# Experiment C Design

## Goal

Test the hypothesis: **Can group tokens help the model predict position i+k as accurately as predicting position i+1?**

The core idea is that instead of predicting pure tokens directly at distant positions, we predict "group tokens" that represent a subset of the vocab. This reduces prediction difficulty by constraining the search space.

---

## Core Design Decisions

### 1. Model: PDLM

We use **PDLM** (Parallel Denoising Language Model), not GPT-2 or BD3LM.

PDLM can predict multiple positions in parallel within a block, which is essential for testing group token generation at scale.

### 2. Input: Pure → Group (No MASK)

**Input transformation**: Pure tokens are noised to their corresponding group tokens.

```
Original:     [tok_0, tok_1, tok_2, tok_3, tok_4, ...]
                 ↓      ↓      ↓      ↓      ↓
Noised:       [tok_0, G_a,   G_b,   G_c,   G_d,   ...]
              (prefix) (noised block positions)
```

- Positions 0..i: **Pure tokens** (context/prefix)
- Positions i+1..i+k: **Group tokens** (noised from pure tokens)
- **NO MASK token** in input - we go directly from pure to group level

### 3. Target: Pure Tokens

The model predicts **pure tokens** - this is the **Group → Pure** denoising stage.

```
Input:   [tok_0, tok_1, ..., tok_i, G_a,   G_b,   G_c,   G_d  ]
Target:  [  -  ,   -  , ...,   -  , tok_a, tok_b, tok_c, tok_d]
```

Where `G_x` is the group token containing `tok_x`, and the model must recover the exact pure token.

**Why this setup?**
- Group tokens provide partial information (the token is somewhere in this group)
- Model must use context + group constraint to predict the exact token
- Tests whether coarse info helps predict distant positions
- This is the second stage of denoising (first stage MASK→Group is for later experiments)

### 4. Metrics

#### Standard Metrics
- **Loss**: Cross-entropy loss on pure token prediction (given group context)
- **Perplexity**: exp(loss), measures prediction uncertainty
- **Accuracy**: % of positions where predicted pure token is correct

#### Causal Compatibility Metric (Key Innovation)

**Problem**: When predicting multiple pure tokens in parallel (e.g., block_size=4), each position may get a good individual loss, but the predictions may not be **mutually compatible** - they weren't conditioned on each other.

**Solution**: Test if generated tokens are consistent when used as context.

```
Step 1: Model generates x1, x2, x3, x4 in parallel (from group tokens G1, G2, G3, G4)
        Each position only sees its own group token, not the other predictions.

Step 2: For each position i, re-predict x_i given the OTHER generated pure tokens:
        - Give model: [prefix..., x1, x2, x3, G4]  → predict x4'
        - Give model: [prefix..., x1, x2, G3, x4]  → predict x3'
        - Give model: [prefix..., x1, G2, x3, x4]  → predict x2'
        - Give model: [prefix..., G1, x2, x3, x4]  → predict x1'

Step 3: Compare: Does x_i == x_i'?
        If model "changes its mind", the parallel predictions weren't compatible.
```

**Compatibility Score** = % of positions where x_i == x_i' (model doesn't change mind)

**Interpretation**:
- High compatibility → Parallel predictions are coherent, model captures dependencies well
- Low compatibility → Parallel predictions are inconsistent, would need iterative refinement

This metric is crucial because:
1. A model might achieve low per-position loss but generate incoherent sequences
2. It measures whether parallel decoding produces consistent outputs
3. It's a proxy for "causal faithfulness" - how well the model respects token dependencies

---

## Experiment Design

### Phase 1: Baseline Measurements

Train PDLM with different noise levels to understand the difficulty landscape.

| Variant | num_groups | tokens/group | Noise Level |
|---------|------------|--------------|-------------|
| `pdlm_g4` | 4 | 1024 | Very high |
| `pdlm_g16` | 16 | 256 | High |
| `pdlm_g64` | 64 | 64 | Medium |
| `pdlm_g256` | 256 | 16 | Low |
| `pdlm_g1024` | 1024 | 4 | Very low |

**Metrics to collect**:
- Pure token prediction loss at each position (i+1, i+2, i+3, i+4)
- Compatibility score for each block_size

### Phase 2: Noise Level Analysis

**Question**: At what noise level does predicting pure token at i+k (given group context) match predicting pure token at i+1 (standard AR)?

```
Compare:
  - PDLM predicting pure token at i+4 (given group tokens at i+1, i+2, i+3, i+4)
  - AR model predicting pure token at i+1 (no hints)

If group-conditioned i+4 loss ≈ AR i+1 loss, we've found the sweet spot.
```

### Phase 3: Compatibility Analysis

For each noise level, measure:
1. **Per-position loss**: How good is each position independently?
2. **Compatibility score**: How consistent are parallel predictions?
3. **Trade-off curve**: Plot loss vs. compatibility across noise levels

**Hypothesis**: Lower noise (more groups) → better individual loss, but possibly lower compatibility (more room for inconsistency).

---

## Training Setup

### Model Architecture

**PDLM Stage Types:**

| Stage | Stage 1 | Stage 2 | wte | lm_head | MASK? |
|-------|---------|---------|-----|---------|-------|
| `stage1_mtp` | MTP | - | pure | groups | No |
| `stage1_mask` | MASK | - | pure + MASK | groups | Yes |
| `stage2` | - | Yes | pure + groups | pure | No |
| `both_mtp` | MTP | Yes | pure + groups | pure + groups | No |
| `both_mask` | MASK | Yes | pure + groups + MASK | pure + groups | Yes |

**For Experiment C, we use `stage2`**: Group → Pure denoising.

```python
# PDLM Config for Experiment C (stage2)
class PDLMConfig:
    pure_vocab_size: int = 4096
    num_groups: int = 64  # varies by experiment
    stage: str = "stage2"
    # wte: pure_vocab + num_groups
    # lm_head: pure_vocab_size
```

### Input/Output Format

```python
# Training batch
input_ids:  (B, L)  # [pure_prefix..., group_tokens...]
target_ids: (B, L)  # [ignore..., pure_targets...]  <- pure tokens!
loss_mask:  (B, L)  # [0..., 1, 1, 1, 1] for block positions
```

### Loss Computation

```python
# Standard CE loss on pure tokens (Group → Pure denoising)
logits = model.lm_head(hidden_states)  # (B, L, pure_vocab_size)
loss = F.cross_entropy(logits, target_ids, reduction='none')
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

### Compatibility Evaluation

```python
def compute_compatibility(model, input_ids, generated_pure, group_tokens, block_positions):
    """
    Test if model changes predictions when conditioned on other generated pure tokens.

    Args:
        input_ids: Original input with prefix + group tokens at block positions
        generated_pure: (B, block_size) pure tokens generated in parallel
        group_tokens: (B, block_size) group tokens at block positions
        block_positions: Indices of block positions in input_ids

    Returns:
        compatibility_score: float in [0, 1]
    """
    B, block_size = generated_pure.shape
    agreements = 0
    total = 0

    for i in range(block_size):
        # Create input: other positions get generated pure tokens, position i keeps group token
        test_input = input_ids.clone()
        for j in range(block_size):
            if j != i:
                # Replace group token with generated pure token
                test_input[:, block_positions[j]] = generated_pure[:, j]
            else:
                # Keep group token at position i
                test_input[:, block_positions[j]] = group_tokens[:, j]

        # Re-predict position i with other pure tokens as context
        logits = model(test_input)
        repredicted = logits[:, block_positions[i]].argmax(dim=-1)

        # Check if model changes its mind
        agreements += (repredicted == generated_pure[:, i]).sum().item()
        total += B

    return agreements / total
```

---

## Training Matrix

### Tokenizer Configs

**Stage 1 (MASK → Group)**: Use overlap_k = 1 (each token belongs to exactly one group)

| Config | num_groups | overlap_k | tokens/group |
|--------|------------|-----------|--------------|
| `g4_k1` | 4 | 1 | 1024 |
| `g16_k1` | 16 | 1 | 256 |
| `g64_k1` | 64 | 1 | 64 |
| `g256_k1` | 256 | 1 | 16 |
| `g1024_k1` | 1024 | 1 | 4 |

**Stage 2 (Group → Pure)**: Also explore overlap_k > 1 to see effect

| Config | num_groups | overlap_k | tokens/group | Notes |
|--------|------------|-----------|--------------|-------|
| `g64_k1` | 64 | 1 | 64 | Baseline |
| `g64_k2` | 64 | 2 | 64 | Each token in 2 groups |
| `g64_k4` | 64 | 4 | 64 | Each token in 4 groups |
| `g16_k1` | 16 | 1 | 256 | High noise baseline |
| `g16_k4` | 16 | 4 | 256 | High noise + high overlap |

**Overlap semantics**: When overlap_k > 1, during training the model may see different group tokens for the same pure target. This tests whether the model can learn to denoise regardless of which group is given.

### Models to Train

| Model | num_groups | block_size | Notes |
|-------|------------|------------|-------|
| `expc_g64_b4` | 64 | 4 | Main experiment |
| `expc_g16_b4` | 16 | 4 | High noise |
| `expc_g256_b4` | 256 | 4 | Low noise |
| `expc_g64_b8` | 64 | 8 | Larger block |

---

## Evaluation Plan

### Per-Model Metrics

1. **Loss by position**: loss[i+1], loss[i+2], loss[i+3], loss[i+4]
2. **Accuracy by position**: % correct pure token predictions
3. **Compatibility score**: Overall and per-position

### Cross-Model Analysis

1. **Loss vs. noise level**: Does lower noise always help?
2. **Compatibility vs. noise level**: Trade-off curve
3. **Position degradation**: How much does loss increase with distance?
4. **Sweet spot identification**: Where is the best loss-compatibility trade-off?

### Visualization

```
Plot 1: Loss by Position
  X-axis: Position (i+1 to i+block_size)
  Y-axis: Cross-entropy loss
  Lines: Different noise levels (g4, g16, g64, g256, g1024)

Plot 2: Compatibility vs. Noise Level
  X-axis: num_groups (log scale)
  Y-axis: Compatibility score
  Point: Each model

Plot 3: Loss-Compatibility Trade-off
  X-axis: Average loss
  Y-axis: Compatibility score
  Points: Each (noise_level, block_size) config
```

---

## Infrastructure

### Already Built
- [x] `group_tokenizer/builder.py` - Build tokenizer variants
- [x] `group_tokenizer/config.py` - Configuration
- [x] `group_tokenizer/clustering.py` - K-means clustering
- [x] `group_tokenizer/token_map.py` - Runtime token maps
- [x] `group_tokenizer/dump.py` - Inspect token maps

### To Build
- [ ] Modify PDLM wte to include group tokens, lm_head outputs pure_vocab
- [ ] Update dataloader for pure→group noising (with overlap_k support)
- [ ] Implement compatibility evaluation
- [ ] Create training script for Experiment C
- [ ] Create evaluation/analysis scripts
- [ ] Overlap ablation: compare k=1 vs k=2 vs k=4

---

## Key Questions

1. **Group context benefit**: How much does knowing the group (1-of-64 → 1-of-64) reduce prediction difficulty vs. blind prediction (1-of-4096)?

2. **Position degradation with groups**: Does loss still increase with distance, or do groups equalize it?

3. **Compatibility trade-off**: Do more groups (lower noise) lead to less compatible predictions?

4. **Optimal operating point**: What num_groups gives best loss while maintaining >90% compatibility?

5. **Block size scaling**: Does compatibility degrade faster with larger blocks?

6. **Overlap effect (Stage 2)**: When overlap_k > 1, each pure token belongs to multiple groups. Does training with varied group assignments improve robustness or hurt accuracy?

---

## Next Steps

1. Generate tokenizer variants with overlap_k=1 (g4, g16, g64, g256, g1024)
2. Modify PDLM: wte includes group tokens, lm_head outputs pure_vocab
3. Update dataloader for pure→group noising
4. Train first model: `expc_g64_k1_b4` (baseline)
5. Implement compatibility evaluation
6. Analyze baseline results
7. Generate tokenizer variants with overlap_k > 1 (g64_k2, g64_k4, etc.)
8. Train overlap ablation models
9. Compare overlap=1 vs overlap>1 results

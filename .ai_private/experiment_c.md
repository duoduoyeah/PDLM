# Experiment C Design

## Goal

Test the hypothesis: **Can group tokens help the model predict position i+k as accurately as predicting position i+1?**

The core idea is that instead of predicting pure tokens directly at distant positions, we predict "group tokens" that represent a subset of the vocab. This reduces prediction difficulty by constraining the search space.

---

## Core Design Decisions

### 1. Model: PDLM

We use **PDLM** (Parallel Denoising Language Model), not GPT-2 or BD3LM.

PDLM can predict multiple positions in parallel within a block, which is essential for testing group token generation at scale.

### 2. Input: 2L Structure (Same as BD3LM)

PDLM uses the same **2L input structure** as BD3LM, following `attn_masks.py`:

```
Total input: [xt | x0]  where each half has length L

xt (first L):  Noised tokens - group tokens at block positions, pure at prefix
x0 (second L): Clean tokens - always pure tokens
```

**Input transformation**: Pure tokens are noised to their corresponding group tokens in xt.

```
x0 (clean):   [tok_0, tok_1, tok_2, tok_3, tok_4, ...]  <- always pure
                 ↓      ↓      ↓      ↓      ↓
xt (noised):  [tok_0, G_a,   G_b,   G_c,   G_d,   ...]
              (prefix) (noised block positions)
```

- **xt prefix (0..i)**: Pure tokens (context)
- **xt block positions (i+1..i+k)**: Group tokens (noised from pure)
- **x0 (all positions)**: Pure tokens (clean reference)
- **NO MASK token** in input - we go directly from pure to group level

### 3. Target: Pure Tokens

The model predicts **pure tokens** at block positions - this is the **Group → Pure** denoising stage.

```
Full input (2L):
  xt:     [tok_0, tok_1, ..., tok_i, G_a,   G_b,   G_c,   G_d  ]  (L tokens)
  x0:     [tok_0, tok_1, ..., tok_i, tok_a, tok_b, tok_c, tok_d]  (L tokens)

Target:   [  -  ,   -  , ...,   -  , tok_a, tok_b, tok_c, tok_d]  (loss only on block positions)
```

Where `G_x` is the group token containing `tok_x`, and the model must recover the exact pure token.

### Loss Computation (CRITICAL)

**Loss is computed on the FIRST half (xt), at block positions [prefix_len, L).**

```
Position:    0 -------- L-1 | L -------- 2L-1
             |---- xt ----|  |---- x0 ----|

xt INPUT:    [pure_prefix | group_tokens]   <- model sees this
xt TARGET:   [    -       | pure_tokens ]   <- LOSS COMPUTED HERE

x0:          [pure_prefix | pure_tokens ]   <- clean reference (attention only, NO loss)

Loss mask:   [0...0       | 1, 1, 1, 1  ]   [0...0 | 0...0]
             ^--prefix---^ ^--blocks----^   ^--- x0 (no loss) ---^
```

- xt block positions: **input = group token, target = pure token** → loss here
- x0 is the clean reference for attention (no loss on x0)

### BD3LM vs PDLM Stage2

| Aspect | BD3LM | PDLM Stage2 |
|--------|-------|-------------|
| Noise token | MASK | Group token |
| Noised positions | **Random** (sampled by t) | **All** block positions |
| Loss positions | Only MASKED positions | All block positions |
| Information | Binary: revealed or hidden | Coarse: group hint everywhere |

```
BD3LM (random masking, e.g. 2 of 4):
  xt blocks: [pure, MASK, pure, MASK]
  loss:      [  - , yes ,   - , yes ]  ← only on MASK positions

PDLM stage2 (all noised):
  xt blocks: [G_a,  G_b,  G_c,  G_d ]
  loss:      [yes,  yes,  yes,  yes ]  ← ALL block positions
```

**Attention mask** (from `attn_masks.py`):
- Block diagonal: xt block attends to itself, x0 block attends to itself
- Offset block causal: xt blocks attend to previous x0 blocks
- Block causal: x0 blocks attend to current and previous x0 blocks
- Prefix: bidirectional within prefix, blocks attend to prefix

**Why this setup?**
- Group tokens provide partial information (the token is somewhere in this group)
- Model must use context + group constraint to predict the exact token
- Tests whether coarse info helps predict distant positions
- This is the second stage of denoising (first stage MASK→Group is for later experiments)

### 4. Metrics

- **Loss**: Cross-entropy loss on pure token prediction
- **Accuracy**: % correct pure token predictions per position

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

## Infrastructure

### Already Built
- [x] `group_tokenizer/builder.py` - Build tokenizer variants
- [x] `group_tokenizer/config.py` - Configuration
- [x] `group_tokenizer/clustering.py` - K-means clustering
- [x] `group_tokenizer/token_map.py` - Runtime token maps
- [x] `group_tokenizer/dump.py` - Inspect token maps
- [x] Modify PDLM wte to include group tokens, lm_head outputs pure_vocab (stage-based in `pdlm.py`)
- [x] Generate tokenizer variants: g16_k1, g64_k1, g256_k1
- [x] `launch/build_group_tokenizer.sh` - Build script with dump

### To Build
- [ ] Update dataloader for pure→group noising (2L structure, with overlap_k support)
- [ ] Implement compatibility evaluation
- [ ] Create training script for Experiment C (`launch/run_expc.sh`)
- [ ] Create evaluation/analysis scripts
- [ ] Overlap ablation: compare k=1 vs k=2 vs k=4

---

## Next Steps

1. ~~Generate tokenizer variants with overlap_k=1 (g16, g64, g256)~~ ✓
2. ~~Modify PDLM: wte includes group tokens, lm_head outputs pure_vocab~~ ✓
3. **Update dataloader for pure→group noising (2L structure)** ← CURRENT
4. Create `launch/run_expc.sh` training script
5. Train first model: `expc_g64_k1_b4` (baseline)
6. Implement compatibility evaluation
7. Analyze baseline results
8. Generate tokenizer variants with overlap_k > 1 (g64_k2, g64_k4, etc.)
9. Train overlap ablation models
10. Compare overlap=1 vs overlap>1 results

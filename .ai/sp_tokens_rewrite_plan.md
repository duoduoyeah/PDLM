# sp_tokens Rewrite Plan

## Current State Analysis

### Directory Structure
```
nanochat/sp_tokens/
├── token_map.py          # Runtime TokenMap class
├── get_mask_id.py        # Utility to lookup MASK token ID
├── implementation.md     # Scattered notes (messy)
├── creation/
│   ├── inject_mask_token.py   # Injects single <|MASK|> token
│   ├── inject_tokens.py       # Main hierarchical group token injection
│   └── kmeans.py              # K-means clustering implementation
└── dump/
    ├── dump_maps.py           # Debug: dump maps to text
    ├── dump_token_ids.py      # Debug: dump token IDs
    ├── dump_val_dataset.py    # Debug: dump validation data
    └── dump.sh                # Shell wrapper
```

### Tokenizer File Structure

A tokenizer directory contains two files:

| File | Contents | Purpose |
|------|----------|---------|
| `tokenizer.pkl` | Pickled `tiktoken.Encoding` | The actual tokenizer: `mergeable_ranks` (BPE merges), `special_tokens` dict, regex `pat_str` |
| `token_bytes.pt` | Tensor `(vocab_size,)` int32 | Maps `token_id → num_utf8_bytes`. Used for bits-per-byte (bpb) metric. Special tokens = 0 |

**Why token_bytes.pt?**

Bits-per-byte (bpb) is a vocab-size-invariant evaluation metric. Instead of averaging loss per token, we weight by byte length:

```python
# From scripts/tok_train.py
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)  # special tokens not counted in bpb
    else:
        token_bytes.append(len(token_str.encode("utf-8")))
```

Example: "hello" → 5 bytes, "你好" → 6 bytes (3 per CJK char), `<|MASK|>` → 0 bytes

**When extending tokenizer**: New group tokens and MASK should have `token_bytes = 0` (they're special tokens, not real text).

---

## Current Features

### 1. Token Types Added to Base Tokenizer

| Token Type | Example | Purpose |
|------------|---------|---------|
| **MASK token** | `<\|MASK\|>` | Root token representing ALL pure tokens |
| **Group tokens** | `<\|G_0\|>`, `<\|G_0_1\|>`, `<\|G_0_1_2\|>` | Hierarchical groupings via k-means path |
| **Pair overlap tokens** | `<\|G_PAIR_0_1\|>` | Groups covering 2 top-level clusters |
| **Triplet overlap tokens** | `<\|G_TRIP_0_1_2\|>` | Groups covering 3 top-level clusters |

### 2. Hierarchy Structure

The hierarchy is built via **bottom-up k-means clustering** (k=4) on lm_head embeddings:
- **Level 0**: Pure tokens (original vocab, ~4096)
- **Level 1**: 1024 groups
- **Level 2**: 256 groups
- **Level 3**: 64 groups
- **Level 4**: 16 groups
- **Level 5**: 4 top-level groups
- **Overlap levels**: Pair (2-of-4) and Triplet (3-of-4) combinations
- **Root level**: Single MASK token covering all

### 3. Data Structures Created

#### `pure_to_noisy_map` (Tensor)
- **Shape**: `(vocab_size, num_levels, fanout)`
- **Purpose**: For each pure token, at each noise level, what group token(s) can represent it
- **Fanout**: Multiple options per level (for overlap tokens); sampling picks one

#### `noisy_level_map` (Tensor)
- **Shape**: `(total_vocab, 2)` → `[low_level, high_level]`
- **Purpose**: For any token, what noise levels it can appear at
- Level 0 = pure token, higher = coarser group

---

## How Tokens Are Added

### Method 1: Single MASK Token (`inject_mask_token.py`)
1. Load base tokenizer (tiktoken-based `RustBPETokenizer`)
2. Extract existing special tokens and mergeable ranks
3. Assign new ID = `vocab_size` to `<|MASK|>`
4. Create new `tiktoken.Encoding` with updated special tokens dict
5. Save as `tokenizer.pkl`
6. Optionally extend `token_bytes.pt` with padding

### Method 2: Full Hierarchy (`inject_tokens.py`)
1. **Load model and tokenizer** via `load_model_from_dir()`
2. **Extract embeddings** from `model.lm_head.weight`
3. **Run hierarchical k-means**:
   - `perform_hierarchical_kmeans(X, k=4, max_depth=5)`
   - Uses cosine similarity (L2-normalized vectors)
   - Returns `ancestry` tensor: `(N, depth)` with cluster indices 0-3 at each level
4. **Generate token strings** from ancestry paths:
   - Root `()` → `<|MASK|>`
   - Path `(0, 3, 1)` → `<|G_0_3_1|>`
   - Filters out singleton groups (only 1 pure token)
5. **Add overlap tokens** for top-level (if k=4):
   - Pairs: `[0,1], [1,2], [2,3], [3,0]`
   - Triplets: all 3-of-4 combinations
6. **Build token maps**:
   - `pure_to_noisy_map`: fill each level with appropriate group IDs
   - `noisy_level_map`: compute min/max level for each token
7. **Save artifacts**:
   - `tokenizer_{timestamp}.pkl`
   - `token_maps_{timestamp}.pt`

---

## Runtime Usage (`token_map.py`)

### TokenMap Class Methods

| Method | Purpose |
|--------|---------|
| `noise_tokens(pure_ids, noisy_levels)` | Convert pure tokens to group tokens at specified levels |
| `transit_noisy_tokens(pure_ids, noisy_ids, ...)` | Transition from current noisy state to next level |
| `is_all_pure_tokens(ids)` | Check if all tokens are at level 0 |
| `get_random_noisy_level(ids, step, total_steps, ...)` | Sample noise levels (binomial or uniform) |

### Loading
```python
from nanochat.sp_tokens.token_map import get_token_map
token_map = get_token_map(tokenizer_dir, device="cuda")
```

---

## Issues / Code Smells

1. **Scattered logic**: Creation split across `inject_mask_token.py` and `inject_tokens.py` with overlap
2. **Hardcoded values**: k=4, depth=5, min_group_size=8 baked into scripts
3. **No clear API**: Must know to run specific scripts in order
4. **Timestamp-based filenames**: Requires manual renaming to `tokenizer.pkl` / `token_maps.pt`
5. **`implementation.md`**: Contains outdated/partial notes, not actual documentation
6. **Dump scripts**: Scattered debug utilities without unified interface

---

## Questions for Rewrite

1. Should we consolidate `inject_mask_token.py` into `inject_tokens.py` or keep them separate?
2. What configuration should be exposed (k, depth, min_group_size, overlap types)?
3. Should we add a CLI or config-file based approach?
4. How to handle versioning of tokenizers/maps?

---

## Rewrite Goals: Flexible Tokenizer Generation

### Goal 1: Configurable Layer Variants

From a single base (pure) tokenizer, generate **different tokenizer variants**:

| Variant | Description |
|---------|-------------|
| **MASK-only** | Just add `<\|MASK\|>` token (1 extra token) |
| **Single-layer groups** | Add 1 level of group tokens (e.g., 4 groups) + MASK |
| **Multi-layer groups** | Add N levels of hierarchy (configurable depth) + MASK |
| **Full hierarchy** | Current behavior: 5 levels + overlap + MASK |

Configuration should allow:
- Number of group layers (0, 1, 2, ..., N)
- Whether to include MASK token
- k value (branching factor) per level or global

### Goal 2: Optional Overlap Layers

Overlap tokens (pair, triplet, etc.) should be **optional and configurable**:
- Enable/disable overlap layers entirely
- Configure which overlap types: pairs (2-of-k), triplets (3-of-k), etc.
- Overlap layers sit between the finest group level and MASK

### Goal 3: Multi-Membership Per Layer (Future)

**Current behavior**: At each hierarchy level, each pure token belongs to **exactly 1** group token.
- Exception: Overlap layers allow multiple membership by design

**Future behavior**: Allow **soft clustering** where each pure token can belong to **multiple groups** at the same level.

| Aspect | Current | Future |
|--------|---------|--------|
| Groups per pure token per level | 1 | 1 or more (configurable) |
| Fanout meaning | Only for overlap sampling | Also for multi-membership at any level |
| Clustering method | Hard k-means | Soft k-means / overlapping clusters |

This enables:
- Smoother transitions between noise levels
- Better coverage for tokens near cluster boundaries
- More flexible denoising paths

### Architecture Sketch

```
BaseTokenizer (pure)
       │
       ▼
┌──────────────────────────────────┐
│     TokenizerBuilder             │
│  ─────────────────────────────   │
│  - base_tokenizer                │
│  - embeddings (lm_head.weight)   │
│  - config:                       │
│      - num_layers: int           │
│      - k: int (branching factor) │
│      - include_mask: bool        │
│      - include_overlap: bool     │
│      - multi_membership: bool    │
│      - membership_count: int     │
└──────────────────────────────────┘
       │
       ▼
  Different Output Tokenizers
  ├── tokenizer_mask_only/
  ├── tokenizer_1layer/
  ├── tokenizer_3layer/
  ├── tokenizer_full/
  └── tokenizer_soft_cluster/
```

---

## Stage 1: Single-Layer Group Tokens

### Base Tokenizers

| Base Tokenizer | Pure Vocab | With MASK | Notes |
|----------------|------------|-----------|-------|
| **small** | 4096 | 4096 + 1 = 4097 | |
| **medium** | 8192 | 8192 + 1 = 8193 | |

### Parameters to Explore

| Parameter | Name | Meaning |
|-----------|------|---------|
| **A** | `num_groups` | Number of group tokens at this level. Fewer groups = higher noise (each group covers more pure tokens) |
| **B** | `overlap_k` | Each pure token belongs to k different group tokens. k=1 is hard clustering, k>1 is overlapping |

### Model Architecture Reference (pdlm.py)

**Current design:**
- `wte`: `all_vocab_size` (pure + group + MASK)
- `lm_head`: `pure_vocab_size` only

**Proposed design (Stage 1 rewrite):**
- `wte`: `all_vocab_size` (pure + group + MASK)
- `lm_head`: `pure_vocab_size + num_groups` (pure + group, NO MASK)
- **MASK token at end of vocab** - input-only, never predicted

**Two-stage denoising:**
1. MASK → Group: model predicts group token directly (CE loss on group_id)
2. Group → Pure: model predicts pure token directly (CE loss on pure_id)

Both stages use standard CE loss and argmax - symmetric and compatible.

### Parameter Space

**num_groups** range (for 4096 base):
| num_groups | tokens/group | noise level |
|------------|--------------|-------------|
| 1024 | 4 | lowest |
| 256 | 16 | low |
| 64 | 64 | medium |
| 16 | 256 | high |
| 4 | 1024 | highest |

**overlap_k**: 1, 2, 4

### Training Matrix (Lower Triangle)

Higher noise allows higher k. Low noise → k=1 only.

```
                   num_groups (noise level →)
                 1024   256    64    16     4
              ┌─────────────────────────────────
overlap_k=1   │  ✓      ✓      ✓     ✓      ✓
overlap_k=2   │  -      -      ✓     ✓      ✓
overlap_k=4   │  -      -      -     ✓      ✓
```

Same pattern applies to 8192 base (2048, 512, 128, 32, 8 groups).

**Stage 1 scope**: 4096 base only → 10 models

### Design Decisions

**Unequal group sizes from k-means**: Accept it (default). Same as how LM training accepts unequal token lengths - natural imbalance, don't over-engineer. Track stats (min/max/std) for visibility.

**Alternative: bpb-style weighting**: Weight loss by 1/group_size, analogous to how bpb spreads loss over bytes. Large groups get "tolerated" more. Optional experiment for later.

**Evaluation plan**: Analyze per-group accuracy after training - which groups are easy/hard to denoise? Correlate with group size, embedding spread, etc. Reference: `loss_eval.py` uses bpb (bits-per-byte) which weights tokens by byte length - similar idea of non-uniform eval when it makes sense.

**Logit leakage analysis**: In Stage 2 (Group → Pure), check how much probability mass goes to tokens outside the current group. If model learns group structure well, most mass should stay within group members.

### Math Directions (to explore later)

- **Information theory**: H = log₂(tokens/group) bits uncertainty; overlap_k ≤ √(tokens/group) as heuristic
- **Diffusion analogy**: higher noise needs more redundancy, similar to variance-dependent sampling in score matching
- **Clustering**: overlap helps boundary tokens; less useful when clusters are small/tight

### Overlap via Sub-groups (see `.ai/overlap_design.md`)

Instead of topk post-hoc overlap, use **combinatorial sub-group design**:
- Create `num_sub` sub-groups via clustering
- Final groups = combinations of `sub_per_final` sub-groups
- Complete design: num_final = C(num_sub, sub_per_final), overlap_k = C(num_sub-1, sub_per_final-1)

Practical configs for noise level 1024 (vocab 4096):
| Config | num_final | overlap_k |
|--------|-----------|-----------|
| (4,1) | 4 | 1 (baseline) |
| (8,2) | 28 | 7 |
| (12,3) | 220 | 55 |

Tool: `uv run -m scripts.overlap_calc --target_noise 1024`

---

## Training Design: Two-Stage Loss with Overlap

### Overview

When `overlap_k > 1`, each pure token maps to **k valid group tokens**. Training involves two stages in the same batch:

| Stage | Input Token | Target | Output Space |
|-------|-------------|--------|--------------|
| 1 | MASK | One of k valid groups | `num_groups` |
| 2 | GROUP | Single pure token | `pure_vocab_size` |

### Mixed Batch Design (Recommended)

Within one B×2L batch, positions are mixed:
- Some positions: MASK → Group (Stage 1)
- Other positions: GROUP → Pure (Stage 2)

```python
# Training data contains:
# - noisy_ids: (B, L) - mix of MASK and GROUP tokens
# - stage_mask: (B, L) - True where input is MASK (Stage 1)
# - valid_groups: (B, L, k) - k valid group targets for Stage 1 positions
# - pure_targets: (B, L) - pure token targets for Stage 2 positions
```

### Logit Masking by Input Type

**Key design**: Mask out invalid output logits based on input token type.

Modify `pdlm.py` so that:
- When input is **MASK** → logits for pure tokens = -inf (only group logits active)
- When input is **GROUP** → logits for group tokens = -inf (only pure logits active)

```python
# lm_head outputs: (B, L, pure_vocab_size + num_groups)
logits = self.lm_head(x)

# Mask invalid outputs based on input type
# is_mask_input: (B, L) - True where input token is MASK
logits_pure = logits[:, :, :pure_vocab_size]
logits_group = logits[:, :, pure_vocab_size:]

# Zero out (or -inf) invalid parts
logits_pure = logits_pure.masked_fill(is_mask_input.unsqueeze(-1), float('-inf'))
logits_group = logits_group.masked_fill(~is_mask_input.unsqueeze(-1), float('-inf'))

logits = torch.cat([logits_pure, logits_group], dim=-1)
```

This ensures:
- Stage 1: softmax only over group tokens
- Stage 2: softmax only over pure tokens

### Loss Computation

**Stage 1 (MASK → Group) with overlap_k > 1:**

Use marginalized likelihood - probability of predicting ANY valid group:

```python
# valid_groups: (B, L, k) - indices of k valid groups for each position
log_probs = F.log_softmax(logits_group, dim=-1)  # (B, L, num_groups)
valid_log_probs = log_probs.gather(-1, valid_groups)  # (B, L, k)
log_p_any_valid = torch.logsumexp(valid_log_probs, dim=-1)  # (B, L)
stage1_loss = -log_p_any_valid  # NLL
```

Semantics: if model gives P=0.3 to group A and P=0.2 to group B (both valid), then P(correct) = 0.5, loss = -log(0.5).

**Stage 2 (Group → Pure):**

Standard single-target cross-entropy:

```python
stage2_loss = F.cross_entropy(logits_pure, pure_targets, reduction='none')
```

**Combined loss:**

```python
loss = torch.where(stage_mask, stage1_loss, stage2_loss).mean()
```

### Loss Scale Difference

**Natural imbalance:**
- Stage 1: picking 1-of-28 groups (with k=7 valid) → random baseline ≈ log(4) ≈ 1.4
- Stage 2: picking 1-of-4096 pure tokens → random baseline ≈ log(4096) ≈ 8.3

Stage 1 loss is naturally ~6x smaller.

**Approach:**
1. **First**: No adjustment - train with natural loss scales, observe behavior
2. **Later**: If needed, add explicit scaling:
   ```python
   scale = math.log(pure_vocab_size) / math.log(num_groups)
   loss = torch.where(stage_mask, stage1_loss * scale, stage2_loss).mean()
   ```

### Computation Overhead

The multi-target loss adds negligible overhead:
- `gather` over k indices: O(B × L × k)
- `logsumexp` over k: O(B × L × k)

With k=7, this is ~0.1% of total training cost (dominated by transformer forward pass).

# Block-PDLM Design

## Overview

Block-PDLM is an autoregressive language generation model that generates text block by block, with each block produced in two steps. During training, the model uses a 2L input structure `[xt | x0]` where almost every token position contributes to the loss: the first L positions (xt half) compute Stage 2 loss (Group → Pure denoising), while the second L positions (x0 half) compute Stage 1 loss (block→block group prediction). During inference, the input is just the prompt, and the model generates autoregressively - each block requires two forward steps: first predict the group tokens for the block, then denoise them to pure tokens.

## Two-Stage Architecture

```
Input:  [xt (group tokens) | x0 (pure tokens)]
         ↓                   ↓
Stage 2: Group → Pure       Stage 1: Block → Block (predict next block's tokens/groups)
         ↓                   ↓
Output:  pure vocab logits   pure/group vocab logits (mode-dependent)
```

## Training

**Stage configs:** `pdlm_stage=both_block`

**Loss:**
```
combined_loss = mtp_loss_weight * stage1_loss + stage2_loss
```

### Stage 1 Target Modes

**Pure-target mode (default):**
- `lm_head` shape: `(pure_vocab_size, n_embd)` - full pure vocabulary
- Stage 1 loss: CE against **pure token targets** (blocks 1..N-1)
- At inference: pre-compute `group_head = group_to_pure_mask @ lm_head.weight` to predict groups

**Group-target mode (`--stage1_target_mode=group`):**
- `lm_head` shape: `(num_groups, n_embd)` - group vocabulary only
- Stage 1 loss: any-correct CE against **group token targets** (blocks 1..N-1)

### Stage 2

- Stage 2 loss: CE against pure token targets (all block positions)

### Why Pure-Target Mode?

The key insight is that **Stage 2 acts as error correction**. Even if Stage 1 doesn't perfectly predict the next token, as long as it assigns high probability to tokens within the correct group, Stage 2 can recover.

When we collapse pure logits to group logits via `group_to_pure_mask @ lm_head.weight`:
- The group with highest summed member logits wins
- A "near miss" (high prob on wrong token but same group) still leads to correct group prediction
- Stage 2 then refines within the group

This makes the two-stage pipeline more robust than requiring Stage 1 to directly predict coarse groups perfectly.

## Generation

Two steps per block:
1. **Predict groups**: Forward pure context with causal attention → argmax group logits
2. **Denoise**: Build `[xt | x0]` with predicted groups → argmax pure logits

## RoPE Position Analysis

**Question:** In block-to-block prediction with block_size=4, the last block's tokens have RoPE positions [T-4, T-3, T-2, T-1] but predict targets at positions [T, T+1, T+2, T+3]. Is this a problem?

**Answer:** No. The gap between input position and prediction position is a constant `block_size` for every token — the same pattern as standard AR (where the gap is a constant 1). The model learns this fixed offset during training since every block boundary has the same structure. RoPE correctly encodes the true positions of context tokens for attention computation; the lm_head simply learns "predict block_size ahead" instead of "predict 1 ahead."

**Why within-block bidirectional attention is fine with sequential RoPE:** All tokens in a block see the same context (bidirectional within block + causal to prefix), but produce different predictions because their RoPE positions give each a unique identity. RoPE naturally handles both forward and backward relative positions, so bidirectional attention within a block works correctly.

## Open Questions

**Loss combination:** Two standard approaches exist - (1) **Joint**: sum losses and backprop once, (2) **Alternating**: backprop each loss separately with separate updates. Current implementation uses joint.

**Loss weighting problem:** With pure-target mode, both Stage 1 and Stage 2 now predict pure tokens, so their loss magnitudes are similar. The `mtp_loss_weight` parameter still allows tuning if needed. Options from multi-task learning:
- **DWA (Dynamic Weight Average)**: Adjust weights based on loss rate-of-change, prioritizes slower-converging tasks. Simple, low cost.
- **Uncertainty Weighting**: Learn σ per task, `L = L1/σ1² + L2/σ2² + log(σ1) + log(σ2)`. Principled, auto-balances.
- **GradNorm**: Directly balance gradient magnitudes across tasks. Medium cost.
- **Fixed ratio**: Compute initial L2/L1 ratio, use as constant w1. Simplest baseline.

**RoPE should not be adjusted for prediction offset:** Stage 1 (position i → predict i+B) and Stage 2 (position i → predict i) have different target offsets. However, RoPE encodes **relative position for attention** (`q_m^T k_n` depends on `m-n`), not prediction target. Adjusting RoPE to match "target position" would break attention patterns. Keep RoPE as actual sequence positions; the model learns prediction offsets through lm_head and embeddings.

**Stage disambiguation:** Currently the model must disambiguate Stage 1 vs Stage 2 behavior purely from input embedding type (pure vs group token). This is unverified — the model might conflate them. If implicit disambiguation fails, consider explicit stage embedding (similar to timestep embedding in diffusion/flow matching models):

- **Option A (simple):** Add learned stage embedding to input embeddings: `input_emb + stage_emb[stage_id]`
- **Option B (per-block, preferred):** Inject stage embedding into every transformer block via AdaLN (Adaptive LayerNorm) — timestep/stage modulates layernorm scale & shift. Stronger conditioning, less signal washout through layers.

When trying stage embedding, start with Option B (per-block AdaLN).

---

# block_pdlm_inference (Iterative Denoising Variant)

**Stage name:** `pdlm_stage=block_pdlm_inference`

## Motivation

The original block-pdlm generates each block in 2 steps (predict groups, then denoise all at once). This has two problems:
1. Stage 2 always sees all-group input during training but needs to work with mixed pure/group input during iterative inference
2. The model cannot be used for iterative (one-position-at-a-time) generation within a block

block_pdlm_inference trains Stage 2 with **mixed pure/group inputs**, so it learns to denoise from any partial state. This enables iterative inference where each block is generated one token at a time.

## Stage 2 Training Change

**Original block-pdlm:**
```
Input:  {g, g, g, g}  →  Target: {p, p, p, p}    (denoise all)
```

**block_pdlm_inference:**
For each block during training, randomly sample `r ∈ {0, 1, 2, 3}` — the number of "already revealed" pure tokens (left-to-right). **All targets are pure tokens** (pure-target mode):

```
r=0: {g, g, g, g} → {p_i0, p_i1, p_i2, p_i3}    (denoise current block, same as original)
r=1: {p, g, g, g} → {p_j0, p_i1, p_i2, p_i3}    (pos 0: predict next block; pos 1-3: denoise)
r=2: {p, p, g, g} → {p_j0, p_j1, p_i2, p_i3}    (pos 0-1: predict next block; pos 2-3: denoise)
r=3: {p, p, p, g} → {p_j0, p_j1, p_j2, p_i3}    (pos 0-2: predict next block; pos 3: denoise)
```

Where `p_i*` = pure token in current block i, `p_j*` = pure token in next block i+1.

**Rule:** Group input → target is pure token at same position in **current block** (denoise). Pure input → target is pure token at same position in **next block** (predict ahead, like Stage 1).

This means the model learns all intermediate denoising states. The lm_head always outputs pure vocab logits. The task (denoise vs predict-ahead) is implicitly determined by whether the model sees a group or pure embedding.

**Stage 1 remains pure-target mode** (unchanged from base block-pdlm).

## Inference (5 Steps Per Block)

For block_size=4, generating one block requires 1 + block_size = 5 forward steps.
Group tokens are **progressively refined**: at each step, the model's denoising predictions for remaining group positions are collapsed back to updated group tokens.

```
Step 1 (Stage 1): pure context → pure logits → collapse → g0, g1, g2, g3

Step 2: {g0,g1,g2,g3} → predict p0',p1',p2',p3' (all denoise current block)
        Sample p0. Update: g1=collapse(p1'), g2=collapse(p2'), g3=collapse(p3')

Step 3: {p0, g1,g2,g3} → predict _,p1',p2',p3'  (pos 0 predicts next block; rest denoise)
        Sample p1. Update: g2=collapse(p2'), g3=collapse(p3')

Step 4: {p0,p1,g2,g3}  → predict _,_,p2',p3'     (pos 0-1 predict next block; rest denoise)
        Sample p2. Update: g3=collapse(p3')

Step 5: {p0,p1,p2,g3}  → predict _,_,_,p3'        (pos 0-2 predict next block; pos 3 denoises)
        Sample p3.
```

- Stage 1 uses **argmax** (deterministic group selection via pure→group collapse)
- Stage 2 uses **sampling** (temperature=1.0 default)
- Group collapse: `pure_probs = softmax(pure_logits)` → `group_probs = group_to_pure_mask @ pure_probs` → argmax
- Each remaining group token gets refined with more context at each step (e.g., g3 is updated 3 times before being used for final denoising)

## Evaluation

**Dump eval:** Given a prompt of K tokens, generate 128 or 256 tokens using the iterative inference above. Dump the decoded text for manual inspection by researchers.

## Pure-to-Group Collapse with Top-k Filtering

At inference, when collapsing pure logits to group probabilities (for Stage 1 group prediction and group token refinement), use top-k filtering to ignore the long tail of unlikely tokens:

```python
# collapse_topk: default k=256, set k=-1 to disable
if collapse_topk > 0:
    topk_vals, topk_ids = torch.topk(pure_logits, k=collapse_topk)
    topk_probs = softmax(topk_vals)
    topk_groups = pure_to_group[topk_ids]       # (*, k, overlap_k)
    # scatter_add into group_probs
else:
    # no filtering, dense collapse
    group_probs = group_to_pure_mask @ softmax(pure_logits)
```

- **Default:** `collapse_topk=256`
- **Disable:** `collapse_topk=-1` (use all vocab tokens)
- Applied everywhere pure→group collapse happens: Stage 1 group prediction, group token refinement during iterative denoising

## References

- Stage 1 block approach: `design/experiment_d.md`
- Base implementation: `nanochat/pdlm.py` (`_forward_both_block`)
- Base dataloader: `nanochat/dataloader_pdlm.py` (`both_block` branch)
- Training script: `launch/run_block_pdlm.sh`

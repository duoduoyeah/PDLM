# Learned Group Mapping

## Motivation

The pdlm_s1b model (block-to-block Stage 1) trained with pure token targets produces similar loss regardless of tokenizer/grouping, since no group information appears during training. This means the model checkpoint is **tokenizer-agnostic** — it learns a pure-token distribution that is independent of any particular grouping.

However, when evaluating group accuracy by collapsing pure logits via a fixed `group_to_pure_mask`, we observe a counterintuitive pattern:

```
pos 0: loss=1.5728, ppl=4.82,  accuracy=36.52%
pos 1: loss=2.5910, ppl=13.34, accuracy=53.96%
pos 2: loss=3.4217, ppl=30.62, accuracy=64.53%
pos 3: loss=4.0461, ppl=57.17, accuracy=68.60%
```

Loss increases with position (expected — distant tokens are harder), but group accuracy **also increases**. This is because:

- **Confident predictions (pos 0):** The model pushes one token's logit very high and suppresses others, including same-group members. When summing raw logits per group, the correct group's sum can lose to another group with several moderately positive logits.
- **Uncertain predictions (pos 3):** Logits are softer, less intra-group suppression. Tokens within the same group tend to get similar moderate logits, so the correct group accumulates well via summation.

This logit-sum degeneration is a property of how groups are defined relative to the model's learned logit structure. Different tokenizers show this effect to varying degrees. The fix: **learn the grouping that is optimal for the model's actual prediction patterns**.

## Approach

Given a frozen pdlm_s1b checkpoint, train a group assignment matrix that maximizes group-level accuracy. The LM is frozen — the only trainable parameter is the group mapping itself.

### Pos 0 Change

Position 0 in each block will be decoded directly as a **pure token** (not a group token). This guarantees at least 1 token decoded per forward pass (lower bound of 1 tps). During training of the group mapping, we only train on positions 1..B-1. This also sidesteps the pos 0 low-accuracy problem entirely.

### Training Pipeline

1. **Data**: Pure token sequences from dataset, loaded via existing dataloader (`B*T + K` tokens per batch).
2. **Frozen model forward**: Input `(batch, T)` → pure logits `(batch, T, K, V_pure)`. One logit vector per position per future step. Since the model is frozen, logits can be pre-computed and cached.
3. **Group mapping**: `pure_logits @ A` → `(batch, T, K, G)` where `A` is the trainable assignment matrix `(V_pure, G)`.
4. **Target**: The real pure token at each `(t, k)` position → look up its group under current `A` → `(batch, T, K)`.
5. **Loss**: Cross-entropy over the `G` dimension. All positions in `T` and all `k` in `K` contribute. Backprop only updates `A`.

Pos 0 variant: either include all positions, or exclude pos 0 per block (see Pos 0 Change above). Both should be tested.

### Trainable Group Assignment (Approach C)

The `group_to_pure_mask` (shape `(num_groups, pure_vocab_size)`, binary) is replaced with a trainable float assignment matrix. Training operates in continuous space; binarization happens after training.

```python
# Learnable parameter — the only thing being trained
A = nn.Parameter(torch.randn(pure_vocab_size, num_groups))  # (V, G)

# Forward pass (LM is frozen, only A is updated)
soft_assign = torch.sigmoid(A)                               # (V, G), continuous [0, 1]
group_logits = pure_logits @ soft_assign                     # (batch, T, K, G)
target_membership = soft_assign[target_pure_token]           # (batch, T, K, G)
loss_task = BCE(sigmoid(group_logits), target_membership)
```

Key properties:
- **Frozen model:** No backprop through the LM. Can pre-compute and cache pure logits for the whole dataset, then train the assignment matrix on cached logits. This makes sweeping `num_groups` very cheap.
- **Float training, binary inference:** During training `soft_assign` is continuous — each entry is "how strongly does token v belong to group g". After training, threshold to 0/1.
- **BCE loss:** Each group is an independent binary question — "does the target token belong to this group?" No mutual exclusivity assumption, naturally handles overlap.
- **Target depends on assignment:** As `A` learns, target membership shifts. Handled naturally since both group_logits and target_membership derive from the same `soft_assign`.

### Loss

```python
loss = loss_task + λ_noise * loss_noise + λ_overlap * loss_overlap + sharp_schedule(step) * loss_sharp
```

- **Task loss (BCE):** Primary objective. Per-group binary prediction against soft membership target.
- **Noise penalty (column constraint):** `relu(col_sums - max_size_soft).mean()` — discourages groups from being too large.
- **Overlap penalty (row constraint):** `relu(min_overlap_soft - row_sums).mean()` — encourages tokens to appear in multiple groups.
- **Sharpening (annealed):** `(soft_assign * (1 - soft_assign)).mean()` — pushes values toward 0 or 1. Weight starts at 0 (free exploration), ramps up during training (force commitment). Ensures clean binarization at the end.

All penalties are soft — the model can violate them if the task loss benefits enough.

### Hyperparameters

- `num_groups (G)`: Controls granularity / noise level. Fewer groups = easier for Stage 1 but more work for Stage 2. Sweep to trace a granularity-accuracy tradeoff curve.
- `max_size_soft`: Soft upper bound on group size (column sum). Controls noise level.
- `min_overlap_soft`: Soft lower bound on token overlap (row sum). Controls how many groups a token belongs to.
- `λ_noise, λ_overlap`: Weights for size/overlap penalties.
- `sharp_schedule`: Annealing schedule for sharpening penalty (e.g., linear ramp from 0 to λ_sharp over training).

### Coarse Hyperparameter Sweep

Three-phase sweep: isolate G first, then penalties, then refine.

**Phase 1: Probe G × penalty interaction (12 runs)**

Fix: `max_size_soft = V/G × 4`, `min_overlap_soft = 4`, `λ_sharp = 1.0` (ramp start 70%), `lr = 0.1`, Adam.

| G | `λ_noise` | `λ_overlap` | Description |
|---|-----------|-------------|-------------|
| 64 | 0.1 | 0 | Noise only |
| 64 | 1.0 | 0 | Noise strong |
| 64 | 0.1 | 0.1 | Both moderate |
| 64 | 1.0 | 0.1 | Noise strong + overlap moderate |
| 256 | 0.1 | 0 | Noise only |
| 256 | 1.0 | 0 | Noise strong |
| 256 | 0.1 | 0.1 | Both moderate |
| 256 | 1.0 | 0.1 | Noise strong + overlap moderate |
| 512 | 0.1 | 0 | Noise only |
| 512 | 1.0 | 0 | Noise strong |
| 512 | 0.1 | 0.1 | Both moderate |
| 512 | 1.0 | 0.1 | Noise strong + overlap moderate |

Noise penalty is always on (prevents degenerate all-ones solution). Overlap is secondary, tested on/off.

Goal: find which G range works, and whether noise strength matters more than overlap.

**Phase 2: Refine at best G (sweep penalty strength and soft targets)**

At the best 1-2 G values from Phase 1, sweep:

| Param | Values |
|-------|--------|
| `λ_noise` | 0.01, 0.1, 0.5, 1.0 |
| `max_size_soft` | V/G × 1, V/G × 2, V/G × 4, V/G × 8 |
| `λ_overlap` | 0, 0.1 |

**Phase 3: Refine sharpening and lr**

| Param | Values |
|-------|--------|
| `λ_sharp` (final) | 0.1, 1.0, 10.0 |
| Ramp start | 50%, 70%, 90% of total steps |
| `lr` | 1e-2, 1e-1, 1.0 |

LR can be higher than typical deep learning because the "model" is a single `(V, G)` matrix — no depth, no compounding gradients.

## Prior Art

The current tokenizer grouping was produced via weight-space clustering (k-means on `lm_head.weight` rows from another model). This is approach A — non-parametric, no training, uses a different model's representations. The learned approach directly optimizes grouping for the actual model being used.

## References

- Pure-target vs group-target analysis: `design/experiment_d.md`
- Block-PDLM architecture: `design/block_pdlm.md`
- Implementation target: TBD

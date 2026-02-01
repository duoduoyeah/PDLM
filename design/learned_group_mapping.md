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

### Trainable Group Assignment (Approach C)

The `group_to_pure_mask` (shape `(num_groups, pure_vocab_size)`, binary) is replaced with a trainable soft assignment matrix.

```python
# Learnable parameter — the only thing being trained
group_assign = nn.Parameter(torch.randn(pure_vocab_size, num_groups))

# Forward pass (LM is frozen, only group_assign is updated)
pure_logits = hidden @ lm_head.weight.T                        # (B, T, V) — from frozen model
soft_assign = gumbel_softmax(group_assign, hard=True, dim=-1)  # (V, G) — differentiable discrete
group_logits = pure_logits @ soft_assign                       # (B, T, G)
target_group = soft_assign[target_pure_token].argmax(dim=-1)   # target shifts with assignment
loss = CE(group_logits, target_group)
```

Key properties:
- **Frozen model:** No backprop through the LM. Can pre-compute and cache pure logits for the whole dataset, then train the assignment matrix on cached logits. This makes sweeping `num_groups` very cheap.
- **Gumbel-softmax:** Gives differentiable discrete assignments. Each pure token is assigned to exactly one group during forward pass, but gradients flow through the soft relaxation.
- **Target depends on assignment:** As the assignment matrix learns, `target_group_id` for each pure token shifts. This is handled naturally since both the group logits and the target are derived from the same `soft_assign`.

### Hyperparameters

- `num_groups`: Controls granularity / noise level. Fewer groups = easier for Stage 1 but more work for Stage 2. Sweep to trace a granularity-accuracy tradeoff curve.
- Group size regularization: Optional penalty to encourage balanced group sizes.
- `overlap`: Whether a pure token can belong to multiple groups (future extension).

## Prior Art

The current tokenizer grouping was produced via weight-space clustering (k-means on `lm_head.weight` rows from another model). This is approach A — non-parametric, no training, uses a different model's representations. The learned approach directly optimizes grouping for the actual model being used.

## References

- Pure-target vs group-target analysis: `design/experiment_d.md`
- Block-PDLM architecture: `design/block_pdlm.md`
- Implementation target: TBD

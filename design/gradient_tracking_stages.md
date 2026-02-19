# Gradient Tracking: Stage 1 vs Stage 2 in both_block

Reuse the per-position gradient tracking infrastructure, but with 2 "positions": Stage 1 loss and Stage 2 loss.

## Key metric

`gradient/cos_s1_s2` — cosine similarity between ∇L_stage1 and ∇L_stage2 on shared trunk parameters.
- cos > 0: stages cooperate (joint training helps)
- cos ≈ 0: stages are independent (joint training is neutral)
- cos < 0: stages conflict (joint training hurts both)

## Also logged

`gradient/s1_grad_norm`, `s2_grad_norm`, `s1_grad_pct`, `s2_grad_pct`, `s1_loss`, `s2_loss` — magnitude and loss for each stage.

Per-layer norms (`gradient/layer_{block_N,wte,lm_head}_s{1,2}_grad_norm`) are computed as a side effect of the existing infrastructure; not the focus here.

## Implementation

Treat Stage 1 and Stage 2 as "pos 0" and "pos 1" in the existing `compute_gradient_metrics`. Requires `return_nll`-style separation of the two stage losses from `_forward_both_block`.

For background on per-position gradient tracking methodology, see `gradient_tracking_positions.md`.

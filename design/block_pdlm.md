# Block-PDLM Design

## Overview

Block-PDLM is an autoregressive language generation model that generates text block by block, with each block produced in two steps. During training, the model uses a 2L input structure `[xt | x0]` where almost every token position contributes to the loss: the first L positions (xt half) compute Stage 2 loss (Group → Pure denoising), while the second L positions (x0 half) compute Stage 1 loss (block→block group prediction). During inference, the input is just the prompt, and the model generates autoregressively - each block requires two forward steps: first predict the group tokens for the block, then denoise them to pure tokens.

## Two-Stage Architecture

```
Input:  [xt (group tokens) | x0 (pure tokens)]
         ↓                   ↓
Stage 2: Group → Pure       Stage 1: Block → Block (predict next block's groups)
         ↓                   ↓
Output:  pure vocab logits   group vocab logits
```

## Training

**Stage configs:** `pdlm_stage=both_block`

**Loss:**
```
combined_loss = mtp_loss_weight * stage1_loss + stage2_loss
```

- Stage 1: any-correct CE loss on group targets (blocks 1..N-1)
- Stage 2: CE loss on pure targets (all block positions)

## Generation

Two steps per block:
1. **Predict groups**: Forward pure context with causal attention → argmax group logits
2. **Denoise**: Build `[xt | x0]` with predicted groups → argmax pure logits

## Open Questions

**Q: lm_head outputs both pure and group logits - how does the model ensure xt positions only produce pure logits and x0 positions only produce group logits?**

**Loss combination:** Two standard approaches exist - (1) **Joint**: sum losses and backprop once, (2) **Alternating**: backprop each loss separately with separate updates. Current implementation uses joint.

**Loss weighting problem:** Stage 1 (Pure→Group) has lower loss (~0.3) than Stage 2 (Group→Pure, ~1.5) due to different target space sizes. With w1=1, Stage 2 dominates gradients (~83%). Options from multi-task learning:
- **DWA (Dynamic Weight Average)**: Adjust weights based on loss rate-of-change, prioritizes slower-converging tasks. Simple, low cost.
- **Uncertainty Weighting**: Learn σ per task, `L = L1/σ1² + L2/σ2² + log(σ1) + log(σ2)`. Principled, auto-balances.
- **GradNorm**: Directly balance gradient magnitudes across tasks. Medium cost.
- **Fixed ratio**: Compute initial L2/L1 ratio, use as constant w1. Simplest baseline.

## References

- Stage 1 block approach: `design/experiment_d.md`
- Implementation: `nanochat/pdlm.py` (`_forward_both_block`)
- Dataloader: `nanochat/dataloader_pdlm.py` (`both_block` branch)
- Training script: `launch/run_block_pdlm.sh`

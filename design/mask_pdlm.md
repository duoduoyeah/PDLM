# Mask PDLM

## Overview

Mask PDLM is a two-stage model where every position predicts **itself**, not the next block's token. This is the core difference from block-PDLM:

- **Block-PDLM:** block i's position k, when pure, predicts block i+1's position k (predict ahead)
- **Mask-PDLM:** block i's position k always predicts its own pure token (predict self)

Since Stage 1 has no "predict ahead" mechanism, a **mask token** is needed as a placeholder for unknown positions. The mask token is the initial fully-noisy state before any prediction.

## Comparison with Block-PDLM

| | Block-PDLM | Mask-PDLM |
|---|---|---|
| Target | Dual: pure→next block, group→self | Consistent: always self |
| Mask token | Not needed | Required |
| Data utilization | ~100% (both xt and x0 compute loss) | ~35% (only non-pure xt positions) |
| Stage disambiguation | Implicit from embedding type (ambiguous) | Not needed (single target) |

## Training

**Input structure:** `[xt | x0]` — same dual structure as block-PDLM. x0 provides clean prefix context only, **no loss computed on x0**.

**5 block states:** For each block on the xt side, randomly sample r ∈ {0,1,2,3,4}:

```
r=0: {m, m, m, m}  — all mask (Stage 1)
r=1: {g, g, g, g}  — all group (full denoise)
r=2: {p, g, g, g}  — 1 pure, 3 group
r=3: {p, p, g, g}  — 2 pure, 2 group
r=4: {p, p, p, g}  — 3 pure, 1 group
```

**Target:** Always self's pure token. Pure positions are **masked off** from loss (trivially correct).

**Loss:** Unified CE loss over all non-pure positions in xt. No stage1/stage2 split.

**Data efficiency:** Average loss-contributing positions per block: (4+4+3+2+1)/5 = 2.8/4 = 70% of xt. Since x0 computes no loss, effective utilization is ~35% of total positions.

### Padding-1

**4 block states (merged variant):** Merge r=0 (all mask) and r=1 (all group) into a single state. The first step goes directly from all-mask to decoding the first position, so each block takes 4 steps instead of 5. For each block, randomly sample r ∈ {0,1,2,3}:

```
r=0: {m, m, m, m}  — all mask (first step decodes pos 0)
r=1: {p, g, g, g}  — 1 pure, 3 group
r=2: {p, p, g, g}  — 2 pure, 2 group
r=3: {p, p, p, g}  — 3 pure, 1 group
```

Data efficiency: (4+3+2+1)/4 = 2.5/4 = 62.5% of xt. Overall ~31.25%.

**Dynamic softmapping ratio:** Instead of hardcoding the softmapping (top-k collapse) ratio, make it adaptive during training:
1. First train the model on `{m,m,m,m} → {p,g,g,g}` (the r=0 state)
2. Compute the accuracy rate from the model's predictions
3. Use the accuracy rate to dynamically set the softmapping ratio as training progresses

This replaces the fixed top-k with a curriculum that tightens the collapse as the model improves.

## Inference

5 steps per block (block_size=4):

```
Step 1 (Stage 1): append {m,m,m,m} → pure logits → collapse → {g0,g1,g2,g3}

Step 2: {g0,g1,g2,g3} → predict p0',p1',p2',p3'
        Sample p0. Refine: g1=collapse(p1'), g2=collapse(p2'), g3=collapse(p3')

Step 3: {p0,g1,g2,g3} → predict _,p1',p2',p3'
        Sample p1. Refine: g2=collapse(p2'), g3=collapse(p3')

Step 4: {p0,p1,g2,g3} → predict _,_,p2',p3'
        Sample p2. Refine: g3=collapse(p3')

Step 5: {p0,p1,p2,g3} → predict _,_,_,p3'
        Sample p3. Block complete → next block.
```

**Top-k collapse** (same as block-PDLM, default k=256): Applied at every pure→group conversion — Stage 1 group prediction and group token refinement during iterative steps.

## Attention

Same as block-PDLM: **causal across blocks, bidirectional within block**.

## Evaluation

**First impl:** Unified CE loss on held-out data + dump eval (generate tokens, inspect manually). Granular breakdowns (loss by state, per-stage) deferred to later.

## References

- Block-PDLM design: `design/block_pdlm.md`
- Base implementation: `nanochat/pdlm.py`
- Base dataloader: `nanochat/dataloader_pdlm.py`
- Training script: `launch/run_block_pdlm.sh`

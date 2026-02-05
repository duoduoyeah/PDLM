# Gradient Tracking for Block-PDLM

## What It Is

Per-position gradient tracking measures the gradient magnitude contributed by each token position in a block during training.

In block-PDLM with `block_size=4`, we predict 4 tokens simultaneously. The total loss is:

```
Total Loss = L_pos0 + L_pos1 + L_pos2 + L_pos3
```

Gradient tracking computes the gradient norm for each position's loss **before** they're combined:

```
∇L_pos0, ∇L_pos1, ∇L_pos2, ∇L_pos3
```

This reveals which positions actually drive parameter updates during optimization.

## Why We Do It

**Problem:** Loss values don't tell us which positions dominate training.

Current s1b eval shows:
- pos 0: loss=1.57 (22% prob)
- pos 1: loss=2.59 (8% prob)
- pos 2: loss=3.42 (3.3% prob)
- pos 3: loss=4.05 (1.8% prob)

**Two very different scenarios are possible:**

**Scenario 1:** Position 0 has small gradients (saturated), positions 2-3 have huge gradients.
- Interpretation: Model IS trying to learn later positions, they're just fundamentally harder
- Action: May need architectural changes or different training strategy

**Scenario 2:** All positions have similar gradient magnitudes.
- Interpretation: Model treats all positions equally, pos 0 isn't "cheating"
- Action: May need to reweight positions or investigate why later positions are harder

**Scenario 3:** Position 0 has huge gradients, later positions have small gradients.
- Interpretation: Model is overfitting to make pos 0 easy, neglecting joint generation
- Action: Need to rebalance or constrain pos 0 predictions

Gradient tracking tells us which scenario we're in, enabling targeted fixes rather than guessing.

## How to Read Gradient Metrics

For each position, we track:
- **Loss value**: How wrong the prediction is
- **Gradient norm** (`||∇L_pos||`): How much this position pushes parameter updates

**Example interpretation:**

```
Position | Loss  | Grad Norm | % of Total Grad
---------|-------|-----------|----------------
pos 0    | 1.57  | 12.3      | 15%
pos 1    | 2.59  | 18.7      | 23%
pos 2    | 3.42  | 25.4      | 31%
pos 3    | 4.05  | 25.1      | 31%
```

**Reading:** Loss degrades across positions (1.57→4.05), but gradient contribution is roughly balanced. Position 0 has the lowest loss AND lowest gradient influence. This suggests:
- pos 0 is easier to predict (low loss)
- pos 0 is saturated (low gradient)
- Later positions drive most of the learning (60%+ of gradient mass)

**Contrast with:**

```
Position | Loss  | Grad Norm | % of Total Grad
---------|-------|-----------|----------------
pos 0    | 1.57  | 45.2      | 72%
pos 1    | 2.59  | 8.1       | 13%
pos 2    | 3.42  | 5.3       | 8%
pos 3    | 4.05  | 4.4       | 7%
```

**Reading:** pos 0 dominates training (72% of gradient). Despite having the lowest loss, it produces the most "learning signal". This is suspicious - suggests pos 0 might be "cheating" by learning surface patterns that are easy to optimize but don't help joint generation.

## Experimental Setup

We will train two models with gradient tracking enabled:

1. **Model A**: `block_size=4` (baseline, matches current s1b)
2. **Model B**: `block_size=8` (2x longer blocks)

**Hypothesis test:**

If the degradation pattern (pos 0 good, later positions poor) is fundamental to block prediction:
- Model B should show **steeper** degradation: pos 0 OK, pos 7 terrible
- Gradient distribution should look similar proportionally

If pos 0 is "cheating" by learning trivial patterns:
- Model B should show similar pos 0 performance
- Gradient % for pos 0 should remain high across both models

If later positions are just harder due to task structure:
- Both models should show gradient concentrated on later positions
- Model B's later positions should get more gradient mass

## Per-Group Gradient Direction Compatibility

Beyond gradient **magnitude** (how much each group pushes), we can measure gradient **direction** (do groups push the same way or fight each other).

**Setup:**
- Use `loss_reduction='none'` to get per-token losses
- Group tokens by kind (e.g., 4 kinds × 128 tokens each)
- Mean loss per group, then 1 backward pass per group with `retain_graph=True`
- Compare gradient vectors via cosine similarity

**Cost:** 1 forward + K backward (K = number of groups). Cheap enough to run every N steps.

**Which layers to check:**
- lm_head: too shallow, only reflects output distribution agreement
- Middle transformer block: most representative of overall optimization dynamics
- Best: check early/middle/late + lm_head for a depth profile
- Checking more layers is free — backward already computes all gradients, just snapshot more

**Interpreting cosine similarity between group gradients:**
- cos > 0: groups cooperate (training one helps the other)
- cos ≈ 0: groups are orthogonal (independent, no conflict)
- cos < 0: groups conflict (improving one hurts the other)

**Key detail:** gradient compatibility can differ by depth. Two groups may agree in early layers but conflict in late layers (or vice versa). Always check the profile, not a single layer.

## Key Insight

From multitask_learning.md: **gradient magnitude determines what drives optimization, not loss value.** A position with 2x higher loss might contribute 10x less to parameter updates if its gradients are smaller.

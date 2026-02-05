# Understanding Multi-Task Learning: Loss Values vs Gradient Magnitudes

## The Core Problem

When training a model with multiple tasks, we combine individual task losses into a single total loss:

```
Total Loss = w₁ × L₁ + w₂ × L₂ + w₃ × L₃ + w₄ × L₄
```

**A common misconception:** If task losses are 1.5, 2.5, 3.5, and 4.5, the model will optimize hardest toward the task with the highest loss (4.5).

**The reality:** What drives optimization is **gradient magnitude**, not loss value. These two things are not the same.

## Why This Matters

Consider a scenario where:
- **Task 1** is the most important but has a low loss (1.5)
- **Tasks 2-4** are less important but have higher losses (2.5, 3.5, 4.5)

Without checking gradient magnitudes, you might assume the model is focusing on the harder tasks (2-4). But if those tasks also produce much larger gradients, they will **dominate** the parameter updates, causing your important Task 1 to be neglected.

## The Disconnect Between Loss and Gradients

**Loss value** tells you how far you are from optimal performance on a task.

**Gradient magnitude** tells you how much the model's parameters will actually change in response to that task.

A task can have:
- High loss but small gradients (barely influences training)
- Low loss but large gradients (dominates the update direction)

This disconnect happens because:
- Different loss functions (cross-entropy vs MSE vs cosine) have inherently different scales
- Task difficulty doesn't correlate with gradient size
- Data distributions affect gradient magnitudes independently of loss values

## How Tasks Actually Compete

When you do backpropagation with combined loss, the final gradient is a weighted sum:

```
∇Total = w₁∇L₁ + w₂∇L₂ + w₃∇L₃ + w₄∇L₄
```

If `||∇L₄||` is 10x larger than `||∇L₁||`, then even with equal weights (w₁ = w₂ = w₃ = w₄ = 1), Task 4 will dominate the parameter updates. The model effectively "pushes harder" toward Task 4, regardless of its loss value.

Research has shown that this gradient dominance doesn't necessarily correlate with learning gains—tasks with large gradients can achieve similar or even lower performance improvements than tasks with smaller gradients.

## Measuring What Really Matters

The key insight: **you can measure individual task gradients even though you combine losses**.

While the final backward pass computes one combined gradient, you can separately compute each task's gradient before combining them. This lets you:

1. **Diagnose** which tasks are actually dominating optimization
2. **Adjust** loss weights to prioritize what matters
3. **Monitor** gradient balance throughout training

Importantly, this measurement can be done on the **validation set without updating the model**. You're simply computing gradients to inspect them, not applying them. This makes it a safe, non-invasive way to understand your model's training dynamics.

## Practical Implications

If Task 1 is your priority but you see:
- Task 1: Loss=1.5, Gradient norm=3.2
- Task 4: Loss=4.5, Gradient norm=35.1

Then Task 4 is getting ~11x more influence on your model's parameters. To fix this, you need to either:
- **Increase Task 1's weight** substantially (e.g., w₁ = 10, w₄ = 1)
- **Normalize gradients** across tasks to equal magnitudes
- **Apply gradient balancing methods** like GradNorm or dual-balancing

## Common Solutions from Research

**Loss-scale balancing:** Apply logarithm transformation to make all loss scales similar before combining.

**Gradient normalization:** Normalize all task gradients to the same magnitude (often the maximum gradient norm).

**Dynamic weighting:** Automatically adjust weights during training based on gradient norms and training rates.

**Gradient surgery:** Modify conflicting gradients to prevent tasks from interfering with each other.

## The Bottom Line

Don't trust loss values alone. In multi-task learning, **gradient magnitudes determine which tasks actually drive optimization**. Measure them, especially on validation data, to understand what your model is really learning—and adjust accordingly.

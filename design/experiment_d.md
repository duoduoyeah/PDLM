# Experiment D Design

## Goal

Prove that Stage 1 (MASK/MTP → Group) can achieve high transition accuracy. If successful, combined with Stage 2 (Group → Pure, validated in Experiment C), PDLM achieves B tokens per 2 forward passes - a real throughput gain over AR (B passes) and BD3LM (~T passes).

## Two-Stage Design

```
Stage 1: MASK/MTP → Group    (predict which group the token belongs to)
Stage 2: Group → Pure        (predict exact token given group hint)
```

**Why decompose?** Predicting distant tokens (i+k) directly is hard. Group tokens reduce the search space - instead of choosing from 4096 tokens, first narrow to ~64 groups, then pick from ~64 tokens within the group.

## Stage 1 Approaches

| Approach | Input | Output | Notes |
|----------|-------|--------|-------|
| MASK-based | `[prefix, MASK, MASK, ...]` | group tokens | Explicit position markers |
| MTP-based | `[prefix]` | group tokens via MTP head | Simpler, no MASK needed |
| Block-based | `[pure tokens]` | group tokens via lm_head | No extra head, block-causal mask |

**Block-based approach**: Position k in block i predicts the group token at position k in block i+1. Uses block-causal attention (bidirectional within block, causal across blocks). Loss computed on blocks 1..N-1 (block 0 has no prior context). Simpler than MTP - reuses lm_head, no separate MTP head needed.

## Loss Weighting

Exponential decay: `weight[k] = β^k / Σβ^k` (β=0.8, or β=1.0 for uniform)

**Why?** Distant tokens (t+3, t+4) have higher loss than near tokens (t+1). Without weighting, high-loss distant tokens dominate gradients. Exponential decay balances contribution across positions.

Note: This differs from FastMTP's motivation (speculative decoding acceptance). Here all positions matter equally for end-to-end decoding - weighting is purely for training stability.

## Overlap Support (overlap_k > 1)

When `overlap_k > 1`, each pure token belongs to **multiple** valid groups.

**Any-correct loss:**
```python
# valid_targets: (*, overlap_k) - all valid group indices
log_probs = F.log_softmax(logits, dim=-1)
valid_log_probs = torch.gather(log_probs, dim=-1, index=valid_targets)
loss = -logsumexp(valid_log_probs, dim=-1).mean()
```

**Key properties:**
- `loss = -log(Σ P(g) for g in valid_groups)`
- If model puts 100% prob on valid groups → loss = 0
- Penalizes probability "leaked" to invalid groups
- Degenerates to standard CE when overlap_k = 1

## Success Metric: Transition Accuracy

A transition is **successful** if the predicted group contains the target token.

```python
transition_accuracy = mean(target_token in group_members[predicted_group])
```

- Baseline (random): 1/num_groups
- Target: >90% for viable end-to-end decoding

## TODO

- [x] Implement Stage 1 MTP training (predict group tokens from pure prefix)
- [ ] Evaluate transition accuracy per position (does i+1 differ from i+4?)
- [ ] End-to-end decoding: Stage 1 + Stage 2 combined

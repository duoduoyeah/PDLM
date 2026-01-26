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
| MTP-based | `[prefix]` | group tokens | Simpler, no MASK needed |

**Start with**: MTP-based (simpler architecture, no new vocab tokens needed).

## Success Metric: Transition Accuracy

A transition is **successful** if the predicted group contains the target token.

```python
transition_accuracy = mean(target_token in group_members[predicted_group])
```

- Baseline (random): 1/num_groups
- Target: >90% for viable end-to-end decoding

## TODO

- [ ] Implement Stage 1 MTP training (predict group tokens from pure prefix)
- [ ] Evaluate transition accuracy per position (does i+1 differ from i+4?)
- [ ] End-to-end decoding: Stage 1 + Stage 2 combined

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

**MTP-based approach**: Predicts multiple future tokens by iteratively feeding the previous target's group embedding back into the MTP block.

```
Current Flow:
k=0: h → lm_head → pure_logits
k=1: h + group_wte[group_of(pure_target[0])] → MTP_block → lm_head → pure_logits
k=2: h + group_wte[group_of(pure_target[1])] → MTP_block → lm_head → pure_logits
k=3: h + group_wte[group_of(pure_target[2])] → MTP_block → lm_head → pure_logits
```

### MTP Head Input: group_wte vs pure_wte

The MTP head uses **teacher forcing**: during training, feed ground truth targets (not model predictions) as input to subsequent steps. This prevents error compounding.

Current implementation uses **group_wte** (not pure_wte) for teacher forcing. This affects hidden representation:

| Input embedding | Hidden becomes | Implication |
|-----------------|----------------|-------------|
| group_wte | group-aligned | Different tokenizers → different losses; fits Stage 2 interface |
| pure_wte | pure-aligned | Tokenizer-independent loss; Stage 2 interface mismatch |

**Current choice: group_wte** — Stage 2 expects group-aligned hidden states, so Stage 1 must "think in groups".

**Block-based approach**: Position k in block i predicts the group token at position k in block i+1. Uses block-causal attention (bidirectional within block, causal across blocks). Loss computed on blocks 1..N-1 (block 0 has no prior context). Simpler than MTP - reuses lm_head, no separate MTP head needed.

## Stage 1 Target Modes

Two modes for Stage 1 training targets:

### Pure-Target Mode (Default)

**Training:**
- `lm_head` shape: `(pure_vocab_size, n_embd)` e.g., `(4096, 768)`
- Output: logits over **pure tokens**
- Loss: standard CE against **pure token targets**

**Inference:**
- Pre-compute: `group_head = group_to_pure_mask @ lm_head.weight` → `(num_groups, n_embd)`
- Use `group_head` like any other linear layer
- Argmax over group logits = group with highest summed pure logits among its members

```python
# Pre-compute once before inference
group_head = token_map.group_to_pure_mask.float() @ lm_head.weight  # (64, 768)

# During inference
group_logits = hidden @ group_head.T  # (B, T, 64)
predicted_group = group_logits.argmax(dim=-1)
```

**Why pure-target mode?**
1. **Stage 2 error correction**: Even if Stage 1 doesn't predict the exact token, it may still put high probability on tokens within the correct group. Stage 2 can then correct within-group errors. This makes the two-stage system more robust than requiring Stage 1 to perfectly predict groups.
2. **Simpler training**: Standard next-token prediction objective, no special group-aware loss needed.
3. **Richer signal**: Learning full token distribution provides more gradient signal than coarse group targets.

### Group-Target Mode (Legacy)

**Training:**
- `lm_head` shape: `(num_groups, n_embd)` e.g., `(64, 768)`
- Output: logits over **group tokens**
- Loss: any-correct CE against **group token targets** (handles overlap_k > 1)

**Inference:**
- Directly use `lm_head` for group predictions

Use `--stage1_target_mode=group` to enable legacy mode.

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
- [x] Design pure-target mode for Stage 1 (train with pure targets, collapse to groups at inference)
- [ ] Implement pure-target mode in code (default) with group-target mode as legacy option
- [ ] Evaluate transition accuracy per position (does i+1 differ from i+4?)
- [ ] End-to-end decoding: Stage 1 + Stage 2 combined

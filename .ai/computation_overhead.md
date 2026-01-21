# PDLM Computational Overhead Analysis

## vs Standard GPT

Standard GPT forward pass:
- Input: (B, T) token ids
- Embedding lookup: (B, T) → (B, T, D)
- Transformer blocks: (B, T, D) → (B, T, D)
- LM head: (B, T, D) → (B, T, V)
- Loss: cross-entropy on (B, T, V) vs targets

---

## PDLM Extra Computations

### 1. Training: Doubled Sequence (pdlm.py:254)

```python
idx = torch.cat((idx, targets), dim=1)  # (B, T) → (B, 2T)
```

- Transformer processes 2T instead of T
- Attention is O((2T)²) = 4x attention cost
- But only first T positions contribute to loss

### 2. Larger Embedding Table (pdlm.py:144)

```python
wte = nn.Embedding(all_vocab_size, n_embd)  # pure + group + MASK tokens
```

- Memory: extra (num_group_tokens + 1) * D parameters
- Compute: same (just lookup)

### 3. Token Map Operations (token_map.py)

#### 3.1 noise_tokens() - Training time

```python
options = pure_to_noisy_map[pure_ids, noisy_levels]  # (B, T) → (B, T, fanout)
# then sample one from fanout
```

- Tensor indexing: O(B * T)
- Random sampling: O(B * T)

#### 3.2 transit_noisy_tokens() - Inference time (OLD design)

```python
# OLD: complex aggregation with topk and scatter_add
acc_probs.scatter_add_(2, flat_options, flat_weights)
return acc_probs.argmax(dim=-1)
```

- Scatter add: O(B * T * K * fanout) - expensive
- Argmax over vocab: O(B * T * V)

#### 3.2 NEW design: Direct prediction

With lm_head outputting `pure + group` tokens:
- Stage 1 (MASK → Group): argmax over group token range
- Stage 2 (Group → Pure): argmax over pure token range

No scatter_add needed. Just masked argmax: O(B * T)

#### 3.3 get_random_noisy_level() - Training time

```python
dist = Binomial(max_level, probs=step/total_steps)
levels = dist.sample()
```

- Binomial sampling: O(B * T)

---

## Summary Table

| Operation | When | Complexity | Notes |
|-----------|------|------------|-------|
| 2x sequence length | Train | O(4T² attention) | Main overhead |
| Larger wte | Both | O(1) extra memory | Negligible compute |
| Larger lm_head | Both | O(D * num_groups) extra | NEW: output group tokens |
| noise_tokens | Train | O(B*T) | Cheap |
| get_random_noisy_level | Train | O(B*T) | Cheap |
| transit (OLD) | Inference | O(B*T*K*fanout) + O(B*T*V) | scatter_add - expensive |
| transit (NEW) | Inference | O(B*T) | masked argmax - cheap |

---

## Questions to Address

1. Can we reduce the 2T training overhead? (e.g., separate forward for noisy/clean?)
2. transit_noisy_tokens with topk: is scatter_add + argmax the bottleneck?
3. For single-layer group tokens, do we need all this complexity?

---

## TODO: Redesign for Stage 1

For single-layer group tokens with simple parameters (num_groups, overlap_k), what's the minimal token_map interface?

---

## Profiling Tools

PyTorch profiling options:
- `torch.profiler` - built-in, shows CPU/CUDA time per op
- `torch.cuda.Event` - simple start/end timing
- `torch.autograd.profiler` - legacy but still useful

Example:
```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Scope

**Focus now**: Inference/eval overhead

**TODO later**: Training overhead analysis (2x sequence, noise_tokens, etc.)

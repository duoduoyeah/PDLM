# Overlap Design via Sub-groups

## Three Key Variables

```
num_sub        = number of sub-groups (built by clustering)
sub_per_final  = sub-groups per final group
overlap_k      = how many final groups each token appears in
```

**Derived:**
```
num_final = (num_sub × overlap_k) / sub_per_final
tokens_per_final = (vocab_size / num_sub) × sub_per_final
```

---

## Example: 8 sub-groups, 2 per final, overlap_k=3

```
num_final = (8 × 3) / 2 = 12 final groups
tokens_per_final = (4096 / 8) × 2 = 1024 tokens
```

```
          S1  S2  S3  S4  S5  S6  S7  S8  │ row sum
──────────────────────────────────────────────────
G1         1   1   0   0   0   0   0   0  │   2
G2         1   0   1   0   0   0   0   0  │   2
G3         1   0   0   1   0   0   0   0  │   2
G4         0   1   1   0   0   0   0   0  │   2
G5         0   1   0   1   0   0   0   0  │   2
G6         0   0   1   1   0   0   0   0  │   2
G7         0   0   0   0   1   1   0   0  │   2
G8         0   0   0   0   1   0   1   0  │   2
G9         0   0   0   0   1   0   0   1  │   2
G10        0   0   0   0   0   1   1   0  │   2
G11        0   0   0   0   0   1   0   1  │   2
G12        0   0   0   0   0   0   1   1  │   2
──────────────────────────────────────────────────
col sum    3   3   3   3   3   3   3   3    ← overlap_k=3 ✓
```

---

## Comparison: (12,3,3) vs (8,2,3)

Both achieve **noise level = 1024 tokens/final group** with **overlap_k=3**:

| Setting | num_sub | sub_per_final | overlap_k | num_final | tokens/sub | tokens/final |
|---------|---------|---------------|-----------|-----------|------------|--------------|
| A | 12 | 3 | 3 | 12 | 341 | 1024 |
| B | 8 | 2 | 3 | 12 | 512 | 1024 |

### Differences

| Aspect | (12,3,3) | (8,2,3) |
|--------|----------|---------|
| **Sub-group size** | 341 tokens (finer) | 512 tokens (coarser) |
| **Clustering precision** | More precise (smaller clusters) | Less precise (larger clusters) |
| **Pairwise final group overlap** | Can share 1/3, 2/3, or 3/3 | Can share 0 or 1/2 only |
| **Semantic coherence** | Sub-groups more coherent | Sub-groups more diverse |

### Pairwise overlap between final groups

**(12,3,3):** Two final groups can share 0, 1, 2, or 3 sub-groups
- Share 1 sub: 341 tokens overlap (1/3 of final)
- Share 2 sub: 682 tokens overlap (2/3 of final)
- More gradual overlap spectrum

**(8,2,3):** Two final groups can share 0 or 1 sub-group
- Share 1 sub: 512 tokens overlap (1/2 of final)
- Binary: either half-overlap or no overlap

### Which to choose?

- **(12,3,3)**: Finer clustering, smoother overlap gradients, more sub-groups to manage
- **(8,2,3)**: Coarser clustering, simpler structure, fewer sub-groups

**Recommendation**: Start with (8,2,3) for simplicity, try (12,3,3) if need finer control.

---

## Complete Design: All Combinations C(num_sub, sub_per_final)

Instead of controlling overlap_k exactly, use **all possible combinations**.

**Formula:**
```
num_final = C(num_sub, sub_per_final)
overlap_k = C(num_sub - 1, sub_per_final - 1)
tokens_per_final = (vocab_size / num_sub) × sub_per_final
```

### All configs achieving noise level = 1024 (vocab_size = 4096)

| num_sub | sub_per_final | num_final | overlap_k | practical? |
|---------|---------------|-----------|-----------|------------|
| 4 | 1 | 4 | 1 | ✓ baseline (no overlap) |
| 8 | 2 | 28 | 7 | ✓ good |
| 12 | 3 | 220 | 55 | ✓ maybe |
| 16 | 4 | 1,820 | 455 | ? too many groups |
| 20 | 5 | 15,504 | 3,876 | ✗ explodes |

**Key insight:** num_final grows as C(n, n/4) which explodes quickly.

### Practical choices for experiments

| Config | num_final | overlap_k | Use case |
|--------|-----------|-----------|----------|
| (4,1) | 4 | 1 | Baseline, no overlap |
| (8,2) | 28 | 7 | Moderate overlap |
| (12,3) | 220 | 55 | High overlap (if lm_head size manageable) |

### Tool

Use `scripts/overlap_calc.py` to explore:
```bash
uv run -m scripts.overlap_calc --num_sub 8 --sub_per_final 2
uv run -m scripts.overlap_calc --target_noise 1024 --vocab_size 4096
```

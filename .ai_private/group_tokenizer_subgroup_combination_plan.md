# Group Tokenizer Rewrite: Sub-group Combination Approach

## Overview

Rewrite the group tokenizer to support two overlap approaches:
1. **all_combos**: All C(num_sub, sub_per_final) combinations, overlap_k derived
2. **flexible**: Custom overlap_k with balanced design algorithm

## Target Configs (14 total)

### All Combos Mode (9 configs)
| Name | num_sub | sub_per_final | groups | k |
|------|---------|---------------|--------|---|
| n1024_k1_g4 | 4 | 1 | 4 | 1 |
| n1024_k7_g28 | 8 | 2 | 28 | 7 |
| n1024_k55_g220 | 12 | 3 | 220 | 55 |
| n512_k1_g8 | 8 | 1 | 8 | 1 |
| n512_k15_g120 | 16 | 2 | 120 | 15 |
| n256_k1_g16 | 16 | 1 | 16 | 1 |
| n256_k31_g496 | 32 | 2 | 496 | 31 |
| n64_k1_g64 | 64 | 1 | 64 | 1 |
| n16_k1_g256 | 256 | 1 | 256 | 1 |

### Flexible Mode (3 configs)
| Name | num_sub | sub_per_final | k | groups |
|------|---------|---------------|---|--------|
| n64_k4_g256 | 128 | 2 | 4 | 256 |
| n16_k2_g512 | 512 | 2 | 2 | 512 |
| n16_k4_g1024 | 512 | 2 | 4 | 1024 |

Plus `_mask` variants = 28 tokenizers total.

---

## Files to Modify

### 1. `nanochat/group_tokenizer/config.py`
**Replace config class with new parameters:**

```python
@dataclass
class GroupTokenizerConfig:
    num_sub: int                                    # Sub-groups from clustering
    sub_per_final: int = 1                          # Sub-groups per final group
    overlap_mode: Literal["all_combos", "flexible"] = "all_combos"
    overlap_k: Optional[int] = None                 # Only for flexible mode
    include_mask: bool = True
    clustering_method: str = "kmeans"
    random_seed: int = 42

    # Properties: num_groups, effective_overlap_k (derived)
    # Method: get_output_name() -> "n1024_k7_g28(_mask)"
```

### 2. `nanochat/group_tokenizer/clustering.py`
**Add new functions, remove old overlap logic:**

```python
# KEEP: kmeans_clustering(), random_clustering()

# ADD:
def build_all_combos_assignment(num_sub, sub_per_final) -> Tensor:
    """Generate all C(num_sub, sub_per_final) combinations.
    Returns: (num_final, num_sub) boolean matrix"""

def build_flexible_assignment(num_sub, sub_per_final, overlap_k, seed) -> Tensor:
    """Greedy balanced design: each row has sub_per_final, each col has overlap_k.
    Returns: (num_final, num_sub) boolean matrix"""

def derive_pure_to_group(sub_assignments, sub_to_final) -> Tuple[Tensor, Tensor]:
    """Convert sub-group assignments + assignment matrix to final tensors.
    Returns: pure_to_group (vocab, k), group_to_pure_mask (groups, vocab)"""

# REMOVE: compute_overlap_assignments() (old top-k centroid approach)
```

### 3. `nanochat/group_tokenizer/builder.py`
**Update build flow:**

```python
def build(self, config):
    # Step 1: Cluster into SUB-GROUPS (not final groups)
    sub_assignments = kmeans_clustering(embeddings, config.num_sub, ...)

    # Step 2: Build assignment matrix based on mode
    if config.overlap_mode == "all_combos":
        sub_to_final = build_all_combos_assignment(num_sub, sub_per_final)
    else:  # flexible
        sub_to_final = build_flexible_assignment(num_sub, sub_per_final, overlap_k, seed)

    # Step 3: Derive final tensors
    pure_to_group, group_to_pure_mask = derive_pure_to_group(sub_assignments, sub_to_final)

    # Step 4-5: Build token maps and extended tokenizer (mostly unchanged)
```

### 4. `scripts/build_group_tokenizer.py`
**New CLI interface:**

```bash
# All combos mode:
uv run -m scripts.build_group_tokenizer \
    --checkpoint-dir PATH --output-dir PATH \
    --mode all_combos --num-sub 8 --sub-per-final 2

# Flexible mode:
uv run -m scripts.build_group_tokenizer \
    --checkpoint-dir PATH --output-dir PATH \
    --mode flexible --num-sub 128 --sub-per-final 2 --overlap-k 4

# With mask:
uv run -m scripts.build_group_tokenizer ... --mask  # (default: no mask)
```

### 5. `nanochat/group_tokenizer/overlap_calc.py`
**Add flexible mode calculation:**

```python
def calc_flexible_design(num_sub, sub_per_final, overlap_k, vocab_size):
    """Calculate params for flexible mode."""
    num_final = (num_sub * overlap_k) // sub_per_final
    ...
```

---

## Key Algorithm: Balanced Design (flexible mode)

```python
def build_flexible_assignment(num_sub, sub_per_final, overlap_k, seed):
    num_final = (num_sub * overlap_k) // sub_per_final
    assignment = zeros(num_final, num_sub, bool)
    col_counts = zeros(num_sub)  # Track appearances per sub-group

    for f in range(num_final):
        # Find sub-groups with room (col_counts < overlap_k)
        available = where(col_counts < overlap_k)
        # Pick sub_per_final with lowest counts (greedy balance)
        # Random tie-breaking
        selected = topk_lowest(col_counts[available], sub_per_final)
        assignment[f, selected] = True
        col_counts[selected] += 1

    # Verify: each row = sub_per_final, each col = overlap_k
    return assignment
```

---

## Data Flow

```
Token t  -->  Sub-group s  -->  Final groups [f1, f2, ..., fk]
         (kmeans)          (assignment matrix)

sub_assignments: (vocab_size,) - which sub-group each token belongs to
sub_to_final: (num_final, num_sub) - which sub-groups in each final group
pure_to_group: (vocab_size, overlap_k) - final group IDs for each token
group_to_pure_mask: (num_final, vocab_size) - member tokens per group
```

---

## Implementation Order

1. **config.py** - New config class with derived properties
2. **clustering.py** - Add assignment matrix builders
3. **builder.py** - Update build flow
4. **build_group_tokenizer.py** - New CLI
5. **overlap_calc.py** - Add flexible mode (optional)

---

## Verification

After implementation, test with:
```bash
# Build one config
uv run -m scripts.build_group_tokenizer \
    --checkpoint-dir /path/to/checkpoints \
    --output-dir /tmp/test_tokenizer \
    --mode all_combos --num-sub 8 --sub-per-final 2

# Verify output
uv run -m nanochat.group_tokenizer.dump /tmp/test_tokenizer
```

Check:
- `token_maps.pt` has correct shapes
- `pure_to_group.shape == (4096, 7)` for n1024_k7_g28
- `group_to_pure_mask.shape == (28, 4096)`
- Each token appears in exactly k=7 groups
- Each group has ~1024 tokens (noise level)

---

## No Changes Needed

- `token_map.py` - Runtime interface unchanged (same tensor format)
- `dump.py` - Should work with new format
- Dataloaders - Use same pure_to_group/group_to_pure_mask tensors

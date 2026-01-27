# Group Tokenizer Usage Guide

## Build Tokenizer

```bash
# Build n1024_k7_g28 (8 sub-groups, 2 per final = C(8,2)=28 groups, k=7 overlap)
LOCAL_OUTPUT=/path/to/output ./launch/build_group_tokenizer.sh --mode all_combos --num-sub 8 --sub-per-final 2

# Build n512_k15_g120 (16 sub-groups, 2 per final = C(16,2)=120 groups, k=15 overlap)
LOCAL_OUTPUT=/path/to/output ./launch/build_group_tokenizer.sh --mode all_combos --num-sub 16 --sub-per-final 2

# Build with mask token
LOCAL_OUTPUT=/path/to/output ./launch/build_group_tokenizer.sh --mode all_combos --num-sub 8 --sub-per-final 2 --mask

# Flexible mode (custom k)
LOCAL_OUTPUT=/path/to/output ./launch/build_group_tokenizer.sh --mode flexible --num-sub 128 --sub-per-final 2 --overlap-k 4
```

**Environment variables:**
- `LOCAL_OUTPUT`: Output directory (required)
- `DRIVE_CHECKPOINT`: Custom checkpoint path (optional)
- `DRIVE_TOKENIZER`: Custom base tokenizer path (optional)
- `MODEL_TAG`: Specific model checkpoint tag (optional)

## Dump/Inspect Tokenizer

```bash
# Overview (shapes, group size stats)
python -m nanochat.group_tokenizer.dump /path/to/tokenizer

# Specific group details
python -m nanochat.group_tokenizer.dump /path/to/tokenizer --group 5

# Specific token details
python -m nanochat.group_tokenizer.dump /path/to/tokenizer --token 123

# All groups summary
python -m nanochat.group_tokenizer.dump /path/to/tokenizer --all-groups

# Overlap quality analysis (important for k>1)
python -m nanochat.group_tokenizer.dump /path/to/tokenizer --overlap-quality

# Overlap quality with more samples
python -m nanochat.group_tokenizer.dump /path/to/tokenizer --overlap-quality --sample-tokens 10
```

## Overlap Quality Metrics

| Metric | Good Value | Meaning |
|--------|------------|---------|
| Jaccard similarity | Lower | Groups sharing a token are different |
| Shared ratio | Lower | Fewer tokens appear in ALL k groups |
| Diversity score | Higher | Token sees more of vocab across k groups |

**Expected values for all_combos with sub_per_final=2:**
- Jaccard ~= 1 / num_sub
- Shared ratio ~= 1 / num_sub
- Diversity score ~= (num_sub - 1) / num_sub

Example n1024_k7_g28 (num_sub=8):
- Jaccard ~0.125, Shared ~12.5%, Diversity ~87.5%

## Target Configs Reference

### All Combos Mode
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

### Flexible Mode
| Name | num_sub | sub_per_final | k | groups |
|------|---------|---------------|---|--------|
| n64_k4_g256 | 128 | 2 | 4 | 256 |
| n16_k2_g512 | 512 | 2 | 2 | 512 |
| n16_k4_g1024 | 512 | 2 | 4 | 1024 |

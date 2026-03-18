# Robustness to Group Granularity — Experiment Design

## Goal
Show PDLM is robust to group granularity. It works across different group sizes, not that one specific setting is optimal.

## Design

5 data points varying tokens_per_group from fine (64) to coarse (1024).
Finer groups → more num_groups (more choices for the model).
Overlap_k kept at 31 where possible for consistency.

| tokens_per_group | num_groups (g) | num_sub | sub_per_final | overlap_k | mode     | status            |
|------------------|---------------|---------|---------------|-----------|----------|-------------------|
| 64               | 1984          | 64      | 1             | 31        | flexible | new tokenizer+model |
| 128              | 992           | 32      | 1             | 31        | flexible | new tokenizer+model |
| 256              | 496           | 32      | 2             | 31        | all_combos | trained (4s)     |
| 512              | 120           | 16      | 2             | 15        | all_combos | trained (4s)     |
| 1024             | 220           | 16      | 4             | 55        | all_combos | tokenizer exists, need 4s model |

## Training
- All models: 4s (4-state), same p value (p=50 or whichever is standard)
- Same architecture, same data ratio (r=40)
- Block size 4

## Table Design (paper)

| tokens_per_group | num_groups | PPL |
|------------------|-----------|-----|
| 64               | 1984      | --  |
| 128              | 992       | --  |
| 256              | 496       | --  |
| 512              | 120       | --  |
| 1024             | 220       | --  |

# Paper Tables Plan

All experiments use data-to-parameter ratio 40. Model size 30M. Dataset: TinyStories.

## Table 1+2: Exp A/B (Analysis section)
- **Purpose**: Show distance degrades prediction, mask tokens don't help
- **Rows**: target_shift / position k (1, 2, 3, 4)
- **Columns**: AR loss (target_shift=k), BD3-LM loss (predict i+k with mask fill)
- **Models needed**: AR with target_shift 1-4, BD3-LM with block_size 4

## Table 3: Exp C + D — Group tokens as noisy intermediates
- **Purpose**: Show group tokens carry real info (exp-c), and stage 1 can produce them (exp-d)
- **Rows**: num_groups (16, 64, 256, 1024) × position (avg, pos0, pos1, pos2, pos3)
- **Columns**: Oracle PPL (exp-c, ground truth G given), Acc (exp-d, stage-1 group prediction accuracy)
- **Models needed**: mask-PDLM with different num_groups, fixed block_size=4, overlap_k=1

## Table 4: End-to-End Results — PDLM vs BD3-LM vs AR
- **Purpose**: Headline result; also shows block size effect folded in
- **Rows**: AR (no block size), BD3-LM × block_size (2,4,8,16), PDLM × block_size (2,4,8,16)
- **Columns**: PPL (realistic decoding: predicted G, ground truth P), Entropy PPL (target-free)
- **Metric note**: PPL uses teacher-forced P but model-predicted G from previous step (not oracle). Using ground truth G would reduce to exp-c and be artificially lower.
- **Models needed**: AR, BD3-LM with block_size 2/4/8/16, mask-PDLM with block_size 2/4/8/16

## Robustness: Group Granularity
- **Purpose**: Show PDLM is not sensitive to num_groups (group token is an intermediate step, not final design)
- **Rows**: num_groups (64, 256, 1024)
- **Columns**: PPL
- **Models needed**: mask-PDLM with different num_groups, fixed block_size=4

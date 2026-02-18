# Paper Tables Plan

All experiments use data-to-parameter ratio 40. Model size 30M. Dataset: TinyStories.

## Tier 1 — Must Have

### Table 1+2: Exp A/B (Analysis section)
- **Purpose**: Show distance degrades prediction, mask tokens don't help
- **Rows**: target_shift / position k (1, 2, 3, 4)
- **Columns**: AR loss (target_shift=k), BD3-LM loss (predict i+k with mask fill), accuracy
- **Models needed**: AR (GPT-2) with target_shift 1-4, BD3-LM with block_size 4

### Table 4: Comparison — PDLM vs BD3-LM vs AR
- **Purpose**: Headline result, head-to-head comparison
- **Rows**: PDLM, BD3-LM, AR
- **Columns**: perplexity, accuracy
- **Models needed**: PDLM (both stages), BD3-LM, AR baseline

## Tier 2 — Important

### Table 3: Exp C — Group tokens help
- **Purpose**: Show group tokens improve prediction at different noise levels
- **Rows**: num_groups (16, 64, 256, 1024)
- **Columns**: loss, accuracy (overall or per-position)
- **Models needed**: PDLM stage2 with different num_groups, fixed block_size=4, overlap_k=1

### Table 6: Per-position breakdown
- **Purpose**: Show where group tokens help most
- **Rows**: position (i+1, i+2, i+3, i+4)
- **Columns**: AR accuracy, BD3-LM accuracy, PDLM accuracy
- **Models needed**: same as Table 4, just different eval

## Tier 3 — Nice to Have

### Table 5: Ablation — block size
- **Purpose**: Quality vs block size tradeoff
- **Rows**: block_size (4, 8)
- **Columns**: perplexity, accuracy
- **Models needed**: PDLM with block_size 4 and 8

### Table 7: Qualitative — blind head-to-head judging
- **Purpose**: Show generation quality beyond perplexity
- **Rows**: PDLM, BD3-LM, AR
- **Columns**: coherence, grammar, quality
- **Method**: same prefix, three models generate, judge scores blindly
- **Models needed**: same as Table 4, plus eval pipeline

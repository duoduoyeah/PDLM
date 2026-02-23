#!/bin/bash
## Launch 10 mask-PDLM sweep jobs (2 tokenizers x 5 soft_p values)
## At most 4 run in parallel (per-user GPU limit).

SCRIPT="slurms/slurm_mask_pdlm.sh"
COMMON="--data_ratio=40 --test_mode=false --depth=8 --block_size=4"

for TOK in "512 15 120" "256 31 496"; do
    read N K G <<< "$TOK"
    for P in 0.3 0.5 0.7 0.9 1.0; do
        sbatch $SCRIPT --noise_level=$N --overlap_k=$K --num_groups=$G --soft_p_within=$P $COMMON
    done
done

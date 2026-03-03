#!/bin/bash -l
#SBATCH --job-name="analysis_exp1"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3:00:00
#SBATCH --output=/rhome/sli588/temp/analysis_exp1_%j.out

## Run Analysis Experiment 1: AR vs BD3-LM paired comparison.
## Produces the table for tab:mask_signal in Analysis.tex.
##
## Usage:
##   sbatch slurms/slurm_analysis_exp1.sh
##   SKIP_EXISTING=true sbatch slurms/slurm_analysis_exp1.sh   # resume interrupted run

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Run
# ============================================================
bash table_script/table_analysis_exp1.sh

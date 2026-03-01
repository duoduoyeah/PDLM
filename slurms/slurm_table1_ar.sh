#!/bin/bash -l
#SBATCH --job-name="table1_ar"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=2:00:00
#SBATCH --output=/rhome/sli588/temp/table1_ar_%j.out

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Run
# ============================================================
bash table_script/table1_ar.sh

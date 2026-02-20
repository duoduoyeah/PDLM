#!/bin/bash -l
#SBATCH --job-name="table1_ar"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=2:00:00
#SBATCH --output=table1_ar_%j.out

# ============================================================
# Setup
# ============================================================
export UV_CACHE_DIR=$SCRATCH/.uv_cache

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env; }

# ============================================================
# Clone repo
# ============================================================
cd $SCRATCH
git clone git@github.com:duoduoyeah/PDLM.git repo_table1_ar
cd repo_table1_ar

# ============================================================
# Python env (builds rustbpe via maturin; requires Rust)
# ============================================================
uv venv
uv sync --extra gpu
source .venv/bin/activate

# ============================================================
# Copy secrets (not in git)
# ============================================================
cp ~/PDLM/launch/.env launch/.env

# ============================================================
# Run
# ============================================================
bash table_script/table1_ar.sh

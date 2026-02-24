#!/bin/bash
## Common setup sourced by all slurm job scripts.
## Usage: source slurms/setup.sh
## Requires REPO_ROOT to be set by the calling script.

if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: REPO_ROOT not set. Set it before sourcing setup.sh."
    return 1
fi
cd "$REPO_ROOT"

# Point uv cache and venv to scratch (keeps home dir clean, local .venv stays CPU)
export UV_CACHE_DIR=$SCRATCH/.uv_cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv_gpu

# Install/sync GPU dependencies into scratch venv
command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env; }
uv sync --extra gpu

# Activate scratch venv
export PATH="$SCRATCH/.venv_gpu/bin:$PATH"

# Load credentials and export to child processes (wandb_v1_* service account key — do NOT use wandb login)
source launch/.env
export WANDB_API_KEY HF_TOKEN

# Store wandb run data on scratch to avoid filling home quota
export WANDB_DIR=$SCRATCH

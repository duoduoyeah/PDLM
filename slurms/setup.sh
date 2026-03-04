#!/bin/bash
## Common setup sourced by all slurm job scripts.
## Usage: source slurms/setup.sh
## Requires REPO_ROOT to be set by the calling script.
##
## For parallel jobs: if GPU_VENV is already set (exported by parent),
## skips uv sync and just activates the existing venv.

if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: REPO_ROOT not set. Set it before sourcing setup.sh."
    return 1
fi
cd "$REPO_ROOT"

if [ -n "${GPU_VENV}" ]; then
    # Child mode: parent already synced, just activate
    export PATH="${GPU_VENV}/bin:$PATH"
else
    # Normal mode: sync venv in repo directory
    command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env; }
    uv sync --extra gpu

    # Export resolved path for child jobs
    export GPU_VENV="${REPO_ROOT}/.venv"
    export PATH="${REPO_ROOT}/.venv/bin:$PATH"
fi

# Load credentials and export to child processes (wandb_v1_* service account key — do NOT use wandb login)
source launch/.env
export WANDB_API_KEY HF_TOKEN

# Store wandb run data on scratch to avoid filling home quota
export WANDB_DIR=$SCRATCH

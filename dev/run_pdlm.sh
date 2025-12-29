#!/bin/bash

# This script is the code that run training and eval on single A100

# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat/pdlm/pdlm_depth4_bs8_pr1_ratio40_causal_samenoisy"
mkdir -p $NANOCHAT_BASE_DIR

python -c "from nanochat.common import get_base_dir; print(get_base_dir())"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
# [ -d ".venv" ] || uv venv
# install the repo dependencies
# uv sync --extra gpu
# when colab
uv pip install  -e .
# activate venv so that `python` uses the project's venv instead of system python
# source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Data & Tokenizer

# ourdataset currently only 17 has shards\
echo "Waiting for dataset download to complete..."
python -m nanochat.dataset -n 17 
echo "dataset download to complete~~~"
#-----------------------------------------------------------------------------
# Base model (pretraining)

# The d12 model is 90M parameters.
# Chinchilla says #tokens = 20X #params, so we need 90e6 * 20 = 1.8B tokens.
# Assume our tokenizer is 5 chars/token, this is 1.8B * 4 ~= 7.2B chars.
# At 250M chars/shard, this is 7.2B / 250M ~= 56 shards needed for pretraining.
# While we only has 17 shards here, so its not enough

# Number of processes/GPUs to use
NPROC_PER_NODE=1

# pdlm_depth4_bs8_pr1_ratio40_causal
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs8_pr1_ratio40_causal \
    --depth=4 \
    --block_size=8 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40

# pdlm_depth4_bs4_pr1_ratio40_causal
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs4_pr1_ratio40_causal \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40

# pdlm_depth4_bs2_pr1_ratio40_causal
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs2_pr1_ratio40_causal \
    --depth=4 \
    --block_size=2 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40

# pdlm_depth4_bs1_pr1_ratio40_causal
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs1_pr1_ratio40_causal \
    --depth=4 \
    --block_size=1 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40

# pdlm_depth4_bs8_pr1_ratio40_causal_samenoisy
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs8_pr1_ratio40_causal_samenoisy \
    --depth=4 \
    --block_size=8 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40 \
    --noise_total_steps=0 \
    --total_batch_size=65536

# pdlm_depth4_bs4_pr1_ratio40_causal_samenoisy
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs4_pr1_ratio40_causal_samenoisy \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40 \
    --noise_total_steps=0

# pdlm_depth4_bs2_pr1_ratio40_causal_samenoisy
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs2_pr1_ratio40_causal_samenoisy \
    --depth=4 \
    --block_size=2 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40 \
    --noise_total_steps=0

# pdlm_depth4_bs1_pr1_ratio40_causal_samenoisy
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs1_pr1_ratio40_causal_samenoisy \
    --depth=4 \
    --block_size=1 \
    --prefix_pure_tokens=1 \
    --is_causal=True \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40 \
    --noise_total_steps=0

# =================================================
# Non causal

# pdlm_depth4_bs8_pr1_ratio40_non_causal
export NANOCHAT_BASE_DIR="/content/drive/MyDrive/pdlm_depth4_bs8_pr1_ratio40_non_causal/"
mkdir -p $NANOCHAT_BASE_DIR
python -m nanochat.report reset

python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs8_pr1_ratio40_non_causal \
    --depth=4 \
    --block_size=8 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40

# pdlm_depth4_bs8_pr1_ratio40_non_causal_samenoisy
python -m scripts.pdlm_base_train \
    --run=pdlm_depth4_bs8_pr1_ratio40_non_causal_samenoisy \
    --depth=4 \
    --block_size=8 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=1024 \
    --device_batch_size=64 \
    --target_param_data_ratio=40 \
    --noise_total_steps=0


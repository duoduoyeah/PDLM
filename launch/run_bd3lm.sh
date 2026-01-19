#!/bin/bash

## BD3LM Training Script
## Test mode: data_ratio=10
## Production mode: data_ratio=20 (or user specified)
##
## Usage:
##   bash launch/run_bd3lm.sh --variant=normal
##   bash launch/run_bd3lm.sh --variant=ts1 --depth=8 --test_mode=false --data_ratio=30

# ============================================================
# Default values
# ============================================================
VARIANT="normal"  # normal, ts1, ts2, ts4
TEST_MODE="true"
DATA_RATIO="10"  # default 10 for test mode
DEPTH="4"  # model depth
BLOCK_SIZE="4"  # block size

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --variant=*)
            VARIANT="${arg#*=}"
            ;;
        --test_mode=*)
            TEST_MODE="${arg#*=}"
            ;;
        --data_ratio=*)
            DATA_RATIO="${arg#*=}"
            ;;
        --depth=*)
            DEPTH="${arg#*=}"
            ;;
        --block_size=*)
            BLOCK_SIZE="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/run_bd3lm.sh --variant=normal [--depth=4] [--block_size=4] [--test_mode=true] [--data_ratio=10]"
            exit 1
            ;;
    esac
done

# Build base model name from depth, block_size, and variant
BASE_MODEL_NAME="bd3lm_d${DEPTH}_b${BLOCK_SIZE}_${VARIANT}"

WANDB_GROUP="bd3lm_d${DEPTH}"
MODEL_REPO="duoduoyeah/bd3lm_d${DEPTH}"
DRIVE_BASE="/content/drive/MyDrive/nanochat"
BASE_TOKENIZER_REPO="/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer_with_mask"

# Load secrets from .env file (created by: %run launch/setup_secrets.py)
if [ -f "launch/.env" ]; then
    source launch/.env
    echo "Loaded secrets from launch/.env"
else
    echo "Warning: launch/.env not found. Run '%run launch/setup_secrets.py' first"
fi

# Wandb login
if [ -n "${WANDB_API_KEY}" ]; then
    wandb login --relogin "${WANDB_API_KEY}"
fi

# Build model name with ratio suffix, and "_test" suffix in test mode
if [ "${TEST_MODE}" = "true" ]; then
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}_test"
else
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}"
fi

# Export environment variables
export MODEL_NAME
export DATA_RATIO
export DEPTH
export WANDB_GROUP
export MODEL_REPO
export NANOCHAT_BASE_DIR="${DRIVE_BASE}/${MODEL_NAME}"
export BASE_TOKENIZER_REPO

echo "=== Running model: ${MODEL_NAME} ==="
echo "=== Base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Test mode: ${TEST_MODE} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Depth: ${DEPTH} ==="

# ============================================================
# Setup (run once per model)
# ============================================================

# Handle existing model dir
if [ -d "${NANOCHAT_BASE_DIR}" ]; then
    if [ "${TEST_MODE}" = "true" ]; then
        # Test mode: remove old dir
        echo "Test mode: Removing old model dir ${NANOCHAT_BASE_DIR}"
        rm -rf "${NANOCHAT_BASE_DIR}"
    else
        # Production mode: backup old dir with timestamp
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR="${NANOCHAT_BASE_DIR}_backup_${TIMESTAMP}"
        echo "Production mode: Backing up ${NANOCHAT_BASE_DIR} to ${BACKUP_DIR}"
        mv "${NANOCHAT_BASE_DIR}" "${BACKUP_DIR}"
    fi
fi

# Prepare tokenizer
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"
cp "${BASE_TOKENIZER_REPO}/token_bytes.pt" "${BASE_TOKENIZER_REPO}/tokenizer.pkl" "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate base dir
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"

# Install (skip if already installed)
# uv pip install -e .

# Prepare report
python -m nanochat.report reset

# Download dataset (train and validation shards)
echo "Downloading dataset..."
python -m nanochat.dataset -n 10 --split both
echo "Dataset download complete."

# ============================================================
# Training - select config based on MODEL_NAME
# ============================================================

case "${VARIANT}" in
    "normal")
        # Normal BD3LM (random masking, ~62.5% tokens masked for block_size=4)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=${DEPTH} \
            --block_size=${BLOCK_SIZE} \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=${DATA_RATIO} \
            --target_shift=-1 \
            --eval_every=1000 \
            --eval_num_batches=20 \
            --eval_num_batches_final=100
        ;;
    "ts1")
        # BD3LM with target_shift=1 (predict 1st position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=${DEPTH} \
            --block_size=${BLOCK_SIZE} \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=${DATA_RATIO} \
            --target_shift=1 \
            --eval_every=1000 \
            --eval_num_batches=20 \
            --eval_num_batches_final=100
        ;;
    "ts2")
        # BD3LM with target_shift=2 (predict 2nd position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=${DEPTH} \
            --block_size=${BLOCK_SIZE} \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=${DATA_RATIO} \
            --target_shift=2 \
            --eval_every=1000 \
            --eval_num_batches=20 \
            --eval_num_batches_final=100
        ;;
    "ts3")
        # BD3LM with target_shift=3 (predict 3rd position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=${DEPTH} \
            --block_size=${BLOCK_SIZE} \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=${DATA_RATIO} \
            --target_shift=3 \
            --eval_every=1000 \
            --eval_num_batches=20 \
            --eval_num_batches_final=100
        ;;
    "ts4")
        # BD3LM with target_shift=4 (predict 4th/last position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=${DEPTH} \
            --block_size=${BLOCK_SIZE} \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=${DATA_RATIO} \
            --target_shift=4 \
            --eval_every=1000 \
            --eval_num_batches=20 \
            --eval_num_batches_final=100
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        echo "Available: normal, ts1, ts2, ts3, ts4"
        exit 1
        ;;
esac

# ============================================================
# Post-training: cleanup and upload
# ============================================================

echo "=== Training complete for ${MODEL_NAME} ==="

# Remove dataset to save space (optional)
# rm -rf "${NANOCHAT_BASE_DIR}/tiny_story_data"

# Skip upload in test mode
if [ "${TEST_MODE}" = "true" ]; then
    echo "Test mode: Skipping HuggingFace upload"
else


    # Prepare upload folder
    mkdir -p /content/upload_to_huggingface
    cp -r "${NANOCHAT_BASE_DIR}" /content/upload_to_huggingface/

    # Upload to HuggingFace
    python -c "
import os
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN', '')
if not token:
    print('Warning: HF_TOKEN not set, skipping upload')
else:
    api = HfApi(token=token)
    model_repo = os.environ['MODEL_REPO']
    print(f'Uploading to {model_repo}...')
    api.upload_large_folder(
        folder_path='/content/upload_to_huggingface',
        repo_id=model_repo,
        repo_type='model',
    )
    print('Upload complete!')
"

    # Cleanup upload folder for next model
    rm -rf /content/upload_to_huggingface
fi

echo "=== Done with ${MODEL_NAME} ==="

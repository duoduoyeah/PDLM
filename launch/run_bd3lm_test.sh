#!/bin/bash

## BD3LM Dry Run Configurations
## depth=4, block_size=4, target_param_data_ratio=10 for quick testing

# ============================================================
# Configuration - modify these for each run
# ============================================================
MODEL_NAME="${1:-bd3lm_d4_b4_normal}"  # pass as first arg, default to normal
TEST_MODE="${2:-true}"  # pass as second arg, default to true (removes old model dir)
WANDB_GROUP="bd3lm_d4"
MODEL_REPO="duoduoyeah/bd3lm_d4"
DRIVE_BASE="/content/drive/MyDrive/nanochat"
BASE_TOKENIZER_REPO="/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer_with_mask"

# Secrets should be set by running: %run launch/test_colab_secrets.py
# Check if they are set
if [ -z "${HF_TOKEN}" ]; then
    echo "Warning: HF_TOKEN not set. Run '%run launch/test_colab_secrets.py' first"
fi
if [ -z "${WANDB_API_KEY}" ]; then
    echo "Warning: WANDB_API_KEY not set. Run '%run launch/test_colab_secrets.py' first"
fi

# Export environment variables
export MODEL_NAME
export WANDB_GROUP
export MODEL_REPO
export NANOCHAT_BASE_DIR="${DRIVE_BASE}/${MODEL_NAME}"
export BASE_TOKENIZER_REPO

echo "=== Running model: ${MODEL_NAME} ==="
echo "=== Base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Test mode: ${TEST_MODE} ==="

# ============================================================
# Setup (run once per model)
# ============================================================

# In test mode, remove old model dir if exists (for repeated testing)
if [ "${TEST_MODE}" = "true" ] && [ -d "${NANOCHAT_BASE_DIR}" ]; then
    echo "Test mode: Removing old model dir ${NANOCHAT_BASE_DIR}"
    rm -rf "${NANOCHAT_BASE_DIR}"
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

# Download dataset
echo "Downloading dataset..."
python -m nanochat.dataset -n 10
echo "Dataset download complete."

# ============================================================
# Training - select config based on MODEL_NAME
# ============================================================

case "${MODEL_NAME}" in
    "bd3lm_d4_b4_normal")
        # Run 1: Normal BD3LM (random masking, ~50% tokens masked)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=4 \
            --block_size=4 \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=10 \
            --target_shift=-1 \
            --bd3lm_effective_ratio=0.5
        ;;
    "bd3lm_d4_b4_ts1")
        # Run 2: BD3LM with target_shift=1 (predict 1st position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=4 \
            --block_size=4 \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=10 \
            --target_shift=1 \
            --bd3lm_effective_ratio=0.25
        ;;
    "bd3lm_d4_b4_ts2")
        # Run 3: BD3LM with target_shift=2 (predict 2nd position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=4 \
            --block_size=4 \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=10 \
            --target_shift=2 \
            --bd3lm_effective_ratio=0.25
        ;;
    "bd3lm_d4_b4_ts4")
        # Run 4: BD3LM with target_shift=4 (predict 4th/last position in each block)
        python -m scripts.base_train \
            --run="${MODEL_NAME}" \
            --wandb_group="${WANDB_GROUP}" \
            --depth=4 \
            --block_size=4 \
            --prefix_pure_tokens=1 \
            --is_causal=False \
            --max_seq_len=512 \
            --device_batch_size=128 \
            --target_param_data_ratio=10 \
            --target_shift=4 \
            --bd3lm_effective_ratio=0.25
        ;;
    *)
        echo "Unknown model: ${MODEL_NAME}"
        echo "Available: bd3lm_d4_b4_normal, bd3lm_d4_b4_ts1, bd3lm_d4_b4_ts2, bd3lm_d4_b4_ts4"
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

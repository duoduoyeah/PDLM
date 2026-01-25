#!/bin/bash

## Experiment C Training Script (PDLM Stage 2: Group → Pure)
## Test mode: data_ratio=10
## Production mode: data_ratio=20 (or user specified)
##
## Usage:
##   bash launch/run_expc.sh --variant=g64
##   bash launch/run_expc.sh --variant=g16 --depth=8 --test_mode=false --data_ratio=30
##   bash launch/run_expc.sh --variant=g256 --block_size=8

# ============================================================
# Default values
# ============================================================
VARIANT="g64"  # g16, g64, g256 (num_groups)
OVERLAP_K="1"  # overlap_k for group tokenizer
TEST_MODE="true"
DATA_RATIO="10"  # default 10 for test mode
DEPTH="4"  # model depth
BLOCK_SIZE="4"  # block size

# Common training arguments
PREFIX_PURE_TOKENS="1"
IS_CAUSAL="False"
MAX_SEQ_LEN="512"
DEVICE_BATCH_SIZE="128"
EVAL_EVERY="2500"
EVAL_NUM_BATCHES="20"
EVAL_NUM_BATCHES_FINAL="100"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --variant=*)
            VARIANT="${arg#*=}"
            ;;
        --overlap_k=*)
            OVERLAP_K="${arg#*=}"
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
        --prefix_pure_tokens=*)
            PREFIX_PURE_TOKENS="${arg#*=}"
            ;;
        --is_causal=*)
            IS_CAUSAL="${arg#*=}"
            ;;
        --max_seq_len=*)
            MAX_SEQ_LEN="${arg#*=}"
            ;;
        --device_batch_size=*)
            DEVICE_BATCH_SIZE="${arg#*=}"
            ;;
        --eval_every=*)
            EVAL_EVERY="${arg#*=}"
            ;;
        --eval_num_batches=*)
            EVAL_NUM_BATCHES="${arg#*=}"
            ;;
        --eval_num_batches_final=*)
            EVAL_NUM_BATCHES_FINAL="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/run_expc.sh --variant=g64 [--depth=4] [--block_size=4] [--test_mode=true] [--data_ratio=10]"
            echo "       [--overlap_k=1] [--prefix_pure_tokens=1] [--is_causal=False] [--max_seq_len=512]"
            echo "       [--device_batch_size=128] [--eval_every=2500]"
            echo ""
            echo "Variants: g16, g64, g256 (num_groups for group tokenizer)"
            exit 1
            ;;
    esac
done

# Validate variant
case "${VARIANT}" in
    "g16"|"g64"|"g256")
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        echo "Available: g16, g64, g256"
        exit 1
        ;;
esac

# Build model name
BASE_MODEL_NAME="expc_d${DEPTH}_b${BLOCK_SIZE}_${VARIANT}_k${OVERLAP_K}"

WANDB_GROUP="expc_d${DEPTH}"
MODEL_REPO="duoduoyeah/expc_d${DEPTH}"
DRIVE_BASE="/content/drive/MyDrive/nanochat"

# Local training base (faster than Drive)
LOCAL_TRAIN_BASE="/content/pdlm_temp_train"

# Group tokenizer path on Drive (built by build_group_tokenizer.sh)
# Contains: tokenizer.pkl, token_maps.pt (self-contained, no need for base tokenizer)
GROUP_TOKENIZER_PATH="${DRIVE_BASE}/group_tokenizers/${VARIANT}_k${OVERLAP_K}"

# Load secrets from .env file
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

# Build model name with ratio suffix and test suffix
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
export NANOCHAT_BASE_DIR="${LOCAL_TRAIN_BASE}/${MODEL_NAME}"

echo "=== Running Experiment C: ${MODEL_NAME} ==="
echo "=== Local base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Drive base: ${DRIVE_BASE} ==="
echo "=== Test mode: ${TEST_MODE} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Depth: ${DEPTH} ==="
echo "=== Block size: ${BLOCK_SIZE} ==="
echo "=== Variant: ${VARIANT} (overlap_k=${OVERLAP_K}) ==="
echo "=== Group tokenizer: ${GROUP_TOKENIZER_PATH} ==="

# ============================================================
# Setup (run once per model)
# ============================================================

# Verify group tokenizer exists on Drive
if [ ! -f "${GROUP_TOKENIZER_PATH}/token_maps.pt" ]; then
    echo "ERROR: Group tokenizer not found at ${GROUP_TOKENIZER_PATH}"
    echo "Run build_group_tokenizer.sh first to generate group tokenizers"
    exit 1
fi

# Handle existing local model dir
if [ -d "${NANOCHAT_BASE_DIR}" ]; then
    if [ "${TEST_MODE}" = "true" ]; then
        echo "Test mode: Removing old local model dir ${NANOCHAT_BASE_DIR}"
        rm -rf "${NANOCHAT_BASE_DIR}"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR="${NANOCHAT_BASE_DIR}_backup_${TIMESTAMP}"
        echo "Production mode: Backing up ${NANOCHAT_BASE_DIR} to ${BACKUP_DIR}"
        mv "${NANOCHAT_BASE_DIR}" "${BACKUP_DIR}"
    fi
fi

# Create local base directory
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"

# Copy tokenizer files from Drive to local (group tokenizer is self-contained)
echo "Copying tokenizer files from Drive to local..."
cp "${GROUP_TOKENIZER_PATH}/tokenizer.pkl" "${NANOCHAT_BASE_DIR}/tokenizer/"
cp "${GROUP_TOKENIZER_PATH}/token_maps.pt" "${NANOCHAT_BASE_DIR}/tokenizer/"

echo "Tokenizer setup complete:"
ls -la "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate base dir
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"

# Validate token map
python -c "
from nanochat.group_tokenizer.token_map import get_token_map
tm = get_token_map()
print(f'Token map: pure_vocab={tm.pure_vocab_size}, num_groups={tm.num_groups}, overlap_k={tm.overlap_k}')
"

# Prepare report
python -m nanochat.report reset

# Download dataset to local base dir
echo "Downloading dataset to local base dir..."
python -m nanochat.dataset -n 10 --split both
echo "Dataset download complete."

# ============================================================
# Training - PDLM Stage 2 (Group → Pure)
# ============================================================

echo "Starting PDLM Stage 2 training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=stage2 \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --prefix_pure_tokens=${PREFIX_PURE_TOKENS} \
    --is_causal=${IS_CAUSAL} \
    --max_seq_len=${MAX_SEQ_LEN} \
    --device_batch_size=${DEVICE_BATCH_SIZE} \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=${EVAL_EVERY} \
    --eval_num_batches=${EVAL_NUM_BATCHES} \
    --eval_num_batches_final=${EVAL_NUM_BATCHES_FINAL}

# ============================================================
# Post-training: copy to Drive and upload
# ============================================================

echo "=== Training complete for ${MODEL_NAME} ==="

# Remove dataset from local (keep checkpoints and tokenizer)
rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

# Copy results to Drive for persistence
DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${MODEL_NAME}"
echo "Copying results to Drive: ${DRIVE_OUTPUT_DIR}"
mkdir -p "${DRIVE_OUTPUT_DIR}"
cp -r "${NANOCHAT_BASE_DIR}"/* "${DRIVE_OUTPUT_DIR}/"
echo "Results saved to Drive."

# Skip HF upload in test mode
if [ "${TEST_MODE}" = "true" ]; then
    echo "Test mode: Skipping HuggingFace upload"
else
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
    drive_output = '${DRIVE_OUTPUT_DIR}'
    print(f'Uploading {drive_output} to {model_repo}...')
    api.upload_large_folder(
        folder_path=drive_output,
        repo_id=model_repo,
        repo_type='model',
    )
    print('Upload complete!')
"
fi

# Cleanup local training dir
echo "Cleaning up local training dir..."
rm -rf "${NANOCHAT_BASE_DIR}"

echo "=== Done with ${MODEL_NAME} ==="

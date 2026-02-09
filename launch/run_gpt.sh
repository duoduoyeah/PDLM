#!/bin/bash

## GPT (next-token AR) Training Script
## For gradient tracking comparison with PDLM stage1_block.
## Uses --gradient_block_size to create virtual block positions for gradient decomposition.
##
## Usage:
##   bash launch/run_gpt.sh --depth=8 --data_ratio=20 --gradient_track_every=200

# ============================================================
# Default values
# ============================================================
TEST_MODE="true"
DATA_RATIO="10"
DEPTH="4"
GRADIENT_TRACK_EVERY="0"
GRADIENT_BLOCK_SIZE="4"
DRIVE_OUTPUT_FOLDER=""
TOKENIZER_DIR=""

MAX_SEQ_LEN="512"
DEVICE_BATCH_SIZE="64"
EVAL_EVERY="2500"
EVAL_NUM_BATCHES="20"
EVAL_NUM_BATCHES_FINAL="100"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --test_mode=*) TEST_MODE="${arg#*=}" ;;
        --data_ratio=*) DATA_RATIO="${arg#*=}" ;;
        --depth=*) DEPTH="${arg#*=}" ;;
        --gradient_track_every=*) GRADIENT_TRACK_EVERY="${arg#*=}" ;;
        --gradient_block_size=*) GRADIENT_BLOCK_SIZE="${arg#*=}" ;;
        --max_seq_len=*) MAX_SEQ_LEN="${arg#*=}" ;;
        --device_batch_size=*) DEVICE_BATCH_SIZE="${arg#*=}" ;;
        --eval_every=*) EVAL_EVERY="${arg#*=}" ;;
        --eval_num_batches=*) EVAL_NUM_BATCHES="${arg#*=}" ;;
        --eval_num_batches_final=*) EVAL_NUM_BATCHES_FINAL="${arg#*=}" ;;
        --drive_output_folder=*) DRIVE_OUTPUT_FOLDER="${arg#*=}" ;;
        --tokenizer_dir=*) TOKENIZER_DIR="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Build model name
BASE_MODEL_NAME="gpt_d${DEPTH}"

if [ "${GRADIENT_TRACK_EVERY}" != "0" ]; then
    WANDB_GROUP="gpt_d${DEPTH}_gradient_tracking"
else
    WANDB_GROUP="gpt_d${DEPTH}"
fi

DRIVE_BASE="/content/drive/MyDrive/nanochat"
LOCAL_TRAIN_BASE="/content/gpt_temp_train"

# Build model name with ratio and test suffix
if [ "${TEST_MODE}" = "true" ]; then
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}_test"
else
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}"
fi

export MODEL_NAME
export WANDB_GROUP
export NANOCHAT_BASE_DIR="${LOCAL_TRAIN_BASE}/${MODEL_NAME}"

echo "=== Running GPT: ${MODEL_NAME} ==="
echo "=== Local base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Depth: ${DEPTH} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Gradient track every: ${GRADIENT_TRACK_EVERY} ==="
echo "=== Gradient block size: ${GRADIENT_BLOCK_SIZE} ==="

# ============================================================
# Setup
# ============================================================

# Load secrets
if [ -f "launch/.env" ]; then
    source launch/.env
    echo "Loaded secrets from launch/.env"
fi

if [ -n "${WANDB_API_KEY}" ]; then
    wandb login --relogin "${WANDB_API_KEY}"
fi

# Handle existing local dir
if [ -d "${NANOCHAT_BASE_DIR}" ]; then
    if [ "${TEST_MODE}" = "true" ]; then
        echo "Test mode: Removing old local dir ${NANOCHAT_BASE_DIR}"
        rm -rf "${NANOCHAT_BASE_DIR}"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mv "${NANOCHAT_BASE_DIR}" "${NANOCHAT_BASE_DIR}_backup_${TIMESTAMP}"
    fi
fi

mkdir -p "${NANOCHAT_BASE_DIR}"

# Validate base dir
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"

# Prepare report
python -m nanochat.report reset

# Tokenizer setup (required)
if [ -z "${TOKENIZER_DIR}" ]; then
    echo "ERROR: --tokenizer_dir is required"
    exit 1
fi
if [ ! -f "${TOKENIZER_DIR}/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found at ${TOKENIZER_DIR}"
    exit 1
fi
echo "Copying tokenizer from ${TOKENIZER_DIR}..."
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"
cp "${TOKENIZER_DIR}/tokenizer.pkl" "${NANOCHAT_BASE_DIR}/tokenizer/"
if [ -f "${TOKENIZER_DIR}/token_bytes.pt" ]; then
    cp "${TOKENIZER_DIR}/token_bytes.pt" "${NANOCHAT_BASE_DIR}/tokenizer/"
fi
echo "Tokenizer setup complete:"
ls -la "${NANOCHAT_BASE_DIR}/tokenizer/"

# Download dataset (skip if already present)
if [ -d "${NANOCHAT_BASE_DIR}/tokenized_data" ]; then
    echo "Dataset already present, skipping download."
else
    echo "Downloading dataset..."
    python -m nanochat.dataset -n 10 --split both
fi

# ============================================================
# Training
# ============================================================

echo "Starting GPT training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --depth=${DEPTH} \
    --max_seq_len=${MAX_SEQ_LEN} \
    --device_batch_size=${DEVICE_BATCH_SIZE} \
    --model_type=next_token_ar \
    --target_shift=1 \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=${EVAL_EVERY} \
    --eval_num_batches=${EVAL_NUM_BATCHES} \
    --eval_num_batches_final=${EVAL_NUM_BATCHES_FINAL} \
    --gradient_track_every=${GRADIENT_TRACK_EVERY} \
    --gradient_block_size=${GRADIENT_BLOCK_SIZE}

# ============================================================
# Post-training: copy to Drive
# ============================================================

echo "=== Training complete for ${MODEL_NAME} ==="

rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

if [ -n "${DRIVE_OUTPUT_FOLDER}" ]; then
    DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${DRIVE_OUTPUT_FOLDER}/${MODEL_NAME}"
else
    DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${MODEL_NAME}"
fi
if [ -d "${DRIVE_OUTPUT_DIR}" ]; then
    if [ "${TEST_MODE}" = "true" ]; then
        rm -rf "${DRIVE_OUTPUT_DIR}"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mv "${DRIVE_OUTPUT_DIR}" "${DRIVE_OUTPUT_DIR}_backup_${TIMESTAMP}"
    fi
fi
echo "Copying results to Drive: ${DRIVE_OUTPUT_DIR}"
mkdir -p "${DRIVE_OUTPUT_DIR}"
cp -r "${NANOCHAT_BASE_DIR}"/* "${DRIVE_OUTPUT_DIR}/"

rm -rf "${NANOCHAT_BASE_DIR}"
echo "=== Done with ${MODEL_NAME} ==="

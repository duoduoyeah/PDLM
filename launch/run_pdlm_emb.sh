#!/bin/bash

## PDLM Embedding Training Script
## Replaces discrete group token IDs with continuous averaged embeddings as noise input.
## Constructs norm(mean(norm(wte(tok_i)) for tok_i in noise_set)) at block positions.
## Eliminates group tokenizer infrastructure entirely — no clustering, no token_maps.pt.
##
## Architecture uses 2L input: [xt | x0] with block diffusion mask.
## wte: pure_vocab_size only, lm_head: pure_vocab_size
##
## Usage:
##   bash launch/run_pdlm_emb.sh
##   bash launch/run_pdlm_emb.sh --noise_count=64 --depth=4 --block_size=4
##   bash launch/run_pdlm_emb.sh --test_mode=false --data_ratio=20

# ============================================================
# Default values
# ============================================================
NOISE_COUNT="64"           # total tokens in noise average (including target)
TEST_MODE="true"
DATA_RATIO="10"            # default 10 for test mode
DEPTH="4"                  # model depth
BLOCK_SIZE="4"             # bucket_size for block diffusion
SOFT_P_WITHIN="1.0"        # prob of including correct target in noise (1.0 = always include)
DRIVE_OUTPUT_FOLDER=""     # subfolder under DRIVE_BASE for outputs (empty = save directly under DRIVE_BASE)
GRADIENT_TRACK_EVERY="0"   # 0 = disabled, >0 = log gradient metrics every N steps

# Common training arguments
MAX_SEQ_LEN="512"
DEVICE_BATCH_SIZE="64"     # Lower than Stage 1 due to 2L input (doubled sequence)
EVAL_EVERY="2500"
EVAL_NUM_BATCHES="20"
EVAL_NUM_BATCHES_FINAL="100"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --noise_count=*)
            NOISE_COUNT="${arg#*=}"
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
        --soft_p_within=*)
            SOFT_P_WITHIN="${arg#*=}"
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
        --drive_output_folder=*)
            DRIVE_OUTPUT_FOLDER="${arg#*=}"
            ;;
        --gradient_track_every=*)
            GRADIENT_TRACK_EVERY="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/run_pdlm_emb.sh [--noise_count=64] [--depth=4] [--block_size=4]"
            echo "       [--test_mode=true] [--data_ratio=10]"
            echo "       [--max_seq_len=512] [--device_batch_size=64]"
            echo "       [--drive_output_folder=<folder>]"
            exit 1
            ;;
    esac
done

# Build model name
BASE_MODEL_NAME="pdlm_emb_d${DEPTH}_b${BLOCK_SIZE}_nc${NOISE_COUNT}"

WANDB_GROUP="pdlm_emb_d${DEPTH}"
DRIVE_BASE="/content/drive/MyDrive/nanochat"

# Local training base (faster than Drive)
LOCAL_TRAIN_BASE="/content/pdlm_emb_temp_train"

# Tokenizer path: any existing group tokenizer folder that contains tokenizer.pkl
# For pdlm_emb we only need tokenizer.pkl (no token_maps.pt)
# Default: look for a common tokenizer location
TOKENIZER_SOURCE="${DRIVE_BASE}/group_tokenizers"

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
export NANOCHAT_BASE_DIR="${LOCAL_TRAIN_BASE}/${MODEL_NAME}"

echo "=== Running PDLM Embedding: ${MODEL_NAME} ==="
echo "=== Local base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Drive base: ${DRIVE_BASE} ==="
echo "=== Drive output folder: ${DRIVE_OUTPUT_FOLDER:-<root>} ==="
echo "=== Test mode: ${TEST_MODE} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Depth: ${DEPTH} ==="
echo "=== Block size (bucket): ${BLOCK_SIZE} ==="
echo "=== Noise count: ${NOISE_COUNT} ==="
echo "=== Soft p within: ${SOFT_P_WITHIN} ==="

# ============================================================
# Setup (run once per model)
# ============================================================

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

# Copy tokenizer.pkl from any existing group tokenizer folder
# pdlm_emb does NOT need token_maps.pt
echo "Looking for tokenizer.pkl..."
FOUND_TOKENIZER=""
if [ -d "${TOKENIZER_SOURCE}" ]; then
    # Find first available tokenizer.pkl in any group tokenizer subfolder
    for dir in "${TOKENIZER_SOURCE}"/*/; do
        if [ -f "${dir}tokenizer.pkl" ]; then
            FOUND_TOKENIZER="${dir}tokenizer.pkl"
            break
        fi
    done
fi

if [ -n "${FOUND_TOKENIZER}" ]; then
    echo "Copying tokenizer from: ${FOUND_TOKENIZER}"
    cp "${FOUND_TOKENIZER}" "${NANOCHAT_BASE_DIR}/tokenizer/"
else
    echo "ERROR: No tokenizer.pkl found in ${TOKENIZER_SOURCE}"
    echo "Please ensure at least one group tokenizer has been built, or copy tokenizer.pkl manually."
    exit 1
fi

echo "Tokenizer setup complete (no token_maps.pt needed for pdlm_emb):"
ls -la "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate base dir
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"

# Prepare report
python -m nanochat.report reset

# Download dataset to local base dir (skip if already exists)
if [ -d "${NANOCHAT_BASE_DIR}/simple_story_data" ]; then
    echo "Dataset already exists, skipping download."
else
    echo "Downloading dataset to local base dir..."
    python -m nanochat.dataset -n 10 --split both
    echo "Dataset download complete."
fi

# ============================================================
# Training - PDLM Embedding
# ============================================================

echo "Starting PDLM Embedding training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=pdlm_emb \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --noise_count=${NOISE_COUNT} \
    --soft_p_within=${SOFT_P_WITHIN} \
    --max_seq_len=${MAX_SEQ_LEN} \
    --device_batch_size=${DEVICE_BATCH_SIZE} \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=${EVAL_EVERY} \
    --eval_num_batches=${EVAL_NUM_BATCHES} \
    --eval_num_batches_final=${EVAL_NUM_BATCHES_FINAL} \
    --gradient_track_every=${GRADIENT_TRACK_EVERY}

# ============================================================
# Post-training: copy to Drive
# ============================================================

echo "=== Training complete for ${MODEL_NAME} ==="

# Remove dataset from local (keep checkpoints and tokenizer)
rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

# Copy results to Drive for persistence
if [ -n "${DRIVE_OUTPUT_FOLDER}" ]; then
    DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${DRIVE_OUTPUT_FOLDER}/${MODEL_NAME}"
else
    DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${MODEL_NAME}"
fi
if [ -d "${DRIVE_OUTPUT_DIR}" ]; then
    if [ "${TEST_MODE}" = "true" ]; then
        echo "Test mode: Removing old Drive output dir ${DRIVE_OUTPUT_DIR}"
        rm -rf "${DRIVE_OUTPUT_DIR}"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR="${DRIVE_OUTPUT_DIR}_backup_${TIMESTAMP}"
        echo "Production mode: Backing up ${DRIVE_OUTPUT_DIR} to ${BACKUP_DIR}"
        mv "${DRIVE_OUTPUT_DIR}" "${BACKUP_DIR}"
    fi
fi
echo "Copying results to Drive: ${DRIVE_OUTPUT_DIR}"
mkdir -p "${DRIVE_OUTPUT_DIR}"
cp -r "${NANOCHAT_BASE_DIR}"/* "${DRIVE_OUTPUT_DIR}/"
echo "Results saved to Drive."

# Cleanup local training dir
echo "Cleaning up local training dir..."
rm -rf "${NANOCHAT_BASE_DIR}"

echo "=== Done with ${MODEL_NAME} ==="

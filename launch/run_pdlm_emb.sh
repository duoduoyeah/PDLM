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
TOKENIZER_PATH=""          # REQUIRED: path to tokenizer dir containing tokenizer.pkl (e.g. .../simplestory_tokenizer/4096/tokenizer)

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
        --tokenizer_path=*)
            TOKENIZER_PATH="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/run_pdlm_emb.sh --tokenizer_path=<path> [--noise_count=64] [--depth=4] [--block_size=4]"
            echo "       [--test_mode=true] [--data_ratio=10]"
            echo "       [--max_seq_len=512] [--device_batch_size=64]"
            echo "       [--drive_output_folder=<folder>]"
            echo ""
            echo "  --tokenizer_path: REQUIRED. Path to tokenizer dir containing tokenizer.pkl"
            echo "                    e.g. /content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer"
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

# Validate required tokenizer_path
if [ -z "${TOKENIZER_PATH}" ]; then
    echo "ERROR: --tokenizer_path is required."
    echo "  e.g. --tokenizer_path=/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer"
    exit 1
fi

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
echo "=== Tokenizer path: ${TOKENIZER_PATH} ==="

# ============================================================
# Setup (run once per model)
# ============================================================

# Handle existing local model dir
if [ -d "${NANOCHAT_BASE_DIR}" ]; then
    if [ "${TEST_MODE}" = "true" ]; then
        echo "Test mode: Cleaning old checkpoints but preserving dataset..."
        rm -rf "${NANOCHAT_BASE_DIR}/base_checkpoints"
        rm -rf "${NANOCHAT_BASE_DIR}/tokenizer"
        rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"
        rm -f "${NANOCHAT_BASE_DIR}/report.json"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR="${NANOCHAT_BASE_DIR}_backup_${TIMESTAMP}"
        echo "Production mode: Backing up ${NANOCHAT_BASE_DIR} to ${BACKUP_DIR}"
        mv "${NANOCHAT_BASE_DIR}" "${BACKUP_DIR}"
    fi
fi

# Create local base directory
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"

# Copy tokenizer.pkl (no token_maps.pt needed for pdlm_emb)
if [ ! -f "${TOKENIZER_PATH}/tokenizer.pkl" ]; then
    echo "ERROR: tokenizer.pkl not found at ${TOKENIZER_PATH}"
    exit 1
fi
echo "Copying tokenizer from: ${TOKENIZER_PATH}"
cp "${TOKENIZER_PATH}/tokenizer.pkl" "${NANOCHAT_BASE_DIR}/tokenizer/"

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

#!/bin/bash

## Combined PDLM (both_mtp) Training Script
## Single-pass model that performs:
##   - Stage 1 (MTP: Pure → K Group tokens) on the second L positions
##   - Stage 2 (Group → Pure denoising) on the first L positions
##
## Architecture uses 2L input: [xt | x0] with block diffusion mask
##
## Usage:
##   bash launch/run_both_mtp.sh --variant=g64
##   bash launch/run_both_mtp.sh --variant=g64 --overlap_k=2 --depth=8
##   bash launch/run_both_mtp.sh --variant=g64 --test_mode=false --data_ratio=20

# ============================================================
# Default values
# ============================================================
VARIANT="g64"              # g16, g64, g256 (num_groups)
OVERLAP_K="1"              # overlap_k for group tokenizer
TEST_MODE="true"
DATA_RATIO="10"            # default 10 for test mode
DEPTH="4"                  # model depth
BLOCK_SIZE="8"             # bucket_size for block diffusion (Stage 2)
N_FUTURE_TOKENS="4"        # K group tokens to predict (Stage 1 MTP)
MTP_LOSS_BETA="0.8"        # exponential decay for MTP loss weighting
MTP_LOSS_WEIGHT="1.0"      # Stage 1 loss weight relative to Stage 2

# Common training arguments
MAX_SEQ_LEN="512"
DEVICE_BATCH_SIZE="64"     # Lower than MTP due to 2L input (doubled sequence)
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
        --n_future_tokens=*)
            N_FUTURE_TOKENS="${arg#*=}"
            ;;
        --mtp_loss_beta=*)
            MTP_LOSS_BETA="${arg#*=}"
            ;;
        --mtp_loss_weight=*)
            MTP_LOSS_WEIGHT="${arg#*=}"
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
            echo "Usage: bash launch/run_both_mtp.sh --variant=g64 [--overlap_k=1] [--depth=4]"
            echo "       [--block_size=8] [--n_future_tokens=4] [--mtp_loss_beta=0.8] [--mtp_loss_weight=1.0]"
            echo "       [--test_mode=true] [--data_ratio=10] [--max_seq_len=512] [--device_batch_size=64]"
            echo ""
            echo "Variants: g16, g64, g256 (num_groups for group tokenizer)"
            echo ""
            echo "Parameters:"
            echo "  --block_size        Bucket size for block diffusion (Stage 2)"
            echo "  --n_future_tokens   K group tokens to predict (Stage 1 MTP)"
            echo "  --mtp_loss_beta     Exponential decay for MTP loss weights (β^k)"
            echo "  --mtp_loss_weight   Stage 1 loss weight relative to Stage 2"
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
BASE_MODEL_NAME="both_mtp_d${DEPTH}_b${BLOCK_SIZE}_K${N_FUTURE_TOKENS}_${VARIANT}_k${OVERLAP_K}"

WANDB_GROUP="both_mtp_d${DEPTH}"
DRIVE_BASE="/content/drive/MyDrive/nanochat"

# Local training base (faster than Drive)
LOCAL_TRAIN_BASE="/content/both_mtp_temp_train"

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
export NANOCHAT_BASE_DIR="${LOCAL_TRAIN_BASE}/${MODEL_NAME}"

echo "=== Running Combined PDLM (both_mtp): ${MODEL_NAME} ==="
echo "=== Local base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Drive base: ${DRIVE_BASE} ==="
echo "=== Test mode: ${TEST_MODE} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Depth: ${DEPTH} ==="
echo "=== Block size (bucket): ${BLOCK_SIZE} ==="
echo "=== N future tokens (K): ${N_FUTURE_TOKENS} ==="
echo "=== MTP loss beta: ${MTP_LOSS_BETA} ==="
echo "=== MTP loss weight: ${MTP_LOSS_WEIGHT} ==="
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
# Training - Combined PDLM (both_mtp)
# ============================================================

echo "Starting Combined PDLM (both_mtp) training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=both_mtp \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --n_future_tokens=${N_FUTURE_TOKENS} \
    --mtp_loss_beta=${MTP_LOSS_BETA} \
    --mtp_loss_weight=${MTP_LOSS_WEIGHT} \
    --max_seq_len=${MAX_SEQ_LEN} \
    --device_batch_size=${DEVICE_BATCH_SIZE} \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=${EVAL_EVERY} \
    --eval_num_batches=${EVAL_NUM_BATCHES} \
    --eval_num_batches_final=${EVAL_NUM_BATCHES_FINAL}

# ============================================================
# Post-training: copy to Drive
# ============================================================

echo "=== Training complete for ${MODEL_NAME} ==="

# Remove dataset from local (keep checkpoints and tokenizer)
rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

# Copy results to Drive for persistence
DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${MODEL_NAME}"
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

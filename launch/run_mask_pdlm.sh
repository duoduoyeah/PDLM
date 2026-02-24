#!/bin/bash

## Mask PDLM Training Script
## Every position predicts itself (not the next block's token).
## Eliminates dual-target ambiguity of block-PDLM at the cost of lower data utilization (~35% vs ~100%).
## A mask token is introduced as the initial fully-noisy state.
##
## Architecture uses 2L input: [xt | x0] with block diffusion mask.
## wte: pure_vocab_size + num_groups + 1 (mask), lm_head: pure_vocab_size
##
## Tokenizer naming: n{noise}_k{overlap_k}_g{num_groups}
##
## Usage:
##   bash launch/run_mask_pdlm.sh --noise_level=64 --overlap_k=1 --num_groups=64
##   bash launch/run_mask_pdlm.sh --noise_level=1024 --overlap_k=7 --num_groups=28 --depth=8
##   bash launch/run_mask_pdlm.sh --noise_level=64 --num_groups=64 --test_mode=false --data_ratio=20

# ============================================================
# Default values
# ============================================================
NOISE_LEVEL="64"           # tokens per final group
OVERLAP_K="1"              # how many groups each token appears in
NUM_GROUPS="64"            # number of final groups
TEST_MODE="true"
DATA_RATIO="10"            # default 10 for test mode
DEPTH="4"                  # model depth
BLOCK_SIZE="4"             # bucket_size for block diffusion
SOFT_P_WITHIN="1.0"        # prob of correct group mapping (1.0 = hard, <1.0 = soft noise on group positions)
MASK_PDLM_4STATE="false"   # use 4-state variant (k states instead of k+1)
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
        --noise_level=*)
            NOISE_LEVEL="${arg#*=}"
            ;;
        --overlap_k=*)
            OVERLAP_K="${arg#*=}"
            ;;
        --num_groups=*)
            NUM_GROUPS="${arg#*=}"
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
        --mask_pdlm_4state=*)
            MASK_PDLM_4STATE="${arg#*=}"
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
            echo "Usage: bash launch/run_mask_pdlm.sh [--noise_level=64] [--overlap_k=1] [--num_groups=64]"
            echo "       [--depth=4] [--block_size=4]"
            echo "       [--test_mode=true] [--data_ratio=10]"
            echo "       [--max_seq_len=512] [--device_batch_size=64]"
            echo "       [--drive_output_folder=<folder>]"
            exit 1
            ;;
    esac
done

# Build tokenizer variant name (matches folder naming convention)
TOKENIZER_VARIANT="n${NOISE_LEVEL}_k${OVERLAP_K}_g${NUM_GROUPS}"

# Convert soft_p_within to integer percentage (e.g. 0.7 -> 70)
SOFT_P_INT=$(python3 -c "print(int(float('${SOFT_P_WITHIN}') * 100))")

# Build model name
BASE_MODEL_NAME="mask_pdlm_d${DEPTH}_b${BLOCK_SIZE}_${TOKENIZER_VARIANT}_p${SOFT_P_INT}"
if [ "${MASK_PDLM_4STATE}" = "true" ]; then
    BASE_MODEL_NAME="${BASE_MODEL_NAME}_4s"
fi

WANDB_GROUP="mask_pdlm_d${DEPTH}"
DRIVE_BASE="/content/drive/MyDrive/nanochat"

# Local training base (faster than Drive)
LOCAL_TRAIN_BASE="/content/mask_pdlm_temp_train"

# Group tokenizer path on Drive
GROUP_TOKENIZER_PATH="${DRIVE_BASE}/group_tokenizers/${TOKENIZER_VARIANT}"

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

echo "=== Running Mask PDLM: ${MODEL_NAME} ==="
echo "=== Local base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Drive base: ${DRIVE_BASE} ==="
echo "=== Drive output folder: ${DRIVE_OUTPUT_FOLDER:-<root>} ==="
echo "=== Test mode: ${TEST_MODE} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Depth: ${DEPTH} ==="
echo "=== Block size (bucket): ${BLOCK_SIZE} ==="
echo "=== Soft p within: ${SOFT_P_WITHIN} ==="
echo "=== 4-state: ${MASK_PDLM_4STATE} ==="
echo "=== Tokenizer: ${TOKENIZER_VARIANT} (noise=${NOISE_LEVEL}, overlap_k=${OVERLAP_K}, num_groups=${NUM_GROUPS}) ==="
echo "=== Group tokenizer path: ${GROUP_TOKENIZER_PATH} ==="

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

# Download dataset to local base dir (skip if already exists)
if [ -d "${NANOCHAT_BASE_DIR}/simple_story_data" ]; then
    echo "Dataset already exists, skipping download."
else
    echo "Downloading dataset to local base dir..."
    python -m nanochat.dataset -n 10 --split both
    echo "Dataset download complete."
fi

# ============================================================
# Training - Mask PDLM
# ============================================================

echo "Starting Mask PDLM training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=mask_pdlm \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --soft_p_within=${SOFT_P_WITHIN} \
    --mask_pdlm_4state=${MASK_PDLM_4STATE} \
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

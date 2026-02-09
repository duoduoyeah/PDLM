#!/bin/bash

## PDLM Stage 1 Block Training Script (Pure → Group, block-to-block)
## Position k in block i predicts the group token at position k in block i+1.
## No MASK tokens, no MTP head — just direct prediction via lm_head with
## an L×L block-causal attention mask.
##
## Tokenizer naming: n{noise}_k{overlap_k}_g{num_groups}
##   - noise_level: tokens per final group (e.g., 64, 1024)
##   - overlap_k: how many groups each token appears in
##   - num_groups: number of final groups
##
## Usage:
##   bash launch/run_pdlm_stage1_block.sh --noise_level=64 --overlap_k=1 --num_groups=64
##   bash launch/run_pdlm_stage1_block.sh --noise_level=1024 --overlap_k=7 --num_groups=28 --depth=8
##   bash launch/run_pdlm_stage1_block.sh --noise_level=64 --num_groups=64 --test_mode=false --data_ratio=20

# ============================================================
# Default values
# ============================================================
NOISE_LEVEL="64"        # tokens per final group
OVERLAP_K="1"           # how many groups each token appears in
NUM_GROUPS="64"         # number of final groups
TEST_MODE="true"
DATA_RATIO="10"         # default 10 for test mode
DEPTH="4"               # model depth
BLOCK_SIZE="4"          # bucket size (parallel block prediction)
PREFIX_PURE_TOKENS="0"  # pure prefix tokens for conditioning
IS_CAUSAL="False"       # bidirectional attention within blocks
STAGE1_TARGET_MODE="pure"  # "pure" (CE over pure_vocab) or "group" (any_correct_ce over num_groups)
GRADIENT_TRACK_EVERY="0"  # gradient tracking interval (0 = disabled)
DRIVE_OUTPUT_FOLDER=""  # subfolder under DRIVE_BASE for outputs (empty = save directly under DRIVE_BASE)

# Common training arguments
MAX_SEQ_LEN="512"
DEVICE_BATCH_SIZE="64"
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
        --prefix_pure_tokens=*)
            PREFIX_PURE_TOKENS="${arg#*=}"
            ;;
        --is_causal=*)
            IS_CAUSAL="${arg#*=}"
            ;;
        --stage1_target_mode=*)
            STAGE1_TARGET_MODE="${arg#*=}"
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
        --gradient_track_every=*)
            GRADIENT_TRACK_EVERY="${arg#*=}"
            ;;
        --drive_output_folder=*)
            DRIVE_OUTPUT_FOLDER="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/run_pdlm_stage1_block.sh [--noise_level=64] [--overlap_k=1] [--num_groups=64]"
            echo "       [--depth=4] [--block_size=8] [--prefix_pure_tokens=0] [--is_causal=False] [--stage1_target_mode=pure]"
            echo "       [--test_mode=true] [--data_ratio=10]"
            echo "       [--max_seq_len=512] [--device_batch_size=64] [--gradient_track_every=0]"
            echo "       [--drive_output_folder=<folder>]"
            echo ""
            echo "Use --drive_output_folder to save outputs to a subfolder under DRIVE_BASE"
            exit 1
            ;;
    esac
done

# Build tokenizer variant name (matches folder naming convention)
TOKENIZER_VARIANT="n${NOISE_LEVEL}_k${OVERLAP_K}_g${NUM_GROUPS}"

# Build model name
BASE_MODEL_NAME="pdlm_s1b_d${DEPTH}_b${BLOCK_SIZE}_${TOKENIZER_VARIANT}"

if [ "${GRADIENT_TRACK_EVERY}" != "0" ]; then
    WANDB_GROUP="s1b_d${DEPTH}_gradient_tracking"
else
    WANDB_GROUP="pdlm_s1b_d${DEPTH}"
fi
DRIVE_BASE="/content/drive/MyDrive/nanochat"

# Local training base (faster than Drive)
LOCAL_TRAIN_BASE="/content/pdlm_s1b_temp_train"

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

echo "=== Running PDLM Stage 1 Block: ${MODEL_NAME} ==="
echo "=== Local base dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Drive base: ${DRIVE_BASE} ==="
echo "=== Drive output folder: ${DRIVE_OUTPUT_FOLDER:-<root>} ==="
echo "=== Test mode: ${TEST_MODE} ==="
echo "=== Data ratio: ${DATA_RATIO} ==="
echo "=== Depth: ${DEPTH} ==="
echo "=== Block size: ${BLOCK_SIZE} ==="
echo "=== Prefix pure tokens: ${PREFIX_PURE_TOKENS} ==="
echo "=== Is causal: ${IS_CAUSAL} ==="
echo "=== Stage1 target mode: ${STAGE1_TARGET_MODE} ==="
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

# Download dataset to local base dir
echo "Downloading dataset to local base dir..."
python -m nanochat.dataset -n 10 --split both
echo "Dataset download complete."

# ============================================================
# Training - PDLM Stage 1 Block (Pure → Group, block-to-block)
# ============================================================

echo "Starting PDLM Stage 1 Block training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=stage1_block \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --prefix_pure_tokens=${PREFIX_PURE_TOKENS} \
    --is_causal=${IS_CAUSAL} \
    --max_seq_len=${MAX_SEQ_LEN} \
    --device_batch_size=${DEVICE_BATCH_SIZE} \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=${EVAL_EVERY} \
    --eval_num_batches=${EVAL_NUM_BATCHES} \
    --eval_num_batches_final=${EVAL_NUM_BATCHES_FINAL} \
    --stage1_target_mode=${STAGE1_TARGET_MODE} \
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

#!/bin/bash

## Train Learned Group Mapping
## Caches logits from a frozen stage1_block model, then trains a group assignment
## matrix to optimize group-level accuracy.
##
## Usage:
##   bash launch/train_learned_group_mapping.sh --ckpt_path=/path/to/model
##   bash launch/train_learned_group_mapping.sh --ckpt_path=/path/to/model --num_groups=512
##   bash launch/train_learned_group_mapping.sh --cache_dir=/path/to/cache --num_groups=256
##
## The checkpoint folder should contain:
##   - base_checkpoints/  (with model_*.pt files)
##   - tokenizer/
##
## Prepared folder structure:
##   LOCAL_DIR/
##   ├── tokenizer/              # copied from CKPT_PATH/tokenizer/
##   │   ├── tokenizer.pkl
##   │   └── token_maps.pt       # original mapping (needed by dataloader)
##   ├── base_checkpoints/       # copied from CKPT_PATH/base_checkpoints/
##   │   └── <model_tag>/
##   │       ├── model_NNNNNN.pt
##   │       └── meta_NNNNNN.json
##   ├── simple_story_data/      # downloaded by nanochat.dataset
##   │   └── val/
##   └── output_g<G>/            # OUTPUT_DIR
##       ├── logit_cache/        # cached pure logits
##       │   ├── batch_0000.pt .. batch_NNNN.pt
##       │   └── cache_meta.json
##       ├── token_maps.pt       # final binarized output (TokenMap-compatible)
##       └── assignment_raw.pt   # raw soft assignment matrix

# ============================================================
# Default values
# ============================================================
CKPT_PATH=""
CACHE_DIR=""
NUM_GROUPS="256"
NUM_EPOCHS="50"
LR="0.1"
LAMBDA_NOISE="0.1"
LAMBDA_OVERLAP="0.0"
LAMBDA_SHARP="1.0"
SHARP_RAMP_START="0.7"
MAX_SIZE_MULT="4"
MIN_OVERLAP_SOFT="4"
CACHE_NUM_BATCHES="200"
CACHE_SPLIT="val"
SKIP_POS0="False"
OUTPUT_DIR=""
LOCAL_DIR="/content/learned_group_mapping"
RUN="dummy"
WANDB_GROUP=""
EVAL_EVERY="5"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --ckpt_path=*)
            CKPT_PATH="${arg#*=}"
            ;;
        --cache_dir=*)
            CACHE_DIR="${arg#*=}"
            ;;
        --num_groups=*)
            NUM_GROUPS="${arg#*=}"
            ;;
        --num_epochs=*)
            NUM_EPOCHS="${arg#*=}"
            ;;
        --lr=*)
            LR="${arg#*=}"
            ;;
        --lambda_noise=*)
            LAMBDA_NOISE="${arg#*=}"
            ;;
        --lambda_overlap=*)
            LAMBDA_OVERLAP="${arg#*=}"
            ;;
        --lambda_sharp=*)
            LAMBDA_SHARP="${arg#*=}"
            ;;
        --sharp_ramp_start=*)
            SHARP_RAMP_START="${arg#*=}"
            ;;
        --max_size_mult=*)
            MAX_SIZE_MULT="${arg#*=}"
            ;;
        --min_overlap_soft=*)
            MIN_OVERLAP_SOFT="${arg#*=}"
            ;;
        --cache_num_batches=*)
            CACHE_NUM_BATCHES="${arg#*=}"
            ;;
        --cache_split=*)
            CACHE_SPLIT="${arg#*=}"
            ;;
        --skip_pos0=*)
            SKIP_POS0="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --local_dir=*)
            LOCAL_DIR="${arg#*=}"
            ;;
        --run=*)
            RUN="${arg#*=}"
            ;;
        --wandb_group=*)
            WANDB_GROUP="${arg#*=}"
            ;;
        --eval_every=*)
            EVAL_EVERY="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/train_learned_group_mapping.sh --ckpt_path=/path/to/model"
            echo "       [--num_groups=256] [--num_epochs=50] [--lr=0.1]"
            echo "       [--cache_dir=...] [--output_dir=...] [--run=wandb_name]"
            exit 1
            ;;
    esac
done

# Load secrets from .env file (for wandb API key)
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

# ============================================================
# Step 0: Validate and setup
# ============================================================
if [ -z "${CKPT_PATH}" ] && [ -z "${CACHE_DIR}" ]; then
    echo "Error: Must provide --ckpt_path or --cache_dir"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${LOCAL_DIR}/output_g${NUM_GROUPS}"
fi

if [ -n "${CKPT_PATH}" ] && [ ! -d "${CKPT_PATH}" ]; then
    echo "Error: Checkpoint path does not exist: ${CKPT_PATH}"
    exit 1
fi

echo "============================================================"
echo "Train Learned Group Mapping"
echo "============================================================"
echo "Checkpoint:       ${CKPT_PATH:-N/A}"
echo "Cache Dir:        ${CACHE_DIR:-will create}"
echo "Num Groups:       ${NUM_GROUPS}"
echo "Num Epochs:       ${NUM_EPOCHS}"
echo "LR:               ${LR}"
echo "Lambda Noise:     ${LAMBDA_NOISE}"
echo "Lambda Overlap:   ${LAMBDA_OVERLAP}"
echo "Lambda Sharp:     ${LAMBDA_SHARP}"
echo "Sharp Ramp Start: ${SHARP_RAMP_START}"
echo "Max Size Mult:    ${MAX_SIZE_MULT}"
echo "Min Overlap Soft: ${MIN_OVERLAP_SOFT}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Local Dir:        ${LOCAL_DIR}"
echo "Run:              ${RUN}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

if [ -n "${CKPT_PATH}" ]; then
    # --ckpt_path mode: copy model to local dir, download data, find checkpoint

    # ============================================================
    # Step 1: Copy tokenizer
    # ============================================================
    echo ""
    echo "Step 1: Copying tokenizer..."

    if [ ! -d "${CKPT_PATH}/tokenizer" ]; then
        echo "Error: No tokenizer directory found at ${CKPT_PATH}/tokenizer"
        exit 1
    fi

    mkdir -p "${LOCAL_DIR}/tokenizer"
    cp "${CKPT_PATH}/tokenizer/"* "${LOCAL_DIR}/tokenizer/"
    echo "  Copied tokenizer to ${LOCAL_DIR}/tokenizer/"

    # ============================================================
    # Step 2: Copy checkpoints
    # ============================================================
    echo ""
    echo "Step 2: Copying checkpoints..."

    if [ ! -d "${CKPT_PATH}/base_checkpoints" ]; then
        echo "Error: No base_checkpoints directory found at ${CKPT_PATH}/base_checkpoints"
        exit 1
    fi

    if [ ! -d "${LOCAL_DIR}/base_checkpoints" ]; then
        cp -r "${CKPT_PATH}/base_checkpoints" "${LOCAL_DIR}/"
    fi
    echo "  Copied base_checkpoints to ${LOCAL_DIR}/base_checkpoints/"

    # ============================================================
    # Step 3: Download dataset
    # ============================================================
    echo ""
    echo "Step 3: Downloading dataset..."

    export NANOCHAT_BASE_DIR="${LOCAL_DIR}"
    python -m nanochat.dataset --split=${CACHE_SPLIT}

    if [ $? -ne 0 ]; then
        echo "Error: Failed to download dataset"
        exit 1
    fi

    echo "  Downloaded data to ${LOCAL_DIR}/simple_story_data/"

    # ============================================================
    # Step 4: Find checkpoint directory
    # ============================================================
    echo ""
    echo "Step 4: Finding checkpoint directory..."

    # Find the directory containing model_*.pt files (handles nested structures)
    INNER_CKPT_DIR=$(find "${LOCAL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | head -1)

    if [ -z "${INNER_CKPT_DIR}" ]; then
        echo "Error: No checkpoints found in ${LOCAL_DIR}/base_checkpoints"
        echo "Available directories:"
        ls -la "${LOCAL_DIR}/base_checkpoints/"
        exit 1
    fi

    echo "  Found checkpoint dir: ${INNER_CKPT_DIR}"

    CKPT_DIR_ARG="--ckpt_dir=${INNER_CKPT_DIR}"
else
    # --cache_dir mode: no model copy needed, just set NANOCHAT_BASE_DIR
    CKPT_DIR_ARG=""
    export NANOCHAT_BASE_DIR="${LOCAL_DIR}"
fi

# Build cache_dir arg
if [ -n "${CACHE_DIR}" ]; then
    CACHE_DIR_ARG="--cache_dir=${CACHE_DIR}"
else
    CACHE_DIR_ARG="--cache_dir=${OUTPUT_DIR}/logit_cache"
fi

# ============================================================
# Step 5: Run training
# ============================================================
echo ""
echo "Step 5: Starting training..."

python -m scripts.train_group_mapping \
    ${CKPT_DIR_ARG} \
    ${CACHE_DIR_ARG} \
    --num_groups=${NUM_GROUPS} \
    --num_epochs=${NUM_EPOCHS} \
    --lr=${LR} \
    --lambda_noise=${LAMBDA_NOISE} \
    --lambda_overlap=${LAMBDA_OVERLAP} \
    --lambda_sharp=${LAMBDA_SHARP} \
    --sharp_ramp_start=${SHARP_RAMP_START} \
    --max_size_soft_multiplier=${MAX_SIZE_MULT} \
    --min_overlap_soft=${MIN_OVERLAP_SOFT} \
    --cache_num_batches=${CACHE_NUM_BATCHES} \
    --cache_split=${CACHE_SPLIT} \
    --skip_pos0=${SKIP_POS0} \
    --output_dir=${OUTPUT_DIR} \
    --run=${RUN} \
    --wandb_group=${WANDB_GROUP} \
    --eval_every_epoch=${EVAL_EVERY}

TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

# ============================================================
# Step 6: Summary
# ============================================================
echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
if [ -f "${OUTPUT_DIR}/token_maps.pt" ]; then
    echo "token_maps.pt saved successfully"
fi
if [ -f "${OUTPUT_DIR}/assignment_raw.pt" ]; then
    echo "assignment_raw.pt saved successfully"
fi
echo "============================================================"

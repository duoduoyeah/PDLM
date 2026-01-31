#!/bin/bash

## PDLM Stage 1 Block Evaluation Script
## Copies model to local temp folder, downloads validation dataset, runs evaluation.
##
## Usage:
##   bash launch/eval_pdlm_stage1_block.sh --ckpt_path=/path/to/model
##   bash launch/eval_pdlm_stage1_block.sh --ckpt_path=/path/to/model --num_batches=50
##   bash launch/eval_pdlm_stage1_block.sh --ckpt_path=/path/to/model --dump_path=/tmp/dump.txt
##
## The checkpoint folder should contain:
##   - base_checkpoints/  (with model_*.pt files)
##   - tokenizer/

# ============================================================
# Default values
# ============================================================
CKPT_PATH=""
NUM_BATCHES="20"
LOCAL_DIR="/tmp/pdlm_s1b_eval"
DUMP_PATH=""
DUMP_SEQUENCES="10"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --ckpt_path=*)
            CKPT_PATH="${arg#*=}"
            ;;
        --num_batches=*)
            NUM_BATCHES="${arg#*=}"
            ;;
        --local_dir=*)
            LOCAL_DIR="${arg#*=}"
            ;;
        --dump_path=*)
            DUMP_PATH="${arg#*=}"
            ;;
        --dump_sequences=*)
            DUMP_SEQUENCES="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_pdlm_stage1_block.sh --ckpt_path=/path/to/model"
            echo "       [--num_batches=20] [--local_dir=/tmp/pdlm_s1b_eval]"
            echo "       [--dump_path=/path/to/dump.txt] [--dump_sequences=10]"
            exit 1
            ;;
    esac
done

# ============================================================
# Step 0: Validate and setup
# ============================================================
if [ -z "${CKPT_PATH}" ]; then
    echo "Error: --ckpt_path is required"
    exit 1
fi

if [ ! -d "${CKPT_PATH}" ]; then
    echo "Error: Checkpoint path does not exist: ${CKPT_PATH}"
    exit 1
fi

echo "============================================================"
echo "PDLM Stage 1 Block Evaluation"
echo "============================================================"
echo "Checkpoint:     ${CKPT_PATH}"
echo "Num Batches:    ${NUM_BATCHES}"
echo "Local Dir:      ${LOCAL_DIR}"
if [ -n "${DUMP_PATH}" ]; then
    echo "Dump Path:      ${DUMP_PATH}"
    echo "Dump Sequences: ${DUMP_SEQUENCES}"
fi
echo "============================================================"

# Handle existing local dir (remove to ensure clean state)
if [ -d "${LOCAL_DIR}" ]; then
    echo ""
    echo "Removing existing local directory: ${LOCAL_DIR}"
    rm -rf "${LOCAL_DIR}"
fi
mkdir -p "${LOCAL_DIR}"

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

cp -r "${CKPT_PATH}/base_checkpoints" "${LOCAL_DIR}/"
echo "  Copied base_checkpoints to ${LOCAL_DIR}/base_checkpoints/"

# ============================================================
# Step 3: Download validation dataset
# ============================================================
echo ""
echo "Step 3: Downloading validation dataset..."

export NANOCHAT_BASE_DIR="${LOCAL_DIR}"
python -m nanochat.dataset --split=val

if [ $? -ne 0 ]; then
    echo "Error: Failed to download validation dataset"
    exit 1
fi

echo "  Downloaded validation data to ${LOCAL_DIR}/simple_story_data/"

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

# ============================================================
# Step 5: Run evaluation
# ============================================================
echo ""
echo "Step 5: Running evaluation..."

python -m scripts.pdlm_eval \
    --ckpt_dir="${INNER_CKPT_DIR}" \
    --num_batches=${NUM_BATCHES} \
    --output_json="${LOCAL_DIR}/eval_result.json"

EVAL_STATUS=$?

if [ $EVAL_STATUS -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

# ============================================================
# Step 6: Optional dump
# ============================================================
if [ -n "${DUMP_PATH}" ]; then
    echo ""
    echo "Step 6: Dumping predictions..."

    python -m scripts.pdlm_eval \
        --ckpt_dir="${INNER_CKPT_DIR}" \
        --dump_stage1_block="${DUMP_PATH}" \
        --dump_sequences=${DUMP_SEQUENCES}

    if [ $? -ne 0 ]; then
        echo "Warning: Dump failed"
    else
        echo "  Dumped predictions to ${DUMP_PATH}"
    fi
fi

# ============================================================
# Step 7: Print summary
# ============================================================
echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${LOCAL_DIR}/eval_result.json"
echo ""
cat "${LOCAL_DIR}/eval_result.json"
echo ""
echo "============================================================"

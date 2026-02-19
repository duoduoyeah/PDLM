#!/bin/bash

## PDLM Both-Block Evaluation Script
## Evaluates both_block models: Stage 1, Stage 2, and End-to-End per-position metrics.
##
## Usage:
##   bash launch/eval_pdlm_block.sh --ckpt_dir=/path/to/model
##   bash launch/eval_pdlm_block.sh --ckpt_dir=/path/to/model --num_batches=50
##   bash launch/eval_pdlm_block.sh --ckpt_dir=/path/to/model --run_compatibility --run_oracle_accuracy

# ============================================================
# Default values
# ============================================================
NUM_BATCHES="20"
CKPT_DIR=""  # Direct checkpoint path (required)
RUN_COMPATIBILITY="false"
RUN_ORACLE_ACCURACY="false"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --ckpt_dir=*)
            CKPT_DIR="${arg#*=}"
            ;;
        --num_batches=*)
            NUM_BATCHES="${arg#*=}"
            ;;
        --run_compatibility)
            RUN_COMPATIBILITY="true"
            ;;
        --run_oracle_accuracy)
            RUN_ORACLE_ACCURACY="true"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_pdlm_block.sh --ckpt_dir=/path/to/model"
            echo "       [--num_batches=20] [--run_compatibility] [--run_oracle_accuracy]"
            exit 1
            ;;
    esac
done

if [ -z "${CKPT_DIR}" ]; then
    echo "Error: --ckpt_dir is required"
    echo "Usage: bash launch/eval_pdlm_block.sh --ckpt_dir=/path/to/model"
    exit 1
fi

echo "============================================================"
echo "PDLM Both-Block Evaluation"
echo "============================================================"
echo "Checkpoint:      ${CKPT_DIR}"
echo "Num Batches:     ${NUM_BATCHES}"
echo "Compatibility:   ${RUN_COMPATIBILITY}"
echo "Oracle Accuracy: ${RUN_ORACLE_ACCURACY}"
echo "============================================================"

# ============================================================
# Step 0: Ensure validation dataset exists
# ============================================================
echo ""
echo "Step 0: Checking for validation dataset..."

if [ -n "${NANOCHAT_BASE_DIR}" ]; then
    DATA_DIR="${NANOCHAT_BASE_DIR}/simple_story_data"
else
    DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
fi

VAL_SHARDS=$(find "${DATA_DIR}" -maxdepth 1 -name "validation_*.parquet" 2>/dev/null | head -1)

if [ -z "${VAL_SHARDS}" ]; then
    echo "No validation data found in ${DATA_DIR}"
    echo "Downloading validation dataset..."
    python -m nanochat.dataset --split=val
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download validation dataset"
        exit 1
    fi
else
    echo "Validation data found in ${DATA_DIR}"
fi

# ============================================================
# Setup: link tokenizer and data
# ============================================================
echo ""
echo "Using checkpoint path: ${CKPT_DIR}"

BASE_DIR=$(dirname "${DATA_DIR}")
TOKENIZER_LINK="${BASE_DIR}/tokenizer"
MODEL_TOKENIZER="${CKPT_DIR}/tokenizer"

if [ -d "${MODEL_TOKENIZER}" ]; then
    if [ -L "${TOKENIZER_LINK}" ]; then
        rm "${TOKENIZER_LINK}"
    fi
    ln -s "${MODEL_TOKENIZER}" "${TOKENIZER_LINK}"
    echo "Created symlink: ${TOKENIZER_LINK} -> ${MODEL_TOKENIZER}"
else
    echo "Warning: No tokenizer found at ${MODEL_TOKENIZER}"
fi

export NANOCHAT_BASE_DIR="${BASE_DIR}"
echo "NANOCHAT_BASE_DIR set to: ${BASE_DIR}"

# Find the directory containing model_*.pt files
ACTUAL_CKPT_DIRS=$(find "${CKPT_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u)
CKPT_COUNT=$(echo "$ACTUAL_CKPT_DIRS" | grep -c . 2>/dev/null || echo 0)

if [ "$CKPT_COUNT" -eq 0 ]; then
    echo "Error: No checkpoints found in ${CKPT_DIR}/base_checkpoints"
    exit 1
elif [ "$CKPT_COUNT" -gt 1 ]; then
    echo "Warning: Multiple checkpoint directories found:"
    echo "$ACTUAL_CKPT_DIRS"
    echo "Please specify a more specific --ckpt_dir path."
    exit 1
fi

ACTUAL_CKPT_DIR="$ACTUAL_CKPT_DIRS"
echo "Checkpoint dir: ${ACTUAL_CKPT_DIR}"

# ============================================================
# Run evaluation
# ============================================================
COMPAT_FLAG=""
if [ "${RUN_COMPATIBILITY}" = "true" ]; then
    COMPAT_FLAG="--run_compatibility"
fi

ORACLE_FLAG=""
if [ "${RUN_ORACLE_ACCURACY}" = "true" ]; then
    ORACLE_FLAG="--run_oracle_accuracy"
fi

python -m scripts.pdlm_eval \
    --ckpt_dir="${ACTUAL_CKPT_DIR}" \
    --num_batches=${NUM_BATCHES} \
    --output_json="${CKPT_DIR}/eval_result.json" \
    ${COMPAT_FLAG} \
    ${ORACLE_FLAG}

echo ""
echo "============================================================"
echo "Results saved to: ${CKPT_DIR}/eval_result.json"
echo "============================================================"

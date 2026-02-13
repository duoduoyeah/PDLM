#!/bin/bash

## Mask PDLM Evaluation Script
## Part 1: Loss eval (unified + end2end with mask/group breakdown)
## Part 2: Generation eval (block-by-block iterative denoising)
##
## Usage:
##   bash launch/eval_mask_pdlm.sh --ckpt_dir=/path/to/model
##   bash launch/eval_mask_pdlm.sh --ckpt_dir=/path/to/model --num_batches=50
##   bash launch/eval_mask_pdlm.sh --ckpt_dir=/path/to/model --generate
##   bash launch/eval_mask_pdlm.sh --ckpt_dir=/path/to/model --generate --num_prompts=3 --prompt_tokens=32 --generate_tokens=64

# ============================================================
# Default values
# ============================================================
NUM_BATCHES="20"
CKPT_DIR=""  # Direct checkpoint path (required)
GENERATE="false"
NUM_PROMPTS="5"
PROMPT_TOKENS="16"
GENERATE_TOKENS="128"
TEMPERATURE="0.0"
TOPK="0"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --ckpt_dir=*)
            CKPT_DIR="${arg#*=}"
            ;;
        --num_batches=*)
            NUM_BATCHES="${arg#*=}"
            ;;
        --generate)
            GENERATE="true"
            ;;
        --num_prompts=*)
            NUM_PROMPTS="${arg#*=}"
            ;;
        --prompt_tokens=*)
            PROMPT_TOKENS="${arg#*=}"
            ;;
        --generate_tokens=*)
            GENERATE_TOKENS="${arg#*=}"
            ;;
        --temperature=*)
            TEMPERATURE="${arg#*=}"
            ;;
        --topk=*)
            TOPK="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_mask_pdlm.sh --ckpt_dir=/path/to/model"
            echo "       [--num_batches=20] [--generate] [--num_prompts=5]"
            echo "       [--prompt_tokens=64] [--generate_tokens=128]"
            echo "       [--temperature=0.0] [--topk=0]"
            exit 1
            ;;
    esac
done

if [ -z "${CKPT_DIR}" ]; then
    echo "Error: --ckpt_dir is required"
    echo "Usage: bash launch/eval_mask_pdlm.sh --ckpt_dir=/path/to/model"
    exit 1
fi

echo "============================================================"
echo "Mask PDLM Evaluation"
echo "============================================================"
echo "Checkpoint:      ${CKPT_DIR}"
echo "Num Batches:     ${NUM_BATCHES}"
echo "Generate:        ${GENERATE}"
if [ "${GENERATE}" = "true" ]; then
    echo "  Num Prompts:   ${NUM_PROMPTS}"
    echo "  Prompt Tokens: ${PROMPT_TOKENS}"
    echo "  Gen Tokens:    ${GENERATE_TOKENS}"
    echo "  Temperature:   ${TEMPERATURE}"
    echo "  Top-k:         ${TOPK}"
fi
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
# Part 1: Loss evaluation
# ============================================================
echo ""
echo "============================================================"
echo "Part 1: Loss Evaluation (unified + end2end)"
echo "============================================================"

python -m scripts.pdlm_eval \
    --ckpt_dir="${ACTUAL_CKPT_DIR}" \
    --num_batches=${NUM_BATCHES} \
    --output_json="${CKPT_DIR}/eval_result.json"

echo ""
echo "Results saved to: ${CKPT_DIR}/eval_result.json"

# ============================================================
# Part 2: Generation (optional)
# ============================================================
if [ "${GENERATE}" = "true" ]; then
    echo ""
    echo "============================================================"
    echo "Part 2: Generation (iterative denoising)"
    echo "============================================================"

    python -m scripts.pdlm_eval \
        --ckpt_dir="${ACTUAL_CKPT_DIR}" \
        --generate \
        --num_prompts=${NUM_PROMPTS} \
        --prompt_tokens=${PROMPT_TOKENS} \
        --generate_tokens=${GENERATE_TOKENS} \
        --temperature=${TEMPERATURE} \
        --topk=${TOPK}
fi

echo ""
echo "============================================================"
echo "Evaluation complete."
echo "============================================================"

#!/bin/bash

## PDLM Stage 2 Evaluation Script
## Downloads models from HuggingFace and runs evaluation locally.
##
## Usage:
##   bash launch/eval_pdlm.sh --depth=8
##   bash launch/eval_pdlm.sh --depth=8 --num_batches=50
##   bash launch/eval_pdlm.sh --ckpt_dir=/path/to/model

# ============================================================
# Default values
# ============================================================
DEPTH="8"
DATA_RATIO="20"
BLOCK_SIZE="4"
NUM_BATCHES="20"
LOCAL_DIR="/tmp/pdlm_eval"
HF_REPO=""  # Empty = use default pattern (duoduoyeah/pdlm_d${DEPTH})
CKPT_DIR=""  # Direct checkpoint path (overrides HF download)
RUN_COMPATIBILITY="false"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --depth=*)
            DEPTH="${arg#*=}"
            ;;
        --data_ratio=*)
            DATA_RATIO="${arg#*=}"
            ;;
        --block_size=*)
            BLOCK_SIZE="${arg#*=}"
            ;;
        --num_batches=*)
            NUM_BATCHES="${arg#*=}"
            ;;
        --local_dir=*)
            LOCAL_DIR="${arg#*=}"
            ;;
        --repo=*)
            HF_REPO="${arg#*=}"
            ;;
        --ckpt_dir=*)
            CKPT_DIR="${arg#*=}"
            ;;
        --run_compatibility)
            RUN_COMPATIBILITY="true"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_pdlm.sh --depth=8 [--data_ratio=20]"
            echo "       [--block_size=4] [--num_batches=20] [--local_dir=/tmp/pdlm_eval]"
            echo "       [--repo=duoduoyeah/pdlm_d8] [--ckpt_dir=/path/to/ckpt]"
            echo "       [--run_compatibility]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "PDLM Stage 2 Evaluation"
echo "============================================================"
echo "Depth:        ${DEPTH}"
echo "Data Ratio:   ${DATA_RATIO}"
echo "Block Size:   ${BLOCK_SIZE}"
echo "Num Batches:  ${NUM_BATCHES}"
echo "Local Dir:    ${LOCAL_DIR}"
echo "Compatibility: ${RUN_COMPATIBILITY}"
echo "============================================================"

# ============================================================
# Step 0: Ensure validation dataset exists
# ============================================================
echo ""
echo "Step 0: Checking for validation dataset..."

# Determine data directory (same logic as nanochat/common.py)
if [ -n "${NANOCHAT_BASE_DIR}" ]; then
    DATA_DIR="${NANOCHAT_BASE_DIR}/simple_story_data"
else
    DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
fi

# Check if any validation shards exist
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
# If direct checkpoint path provided, use it
# ============================================================
if [ -n "${CKPT_DIR}" ]; then
    echo ""
    echo "Using direct checkpoint path: ${CKPT_DIR}"

    # Create symlink in base dir pointing to tokenizer in ckpt folder
    BASE_DIR=$(dirname "${DATA_DIR}")
    TOKENIZER_LINK="${BASE_DIR}/tokenizer"
    MODEL_TOKENIZER="${CKPT_DIR}/tokenizer"

    if [ -d "${MODEL_TOKENIZER}" ]; then
        # Remove existing symlink if it points elsewhere
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

    # Find the directory containing model_*.pt files (handles nested structures)
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

    COMPAT_FLAG=""
    if [ "${RUN_COMPATIBILITY}" = "true" ]; then
        COMPAT_FLAG="--run_compatibility"
    fi

    python -m scripts.pdlm_eval \
        --ckpt_dir="${ACTUAL_CKPT_DIR}" \
        --num_batches=${NUM_BATCHES} \
        --output_json="${CKPT_DIR}/eval_result.json" \
        ${COMPAT_FLAG}

    echo ""
    echo "============================================================"
    echo "Results saved to: ${CKPT_DIR}/eval_result.json"
    echo "============================================================"
    exit 0
fi

# ============================================================
# Step 1: Download from HuggingFace
# ============================================================
# Build HF repo name
if [ -z "${HF_REPO}" ]; then
    HF_REPO="duoduoyeah/pdlm_d${DEPTH}"
fi

echo ""
echo "Step 1: Downloading models from HuggingFace..."
echo "HF Repo: ${HF_REPO}"

python -c "
from huggingface_hub import snapshot_download
import os

repo_id = '${HF_REPO}'
local_dir = '${LOCAL_DIR}'

print(f'Downloading {repo_id} to {local_dir}...')
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    repo_type='model',
)
print('Download complete!')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download from HuggingFace"
    exit 1
fi

# ============================================================
# Step 2: Discover models
# ============================================================
echo ""
echo "Step 2: Discovering models..."

# Build pattern for model directories
BASE_PATTERN="pdlm_d${DEPTH}_b${BLOCK_SIZE}"

# Find all matching model directories
MODELS=()
for dir in "${LOCAL_DIR}"/${BASE_PATTERN}*_r${DATA_RATIO}*; do
    if [ -d "$dir" ]; then
        MODELS+=("$dir")
    fi
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models found matching pattern ${BASE_PATTERN}*_r${DATA_RATIO}"
    echo "Available directories in ${LOCAL_DIR}:"
    ls -la "${LOCAL_DIR}/"
    exit 1
fi

echo "Found ${#MODELS[@]} model(s):"
for m in "${MODELS[@]}"; do
    echo "  - $(basename "$m")"
done

# ============================================================
# Step 3: Run evaluation for each model
# ============================================================
echo ""
echo "Step 3: Running evaluation..."

# Store results for summary
declare -a RESULTS

COMPAT_FLAG=""
if [ "${RUN_COMPATIBILITY}" = "true" ]; then
    COMPAT_FLAG="--run_compatibility"
fi

for MODEL_DIR in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_DIR")

    echo ""
    echo "------------------------------------------------------------"
    echo "Evaluating: ${MODEL_NAME}"
    echo "------------------------------------------------------------"

    # Create symlink to shared data directory so get_base_dir() finds data
    if [ ! -e "${MODEL_DIR}/simple_story_data" ]; then
        ln -s "${DATA_DIR}" "${MODEL_DIR}/simple_story_data"
        echo "  Created symlink: ${MODEL_DIR}/simple_story_data -> ${DATA_DIR}"
    fi

    # Set NANOCHAT_BASE_DIR to model directory so get_base_dir() finds both:
    # - tokenizer at ${MODEL_DIR}/tokenizer
    # - data at ${MODEL_DIR}/simple_story_data (symlink)
    export NANOCHAT_BASE_DIR="${MODEL_DIR}"

    # Find the directory containing model_*.pt files (handles nested structures)
    CKPT_DIRS=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u)
    CKPT_COUNT=$(echo "$CKPT_DIRS" | grep -c . 2>/dev/null || echo 0)

    if [ "$CKPT_COUNT" -eq 0 ]; then
        echo "Error: No checkpoints found in ${MODEL_DIR}/base_checkpoints"
        RESULTS+=("${MODEL_NAME}|-,-|NO CKPT")
        continue
    elif [ "$CKPT_COUNT" -gt 1 ]; then
        echo "Warning: Multiple checkpoint directories found:"
        echo "$CKPT_DIRS"
        echo "Skipping - please specify which one to use."
        RESULTS+=("${MODEL_NAME}|-,-|MULTI CKPT")
        continue
    fi

    CKPT_DIR="$CKPT_DIRS"
    echo "  Checkpoint dir: ${CKPT_DIR}"

    # Run evaluation with direct checkpoint path
    OUTPUT=$(python -m scripts.pdlm_eval \
        --ckpt_dir="${CKPT_DIR}" \
        --num_batches=${NUM_BATCHES} \
        --output_json="${MODEL_DIR}/eval_result.json" \
        ${COMPAT_FLAG} 2>&1)

    EVAL_STATUS=$?
    echo "$OUTPUT"

    if [ $EVAL_STATUS -eq 0 ]; then
        # Extract key metrics from JSON
        if [ -f "${MODEL_DIR}/eval_result.json" ]; then
            METRICS=$(python -c "
import json
with open('${MODEL_DIR}/eval_result.json', 'r') as f:
    result = json.load(f)
print(f\"{result['overall_loss']:.4f},{result['overall_ppl']:.2f}\")
" 2>/dev/null)
            RESULTS+=("${MODEL_NAME}|${METRICS}|OK")
        else
            RESULTS+=("${MODEL_NAME}|-,-|OK (no JSON)")
        fi
    else
        RESULTS+=("${MODEL_NAME}|-,-|FAILED")
    fi
done

# ============================================================
# Step 4: Print summary
# ============================================================
echo ""
echo "============================================================"
echo "EVALUATION SUMMARY"
echo "============================================================"
echo ""
printf "%-40s | %-10s | %-10s | %-10s\n" "Model" "Loss" "PPL" "Status"
printf "%s\n" "-------------------------------------------------------------------------"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r model metrics status <<< "$result"
    IFS=',' read -r loss ppl <<< "$metrics"
    printf "%-40s | %-10s | %-10s | %-10s\n" "$model" "$loss" "$ppl" "$status"
done

echo ""
echo "Results saved to: ${LOCAL_DIR}/*/eval_result.json"
echo "============================================================"

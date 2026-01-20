#!/bin/bash

## BD3LM Evaluation Script
## Downloads models from HuggingFace and runs evaluation locally.
##
## Usage:
##   bash launch/eval_bd3lm.sh --depth=8
##   bash launch/eval_bd3lm.sh --depth=8 --variant=normal
##   bash launch/eval_bd3lm.sh --depth=8 --data_ratio=20 --num_batches=50

# ============================================================
# Default values
# ============================================================
DEPTH="8"
VARIANT=""  # Empty = all variants (normal, ts1, ts2, ts3, ts4)
DATA_RATIO="20"
BLOCK_SIZE="4"
NUM_BATCHES="20"
LOCAL_DIR="/tmp/bd3lm_eval"
ADJUST="true"  # Use _adjust suffix models
HF_REPO=""  # Empty = use default pattern (duoduoyeah/bd3lm_d${DEPTH})

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --depth=*)
            DEPTH="${arg#*=}"
            ;;
        --variant=*)
            VARIANT="${arg#*=}"
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
        --adjust=*)
            ADJUST="${arg#*=}"
            ;;
        --repo=*)
            HF_REPO="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_bd3lm.sh --depth=8 [--variant=normal] [--data_ratio=20]"
            echo "       [--block_size=4] [--num_batches=20] [--local_dir=/tmp/bd3lm_eval] [--adjust=true]"
            echo "       [--repo=duoduoyeah/bd3lm_d8]"
            exit 1
            ;;
    esac
done

# Build HF repo name and suffix
if [ -z "${HF_REPO}" ]; then
    HF_REPO="duoduoyeah/bd3lm_d${DEPTH}"
fi
if [ "${ADJUST}" = "true" ]; then
    SUFFIX="_adjust"
else
    SUFFIX=""
fi

echo "============================================================"
echo "BD3LM Evaluation"
echo "============================================================"
echo "HF Repo:      ${HF_REPO}"
echo "Depth:        ${DEPTH}"
echo "Variant:      ${VARIANT:-all}"
echo "Data Ratio:   ${DATA_RATIO}"
echo "Block Size:   ${BLOCK_SIZE}"
echo "Num Batches:  ${NUM_BATCHES}"
echo "Local Dir:    ${LOCAL_DIR}"
echo "Adjust:       ${ADJUST}"
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
# Step 1: Download from HuggingFace
# ============================================================
echo ""
echo "Step 1: Downloading models from HuggingFace..."

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
# Pattern: bd3lm_d{depth}_b{block}_{variant}_r{ratio}[_adjust]
BASE_PATTERN="bd3lm_d${DEPTH}_b${BLOCK_SIZE}"

# Find all matching model directories
MODELS=()
for dir in "${LOCAL_DIR}"/${BASE_PATTERN}*_r${DATA_RATIO}${SUFFIX}; do
    if [ -d "$dir" ]; then
        MODELS+=("$dir")
    fi
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models found matching pattern ${BASE_PATTERN}*_r${DATA_RATIO}${SUFFIX}"
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

for MODEL_DIR in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_DIR")

    # Skip if variant filter is set and doesn't match the folder name
    if [ -n "${VARIANT}" ] && [[ ! "$MODEL_NAME" == *"_${VARIANT}_"* ]]; then
        echo "Skipping ${MODEL_NAME} (does not match variant filter: ${VARIANT})"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Evaluating: ${MODEL_NAME}"
    echo "  (target_shift auto-detected from checkpoint)"
    echo "------------------------------------------------------------"

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

    # Set NANOCHAT_BASE_DIR so get_tokenizer() finds the tokenizer in the model directory
    export NANOCHAT_BASE_DIR="${MODEL_DIR}"

    # Run evaluation with direct checkpoint path
    # Python script reads target_shift from checkpoint metadata automatically
    OUTPUT=$(python -m scripts.bd3lm_eval \
        --ckpt_dir="${CKPT_DIR}" \
        --num_batches=${NUM_BATCHES} \
        --output_json="${MODEL_DIR}/eval_result.json" 2>&1)

    EVAL_STATUS=$?
    echo "$OUTPUT"

    if [ $EVAL_STATUS -eq 0 ]; then
        # Extract key metrics from JSON
        if [ -f "${MODEL_DIR}/eval_result.json" ]; then
            METRICS=$(python -c "
import json
with open('${MODEL_DIR}/eval_result.json', 'r') as f:
    result = json.load(f)
# Handle both normal and target_shift mode
if 'overall_loss' in result:
    print(f\"{result['overall_loss']:.4f},{result['overall_ppl']:.2f}\")
else:
    print(f\"{result['loss']:.4f},{result['ppl']:.2f}\")
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

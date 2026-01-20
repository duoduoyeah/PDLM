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
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_bd3lm.sh --depth=8 [--variant=normal] [--data_ratio=20]"
            echo "       [--block_size=4] [--num_batches=20] [--local_dir=/tmp/bd3lm_eval] [--adjust=true]"
            exit 1
            ;;
    esac
done

# Build HF repo name and suffix
HF_REPO="duoduoyeah/bd3lm_d${DEPTH}"
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

# Function to get target_shift from variant name
get_target_shift() {
    local model_name="$1"
    if [[ "$model_name" == *"_normal_"* ]]; then
        echo "-1"
    elif [[ "$model_name" == *"_ts1_"* ]]; then
        echo "1"
    elif [[ "$model_name" == *"_ts2_"* ]]; then
        echo "2"
    elif [[ "$model_name" == *"_ts3_"* ]]; then
        echo "3"
    elif [[ "$model_name" == *"_ts4_"* ]]; then
        echo "4"
    else
        echo "-1"  # default to normal
    fi
}

# Function to get variant name from model name
get_variant_name() {
    local model_name="$1"
    if [[ "$model_name" == *"_normal_"* ]]; then
        echo "normal"
    elif [[ "$model_name" == *"_ts1_"* ]]; then
        echo "ts1"
    elif [[ "$model_name" == *"_ts2_"* ]]; then
        echo "ts2"
    elif [[ "$model_name" == *"_ts3_"* ]]; then
        echo "ts3"
    elif [[ "$model_name" == *"_ts4_"* ]]; then
        echo "ts4"
    else
        echo "unknown"
    fi
}

for MODEL_DIR in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    VARIANT_NAME=$(get_variant_name "$MODEL_NAME")
    TARGET_SHIFT=$(get_target_shift "$MODEL_NAME")

    # Skip if variant filter is set and doesn't match
    if [ -n "${VARIANT}" ] && [ "${VARIANT_NAME}" != "${VARIANT}" ]; then
        echo "Skipping ${MODEL_NAME} (variant=${VARIANT_NAME}, filter=${VARIANT})"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Evaluating: ${MODEL_NAME}"
    echo "  Variant: ${VARIANT_NAME}"
    echo "  Target Shift: ${TARGET_SHIFT}"
    echo "------------------------------------------------------------"

    # Set environment and run evaluation
    export NANOCHAT_BASE_DIR="${MODEL_DIR}"

    # Run evaluation and capture output
    OUTPUT=$(python -m scripts.bd3lm_eval \
        --target_shift=${TARGET_SHIFT} \
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
            RESULTS+=("${VARIANT_NAME}|${TARGET_SHIFT}|${METRICS}|OK")
        else
            RESULTS+=("${VARIANT_NAME}|${TARGET_SHIFT}|-,-|OK (no JSON)")
        fi
    else
        RESULTS+=("${VARIANT_NAME}|${TARGET_SHIFT}|-,-|FAILED")
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
printf "%-10s | %-12s | %-10s | %-10s | %-10s\n" "Variant" "Target Shift" "Loss" "PPL" "Status"
printf "%s\n" "---------------------------------------------------------------"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r variant ts metrics status <<< "$result"
    IFS=',' read -r loss ppl <<< "$metrics"
    printf "%-10s | %-12s | %-10s | %-10s | %-10s\n" "$variant" "$ts" "$loss" "$ppl" "$status"
done

echo ""
echo "Results saved to: ${LOCAL_DIR}/*/eval_result.json"
echo "============================================================"

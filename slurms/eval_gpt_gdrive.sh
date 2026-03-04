#!/bin/bash

## GPT (AR) evaluation from Google Drive.
## Downloads model via rclone, runs evaluation, saves results.
##
## Usage (standalone inside a SLURM job or interactive node):
##   bash slurms/eval_gpt_gdrive.sh \
##       --gdrive_folder=gpt_d8 \
##       --model=gpt_d8_next1_r40_v4096_implicit_simple
##   bash slurms/eval_gpt_gdrive.sh \
##       --gdrive_folder=next_token_ar \
##       --model=gpt_d8_next1_10_16_r40 \
##       --total_sequences=3200 \
##       --local_dir=$SCRATCH/gpt_eval

set -e

# ============================================================
# Environment setup (uv/venv/credentials)
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default values
# ============================================================
GDRIVE_ROOT="gdrive:nanochat"
GDRIVE_FOLDER=""
MODEL=""
TOTAL_SEQUENCES="3200"
LOCAL_DIR="${SCRATCH:-/tmp}/gpt_eval"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --gdrive_folder=*)
            GDRIVE_FOLDER="${arg#*=}"
            ;;
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --total_sequences=*)
            TOTAL_SEQUENCES="${arg#*=}"
            ;;
        --local_dir=*)
            LOCAL_DIR="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash slurms/eval_gpt_gdrive.sh \\"
            echo "    --gdrive_folder=gpt_d8 \\"
            echo "    --model=gpt_d8_next1_r40_v4096_implicit_simple \\"
            echo "    [--total_sequences=3200] [--local_dir=\$SCRATCH/gpt_eval]"
            exit 1
            ;;
    esac
done

if [ -z "${GDRIVE_FOLDER}" ] || [ -z "${MODEL}" ]; then
    echo "Error: --gdrive_folder and --model are required"
    exit 1
fi

GDRIVE_PATH="${GDRIVE_ROOT}/${GDRIVE_FOLDER}/${MODEL}"
MODEL_DIR="${LOCAL_DIR}/${MODEL}"

echo "============================================================"
echo "GPT Evaluation (Google Drive)"
echo "============================================================"
echo "GDrive path:     ${GDRIVE_PATH}"
echo "Local dir:       ${MODEL_DIR}"
echo "Total sequences: ${TOTAL_SEQUENCES}"
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
    echo "No validation data found in ${DATA_DIR}. Downloading..."
    python -m nanochat.dataset --split=val
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download validation dataset"
        exit 1
    fi
else
    echo "Validation data found in ${DATA_DIR}"
fi

# ============================================================
# Step 1: Download model from Google Drive
# ============================================================
echo ""
echo "Step 1: Downloading ${MODEL} from Google Drive..."
mkdir -p "${MODEL_DIR}"

rclone copy "${GDRIVE_PATH}/" "${MODEL_DIR}/" \
    --include "base_checkpoints/**" \
    --include "tokenizer/**" \
    --include "report/**" \
    --progress

if [ $? -ne 0 ]; then
    echo "Error: rclone download failed"
    exit 1
fi
echo "Download complete: ${MODEL_DIR}"

# ============================================================
# Step 2: Set up NANOCHAT_BASE_DIR and data symlink
# ============================================================
echo ""
echo "Step 2: Setting up data links..."

if [ ! -e "${MODEL_DIR}/simple_story_data" ]; then
    ln -s "${DATA_DIR}" "${MODEL_DIR}/simple_story_data"
    echo "  Created symlink: ${MODEL_DIR}/simple_story_data -> ${DATA_DIR}"
fi

export NANOCHAT_BASE_DIR="${MODEL_DIR}"
echo "  NANOCHAT_BASE_DIR=${MODEL_DIR}"

# ============================================================
# Step 3: Find checkpoint directory
# ============================================================
CKPT_DIR=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u)
CKPT_COUNT=$(echo "${CKPT_DIR}" | grep -c . 2>/dev/null || echo 0)

if [ "${CKPT_COUNT}" -eq 0 ]; then
    echo "Error: No checkpoints found in ${MODEL_DIR}/base_checkpoints"
    exit 1
elif [ "${CKPT_COUNT}" -gt 1 ]; then
    echo "Warning: Multiple checkpoint subdirectories found, using last:"
    CKPT_DIR=$(echo "${CKPT_DIR}" | tail -1)
fi
echo "  Checkpoint dir: ${CKPT_DIR}"

# ============================================================
# Step 4: Run evaluation
# ============================================================
echo ""
echo "Step 4: Running GPT evaluation (${TOTAL_SEQUENCES} sequences)..."

python -m scripts.gpt_eval \
    --ckpt_dir="${CKPT_DIR}" \
    --total_sequences=${TOTAL_SEQUENCES} \
    --output_json="${MODEL_DIR}/eval_result.json"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo ""
echo "Results saved to: ${MODEL_DIR}/eval_result.json"

echo ""
echo "============================================================"
echo "Done: ${MODEL}"
echo "============================================================"

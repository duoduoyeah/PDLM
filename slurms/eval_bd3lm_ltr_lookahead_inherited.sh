#!/bin/bash

## BD3-LM-Prime L2R sub-token lookahead evaluation.
## Downloads model via rclone if needed, runs eval, optionally pushes results.
##
## Usage:
##   bash slurms/eval_bd3lm_ltr_lookahead.sh \
##       --gdrive_folder=bd3lm_prime_d8 \
##       --model=bd3lm_prime_d8_b4_l2_r40 \
##       [--total_sequences=3200] [--push_results]

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

GDRIVE_ROOT="gdrive:nanochat"
GDRIVE_FOLDER=""
MODEL=""
TOTAL_SEQUENCES="3200"
LOCAL_DIR="${SCRATCH:-/tmp}/bd3lm_eval"
PUSH_RESULTS="false"

for arg in "$@"; do
    case $arg in
        --gdrive_folder=*) GDRIVE_FOLDER="${arg#*=}" ;;
        --model=*) MODEL="${arg#*=}" ;;
        --total_sequences=*) TOTAL_SEQUENCES="${arg#*=}" ;;
        --local_dir=*) LOCAL_DIR="${arg#*=}" ;;
        --push_results) PUSH_RESULTS="true" ;;
        *)
            echo "Unknown argument: $arg"
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
RUN_FOLDER="seq${TOTAL_SEQUENCES}_ltr_sub_lookahead"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "BD3-LM-Prime L2R Sub-Token Lookahead Evaluation"
echo "============================================================"
echo "Model:           ${MODEL}"
echo "Total sequences: ${TOTAL_SEQUENCES}"
echo "============================================================"

# --- Ensure validation data ---
if [ -n "${NANOCHAT_BASE_DIR}" ]; then
    DATA_DIR="${NANOCHAT_BASE_DIR}/simple_story_data"
else
    DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
fi
if [ -z "$(find "${DATA_DIR}" -maxdepth 1 -name 'validation_*.parquet' 2>/dev/null | head -1)" ]; then
    echo "Downloading validation data..."
    python -m nanochat.dataset --split=val
fi

# --- Download model if not present ---
CKPT_DIR=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)
if [ -z "${CKPT_DIR}" ]; then
    echo "Model not found locally, downloading from Google Drive..."
    mkdir -p "${MODEL_DIR}"
    rclone copy "${GDRIVE_PATH}/" "${MODEL_DIR}/" \
        --include "base_checkpoints/**" \
        --include "tokenizer/**" \
        --progress
    CKPT_DIR=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)
    if [ -z "${CKPT_DIR}" ]; then
        echo "Error: No checkpoints found after download"
        exit 1
    fi
else
    echo "Model found locally: ${CKPT_DIR}"
fi

# --- Data symlink ---
if [ ! -e "${MODEL_DIR}/simple_story_data" ]; then
    ln -s "${DATA_DIR}" "${MODEL_DIR}/simple_story_data"
fi
export NANOCHAT_BASE_DIR="${MODEL_DIR}"

# --- Run evaluation ---
OUT_JSON="${MODEL_DIR}/eval_ltr_sub_lookahead.json"
echo ""
echo "Running L2R sub-token lookahead evaluation (${TOTAL_SEQUENCES} sequences)..."

python -m scripts.bd3lm_eval \
    --ckpt_dir="${CKPT_DIR}" \
    --ltr_sub_lookahead \
    --total_sequences=${TOTAL_SEQUENCES} \
    --output_json="${OUT_JSON}"

echo ""
echo "Results saved to: ${OUT_JSON}"

# --- Push results (optional) ---
if [ "${PUSH_RESULTS}" = "true" ]; then
    echo ""
    echo "Uploading results to HuggingFace..."
    bash "${REPO_ROOT}/slurms/hf_upload.sh" \
        --repo_prefix=bd3lm \
        --run_folder="${RUN_FOLDER}" \
        --result_dirs="${MODEL_DIR}" \
        --eval_json=eval_ltr_sub_lookahead.json
fi

echo ""
echo "============================================================"
echo "Done: ${MODEL}"
echo "============================================================"

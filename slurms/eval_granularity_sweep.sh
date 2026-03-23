#!/bin/bash -l
#SBATCH --job-name="gran_sweep"
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:ada6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=/rhome/sli588/temp/gran_sweep_%j.out

## Evaluate a single mask_pdlm model with fresh-mask eval for the granularity sweep.
##
## Usage:
##   sbatch slurms/eval_granularity_sweep.sh --model=mask_pdlm_d8_b4_n64_k31_g1984_p50_r40
##
## Launch all 4:
##   for M in mask_pdlm_d8_b4_n64_k31_g1984_p50_r40 mask_pdlm_d8_b4_n128_k31_g992_p50_r40 \
##            mask_pdlm_d8_b4_n256_k31_g496_p50_r40 mask_pdlm_d8_b4_n512_k15_g120_p50_r40; do
##       sbatch slurms/eval_granularity_sweep.sh --model=$M
##   done

set -e

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Settings
# ============================================================
TOTAL_SEQ=3200
GDRIVE_ROOT="gdrive:nanochat"
GDRIVE_FOLDER="mask_pdlm_soft_sweep"
OUT_DIR="table_script/results/granularity_sweep"
LOCAL_DIR="${SCRATCH}/mask_pdlm_sweep_eval"
MODEL=""

for arg in "$@"; do
    case $arg in
        --model=*) MODEL="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "Error: --model is required"
    exit 1
fi

mkdir -p "${OUT_DIR}"

# ============================================================
# Helper: ensure model is downloaded and find its checkpoint
# ============================================================
MODEL_DIR="${LOCAL_DIR}/${MODEL}"

if [ -d "${MODEL_DIR}/base_checkpoints" ]; then
    echo "  Model already at ${MODEL_DIR}"
else
    echo "  Downloading ${MODEL} from Google Drive..."
    mkdir -p "${MODEL_DIR}"
    rclone copy "${GDRIVE_ROOT}/${GDRIVE_FOLDER}/${MODEL}/" "${MODEL_DIR}/" \
        --include "base_checkpoints/**" \
        --include "tokenizer/**" \
        --progress
    if [ $? -ne 0 ]; then
        echo "  Error: rclone download failed for ${MODEL}"
        exit 1
    fi
fi

# ============================================================
# Set up data links
# ============================================================
if [ -n "${NANOCHAT_BASE_DIR}" ]; then
    DATA_DIR="${NANOCHAT_BASE_DIR}/simple_story_data"
else
    DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
fi

VAL_SHARDS=$(find "${DATA_DIR}" -maxdepth 1 -name "validation_*.parquet" 2>/dev/null | head -1)
if [ -z "${VAL_SHARDS}" ]; then
    echo "No validation data found in ${DATA_DIR}. Downloading..."
    python -m nanochat.dataset --split=val
fi

if [ ! -e "${MODEL_DIR}/simple_story_data" ]; then
    ln -s "${DATA_DIR}" "${MODEL_DIR}/simple_story_data" 2>/dev/null || true
fi

BASE_DIR=$(dirname "${DATA_DIR}")
if [ -d "${MODEL_DIR}/tokenizer" ]; then
    TOKENIZER_LINK="${BASE_DIR}/tokenizer"
    if [ -L "${TOKENIZER_LINK}" ]; then rm "${TOKENIZER_LINK}"; fi
    ln -s "${MODEL_DIR}/tokenizer" "${TOKENIZER_LINK}" 2>/dev/null || true
fi

export NANOCHAT_BASE_DIR="${MODEL_DIR}"

# ============================================================
# Find checkpoint and run eval
# ============================================================
CKPT_DIR=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)
if [ -z "${CKPT_DIR}" ]; then
    echo "Error: No checkpoint found for ${MODEL}"
    exit 1
fi

RESULT_OUT="${OUT_DIR}/${MODEL}.json"

echo ""
echo "========================================================"
echo "Granularity Sweep Eval: ${MODEL}"
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Sequences:  ${TOTAL_SEQ}"
echo "  Output:     ${RESULT_OUT}"
echo "========================================================"

python -m scripts.pdlm_eval \
    --ckpt_dir="${CKPT_DIR}" \
    --total_sequences=${TOTAL_SEQ} \
    --fresh_mask_decode \
    --output_json="${RESULT_OUT}"

echo ""
echo "=== Done: ${MODEL} ==="
echo "Result: ${RESULT_OUT}"

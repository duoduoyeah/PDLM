#!/bin/bash -l
#SBATCH --job-name="mask_pdlm_refresh"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=/rhome/sli588/temp/mask_pdlm_refresh_%j.out

## Evaluate all mask_pdlm sweep models with the refresh end2end eval.
##
## The refresh eval runs an extra forward pass after each teacher-force step
## to update group tokens with the current ground-truth context before
## recording loss. This removes the staleness bias against high-p (soft) models.
##
## Reuses already-downloaded models from mask_pdlm_sweep_eval in $SCRATCH.
## If a model is not found locally, downloads it from Google Drive first.
##
## Results (JSON per model) saved to table_script/results/table_mask_pdlm_refresh/
##
## Usage:
##   sbatch slurms/table_mask_pdlm_refresh.sh
##   SKIP_EXISTING=true sbatch slurms/table_mask_pdlm_refresh.sh   # resume interrupted run
##   PRINT_ONLY=true  sbatch slurms/table_mask_pdlm_refresh.sh     # reprint table from cached JSONs

set -e

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Settings
# ============================================================
TOTAL_SEQ=3200
GDRIVE_ROOT="gdrive:nanochat"
GDRIVE_FOLDER="mask_pdlm_soft_sweep"
OUT_DIR="table_script/results/table_mask_pdlm_refresh"
LOCAL_DIR="${SCRATCH}/mask_pdlm_sweep_eval"   # reuse downloads from table_mask_pdlm_sweep.sh
SKIP_EXISTING="${SKIP_EXISTING:-false}"
PRINT_ONLY="${PRINT_ONLY:-false}"

mkdir -p "${OUT_DIR}"

# ============================================================
# Model lists (same as table_mask_pdlm_parallel.sh)
# ============================================================

P_SWEEP_K31=(
    mask_pdlm_d8_b4_n256_k31_g496_p0_r40
    mask_pdlm_d8_b4_n256_k31_g496_p30_r40
    mask_pdlm_d8_b4_n256_k31_g496_p50_r40
    mask_pdlm_d8_b4_n256_k31_g496_p70_r40
    mask_pdlm_d8_b4_n256_k31_g496_p90_r40
    mask_pdlm_d8_b4_n256_k31_g496_p100_r40
)

P_SWEEP_K15=(
    mask_pdlm_d8_b4_n512_k15_g120_p0_r40
    mask_pdlm_d8_b4_n512_k15_g120_p30_r40
    mask_pdlm_d8_b4_n512_k15_g120_p50_r40
    mask_pdlm_d8_b4_n512_k15_g120_p70_r40
    mask_pdlm_d8_b4_n512_k15_g120_p90_r40
    mask_pdlm_d8_b4_n512_k15_g120_p100_r40
)

B_SWEEP_EXTRA=(
    mask_pdlm_d8_b1_n512_k15_g120_p50_r40
    mask_pdlm_d8_b2_n512_k15_g120_p50_r40
    mask_pdlm_d8_b8_n512_k15_g120_p50_r40
    mask_pdlm_d8_b16_n512_k15_g120_p50_r40
)

ALL_MODELS=(
    "${P_SWEEP_K31[@]}"
    "${P_SWEEP_K15[@]}"
    "${B_SWEEP_EXTRA[@]}"
)

# ============================================================
# Helper: ensure model is downloaded and find its checkpoint
# ============================================================
_ensure_model() {
    local MODEL="$1"
    local MODEL_DIR="${LOCAL_DIR}/${MODEL}"

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
            return 1
        fi
    fi

    if [ -n "${NANOCHAT_BASE_DIR}" ]; then
        DATA_DIR="${NANOCHAT_BASE_DIR}/simple_story_data"
    else
        DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
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
}

_find_ckpt() {
    local MODEL_DIR="$1"
    find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1
}

# ============================================================
# Ensure validation dataset exists
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

# ============================================================
# Run evals
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "Mask PDLM Refresh Decode Evaluation"
    echo "  total_sequences=${TOTAL_SEQ}"
    echo "  models: ${#ALL_MODELS[@]}"
    echo "========================================================"

    for MODEL in "${ALL_MODELS[@]}"; do
        RESULT_OUT="${OUT_DIR}/${MODEL}.json"

        echo ""
        echo "=== ${MODEL} ==="

        if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
            echo "  Skipping (already exists: ${RESULT_OUT})"
            continue
        fi

        _ensure_model "${MODEL}"

        CKPT_DIR=$(_find_ckpt "${LOCAL_DIR}/${MODEL}")
        if [ -z "${CKPT_DIR}" ]; then
            echo "  Error: No checkpoint found for ${MODEL}"
            continue
        fi
        echo "  Checkpoint: ${CKPT_DIR}"

        python -m scripts.pdlm_eval \
            --ckpt_dir="${CKPT_DIR}" \
            --total_sequences=${TOTAL_SEQ} \
            --refresh_decode \
            --output_json="${RESULT_OUT}"

        if [ $? -ne 0 ]; then
            echo "  Error: Evaluation failed for ${MODEL}"
            continue
        fi
        echo "  Saved: ${RESULT_OUT}"
    done
fi

# ============================================================
# Analyze results: print tables + write summary.json + summary.csv
# ============================================================
echo ""
python3 table_script/analyze_mask_pdlm_refresh.py \
    --refresh_dir="${OUT_DIR}" \
    --sequential_dir="table_script/results/table_mask_pdlm_sweep"

echo ""
echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

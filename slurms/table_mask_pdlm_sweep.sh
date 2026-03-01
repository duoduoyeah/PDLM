#!/bin/bash -l
#SBATCH --job-name="mask_pdlm_sweep"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=/rhome/sli588/temp/mask_pdlm_sweep_%j.out

## Evaluate all mask_pdlm sweep models and print comparison tables.
##
## Runs two sweeps from gdrive:nanochat/mask_pdlm_soft_sweep/:
##   1. soft_p sweep: b=4, both tokenizers (k31/g496 and k15/g120), p=0..100
##   2. block_size sweep: p=50, k15/g120, b=1,2,4,8,16
##      (b4/p50/k15 reuses results from sweep 1 — evaluated once)
##
## Results (JSON per model) saved to table_script/results/table_mask_pdlm_sweep/
## and pushed to duoduoyeah/eval_results.
##
## Usage:
##   sbatch slurms/table_mask_pdlm_sweep.sh
##   SKIP_EXISTING=true sbatch slurms/table_mask_pdlm_sweep.sh   # resume interrupted run
##   PRINT_ONLY=true  sbatch slurms/table_mask_pdlm_sweep.sh     # reprint table from cached JSONs

set -e

# ============================================================
# Environment setup (uv/venv/credentials via scratch venv)
# ============================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Settings
# ============================================================
TOTAL_SEQ=3200
GDRIVE_FOLDER="mask_pdlm_soft_sweep"
OUT_DIR="table_script/results/table_mask_pdlm_sweep"
LOCAL_DIR="${SCRATCH}/mask_pdlm_sweep_eval"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
PRINT_ONLY="${PRINT_ONLY:-false}"

mkdir -p "${OUT_DIR}"

# ============================================================
# Model lists
# ============================================================

# soft_p sweep: k31/g496 tokenizer, block_size=4
P_SWEEP_K31=(
    mask_pdlm_d8_b4_n256_k31_g496_p0_r40
    mask_pdlm_d8_b4_n256_k31_g496_p30_r40
    mask_pdlm_d8_b4_n256_k31_g496_p50_r40
    mask_pdlm_d8_b4_n256_k31_g496_p70_r40
    mask_pdlm_d8_b4_n256_k31_g496_p90_r40
    mask_pdlm_d8_b4_n256_k31_g496_p100_r40
)

# soft_p sweep: k15/g120 tokenizer, block_size=4
P_SWEEP_K15=(
    mask_pdlm_d8_b4_n512_k15_g120_p0_r40
    mask_pdlm_d8_b4_n512_k15_g120_p30_r40
    mask_pdlm_d8_b4_n512_k15_g120_p50_r40
    mask_pdlm_d8_b4_n512_k15_g120_p70_r40
    mask_pdlm_d8_b4_n512_k15_g120_p90_r40
    mask_pdlm_d8_b4_n512_k15_g120_p100_r40
)

# block_size sweep: k15/g120, p=50 (b4 reuses P_SWEEP_K15 result — not listed here)
B_SWEEP_EXTRA=(
    mask_pdlm_d8_b1_n512_k15_g120_p50_r40
    mask_pdlm_d8_b2_n512_k15_g120_p50_r40
    mask_pdlm_d8_b8_n512_k15_g120_p50_r40
    mask_pdlm_d8_b16_n512_k15_g120_p50_r40
)

# 16 unique models total (b4/p50/k15 counted once in P_SWEEP_K15)
ALL_MODELS=(
    "${P_SWEEP_K31[@]}"
    "${P_SWEEP_K15[@]}"
    "${B_SWEEP_EXTRA[@]}"
)

# ============================================================
# Run evals
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "Mask PDLM Sweep Evaluation"
    echo "  total_sequences=${TOTAL_SEQ}"
    echo "  gdrive_folder=${GDRIVE_FOLDER}"
    echo "  models: ${#ALL_MODELS[@]} unique (16 total, b4/p50/k15 shared across sweeps)"
    echo "========================================================"

    for MODEL in "${ALL_MODELS[@]}"; do
        RESULT_LOCAL="${LOCAL_DIR}/${MODEL}/eval_result.json"
        RESULT_OUT="${OUT_DIR}/${MODEL}.json"

        echo ""
        echo "=== ${MODEL} ==="

        if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
            echo "  Skipping (already exists: ${RESULT_OUT})"
            continue
        fi

        bash slurms/eval_mask_pdlm_gdrive.sh \
            --gdrive_folder="${GDRIVE_FOLDER}" \
            --model="${MODEL}" \
            --total_sequences=${TOTAL_SEQ} \
            --local_dir="${LOCAL_DIR}" \
            --push_results

        cp "${RESULT_LOCAL}" "${RESULT_OUT}"
        echo "  Saved: ${RESULT_OUT}"
    done
fi

# ============================================================
# Analyze results: print tables + write summary.json + summary.csv
# ============================================================
echo ""
python3 table_script/analyze_mask_pdlm_sweep.py --results_dir="${OUT_DIR}"

echo ""
echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

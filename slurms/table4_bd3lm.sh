#!/bin/bash -l
#SBATCH --job-name="table4_bd3lm"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --output=/rhome/sli588/temp/table4_bd3lm_%j.out

## Evaluate all BD3LM models in left-to-right mode and print Table 4 comparison.
##
## Downloads from gdrive:nanochat/bd3lm_d8/, runs left-to-right eval,
## saves per-model JSONs to table_script/results/table4_bd3lm/,
## then runs analyze_table4.py to print BD3LM-LtoR vs PDLM-p0 comparison.
##
## Usage:
##   sbatch slurms/table4_bd3lm.sh
##   SKIP_EXISTING=true sbatch slurms/table4_bd3lm.sh   # resume interrupted run
##   PRINT_ONLY=true bash slurms/table4_bd3lm.sh        # reprint table from cached JSONs

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
GDRIVE_FOLDER="bd3lm_d8"
OUT_DIR="table_script/results/table4_bd3lm"
LOCAL_DIR="${SCRATCH}/table4_bd3lm_eval"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
PRINT_ONLY="${PRINT_ONLY:-false}"

mkdir -p "${OUT_DIR}"

# ============================================================
# Model list
# ============================================================
BD3LM_MODELS=(
    bd3lm_d8_b2_normal_r40
    bd3lm_d8_b4_normal_r40
    bd3lm_d8_b8_normal_r40
    bd3lm_d8_b16_normal_r40
)

# ============================================================
# Run evals
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "BD3LM Table 4 Evaluation (left-to-right mode)"
    echo "  total_sequences=${TOTAL_SEQ}"
    echo "  gdrive_folder=${GDRIVE_FOLDER}"
    echo "  models: ${#BD3LM_MODELS[@]}"
    echo "========================================================"

    for MODEL in "${BD3LM_MODELS[@]}"; do
        RESULT_LOCAL="${LOCAL_DIR}/${MODEL}/eval_result.json"
        RESULT_OUT="${OUT_DIR}/${MODEL}.json"

        echo ""
        echo "=== ${MODEL} ==="

        if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
            echo "  Skipping (already exists: ${RESULT_OUT})"
            continue
        fi

        bash "${REPO_ROOT}/slurms/eval_bd3lm_gdrive.sh" \
            --gdrive_folder="${GDRIVE_FOLDER}" \
            --model="${MODEL}" \
            --total_sequences=${TOTAL_SEQ} \
            --local_dir="${LOCAL_DIR}"

        cp "${RESULT_LOCAL}" "${RESULT_OUT}"
        echo "  Saved: ${RESULT_OUT}"
    done
fi

# ============================================================
# Analyze results: print Table 4 + write summary files
# ============================================================
echo ""
python3 "${REPO_ROOT}/table_script/analyze_table4.py" \
    --bd3lm_dir="${OUT_DIR}" \
    --pdlm_dir="table_script/results/table_mask_pdlm_sweep"

echo ""
echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

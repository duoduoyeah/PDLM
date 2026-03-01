#!/bin/bash -l
#SBATCH --job-name="table4_bd3lm_par"
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:ada6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0:50:00
#SBATCH --output=/rhome/sli588/temp/table4_bd3lm_par_%j.out

## Evaluate all 4 BD3LM models in parallel (one per GPU) in left-to-right mode.
##
## Requests 4×ada6000 for 50 min, launches each model on GPU 0-3 simultaneously.
## Per-model logs: /rhome/sli588/temp/table4_bd3lm_par_<model>_<jobid>.out
##
## Usage:
##   sbatch slurms/table4_bd3lm_parallel.sh
##   SKIP_EXISTING=true sbatch slurms/table4_bd3lm_parallel.sh
##   PRINT_ONLY=true bash slurms/table4_bd3lm_parallel.sh

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
GDRIVE_FOLDER="bd3lm_d8"
OUT_DIR="table_script/results/table4_bd3lm"
LOCAL_DIR="${SCRATCH}/table4_bd3lm_eval"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
PRINT_ONLY="${PRINT_ONLY:-false}"

mkdir -p "${OUT_DIR}"

# ============================================================
# Model list (4 models → GPU 0-3)
# ============================================================
BD3LM_MODELS=(
    bd3lm_d8_b2_normal_r40
    bd3lm_d8_b4_normal_r40
    bd3lm_d8_b8_normal_r40
    bd3lm_d8_b16_normal_r40
)

# ============================================================
# Run evals in parallel (one per GPU)
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "BD3LM Table 4 Evaluation (parallel, 4×ada6000)"
    echo "  total_sequences=${TOTAL_SEQ}"
    echo "  gdrive_folder=${GDRIVE_FOLDER}"
    echo "  models: ${#BD3LM_MODELS[@]}"
    echo "========================================================"

    PIDS=()
    for i in "${!BD3LM_MODELS[@]}"; do
        MODEL="${BD3LM_MODELS[$i]}"
        RESULT_OUT="${OUT_DIR}/${MODEL}.json"
        LOG_FILE="/rhome/sli588/temp/table4_bd3lm_par_${MODEL}_${SLURM_JOB_ID}.out"

        if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
            echo "  Skipping GPU ${i}: ${MODEL} (already exists)"
            continue
        fi

        echo "  Launching GPU ${i}: ${MODEL} → ${LOG_FILE}"
        CUDA_VISIBLE_DEVICES=${i} bash "${REPO_ROOT}/slurms/eval_bd3lm_gdrive.sh" \
            --gdrive_folder="${GDRIVE_FOLDER}" \
            --model="${MODEL}" \
            --total_sequences=${TOTAL_SEQ} \
            --local_dir="${LOCAL_DIR}" \
            > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "Waiting for ${#PIDS[@]} parallel eval(s) to finish..."
    FAIL=0
    for PID in "${PIDS[@]}"; do
        wait "${PID}" || FAIL=$?
    done
    if [ "${FAIL}" -ne 0 ]; then
        echo "Error: one or more evals failed (exit code ${FAIL})"
        exit "${FAIL}"
    fi
    echo "All evals complete."

    # Copy results to OUT_DIR
    for MODEL in "${BD3LM_MODELS[@]}"; do
        RESULT_LOCAL="${LOCAL_DIR}/${MODEL}/eval_result.json"
        RESULT_OUT="${OUT_DIR}/${MODEL}.json"
        if [ -f "${RESULT_LOCAL}" ]; then
            cp "${RESULT_LOCAL}" "${RESULT_OUT}"
            echo "  Saved: ${RESULT_OUT}"
        else
            echo "  Warning: missing result for ${MODEL}"
        fi
    done
fi

# ============================================================
# Analyze results
# ============================================================
echo ""
python3 "${REPO_ROOT}/table_script/analyze_table4.py" \
    --bd3lm_dir="${OUT_DIR}" \
    --pdlm_dir="table_script/results/table_mask_pdlm_sweep"

echo ""
echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

#!/bin/bash

## Table 4: Equivalence Check — parallel evaluation of 4 models.
##
## Launches AR (full), AR (10/16), BD3-LM, PDLM p=0 evaluations in parallel,
## then runs analysis script to print the table.
##
## Usage:
##   bash slurms/table4_equivalence.sh                          # run from interactive GPU node
##   bash slurms/table4_equivalence.sh --total_sequences=320    # quick test
##   PRINT_ONLY=true bash slurms/table4_equivalence.sh          # reprint from cached results
##   USE_SRUN=true bash slurms/table4_equivalence.sh            # use srun for parallel GPU jobs

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Settings
# ============================================================
TOTAL_SEQUENCES="3200"
OUT_DIR="table_script/results/table4_equivalence"
LOCAL_DIR="${REPO_ROOT}/table_script/results/table4_equiv_eval"
PRINT_ONLY="${PRINT_ONLY:-false}"
USE_SRUN="${USE_SRUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

# srun settings (only used when USE_SRUN=true)
PARTITION="short_gpu"
QOS="short_gpu"
GRES="gpu:ada6000:1"
TIME="0:30:00"
MEM="100G"

for arg in "$@"; do
    case $arg in
        --total_sequences=*)  TOTAL_SEQUENCES="${arg#*=}" ;;
        --partition=*)        PARTITION="${arg#*=}" ;;
        --time=*)             TIME="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

mkdir -p "${OUT_DIR}"

# ============================================================
# Model definitions
# ============================================================
# Format: KEY|EVAL_SCRIPT|GDRIVE_FOLDER|MODEL_NAME
MODELS=(
    "ar_full|eval_gpt_gdrive.sh|gpt_d8|gpt_d8_next1_r40_v4096_implicit_simple"
    "ar_10_16|eval_gpt_gdrive.sh|next_token_ar|gpt_d8_next1_10_16_r40"
    "bd3lm|eval_bd3lm_gdrive.sh|bd3lm_d8|bd3lm_d8_b4_normal_r40"
    "pdlm_p0|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p0_4s_r40"
)

# ============================================================
# Run evaluations
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "Table 4: Equivalence Check Evaluation"
    echo "  total_sequences=${TOTAL_SEQUENCES}"
    echo "  models: ${#MODELS[@]}"
    echo "  use_srun: ${USE_SRUN}"
    echo "========================================================"

    LOG_DIR="${REPO_ROOT}/slurms/logs"
    mkdir -p "${LOG_DIR}"

    PIDS=()
    KEYS=()
    MODEL_NAMES=()

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r KEY EVAL_SCRIPT GDRIVE_FOLDER MODEL_NAME <<< "${entry}"
        RESULT_OUT="${OUT_DIR}/${KEY}.json"
        RESULT_LOCAL="${LOCAL_DIR}/${MODEL_NAME}/eval_result.json"
        LOG_FILE="${LOG_DIR}/table4_${KEY}.log"

        if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
            echo "[skip] ${KEY} (already exists: ${RESULT_OUT})"
            continue
        fi

        echo "[launch] ${KEY}: ${EVAL_SCRIPT} ${MODEL_NAME}"

        if [ "${USE_SRUN}" = "true" ]; then
            srun --export="ALL,GPU_VENV=${GPU_VENV},SLURM_SUBMIT_DIR=${REPO_ROOT}" \
                -p "${PARTITION}" --qos="${QOS}" --gres="${GRES}" --time="${TIME}" --mem="${MEM}" \
                bash "${REPO_ROOT}/slurms/${EVAL_SCRIPT}" \
                    --gdrive_folder="${GDRIVE_FOLDER}" \
                    --model="${MODEL_NAME}" \
                    --total_sequences="${TOTAL_SEQUENCES}" \
                    --local_dir="${LOCAL_DIR}" \
                > "${LOG_FILE}" 2>&1 &
        else
            bash "${REPO_ROOT}/slurms/${EVAL_SCRIPT}" \
                --gdrive_folder="${GDRIVE_FOLDER}" \
                --model="${MODEL_NAME}" \
                --total_sequences="${TOTAL_SEQUENCES}" \
                --local_dir="${LOCAL_DIR}" \
            > "${LOG_FILE}" 2>&1 &
        fi

        PIDS+=($!)
        KEYS+=("${KEY}")
        MODEL_NAMES+=("${MODEL_NAME}")
    done

    # Wait for all jobs
    echo ""
    echo "--- Waiting for ${#PIDS[@]} jobs ---"
    FAILED=0
    for i in "${!PIDS[@]}"; do
        KEY="${KEYS[$i]}"
        PID="${PIDS[$i]}"
        LOG_FILE="${LOG_DIR}/table4_${KEY}.log"

        RESULT_LOCAL="${LOCAL_DIR}/${MODEL_NAMES[$i]}/eval_result.json"
        RESULT_OUT="${OUT_DIR}/${KEY}.json"

        if wait "${PID}"; then
            if [ -f "${RESULT_LOCAL}" ]; then
                cp "${RESULT_LOCAL}" "${RESULT_OUT}"
                echo "[done]   ${KEY} -> ${RESULT_OUT}"
            else
                echo "[WARN]   ${KEY}: eval succeeded but no result JSON at ${RESULT_LOCAL}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "[FAILED] ${KEY} (see ${LOG_FILE})"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ ${FAILED} -gt 0 ]; then
        echo ""
        echo "WARNING: ${FAILED} jobs failed. Check logs in ${LOG_DIR}/"
    fi
fi

# ============================================================
# Analyze results
# ============================================================
echo ""
python3 "${REPO_ROOT}/table_script/analyze_table4_equivalence.py" \
    --results_dir="${OUT_DIR}"

echo ""
echo "========================================================"
echo "JSON results in: ${OUT_DIR}/"
echo "========================================================"

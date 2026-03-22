#!/bin/bash

## Table 5: p-sweep — effect of group token softness on model decisiveness.
##
## Evaluates PDLM n256 (g496) at p=0,30,50,70,90,100 (all 4s models).
## Reports PPL, Accuracy, Ent-PPL, Argmax Prob.
##
## Usage:
##   bash slurms/table5_psweep.sh                          # run from interactive GPU node
##   bash slurms/table5_psweep.sh --total_sequences=320    # quick test
##   PRINT_ONLY=true bash slurms/table5_psweep.sh          # reprint from cached results
##   USE_SRUN=true bash slurms/table5_psweep.sh            # use srun for parallel GPU jobs

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Settings
# ============================================================
TOTAL_SEQUENCES="3200"
OUT_DIR="table_script/results/table5_psweep"
LOCAL_DIR="${SCRATCH:-/tmp}/mask_pdlm_eval"
PRINT_ONLY="${PRINT_ONLY:-false}"
USE_SRUN="${USE_SRUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
EVAL_EXTRA_ARGS="--fresh_mask_decode"

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
# Model definitions: n256 (g496) 4s models at all p values
# ============================================================
# Format: KEY|EVAL_SCRIPT|GDRIVE_FOLDER|MODEL_NAME
MODELS=(
    "p0|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p0_4s_r40"
    "p30|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p30_4s_r40"
    "p50|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p50_4s_r40"
    "p70|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p70_4s_r40"
    "p90|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p90_4s_r40"
    "p100|eval_mask_pdlm_gdrive.sh|mask_pdlm_soft_sweep|mask_pdlm_d8_b4_n256_k31_g496_p100_4s_r40"
)

# ============================================================
# Run evaluations
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "Table 5: p-sweep (Effect of Group Token Softness)"
    echo "  total_sequences=${TOTAL_SEQUENCES}"
    echo "  models: ${#MODELS[@]}"
    echo "  use_srun: ${USE_SRUN}"
    echo "========================================================"

    LOG_DIR="${REPO_ROOT}/slurms/logs"
    mkdir -p "${LOG_DIR}"

    FAILED=0

    if [ "${USE_SRUN}" = "true" ]; then
        # Parallel mode: launch all as separate srun jobs
        PIDS=()
        KEYS=()
        MODEL_NAMES=()

        for entry in "${MODELS[@]}"; do
            IFS='|' read -r KEY EVAL_SCRIPT GDRIVE_FOLDER MODEL_NAME <<< "${entry}"
            RESULT_OUT="${OUT_DIR}/${KEY}.json"
            LOG_FILE="${LOG_DIR}/table5_${KEY}.log"

            if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
                echo "[skip] ${KEY} (already exists: ${RESULT_OUT})"
                continue
            fi

            echo "[launch] ${KEY}: ${EVAL_SCRIPT} ${MODEL_NAME}"
            srun --export="ALL,GPU_VENV=${GPU_VENV},SLURM_SUBMIT_DIR=${REPO_ROOT}" \
                -p "${PARTITION}" --qos="${QOS}" --gres="${GRES}" --time="${TIME}" --mem="${MEM}" \
                bash "${REPO_ROOT}/slurms/${EVAL_SCRIPT}" \
                    --gdrive_folder="${GDRIVE_FOLDER}" \
                    --model="${MODEL_NAME}" \
                    --total_sequences="${TOTAL_SEQUENCES}" \
                    --local_dir="${LOCAL_DIR}" \
                    --eval_args="${EVAL_EXTRA_ARGS}" \
                > "${LOG_FILE}" 2>&1 &

            PIDS+=($!)
            KEYS+=("${KEY}")
            MODEL_NAMES+=("${MODEL_NAME}")
        done

        # Wait for all srun jobs
        echo ""
        echo "--- Waiting for ${#PIDS[@]} jobs ---"
        for i in "${!PIDS[@]}"; do
            KEY="${KEYS[$i]}"
            PID="${PIDS[$i]}"
            LOG_FILE="${LOG_DIR}/table5_${KEY}.log"
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
    else
        # Sequential mode: run one at a time on current GPU
        for entry in "${MODELS[@]}"; do
            IFS='|' read -r KEY EVAL_SCRIPT GDRIVE_FOLDER MODEL_NAME <<< "${entry}"
            RESULT_OUT="${OUT_DIR}/${KEY}.json"
            RESULT_LOCAL="${LOCAL_DIR}/${MODEL_NAME}/eval_result.json"
            LOG_FILE="${LOG_DIR}/table5_${KEY}.log"

            if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
                echo "[skip] ${KEY} (already exists: ${RESULT_OUT})"
                continue
            fi

            echo "[run] ${KEY}: ${EVAL_SCRIPT} ${MODEL_NAME}"
            if bash "${REPO_ROOT}/slurms/${EVAL_SCRIPT}" \
                --gdrive_folder="${GDRIVE_FOLDER}" \
                --model="${MODEL_NAME}" \
                --total_sequences="${TOTAL_SEQUENCES}" \
                --local_dir="${LOCAL_DIR}" \
                --eval_args="${EVAL_EXTRA_ARGS}" \
                > "${LOG_FILE}" 2>&1; then
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
    fi

    if [ ${FAILED} -gt 0 ]; then
        echo ""
        echo "WARNING: ${FAILED} jobs failed. Check logs in ${LOG_DIR}/"
    fi
fi

# ============================================================
# Analyze results
# ============================================================
echo ""
python3 "${REPO_ROOT}/table_script/analyze_table5_psweep.py" \
    --results_dir="${OUT_DIR}"

# ============================================================
# Upload to HuggingFace
# ============================================================
echo ""
echo "Uploading results to HuggingFace..."
python3 -c "
import os
from huggingface_hub import upload_folder

upload_folder(
    folder_path='${OUT_DIR}',
    path_in_repo='table5_psweep/seq${TOTAL_SEQUENCES}_fresh_mask',
    repo_id='duoduoyeah/eval_results',
    repo_type='dataset',
    token=os.environ.get('HF_TOKEN'),
)
print('Uploaded to duoduoyeah/eval_results/table5_psweep/seq${TOTAL_SEQUENCES}_fresh_mask/')
" || echo "Warning: HuggingFace upload failed (results still saved locally)"

echo ""
echo "========================================================"
echo "JSON results in: ${OUT_DIR}/"
echo "========================================================"

#!/bin/bash

## Parallel oracle evaluation (exp-c + exp-d) for all 4s mask_pdlm models.
## Runs 4 models at a time on ada6000 GPUs (1 GPU each).
##
## Usage:
##   bash slurms/eval_mask_pdlm_oracle_parallel.sh
##   bash slurms/eval_mask_pdlm_oracle_parallel.sh --total_sequences=128    # sanity test
##   bash slurms/eval_mask_pdlm_oracle_parallel.sh --total_sequences=3200 --push_results

set -e

# ============================================================
# Environment setup (uv sync once, export GPU_VENV for children)
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default values
# ============================================================
TOTAL_SEQUENCES="3200"
PUSH_FLAG=""
PARALLEL=4
PARTITION="short_gpu"
QOS="short_gpu"
GRES="gpu:ada6000:1"
TIME="0:20:00"
MEM="100G"
LOCAL_DIR="/rhome/sli588/slurms_output/mask_pdlm_eval"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --total_sequences=*)
            TOTAL_SEQUENCES="${arg#*=}"
            ;;
        --push_results)
            PUSH_FLAG="--push_results"
            ;;
        --parallel=*)
            PARALLEL="${arg#*=}"
            ;;
        --partition=*)
            PARTITION="${arg#*=}"
            ;;
        --time=*)
            TIME="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash slurms/eval_mask_pdlm_oracle_parallel.sh \\"
            echo "    [--total_sequences=3200] [--push_results] [--parallel=4] \\"
            echo "    [--partition=short_gpu] [--time=0:20:00]"
            exit 1
            ;;
    esac
done

# ============================================================
# Model list (4s only)
# ============================================================
GDRIVE_FOLDER="mask_pdlm_soft_sweep"

MODELS=(
    "mask_pdlm_d8_b4_n256_k31_g496_p0_4s_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p30_4s_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p50_4s_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p70_4s_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p90_4s_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p100_4s_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p0_4s_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p30_4s_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p50_4s_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p70_4s_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p90_4s_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p100_4s_r40"
)

NUM_MODELS=${#MODELS[@]}
LOG_DIR="${REPO_ROOT}/slurms/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Mask PDLM Oracle Evaluation (exp-c + exp-d) — Parallel"
echo "============================================================"
echo "Models:          ${NUM_MODELS}"
echo "Parallel jobs:   ${PARALLEL}"
echo "Partition:       ${PARTITION} (${GRES})"
echo "Time limit:      ${TIME}"
echo "Total sequences: ${TOTAL_SEQUENCES}"
echo "Push results:    ${PUSH_FLAG:-no}"
echo "============================================================"
echo ""

# ============================================================
# Run in batches of $PARALLEL
# ============================================================
PIDS=()
RUNNING_MODELS=()
FAILED=0
SUCCEEDED=0
SUCCEEDED_MODELS=()

launch_model() {
    local model=$1
    local log_file="${LOG_DIR}/oracle_eval_${model}.log"

    echo "[launch] ${model}"
    srun --export="ALL,GPU_VENV=${GPU_VENV},SLURM_SUBMIT_DIR=${REPO_ROOT}" \
        -p "${PARTITION}" --qos="${QOS}" --gres="${GRES}" --time="${TIME}" --mem="${MEM}" \
        bash "${REPO_ROOT}/slurms/eval_mask_pdlm_oracle.sh" \
            --gdrive_folder="${GDRIVE_FOLDER}" \
            --model="${model}" \
            --total_sequences="${TOTAL_SEQUENCES}" \
            --local_dir="${LOCAL_DIR}" \
        > "${log_file}" 2>&1 &

    PIDS+=($!)
    RUNNING_MODELS+=("${model}")
}

wait_for_batch() {
    echo ""
    echo "--- Waiting for batch of ${#PIDS[@]} jobs ---"
    for i in "${!PIDS[@]}"; do
        local pid=${PIDS[$i]}
        local model=${RUNNING_MODELS[$i]}
        local log_file="${LOG_DIR}/oracle_eval_${model}.log"

        if wait "${pid}"; then
            echo "[done]   ${model}"
            SUCCEEDED=$((SUCCEEDED + 1))
            SUCCEEDED_MODELS+=("${model}")
        else
            echo "[FAILED] ${model} (see ${log_file})"
            FAILED=$((FAILED + 1))
        fi
    done
    PIDS=()
    RUNNING_MODELS=()
}

for i in "${!MODELS[@]}"; do
    launch_model "${MODELS[$i]}"

    # When we've launched $PARALLEL jobs, wait for them all
    if [ $(( (i + 1) % PARALLEL )) -eq 0 ]; then
        wait_for_batch
    fi
done

# Wait for any remaining jobs
if [ ${#PIDS[@]} -gt 0 ]; then
    wait_for_batch
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All done: ${SUCCEEDED} succeeded, ${FAILED} failed out of ${NUM_MODELS}"
echo "============================================================"

# ============================================================
# Push results to HuggingFace (single upload for all succeeded)
# ============================================================
if [ -n "${PUSH_FLAG}" ] && [ ${#SUCCEEDED_MODELS[@]} -gt 0 ]; then
    RUN_FOLDER="seq${TOTAL_SEQUENCES}_oracle"

    # Build space-separated list of result dirs
    RESULT_DIRS=""
    for model in "${SUCCEEDED_MODELS[@]}"; do
        RESULT_DIRS="${RESULT_DIRS} ${LOCAL_DIR}/${model}"
    done

    bash "${REPO_ROOT}/slurms/hf_upload.sh" \
        --repo_prefix=mask_pdlm \
        --run_folder="${RUN_FOLDER}" \
        --eval_json=eval_oracle.json \
        --result_dirs="${RESULT_DIRS}"
    if [ $? -ne 0 ]; then
        echo "Warning: HuggingFace upload failed (results still saved locally in ${LOCAL_DIR})"
    fi
fi

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi

#!/bin/bash

## Parallel threshold decode evaluation for all mask_pdlm models.
## Runs 4 models at a time on ada6000 GPUs (1 GPU each).
##
## Usage:
##   bash slurms/eval_mask_pdlm_threshold_parallel.sh
##   bash slurms/eval_mask_pdlm_threshold_parallel.sh --total_sequences=3200
##   bash slurms/eval_mask_pdlm_threshold_parallel.sh --total_sequences=320 --push_results
##   bash slurms/eval_mask_pdlm_threshold_parallel.sh --only_4s          # 4-state only (12 models)
##   bash slurms/eval_mask_pdlm_threshold_parallel.sh --only_5s          # 5-state only (12 models)

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

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
FILTER="all"   # "all", "4s", "5s"

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
        --only_4s)
            FILTER="4s"
            ;;
        --only_5s)
            FILTER="5s"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash slurms/eval_mask_pdlm_threshold_parallel.sh \\"
            echo "    [--total_sequences=3200] [--push_results] [--parallel=4] \\"
            echo "    [--partition=ada6000] [--time=0:20:00] [--only_4s] [--only_5s]"
            exit 1
            ;;
    esac
done

# ============================================================
# Model list
# ============================================================
GDRIVE_FOLDER="mask_pdlm_soft_sweep"

# 4-state models
MODELS_4S=(
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

# 5-state models
MODELS_5S=(
    "mask_pdlm_d8_b4_n256_k31_g496_p0_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p30_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p50_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p70_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p90_r40"
    "mask_pdlm_d8_b4_n256_k31_g496_p100_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p0_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p30_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p50_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p70_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p90_r40"
    "mask_pdlm_d8_b4_n512_k15_g120_p100_r40"
)

# Build final model list based on filter
MODELS=()
case $FILTER in
    4s)
        MODELS=("${MODELS_4S[@]}")
        ;;
    5s)
        MODELS=("${MODELS_5S[@]}")
        ;;
    all)
        MODELS=("${MODELS_4S[@]}" "${MODELS_5S[@]}")
        ;;
esac

NUM_MODELS=${#MODELS[@]}
LOG_DIR="${REPO_ROOT}/slurms/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Mask PDLM Threshold Decode — Parallel Evaluation"
echo "============================================================"
echo "Filter:          ${FILTER}"
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

launch_model() {
    local model=$1
    local log_file="${LOG_DIR}/threshold_eval_${model}.log"

    echo "[launch] ${model}"
    SLURM_SUBMIT_DIR="${REPO_ROOT}" srun -p "${PARTITION}" --qos="${QOS}" --gres="${GRES}" --time="${TIME}" --mem="${MEM}" \
        bash "${REPO_ROOT}/slurms/eval_mask_pdlm_threshold.sh" \
            --gdrive_folder="${GDRIVE_FOLDER}" \
            --model="${model}" \
            --total_sequences="${TOTAL_SEQUENCES}" \
            ${PUSH_FLAG} \
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
        local log_file="${LOG_DIR}/threshold_eval_${model}.log"

        if wait "${pid}"; then
            echo "[done]   ${model}"
            SUCCEEDED=$((SUCCEEDED + 1))
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

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi

#!/bin/bash

## Parallel BD3-LM-Prime fresh sub-token lookahead evaluation.
## Submits 11 sbatch jobs (τ=0.0, 0.1, ..., 1.0), each on 1 ada6000 GPU.
## Optionally submits a 12th job (--dependency) to merge results and upload to HF.
##
## Usage:
##   bash slurms/eval_bd3lm_ltr_lookahead_parallel.sh
##   bash slurms/eval_bd3lm_ltr_lookahead_parallel.sh --push_results
##   bash slurms/eval_bd3lm_ltr_lookahead_parallel.sh --total_sequences=320  # quick test

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# Default values
# ============================================================
GDRIVE_FOLDER="bd3lm_prime_d8"
MODEL="bd3lm_prime_d8_b4_l2_r40"
TOTAL_SEQUENCES="3200"
PUSH_RESULTS="false"
PARTITION="short_gpu"
QOS="short_gpu"
GRES="gpu:ada6000:1"
TIME="2:00:00"
MEM="100G"
CPUS=8
LOCAL_DIR="/rhome/sli588/slurms_output/bd3lm_eval"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --total_sequences=*)
            TOTAL_SEQUENCES="${arg#*=}"
            ;;
        --push_results)
            PUSH_RESULTS="true"
            ;;
        --time=*)
            TIME="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash slurms/eval_bd3lm_ltr_lookahead_parallel.sh \\"
            echo "    [--total_sequences=3200] [--push_results] [--time=2:00:00]"
            exit 1
            ;;
    esac
done

# ============================================================
# Threshold list (11 values, no τ=∞)
# ============================================================
THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

LOG_DIR="${REPO_ROOT}/slurms/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "BD3-LM-Prime Fresh Sub-Token Lookahead — sbatch Submission"
echo "============================================================"
echo "Model:           ${MODEL}"
echo "Thresholds:      ${#THRESHOLDS[@]} (0.0 to 1.0)"
echo "Partition:       ${PARTITION} (${GRES})"
echo "Time limit:      ${TIME}"
echo "Total sequences: ${TOTAL_SEQUENCES}"
echo "Push results:    ${PUSH_RESULTS}"
echo "Local dir:       ${LOCAL_DIR}"
echo "============================================================"
echo ""

# ============================================================
# Submit 11 eval jobs
# ============================================================
JOB_IDS=()

for tau in "${THRESHOLDS[@]}"; do
    JOB_ID=$(sbatch \
        --partition="${PARTITION}" --qos="${QOS}" \
        --gres="${GRES}" --time="${TIME}" --mem="${MEM}" --cpus-per-task="${CPUS}" \
        --job-name="fresh_tau${tau}" \
        --output="${LOG_DIR}/fresh_tau${tau}_${MODEL}_%j.log" \
        --export="ALL,SLURM_SUBMIT_DIR=${REPO_ROOT}" \
        --wrap="bash ${REPO_ROOT}/slurms/eval_bd3lm_ltr_lookahead.sh \
            --gdrive_folder=${GDRIVE_FOLDER} \
            --model=${MODEL} \
            --total_sequences=${TOTAL_SEQUENCES} \
            --local_dir=${LOCAL_DIR} \
            --threshold=${tau}" \
        --parsable)

    echo "[submitted] τ=${tau}  job=${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

# ============================================================
# Submit merge + upload job (depends on all 11 finishing OK)
# ============================================================
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
MODEL_DIR="${LOCAL_DIR}/${MODEL}"
RUN_FOLDER="seq${TOTAL_SEQUENCES}_ltr_sub_lookahead_fresh"

MERGE_SCRIPT="
set -e
export REPO_ROOT=${REPO_ROOT}
cd ${REPO_ROOT}
source slurms/setup.sh

echo 'Merging per-τ result JSONs...'
python -c \"
import json, glob, os
model_dir = '${MODEL_DIR}'
pattern = os.path.join(model_dir, 'eval_ltr_sub_lookahead_fresh_tau*.json')
files = sorted(glob.glob(pattern))
merged = {'stage': 'bd3lm', 'ltr_sub_lookahead': {}}
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for tau, result in data.get('ltr_sub_lookahead', {}).items():
        merged['ltr_sub_lookahead'][tau] = result
out = os.path.join(model_dir, 'eval_ltr_sub_lookahead_fresh.json')
with open(out, 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'Merged {len(files)} files into {out}')
print(f'Thresholds: {sorted(merged[\"ltr_sub_lookahead\"].keys())}')
\"

if [ '${PUSH_RESULTS}' = 'true' ]; then
    echo 'Uploading to HuggingFace...'
    bash ${REPO_ROOT}/slurms/hf_upload.sh \
        --repo_prefix=bd3lm \
        --run_folder=${RUN_FOLDER} \
        --result_dirs=${MODEL_DIR} \
        --eval_json=eval_ltr_sub_lookahead_fresh.json
fi
echo 'Done.'
"

MERGE_JOB=$(sbatch \
    --partition=batch \
    --time=0:10:00 --mem=4G --cpus-per-task=1 \
    --job-name="fresh_merge" \
    --output="${LOG_DIR}/fresh_merge_${MODEL}_%j.log" \
    --dependency="afterok:${DEP_STR}" \
    --export="ALL,SLURM_SUBMIT_DIR=${REPO_ROOT}" \
    --wrap="${MERGE_SCRIPT}" \
    --parsable)

echo ""
echo "[submitted] merge+upload job=${MERGE_JOB} (depends on ${#JOB_IDS[@]} eval jobs)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOG_DIR}/"

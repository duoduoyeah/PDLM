#!/bin/bash

## Mask PDLM evaluation from Google Drive.
## Downloads model via rclone, runs evaluation, optionally pushes results to HuggingFace.
##
## Usage (standalone inside a SLURM job or interactive node):
##   bash slurms/eval_mask_pdlm_gdrive.sh \
##       --gdrive_folder=mask_pdlm_soft_sweep \
##       --model=mask_pdlm_d8_b4_n512_k15_g120_p50_r40
##   bash slurms/eval_mask_pdlm_gdrive.sh \
##       --gdrive_folder=mask_pdlm_soft_sweep \
##       --model=mask_pdlm_d8_b4_n512_k15_g120_p50_r40 \
##       --total_sequences=3200 \
##       --local_dir=$SCRATCH/mask_pdlm_eval \
##       --push_results

set -e

# ============================================================
# Environment setup (uv/venv/credentials)
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default values
# ============================================================
GDRIVE_ROOT="gdrive:nanochat"
GDRIVE_FOLDER=""
MODEL=""
TOTAL_SEQUENCES="3200"
LOCAL_DIR="${SCRATCH:-/tmp}/mask_pdlm_eval"
PUSH_RESULTS="false"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --gdrive_folder=*)
            GDRIVE_FOLDER="${arg#*=}"
            ;;
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --total_sequences=*)
            TOTAL_SEQUENCES="${arg#*=}"
            ;;
        --local_dir=*)
            LOCAL_DIR="${arg#*=}"
            ;;
        --push_results)
            PUSH_RESULTS="true"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash slurms/eval_mask_pdlm_gdrive.sh \\"
            echo "    --gdrive_folder=mask_pdlm_soft_sweep \\"
            echo "    --model=mask_pdlm_d8_b4_n512_k15_g120_p50_r40 \\"
            echo "    [--total_sequences=3200] [--local_dir=\$SCRATCH/mask_pdlm_eval] [--push_results]"
            exit 1
            ;;
    esac
done

if [ -z "${GDRIVE_FOLDER}" ] || [ -z "${MODEL}" ]; then
    echo "Error: --gdrive_folder and --model are required"
    exit 1
fi

GDRIVE_PATH="${GDRIVE_ROOT}/${GDRIVE_FOLDER}/${MODEL}"
MODEL_DIR="${LOCAL_DIR}/${MODEL}"
RUN_FOLDER="seq${TOTAL_SEQUENCES}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Mask PDLM Evaluation (Google Drive)"
echo "============================================================"
echo "GDrive path:     ${GDRIVE_PATH}"
echo "Local dir:       ${MODEL_DIR}"
echo "Total sequences: ${TOTAL_SEQUENCES}"
echo "Push results:    ${PUSH_RESULTS}"
echo "============================================================"

# ============================================================
# Step 0: Ensure validation dataset exists
# ============================================================
echo ""
echo "Step 0: Checking for validation dataset..."

if [ -n "${NANOCHAT_BASE_DIR}" ]; then
    DATA_DIR="${NANOCHAT_BASE_DIR}/simple_story_data"
else
    DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
fi

VAL_SHARDS=$(find "${DATA_DIR}" -maxdepth 1 -name "validation_*.parquet" 2>/dev/null | head -1)

if [ -z "${VAL_SHARDS}" ]; then
    echo "No validation data found in ${DATA_DIR}. Downloading..."
    python -m nanochat.dataset --split=val
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download validation dataset"
        exit 1
    fi
else
    echo "Validation data found in ${DATA_DIR}"
fi

# ============================================================
# Step 1: Download model from Google Drive
# ============================================================
echo ""
echo "Step 1: Downloading ${MODEL} from Google Drive..."
mkdir -p "${MODEL_DIR}"

rclone copy "${GDRIVE_PATH}/" "${MODEL_DIR}/" \
    --include "base_checkpoints/**" \
    --include "tokenizer/**" \
    --include "report/**" \
    --progress

if [ $? -ne 0 ]; then
    echo "Error: rclone download failed"
    exit 1
fi
echo "Download complete: ${MODEL_DIR}"

# ============================================================
# Step 2: Set up NANOCHAT_BASE_DIR and tokenizer symlink
# ============================================================
echo ""
echo "Step 2: Setting up data links..."

# Symlink simple_story_data so the model finds the dataset
if [ ! -e "${MODEL_DIR}/simple_story_data" ]; then
    ln -s "${DATA_DIR}" "${MODEL_DIR}/simple_story_data"
    echo "  Created symlink: ${MODEL_DIR}/simple_story_data -> ${DATA_DIR}"
fi

# Symlink tokenizer into NANOCHAT_BASE_DIR root (needed by get_base_dir)
BASE_DIR=$(dirname "${DATA_DIR}")
TOKENIZER_LINK="${BASE_DIR}/tokenizer"
MODEL_TOKENIZER="${MODEL_DIR}/tokenizer"

if [ -d "${MODEL_TOKENIZER}" ]; then
    if [ -L "${TOKENIZER_LINK}" ]; then
        rm "${TOKENIZER_LINK}"
    fi
    ln -s "${MODEL_TOKENIZER}" "${TOKENIZER_LINK}"
    echo "  Created symlink: ${TOKENIZER_LINK} -> ${MODEL_TOKENIZER}"
fi

export NANOCHAT_BASE_DIR="${MODEL_DIR}"
echo "  NANOCHAT_BASE_DIR=${MODEL_DIR}"

# ============================================================
# Step 3: Find checkpoint directory
# ============================================================
CKPT_DIR=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u)
CKPT_COUNT=$(echo "${CKPT_DIR}" | grep -c . 2>/dev/null || echo 0)

if [ "${CKPT_COUNT}" -eq 0 ]; then
    echo "Error: No checkpoints found in ${MODEL_DIR}/base_checkpoints"
    exit 1
elif [ "${CKPT_COUNT}" -gt 1 ]; then
    echo "Warning: Multiple checkpoint subdirectories found, using last:"
    CKPT_DIR=$(echo "${CKPT_DIR}" | tail -1)
fi
echo "  Checkpoint dir: ${CKPT_DIR}"

# ============================================================
# Step 4: Run evaluation
# ============================================================
echo ""
echo "Step 4: Running evaluation (${TOTAL_SEQUENCES} sequences)..."

python -m scripts.pdlm_eval \
    --ckpt_dir="${CKPT_DIR}" \
    --total_sequences=${TOTAL_SEQUENCES} \
    --output_json="${MODEL_DIR}/eval_result.json"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo ""
echo "Results saved to: ${MODEL_DIR}/eval_result.json"

# ============================================================
# Step 5: Push results to HuggingFace (optional)
# ============================================================
if [ "${PUSH_RESULTS}" = "true" ]; then
    echo ""
    echo "Step 5: Uploading results to duoduoyeah/eval_results..."

    python -c "
import json, os, tempfile, shutil
from huggingface_hub import upload_folder

model_name    = '${MODEL}'
gdrive_folder = '${GDRIVE_FOLDER}'
run_folder    = '${RUN_FOLDER}'
model_dir     = '${MODEL_DIR}'
timestamp     = '${TIMESTAMP}'

args = {k: v for k, v in {
    'script':          'slurms/eval_mask_pdlm_gdrive.sh',
    'gdrive_folder':   gdrive_folder,
    'model':           model_name,
    'total_sequences': '${TOTAL_SEQUENCES}',
    'local_dir':       '${LOCAL_DIR}',
    'timestamp':       timestamp,
}.items() if v}

tmp = tempfile.mkdtemp()
try:
    shutil.copy(f'{model_dir}/eval_result.json', f'{tmp}/eval_result_{timestamp}.json')
    with open(f'{tmp}/args_{timestamp}.json', 'w') as f:
        json.dump(args, f, indent=2)
    upload_folder(
        folder_path=tmp,
        path_in_repo=f'mask_pdlm/{model_name}/{run_folder}',
        repo_id='duoduoyeah/eval_results',
        repo_type='dataset',
        token=os.environ.get('HF_TOKEN'),
    )
    print(f'Uploaded to duoduoyeah/eval_results/mask_pdlm/{model_name}/{run_folder}/eval_result_{timestamp}.json')
finally:
    shutil.rmtree(tmp)
"
    if [ $? -ne 0 ]; then
        echo "Warning: Upload to HuggingFace failed (results still saved locally)"
    fi
fi

echo ""
echo "============================================================"
echo "Done: ${MODEL}"
echo "============================================================"

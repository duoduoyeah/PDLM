#!/bin/bash

## GPT Evaluation Script
## Either copies from a local path or downloads from HuggingFace, then runs evaluation.
##
## Usage (local):
##   bash launch/eval_gpt.sh --ckpt_path=/path/to/model
##   bash launch/eval_gpt.sh --ckpt_path=/path/to/model --total_sequences=3200
##
## Usage (HuggingFace):
##   bash launch/eval_gpt.sh --repo=duoduoyeah/next_token_predictor --model=gpt_d8_next1_r40_v4096_implicit_simple
##   bash launch/eval_gpt.sh --repo=duoduoyeah/next_token_predictor --model=gpt_d8_next1_r40_v4096_implicit_simple --total_sequences=3200
##
## The model folder should contain:
##   - base_checkpoints/  (with model_*.pt files)
##   - tokenizer/
##
## For private repos, HF_TOKEN must be set in the environment (e.g. via source launch/.env).

# ============================================================
# Default values
# ============================================================
CKPT_PATH=""
HF_REPO=""
HF_MODEL=""
TOTAL_SEQUENCES=""
NUM_BATCHES="20"
LOCAL_DIR=""
PUSH_RESULTS="false"  # If true, upload eval_result.json + args.json to duoduoyeah/eval_results

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --ckpt_path=*)
            CKPT_PATH="${arg#*=}"
            ;;
        --repo=*)
            HF_REPO="${arg#*=}"
            ;;
        --model=*)
            HF_MODEL="${arg#*=}"
            ;;
        --total_sequences=*)
            TOTAL_SEQUENCES="${arg#*=}"
            ;;
        --num_batches=*)
            NUM_BATCHES="${arg#*=}"
            ;;
        --local_dir=*)
            LOCAL_DIR="${arg#*=}"
            ;;
        --push_results)
            PUSH_RESULTS="true"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash launch/eval_gpt.sh --ckpt_path=/path/to/model"
            echo "       bash launch/eval_gpt.sh --repo=owner/repo --model=model_name"
            echo "       [--total_sequences=3200] [--num_batches=20] [--local_dir=/tmp/gpt_eval]"
            echo "       [--push_results]"
            exit 1
            ;;
    esac
done

# Build eval size arg (total_sequences takes priority over num_batches)
if [ -n "${TOTAL_SEQUENCES}" ]; then
    EVAL_SIZE_ARG="--total_sequences=${TOTAL_SEQUENCES}"
    RUN_FOLDER="seq${TOTAL_SEQUENCES}"
else
    EVAL_SIZE_ARG="--num_batches=${NUM_BATCHES}"
    RUN_FOLDER="batches${NUM_BATCHES}"
fi

# ============================================================
# Case A: HuggingFace download
# ============================================================
if [ -n "${HF_REPO}" ]; then
    if [ -z "${HF_MODEL}" ]; then
        echo "Error: --model is required when --repo is set"
        exit 1
    fi

    if [ -z "${LOCAL_DIR}" ]; then
        LOCAL_DIR="/tmp/gpt_eval"
    fi

    echo "============================================================"
    echo "GPT Evaluation (HuggingFace)"
    echo "============================================================"
    echo "Repo:             ${HF_REPO}"
    echo "Model:            ${HF_MODEL}"
    echo "Local Dir:        ${LOCAL_DIR}"
    echo "Eval size arg:    ${EVAL_SIZE_ARG}"
    echo "============================================================"

    mkdir -p "${LOCAL_DIR}"

    # --------------------------------------------------------
    # Step 1: Download model from HuggingFace
    # --------------------------------------------------------
    echo ""
    echo "Step 1: Downloading model from HuggingFace..."

    python -c "
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id='${HF_REPO}',
    local_dir='${LOCAL_DIR}',
    repo_type='model',
    allow_patterns='${HF_MODEL}/**',
    token=os.environ.get('HF_TOKEN'),
)
print('Download complete!')
"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download from HuggingFace"
        exit 1
    fi

    MODEL_DIR="${LOCAL_DIR}/${HF_MODEL}"

# ============================================================
# Case B: Local path
# ============================================================
elif [ -n "${CKPT_PATH}" ]; then
    if [ ! -d "${CKPT_PATH}" ]; then
        echo "Error: Checkpoint path does not exist: ${CKPT_PATH}"
        exit 1
    fi

    if [ -z "${LOCAL_DIR}" ]; then
        MODEL_NAME=$(basename "${CKPT_PATH}")
        LOCAL_DIR="/tmp/gpt_eval/${MODEL_NAME}"
    fi

    echo "============================================================"
    echo "GPT Evaluation (local)"
    echo "============================================================"
    echo "Checkpoint:       ${CKPT_PATH}"
    echo "Local Dir:        ${LOCAL_DIR}"
    echo "Eval size arg:    ${EVAL_SIZE_ARG}"
    echo "============================================================"

    if [ -d "${LOCAL_DIR}" ]; then
        echo ""
        echo "Removing existing local directory: ${LOCAL_DIR}"
        rm -rf "${LOCAL_DIR}"
    fi
    mkdir -p "${LOCAL_DIR}"

    # --------------------------------------------------------
    # Step 1: Copy tokenizer
    # --------------------------------------------------------
    echo ""
    echo "Step 1: Copying tokenizer..."

    if [ ! -d "${CKPT_PATH}/tokenizer" ]; then
        echo "Error: No tokenizer directory found at ${CKPT_PATH}/tokenizer"
        exit 1
    fi

    mkdir -p "${LOCAL_DIR}/tokenizer"
    cp "${CKPT_PATH}/tokenizer/"* "${LOCAL_DIR}/tokenizer/"
    echo "  Copied tokenizer to ${LOCAL_DIR}/tokenizer/"

    # --------------------------------------------------------
    # Step 2: Copy checkpoints
    # --------------------------------------------------------
    echo ""
    echo "Step 2: Copying checkpoints..."

    if [ ! -d "${CKPT_PATH}/base_checkpoints" ]; then
        echo "Error: No base_checkpoints directory found at ${CKPT_PATH}/base_checkpoints"
        exit 1
    fi

    cp -r "${CKPT_PATH}/base_checkpoints" "${LOCAL_DIR}/"
    echo "  Copied base_checkpoints to ${LOCAL_DIR}/base_checkpoints/"

    MODEL_DIR="${LOCAL_DIR}"

else
    echo "Error: either --ckpt_path or --repo+--model is required"
    exit 1
fi

# ============================================================
# Step 3: Download validation dataset (if needed)
# ============================================================
echo ""
echo "Step 3: Downloading validation dataset (if needed)..."

export NANOCHAT_BASE_DIR="${MODEL_DIR}"
python -m nanochat.dataset --split=val

if [ $? -ne 0 ]; then
    echo "Error: Failed to download validation dataset"
    exit 1
fi

# ============================================================
# Step 4: Find checkpoint directory
# ============================================================
echo ""
echo "Step 4: Finding checkpoint directory..."

INNER_CKPT_DIR=$(find "${MODEL_DIR}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | head -1)

if [ -z "${INNER_CKPT_DIR}" ]; then
    echo "Error: No checkpoints found in ${MODEL_DIR}/base_checkpoints"
    echo "Available directories:"
    ls -la "${MODEL_DIR}/base_checkpoints/"
    exit 1
fi

echo "  Found checkpoint dir: ${INNER_CKPT_DIR}"

# ============================================================
# Step 5: Run evaluation
# ============================================================
echo ""
echo "Step 5: Running evaluation..."

python -m scripts.gpt_eval \
    --ckpt_dir="${INNER_CKPT_DIR}" \
    ${EVAL_SIZE_ARG} \
    --output_json="${MODEL_DIR}/eval_result.json"

EVAL_STATUS=$?

if [ $EVAL_STATUS -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

# ============================================================
# Step 6: Print summary
# ============================================================
echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${MODEL_DIR}/eval_result.json"
echo ""
cat "${MODEL_DIR}/eval_result.json"
echo ""
echo "============================================================"

if [ "${PUSH_RESULTS}" = "true" ]; then
    echo ""
    echo "Uploading results to duoduoyeah/eval_results..."
    python -c "
import json, os, tempfile, shutil
from huggingface_hub import upload_folder

model_name = '${HF_MODEL}'
run_folder  = '${RUN_FOLDER}'
model_dir   = '${MODEL_DIR}'

args = {k: v for k, v in {
    'script':          'eval_gpt.sh',
    'hf_repo':         '${HF_REPO}',
    'model':           model_name,
    'ckpt_path':       '${CKPT_PATH}',
    'total_sequences': '${TOTAL_SEQUENCES}',
    'num_batches':     '${NUM_BATCHES}',
    'local_dir':       '${LOCAL_DIR}',
}.items() if v}

tmp = tempfile.mkdtemp()
try:
    shutil.copy(f'{model_dir}/eval_result.json', f'{tmp}/eval_result.json')
    with open(f'{tmp}/args.json', 'w') as f:
        json.dump(args, f, indent=2)
    upload_folder(
        folder_path=tmp,
        path_in_repo=f'gpt/{model_name}/{run_folder}',
        repo_id='duoduoyeah/eval_results',
        repo_type='dataset',
        token=os.environ.get('HF_TOKEN'),
        commit_message=f'eval: gpt {model_name} {run_folder}',
    )
    print(f'Uploaded to duoduoyeah/eval_results/gpt/{model_name}/{run_folder}/')
finally:
    shutil.rmtree(tmp)
" || echo "Warning: Failed to upload results to HuggingFace"
fi

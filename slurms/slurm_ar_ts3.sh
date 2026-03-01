#!/bin/bash -l
#SBATCH --job-name="ar_ts3"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=/rhome/sli588/temp/ar_ts3_%j.out

## Train AR model with target_shift=3 (predicts token 3 positions ahead).
## Fills the missing ts=3 slot alongside existing ts=1,2,4 models on HF.
##
## Tokenizer: standard 4096-vocab tokenizer (no mask token) from Drive.
##   gdrive:nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer/
##
## Outputs (non-test run):
##   Drive: gdrive:nanochat/next_token_ar/gpt_d8_next3_r40/
##   HF:    duoduoyeah/next_token_predictor  (folder: gpt_d8_next3_r40)
##
## Usage:
##   # test run (direct, no sbatch):
##   bash slurms/slurm_ar_ts3.sh --test_mode=true --data_ratio=1
##
##   # full run:
##   sbatch slurms/slurm_ar_ts3.sh

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/slurms/setup.sh"

# Load HF_TOKEN for upload
if [ -f "${REPO_ROOT}/launch/.env" ]; then
    source "${REPO_ROOT}/launch/.env"
fi

# ============================================================
# Default parameters
# ============================================================
DEPTH="8"
DATA_RATIO="40"
TEST_MODE="false"
DRIVE_OUTPUT_FOLDER="next_token_ar"
HF_REPO="duoduoyeah/next_token_predictor"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --depth=*)              DEPTH="${arg#*=}" ;;
        --data_ratio=*)         DATA_RATIO="${arg#*=}" ;;
        --test_mode=*)          TEST_MODE="${arg#*=}" ;;
        --drive_output_folder=*) DRIVE_OUTPUT_FOLDER="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ============================================================
# Derived names
# ============================================================
BASE_MODEL_NAME="gpt_d${DEPTH}_next3_r${DATA_RATIO}"

if [ "${TEST_MODE}" = "true" ]; then
    MODEL_NAME="${BASE_MODEL_NAME}_test"
else
    MODEL_NAME="${BASE_MODEL_NAME}"
fi

WANDB_GROUP="gpt_d${DEPTH}"
DRIVE_BASE="gdrive:nanochat"
NANOCHAT_BASE_DIR="${SCRATCH:-/tmp}/ar_ts3_temp/${MODEL_NAME}"

export MODEL_NAME NANOCHAT_BASE_DIR

echo "=== ar_ts3: ${MODEL_NAME} ==="
echo "=== Scratch dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Depth: ${DEPTH}, data_ratio: ${DATA_RATIO}, test_mode: ${TEST_MODE} ==="

# ============================================================
# Fetch tokenizer: Drive → scratch
# Standard 4096-vocab tokenizer (no mask token, vocab_size=4096)
# ============================================================
TOKENIZER_SRC="${DRIVE_BASE}/tokenizer/simplestory_tokenizer/4096/tokenizer"

mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"
echo "Fetching tokenizer from ${TOKENIZER_SRC} ..."
rclone copy "${TOKENIZER_SRC}/tokenizer.pkl"  "${NANOCHAT_BASE_DIR}/tokenizer/"
rclone copy "${TOKENIZER_SRC}/token_bytes.pt" "${NANOCHAT_BASE_DIR}/tokenizer/"
echo "Tokenizer ready:"
ls -lh "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate tokenizer
python -c "
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
print(f'Tokenizer loaded: vocab_size={tok.vocab_size}')
assert tok.vocab_size == 4096, f'Expected 4096, got {tok.vocab_size}'
print('Tokenizer OK: vocab_size=4096, no mask token (AR)')
"

# ============================================================
# Download dataset to scratch
# ============================================================
echo "Downloading dataset..."
python -m nanochat.dataset -n 10 --split both

# ============================================================
# Training
# ============================================================
echo "Starting AR ts=3 training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=next_token_ar \
    --target_shift=3 \
    --depth=${DEPTH} \
    --max_seq_len=512 \
    --device_batch_size=64 \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=2500 \
    --eval_num_batches=20 \
    --eval_num_batches_final=100

# ============================================================
# Upload results: scratch → Drive + HF
# ============================================================
echo "=== Training complete. Uploading results... ==="

rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

# -- Drive --
DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${DRIVE_OUTPUT_FOLDER}/${MODEL_NAME}"
echo "Uploading to Drive: ${DRIVE_OUTPUT_DIR}"
rclone copy "${NANOCHAT_BASE_DIR}" "${DRIVE_OUTPUT_DIR}"
echo "Drive upload complete."

# -- HuggingFace (non-test runs only) --
if [ "${TEST_MODE}" != "true" ]; then
    echo "Uploading to HF: ${HF_REPO}/${MODEL_NAME}"
    python -c "
import os
from huggingface_hub import upload_folder

model_dir  = '${NANOCHAT_BASE_DIR}'
model_name = '${MODEL_NAME}'
hf_repo    = '${HF_REPO}'
token      = os.environ.get('HF_TOKEN')

if not token:
    print('Warning: HF_TOKEN not set, skipping HF upload')
else:
    upload_folder(
        folder_path=model_dir,
        path_in_repo=model_name,
        repo_id=hf_repo,
        repo_type='model',
        token=token,
        ignore_patterns=['simple_story_data/**', 'tokenized_data/**'],
        commit_message=f'add {model_name}',
    )
    print(f'HF upload complete: {hf_repo}/{model_name}')
"
else
    echo "Test mode: skipping HF upload."
fi

echo "=== Done: ${MODEL_NAME} ==="

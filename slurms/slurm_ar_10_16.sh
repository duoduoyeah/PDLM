#!/bin/bash -l
#SBATCH --job-name="ar_10_16"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=/rhome/sli588/temp/ar_10_16_%j.out

## Train AR model (next1) with 4-state block loss mask (10/16 effective tokens).
## Matches mask_pdlm 4-state supervision ratio for Table 4 equivalence check.
##
## Tokenizer: standard 4096-vocab tokenizer (no mask token) from Drive.
##   gdrive:nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer/
##
## Outputs (non-test run):
##   Drive: gdrive:nanochat/next_token_ar/gpt_d8_next1_10_16_r40/
##
## Usage:
##   # test run (direct, no sbatch):
##   bash slurms/slurm_ar_10_16.sh --test_mode=true --data_ratio=1
##
##   # full run:
##   sbatch slurms/slurm_ar_10_16.sh

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default parameters
# ============================================================
DEPTH="8"
DATA_RATIO="40"
TEST_MODE="false"
DRIVE_OUTPUT_FOLDER="next_token_ar"

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
BASE_MODEL_NAME="gpt_d${DEPTH}_next1_10_16_r${DATA_RATIO}"

if [ "${TEST_MODE}" = "true" ]; then
    MODEL_NAME="${BASE_MODEL_NAME}_test"
else
    MODEL_NAME="${BASE_MODEL_NAME}"
fi

WANDB_GROUP="gpt_d${DEPTH}"
DRIVE_BASE="gdrive:nanochat"
NANOCHAT_BASE_DIR="${SCRATCH:-/tmp}/ar_10_16_temp/${MODEL_NAME}"

export MODEL_NAME NANOCHAT_BASE_DIR

echo "=== ar_10_16: ${MODEL_NAME} ==="
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
echo "Starting AR next1 10/16 training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=next_token_ar \
    --target_shift=1 \
    --depth=${DEPTH} \
    --max_seq_len=512 \
    --device_batch_size=64 \
    --loss_mask_block_size=4 \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=2500 \
    --eval_num_batches=20 \
    --eval_num_batches_final=100

# ============================================================
# Upload results: scratch → Drive
# ============================================================
echo "=== Training complete. Uploading results... ==="

rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

# -- Drive --
DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${DRIVE_OUTPUT_FOLDER}/${MODEL_NAME}"
echo "Uploading to Drive: ${DRIVE_OUTPUT_DIR}"
rclone copy "${NANOCHAT_BASE_DIR}" "${DRIVE_OUTPUT_DIR}"
echo "Drive upload complete."

echo "=== Done: ${MODEL_NAME} ==="

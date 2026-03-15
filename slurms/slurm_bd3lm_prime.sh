#!/bin/bash -l
#SBATCH --job-name="bd3lm_prime"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=/rhome/sli588/temp/bd3lm_prime_%j.out

## Slurm wrapper for BD3-LM-Prime training (Prime sub-token encoding on BD3-LM backbone).
## Same tokenizer as BD3-LM (tokenizer_with_mask, 4097 vocab).
##
## Usage:
##   sbatch slurms/slurm_bd3lm_prime.sh --block_size=4 --depth=8 --data_ratio=40 --test_mode=false
##   sbatch slurms/slurm_bd3lm_prime.sh --block_size=4 --depth=8 --data_ratio=40 --target_length=2 --test_mode=false

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default parameters
# ============================================================
DEPTH="8"
BLOCK_SIZE="4"
TARGET_LENGTH="2"
TEST_MODE="true"
DATA_RATIO="10"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --depth=*)          DEPTH="${arg#*=}" ;;
        --block_size=*)     BLOCK_SIZE="${arg#*=}" ;;
        --target_length=*)  TARGET_LENGTH="${arg#*=}" ;;
        --test_mode=*)      TEST_MODE="${arg#*=}" ;;
        --data_ratio=*)     DATA_RATIO="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ============================================================
# Derived names
# ============================================================
BASE_MODEL_NAME="bd3lm_prime_d${DEPTH}_b${BLOCK_SIZE}_l${TARGET_LENGTH}"

if [ "${TEST_MODE}" = "true" ]; then
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}_test"
else
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}"
fi

WANDB_GROUP="bd3lm_prime_d${DEPTH}"
DRIVE_BASE="gdrive:nanochat"
DRIVE_OUTPUT_DIR="${DRIVE_BASE}/bd3lm_prime_d${DEPTH}/${MODEL_NAME}"
NANOCHAT_BASE_DIR="$SCRATCH/bd3lm_prime_temp_train/${MODEL_NAME}"

export MODEL_NAME DATA_RATIO DEPTH WANDB_GROUP
export NANOCHAT_BASE_DIR

echo "=== bd3lm_prime: ${MODEL_NAME} ==="
echo "=== Scratch dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== depth: ${DEPTH}, block_size: ${BLOCK_SIZE}, target_length: ${TARGET_LENGTH}, data_ratio: ${DATA_RATIO} ==="
echo "=== test_mode: ${TEST_MODE} ==="

# ============================================================
# Fetch tokenizer: Drive → scratch
# ============================================================
TOKENIZER_SRC="${DRIVE_BASE}/tokenizer/simplestory_tokenizer/4096/tokenizer_with_mask"
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"
echo "Fetching tokenizer from ${TOKENIZER_SRC} ..."
rclone copy "${TOKENIZER_SRC}/tokenizer.pkl"  "${NANOCHAT_BASE_DIR}/tokenizer/"
rclone copy "${TOKENIZER_SRC}/token_bytes.pt" "${NANOCHAT_BASE_DIR}/tokenizer/"
echo "Tokenizer ready:"
ls -lh "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate base dir and tokenizer
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"
python -c "
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
vocab_size = tok.get_vocab_size()
mask_id = tok.encode_special('<|MASK|>')
print(f'Tokenizer loaded. vocab_size={vocab_size}, mask_token_id={mask_id}')
assert mask_id is not None, 'ERROR: <|MASK|> token not found — wrong tokenizer!'
assert vocab_size == 4097, f'ERROR: expected vocab_size=4097, got {vocab_size}'
print('Tokenizer OK.')
"

python -m nanochat.report reset

# ============================================================
# Download dataset to scratch
# ============================================================
echo "Downloading dataset..."
python -m nanochat.dataset -n 10 --split both

# ============================================================
# Training
# ============================================================
echo "Starting BD3-LM-Prime training..."
python -m scripts.base_train \
    --model_type=bd3lm_prime \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --target_length=${TARGET_LENGTH} \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=512 \
    --device_batch_size=128 \
    --target_param_data_ratio=${DATA_RATIO} \
    --target_shift=-1 \
    --eval_every=2500 \
    --eval_num_batches=20 \
    --eval_num_batches_final=100 \
    --bd3lm_compute_matched=True

# ============================================================
# Upload results: scratch → Drive (before scratch auto-cleans)
# ============================================================
echo "=== Training complete. Uploading results to Drive... ==="

rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

rclone copy "${NANOCHAT_BASE_DIR}" "${DRIVE_OUTPUT_DIR}"
echo "Results saved to ${DRIVE_OUTPUT_DIR}"

echo "=== Done: ${MODEL_NAME} ==="

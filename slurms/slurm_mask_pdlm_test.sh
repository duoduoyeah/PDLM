#!/bin/bash
## Interactive test script — run inside srun session to debug the full pipeline.
## Usage: bash slurms/slurm_mask_pdlm_test.sh

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

MODEL_NAME="mask_pdlm_d8_b4_n512_k15_g120_p50_r1_test"
TOKENIZER_VARIANT="n512_k15_g120"
DEPTH="8"
BLOCK_SIZE="4"
SOFT_P_WITHIN="0.5"
DATA_RATIO="1"
WANDB_GROUP="mask_pdlm_d8"
NANOCHAT_BASE_DIR="$SCRATCH/mask_pdlm_temp_train/${MODEL_NAME}"
DRIVE_BASE="gdrive:nanochat"

export MODEL_NAME DATA_RATIO DEPTH WANDB_GROUP NANOCHAT_BASE_DIR

echo "=== mask_pdlm test: ${MODEL_NAME} ==="
echo "=== Scratch dir: ${NANOCHAT_BASE_DIR} ==="

# Fetch tokenizer
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"
echo "Fetching tokenizer..."
rclone copy "${DRIVE_BASE}/group_tokenizers/${TOKENIZER_VARIANT}/tokenizer.pkl" "${NANOCHAT_BASE_DIR}/tokenizer/"
rclone copy "${DRIVE_BASE}/group_tokenizers/${TOKENIZER_VARIANT}/token_maps.pt"  "${NANOCHAT_BASE_DIR}/tokenizer/"
echo "Tokenizer ready:"
ls -lh "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"
python -c "
from nanochat.group_tokenizer.token_map import get_token_map
tm = get_token_map()
print(f'Token map: pure_vocab={tm.pure_vocab_size}, num_groups={tm.num_groups}, overlap_k={tm.overlap_k}')
"

python -m nanochat.report reset

# Download dataset
echo "Downloading dataset..."
python -m nanochat.dataset -n 10 --split both

# Train
echo "Starting training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=mask_pdlm \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --soft_p_within=${SOFT_P_WITHIN} \
    --max_seq_len=512 \
    --device_batch_size=64 \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=2500 \
    --eval_num_batches=20 \
    --eval_num_batches_final=100

echo "=== Done ==="

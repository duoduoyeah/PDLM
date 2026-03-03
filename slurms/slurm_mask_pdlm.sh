#!/bin/bash -l
#SBATCH --job-name="mask_pdlm"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=/rhome/sli588/temp/mask_pdlm_%j.out

## Slurm wrapper for mask-PDLM training.
## Tokenizer is pulled from Google Drive via rclone into $SCRATCH.
## Training runs entirely on $SCRATCH (fast, auto-cleaned after job).
## Results are pushed back to Drive via rclone before job ends.
##
## Usage:
##   sbatch slurms/slurm_mask_pdlm.sh --noise_level=512 --overlap_k=15 --num_groups=120 \
##          --soft_p_within=0.5 --data_ratio=40 --test_mode=false
##   sbatch slurms/slurm_mask_pdlm.sh --noise_level=256 --overlap_k=31 --num_groups=496 \
##          --soft_p_within=0.7 --data_ratio=40 --test_mode=false

# ============================================================
# Environment setup
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default parameters
# ============================================================
NOISE_LEVEL="64"
OVERLAP_K="1"
NUM_GROUPS="64"
TEST_MODE="true"
DATA_RATIO="10"
DEPTH="8"
BLOCK_SIZE="4"
SOFT_P_WITHIN="1.0"
MASK_PDLM_4STATE="false"
DRIVE_OUTPUT_FOLDER="mask_pdlm_soft_sweep"   # subfolder under gdrive:nanochat

# Parse named arguments (passed after the script name to sbatch)
for arg in "$@"; do
    case $arg in
        --noise_level=*)        NOISE_LEVEL="${arg#*=}" ;;
        --overlap_k=*)          OVERLAP_K="${arg#*=}" ;;
        --num_groups=*)         NUM_GROUPS="${arg#*=}" ;;
        --test_mode=*)          TEST_MODE="${arg#*=}" ;;
        --data_ratio=*)         DATA_RATIO="${arg#*=}" ;;
        --depth=*)              DEPTH="${arg#*=}" ;;
        --block_size=*)         BLOCK_SIZE="${arg#*=}" ;;
        --soft_p_within=*)      SOFT_P_WITHIN="${arg#*=}" ;;
        --mask_pdlm_4state=*)   MASK_PDLM_4STATE="${arg#*=}" ;;
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
TOKENIZER_VARIANT="n${NOISE_LEVEL}_k${OVERLAP_K}_g${NUM_GROUPS}"
SOFT_P_INT=$(python3 -c "print(int(float('${SOFT_P_WITHIN}') * 100))")
BASE_MODEL_NAME="mask_pdlm_d${DEPTH}_b${BLOCK_SIZE}_${TOKENIZER_VARIANT}_p${SOFT_P_INT}"
if [ "${MASK_PDLM_4STATE}" = "true" ]; then
    BASE_MODEL_NAME="${BASE_MODEL_NAME}_4s"
fi

if [ "${TEST_MODE}" = "true" ]; then
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}_test"
else
    MODEL_NAME="${BASE_MODEL_NAME}_r${DATA_RATIO}"
fi

WANDB_GROUP="mask_pdlm_d${DEPTH}"
DRIVE_BASE="gdrive:nanochat"
NANOCHAT_BASE_DIR="$SCRATCH/mask_pdlm_temp_train/${MODEL_NAME}"

export MODEL_NAME DATA_RATIO DEPTH WANDB_GROUP
export NANOCHAT_BASE_DIR

echo "=== mask_pdlm: ${MODEL_NAME} ==="
echo "=== Scratch dir: ${NANOCHAT_BASE_DIR} ==="
echo "=== Tokenizer: ${TOKENIZER_VARIANT} ==="
echo "=== soft_p_within: ${SOFT_P_WITHIN} (p${SOFT_P_INT}) ==="
echo "=== 4-state: ${MASK_PDLM_4STATE} ==="
echo "=== test_mode: ${TEST_MODE}, data_ratio: ${DATA_RATIO} ==="

# ============================================================
# Fetch tokenizer: Drive → scratch
# ============================================================
mkdir -p "${NANOCHAT_BASE_DIR}/tokenizer"
echo "Fetching tokenizer from ${DRIVE_BASE}/group_tokenizers/${TOKENIZER_VARIANT} ..."
rclone copy "${DRIVE_BASE}/group_tokenizers/${TOKENIZER_VARIANT}/tokenizer.pkl" "${NANOCHAT_BASE_DIR}/tokenizer/"
rclone copy "${DRIVE_BASE}/group_tokenizers/${TOKENIZER_VARIANT}/token_maps.pt"  "${NANOCHAT_BASE_DIR}/tokenizer/"
echo "Tokenizer ready:"
ls -lh "${NANOCHAT_BASE_DIR}/tokenizer/"

# Validate tokenizer
python -c "from nanochat.common import get_base_dir; print('Base dir:', get_base_dir())"
python -c "
from nanochat.group_tokenizer.token_map import get_token_map
tm = get_token_map()
print(f'Token map: pure_vocab={tm.pure_vocab_size}, num_groups={tm.num_groups}, overlap_k={tm.overlap_k}')
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
echo "Starting Mask PDLM training..."
python -m scripts.base_train \
    --run="${MODEL_NAME}" \
    --wandb_group="${WANDB_GROUP}" \
    --model_type=pdlm \
    --pdlm_stage=mask_pdlm \
    --depth=${DEPTH} \
    --block_size=${BLOCK_SIZE} \
    --soft_p_within=${SOFT_P_WITHIN} \
    --mask_pdlm_4state=${MASK_PDLM_4STATE} \
    --max_seq_len=512 \
    --device_batch_size=64 \
    --target_param_data_ratio=${DATA_RATIO} \
    --eval_every=2500 \
    --eval_num_batches=20 \
    --eval_num_batches_final=100

# ============================================================
# Upload results: scratch → Drive (before scratch auto-cleans)
# ============================================================
echo "=== Training complete. Uploading results to Drive... ==="

# Strip dataset/tokenized data — only keep checkpoints + report
rm -rf "${NANOCHAT_BASE_DIR}/simple_story_data"
rm -rf "${NANOCHAT_BASE_DIR}/tokenized_data"

if [ -n "${DRIVE_OUTPUT_FOLDER}" ]; then
    DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${DRIVE_OUTPUT_FOLDER}/${MODEL_NAME}"
else
    DRIVE_OUTPUT_DIR="${DRIVE_BASE}/${MODEL_NAME}"
fi

rclone copy "${NANOCHAT_BASE_DIR}" "${DRIVE_OUTPUT_DIR}"
echo "Results saved to ${DRIVE_OUTPUT_DIR}"

echo "=== Done: ${MODEL_NAME} ==="

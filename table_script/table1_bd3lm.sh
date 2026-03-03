#!/bin/bash

## BD3LM-only: PPL at predicted position i+k (k=1,2,4)
##
## BD3LM model: normal mode (b4, r40) from duoduoyeah/bd3lm_d8
##
## Run: bash table_script/table1_bd3lm.sh

set -e

# ============================================================
# Environment setup (venv, credentials)
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Load secrets (HF_TOKEN etc.)
# ============================================================
if [ -f "launch/.env" ]; then
    source launch/.env
else
    echo "Warning: launch/.env not found. HF_TOKEN may not be set."
fi

# ============================================================
# Settings
# ============================================================
TOTAL_SEQ=3200
BD3LM_REPO="duoduoyeah/bd3lm_d8"
BD3LM_MODEL="bd3lm_d8_b4_normal_r40"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results/table1_bd3lm"
TMP_BASE="${SCRATCH:-/tmp}"

mkdir -p "${OUT_DIR}"

# ============================================================
# BD3LM: normal mode (reports all positions in one run)
# ============================================================
echo ""
echo "=== BD3LM normal ==="
bash launch/eval_bd3lm.sh \
    --repo="${BD3LM_REPO}" \
    --model="${BD3LM_MODEL}" \
    --total_sequences=${TOTAL_SEQ} \
    --local_dir="${TMP_BASE}/table1_bd3lm/bd3lm" \
    --push_results

cp "${TMP_BASE}/table1_bd3lm/bd3lm/${BD3LM_MODEL}/eval_result.json" \
   "${OUT_DIR}/bd3lm_normal.json"

# ============================================================
# Print table
# ============================================================
echo ""
echo "========================================================"
echo "BD3LM: PPL at predicted position i+k"
echo "  total_sequences=${TOTAL_SEQ}, seq_len=512"
echo "========================================================"

python -c "
import json

def load(path):
    with open(path) as f:
        return json.load(f)

bd3lm = load('${OUT_DIR}/bd3lm_normal.json')

print()
print(f'  BD3LM normal (all-masked, predicts position k within block):')
print(f'    k=1 : PPL = {bd3lm[\"positions\"][\"0\"][\"ppl\"]:.2f}')
print(f'    k=2 : PPL = {bd3lm[\"positions\"][\"1\"][\"ppl\"]:.2f}')
print(f'    k=4 : PPL = {bd3lm[\"positions\"][\"3\"][\"ppl\"]:.2f}')
print()
"

echo "========================================================"
echo "Results saved to: ${OUT_DIR}/"
echo "========================================================"

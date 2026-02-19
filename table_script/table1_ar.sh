#!/bin/bash

## AR: PPL at predicted position i+k (k=1,2,4)
##
## AR models: target_shift=1,2,4 from duoduoyeah/next_token_predictor
##
## Run: bash table_script/table1_ar.sh

set -e

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
GPT_REPO="duoduoyeah/next_token_predictor"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results/table1_ar"
TMP_BASE="${SCRATCH:-/tmp}"

mkdir -p "${OUT_DIR}"

# ============================================================
# GPT: target_shift=1
# ============================================================
echo ""
echo "=== GPT target_shift=1 ==="
bash launch/eval_gpt.sh \
    --repo="${GPT_REPO}" \
    --model="gpt_d8_next1_r40_v4096_implicit_simple" \
    --total_sequences=${TOTAL_SEQ} \
    --local_dir="${TMP_BASE}/table1_ar/gpt" \
    --push_results

cp "${TMP_BASE}/table1_ar/gpt/gpt_d8_next1_r40_v4096_implicit_simple/eval_result.json" \
   "${OUT_DIR}/gpt_ts1.json"

# ============================================================
# GPT: target_shift=2
# ============================================================
echo ""
echo "=== GPT target_shift=2 ==="
bash launch/eval_gpt.sh \
    --repo="${GPT_REPO}" \
    --model="gpt_d8_next2_r40_v4096_implicit_simple" \
    --total_sequences=${TOTAL_SEQ} \
    --local_dir="${TMP_BASE}/table1_ar/gpt" \
    --push_results

cp "${TMP_BASE}/table1_ar/gpt/gpt_d8_next2_r40_v4096_implicit_simple/eval_result.json" \
   "${OUT_DIR}/gpt_ts2.json"

# ============================================================
# GPT: target_shift=4
# ============================================================
echo ""
echo "=== GPT target_shift=4 ==="
bash launch/eval_gpt.sh \
    --repo="${GPT_REPO}" \
    --model="gpt_d8_next4_r40_v4096_implicit_simple" \
    --total_sequences=${TOTAL_SEQ} \
    --local_dir="${TMP_BASE}/table1_ar/gpt" \
    --push_results

cp "${TMP_BASE}/table1_ar/gpt/gpt_d8_next4_r40_v4096_implicit_simple/eval_result.json" \
   "${OUT_DIR}/gpt_ts4.json"

# ============================================================
# Print table
# ============================================================
echo ""
echo "========================================================"
echo "AR: PPL at predicted position i+k"
echo "  total_sequences=${TOTAL_SEQ}, seq_len=512"
echo "========================================================"

python -c "
import json

def load(path):
    with open(path) as f:
        return json.load(f)

gpt1 = load('${OUT_DIR}/gpt_ts1.json')
gpt2 = load('${OUT_DIR}/gpt_ts2.json')
gpt4 = load('${OUT_DIR}/gpt_ts4.json')

print()
print(f'  Autoregressive (target_shift=k, predicts i+k):')
print(f'    k=1 : PPL = {gpt1[\"overall_ppl\"]:.2f}')
print(f'    k=2 : PPL = {gpt2[\"overall_ppl\"]:.2f}')
print(f'    k=4 : PPL = {gpt4[\"overall_ppl\"]:.2f}')
print()
"

echo "========================================================"
echo "Results saved to: ${OUT_DIR}/"
echo "========================================================"

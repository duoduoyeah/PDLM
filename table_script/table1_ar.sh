#!/bin/bash

## AR vs BD3LM: PPL at predicted position i+k (k=1,2,4)
##
## AR models:   target_shift=1,2,4 from duoduoyeah/next_token_predictor
## BD3LM model: normal mode (b4, r40) from duoduoyeah/bd3lm_d8
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
BD3LM_REPO="duoduoyeah/bd3lm_d8"
BD3LM_MODEL="bd3lm_d8_b4_normal_r40"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results/table1_ar"

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
    --local_dir="/tmp/table1_ar/gpt"

cp "/tmp/table1_ar/gpt/gpt_d8_next1_r40_v4096_implicit_simple/eval_result.json" \
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
    --local_dir="/tmp/table1_ar/gpt"

cp "/tmp/table1_ar/gpt/gpt_d8_next2_r40_v4096_implicit_simple/eval_result.json" \
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
    --local_dir="/tmp/table1_ar/gpt"

cp "/tmp/table1_ar/gpt/gpt_d8_next4_r40_v4096_implicit_simple/eval_result.json" \
   "${OUT_DIR}/gpt_ts4.json"

# ============================================================
# BD3LM: normal mode (reports all positions in one run)
# ============================================================
echo ""
echo "=== BD3LM normal ==="
bash launch/eval_bd3lm.sh \
    --repo="${BD3LM_REPO}" \
    --variant="normal" \
    --data_ratio=40 \
    --total_sequences=${TOTAL_SEQ} \
    --local_dir="/tmp/table1_ar/bd3lm"

cp "/tmp/table1_ar/bd3lm/${BD3LM_MODEL}/eval_result.json" \
   "${OUT_DIR}/bd3lm_normal.json"

# ============================================================
# Print table
# ============================================================
echo ""
echo "========================================================"
echo "AR vs BD3LM: PPL at predicted position i+k"
echo "  total_sequences=${TOTAL_SEQ}, seq_len=512"
echo "========================================================"

python -c "
import json

def load(path):
    with open(path) as f:
        return json.load(f)

gpt1  = load('${OUT_DIR}/gpt_ts1.json')
gpt2  = load('${OUT_DIR}/gpt_ts2.json')
gpt4  = load('${OUT_DIR}/gpt_ts4.json')
bd3lm = load('${OUT_DIR}/bd3lm_normal.json')

print()
print(f'  Autoregressive (target_shift=k, predicts i+k):')
print(f'    k=1 : PPL = {gpt1[\"overall_ppl\"]:.2f}')
print(f'    k=2 : PPL = {gpt2[\"overall_ppl\"]:.2f}')
print(f'    k=4 : PPL = {gpt4[\"overall_ppl\"]:.2f}')
print()
print(f'  BD3LM normal (all-masked, predicts position k within block):')
print(f'    k=1 : PPL = {bd3lm[\"positions\"][0][\"ppl\"]:.2f}')
print(f'    k=2 : PPL = {bd3lm[\"positions\"][1][\"ppl\"]:.2f}')
print(f'    k=4 : PPL = {bd3lm[\"positions\"][3][\"ppl\"]:.2f}')
print()
"

echo "========================================================"
echo "Results saved to: ${OUT_DIR}/"
echo "========================================================"

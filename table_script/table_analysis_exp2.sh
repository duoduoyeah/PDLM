#!/bin/bash

## Analysis Experiment 2: BD3-LM suffix sweep.
##
## Fixes a rich clean prefix (all previous blocks), moves target position
## left-to-right within the current block. Suffix mask count decreases
## 3→0 while prefix positions within the block are revealed (clean).
##
## Block configs:
##   T,M,M,M  pos=0, 0 clean prefix, 3 suffix masks
##   P,T,M,M  pos=1, 1 clean prefix, 2 suffix masks
##   P,P,T,M  pos=2, 2 clean prefix, 1 suffix mask
##   P,P,P,T  pos=3, 3 clean prefix, 0 suffix masks
##
## Model: bd3lm_d8_b4_normal_r40 (HF: duoduoyeah/bd3lm_d8)
## Reuses model download from Exp 1 if available, else re-downloads.
##
## Run: bash table_script/table_analysis_exp2.sh
## Or:  SKIP_EXISTING=true bash table_script/table_analysis_exp2.sh

set -e

# ============================================================
# Environment setup (venv, credentials)
# ============================================================
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Load secrets (HF_TOKEN etc.)
# ============================================================
if [ -f "${REPO_ROOT}/launch/.env" ]; then
    source "${REPO_ROOT}/launch/.env"
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
OUT_DIR="${SCRIPT_DIR}/results/table_analysis_exp2"
TMP_BASE="${SCRATCH:-/tmp}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

# Reuse Exp 1 download if present, otherwise download fresh
BD3LM_LOCAL="${TMP_BASE}/analysis_exp1/bd3lm/${BD3LM_MODEL}"
if [ ! -d "${BD3LM_LOCAL}/base_checkpoints" ]; then
    echo "BD3-LM model not found at ${BD3LM_LOCAL}, downloading..."
    mkdir -p "${BD3LM_LOCAL}"
    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${BD3LM_REPO}',
    local_dir='${TMP_BASE}/analysis_exp1/bd3lm',
    repo_type='model',
    allow_patterns='${BD3LM_MODEL}/**',
    token=os.environ.get('HF_TOKEN'),
)
print('Download complete.')
"
fi

# Ensure data symlink exists for NANOCHAT_BASE_DIR
DATA_DIR="${HOME}/.cache/nanochat/simple_story_data"
if [ ! -d "${DATA_DIR}" ]; then
    python -m nanochat.dataset --split=val
fi
if [ ! -e "${BD3LM_LOCAL}/simple_story_data" ]; then
    ln -s "${DATA_DIR}" "${BD3LM_LOCAL}/simple_story_data"
fi
export NANOCHAT_BASE_DIR="${BD3LM_LOCAL}"

mkdir -p "${OUT_DIR}"

# ============================================================
# BD3-LM left-to-right eval
# ============================================================
BD3LM_LTR_JSON="${OUT_DIR}/bd3lm_ltr.json"
if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${BD3LM_LTR_JSON}" ]; then
    echo "Skipping BD3-LM left-to-right eval (cached)"
else
    echo ""
    echo "=== BD3-LM left-to-right eval ==="
    CKPT_DIR=$(find "${BD3LM_LOCAL}/base_checkpoints" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | head -1)
    python -m scripts.bd3lm_eval \
        --ckpt_dir="${CKPT_DIR}" \
        --total_sequences=${TOTAL_SEQ} \
        --left_to_right \
        --output_json="${BD3LM_LTR_JSON}"
fi

# ============================================================
# Print table (Analysis.tex tab:suffix_signal)
# ============================================================
echo ""
echo "========================================================"
echo "Analysis Experiment 2: BD3-LM suffix sweep"
echo "  total_sequences=${TOTAL_SEQ}"
echo "========================================================"

python3 - "${BD3LM_LTR_JSON}" << 'PYEOF'
import sys, json, math

with open(sys.argv[1]) as f:
    data = json.load(f)

ltr = data["left_to_right"]
positions = ltr["positions"]

configs = [
    ("T,M,M,M", 0),
    ("P,T,M,M", 1),
    ("P,P,T,M", 2),
    ("P,P,P,T", 3),
]

W_CFG, W_PPL, W_ARG, W_ENT = 12, 8, 13, 8
hdr = (f"{'Config':<{W_CFG}}  {'PPL':>{W_PPL}}"
       f"  {'Argmax prob':>{W_ARG}}  {'Ent-PPL':>{W_ENT}}")
sep = "-" * len(hdr)

print()
print(hdr)
print(sep)

for cfg, k in configs:
    pos_data = positions.get(k, positions.get(str(k), {}))
    ppl      = pos_data.get("ppl",         float("nan"))
    arg      = pos_data.get("argmax_prob",  float("nan"))
    ent      = pos_data.get("entropy_ppl",  float("nan"))
    print(f"{cfg:<{W_CFG}}  {ppl:{W_PPL}.2f}  {arg:{W_ARG}.4f}  {ent:{W_ENT}.2f}")

print(sep)
print()
PYEOF

echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

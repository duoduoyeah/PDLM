#!/bin/bash

## Analysis Experiment 1: AR vs BD3-LM paired comparison.
##
## Compares AR(shift=k) vs BD3-LM(pos=k, all non-target positions masked)
## at each block position k=0..3. Reports PPL, Argmax prob, and Ent-PPL
## for both models at each position (table tab:mask_signal in Analysis.tex).
##
## AR models:
##   ts=1,2,4 — downloaded from HF (duoduoyeah/next_token_predictor)
##   ts=3     — downloaded from Drive (gdrive:nanochat/next_token_ar/gpt_d8_next3_r40)
##
## BD3-LM model: bd3lm_d8_b4_normal_r40 (normal mode, all positions masked)
##              downloaded from HF (duoduoyeah/bd3lm_d8)
##
## Run: bash table_script/table_analysis_exp1.sh
## Or:  SKIP_EXISTING=true bash table_script/table_analysis_exp1.sh

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
OUT_DIR="${SCRIPT_DIR}/results/table_analysis_exp1"
TMP_BASE="${SCRATCH:-/tmp}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

mkdir -p "${OUT_DIR}"

# ============================================================
# AR ts=1 (HuggingFace)
# ============================================================
AR_TS1_JSON="${OUT_DIR}/ar_ts1.json"
AR_TS1_MODEL="gpt_d8_next1_r40_v4096_implicit_simple"
if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${AR_TS1_JSON}" ]; then
    echo "Skipping AR ts=1 (cached)"
else
    echo ""
    echo "=== AR ts=1 ==="
    bash launch/eval_gpt.sh \
        --repo="${GPT_REPO}" \
        --model="${AR_TS1_MODEL}" \
        --total_sequences=${TOTAL_SEQ} \
        --local_dir="${TMP_BASE}/analysis_exp1/ar"
    cp "${TMP_BASE}/analysis_exp1/ar/${AR_TS1_MODEL}/eval_result.json" "${AR_TS1_JSON}"
fi

# ============================================================
# AR ts=2 (HuggingFace)
# ============================================================
AR_TS2_JSON="${OUT_DIR}/ar_ts2.json"
AR_TS2_MODEL="gpt_d8_next2_r40_v4096_implicit_simple"
if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${AR_TS2_JSON}" ]; then
    echo "Skipping AR ts=2 (cached)"
else
    echo ""
    echo "=== AR ts=2 ==="
    bash launch/eval_gpt.sh \
        --repo="${GPT_REPO}" \
        --model="${AR_TS2_MODEL}" \
        --total_sequences=${TOTAL_SEQ} \
        --local_dir="${TMP_BASE}/analysis_exp1/ar"
    cp "${TMP_BASE}/analysis_exp1/ar/${AR_TS2_MODEL}/eval_result.json" "${AR_TS2_JSON}"
fi

# ============================================================
# AR ts=3 (Google Drive)
# ============================================================
AR_TS3_JSON="${OUT_DIR}/ar_ts3.json"
AR_TS3_MODEL="gpt_d8_next3_r40"
AR_TS3_LOCAL="${TMP_BASE}/analysis_exp1/ar/${AR_TS3_MODEL}"
if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${AR_TS3_JSON}" ]; then
    echo "Skipping AR ts=3 (cached)"
else
    echo ""
    echo "=== AR ts=3 (from Drive) ==="
    mkdir -p "${AR_TS3_LOCAL}"
    rclone copy "gdrive:nanochat/next_token_ar/${AR_TS3_MODEL}" "${AR_TS3_LOCAL}" \
        --include "base_checkpoints/**" \
        --include "tokenizer/**" \
        --progress
    bash launch/eval_gpt.sh \
        --ckpt_path="${AR_TS3_LOCAL}" \
        --total_sequences=${TOTAL_SEQ} \
        --local_dir="${AR_TS3_LOCAL}"
    cp "${AR_TS3_LOCAL}/eval_result.json" "${AR_TS3_JSON}"
fi

# ============================================================
# AR ts=4 (HuggingFace)
# ============================================================
AR_TS4_JSON="${OUT_DIR}/ar_ts4.json"
AR_TS4_MODEL="gpt_d8_next4_r40_v4096_implicit_simple"
if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${AR_TS4_JSON}" ]; then
    echo "Skipping AR ts=4 (cached)"
else
    echo ""
    echo "=== AR ts=4 ==="
    bash launch/eval_gpt.sh \
        --repo="${GPT_REPO}" \
        --model="${AR_TS4_MODEL}" \
        --total_sequences=${TOTAL_SEQ} \
        --local_dir="${TMP_BASE}/analysis_exp1/ar"
    cp "${TMP_BASE}/analysis_exp1/ar/${AR_TS4_MODEL}/eval_result.json" "${AR_TS4_JSON}"
fi

# ============================================================
# BD3-LM normal mode (HuggingFace)
# ============================================================
BD3LM_JSON="${OUT_DIR}/bd3lm_normal.json"
if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${BD3LM_JSON}" ]; then
    echo "Skipping BD3-LM (cached)"
else
    echo ""
    echo "=== BD3-LM normal mode ==="
    bash launch/eval_bd3lm.sh \
        --repo="${BD3LM_REPO}" \
        --model="${BD3LM_MODEL}" \
        --total_sequences=${TOTAL_SEQ} \
        --local_dir="${TMP_BASE}/analysis_exp1/bd3lm"
    cp "${TMP_BASE}/analysis_exp1/bd3lm/${BD3LM_MODEL}/eval_result.json" "${BD3LM_JSON}"
fi

# ============================================================
# Print table (Analysis.tex tab:mask_signal)
# ============================================================
echo ""
echo "========================================================"
echo "Analysis Experiment 1: AR vs BD3-LM (mask tokens carry no signal)"
echo "  total_sequences=${TOTAL_SEQ}"
echo "========================================================"

python3 - "${AR_TS1_JSON}" "${AR_TS2_JSON}" "${AR_TS3_JSON}" "${AR_TS4_JSON}" "${BD3LM_JSON}" << 'PYEOF'
import sys, json, math

def load(path):
    with open(path) as f:
        return json.load(f)

ar1, ar2, ar3, ar4, bd = [load(p) for p in sys.argv[1:]]

configs = [
    ("T,M,M,M", "AR shift=1",   ar1, "BD3-LM pos=0", bd["positions"]["0"]),
    ("M,T,M,M", "AR shift=2",   ar2, "BD3-LM pos=1", bd["positions"]["1"]),
    ("M,M,T,M", "AR shift=3",   ar3, "BD3-LM pos=2", bd["positions"]["2"]),
    ("M,M,M,T", "AR shift=4",   ar4, "BD3-LM pos=3", bd["positions"]["3"]),
]

W_CFG, W_MDL, W_PPL, W_ARG, W_ENT = 12, 16, 8, 13, 8
hdr = (f"{'Config':<{W_CFG}}  {'Model':<{W_MDL}}  {'PPL':>{W_PPL}}"
       f"  {'Argmax prob':>{W_ARG}}  {'Ent-PPL':>{W_ENT}}")
sep = "-" * len(hdr)

print()
print(hdr)
print(sep)

for cfg, ar_lbl, ar, bd_lbl, bd_pos in configs:
    ar_ppl  = ar.get("overall_ppl", float("nan"))
    ar_arg  = ar.get("overall_argmax_prob", float("nan"))
    ar_ent  = ar.get("overall_entropy_ppl", float("nan"))
    bd_ppl  = bd_pos.get("ppl", float("nan"))
    bd_arg  = bd_pos.get("argmax_prob", float("nan"))
    bd_ent  = bd_pos.get("entropy_ppl", float("nan"))

    print(f"{cfg:<{W_CFG}}  {ar_lbl:<{W_MDL}}  {ar_ppl:{W_PPL}.2f}"
          f"  {ar_arg:{W_ARG}.4f}  {ar_ent:{W_ENT}.2f}")
    print(f"{'':>{W_CFG}}  {bd_lbl:<{W_MDL}}  {bd_ppl:{W_PPL}.2f}"
          f"  {bd_arg:{W_ARG}.4f}  {bd_ent:{W_ENT}.2f}")
    print(sep)

print()
PYEOF

echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

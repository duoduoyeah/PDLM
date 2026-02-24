#!/bin/bash -l
#SBATCH --job-name="mask_pdlm_sweep"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=/rhome/sli588/temp/mask_pdlm_sweep_%j.out

## Evaluate all mask_pdlm sweep models and print comparison tables.
##
## Runs two sweeps from gdrive:nanochat/mask_pdlm_soft_sweep/:
##   1. soft_p sweep: b=4, both tokenizers (k31/g496 and k15/g120), p=0..100
##   2. block_size sweep: p=50, k15/g120, b=1,2,4,8,16
##      (b4/p50/k15 reuses results from sweep 1 — evaluated once)
##
## Results (JSON per model) saved to table_script/results/table_mask_pdlm_sweep/
## and pushed to duoduoyeah/eval_results.
##
## Usage:
##   sbatch slurms/table_mask_pdlm_sweep.sh
##   SKIP_EXISTING=true sbatch slurms/table_mask_pdlm_sweep.sh   # resume interrupted run
##   PRINT_ONLY=true  sbatch slurms/table_mask_pdlm_sweep.sh     # reprint table from cached JSONs

set -e

# ============================================================
# Environment setup (uv/venv/credentials via scratch venv)
# ============================================================
source ~/pdlm_mask/slurms/setup.sh
# setup.sh does: cd ~/pdlm_mask, uv sync --extra gpu, activates scratch venv,
#                exports HF_TOKEN, WANDB_API_KEY, WANDB_DIR

# ============================================================
# Settings
# ============================================================
TOTAL_SEQ=3200
GDRIVE_FOLDER="mask_pdlm_soft_sweep"
OUT_DIR="table_script/results/table_mask_pdlm_sweep"
LOCAL_DIR="${SCRATCH}/mask_pdlm_sweep_eval"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
PRINT_ONLY="${PRINT_ONLY:-false}"

mkdir -p "${OUT_DIR}"

# ============================================================
# Model lists
# ============================================================

# soft_p sweep: k31/g496 tokenizer, block_size=4
P_SWEEP_K31=(
    mask_pdlm_d8_b4_n256_k31_g496_p0_r40
    mask_pdlm_d8_b4_n256_k31_g496_p30_r40
    mask_pdlm_d8_b4_n256_k31_g496_p50_r40
    mask_pdlm_d8_b4_n256_k31_g496_p70_r40
    mask_pdlm_d8_b4_n256_k31_g496_p90_r40
    mask_pdlm_d8_b4_n256_k31_g496_p100_r40
)

# soft_p sweep: k15/g120 tokenizer, block_size=4
P_SWEEP_K15=(
    mask_pdlm_d8_b4_n512_k15_g120_p0_r40
    mask_pdlm_d8_b4_n512_k15_g120_p30_r40
    mask_pdlm_d8_b4_n512_k15_g120_p50_r40
    mask_pdlm_d8_b4_n512_k15_g120_p70_r40
    mask_pdlm_d8_b4_n512_k15_g120_p90_r40
    mask_pdlm_d8_b4_n512_k15_g120_p100_r40
)

# block_size sweep: k15/g120, p=50 (b4 reuses P_SWEEP_K15 result — not listed here)
B_SWEEP_EXTRA=(
    mask_pdlm_d8_b1_n512_k15_g120_p50_r40
    mask_pdlm_d8_b2_n512_k15_g120_p50_r40
    mask_pdlm_d8_b8_n512_k15_g120_p50_r40
    mask_pdlm_d8_b16_n512_k15_g120_p50_r40
)

# 16 unique models total (b4/p50/k15 counted once in P_SWEEP_K15)
ALL_MODELS=(
    "${P_SWEEP_K31[@]}"
    "${P_SWEEP_K15[@]}"
    "${B_SWEEP_EXTRA[@]}"
)

# ============================================================
# Run evals
# ============================================================
if [ "${PRINT_ONLY}" != "true" ]; then
    echo ""
    echo "========================================================"
    echo "Mask PDLM Sweep Evaluation"
    echo "  total_sequences=${TOTAL_SEQ}"
    echo "  gdrive_folder=${GDRIVE_FOLDER}"
    echo "  models: ${#ALL_MODELS[@]} unique (16 total, b4/p50/k15 shared across sweeps)"
    echo "========================================================"

    for MODEL in "${ALL_MODELS[@]}"; do
        RESULT_LOCAL="${LOCAL_DIR}/${MODEL}/eval_result.json"
        RESULT_OUT="${OUT_DIR}/${MODEL}.json"

        echo ""
        echo "=== ${MODEL} ==="

        if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${RESULT_OUT}" ]; then
            echo "  Skipping (already exists: ${RESULT_OUT})"
            continue
        fi

        bash slurms/eval_mask_pdlm_gdrive.sh \
            --gdrive_folder="${GDRIVE_FOLDER}" \
            --model="${MODEL}" \
            --total_sequences=${TOTAL_SEQ} \
            --local_dir="${LOCAL_DIR}" \
            --push_results

        cp "${RESULT_LOCAL}" "${RESULT_OUT}"
        echo "  Saved: ${RESULT_OUT}"
    done
fi

# ============================================================
# Print tables
# ============================================================
echo ""
echo "========================================================"
echo "Mask PDLM Sweep Results  (total_sequences=${TOTAL_SEQ})"
echo "========================================================"

python3 - "${OUT_DIR}" <<'PYEOF'
import json, os, sys

OUT_DIR = sys.argv[1]

def load(model):
    path = os.path.join(OUT_DIR, f"{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fmt_ppl(v):
    return f"{v:.2f}" if v is not None else "N/A"

def fmt_acc(v):
    return f"{v:.1%}" if v is not None else "N/A"

def get_metrics(d):
    if d is None:
        return dict(uppl=None, ueppl=None, e2e_ppl=None, e2e_eppl=None, e2e_acc=None)
    u = d.get("unified", {})
    e = d.get("end2end", {})
    return dict(
        uppl     = u.get("overall_ppl"),
        ueppl    = u.get("overall_entropy_ppl"),
        e2e_ppl  = e.get("overall_ppl"),
        e2e_eppl = e.get("overall_entropy_ppl"),
        e2e_acc  = e.get("overall_accuracy"),
    )

def get_pos_list(d, section):
    if d is None:
        return []
    positions = d.get(section, {}).get("positions", {})
    n = len(positions)
    return [positions.get(i, positions.get(str(i), {})) for i in range(n)]

def print_sweep_table(title, col_labels, models):
    data    = [load(m) for m in models]
    metrics = [get_metrics(d) for d in data]

    CW = 11
    print()
    print(title)
    print("─" * (22 + CW * len(col_labels)))
    row = f"{'metric':<22}" + "".join(f"{str(c):>{CW}}" for c in col_labels)
    print(row)
    print("─" * len(row))

    rows = [
        ("unified_ppl",     [fmt_ppl(m["uppl"])     for m in metrics]),
        ("unified_ent_ppl", [fmt_ppl(m["ueppl"])    for m in metrics]),
        ("end2end_ppl",     [fmt_ppl(m["e2e_ppl"])  for m in metrics]),
        ("end2end_ent_ppl", [fmt_ppl(m["e2e_eppl"]) for m in metrics]),
        ("end2end_acc",     [fmt_acc(m["e2e_acc"])  for m in metrics]),
    ]
    for label, vals in rows:
        print(f"{label:<22}" + "".join(f"{v:>{CW}}" for v in vals))

    # Per-position end2end breakdown
    sample = next((d for d in data if d is not None), None)
    if sample:
        n_pos = len(sample.get("end2end", {}).get("positions", {}))
        if n_pos > 0:
            print()
            print("  end2end per-position breakdown:")
            CW2 = 16
            hdr2 = f"  {'pos':<6}" + "".join(f"{str(c):>{CW2}}" for c in col_labels)
            print(hdr2)
            print("  " + "─" * (6 + CW2 * len(col_labels)))
            for pos in range(n_pos):
                vals = []
                for d in data:
                    pm = get_pos_list(d, "end2end")
                    if pos < len(pm):
                        ppl = pm[pos].get("ppl")
                        acc = pm[pos].get("accuracy")
                        vals.append(f"{fmt_ppl(ppl)} / {fmt_acc(acc)}")
                    else:
                        vals.append("—")
                print(f"  {pos:<6}" + "".join(f"{v:>{CW2}}" for v in vals))

# ── Sweep 1a: soft_p, k31/g496 ────────────────────────────────────────────
P_VALUES = [0, 30, 50, 70, 90, 100]
print_sweep_table(
    title      = "soft_p_within sweep — b=4, k31/g496 (n=256, k=31, g=496)",
    col_labels = [f"p={p}" for p in P_VALUES],
    models     = [f"mask_pdlm_d8_b4_n256_k31_g496_p{p}_r40" for p in P_VALUES],
)

# ── Sweep 1b: soft_p, k15/g120 ────────────────────────────────────────────
print_sweep_table(
    title      = "soft_p_within sweep — b=4, k15/g120 (n=512, k=15, g=120)",
    col_labels = [f"p={p}" for p in P_VALUES],
    models     = [f"mask_pdlm_d8_b4_n512_k15_g120_p{p}_r40" for p in P_VALUES],
)

# ── Sweep 2: block_size ────────────────────────────────────────────────────
B_VALUES = [1, 2, 4, 8, 16]
print_sweep_table(
    title      = "block_size sweep — p=50, k15/g120 (n=512, k=15, g=120)",
    col_labels = [f"b={b}" for b in B_VALUES],
    models     = [f"mask_pdlm_d8_b{b}_n512_k15_g120_p50_r40" for b in B_VALUES],
)

print()
PYEOF

echo "========================================================"
echo "JSON results saved to: ${OUT_DIR}/"
echo "========================================================"

"""
Analyze Table 4: BD3-LM (left-to-right) vs PDLM p=0.

Validates the claim that "hard group token ≈ mask token" by comparing:
  - BD3-LM evaluated in left-to-right teacher-forced mode
  - PDLM p=0 evaluated in end2end iterative inference mode

Both use block_size=4, both skip block 0, both compute per-position ppl/accuracy.

Loads:
  - BD3-LM JSONs from --bd3lm_dir (table4_bd3lm/)
  - PDLM p=0 JSONs from --pdlm_dir (table_mask_pdlm_sweep/)

Prints comparison table and writes summary.json + summary.csv.

Usage:
    python3 table_script/analyze_table4.py
    python3 table_script/analyze_table4.py \\
        --bd3lm_dir=table_script/results/table4_bd3lm \\
        --pdlm_dir=table_script/results/table_mask_pdlm_sweep
"""

import argparse
import csv
import json
import os

# ── Model definitions ─────────────────────────────────────────────────────────

BD3LM_MODELS = [
    "bd3lm_d8_b2_normal_r40",
    "bd3lm_d8_b4_normal_r40",
    "bd3lm_d8_b8_normal_r40",
    "bd3lm_d8_b16_normal_r40",
]

# PDLM p=0 counterparts for b=4 (two tokenizers)
PDLM_P0_K15 = "mask_pdlm_d8_b4_n512_k15_g120_p0_r40"
PDLM_P0_K31 = "mask_pdlm_d8_b4_n256_k31_g496_p0_r40"

# Primary comparison: BD3LM-b4 vs PDLM-p0 (both tokenizers)
PRIMARY_BD3LM = "bd3lm_d8_b4_normal_r40"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(directory, model):
    path = os.path.join(directory, f"{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract_bd3lm_metrics(data):
    """Extract left_to_right metrics from BD3LM JSON."""
    if data is None:
        return {}
    ltr = data.get("left_to_right", {})
    m = {
        "ppl":      ltr.get("overall_ppl"),
        "accuracy": ltr.get("overall_accuracy"),
        "loss":     ltr.get("overall_loss"),
    }
    positions = ltr.get("positions", {})
    for i in range(len(positions)):
        pm = positions.get(i, positions.get(str(i), {}))
        m[f"pos{i}_ppl"]      = pm.get("ppl")
        m[f"pos{i}_accuracy"] = pm.get("accuracy")
        m[f"pos{i}_loss"]     = pm.get("loss")
    return m


def extract_pdlm_metrics(data):
    """Extract end2end metrics from PDLM p=0 JSON."""
    if data is None:
        return {}
    e2e = data.get("end2end", {})
    m = {
        "ppl":      e2e.get("overall_ppl"),
        "accuracy": e2e.get("overall_accuracy"),
        "loss":     e2e.get("overall_loss"),
    }
    positions = e2e.get("positions", {})
    for i in range(len(positions)):
        pm = positions.get(i, positions.get(str(i), {}))
        m[f"pos{i}_ppl"]      = pm.get("ppl")
        m[f"pos{i}_accuracy"] = pm.get("accuracy")
        m[f"pos{i}_loss"]     = pm.get("loss")
    return m


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_ppl(v, w=8):
    return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"

def fmt_acc(v, w=8):
    return f"{v:>{w}.1%}" if v is not None else f"{'N/A':>{w}}"

def fmt_ppl_acc(ppl, acc):
    ppl_s = f"{ppl:.2f}" if ppl is not None else "N/A"
    acc_s = f"{acc:.1%}" if acc is not None else "N/A"
    return f"{ppl_s} / {acc_s}"


# ── Table printers ────────────────────────────────────────────────────────────

def print_primary_table(bd3lm_data, pdlm_k15_data, pdlm_k31_data, block_size=4):
    """
    Print primary comparison: BD3LM-b4 vs PDLM-p0 (two tokenizers).

    Table format:
      metric                 BD3LM-b4     PDLM-p0-k15  PDLM-p0-k31
      overall_ppl              X.XX         X.XX         X.XX
      overall_accuracy        XX.X%        XX.X%        XX.X%
      per-position (ppl / accuracy):
        k     BD3LM-b4       PDLM-p0-k15    PDLM-p0-k31
        0     X.XX / XX.X%   ...
        ...
    """
    bd3lm_m  = extract_bd3lm_metrics(bd3lm_data)
    pdlm_k15 = extract_pdlm_metrics(pdlm_k15_data)
    pdlm_k31 = extract_pdlm_metrics(pdlm_k31_data)

    col_labels = ["BD3LM-b4", "PDLM-p0-k15", "PDLM-p0-k31"]
    CW = 16
    sep = "─" * (24 + CW * len(col_labels))

    print()
    print("Table 4: BD3-LM (left-to-right) vs PDLM p=0")
    print(sep)
    header = f"{'metric':<24}" + "".join(f"{c:>{CW}}" for c in col_labels)
    print(header)
    print(sep)

    def row(label, vals):
        print(f"{label:<24}" + "".join(f"{v:>{CW}}" for v in vals))

    row("overall_ppl", [
        fmt_ppl(bd3lm_m.get("ppl"),      10),
        fmt_ppl(pdlm_k15.get("ppl"),     10),
        fmt_ppl(pdlm_k31.get("ppl"),     10),
    ])
    row("overall_accuracy", [
        fmt_acc(bd3lm_m.get("accuracy"),  10),
        fmt_acc(pdlm_k15.get("accuracy"), 10),
        fmt_acc(pdlm_k31.get("accuracy"), 10),
    ])

    print()
    print("  per-position (ppl / accuracy):")
    pos_header = f"  {'k':<6}" + "".join(f"{c:>{CW + 4}}" for c in col_labels)
    print(pos_header)
    print("  " + "─" * (6 + (CW + 4) * len(col_labels)))

    for k in range(block_size):
        vals = [
            fmt_ppl_acc(bd3lm_m.get(f"pos{k}_ppl"),  bd3lm_m.get(f"pos{k}_accuracy")),
            fmt_ppl_acc(pdlm_k15.get(f"pos{k}_ppl"), pdlm_k15.get(f"pos{k}_accuracy")),
            fmt_ppl_acc(pdlm_k31.get(f"pos{k}_ppl"), pdlm_k31.get(f"pos{k}_accuracy")),
        ]
        print(f"  {k:<6}" + "".join(f"{v:>{CW + 4}}" for v in vals))

    print(sep)


def print_block_size_table(all_bd3lm, block_sizes=(2, 4, 8, 16)):
    """Print BD3LM block_size sweep (left-to-right metrics)."""
    col_labels = [f"b={b}" for b in block_sizes]
    CW = 14
    sep = "─" * (20 + CW * len(col_labels))

    print()
    print("BD3-LM block_size sweep (left-to-right)")
    print(sep)
    header = f"{'metric':<20}" + "".join(f"{c:>{CW}}" for c in col_labels)
    print(header)
    print(sep)

    def row(label, vals):
        print(f"{label:<20}" + "".join(f"{v:>{CW}}" for v in vals))

    metrics = [extract_bd3lm_metrics(all_bd3lm.get(f"bd3lm_d8_b{b}_normal_r40")) for b in block_sizes]

    row("overall_ppl",      [fmt_ppl(m.get("ppl"),      9) for m in metrics])
    row("overall_accuracy", [fmt_acc(m.get("accuracy"), 9) for m in metrics])

    # Per-position rows (use max block_size for position count)
    max_pos = max(block_sizes)
    print()
    print("  per-position ppl (left-to-right):")
    print(f"  {'k':<6}" + "".join(f"{c:>{CW}}" for c in col_labels))
    print("  " + "─" * (6 + CW * len(col_labels)))
    for k in range(max(block_sizes)):
        vals = []
        for i, b in enumerate(block_sizes):
            if k < b:
                vals.append(fmt_ppl(metrics[i].get(f"pos{k}_ppl"), 9))
            else:
                vals.append(f"{'—':>9}")
        print(f"  {k:<6}" + "".join(f"{v:>{CW}}" for v in vals))

    print(sep)


# ── Summary files ─────────────────────────────────────────────────────────────

def write_summary_json(out_dir, all_bd3lm, pdlm_k15_data, pdlm_k31_data):
    summary = {}
    for model in BD3LM_MODELS:
        summary[model] = {"bd3lm": extract_bd3lm_metrics(all_bd3lm.get(model))}
    summary[PDLM_P0_K15] = {"pdlm_p0": extract_pdlm_metrics(pdlm_k15_data)}
    summary[PDLM_P0_K31] = {"pdlm_p0": extract_pdlm_metrics(pdlm_k31_data)}

    out = os.path.join(out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written: {out}")
    return out


def write_summary_csv(out_dir, all_bd3lm, pdlm_k15_data, pdlm_k31_data):
    rows = []
    fieldnames = ["model", "type"]

    # Gather all metric keys
    all_keys = set()
    for model in BD3LM_MODELS:
        m = extract_bd3lm_metrics(all_bd3lm.get(model))
        all_keys.update(m.keys())
    for data in [pdlm_k15_data, pdlm_k31_data]:
        m = extract_pdlm_metrics(data)
        all_keys.update(m.keys())
    metric_keys = sorted(all_keys)
    fieldnames.extend(metric_keys)

    for model in BD3LM_MODELS:
        m = extract_bd3lm_metrics(all_bd3lm.get(model))
        rows.append({"model": model, "type": "bd3lm_ltr", **m})

    m = extract_pdlm_metrics(pdlm_k15_data)
    rows.append({"model": PDLM_P0_K15, "type": "pdlm_p0", **m})

    m = extract_pdlm_metrics(pdlm_k31_data)
    rows.append({"model": PDLM_P0_K31, "type": "pdlm_p0", **m})

    out = os.path.join(out_dir, "summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Written: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bd3lm_dir",
        default="table_script/results/table4_bd3lm",
        help="Directory with BD3LM left-to-right JSON results",
    )
    parser.add_argument(
        "--pdlm_dir",
        default="table_script/results/table_mask_pdlm_sweep",
        help="Directory with PDLM sweep JSON results (for p=0 models)",
    )
    args = parser.parse_args()

    bd3lm_dir = args.bd3lm_dir
    pdlm_dir  = args.pdlm_dir

    # Load BD3LM results
    all_bd3lm = {m: load_json(bd3lm_dir, m) for m in BD3LM_MODELS}

    # Load PDLM p=0 results
    pdlm_k15_data = load_json(pdlm_dir, PDLM_P0_K15)
    pdlm_k31_data = load_json(pdlm_dir, PDLM_P0_K31)

    # Report missing
    missing_bd3lm = [m for m, d in all_bd3lm.items() if d is None]
    if missing_bd3lm:
        print(f"Warning: {len(missing_bd3lm)} BD3LM result(s) missing:")
        for m in missing_bd3lm:
            print(f"  {m}")
    if pdlm_k15_data is None:
        print(f"Warning: PDLM p=0 k15 result missing: {PDLM_P0_K15}")
    if pdlm_k31_data is None:
        print(f"Warning: PDLM p=0 k31 result missing: {PDLM_P0_K31}")

    print("\n" + "=" * 80)
    print("Table 4 Analysis: BD3-LM (left-to-right) vs PDLM p=0")
    print("  BD3-LM eval: left-to-right teacher-forced (prefix 0..k-1 revealed per step)")
    print("  PDLM p=0:    end2end iterative inference (hard group token = mask token)")
    print("=" * 80)

    # Primary comparison: b=4
    print_primary_table(
        bd3lm_data     = all_bd3lm.get(PRIMARY_BD3LM),
        pdlm_k15_data  = pdlm_k15_data,
        pdlm_k31_data  = pdlm_k31_data,
        block_size     = 4,
    )

    # BD3LM block_size sweep
    print_block_size_table(all_bd3lm, block_sizes=(2, 4, 8, 16))

    print("\n" + "=" * 80)
    print("Writing summary files...")
    os.makedirs(bd3lm_dir, exist_ok=True)
    write_summary_json(bd3lm_dir, all_bd3lm, pdlm_k15_data, pdlm_k31_data)
    write_summary_csv(bd3lm_dir, all_bd3lm, pdlm_k15_data, pdlm_k31_data)
    print("=" * 80)


if __name__ == "__main__":
    main()

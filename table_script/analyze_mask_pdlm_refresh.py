"""
Analyze mask_pdlm refresh decode sweep results.

Loads:
  - Sequential end2end results from --sequential_dir (existing table_mask_pdlm_sweep JSONs)
  - Refresh decode results from --refresh_dir (new table_mask_pdlm_refresh JSONs)

Prints comparison tables and writes summary.json + summary.csv.

Usage:
    python3 table_script/analyze_mask_pdlm_refresh.py
    python3 table_script/analyze_mask_pdlm_refresh.py \
        --refresh_dir=table_script/results/table_mask_pdlm_refresh \
        --sequential_dir=table_script/results/table_mask_pdlm_sweep
"""

import argparse
import csv
import json
import os

# ── Model definitions (same as analyze_mask_pdlm_sweep.py) ────────────────────

P_VALUES = [0, 30, 50, 70, 90, 100]
B_VALUES = [1, 2, 4, 8, 16]

P_SWEEP_K31 = [f"mask_pdlm_d8_b4_n256_k31_g496_p{p}_r40"  for p in P_VALUES]
P_SWEEP_K15 = [f"mask_pdlm_d8_b4_n512_k15_g120_p{p}_r40"  for p in P_VALUES]
B_SWEEP     = [f"mask_pdlm_d8_b{b}_n512_k15_g120_p50_r40"  for b in B_VALUES]

ALL_MODELS = list(dict.fromkeys(P_SWEEP_K31 + P_SWEEP_K15 + B_SWEEP))


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_seq(seq_dir, model):
    path = os.path.join(seq_dir, f"{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_ref(ref_dir, model):
    path = os.path.join(ref_dir, f"{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract_metrics(seq_data, ref_data):
    """
    Flat dict with prefixed keys for sequential and refresh.

    Keys:
      seq_ppl, seq_ent_ppl, seq_acc
      seq_pos{i}_ppl, seq_pos{i}_acc, seq_pos{i}_ent_ppl
      ref_ppl, ref_ent_ppl, ref_acc
      ref_pos{i}_ppl, ref_pos{i}_acc, ref_pos{i}_ent_ppl
    """
    m = {}

    # Sequential (from existing sweep JSON, end2end section)
    if seq_data is not None:
        e = seq_data.get("end2end", {})
        m["seq_ppl"]     = e.get("overall_ppl")
        m["seq_ent_ppl"] = e.get("overall_entropy_ppl")
        m["seq_acc"]     = e.get("overall_accuracy")
        positions = e.get("positions", {})
        for i in range(len(positions)):
            pm = positions.get(i, positions.get(str(i), {}))
            m[f"seq_pos{i}_ppl"]     = pm.get("ppl")
            m[f"seq_pos{i}_acc"]     = pm.get("accuracy")
            m[f"seq_pos{i}_ent_ppl"] = pm.get("entropy_ppl")

    # Refresh (from new refresh JSON, end2end_refresh section)
    if ref_data is not None:
        section = ref_data.get("end2end_refresh", {})
        m["ref_ppl"]     = section.get("overall_ppl")
        m["ref_ent_ppl"] = section.get("overall_entropy_ppl")
        m["ref_acc"]     = section.get("overall_accuracy")
        positions = section.get("positions", {})
        for i in range(len(positions)):
            pm = positions.get(i, positions.get(str(i), {}))
            m[f"ref_pos{i}_ppl"]     = pm.get("ppl")
            m[f"ref_pos{i}_acc"]     = pm.get("accuracy")
            m[f"ref_pos{i}_ent_ppl"] = pm.get("entropy_ppl")

    return m


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_ppl(v, w=8):
    return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"

def fmt_acc(v, w=8):
    return f"{v:>{w}.1%}" if v is not None else f"{'N/A':>{w}}"


# ── Table printer ─────────────────────────────────────────────────────────────

def print_sweep_table(title, col_labels, models, seq_data_list, ref_data_list):
    CW = 13
    sep = "─" * (24 + CW * len(col_labels))
    print()
    print(title)
    print(sep)
    header = f"{'metric':<24}" + "".join(f"{str(c):>{CW}}" for c in col_labels)
    print(header)
    print("─" * len(header))

    def row(label, vals):
        print(f"{label:<24}" + "".join(f"{v:>{CW}}" for v in vals))

    metrics = [extract_metrics(sd, rd) for sd, rd in zip(seq_data_list, ref_data_list)]

    row("seq_ppl",     [fmt_ppl(m.get("seq_ppl"),     8) for m in metrics])
    row("seq_ent_ppl", [fmt_ppl(m.get("seq_ent_ppl"), 8) for m in metrics])
    row("seq_acc",     [fmt_acc(m.get("seq_acc"),     8) for m in metrics])
    row("ref_ppl",     [fmt_ppl(m.get("ref_ppl"),     8) for m in metrics])
    row("ref_ent_ppl", [fmt_ppl(m.get("ref_ent_ppl"), 8) for m in metrics])
    row("ref_acc",     [fmt_acc(m.get("ref_acc"),     8) for m in metrics])

    # Per-position: ppl/acc for seq and ref
    sample_seq = next((d for d in seq_data_list if d is not None), None)
    if sample_seq:
        n_pos = len(sample_seq.get("end2end", {}).get("positions", {}))
        if n_pos > 0:
            for variant_label, prefix in [("seq", "seq"), ("ref", "ref")]:
                print()
                print(f"  per-position  [{variant_label}]  (ppl / acc):")
                print(f"  {'pos':<6}" + "".join(f"{str(c):>{CW}}" for c in col_labels))
                print("  " + "─" * (6 + CW * len(col_labels)))
                for pos in range(n_pos):
                    vals = []
                    for m in metrics:
                        ppl = m.get(f"{prefix}_pos{pos}_ppl")
                        acc = m.get(f"{prefix}_pos{pos}_acc")
                        if ppl is not None and acc is not None:
                            vals.append(f"{ppl:5.2f}/{acc:.1%}")
                        else:
                            vals.append("—")
                    print(f"  {pos:<6}" + "".join(f"{v:>{CW}}" for v in vals))

    # Delta table: (ref_acc - seq_acc) per position
    if sample_seq:
        n_pos = len(sample_seq.get("end2end", {}).get("positions", {}))
        if n_pos > 0:
            print()
            print(f"  Δacc  [ref - seq]  (positive = refresh helps):")
            print(f"  {'pos':<6}" + "".join(f"{str(c):>{CW}}" for c in col_labels))
            print("  " + "─" * (6 + CW * len(col_labels)))
            for pos in range(n_pos):
                vals = []
                for m in metrics:
                    seq_acc = m.get(f"seq_pos{pos}_acc")
                    ref_acc = m.get(f"ref_pos{pos}_acc")
                    if seq_acc is not None and ref_acc is not None:
                        delta = ref_acc - seq_acc
                        vals.append(f"{delta:+.1%}")
                    else:
                        vals.append("—")
                print(f"  {pos:<6}" + "".join(f"{v:>{CW}}" for v in vals))


# ── Summary files ─────────────────────────────────────────────────────────────

def write_summary_json(out_dir, all_seq, all_ref):
    summary = {}
    for model in ALL_MODELS:
        summary[model] = extract_metrics(all_seq.get(model), all_ref.get(model))
    out = os.path.join(out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written: {out}")
    return out


def write_summary_csv(out_dir, all_seq, all_ref):
    rows = []
    fieldnames_set = ["model"]
    for model in ALL_MODELS:
        m = extract_metrics(all_seq.get(model), all_ref.get(model))
        rows.append({"model": model, **m})
        for k in m:
            if k not in fieldnames_set:
                fieldnames_set.append(k)

    out = os.path.join(out_dir, "summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_set, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Written: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh_dir",
        default="table_script/results/table_mask_pdlm_refresh",
        help="Directory with refresh decode JSON results",
    )
    parser.add_argument(
        "--sequential_dir",
        default="table_script/results/table_mask_pdlm_sweep",
        help="Directory with sequential (existing sweep) JSON results",
    )
    args = parser.parse_args()

    ref_dir = args.refresh_dir
    seq_dir = args.sequential_dir

    all_seq = {m: load_seq(seq_dir, m) for m in ALL_MODELS}
    all_ref = {m: load_ref(ref_dir, m) for m in ALL_MODELS}

    missing_seq = [m for m, d in all_seq.items() if d is None]
    missing_ref = [m for m, d in all_ref.items() if d is None]
    if missing_seq:
        print(f"Warning: {len(missing_seq)} sequential result(s) missing:")
        for m in missing_seq:
            print(f"  {m}")
    if missing_ref:
        print(f"Warning: {len(missing_ref)} refresh result(s) missing:")
        for m in missing_ref:
            print(f"  {m}")

    print("\n" + "=" * 100)
    print("Mask PDLM Refresh Decode Analysis")
    print("  seq = sequential end2end (1 forward per step, no group refresh)")
    print("  ref = refresh end2end   (2 forwards per step: refresh G, then eval)")
    print("=" * 100)

    print_sweep_table(
        title      = "soft_p sweep — b=4, k31/g496",
        col_labels = [f"p={p}" for p in P_VALUES],
        models     = P_SWEEP_K31,
        seq_data_list = [all_seq[m] for m in P_SWEEP_K31],
        ref_data_list = [all_ref[m] for m in P_SWEEP_K31],
    )

    print_sweep_table(
        title      = "soft_p sweep — b=4, k15/g120",
        col_labels = [f"p={p}" for p in P_VALUES],
        models     = P_SWEEP_K15,
        seq_data_list = [all_seq[m] for m in P_SWEEP_K15],
        ref_data_list = [all_ref[m] for m in P_SWEEP_K15],
    )

    print_sweep_table(
        title      = "block_size sweep — p=50, k15/g120",
        col_labels = [f"b={b}" for b in B_VALUES],
        models     = B_SWEEP,
        seq_data_list = [all_seq[m] for m in B_SWEEP],
        ref_data_list = [all_ref[m] for m in B_SWEEP],
    )

    print("\n" + "=" * 100)
    print("Writing summary files...")
    write_summary_json(ref_dir, all_seq, all_ref)
    write_summary_csv(ref_dir, all_seq, all_ref)
    print("=" * 100)


if __name__ == "__main__":
    main()

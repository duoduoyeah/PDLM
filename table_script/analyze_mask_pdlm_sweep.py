"""
Analyze mask_pdlm sweep results.

Reads per-model JSONs from table_script/results/table_mask_pdlm_sweep/,
prints comparison tables, and writes summary.json + summary.csv.

Usage:
    python3 table_script/analyze_mask_pdlm_sweep.py
    python3 table_script/analyze_mask_pdlm_sweep.py --results_dir=path/to/jsons
"""

import argparse
import csv
import json
import os

# ── Model definitions ─────────────────────────────────────────────────────────

P_VALUES = [0, 30, 50, 70, 90, 100]
B_VALUES = [1, 2, 4, 8, 16]

P_SWEEP_K31 = [f"mask_pdlm_d8_b4_n256_k31_g496_p{p}_r40"    for p in P_VALUES]
P_SWEEP_K15 = [f"mask_pdlm_d8_b4_n512_k15_g120_p{p}_r40"    for p in P_VALUES]
B_SWEEP     = [f"mask_pdlm_d8_b{b}_n512_k15_g120_p50_r40"   for b in B_VALUES]

ALL_MODELS  = list(dict.fromkeys(P_SWEEP_K31 + P_SWEEP_K15 + B_SWEEP))  # unique, ordered


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(results_dir, model):
    path = os.path.join(results_dir, f"{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_ppl(v, w=8):
    return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"

def fmt_acc(v, w=8):
    return f"{v:>{w}.1%}" if v is not None else f"{'N/A':>{w}}"

def fmt_loss(v, w=8):
    return f"{v:>{w}.4f}" if v is not None else f"{'N/A':>{w}}"


def get_pos_list(d, section):
    """Return list of per-position metric dicts sorted by index."""
    positions = d.get(section, {}).get("positions", {})
    n = len(positions)
    return [positions.get(i, positions.get(str(i), {})) for i in range(n)]


def extract_metrics(d):
    """Flat dict of key metrics for a single model result."""
    if d is None:
        return {}
    u = d.get("unified", {})
    e = d.get("end2end", {})
    m = {
        "unified_loss":     u.get("overall_loss"),
        "unified_ppl":      u.get("overall_ppl"),
        "unified_ent_ppl":  u.get("overall_entropy_ppl"),
        "e2e_loss":         e.get("overall_loss"),
        "e2e_ppl":          e.get("overall_ppl"),
        "e2e_ent_ppl":      e.get("overall_entropy_ppl"),
        "e2e_acc":          e.get("overall_accuracy"),
    }
    # Per-position end2end
    for pos_metrics in get_pos_list(d, "end2end"):
        pos = pos_metrics.get("pos", get_pos_list(d, "end2end").index(pos_metrics))
    for i, pm in enumerate(get_pos_list(d, "end2end")):
        m[f"e2e_pos{i}_ppl"]      = pm.get("ppl")
        m[f"e2e_pos{i}_acc"]      = pm.get("accuracy")
        m[f"e2e_pos{i}_ent_ppl"]  = pm.get("entropy_ppl")
    return m


# ── Table printer ─────────────────────────────────────────────────────────────

def print_sweep_table(title, col_labels, models, data_list):
    CW = 11
    sep = "─" * (22 + CW * len(col_labels))
    print()
    print(title)
    print(sep)
    header = f"{'metric':<22}" + "".join(f"{str(c):>{CW}}" for c in col_labels)
    print(header)
    print("─" * len(header))

    def row(label, vals):
        print(f"{label:<22}" + "".join(f"{v:>{CW}}" for v in vals))

    metrics = [extract_metrics(d) for d in data_list]

    row("unified_ppl",     [fmt_ppl(m.get("unified_ppl"),     7) for m in metrics])
    row("unified_ent_ppl", [fmt_ppl(m.get("unified_ent_ppl"), 7) for m in metrics])
    row("end2end_ppl",     [fmt_ppl(m.get("e2e_ppl"),         7) for m in metrics])
    row("end2end_ent_ppl", [fmt_ppl(m.get("e2e_ent_ppl"),     7) for m in metrics])
    row("end2end_acc",     [fmt_acc(m.get("e2e_acc"),         7) for m in metrics])

    # Per-position end2end: ppl / acc
    sample = next((d for d in data_list if d is not None), None)
    if sample:
        n_pos = len(sample.get("end2end", {}).get("positions", {}))
        if n_pos > 0:
            print()
            print("  end2end per-position  (ppl / acc):")
            CW2 = 16
            print(f"  {'pos':<6}" + "".join(f"{str(c):>{CW2}}" for c in col_labels))
            print("  " + "─" * (6 + CW2 * len(col_labels)))
            for pos in range(n_pos):
                vals = []
                for d in data_list:
                    pm = get_pos_list(d, "end2end")
                    if pos < len(pm):
                        ppl = pm[pos].get("ppl")
                        acc = pm[pos].get("accuracy")
                        vals.append(f"{ppl:5.2f} / {acc:.1%}" if ppl and acc else "—")
                    else:
                        vals.append("—")
                print(f"  {pos:<6}" + "".join(f"{v:>{CW2}}" for v in vals))

    # Per-position end2end: entropy_ppl
    if sample:
        n_pos = len(sample.get("end2end", {}).get("positions", {}))
        has_eppl = any(
            get_pos_list(d, "end2end")[0].get("entropy_ppl") is not None
            for d in data_list if d and get_pos_list(d, "end2end")
        )
        if n_pos > 0 and has_eppl:
            print()
            print("  end2end per-position  (entropy_ppl):")
            CW2 = 16
            print(f"  {'pos':<6}" + "".join(f"{str(c):>{CW2}}" for c in col_labels))
            print("  " + "─" * (6 + CW2 * len(col_labels)))
            for pos in range(n_pos):
                vals = []
                for d in data_list:
                    pm = get_pos_list(d, "end2end")
                    if pos < len(pm):
                        v = pm[pos].get("entropy_ppl")
                        vals.append(f"{v:>8.2f}" if v else "—")
                    else:
                        vals.append("—")
                print(f"  {pos:<6}" + "".join(f"{v:>{CW2}}" for v in vals))


# ── Summary files ─────────────────────────────────────────────────────────────

def write_summary_json(results_dir, all_data):
    """Write summary.json: {model: flat_metrics_dict}."""
    summary = {}
    for model, d in all_data.items():
        summary[model] = extract_metrics(d)
    out = os.path.join(results_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written: {out}")
    return out


def write_summary_csv(results_dir, all_data):
    """Write summary.csv: one row per model, columns = all metrics."""
    rows = []
    fieldnames_set = ["model"]
    for model, d in all_data.items():
        m = extract_metrics(d)
        rows.append({"model": model, **m})
        for k in m:
            if k not in fieldnames_set:
                fieldnames_set.append(k)

    out = os.path.join(results_dir, "summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_set, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Written: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="table_script/results/table_mask_pdlm_sweep",
        help="Directory containing per-model result JSONs",
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    # Load all models
    all_data = {m: load(results_dir, m) for m in ALL_MODELS}
    missing = [m for m, d in all_data.items() if d is None]
    if missing:
        print(f"Warning: {len(missing)} model(s) missing results:")
        for m in missing:
            print(f"  {m}")

    # ── Tables ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("Mask PDLM Sweep Analysis")
    print("=" * 88)

    print_sweep_table(
        title      = "soft_p_within sweep — b=4, k31/g496 (n=256, k=31, g=496)",
        col_labels = [f"p={p}" for p in P_VALUES],
        models     = P_SWEEP_K31,
        data_list  = [all_data[m] for m in P_SWEEP_K31],
    )

    print_sweep_table(
        title      = "soft_p_within sweep — b=4, k15/g120 (n=512, k=15, g=120)",
        col_labels = [f"p={p}" for p in P_VALUES],
        models     = P_SWEEP_K15,
        data_list  = [all_data[m] for m in P_SWEEP_K15],
    )

    print_sweep_table(
        title      = "block_size sweep — p=50, k15/g120 (n=512, k=15, g=120)",
        col_labels = [f"b={b}" for b in B_VALUES],
        models     = B_SWEEP,
        data_list  = [all_data[m] for m in B_SWEEP],
    )

    # ── Summary files ─────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("Writing summary files...")
    write_summary_json(results_dir, all_data)
    write_summary_csv(results_dir, all_data)
    print("=" * 88)


if __name__ == "__main__":
    main()

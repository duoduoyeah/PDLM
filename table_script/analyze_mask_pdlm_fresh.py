"""
Analyze mask_pdlm fresh-mask-G decode sweep results.

Two goals:
  1. Inertia isolation: compare seq_acc vs fresh_acc per position.
     If fresh_acc at pos1+ recovers toward p=0 levels, the accuracy drop in
     normal end2end eval is from G-token trajectory inertia, not model capability.

  2. Confidence: report argmax_prob, top3_sum, top5_sum per position across p values.
     Higher p should yield higher argmax_prob, supporting the claim that soft group
     tokens make the model more confident than uninformative mask tokens.

Loads:
  - Sequential end2end results from --sequential_dir (table_mask_pdlm_sweep JSONs)
  - Fresh-mask-G results from --fresh_dir (table_mask_pdlm_fresh JSONs)

Prints comparison tables and writes summary.json + summary.csv.

Usage:
    python3 table_script/analyze_mask_pdlm_fresh.py
    python3 table_script/analyze_mask_pdlm_fresh.py \\
        --fresh_dir=table_script/results/table_mask_pdlm_fresh \\
        --sequential_dir=table_script/results/table_mask_pdlm_sweep
"""

import argparse
import csv
import json
import os

# ── Model definitions ─────────────────────────────────────────────────────────

P_VALUES = [0, 30, 50, 70, 90, 100]
B_VALUES = [1, 2, 4, 8, 16]

P_SWEEP_K31 = [f"mask_pdlm_d8_b4_n256_k31_g496_p{p}_r40"  for p in P_VALUES]
P_SWEEP_K15 = [f"mask_pdlm_d8_b4_n512_k15_g120_p{p}_r40"  for p in P_VALUES]
B_SWEEP     = [f"mask_pdlm_d8_b{b}_n512_k15_g120_p50_r40"  for b in B_VALUES]

ALL_MODELS = list(dict.fromkeys(P_SWEEP_K31 + P_SWEEP_K15 + B_SWEEP))

_CONF_KEYS = ("argmax_prob", "second_prob", "third_prob", "top3_sum", "top5_sum")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(directory, model):
    path = os.path.join(directory, f"{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract_metrics(seq_data, fresh_data):
    """
    Flat dict with prefixed keys.

    Keys:
      seq_ppl, seq_ent_ppl, seq_acc
      seq_pos{i}_ppl, seq_pos{i}_acc, seq_pos{i}_ent_ppl
      fresh_ppl, fresh_ent_ppl, fresh_acc
      fresh_pos{i}_ppl, fresh_pos{i}_acc, fresh_pos{i}_ent_ppl
      fresh_pos{i}_argmax_prob, fresh_pos{i}_second_prob, fresh_pos{i}_third_prob
      fresh_pos{i}_top3_sum, fresh_pos{i}_top5_sum
      fresh_overall_argmax_prob, fresh_overall_top3_sum, fresh_overall_top5_sum
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

    # Fresh-mask-G (from new fresh JSON, end2end_fresh_mask_g section)
    if fresh_data is not None:
        section = fresh_data.get("end2end_fresh_mask_g", {})
        m["fresh_ppl"]     = section.get("overall_ppl")
        m["fresh_ent_ppl"] = section.get("overall_entropy_ppl")
        m["fresh_acc"]     = section.get("overall_accuracy")
        # Overall confidence metrics
        for key in _CONF_KEYS:
            v = section.get(f"overall_{key}")
            if v is not None:
                m[f"fresh_overall_{key}"] = v
        positions = section.get("positions", {})
        for i in range(len(positions)):
            pm = positions.get(i, positions.get(str(i), {}))
            m[f"fresh_pos{i}_ppl"]     = pm.get("ppl")
            m[f"fresh_pos{i}_acc"]     = pm.get("accuracy")
            m[f"fresh_pos{i}_ent_ppl"] = pm.get("entropy_ppl")
            for key in _CONF_KEYS:
                v = pm.get(key)
                if v is not None:
                    m[f"fresh_pos{i}_{key}"] = v

    return m


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_ppl(v, w=8):
    return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"

def fmt_acc(v, w=8):
    return f"{v:>{w}.1%}" if v is not None else f"{'N/A':>{w}}"

def fmt_prob(v, w=8):
    return f"{v:>{w}.4f}" if v is not None else f"{'N/A':>{w}}"


# ── Table printers ────────────────────────────────────────────────────────────

def print_accuracy_table(title, col_labels, seq_data_list, fresh_data_list):
    """Print seq_acc vs fresh_acc per position."""
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

    metrics = [extract_metrics(sd, fd) for sd, fd in zip(seq_data_list, fresh_data_list)]

    row("seq_ppl",     [fmt_ppl(m.get("seq_ppl"),     8) for m in metrics])
    row("seq_ent_ppl", [fmt_ppl(m.get("seq_ent_ppl"), 8) for m in metrics])
    row("seq_acc",     [fmt_acc(m.get("seq_acc"),     8) for m in metrics])
    row("fresh_ppl",   [fmt_ppl(m.get("fresh_ppl"),   8) for m in metrics])
    row("fresh_ent_ppl",[fmt_ppl(m.get("fresh_ent_ppl"),8) for m in metrics])
    row("fresh_acc",   [fmt_acc(m.get("fresh_acc"),   8) for m in metrics])

    # Per-position acc: seq vs fresh
    sample_seq = next((d for d in seq_data_list if d is not None), None)
    if sample_seq:
        n_pos = len(sample_seq.get("end2end", {}).get("positions", {}))
        if n_pos > 0:
            for variant_label, prefix in [("seq", "seq"), ("fresh", "fresh")]:
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

            # Δacc = fresh_acc - seq_acc per position
            print()
            print(f"  Δacc  [fresh - seq]  (positive = inertia was hurting seq):")
            print(f"  {'pos':<6}" + "".join(f"{str(c):>{CW}}" for c in col_labels))
            print("  " + "─" * (6 + CW * len(col_labels)))
            for pos in range(n_pos):
                vals = []
                for m in metrics:
                    seq_acc   = m.get(f"seq_pos{pos}_acc")
                    fresh_acc = m.get(f"fresh_pos{pos}_acc")
                    if seq_acc is not None and fresh_acc is not None:
                        vals.append(f"{fresh_acc - seq_acc:+.1%}")
                    else:
                        vals.append("—")
                print(f"  {pos:<6}" + "".join(f"{v:>{CW}}" for v in vals))


def print_confidence_table(title, col_labels, fresh_data_list):
    """Print argmax_prob, top3_sum, top5_sum per position from fresh-mask-G eval."""
    CW = 10
    # Determine max positions from data
    n_pos = 0
    for fd in fresh_data_list:
        if fd is not None:
            section = fd.get("end2end_fresh_mask_g", {})
            n_pos = max(n_pos, len(section.get("positions", {})))
    if n_pos == 0:
        return

    sep = "─" * (20 + CW * len(col_labels))
    print()
    print(title)
    print(sep)
    header = f"{'metric':<20}" + "".join(f"{str(c):>{CW}}" for c in col_labels)
    print(header)
    print("─" * len(header))

    def row(label, vals):
        print(f"{label:<20}" + "".join(f"{v:>{CW}}" for v in vals))

    def get_conf(fd, pos, key):
        if fd is None:
            return None
        pm = fd.get("end2end_fresh_mask_g", {}).get("positions", {})
        entry = pm.get(pos, pm.get(str(pos), {}))
        return entry.get(key)

    for conf_key, conf_label in [
        ("argmax_prob", "argmax_prob"),
        ("top3_sum",    "top3_sum"),
        ("top5_sum",    "top5_sum"),
    ]:
        print()
        print(f"  {conf_label}  (avg prob per token, higher = more confident):")
        print(f"  {'pos':<6}" + "".join(f"{str(c):>{CW}}" for c in col_labels))
        print("  " + "─" * (6 + CW * len(col_labels)))
        for pos in range(n_pos):
            vals = []
            for fd in fresh_data_list:
                v = get_conf(fd, pos, conf_key)
                vals.append(f"{v:{CW}.4f}" if v is not None else f"{'—':>{CW}}")
            print(f"  {pos:<6}" + "".join(vals))

    # Also print second_prob and third_prob for completeness
    print()
    print("  Probability rank breakdown at pos0 (sequential, no G bias):")
    print(f"  {'rank':<10}" + "".join(f"{str(c):>{CW}}" for c in col_labels))
    print("  " + "─" * (10 + CW * len(col_labels)))
    for conf_key, rank_label in [
        ("argmax_prob", "P(top-1)"),
        ("second_prob", "P(top-2)"),
        ("third_prob",  "P(top-3)"),
    ]:
        vals = []
        for fd in fresh_data_list:
            v = get_conf(fd, 0, conf_key)
            vals.append(f"{v:{CW}.4f}" if v is not None else f"{'—':>{CW}}")
        print(f"  {rank_label:<10}" + "".join(vals))


# ── Summary files ─────────────────────────────────────────────────────────────

def write_summary_json(out_dir, all_seq, all_fresh):
    summary = {}
    for model in ALL_MODELS:
        summary[model] = extract_metrics(all_seq.get(model), all_fresh.get(model))
    out = os.path.join(out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written: {out}")
    return out


def write_summary_csv(out_dir, all_seq, all_fresh):
    rows = []
    fieldnames_set = ["model"]
    for model in ALL_MODELS:
        m = extract_metrics(all_seq.get(model), all_fresh.get(model))
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
        "--fresh_dir",
        default="table_script/results/table_mask_pdlm_fresh",
        help="Directory with fresh-mask-G decode JSON results",
    )
    parser.add_argument(
        "--sequential_dir",
        default="table_script/results/table_mask_pdlm_sweep",
        help="Directory with sequential (existing sweep) JSON results",
    )
    args = parser.parse_args()

    fresh_dir = args.fresh_dir
    seq_dir   = args.sequential_dir

    all_seq   = {m: load_json(seq_dir,   m) for m in ALL_MODELS}
    all_fresh = {m: load_json(fresh_dir, m) for m in ALL_MODELS}

    missing_seq   = [m for m, d in all_seq.items()   if d is None]
    missing_fresh = [m for m, d in all_fresh.items() if d is None]
    if missing_seq:
        print(f"Warning: {len(missing_seq)} sequential result(s) missing:")
        for m in missing_seq:
            print(f"  {m}")
    if missing_fresh:
        print(f"Warning: {len(missing_fresh)} fresh-mask-G result(s) missing:")
        for m in missing_fresh:
            print(f"  {m}")

    print("\n" + "=" * 100)
    print("Mask PDLM Fresh-Mask-G Analysis")
    print("  seq   = sequential end2end (1 forward/step, G from stale collapse)")
    print("  fresh = fresh-mask-G      (2 forwards/step: MASK→G, then eval — no inertia)")
    print("=" * 100)

    # ── Accuracy tables ───────────────────────────────────────────────────────
    print_accuracy_table(
        title      = "soft_p sweep — b=4, k31/g496  [Δacc: fresh - seq]",
        col_labels = [f"p={p}" for p in P_VALUES],
        seq_data_list   = [all_seq[m]   for m in P_SWEEP_K31],
        fresh_data_list = [all_fresh[m] for m in P_SWEEP_K31],
    )

    print_accuracy_table(
        title      = "soft_p sweep — b=4, k15/g120  [Δacc: fresh - seq]",
        col_labels = [f"p={p}" for p in P_VALUES],
        seq_data_list   = [all_seq[m]   for m in P_SWEEP_K15],
        fresh_data_list = [all_fresh[m] for m in P_SWEEP_K15],
    )

    print_accuracy_table(
        title      = "block_size sweep — p=50, k15/g120  [Δacc: fresh - seq]",
        col_labels = [f"b={b}" for b in B_VALUES],
        seq_data_list   = [all_seq[m]   for m in B_SWEEP],
        fresh_data_list = [all_fresh[m] for m in B_SWEEP],
    )

    # ── Confidence tables ─────────────────────────────────────────────────────
    print_confidence_table(
        title      = "Confidence (argmax_prob, top3_sum, top5_sum) — soft_p sweep, b=4, k31/g496",
        col_labels = [f"p={p}" for p in P_VALUES],
        fresh_data_list = [all_fresh[m] for m in P_SWEEP_K31],
    )

    print_confidence_table(
        title      = "Confidence (argmax_prob, top3_sum, top5_sum) — soft_p sweep, b=4, k15/g120",
        col_labels = [f"p={p}" for p in P_VALUES],
        fresh_data_list = [all_fresh[m] for m in P_SWEEP_K15],
    )

    print_confidence_table(
        title      = "Confidence (argmax_prob, top3_sum, top5_sum) — block_size sweep, p=50, k15/g120",
        col_labels = [f"b={b}" for b in B_VALUES],
        fresh_data_list = [all_fresh[m] for m in B_SWEEP],
    )

    print("\n" + "=" * 100)
    print("Writing summary files...")
    write_summary_json(fresh_dir, all_seq, all_fresh)
    write_summary_csv(fresh_dir, all_seq, all_fresh)
    print("=" * 100)


if __name__ == "__main__":
    main()

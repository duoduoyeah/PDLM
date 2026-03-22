"""
Analyze Table 5: p-sweep — effect of group token softness on model decisiveness.

Reads eval JSONs for PDLM n256 (g496) at p=0,30,50,70,90,100
and prints a comparison table with PPL, Accuracy, Ent-PPL, Argmax Prob.

Usage:
    python3 table_script/analyze_table5_psweep.py
    python3 table_script/analyze_table5_psweep.py --results_dir=table_script/results/table5_psweep
"""

import argparse
import csv
import json
import os

# ── Model definitions ─────────────────────────────────────────────────────────

MODELS = [
    {"key": "p0",   "label": "p=0"},
    {"key": "p30",  "label": "p=30"},
    {"key": "p50",  "label": "p=50"},
    {"key": "p70",  "label": "p=70"},
    {"key": "p90",  "label": "p=90"},
    {"key": "p100", "label": "p=100"},
]


def extract_metrics(result):
    """Extract PPL, Accuracy, Ent-PPL, Argmax Prob from PDLM eval result JSON."""
    # Support both fresh_mask and default eval formats
    e2e = result.get("end2end_fresh_mask_g", result.get("end2end", result))
    return {
        "ppl": e2e.get("overall_ppl"),
        "accuracy": e2e.get("overall_accuracy"),
        "ent_ppl": e2e.get("overall_entropy_ppl"),
        "argmax_prob": e2e.get("overall_argmax_prob"),
    }


def fmt(val, is_pct=False):
    """Format a metric value."""
    if val is None:
        return "--"
    if is_pct:
        return f"{val:.2%}"
    return f"{val:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze Table 5: p-sweep")
    parser.add_argument("--results_dir", type=str,
                        default="table_script/results/table5_psweep",
                        help="Directory with per-model JSON results")
    args = parser.parse_args()

    # Load results
    rows = []
    for model in MODELS:
        json_path = os.path.join(args.results_dir, f"{model['key']}.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping {model['label']}")
            rows.append({"label": model["label"], "ppl": None, "accuracy": None, "ent_ppl": None, "argmax_prob": None})
            continue
        with open(json_path) as f:
            result = json.load(f)
        metrics = extract_metrics(result)
        metrics["label"] = model["label"]
        rows.append(metrics)

    # Print table
    print()
    print("=" * 70)
    print("Table 5: Effect of Group Token Softness (n256, g496, 4s)")
    print("=" * 70)
    header = f"{'p':<10s} {'PPL':>8s} {'Accuracy':>10s} {'Ent-PPL':>10s} {'Argmax Prob':>12s}"
    print(header)
    print("-" * 70)
    for r in rows:
        line = f"{r['label']:<10s} {fmt(r['ppl']):>8s} {fmt(r['accuracy'], is_pct=True):>10s} {fmt(r['ent_ppl']):>10s} {fmt(r['argmax_prob']):>12s}"
        print(line)
    print("=" * 70)

    # Save CSV
    csv_path = os.path.join(args.results_dir, "table5_psweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "ppl", "accuracy", "ent_ppl", "argmax_prob"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {csv_path}")

    # Save JSON summary
    json_path = os.path.join(args.results_dir, "table5_psweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()

"""
Analyze Table 4: Equivalence Check.

Reads eval JSONs for AR (full), AR (10/16), BD3-LM, PDLM p=0
and prints a comparison table with PPL, Accuracy, Ent-PPL, Argmax Prob.

Usage:
    python3 table_script/analyze_table4_equivalence.py
    python3 table_script/analyze_table4_equivalence.py --results_dir=table_script/results/table4_equivalence
"""

import argparse
import csv
import json
import os

# ── Model definitions ─────────────────────────────────────────────────────────

MODELS = [
    {"key": "ar_full",  "label": "AR (full)"},
    {"key": "ar_10_16", "label": "AR (10/16)"},
    {"key": "bd3lm",    "label": "BD3-LM"},
    {"key": "pdlm_p0",  "label": "PDLM p=0"},
]


def extract_metrics(result, key):
    """Extract PPL, Accuracy, Ent-PPL, Argmax Prob from eval result JSON."""
    # AR models: metrics at top level
    if key in ("ar_full", "ar_10_16"):
        return {
            "ppl": result.get("overall_ppl"),
            "accuracy": result.get("overall_accuracy"),
            "ent_ppl": result.get("overall_entropy_ppl"),
            "argmax_prob": result.get("overall_argmax_prob"),
        }
    # BD3-LM: metrics inside "left_to_right" sub-dict
    elif key == "bd3lm":
        ltr = result.get("left_to_right", result)
        return {
            "ppl": ltr.get("overall_ppl"),
            "accuracy": ltr.get("overall_accuracy"),
            "ent_ppl": ltr.get("overall_entropy_ppl"),
            "argmax_prob": ltr.get("overall_argmax_prob"),
        }
    # PDLM: metrics inside "end2end" sub-dict
    elif key == "pdlm_p0":
        e2e = result.get("end2end", result)
        return {
            "ppl": e2e.get("overall_ppl"),
            "accuracy": e2e.get("overall_accuracy"),
            "ent_ppl": e2e.get("overall_entropy_ppl"),
            "argmax_prob": e2e.get("overall_argmax_prob"),
        }
    return {}


def fmt(val, is_pct=False):
    """Format a metric value."""
    if val is None:
        return "--"
    if is_pct:
        return f"{val:.2%}"
    return f"{val:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze Table 4: Equivalence Check")
    parser.add_argument("--results_dir", type=str,
                        default="table_script/results/table4_equivalence",
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
        metrics = extract_metrics(result, model["key"])
        metrics["label"] = model["label"]
        rows.append(metrics)

    # Print table
    print()
    print("=" * 70)
    print("Table 4: Equivalence Check")
    print("=" * 70)
    header = f"{'Model':<15s} {'PPL':>8s} {'Accuracy':>10s} {'Ent-PPL':>10s} {'Argmax Prob':>12s}"
    print(header)
    print("-" * 70)
    for r in rows:
        line = f"{r['label']:<15s} {fmt(r['ppl']):>8s} {fmt(r['accuracy'], is_pct=True):>10s} {fmt(r['ent_ppl']):>10s} {fmt(r['argmax_prob']):>12s}"
        print(line)
    print("=" * 70)

    # Save CSV
    csv_path = os.path.join(args.results_dir, "table4_equivalence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "ppl", "accuracy", "ent_ppl", "argmax_prob"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {csv_path}")

    # Save JSON summary
    json_path = os.path.join(args.results_dir, "table4_equivalence_summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()

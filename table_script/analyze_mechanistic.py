"""
Mechanistic validation: group token as prior.

Plots Δ_random and Δ_true (y) vs p (x) for n256 and n512.
Two subplots: n256 (left) and n512 (right).
Each subplot has two lines: Δ_random and Δ_true.

Usage:
    python3 table_script/analyze_mechanistic.py
    python3 table_script/analyze_mechanistic.py --summary=table_script/results/mechanistic_summary.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


P_VALUES = [0, 30, 50, 70, 90, 100]
NUM_GROUPS = [256, 512]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="table_script/results/mechanistic_summary.json")
    parser.add_argument("--output", default="table_script/results/mechanistic_psweep.pdf")
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, ng in zip(axes, NUM_GROUPS):
        delta_rand = []
        delta_true = []
        for p in P_VALUES:
            key = f"n{ng}_p{p}"
            if key not in data:
                print(f"[skip] {key}: not found")
                delta_rand.append(None)
                delta_true.append(None)
                continue
            delta_rand.append(data[key]["avg"]["avg_delta_random"])
            delta_true.append(data[key]["avg"]["avg_delta_true"])

        ax.plot(P_VALUES, delta_rand, "-o", markersize=5, linewidth=2,
                color="#1f77b4", label=r"$\Delta_{\mathrm{rand}}$")
        ax.plot(P_VALUES, delta_true, "-s", markersize=5, linewidth=2,
                color="#d62728", label=r"$\Delta_{\mathrm{true}}$")

        ax.set_xlabel(r"Softness $p$", fontsize=11)
        ax.set_title(rf"$n_{{\mathrm{{groups}}}}={ng}$", fontsize=12)
        ax.set_xticks(P_VALUES)
        ax.set_ylim(-0.05, 1.0)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10, loc="upper left")

    axes[0].set_ylabel(r"$\Delta$ Coverage", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    png_output = args.output.rsplit(".", 1)[0] + ".png"
    fig.savefig(png_output, bbox_inches="tight", dpi=150)
    print(f"Saved: {args.output}")
    print(f"Saved: {png_output}")

    # Summary table
    print(f"\n{'Config':<12} {'Δ_rand':>10} {'Δ_true':>10}")
    print("-" * 35)
    for ng in NUM_GROUPS:
        for p in P_VALUES:
            key = f"n{ng}_p{p}"
            if key not in data:
                continue
            d = data[key]["avg"]
            print(f"n{ng} p={p:<3d}  {d['avg_delta_random']:>10.4f} {d['avg_delta_true']:>10.4f}")


if __name__ == "__main__":
    main()

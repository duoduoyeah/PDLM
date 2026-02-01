"""
Plot MTP evaluation results.

Usage:
    python scripts/plot_mtp_eval.py temp_eval_result.json
    python scripts/plot_mtp_eval.py temp_eval_result.json --output_dir /tmp/plots
"""

import json
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_position_metrics(results, output_dir):
    """Bar charts for loss, ppl, accuracy per position k."""
    positions = results["positions"]
    ks = sorted(positions.keys(), key=int)
    K = len(ks)

    losses = [positions[k]["loss"] for k in ks]
    ppls = [positions[k]["ppl"] for k in ks]
    accs = [positions[k]["accuracy"] for k in ks]
    k_labels = [f"k={k}" for k in ks]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    bars = axes[0].bar(k_labels, losses, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Loss by Position")
    axes[0].set_ylabel("Loss (CE)")
    for bar, v in zip(bars, losses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # PPL
    bars = axes[1].bar(k_labels, ppls, color="#DD8452", edgecolor="white")
    axes[1].set_title("Perplexity by Position")
    axes[1].set_ylabel("Perplexity")
    for bar, v in zip(bars, ppls):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Accuracy
    bars = axes[2].bar(k_labels, accs, color="#55A868", edgecolor="white")
    axes[2].set_title("Group Accuracy by Position")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1.05)
    for bar, v in zip(bars, accs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.1%}", ha="center", va="bottom", fontsize=9)

    overall = results["overall_accuracy"]
    axes[2].axhline(y=overall, color="red", linestyle="--", linewidth=1, label=f"Overall: {overall:.1%}")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"MTP Eval — Overall: loss={results['overall_loss']:.3f}, "
                 f"ppl={results['overall_ppl']:.2f}, acc={results['overall_accuracy']:.1%}",
                 fontsize=11)
    fig.tight_layout()
    path = f"{output_dir}/position_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_group_accuracy_distribution(results, output_dir):
    """Histogram + sorted curve of per-group accuracy."""
    groups = results["per_group_accuracy"]
    accs = []
    totals = []
    for g in groups.values():
        if g["total"] > 0:
            accs.append(g["accuracy"])
            totals.append(g["total"])

    accs = np.array(accs)
    totals = np.array(totals)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Histogram
    axes[0].hist(accs, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.2%}")
    axes[0].axvline(np.median(accs), color="orange", linestyle="--", label=f"Median: {np.median(accs):.2%}")
    axes[0].set_title("Distribution of Per-Group Accuracy")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_ylabel("Number of Groups")
    axes[0].legend(fontsize=8)

    # Sorted accuracy curve
    sorted_accs = np.sort(accs)[::-1]
    axes[1].plot(sorted_accs, color="#DD8452", linewidth=1.5)
    axes[1].set_title("Per-Group Accuracy (sorted)")
    axes[1].set_xlabel("Group rank")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8)

    # Accuracy vs group size (total)
    axes[2].scatter(totals, accs, alpha=0.4, s=12, color="#55A868")
    axes[2].set_title("Accuracy vs Group Size")
    axes[2].set_xlabel("Group total (weighted count)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.02, 1.02)

    fig.suptitle(f"Per-Group Accuracy Analysis — {len(accs)} groups with data", fontsize=11)
    fig.tight_layout()
    path = f"{output_dir}/group_accuracy_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_bottom_groups(results, output_dir, n=20):
    """Horizontal bar chart of top-N and bottom-N groups."""
    groups = results["per_group_accuracy"]
    items = [(int(g), d) for g, d in groups.items() if d["total"] > 10]  # filter tiny groups
    items.sort(key=lambda x: x[1]["accuracy"], reverse=True)

    top = items[:n]
    bottom = items[-n:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Top N
    labels = [f"G{g} (n={d['total']:.0f})" for g, d in top]
    vals = [d["accuracy"] for _, d in top]
    y = range(len(labels))
    axes[0].barh(y, vals, color="#55A868", edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title(f"Top {n} Groups by Accuracy")
    axes[0].invert_yaxis()
    for i, v in enumerate(vals):
        axes[0].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=7)

    # Bottom N
    labels = [f"G{g} (n={d['total']:.0f})" for g, d in bottom]
    vals = [d["accuracy"] for _, d in bottom]
    y = range(len(labels))
    axes[1].barh(y, vals, color="#C44E52", edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlim(0, max(vals) * 1.3 + 0.01 if max(vals) > 0 else 0.1)
    axes[1].set_title(f"Bottom {n} Groups by Accuracy (total > 10)")
    axes[1].invert_yaxis()
    for i, v in enumerate(vals):
        axes[1].text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=7)

    fig.tight_layout()
    path = f"{output_dir}/top_bottom_groups.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MTP eval results")
    parser.add_argument("json_path", help="Path to eval_result.json")
    parser.add_argument("--output_dir", default=".", help="Directory for output PNGs")
    args = parser.parse_args()

    results = load_results(args.json_path)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_position_metrics(results, args.output_dir)
    plot_group_accuracy_distribution(results, args.output_dir)
    plot_top_bottom_groups(results, args.output_dir)

    print("\nDone. Generated 3 plots.")


if __name__ == "__main__":
    main()

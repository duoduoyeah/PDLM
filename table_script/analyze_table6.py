"""
Table 6: Threshold-based parallel decoding.

Plots avg tokens/step (y) vs threshold τ (x) for:
  - BD3-LM baseline (dashed)
  - PDLM with varying p (0, 30, 50, 70, 90, 100)

Two subplots: n256 (left) and n512 (right).

Usage:
    python3 table_script/analyze_table6.py
    python3 table_script/analyze_table6.py \
        --bd3lm_dir=table_script/results/table6_bd3lm \
        --pdlm_dir=table_script/results/table6_mask_pdlm
"""

import argparse
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


BLOCK_SIZE = 4
P_VALUES = [0, 30, 50, 70, 90, 100]

BD3LM_MODEL = "bd3lm_d8_b4_normal_r40"

CONFIGS = {
    "n256": {
        "title": r"$n_{\mathrm{groups}}=256$",
        "models": {
            p: f"mask_pdlm_d8_b4_n256_k31_g496_p{p}_r40" for p in P_VALUES
        },
    },
    "n512": {
        "title": r"$n_{\mathrm{groups}}=512$",
        "models": {
            p: f"mask_pdlm_d8_b4_n512_k15_g120_p{p}_r40" for p in P_VALUES
        },
    },
}


def load_threshold_json(result_dir, model_name, subfolder_prefix):
    """Find and load the latest eval_threshold_*.json for a model."""
    pattern = os.path.join(
        result_dir, subfolder_prefix, model_name, "seq*_threshold", "eval_threshold_*.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def extract_curve(data):
    """Extract (thresholds, tokens_per_step) from eval JSON."""
    td = data["threshold_decode"]
    thresholds = []
    tokens_per_step = []
    for key in sorted(td.keys(), key=float):
        thresholds.append(td[key]["threshold"])
        tokens_per_step.append(BLOCK_SIZE / td[key]["avg_steps"])
    return thresholds, tokens_per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bd3lm_dir", default="table_script/results/table6_bd3lm")
    parser.add_argument("--pdlm_dir", default="table_script/results/table6_mask_pdlm")
    parser.add_argument("--output", default="table_script/results/table6_threshold_decode_tau0.4_0.9.pdf")
    args = parser.parse_args()

    # Load BD3-LM baseline
    bd3lm_data = load_threshold_json(args.bd3lm_dir, BD3LM_MODEL, "bd3lm")

    # Color map: p=0 (light) → p=100 (dark)
    cmap = plt.get_cmap("Blues", len(P_VALUES) + 2)
    p_colors = {p: cmap(i + 2) for i, p in enumerate(P_VALUES)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (config_key, config) in zip(axes, CONFIGS.items()):
        # BD3-LM baseline
        if bd3lm_data:
            tau, tps = extract_curve(bd3lm_data)
            ax.plot(tau, tps, color="black", linestyle="--", linewidth=2,
                    label="BD3-LM", zorder=10)

        # PDLM lines
        for p in P_VALUES:
            model_name = config["models"][p]
            data = load_threshold_json(args.pdlm_dir, model_name, "mask_pdlm")
            if not data:
                print(f"[skip] {config_key} p={p}: not found")
                continue
            tau, tps = extract_curve(data)
            ax.plot(tau, tps, "-o", markersize=2.5, linewidth=1.5,
                    color=p_colors[p], label=f"$p={p}$")

        ax.set_xlabel(r"Threshold $\tau$", fontsize=11)
        ax.set_xlim(0.4, 0.9)
        ax.set_ylim(1.0, 1.5)
        ax.set_title(config["title"], fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Avg tokens / step", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"Saved: {args.output}")

    # Summary table
    print(f"\n{'Model':<22} {'τ=0.2':>8} {'τ=0.3':>8} {'τ=0.5':>8}")
    print("-" * 50)
    if bd3lm_data:
        td = bd3lm_data["threshold_decode"]
        print(f"{'BD3-LM':<22} {BLOCK_SIZE/td['0.2']['avg_steps']:>8.2f} "
              f"{BLOCK_SIZE/td['0.3']['avg_steps']:>8.2f} {BLOCK_SIZE/td['0.5']['avg_steps']:>8.2f}")
    for config_key, config in CONFIGS.items():
        for p in P_VALUES:
            data = load_threshold_json(args.pdlm_dir, config["models"][p], "mask_pdlm")
            if not data:
                continue
            td = data["threshold_decode"]
            label = f"{config_key} p={p}"
            print(f"{label:<22} {BLOCK_SIZE/td['0.2']['avg_steps']:>8.2f} "
                  f"{BLOCK_SIZE/td['0.3']['avg_steps']:>8.2f} {BLOCK_SIZE/td['0.5']['avg_steps']:>8.2f}")


if __name__ == "__main__":
    main()

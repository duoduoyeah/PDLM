"""
Fetch run summaries and loss curves from WandB.

Usage:
    python wandb/fetch_runs.py                        # list all runs
    python wandb/fetch_runs.py --group mask_pdlm_d8   # filter by group
    python wandb/fetch_runs.py --group mask_pdlm_d8 --curves  # also download loss curves
"""

import argparse
import pandas as pd
import wandb

ENTITY = "lizhicuocuocuo-university-of-california-riverside"
PROJECT = "nanochat"


def get_runs(group=None):
    api = wandb.Api()
    filters = {"group": group} if group else {}
    return api.runs(f"{ENTITY}/{PROJECT}", filters=filters)


def runs_summary(runs):
    rows = []
    for r in runs:
        row = {"name": r.name, "state": r.state, "id": r.id, "group": r.config.get("wandb_group", "")}
        row.update({k: v for k, v in r.summary.items() if not k.startswith("_")})
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_curves(runs, keys=("train/loss", "eval/loss")):
    dfs = []
    for r in runs:
        print(f"  fetching {r.name} ...")
        history = pd.DataFrame(r.scan_history(keys=list(keys) + ["_step"]))
        history["run"] = r.name
        dfs.append(history)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None)
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--out", default="wandb/out")
    args = parser.parse_args()

    import os; os.makedirs(args.out, exist_ok=True)

    runs = list(get_runs(args.group))
    print(f"Found {len(runs)} runs")

    df_summary = runs_summary(runs)
    out_summary = f"{args.out}/summary.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"Summary saved to {out_summary}")
    print(df_summary[["name", "state"]].to_string(index=False))

    if args.curves:
        print("Downloading loss curves (full resolution)...")
        df_curves = fetch_curves(runs)
        out_curves = f"{args.out}/curves.csv"
        df_curves.to_csv(out_curves, index=False)
        print(f"Curves saved to {out_curves}")

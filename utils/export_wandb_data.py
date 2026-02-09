"""Export gradient tracking and eval metrics from wandb to CSV.

Usage:
    # Export a single run by name
    python utils/export_wandb_data.py --run <run_name>

    # Export all runs in a group
    python utils/export_wandb_data.py --group s1b_d4_gradient_tracking

    # Export only gradient metrics
    python utils/export_wandb_data.py --run <run_name> --prefix gradient/

    # Export only eval metrics
    python utils/export_wandb_data.py --run <run_name> --prefix eval/

    # Custom output directory
    python utils/export_wandb_data.py --run <run_name> --out_dir ./exported_data

    # Specify wandb entity (if not default)
    python utils/export_wandb_data.py --run <run_name> --entity <your_entity>
"""

import argparse
import json
import os

import pandas as pd
import wandb


def export_run(run, out_dir: str, prefix: str | None = None):
    """Export a single wandb run's history to CSV files."""
    run_name = run.name
    run_id = run.id
    print(f"Exporting run: {run_name} (id={run_id})")

    # Fetch full history (all logged keys)
    history = run.history(samples=50000)  # large enough to get all rows

    if history.empty:
        print(f"  No data found for run {run_name}, skipping.")
        return

    # Separate gradient and eval columns
    all_cols = list(history.columns)
    gradient_cols = [c for c in all_cols if c.startswith("gradient/")]
    eval_cols = [c for c in all_cols if c.startswith("eval/")]
    step_col = "step" if "step" in all_cols else "_step"

    run_dir = os.path.join(out_dir, f"{run_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    exported = []

    if (prefix is None or prefix == "gradient/") and gradient_cols:
        cols = [step_col] + sorted(gradient_cols)
        df = history[cols].dropna(how="all", subset=gradient_cols)
        path = os.path.join(run_dir, "gradient_metrics.csv")
        df.to_csv(path, index=False)
        exported.append(("gradient_metrics.csv", len(df), len(gradient_cols)))

    if (prefix is None or prefix == "eval/") and eval_cols:
        cols = [step_col] + sorted(eval_cols)
        df = history[cols].dropna(how="all", subset=eval_cols)
        path = os.path.join(run_dir, "eval_metrics.csv")
        df.to_csv(path, index=False)
        exported.append(("eval_metrics.csv", len(df), len(eval_cols)))

    # If a custom prefix is given that's neither gradient/ nor eval/, export that
    if prefix and prefix not in ("gradient/", "eval/"):
        matching_cols = [c for c in all_cols if c.startswith(prefix)]
        if matching_cols:
            cols = [step_col] + sorted(matching_cols)
            df = history[cols].dropna(how="all", subset=matching_cols)
            safe_name = prefix.rstrip("/").replace("/", "_")
            path = os.path.join(run_dir, f"{safe_name}_metrics.csv")
            df.to_csv(path, index=False)
            exported.append((f"{safe_name}_metrics.csv", len(df), len(matching_cols)))

    # Also export full history if no prefix filter
    if prefix is None:
        path = os.path.join(run_dir, "all_metrics.csv")
        history.to_csv(path, index=False)
        exported.append(("all_metrics.csv", len(history), len(all_cols)))

    for fname, nrows, ncols in exported:
        print(f"  Saved {fname}: {nrows} rows, {ncols} metric columns")

    # Save run config as JSON
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(dict(run.config), f, indent=2)
    print(f"  Saved config.json")


def main():
    parser = argparse.ArgumentParser(description="Export wandb data to CSV")
    parser.add_argument("--run", type=str, help="Run name to export")
    parser.add_argument("--group", type=str, help="Export all runs in this group")
    parser.add_argument("--project", type=str, default="nanochat", help="Wandb project name")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity (team/user). Uses default if not set.")
    parser.add_argument("--prefix", type=str, default=None, help="Only export metrics with this prefix (e.g., 'gradient/', 'eval/')")
    parser.add_argument("--out_dir", type=str, default="./wandb_export", help="Output directory")
    args = parser.parse_args()

    if not args.run and not args.group:
        parser.error("Must specify --run or --group")

    api = wandb.Api()
    entity = args.entity or api.default_entity
    project_path = f"{entity}/{args.project}"

    if args.run:
        runs = api.runs(project_path, filters={"display_name": args.run})
        runs = list(runs)
        if not runs:
            print(f"No runs found with name '{args.run}' in {project_path}")
            return
    elif args.group:
        runs = api.runs(project_path, filters={"group": args.group})
        runs = list(runs)
        if not runs:
            print(f"No runs found in group '{args.group}' in {project_path}")
            return

    print(f"Found {len(runs)} run(s) to export")
    os.makedirs(args.out_dir, exist_ok=True)

    for run in runs:
        export_run(run, args.out_dir, args.prefix)

    print(f"\nAll exports saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

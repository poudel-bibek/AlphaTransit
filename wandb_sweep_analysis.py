#!/usr/bin/env python3
"""Analyze two WandB sweeps using the WandB API."""

import wandb
import json
from collections import defaultdict

ENTITY = "bibek-poudel"
PROJECT = "transit_design"
SWEEP_IDS = ["c75xr0vy", "cekp5t6y"]

def find_reward_metrics(summary):
    """Find all reward-related metrics in summary."""
    found = {}
    for key, val in summary.items():
        key_lower = key.lower()
        if any(rk in key_lower for rk in ["reward", "return", "score", "objective"]):
            if isinstance(val, (int, float)):
                found[key] = val
    return found

def get_peak_metrics_from_history(run, metric_keys):
    """Scan full training history to find peak values for given metrics."""
    peaks = {}
    try:
        history = run.scan_history(keys=metric_keys, page_size=1000)
        for row in history:
            for key in metric_keys:
                if key in row and row[key] is not None:
                    try:
                        val = float(row[key])
                        if key not in peaks or val > peaks[key]:
                            peaks[key] = val
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        print(f"    [Warning] Could not scan history for run {run.name}: {e}")
    return peaks

def analyze_sweep(api, sweep_id):
    """Analyze a single sweep."""
    sweep_path = f"{ENTITY}/{PROJECT}/{sweep_id}"
    print(f"\n{'='*80}")
    print(f"SWEEP: {sweep_path}")
    print(f"{'='*80}")

    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"ERROR: Could not fetch sweep {sweep_path}: {e}")
        return

    # Sweep metadata
    print(f"\nSweep Name: {getattr(sweep, 'name', 'N/A')}")
    print(f"Sweep ID: {sweep.id}")
    print(f"Sweep State: {sweep.state}")

    # Sweep config (search space)
    print(f"\n--- Sweep Configuration (Search Space) ---")
    sweep_config = sweep.config
    print(json.dumps(sweep_config, indent=2, default=str))

    # Sweep metric
    metric_config = sweep_config.get("metric", {})
    metric_name = metric_config.get("name", "unknown")
    metric_goal = metric_config.get("goal", "unknown")
    print(f"\nOptimization Metric: {metric_name} (goal: {metric_goal})")

    # Runs
    runs = list(sweep.runs)
    print(f"\nNumber of Runs: {len(runs)}")

    # First pass: identify all reward-related metric keys across all runs
    all_reward_keys = set()
    for run in runs:
        reward_metrics = find_reward_metrics(run.summary)
        all_reward_keys.update(reward_metrics.keys())
    if metric_name != "unknown":
        all_reward_keys.add(metric_name)

    print(f"\nReward-related metrics found across runs: {sorted(all_reward_keys)}")

    # Second pass: get full history peaks for each run
    best_run = None
    best_metric_val = None
    run_peaks_map = {}

    print(f"\n--- Individual Runs ---")
    for i, run in enumerate(runs):
        print(f"\n  Run {i+1}/{len(runs)}: {run.name}")
        print(f"    ID: {run.id}")
        print(f"    State: {run.state}")
        print(f"    URL: {run.url}")

        # Config (hyperparameters)
        print(f"    Config (hyperparameters):")
        run_config = run.config
        for key, val in sorted(run_config.items()):
            if key.startswith("_") or key.startswith("wandb"):
                continue
            print(f"      {key}: {val}")

        # Summary metrics (last logged)
        reward_summary = find_reward_metrics(run.summary)
        print(f"    Summary reward metrics (last logged):")
        if reward_summary:
            for key, val in sorted(reward_summary.items()):
                print(f"      {key}: {val}")
        else:
            print(f"      (none found)")

        # Peak metrics from full history
        peaks = {}
        if all_reward_keys:
            peaks = get_peak_metrics_from_history(run, list(all_reward_keys))
            print(f"    Peak reward metrics (from full training history):")
            if peaks:
                for key, val in sorted(peaks.items()):
                    print(f"      {key}: {val}")
            else:
                print(f"      (no history data found)")

            # Track best run by sweep metric
            if metric_name in peaks:
                val = peaks[metric_name]
                if best_metric_val is None:
                    best_metric_val = val
                    best_run = run
                elif metric_goal == "maximize" and val > best_metric_val:
                    best_metric_val = val
                    best_run = run
                elif metric_goal == "minimize" and val < best_metric_val:
                    best_metric_val = val
                    best_run = run

        run_peaks_map[run.id] = peaks

        # Also print key non-reward summary stats
        print(f"    Other summary stats:")
        for key, val in sorted(run.summary.items()):
            if isinstance(val, (int, float)):
                key_lower = key.lower()
                if any(s in key_lower for s in ["step", "epoch", "episode", "time", "_runtime", "_step"]):
                    print(f"      {key}: {val}")

    # Best run
    print(f"\n--- Best Run (by peak {metric_name}, goal: {metric_goal}) ---")
    if best_run:
        print(f"  Name: {best_run.name}")
        print(f"  ID: {best_run.id}")
        print(f"  Peak {metric_name}: {best_metric_val}")
        print(f"  Config:")
        for key, val in sorted(best_run.config.items()):
            if not key.startswith("_") and not key.startswith("wandb"):
                print(f"    {key}: {val}")
    else:
        print(f"  Could not determine best run (metric '{metric_name}' not found in history).")

    # Also check if sweep has a best_run attribute
    try:
        sr_best = sweep.best_run()
        if sr_best:
            print(f"\n  WandB reported best run: {sr_best.name} (ID: {sr_best.id})")
            best_summary = find_reward_metrics(sr_best.summary)
            if best_summary:
                for key, val in sorted(best_summary.items()):
                    print(f"    {key}: {val}")
    except Exception:
        pass


def main():
    api = wandb.Api(timeout=60)
    for sweep_id in SWEEP_IDS:
        analyze_sweep(api, sweep_id)
    print(f"\n{'='*80}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()

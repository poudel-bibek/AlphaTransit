"""Find best policy checkpoint per training run across sweeps 7-12.

For each AT training run:
  - Read its wandb scan_history CSV
  - Find iteration with peak eval/episode_terminal_reward
  - Locate matching policy_up_X_step_Y.pth file under <sweep>/<param_value>/mcts_policies/
Output: scripts/best_policies.json
"""
import json
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEPS_BASE = REPO / "training_data/NeurIPS_results/NeurIPS_sweeps"

# Map: sweep_name -> (alpha, param_key, mcts_eval_every)
SWEEPS = {
    "nips_7_mcts_n_iter_alpha_0_3":      (0.3, "n_iter",            5),
    "nips_8_mcts_n_iter_alpha_1_0":      (1.0, "n_iter",            5),
    "nips_9_mcts_ep_per_iter_alpha_0_3": (0.3, "ep_per_iter",       5),
    "nips_10_mcts_ep_per_iter_alpha_1_0":(1.0, "ep_per_iter",       5),
    "nips_11_mcts_model_size_alpha_0_3": (0.3, "num_gat_blocks",    5),
    "nips_12_mcts_model_size_alpha_1_0": (1.0, "num_gat_blocks",    5),
}

# Each CSV name pattern: <param>_<value>_<runid>.csv
# Each param dir name: <param>_<value>/
# Need to map CSV -> param value -> param dir

results = []

for sweep_name, (alpha, param_key, eval_every) in SWEEPS.items():
    sweep_dir = SWEEPS_BASE / sweep_name
    csv_dir = sweep_dir / "wandb_data/wandb_scan_history"

    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        print(f"WARN no CSVs in {csv_dir}")
        continue

    for csv_path in csvs:
        # Parse param value from filename: e.g. n_iter_100_281nyj0i.csv -> 100
        # or ep_per_iter_8_8dnd5x8i.csv -> 8
        # or num_gat_blocks_16_phoxxit9.csv -> 16
        stem = csv_path.stem
        m = re.match(rf"^{re.escape(param_key)}_(\d+)_(\w+)$", stem)
        if not m:
            print(f"WARN cannot parse {csv_path.name}")
            continue
        param_value = int(m.group(1))
        run_id = m.group(2)

        # Read CSV, find peak eval/episode_terminal_reward
        with open(csv_path) as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        peak_reward = float("-inf")
        peak_iter = None
        peak_step = None
        for r in rows:
            v = r.get("eval/episode_terminal_reward", "")
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            # Use eval/iteration if available
            it_str = r.get("eval/iteration", "")
            try:
                it = int(float(it_str)) if it_str else None
            except ValueError:
                it = None
            step_str = r.get("_step", "")
            try:
                step = int(float(step_str)) if step_str else None
            except ValueError:
                step = None
            if fv > peak_reward:
                peak_reward = fv
                peak_iter = it
                peak_step = step

        # Find matching policy file
        # Policy dir: <sweep>/<param>_<value>/mcts_policies/policy_up_<iter>_step_<step>.pth
        # If name uses different mapping for "ep_per_iter" vs "episodes_per_iter", check
        param_dir_name = f"{param_key}_{param_value}"
        policy_dir = sweep_dir / param_dir_name / "mcts_policies"
        if not policy_dir.is_dir():
            # Try alternate dir name (some sweeps use different conventions)
            alt = sweep_dir / f"{param_key}_{param_value}" / "mcts_policies"
            print(f"WARN policy_dir missing: {policy_dir}")
            continue

        policy_files = sorted(policy_dir.glob(f"policy_up_{peak_iter}_step_*.pth")) if peak_iter else []
        if not policy_files:
            # Find closest checkpoint to peak_iter
            all_pol = sorted(policy_dir.glob("policy_up_*.pth"))
            iter_to_path = {}
            for p in all_pol:
                pm = re.match(r"policy_up_(\d+)_step_(\d+)\.pth", p.name)
                if pm:
                    iter_to_path[int(pm.group(1))] = p
            if iter_to_path and peak_iter:
                # Pick the iteration <= peak_iter that's closest
                candidates = [i for i in iter_to_path.keys() if i <= peak_iter]
                if candidates:
                    chosen_iter = max(candidates)
                    policy_path = iter_to_path[chosen_iter]
                else:
                    chosen_iter = min(iter_to_path.keys())
                    policy_path = iter_to_path[chosen_iter]
                print(f"NOTE no policy at iter {peak_iter} for {sweep_name}/{param_dir_name}; using closest available iter {chosen_iter}")
            else:
                print(f"WARN no policy files found in {policy_dir}")
                continue
        else:
            policy_path = policy_files[0]

        results.append({
            "sweep": sweep_name,
            "alpha": alpha,
            "param_key": param_key,
            "param_value": param_value,
            "run_id": run_id,
            "peak_iter": peak_iter,
            "peak_step": peak_step,
            "peak_reward": peak_reward,
            "policy_path": str(policy_path.relative_to(REPO)),
            "param_dir": str((sweep_dir / param_dir_name).relative_to(REPO)),
        })

# Sort: by alpha, then param_key, then param_value
results.sort(key=lambda r: (r["alpha"], r["param_key"], r["param_value"]))

out_path = Path(__file__).parent / "best_policies.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out_path}")
print(f"Total: {len(results)} models")

# Print summary table
print(f"\n{'sweep':45s} {'alpha':6s} {'param':16s} {'val':5s} {'iter':5s} {'reward':>8s} {'policy_exists':>14s}")
for r in results:
    exists = (REPO / r["policy_path"]).is_file()
    print(f"{r['sweep']:45s} {r['alpha']:<6.1f} {r['param_key']:16s} {r['param_value']:<5d} {r['peak_iter'] or '?':5} {r['peak_reward']:>8.2f} {str(exists):>14s}")

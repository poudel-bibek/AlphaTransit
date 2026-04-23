"""Run alpha_eval sequentially for every model in best_policies.json.

For each entry:
  - Determine training config (n_iter, episodes_per_iter, num_gat_blocks)
  - Build save_dir = .../alphatransit/<sweep_dim>/<param_value>/alphatransit
    (alpha_eval auto-appends _<timestamp>)
  - Skip if any alphatransit_<ts>/eval_results_summary.json already exists
  - Launch as subprocess (1 MCTS worker, 10 seeds, blocking)
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEST = REPO / "scripts/best_policies.json"

# Map sweep name suffix -> sweep dim folder name + per-model params
def parse_sweep(m):
    sweep = m["sweep"]
    pv = m["param_value"]
    alpha = m["alpha"]
    alpha_dir = "0_3" if alpha == 0.3 else "1_0"

    if "n_iter" in sweep:
        sweep_dir = f"nips_{'7' if alpha == 0.3 else '8'}_n_iter"
        param_dir = f"n_iter_{pv}"
        n_iter, eps, blocks = pv, 16, 4
    elif "ep_per_iter" in sweep:
        sweep_dir = f"nips_{'9' if alpha == 0.3 else '10'}_ep_per_iter"
        param_dir = f"ep_per_iter_{pv}"
        n_iter, eps, blocks = 200, pv, 4
    elif "model_size" in sweep:
        sweep_dir = f"nips_{'11' if alpha == 0.3 else '12'}_num_gat_blocks"
        param_dir = f"num_gat_blocks_{pv}"
        n_iter, eps, blocks = 200, 16, pv
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    parent = Path("training_data/NeurIPS_results") / alpha_dir / "alphatransit" / sweep_dir / param_dir
    save_dir = parent / "alphatransit"
    return {
        "alpha": alpha,
        "n_iter": n_iter,
        "eps": eps,
        "blocks": blocks,
        "policy_path": m["policy_path"],
        "save_dir": str(save_dir),
        "parent_dir": parent,
        "label": f"{sweep_dir}/{param_dir}",
    }


def already_done(parent: Path) -> bool:
    if not parent.is_dir():
        return False
    for d in parent.glob("alphatransit_*"):
        if (d / "eval_results_summary.json").is_file():
            return True
    return False


def main():
    with open(BEST) as f:
        models = json.load(f)

    print(f"Loaded {len(models)} models from best_policies.json")
    todo, skip = [], []
    for m in models:
        args = parse_sweep(m)
        if already_done(args["parent_dir"]):
            skip.append(args)
        else:
            todo.append((m, args))

    print(f"\n{len(skip)} already done, {len(todo)} to run\n")

    for i, (m, args) in enumerate(todo, 1):
        print(f"\n{'='*60}\n[{i}/{len(todo)}] {args['label']}\n{'='*60}")
        cmd = [
            ".venv/bin/python", "main.py",
            "--algorithm", "alphatransit",
            "--mode", "eval",
            "--gpu",
            "--alpha", str(args["alpha"]),
            "--apply_best_params",
            "--n_iter", str(args["n_iter"]),
            "--episodes_per_iter", str(args["eps"]),
            "--num_gat_blocks", str(args["blocks"]),
            "--saved_policy_path", args["policy_path"],
            "--num_eval_runs", "10",
            "--num_mcts_workers", "1",
            "--eval_seed_offset", "2",
            "--seed", "42",
            "--save_dir", args["save_dir"],
        ]
        print(" ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            print(f"!! eval failed (rc={rc}) for {args['label']}")
            sys.exit(rc)
    print("\nALL EVALS DONE")


if __name__ == "__main__":
    main()

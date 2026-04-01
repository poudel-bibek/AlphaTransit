"""
Wall-clock scaling experiment: Pure MCTS vs AlphaTransit.

Controlled single-route CPU benchmark. Measures wall-clock time for designing
1 transit route using both methods across MCTS n_iter values {100..500} and
alpha values {0.3, 1.0}. All runs are single-process, threads capped to 1.

Pure MCTS: sequential UXsim rollouts (~8s each per leaf evaluation).
AlphaTransit: sequential neural network forward passes on CPU (~0.1s each).

Output:
    plots/wall_clock_scaling/results.csv

Usage:
    python plots/experiment_wall_clock.py                        # Run all
    python plots/experiment_wall_clock.py --alpha 0.3            # One alpha
    python plots/experiment_wall_clock.py --n-iter 100           # One n_iter
    python plots/experiment_wall_clock.py --method alphatransit  # One method
    python plots/experiment_wall_clock.py --resume               # Skip completed
    python plots/experiment_wall_clock.py --dry-run              # Print grid only
"""

from __future__ import annotations

# ── Thread capping (must happen before any numpy/torch import) ──────────────
import os
for _var in [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
]:
    os.environ[_var] = "1"

import sys
import csv
import time
import random
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ── Project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import BEST_PARAMS, set_global_seeds
from rl.env import TransitEnv
from rl.mcts_utils import MCTSState, MCTSTree
from rl.mcts_worker import run_mcts_simulations, _create_mcts_state, get_state_tensor
from rl.mcts_inference import batch_network_forward
from rl.models import GATV2ActorCritic
from alpha import get_policy_kwargs_alpha

# ── Constants ───────────────────────────────────────────────────────────────
ALPHA_VALUES = [0.3, 1.0]
N_ITER_VALUES = [100, 200, 300, 400, 500]
SEED = 42
OUTPUT_DIR = ROOT / "plots" / "wall_clock_scaling"
RESULTS_CSV = OUTPUT_DIR / "results.csv"

NUM_WARMUP_ROUTES = 5  # Design 5 routes total; only the last is timed.
BENCHMARK_ROUTE_IDX = NUM_WARMUP_ROUTES - 1  # 0-indexed

CSV_FIELDS = ["method", "alpha", "n_iter", "seed", "total_seconds", "route_length", "num_steps", "benchmark_route_idx"]


# ── Config builder ──────────────────────────────────────────────────────────

def build_config(alpha: float, n_iter: int, workers: int = 8) -> dict:
    """Build a config dict programmatically (no argparse dependency).

    Designs NUM_WARMUP_ROUTES routes so the MCTS tree is deep enough to avoid
    dead ends; only the last route is timed for the benchmark.
    """
    at_params = BEST_PARAMS.get("alphatransit", {}).get(alpha, {})
    return {
        # Network & simulation
        "network": "bloomington",
        "alpha": alpha,
        "horizon": 10000,
        "delta_t": 1,
        "delta_n": 5,
        "bus_capacity": 40,
        "stop_duration": 60,
        "service_frequency_mode": "max_load",
        "stop_spacing": 1,
        "comfort_threshold": 1.0,
        "radius": 0.5,
        "demand_warmup": 0.15,
        # Route constraints
        "num_routes": NUM_WARMUP_ROUTES,
        "max_route_length": 14,
        "min_route_length": 2,
        "route_init": "transit_center",
        "transit_center_node": "96",
        # MCTS — keep mcts_batch_size at algorithm default (8) for fair timing
        "n_iter": n_iter,
        "c_puct": at_params.get("c_puct", 1.0),
        "mcts_batch_size": 8,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25,
        # Workers
        "num_mcts_rollout_workers": workers,
        # Seed & logging
        "seed": SEED,
        "wandb_off": True,
        "gpu": False,
        # Paths (PureMCTS needs save_dir and baseline_type)
        "save_dir": str(OUTPUT_DIR / "tmp_runs"),
        "baseline_type": "mcts",
        # Model
        "num_gat_blocks": at_params.get("num_gat_blocks", 4),
        "activation": at_params.get("activation", "tanh"),
        "concat_heads": False,
        "num_eval_runs": 1,
        "eval_seed_offset": 2,
        "mode": "baseline",
        "algorithm": "alphatransit",
        "apply_best_params": False,
        "ignore_unserved": False,
    }


# ── Local inference client (CPU-only, replaces GPU server) ──────────────────

class LocalInferenceClient:
    """CPU-only neural network inference for MCTS leaf evaluation.

    Drop-in replacement for RemoteInferenceClient that runs the GATv2
    forward pass locally on CPU instead of delegating to a GPU server.
    """

    def __init__(self, model: GATV2ActorCritic, device: str = "cpu"):
        self.model = model
        self.device = device
        self.policy_version = 0

    def infer_batch(self, payloads: list) -> list:
        """Evaluate a batch of MCTS leaf states through the neural network."""
        state_dicts = [p["state_dict"] for p in payloads]
        valid_actions_list = [p["valid_actions"] for p in payloads]
        return batch_network_forward(
            self.model, state_dicts, valid_actions_list, self.device
        )

    def infer_single(self, state_key, state_dict, valid_actions):
        """Evaluate a single MCTS leaf state."""
        return self.infer_batch([{
            "state_key": state_key,
            "state_dict": state_dict,
            "valid_actions": valid_actions,
        }])[0]


# ── Model loading ───────────────────────────────────────────────────────────

def create_alphatransit_model(config: dict, env: TransitEnv):
    """Create a randomly initialized AlphaTransit policy-value network on CPU.

    Trained weights are not needed for wall-clock benchmarking — the forward
    pass cost through GATv2 is identical regardless of weight values.
    """
    policy_kwargs = get_policy_kwargs_alpha(config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES)
    model = GATV2ActorCritic(**policy_kwargs)
    model.eval()
    print(f"  Created AlphaTransit model (random init, {sum(p.numel() for p in model.parameters()):,} params)")
    return model


# ── AlphaTransit timed evaluation ──────────────────────────────────────────

def run_alphatransit_timed(config: dict, env: TransitEnv, model: GATV2ActorCritic) -> dict:
    """Design NUM_WARMUP_ROUTES routes using AlphaTransit on CPU, time only the last.

    Routes 0..N-2 warm up the MCTS tree so the last route avoids dead ends.
    Only the wall-clock time of route N-1 is recorded.

    Returns dict with total_seconds, route_length, num_steps.
    """
    set_global_seeds(SEED)
    env.reset(seed=SEED)
    client = LocalInferenceClient(model, device="cpu")

    mcts_state = _create_mcts_state(env)
    tree = MCTSTree(mcts_state)
    last_route_steps = 0
    last_route_start = None

    while not mcts_state.is_terminal():
        valid_actions = mcts_state.get_valid_actions()
        if not valid_actions:
            mcts_state = mcts_state.force_route_end()
            tree = MCTSTree(mcts_state)
            continue

        # Start timing when we enter the last route
        if mcts_state.current_route_index == BENCHMARK_ROUTE_IDX and last_route_start is None:
            last_route_start = time.perf_counter()

        policy = run_mcts_simulations(
            env=env,
            tree=tree,
            tau=0.1,
            config=config,
            inference_client=client,
            add_noise=False,
        )

        valid_policy = policy[valid_actions]
        if valid_policy.sum() > 0:
            action = valid_actions[np.argmax(valid_policy)]
        else:
            action = valid_actions[0]

        mcts_state = mcts_state.apply_action(action)
        tree.advance(action)

        if mcts_state.current_route_index >= BENCHMARK_ROUTE_IDX:
            last_route_steps += 1

    elapsed = time.perf_counter() - last_route_start if last_route_start else 0.0
    route_length = len(mcts_state.all_routes[BENCHMARK_ROUTE_IDX]) if len(mcts_state.all_routes) > BENCHMARK_ROUTE_IDX else 0
    print(f"    AlphaTransit: {elapsed:.1f}s ({elapsed/60:.1f} min), "
          f"route_len={route_length}, steps={last_route_steps} (route {BENCHMARK_ROUTE_IDX})")
    return {"total_seconds": elapsed, "route_length": route_length, "num_steps": last_route_steps}


# ── Pure MCTS timed evaluation ──────────────────────────────────────────────

def run_pure_mcts_timed(config: dict, env: TransitEnv) -> dict:
    """Design NUM_WARMUP_ROUTES routes using Pure MCTS on CPU, time only the last.

    Uses MCTS_LOG_CSV env var to capture per-step timing from construct_path(),
    then extracts step_time_s for route_idx == BENCHMARK_ROUTE_IDX.
    No modifications to baselines/mcts.py needed.

    Returns dict with total_seconds, route_length, num_steps.
    """
    import tempfile
    from baselines.mcts import PureMCTS

    set_global_seeds(SEED)
    env.reset(seed=SEED)
    baseline = PureMCTS(env, config, num_runs=1, base_seed=SEED)
    baseline._resolve_search_params = lambda: (config["n_iter"], config["c_puct"])

    # Enable per-step CSV logging to extract last-route timing
    log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir=str(OUTPUT_DIR))
    log_path = log_file.name
    log_file.close()
    os.environ["MCTS_LOG_CSV"] = log_path

    try:
        baseline.construct_path(None)

        # Parse the CSV: sum step_time_s for the benchmark route
        import pandas as pd
        df = pd.read_csv(log_path)
        last_route = df[df["route_idx"] == BENCHMARK_ROUTE_IDX]
        elapsed = last_route["step_time_s"].astype(float).sum()
        num_steps = len(last_route)
        route_length = num_steps + 1  # steps + initial node
    finally:
        os.environ.pop("MCTS_LOG_CSV", None)
        os.unlink(log_path)

    print(f"    Pure MCTS:    {elapsed:.1f}s ({elapsed/60:.1f} min), "
          f"route_len={route_length}, steps={num_steps} (route {BENCHMARK_ROUTE_IDX})")
    return {"total_seconds": elapsed, "route_length": route_length, "num_steps": num_steps}


# ── CSV I/O ─────────────────────────────────────────────────────────────────

def load_completed() -> set:
    """Load already-completed (method, alpha, n_iter) tuples from results CSV."""
    completed = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["method"], float(row["alpha"]), int(row["n_iter"])))
    return completed


def append_result(method: str, alpha: float, n_iter: int, result: dict) -> None:
    """Append one result row to the CSV (creates file + header if needed)."""
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "method": method,
            "alpha": alpha,
            "n_iter": n_iter,
            "seed": SEED,
            "total_seconds": f"{result['total_seconds']:.2f}",
            "route_length": result["route_length"],
            "num_steps": result["num_steps"],
            "benchmark_route_idx": BENCHMARK_ROUTE_IDX,
        })


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Run the wall-clock scaling experiment."""
    parser = argparse.ArgumentParser(description="Wall-clock scaling experiment")
    parser.add_argument("--alpha", type=float, choices=[0.3, 1.0], default=None,
                        help="Run only this alpha value")
    parser.add_argument("--n-iter", type=int, choices=N_ITER_VALUES, default=None,
                        help="Run only this n_iter value")
    parser.add_argument("--method", choices=["pure_mcts", "alphatransit"], default=None,
                        help="Run only this method")
    parser.add_argument("--workers", type=int, default=8,
                        help="Rollout workers for Pure MCTS (default 8)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all grid cells as parallel subprocesses")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (method, alpha, n_iter) combos already in results.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment grid without running")
    args = parser.parse_args()

    alphas = [args.alpha] if args.alpha else ALPHA_VALUES
    n_iters = [args.n_iter] if args.n_iter else N_ITER_VALUES
    methods = [args.method] if args.method else ["pure_mcts", "alphatransit"]

    # Build experiment grid
    grid = []
    for alpha in alphas:
        for n_iter in n_iters:
            for method in methods:
                grid.append((method, alpha, n_iter))

    completed = load_completed() if args.resume else set()

    print(f"Wall-clock scaling experiment")
    print(f"  Methods:  {methods}")
    print(f"  Alphas:   {alphas}")
    print(f"  N_iters:  {n_iters}")
    print(f"  Seed:     {SEED}")
    print(f"  Routes:   {NUM_WARMUP_ROUTES} total, timing route {BENCHMARK_ROUTE_IDX} only")
    print(f"  Rollout workers: {args.workers} (Pure MCTS only)")
    print(f"  Parallel: {args.parallel}")
    print(f"  Grid:     {len(grid)} runs")
    if args.resume:
        skip = sum(1 for m, a, n in grid if (m, a, n) in completed)
        print(f"  Skipping: {skip} already completed")
    print()

    if args.dry_run:
        for method, alpha, n_iter in grid:
            status = "SKIP" if (method, alpha, n_iter) in completed else "TODO"
            print(f"  [{status}] {method:15s}  alpha={alpha}  n_iter={n_iter}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parallel mode: launch each grid cell as a separate subprocess
    if args.parallel:
        todo = [(m, a, n) for m, a, n in grid if (m, a, n) not in completed]
        if not todo:
            print("All runs already completed.")
            return
        print(f"Launching {len(todo)} parallel subprocesses...")
        procs = []
        for method, alpha, n_iter in todo:
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--method", method,
                "--alpha", str(alpha),
                "--n-iter", str(n_iter),
                "--workers", str(args.workers),
                "--resume",
            ]
            print(f"  {method} alpha={alpha} n_iter={n_iter} (pid launching...)")
            procs.append((method, alpha, n_iter, subprocess.Popen(cmd)))
        print()
        for method, alpha, n_iter, p in procs:
            p.wait()
            status = "OK" if p.returncode == 0 else f"FAIL (rc={p.returncode})"
            print(f"  [{status}] {method} alpha={alpha} n_iter={n_iter}")
        print(f"\nAll done. Results in {RESULTS_CSV}")
        return

    # Sequential mode
    envs = {}
    models = {}

    for i, (method, alpha, n_iter) in enumerate(grid):
        if (method, alpha, n_iter) in completed:
            print(f"[{i+1}/{len(grid)}] SKIP {method} alpha={alpha} n_iter={n_iter}")
            continue

        print(f"[{i+1}/{len(grid)}] {method} alpha={alpha} n_iter={n_iter}")
        config = build_config(alpha, n_iter, workers=args.workers)

        # Reuse env for same alpha
        if alpha not in envs:
            print(f"  Creating TransitEnv for alpha={alpha}...")
            envs[alpha] = TransitEnv(config)

        env = envs[alpha]
        env.config.update({"n_iter": n_iter, "c_puct": config["c_puct"]})

        if method == "alphatransit":
            if alpha not in models:
                models[alpha] = create_alphatransit_model(config, env)
            result = run_alphatransit_timed(config, env, models[alpha])
        else:
            result = run_pure_mcts_timed(config, env)

        append_result(method, alpha, n_iter, result)
        print()

    print(f"Done. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

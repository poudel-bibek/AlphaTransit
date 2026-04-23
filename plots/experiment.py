"""
Wall-clock scaling experiment: Pure MCTS vs AlphaTransit.

Measures wall-clock time for designing 1 transit route (14 nodes) using both
methods across MCTS n_iter values {100..500} and alpha values {0.3, 1.0}.
All runs single-process, threads capped to 1.

If a run produces a route shorter than 14 nodes, it retries with the next seed.

Output:
    figures/wall_clock_scaling/results.csv

Usage:
    python plots/main.py experiment
    python plots/main.py experiment --alpha 0.3
    python plots/main.py experiment --n-iter 100
    python plots/main.py experiment --method alphatransit
    python plots/main.py experiment --resume
    python plots/main.py experiment --dry-run
"""

from __future__ import annotations

import os

for _var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[_var] = "1"

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .common import FIGURES_DIR
except ImportError:
    from common import FIGURES_DIR


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALPHA_VALUES = [0.3, 1.0]
N_ITER_VALUES = [100, 200, 300, 400, 500]
TARGET_ROUTE_LENGTH = 14
MAX_RETRIES = 50
BASE_SEED = 42
ROUTES_BY_ALPHA = {0.3: 1, 1.0: 3}
OUTPUT_DIR = FIGURES_DIR / "wall_clock_scaling"
RESULTS_CSV = OUTPUT_DIR / "results.csv"

CSV_FIELDS = [
    "method",
    "alpha",
    "n_iter",
    "seed",
    "total_seconds",
    "route_length",
    "num_steps",
]

_RUNTIME_LOADED = False


def _load_runtime_dependencies() -> None:
    global _RUNTIME_LOADED
    global BEST_PARAMS, set_global_seeds
    global TransitEnv, batch_network_forward, MCTSTree
    global _create_mcts_state, run_mcts_simulations
    global GATV2ActorCritic, get_policy_kwargs_alpha

    if _RUNTIME_LOADED:
        return

    from alpha import get_policy_kwargs_alpha as _get_policy_kwargs_alpha
    from config import BEST_PARAMS as _BEST_PARAMS, set_global_seeds as _set_global_seeds
    from rl.env import TransitEnv as _TransitEnv
    from rl.mcts_inference import batch_network_forward as _batch_network_forward
    from rl.mcts_utils import MCTSTree as _MCTSTree
    from rl.mcts_worker import _create_mcts_state as __create_mcts_state, run_mcts_simulations as _run_mcts_simulations
    from rl.models import GATV2ActorCritic as _GATV2ActorCritic

    BEST_PARAMS = _BEST_PARAMS
    set_global_seeds = _set_global_seeds
    TransitEnv = _TransitEnv
    batch_network_forward = _batch_network_forward
    MCTSTree = _MCTSTree
    _create_mcts_state = __create_mcts_state
    run_mcts_simulations = _run_mcts_simulations
    GATV2ActorCritic = _GATV2ActorCritic
    get_policy_kwargs_alpha = _get_policy_kwargs_alpha
    _RUNTIME_LOADED = True


def build_config(alpha: float, n_iter: int, seed: int, workers: int = 8) -> dict:
    """Build a config dict. Always designs 1 route."""
    _load_runtime_dependencies()
    at_params = BEST_PARAMS.get("alphatransit", {}).get(alpha, {})
    return {
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
        "num_routes": ROUTES_BY_ALPHA[alpha],
        "max_route_length": TARGET_ROUTE_LENGTH,
        "min_route_length": 2,
        "route_init": "transit_center",
        "transit_center_node": "96",
        "n_iter": n_iter,
        "c_puct": at_params.get("c_puct", 1.0),
        "mcts_batch_size": 8,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25,
        "num_mcts_rollout_workers": workers,
        "seed": seed,
        "wandb_off": True,
        "gpu": False,
        "save_dir": str(OUTPUT_DIR / "tmp_runs"),
        "baseline_type": "mcts",
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


class LocalInferenceClient:
    """CPU-only neural network inference for MCTS leaf evaluation."""

    def __init__(self, model: GATV2ActorCritic, device: str = "cpu"):
        self.model = model
        self.device = device
        self.policy_version = 0

    def infer_batch(self, payloads: list) -> list:
        _load_runtime_dependencies()
        state_dicts = [payload["state_dict"] for payload in payloads]
        valid_actions_list = [payload["valid_actions"] for payload in payloads]
        return batch_network_forward(
            self.model,
            state_dicts,
            valid_actions_list,
            self.device,
        )

    def infer_single(self, state_key, state_dict, valid_actions):
        return self.infer_batch(
            [
                {
                    "state_key": state_key,
                    "state_dict": state_dict,
                    "valid_actions": valid_actions,
                }
            ]
        )[0]


def create_alphatransit_model(config: dict, env: TransitEnv):
    """Create a randomly initialized AlphaTransit model on CPU."""
    _load_runtime_dependencies()
    policy_kwargs = get_policy_kwargs_alpha(
        config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES
    )
    model = GATV2ActorCritic(**policy_kwargs)
    model.eval()
    print(
        f"  Created AlphaTransit model ({sum(param.numel() for param in model.parameters()):,} params)"
    )
    return model


def run_alphatransit_timed(
    config: dict,
    env: TransitEnv,
    model: GATV2ActorCritic,
) -> dict:
    """Design route(s) using AlphaTransit on CPU, time only the last route."""
    _load_runtime_dependencies()
    benchmark_idx = config["num_routes"] - 1
    seed = config["seed"]
    set_global_seeds(seed)
    env.reset(seed=seed)
    client = LocalInferenceClient(model, device="cpu")

    mcts_state = _create_mcts_state(env)
    tree = MCTSTree(mcts_state)
    num_steps = 0
    t0 = None

    while not mcts_state.is_terminal():
        valid_actions = mcts_state.get_valid_actions()
        if not valid_actions:
            mcts_state = mcts_state.force_route_end()
            tree = MCTSTree(mcts_state)
            continue

        if mcts_state.current_route_index == benchmark_idx and t0 is None:
            t0 = time.perf_counter()

        policy = run_mcts_simulations(
            env=env,
            tree=tree,
            tau=0.1,
            config=config,
            inference_client=client,
            add_noise=False,
        )

        valid_policy = policy[valid_actions]
        action = (
            valid_actions[np.argmax(valid_policy)]
            if valid_policy.sum() > 0
            else valid_actions[0]
        )
        mcts_state = mcts_state.apply_action(action)
        tree.advance(action)
        if mcts_state.current_route_index >= benchmark_idx:
            num_steps += 1

    elapsed = time.perf_counter() - t0 if t0 else 0.0
    route_length = (
        len(mcts_state.all_routes[benchmark_idx])
        if len(mcts_state.all_routes) > benchmark_idx
        else 0
    )
    return {
        "total_seconds": elapsed,
        "route_length": route_length,
        "num_steps": num_steps,
    }


def run_pure_mcts_timed(config: dict, env: TransitEnv) -> dict:
    """Design route(s) using Pure MCTS on CPU, time only the last route."""
    _load_runtime_dependencies()
    benchmark_idx = config["num_routes"] - 1
    import tempfile

    from baselines.mcts import PureMCTS

    seed = config["seed"]
    set_global_seeds(seed)
    env.reset(seed=seed)
    baseline = PureMCTS(env, config, num_runs=1, base_seed=seed)

    log_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        dir=str(OUTPUT_DIR),
    )
    log_path = log_file.name
    log_file.close()
    os.environ["MCTS_LOG_CSV"] = log_path

    try:
        baseline.construct_path(None)

        df = np.recfromcsv(log_path, encoding="utf-8")
        route_mask = df["route_idx"] == benchmark_idx
        step_times = np.asarray(df["step_time_s"][route_mask], dtype=float)
        elapsed = step_times.sum()
        num_steps = int(route_mask.sum())
        route_length = num_steps + 1
    finally:
        os.environ.pop("MCTS_LOG_CSV", None)
        os.unlink(log_path)

    return {
        "total_seconds": elapsed,
        "route_length": route_length,
        "num_steps": num_steps,
    }


def load_completed() -> set[tuple[str, float, int]]:
    """Load already-completed (method, alpha, n_iter) tuples from results CSV."""
    completed: set[tuple[str, float, int]] = set()
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                completed.add((row["method"], float(row["alpha"]), int(row["n_iter"])))
    return completed


def append_result(
    method: str,
    alpha: float,
    n_iter: int,
    seed: int,
    result: dict,
) -> None:
    """Append one result row to the CSV."""
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "method": method,
                "alpha": alpha,
                "n_iter": n_iter,
                "seed": seed,
                "total_seconds": f"{result['total_seconds']:.2f}",
                "route_length": result["route_length"],
                "num_steps": result["num_steps"],
            }
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Wall-clock scaling experiment")
    parser.add_argument("--alpha", type=float, choices=[0.3, 1.0])
    parser.add_argument("--n-iter", type=int, choices=N_ITER_VALUES)
    parser.add_argument("--method", choices=["pure_mcts", "alphatransit"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    alphas = [args.alpha] if args.alpha else ALPHA_VALUES
    n_iters = [args.n_iter] if args.n_iter else N_ITER_VALUES
    methods = [args.method] if args.method else ["pure_mcts", "alphatransit"]

    grid = [(method, alpha, n_iter) for alpha in alphas for n_iter in n_iters for method in methods]
    completed = load_completed() if args.resume else set()

    print("Wall-clock scaling experiment")
    print(f"  Grid: {len(grid)} runs ({methods} x alpha={alphas} x n_iter={n_iters})")
    print(f"  Target route length: {TARGET_ROUTE_LENGTH} (retry up to {MAX_RETRIES}x)")
    print(f"  Workers: {args.workers}")
    if args.resume:
        print(f"  Skipping: {sum(1 for item in grid if item in completed)} completed")
    print()

    if args.dry_run:
        for method, alpha, n_iter in grid:
            status = "SKIP" if (method, alpha, n_iter) in completed else "TODO"
            print(f"  [{status}] {method:15s}  alpha={alpha}  n_iter={n_iter}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    envs: dict[float, TransitEnv] = {}
    models: dict[float, GATV2ActorCritic] = {}

    for index, (method, alpha, n_iter) in enumerate(grid, start=1):
        if (method, alpha, n_iter) in completed:
            print(f"[{index}/{len(grid)}] SKIP {method} alpha={alpha} n_iter={n_iter}")
            continue

        print(f"[{index}/{len(grid)}] {method} alpha={alpha} n_iter={n_iter}")

        if alpha not in envs:
            print(f"  Creating TransitEnv for alpha={alpha}...")
            envs[alpha] = TransitEnv(build_config(alpha, n_iter, BASE_SEED, args.workers))

        env = envs[alpha]

        for attempt in range(MAX_RETRIES):
            seed = BASE_SEED + attempt
            config = build_config(alpha, n_iter, seed, args.workers)
            env.config.update(
                {"n_iter": n_iter, "c_puct": config["c_puct"], "seed": seed}
            )

            if method == "alphatransit":
                if alpha not in models:
                    models[alpha] = create_alphatransit_model(config, env)
                result = run_alphatransit_timed(config, env, models[alpha])
            else:
                result = run_pure_mcts_timed(config, env)

            route_length = result["route_length"]
            if route_length == TARGET_ROUTE_LENGTH:
                print(
                    f"    {method}: {result['total_seconds']:.1f}s, "
                    f"route_len={route_length}, steps={result['num_steps']}, seed={seed}"
                )
                append_result(method, alpha, n_iter, seed, result)
                break
            print(
                f"    attempt {attempt + 1}: route_len={route_length} != {TARGET_ROUTE_LENGTH}, "
                f"retrying seed={seed + 1}..."
            )
        else:
            print(
                f"    WARNING: no {TARGET_ROUTE_LENGTH}-node route after {MAX_RETRIES} attempts"
            )
        print()

    print(f"Done. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

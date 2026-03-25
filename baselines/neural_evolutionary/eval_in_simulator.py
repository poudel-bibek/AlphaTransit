"""
Evaluate Holliday et al. NEA/EA routes in AlphaTransit's UXsim simulator.

Takes route pickle files produced by bee_colony.py and evaluates them
through the same simulation pipeline used for all other AlphaTransit baselines.

Usage:
    conda activate icml_rebuttal
    cd /home/bibek/Desktop/ICML/AlphaTransit
    python baselines/neural_evolutionary/eval_in_simulator.py --routes_pkl <path_to_routes.pkl> --alpha 0.3
"""

import argparse
import pickle
import json
import sys
import os
import csv
import numpy as np

# Add AlphaTransit root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ALPHA_ROOT)


def load_node_mapping(mapping_path):
    """Load Mumford index -> AlphaTransit node name mapping."""
    idx_to_name = {}
    with open(mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["mumford_idx"])
            name = row["original_name"]
            idx_to_name[idx] = name
    return idx_to_name


def load_holliday_routes(pkl_path, idx_to_name):
    """Load routes from Holliday's pickle format and convert to AlphaTransit format.

    Holliday stores routes as a torch tensor of shape (n_routes, max_len)
    with -1 padding, where values are 0-indexed Mumford node indices.

    Returns: List[List[str]] — AlphaTransit route format.
    """
    import torch
    with open(pkl_path, "rb") as f:
        routes_tensor = pickle.load(f)

    if isinstance(routes_tensor, torch.Tensor):
        routes_tensor = routes_tensor.numpy()

    routes = []
    for i in range(routes_tensor.shape[0]):
        route_indices = routes_tensor[i]
        # Filter out padding (-1)
        valid = route_indices[route_indices >= 0]
        route_names = [idx_to_name[int(idx)] for idx in valid]
        routes.append(route_names)

    return routes


def evaluate_routes(routes, alpha, num_seeds=5, base_seed=42):
    """Evaluate routes in AlphaTransit's UXsim simulator."""
    from config import get_config
    from rl.env import TransitEnv
    from baselines.utils import simulate_baseline_routes, create_main_save_dir

    # Build config
    sys.argv = [
        "eval_in_simulator.py",
        "--mode", "baseline",
        "--baseline_type", "real_world",
        "--alpha", str(alpha),
        "--network", "bloomington",
        "--num_eval_runs", str(num_seeds),
        "--seed", str(base_seed),
    ]
    config = get_config()
    config["alpha"] = alpha

    env = TransitEnv(config)

    results = []
    for seed_offset in range(num_seeds):
        seed = base_seed + seed_offset * 2
        np.random.seed(seed)

        save_dir = os.path.join(ALPHA_ROOT, "training_data", "NEA_eval",
                                f"alpha_{alpha}", f"seed_{seed}")
        os.makedirs(save_dir, exist_ok=True)

        sim_result = simulate_baseline_routes(
            env, config, routes,
            img_dir=save_dir,
            baseline_save_dir=save_dir
        )
        results.append(sim_result)
        print(f"  Seed {seed}: service_rate={sim_result.get('service_rate', 'N/A'):.3f}, "
              f"wait={sim_result.get('avg_wait_time', 'N/A'):.1f}s")

    # Aggregate
    metric_names = [
        "service_rate", "avg_wait_time", "avg_movement_time",
        "fleet_size", "transfer_rate", "route_efficiency",
        "bus_utilization"
    ]
    aggregated = {}
    for m in metric_names:
        values = [r.get(m, 0) for r in results if m in r]
        if values:
            aggregated[m] = {
                "avg": np.mean(values),
                "std": np.std(values),
            }

    return results, aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate NEA routes in AlphaTransit simulator")
    parser.add_argument("--routes_pkl", required=True, help="Path to routes pickle file")
    parser.add_argument("--alpha", type=float, default=0.3, help="Modal split alpha")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of evaluation seeds")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    # Load node mapping
    mapping_path = os.path.join(SCRIPT_DIR, "datasets", "bloomington", "node_mapping.csv")
    idx_to_name = load_node_mapping(mapping_path)

    # Load routes
    routes = load_holliday_routes(args.routes_pkl, idx_to_name)
    print(f"Loaded {len(routes)} routes from {args.routes_pkl}")
    for i, r in enumerate(routes):
        print(f"  Route {i+1}: {len(r)} stops — {r[:3]}...{r[-3:]}")

    # Evaluate
    print(f"\nEvaluating with alpha={args.alpha}, {args.num_seeds} seeds...")
    results, aggregated = evaluate_routes(routes, args.alpha, args.num_seeds, args.base_seed)

    print(f"\n{'='*60}")
    print(f"NEA Routes — Alpha={args.alpha}")
    print(f"{'='*60}")
    for metric, stats in aggregated.items():
        print(f"  {metric}: {stats['avg']:.4f} ± {stats['std']:.4f}")

    # Save results
    out_path = os.path.join(ALPHA_ROOT, "training_data", "NEA_eval",
                            f"nea_results_alpha_{args.alpha}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "alpha": args.alpha,
            "routes_pkl": args.routes_pkl,
            "num_seeds": args.num_seeds,
            "aggregated": {k: {"avg": float(v["avg"]), "std": float(v["std"])}
                          for k, v in aggregated.items()},
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

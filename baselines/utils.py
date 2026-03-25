"""
Shared utilities for all baselines.

Includes simulation evaluation, visualization, result aggregation,
and seed management functions used by heuristic, genetic, and
neural evolutionary baselines.
"""

import os
import csv
import json
import pickle
import shutil
import numpy as np
import random
import torch
from datetime import datetime
import multiprocessing as mp
mp_ctx = mp.get_context('spawn')
from scipy.special import softmax
from rl.env_utils import (
    plot_network_and_demand,
    plot_network_demand_and_path,
    initialize_route,
    aggregate_results,
    write_results_summary,
    ensure_eval_step_update_dir,
    make_seed_output_dir,
    save_routes_json,
)
def set_global_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_main_save_dir(config):
    """
    Create main save directory for baseline results.
    """
    now = datetime.now()
    main_save_dir = os.path.join(
        config.get("save_dir"),
        f"{config.get('baseline_type')}_{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}"
    )
    os.makedirs(main_save_dir, exist_ok=True)
    return main_save_dir, main_save_dir

def create_initial_network_plot(env, config, img_dir):
    """
    Create initial network + demand visualization, similar to RL training.
    """
    # Build a temporary world for initial visualization
    temp_world = env.build_world(config.get("network"))
    temp_world.name = config.get("network")  # Set proper network name
    # Load demand data for visualization
    env.load_demand_for_plotting(temp_world)
    output_path = os.path.join(img_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path)

def create_path_visualization(env, config, routes, img_dir):
    """
    Create network + path overlay visualization for multiple routes.
    """
    output_path = os.path.join(img_dir, f"{config.get('baseline_type')}_final_routes.png")
    plot_network_demand_and_path(env.world, routes, output_path)
    print(f"Routes visualization saved to: {output_path}")

def create_fancy_animations(env, config, baseline_save_dir):
    """
    """
    all_vehicles_anim_path = os.path.join(baseline_save_dir, f"{config.get('baseline_type')}_anim_all_vehicles.gif")
    env.world.analyzer.network_fancy(
        animation_speed_inverse=10,
        figsize=11,
        sample_ratio=1.0,
        interval=5,
        trace_length=5,
        network_font_size=11,
        antialiasing=False,
        file_name=all_vehicles_anim_path,
        save_as_mp4=False,
        # bus_only=True
    )
    print(f"All vehicles animation saved to: {all_vehicles_anim_path}")

def simulate_baseline_routes(env, config, routes, img_dir, baseline_save_dir):
    """
    Shared across all baselines.
    Simulate routes to get the performance results.
    Includes visualization and animation generation similar to RL training.
    """
    # Create initial network visualization before simulation
    create_initial_network_plot(env, config, img_dir)

    # For baselines, we get the completed routes and then simulate the final.
    env.all_routes = routes
    env.current_route = []  # No current route being built for baselines
    env.current_route_index = len(routes)  # All routes completed
    env.is_baseline = True  # Set baseline flag

    # Build world with correct demand allocation
    env.world = env.build_world(config.get("network"))

    # Apply buses using env's method (this will handle all routes)
    env._apply_action()  # Will automatically skip route completion checks due to is_baseline flag

    # Run simulation and get metrics
    sim_result = env._step_until(env.horizon, print_metrics=True)

    # Generate visualizations using standalone functions
    create_path_visualization(env, config, routes, img_dir)

    if config["save_animations"]:
        create_fancy_animations(env, config, baseline_save_dir)

    save_routes_json(baseline_save_dir, routes)
    return {
        'sim_result': sim_result
    }

def print_results(results_list, aggregated):
    eval_metrics = {
        # Waiting metrics (seconds)
        'total_wait_completed': 'seconds',
        'total_wait_ongoing': 'seconds',
        'total_wait_unserved': 'seconds',
        'sim_end_waiting_passengers_count': 'count',

        # Movement / travel metrics (seconds)
        'total_movement_completed': 'seconds',
        'total_movement_ongoing': 'seconds',
        'total_travel_completed': 'seconds',
        'total_travel_ongoing': 'seconds',

        # Per-passenger averages (seconds)
        'avg_wait_time_completed': 'seconds',
        'avg_wait_time_ongoing': 'seconds',
        'avg_movement_time_completed': 'seconds',
        'avg_movement_time_ongoing': 'seconds',
        'avg_travel_time_completed': 'seconds',
        'avg_travel_time_ongoing': 'seconds',

        # Distributions (seconds)
        'waiting_time_dstr': 'distribution',
        'movement_time_dstr': 'distribution',
        'travel_time_dstr': 'distribution',

        # Passenger counts
        'completed_passengers': 'count',
        'ongoing_passengers': 'count',
        'total_onboarded_count': 'count',
        'wanting_to_onboard': 'count',

        # Demand / reward components
        'demand_coverage_potential': '%',
        'demand_coverage_actual': '%',
        'service_rate': '%',
        'completed_rate': '%',
        'transfer_rate': '%',

        # Route/network metrics
        'route_overlap_ratio': 'ratio',
        'route_length': 'meters',
        'bus_utilization': '%',
        'average_bus_speed': 'm/s',
        'fleet_size': 'count',
        'route_efficiency': 'passengers/km',
        'node_coverage': '%',
    }

    scalar_series = aggregated.get('_scalar_series', {})
    per_run_stats = aggregated.get('_per_run_stats', [])

    print("\n=== Individual Run Results ===")
    max_key_len = max(len(key) for key in eval_metrics.keys())

    for i, res in enumerate(results_list, 1):
        print(f"Run {i}:")
        sim = res['sim_result']
        served = sim['completed_passengers'] + sim['ongoing_passengers']
        wait_seconds = sim['total_wait_completed'] + sim['total_wait_ongoing']
        travel_seconds = sim['total_travel_completed'] + sim['total_travel_ongoing']

        for key in eval_metrics:
            val = sim[key]
            if isinstance(val, (int, float)):
                val_str = f"{val:.2f}" if isinstance(val, float) else f"{val}"
                print(f"  {key:<{max_key_len}} : {val_str} ({eval_metrics[key]})")
            elif isinstance(val, (list, tuple)):
                print(f"  {key:<{max_key_len}} : n={len(val)} ({eval_metrics[key]})")

        if served > 0:
            print(f"  avg_wait_time_served         : {wait_seconds/served:.2f} (seconds)")
            print(f"  avg_travel_time_served       : {travel_seconds/served:.2f} (seconds)")
        else:
            print("  avg_wait_time_served         : 0.00 (seconds)")
            print("  avg_travel_time_served       : 0.00 (seconds)")
        print("---")

    print("\n=== Averaged Results ===")
    for key, values in scalar_series.items():
        if key not in eval_metrics or not values:
            continue
        str_values = [f"{v:.2f}" for v in values]
        avg_val = aggregated.get(key)
        avg_str = f"{avg_val:.2f}" if avg_val is not None else "N/A"
        print(f"  {key:<{max_key_len}} : ({' + '.join(str_values)}) / {len(values)} = {avg_str} ({eval_metrics[key]})")

    total_wait_seconds = sum(stat['combined_wait_seconds'] for stat in per_run_stats)
    total_travel_seconds = sum(stat['combined_travel_seconds'] for stat in per_run_stats)
    total_served = sum(stat['served_passengers'] for stat in per_run_stats)

    combined_avg_wait_minutes = (total_wait_seconds / total_served) / 60 if total_served > 0 else 0.0
    combined_avg_travel_minutes = (total_travel_seconds / total_served) / 60 if total_served > 0 else 0.0

    route_eff = aggregated.get('route_efficiency')
    fleet_size = aggregated.get('fleet_size')
    bus_util = aggregated.get('bus_utilization')
    service_rate = aggregated.get('service_rate')
    transfer_rate = aggregated.get('transfer_rate')

    for dist_key in ['waiting_time_dstr', 'movement_time_dstr', 'travel_time_dstr']:
        combined_stats = [k for k in aggregated.keys() if k.startswith(f'{dist_key}_combined')]
        if not combined_stats:
            continue
        print(f"\n  === Combined {dist_key.replace('_', ' ').title()} Statistics ===")
        for stat_key in combined_stats:
            stat_label = stat_key.replace(f'{dist_key}_combined_', '').replace('_', ' ').title()
            stat_value = aggregated[stat_key]
            if 'passengers' in stat_key:
                print(f"  {stat_label:<{max_key_len-2}} : {stat_value}")
            else:
                print(f"  {stat_label:<{max_key_len-2}} : {stat_value:.2f} seconds")

    print("\n")
    print(f"& ${service_rate:.2f}$ & ${combined_avg_wait_minutes:.2f}$ & ${transfer_rate:.2f}$ & ${combined_avg_travel_minutes:.2f}$ & ${route_eff:.2f}$ & ${fleet_size:.0f}$ & ${bus_util:.0f}$")

def execute_runs(baseline, num_runs, base_seed):
    """
    Execute multiple runs for a baseline.
    """
    eval_offset = baseline.config["eval_seed_offset"]
    starting_seed = base_seed 
    results = []
    for run in range(num_runs):
        current_seed = starting_seed + (run * eval_offset)
        set_global_seeds(current_seed)
        print(f"\n=== Run {run+1} (Seed: {current_seed}) ===")
        
        # Create seed-specific directory
        seed_dir, img_dir = make_seed_output_dir(baseline.eval_root_dir, current_seed)

        # Reset env with current seed so simulation varies per run
        state, _ = baseline.env.reset(seed=current_seed)

        routes = baseline.construct_path(state)
        print(f"Routes: {routes}")

        result = simulate_baseline_routes(baseline.env, baseline.config, routes, img_dir, seed_dir)
        results.append(result)
    
    aggregated = aggregate_results(results)
    print_results(results, aggregated)
    return results, aggregated


# ---------------------------------------------------------------------------
# Holliday et al. route loading and evaluation utilities
# ---------------------------------------------------------------------------

NEURAL_EVOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_evolutionary")


def load_node_mapping(mapping_path=None):
    """Load Mumford index -> AlphaTransit node name mapping."""
    if mapping_path is None:
        mapping_path = os.path.join(NEURAL_EVOL_DIR, "datasets", "bloomington", "node_mapping.csv")
    idx_to_name = {}
    with open(mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["mumford_idx"])
            name = row["original_name"]
            idx_to_name[idx] = name
    return idx_to_name


def load_holliday_routes(pkl_path, idx_to_name=None):
    """Load routes from Holliday's pickle format and convert to AlphaTransit format.

    Holliday stores routes as a torch tensor of shape (n_routes, max_len)
    with -1 padding, where values are 0-indexed Mumford node indices.

    Returns: List[List[str]] — AlphaTransit route format.
    """
    if idx_to_name is None:
        idx_to_name = load_node_mapping()

    with open(pkl_path, "rb") as f:
        routes_tensor = pickle.load(f)

    # Handle list[tensor] wrapper produced by some Holliday scripts
    if isinstance(routes_tensor, list):
        routes_tensor = routes_tensor[0]

    if isinstance(routes_tensor, torch.Tensor):
        routes_tensor = routes_tensor.numpy()

    routes = []
    for i in range(routes_tensor.shape[0]):
        route_indices = routes_tensor[i]
        valid = route_indices[route_indices >= 0]
        route_names = [idx_to_name[int(idx)] for idx in valid]
        routes.append(route_names)

    return routes


def evaluate_holliday_routes(routes, alpha, num_seeds=5, base_seed=42):
    """Evaluate Holliday et al. routes in AlphaTransit's UXsim simulator.

    Uses the same simulation pipeline as all other baselines.
    Returns (results_list, aggregated_dict).
    """
    import sys
    from config import get_config
    from rl.env import TransitEnv

    # Build config via argparse
    original_argv = sys.argv
    sys.argv = [
        "eval_holliday",
        "--mode", "baseline",
        "--baseline_type", "real_world",
        "--alpha", str(alpha),
        "--network", "bloomington",
        "--num_eval_runs", str(num_seeds),
        "--seed", str(base_seed),
    ]
    config = get_config()
    sys.argv = original_argv

    config["alpha"] = alpha
    env = TransitEnv(config)

    results = []
    for seed_offset in range(num_seeds):
        seed = base_seed + seed_offset * 2
        np.random.seed(seed)

        save_dir = os.path.join(
            config.get("save_dir", "./training_data"), "NEA_eval",
            f"alpha_{alpha}", f"seed_{seed}")
        os.makedirs(save_dir, exist_ok=True)

        sim_result = simulate_baseline_routes(
            env, config, routes, img_dir=save_dir, baseline_save_dir=save_dir)
        results.append(sim_result)
        print(f"  Seed {seed}: service_rate={sim_result.get('service_rate', 'N/A')}")

    aggregated = aggregate_results(results)
    print_results(results, aggregated)
    return results, aggregated

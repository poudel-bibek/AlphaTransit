"""
Heuristic Baselines: 
- Neighbourhood search algorithms
- Baselines mostly setup in a way that "to construct the path" we dont need to simulate. 
- However, to get the performance results on the path, we need to simulate.
- Repurposing the RL env setup for the baselines as well.

0. Random baseline: 
    - Random selection of next node.

1. Greedy Demand Coverage: 
   - Build route by greedily selecting the nodes that maximize the immediate demand coverage.

2. Greedy Shortest Path: 
    - Greedily select the node that is closest to the current path.

3. Greedy Reward Maximization: 
    - Greedily select the node that maximizes the immediate reward.

---------------

These baselines still need to respect constraints such as: 
- MAX_ROUTE_LENGTH
- SERVICE_FREQUENCY (This will be taken care by the sim bus handler)
- alpha (facor to determine the % of demand allocated to bus)
- For each baseline, we need to form a path and then simulate to get results.

---------------
Additional notes (ignore)
- In case the immediate next node does not add any new demand coverage (because trips are not completed by just adding 1 node)
- These baselines have the same results across seeds: 
    - Greedy Shortest Path
    - Greedy Demand Coverage
"""

import os
import json
import numpy as np
import random
import torch
from datetime import datetime
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
    # Baselines have no updates; standardize on (update='baseline', steps=0)
    eval_root = ensure_eval_step_update_dir(main_save_dir, update="baseline", steps=0, folder_name="eval_results")
    print(f"Baseline results will be saved to: {eval_root}")
    return main_save_dir, eval_root

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
    print(f"Initial network plot saved to: {output_path}")

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
        
        # Reset env
        state, _ = baseline.env.reset()

        routes = baseline.construct_path(state)
        print(f"Routes: {routes}")

        result = simulate_baseline_routes(baseline.env, baseline.config, routes, img_dir, seed_dir)
        results.append(result)
    
    aggregated = aggregate_results(results)
    print_results(results, aggregated)
    return results, aggregated

#####################################
# Heuristic Baselines: 
#####################################

class RandomWalk:
    """
    Random Neighbor Baseline.
    """
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)

    def run(self):
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir, 'eval_results_summary.json')
        return results, aggregated

    def construct_path(self, state):
        """
        Multi-route version:
        Algorithm:
        Step 1: Initialize first route (same initialization as RL)
        Step 2: For each route, select random neighbors until max length or no valid moves
        Step 3: Move to next route and repeat
        Step 4: Return all completed routes
        """
        
        current_route_index = 0
        while current_route_index < self.env.NUM_ROUTES:
            print(f"\n=== Building Route {current_route_index + 1} ===")

            # Initialize current route (exactly like RL env initialization)
            current_route = initialize_route(self.env)

            while len(current_route) < self.env.MAX_ROUTE_LENGTH:
                # Get current frontier
                frontier = current_route[-1]

                # Get valid neighbors (adjacent and not already in current route)
                route_set = set(current_route)
                valid_neighbors = [n for n in self.env.adj[frontier] if n not in route_set]
                if not valid_neighbors:
                    print(f"No valid neighbors found for frontier: {frontier}")
                    break

                # Select random neighbor
                next_node = random.choice(valid_neighbors)

                current_route.append(next_node)
                print(f"Route {current_route_index + 1} so far: {current_route}")

            # Add completed route to all_routes
            self.env.all_routes.append(current_route)
            print(f"Route {current_route_index + 1} completed: {current_route}")
            current_route_index += 1        

        print(f"\nAll routes completed: {self.env.all_routes}")
        return self.env.all_routes

class DemandCoverage:
    """
    Build route by greedily selecting the nodes that maximize the immediate demand coverage.
    """
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)

    def run(self):
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir, 'eval_results_summary.json')
        return results, aggregated

    def construct_path(self, state):
        """
        Algorithm:
        Step 1: Initialization (same initialization as RL), happens in main at reset.
        Step 2: At each step, from valid neighboring nodes
            - Calculate the incremental demand that would be served by adding that node to the path.
            - Consider the combined demand (d_out + d_in) for that node and select the one with the highest.
        Step 3: Repeat step 2 until reaching the max path length.
        """
        current_route_index = 0
        while current_route_index < self.env.NUM_ROUTES:
            print(f"\n=== Building Route {current_route_index + 1} ===")

            # Initialize current route (exactly like RL env initialization)
            current_route = initialize_route(self.env)
            self.env.current_route = current_route.copy()

            while len(current_route) < self.env.MAX_ROUTE_LENGTH:

                # Get current frontier
                frontier = current_route[-1]

                # Get valid neighbors (adjacent and not already in path)
                route_set = set(current_route)
                valid_neighbors = [n for n in self.env.adj[frontier] if n not in route_set]
                # print(f"Frontier: {frontier}, Valid neighbors: {valid_neighbors}")
                if not valid_neighbors:
                    break

                # Convert to indices for demand calculation
                route_indices = np.array([self.env.node_to_idx[node] for node in current_route])

                # For each valid neighbor, compute incremental demand (d_out_path + d_in_path)
                best_score = -np.inf
                best_node = None

                for neighbor in valid_neighbors:
                    neigh_idx = self.env.node_to_idx[neighbor]

                    # Incremental outgoing: flows from neighbor to current path nodes
                    d_out_inc = self.env.od_matrix[neigh_idx, route_indices].sum()

                    # Incremental incoming: flows from current path nodes to neighbor
                    d_in_inc = self.env.od_matrix[route_indices, neigh_idx].sum()

                    score = d_out_inc + d_in_inc

                    if score > best_score:
                        best_score = score
                        best_node = neighbor

                current_route.append(best_node)
                print(f"Route {current_route_index + 1} so far: {current_route}")

            self.env.all_routes.append(current_route)
            print(f"Route {current_route_index + 1} completed: {current_route}")
            current_route_index += 1

        print(f"\nAll routes completed: {self.env.all_routes}")
        return self.env.all_routes
    
class ShortestPath:
    """
    Greedily select the node that is closest to the current path.
    """
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)

    def run(self):
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir, 'eval_results_summary.json')
        return results, aggregated

    def construct_path(self, state):
        """
        Algorithm:
        Step 1: Initialization (same initialization as RL), happens in main at reset.
        Step 2: At each step, from valid neighboring nodes
            - Calculate the shortest path between the current path and the valid neighboring nodes.
            - Select the node with the shortest path.
        Step 3: Repeat step 2 until reaching the max path length.

        Note:
        - This may not result in overall shortest path but is a greedy selection of shortest path.
        """
        current_route_index = 0
        while current_route_index < self.env.NUM_ROUTES:
            print(f"\n=== Building Route {current_route_index + 1} ===")

            # Initialize current route (exactly like RL env initialization)
            current_route = initialize_route(self.env)

            while len(current_route) < self.env.MAX_ROUTE_LENGTH:

                # Get current frontier
                frontier = current_route[-1]

                # Get valid neighbors (adjacent and not already in path)
                route_set = set(current_route)
                valid_neighbors = [n for n in self.env.adj[frontier] if n not in route_set]
                if not valid_neighbors:
                    print(f"No valid neighbors found for frontier: {frontier}")
                    break

                # For each valid neighbor, compute the edge length as score
                best_score = np.inf
                best_node = None

                for neighbor in valid_neighbors:
                    # Get the edge length between frontier and neighbor
                    edge_length = self.env.link_lengths.get((frontier, neighbor), np.inf)

                    if edge_length < best_score:
                        best_score = edge_length
                        best_node = neighbor

                current_route.append(best_node)
                print(f"Route {current_route_index + 1} so far: {current_route}")

            self.env.all_routes.append(current_route)
            print(f"Route {current_route_index + 1} completed: {current_route}")
            current_route_index += 1

        print(f"\nAll routes completed: {self.env.all_routes}")
        return self.env.all_routes

class RewardMaximization:
    """
    Greedily select the node that maximizes the immediate (short-term) reward.
    """
    def __init__(self, env, config, num_runs, base_seed):

        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)

    def run(self):
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir, 'eval_results_summary.json')
        return results, aggregated

    def construct_path(self, state):
        """
        Algorithm:
        Step 1: Initialization (same initialization as RL), happens in main at reset.
        Step 2: For each valid neighbor, score the candidate using the exact reward
                formulation from RL training. Partial routes rely on fast metrics only;
                full simulations are triggered when the candidate completes or forces the
                route to end.
        Step 3: If the chosen node leads to a forced termination (no continuations), run
                the forced-end simulation, apply the same penalty as RL, and advance to
                the next route.
        Step 4: Repeat until all routes reach MAX_ROUTE_LENGTH or terminate early.

        * Because the route ending candidates need full simulation to choose between them, takes long time to finish,
        """

        def snapshot_env_state():
            return {
                "world": self.env.world,
                "all_routes": [route.copy() for route in self.env.all_routes],
                "current_route": self.env.current_route.copy(),
                "current_route_index": self.env.current_route_index,
                "is_baseline": self.env.is_baseline,
            }

        def restore_env_state(snapshot):
            self.env.world = snapshot["world"]
            self.env.all_routes = [route.copy() for route in snapshot["all_routes"]]
            self.env.current_route = snapshot["current_route"].copy()
            self.env.current_route_index = snapshot["current_route_index"]
            self.env.is_baseline = snapshot["is_baseline"]

        current_route_index = 0
        while current_route_index < self.env.NUM_ROUTES:
            print(f"\n=== Building Route {current_route_index + 1} ===")

            # Initialize current route (exactly like RL env initialization)
            current_route = initialize_route(self.env)
            self.env.current_route = current_route.copy()
            self.env.current_route_index = current_route_index

            while len(current_route) < self.env.MAX_ROUTE_LENGTH:

                # Get current frontier
                frontier = current_route[-1]

                # Get valid neighbors (adjacent and not already in path)
                route_set = set(current_route)
                valid_neighbors = [n for n in self.env.adj[frontier] if n not in route_set]
                if not valid_neighbors:
                    print(f"No valid neighbors found for frontier: {frontier}")
                    break

                best_reward = -np.inf
                best_node = None
                best_is_route_end = False
                best_is_forced_end = False

                for neighbor in valid_neighbors:
                    temp_route = [str(node) for node in current_route + [neighbor]]
                    is_route_end = len(temp_route) >= self.env.MAX_ROUTE_LENGTH
                    temp_route_set = set(temp_route)
                    next_valid_neighbors = [nxt for nxt in self.env.adj[neighbor] if nxt not in temp_route_set]
                    is_forced_end = (not is_route_end) and (len(next_valid_neighbors) == 0)

                    snapshot = snapshot_env_state()

                    self.env.current_route = temp_route.copy()
                    self.env.current_route_index = current_route_index

                    if is_route_end or is_forced_end:
                        self.env.all_routes = [route.copy() for route in snapshot["all_routes"]]
                        self.env.all_routes.append(temp_route.copy())
                        self.env.world = self.env.build_world(self.env.config.get("network"))
                        self.env._apply_action()
                        sim_result = self.env._step_until(self.env.horizon, print_metrics=False)
                        sim_result['route_completed'] = is_route_end
                        sim_result['route_forced_end'] = is_forced_end
                        reward = self.env.compute_reward(sim_result, is_route_end, is_forced_end)
                    else:
                        partial_metrics = self.env._get_partial_route_metrics()
                        sim_result = {
                            'wanting_to_onboard': partial_metrics['wanting_to_onboard'],
                            'total_demand': partial_metrics['total_demand'],
                            'demand_coverage_potential': partial_metrics['demand_coverage_potential'],
                            'route_overlap_ratio': partial_metrics['route_overlap_ratio'],
                            'route_completed': False,
                            'route_forced_end': False,
                        }

                        reward = self.env.compute_reward(sim_result, False, False)

                    restore_env_state(snapshot)

                    print(f"\nReward obtained from choice of {neighbor} = {reward}\n")
                    if reward > best_reward:
                        best_reward = reward
                        best_node = neighbor
                        best_is_route_end = is_route_end
                        best_is_forced_end = is_forced_end

                if best_node is None:
                    print("No valid neighbor produced a reward; stopping route construction.")
                    break

                # Restore original path after evaluations
                current_route.append(best_node)
                self.env.current_route = current_route.copy()
                print(f"\nRoute so far: {current_route}\n")

                if best_is_route_end:
                    print("Reached max route length; completing route.")
                    break

                if best_is_forced_end:
                    print("No valid continuations remain after this node; forcing route to end.")
                    break

            self.env.all_routes.append(current_route.copy())
            print(f"Route {current_route_index + 1} completed: {current_route}")
            current_route_index += 1
            self.env.current_route_index = current_route_index

        print(f"\nAll routes completed: {self.env.all_routes}")
        return self.env.all_routes

#####################################
# Real-world Baseline: 
#####################################

class RealWorld:
    """
    Use the existing routes in the real-world.
    """
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)
    
    def run(self):
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir, 'eval_results_summary.json')
        return results, aggregated

    def construct_path(self, state):
        """
        Algorithm:
        - In this baseline, we use the existing routes in the real-world i.e., no path construction is done.
        - We simply return the existing routes.
        - For fair comparison with RL agent i.e., when number of routes to construct is less than the total existing routes,
          we select a subset of routes based on "best_coverage" strategy i.e., Select routes with highest demand coverage potential
        """
        with open(self.env.network_dir / f"{self.env.config.get('network')}_existing_routes.json", "r") as f:
            all_routes = json.load(f)

        num_routes = self.env.config.get("num_routes", len(all_routes))
        total_existing = len(all_routes)

        print(f"Total existing routes: {total_existing}, Requested routes: {num_routes}")

        if num_routes == total_existing:
            # Use all routes if requested number is equal to available
            selected_routes = all_routes
            print(f"Using all {total_existing} existing routes")
        elif num_routes < total_existing:
            # Select subset based on best coverage
            selected_routes = self._select_best_coverage_subset(all_routes, num_routes)
        else:
            raise ValueError(f"Requested number of routes ({num_routes}) is greater than total existing routes ({total_existing})")

        print(f"Selected {len(selected_routes)} routes.")
        for i, route in enumerate(selected_routes):
            print(f"  Route {i+1}: {len(route['nodes'])} nodes - {route.get('name', 'Unnamed')}")

        # Convert dict to lists and int to str
        return [[str(node) for node in route['nodes']] for route in selected_routes]

    def _select_best_coverage_subset(self, all_routes, num_routes):
        """
        Select routes that cover the most demand using the best coverage strategy.

        Formula:
        For each route: 
            score = route_length × average_node_demand

        - route_length = number of nodes in the route (len(route['nodes']))
        - average_node_demand = (sum of (demand_out + demand_in) for all nodes in route) / route_length

        Routes are ranked by this score in descending order, and the top num_routes are selected.
        """
        route_scores = []
        for route in all_routes:
            nodes = route['nodes']
            # score based on route length and average node demand
            route_length = len(nodes)
            # Get average demand for nodes in this route
            avg_demand = sum(self._get_node_demand(node) for node in nodes) / len(nodes)
            # Combine length and demand coverage
            score = route_length * avg_demand
            route_scores.append((score, route))

        # Sort by score and select top routes
        route_scores.sort(key=lambda x: x[0], reverse=True)
        return [route for _, route in route_scores[:num_routes]]

    def _get_node_demand(self, node_id):
        """
        Get total demand (in + out) for a node as a proxy for its importance.
        """
        node_idx = self.env.node_to_idx.get(str(node_id))
        if node_idx is not None:
            return self.env.demand_out[node_idx] + self.env.demand_in[node_idx]
        else:
            raise ValueError(f"Node {node_id} not found in the network")
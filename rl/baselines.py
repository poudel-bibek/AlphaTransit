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
from rl.env_utils import plot_network_and_demand, plot_network_demand_and_path


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

def initialize_route(env, route_index, use_random=None, avoid_completed_routes=True):
    """
    Initialize a route exactly like the RL environment.

    Args:
        env: TransitEnv instance
        route_index: Index of route being initialized (0, 1, 2, ...)
        use_random: Whether to use random initialization. If None, uses env.random_path_init
        avoid_completed_routes: Whether to avoid starting from nodes in completed routes

    Returns:
        List containing the starting node for the route
    """
    # Use same logic as RL env initialization
    all_nodes = list(env.demand_df_cached["orig"].unique())

    # Determine if we should use random initialization
    if use_random is None:
        use_random = env.random_path_init

    # Filter nodes to avoid completed routes (same as RL env)
    if avoid_completed_routes:
        choice_nodes = [node for node in all_nodes if node not in env.all_routes]
    else:
        choice_nodes = all_nodes

    if use_random:
        choice = random.choice(choice_nodes)
        print(f"Initializing route {route_index} randomly at node: {choice}")
        return [choice]

    else:
        # Rank all nodes by highest demand emanating from them (total volume leaving each node)
        demand_df_grouped = env.demand_df_cached.groupby("orig").sum(numeric_only=True).reset_index()
        demand_df_ranked = demand_df_grouped.sort_values("volume", ascending=False)

        # Choose the highest ranking node from available choice_nodes
        for _, row in demand_df_ranked.iterrows():
            candidate_node = row["orig"]
            if candidate_node in choice_nodes:
                choice = candidate_node
                print(f"Initializing route {route_index} at highest available demand node: {choice}")
                return [choice]

def create_main_save_dir(config):
    """
    Create main save directory for baseline results.
    """
    now = datetime.now()
    main_save_dir = os.path.join(
        config.get("save_dir", "./baseline_data"), 
        f"{config.get('baseline_type', 'unknown')}_{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}"
    )
    os.makedirs(main_save_dir, exist_ok=True)
    print(f"Baseline results will be saved to: {main_save_dir}")
    return main_save_dir

def create_initial_network_plot(env, config, img_dir):
    """
    Create initial network + demand visualization, similar to RL training.
    """
    # Build a temporary world for initial visualization
    temp_world = env.build_world(config.get("network"))
    temp_world.name = config.get("network", "Unknown")  # Set proper network name
    # Load demand data for visualization
    env.load_demand_for_plotting(temp_world)
    output_path = os.path.join(img_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path)
    print(f"Initial network plot saved to: {output_path}")

def create_path_visualization(env, config, routes, img_dir):
    """
    Create network + path overlay visualization for multiple routes.
    """
    output_path = os.path.join(img_dir, f"{config.get('baseline_type', 'unknown')}_final_routes.png")
    plot_network_demand_and_path(env.world, routes, output_path)
    print(f"Routes visualization saved to: {output_path}")

def create_fancy_animations(env, config, baseline_save_dir):
    """
    """
    try:
        # Create fancy animation with all vehicles
        all_vehicles_anim_path = os.path.join(baseline_save_dir, f"{config.get('baseline_type', 'unknown')}_anim_all_vehicles.gif")
        env.world.analyzer.network_fancy(
            animation_speed_inverse=10,
            figsize=11,
            sample_ratio=1.0,
            interval=5,
            trace_length=5,
            network_font_size=11,
            antialiasing=False,
            file_name=all_vehicles_anim_path,
            save_as_mp4=True
        )
        print(f"All vehicles animation saved to: {all_vehicles_anim_path}")
        
        # # Create fancy animation with buses only
        # bus_only_anim_path = os.path.join(baseline_save_dir, f"{config.get('baseline_type', 'unknown')}_anim_bus_only.gif")
        # env.world.analyzer.network_fancy(
        #     animation_speed_inverse=10,
        #     sample_ratio=1.0,
        #     interval=5,
        #     trace_length=5,
        #     network_font_size=14,
        #     antialiasing=False,
        #     file_name=bus_only_anim_path,
        #     save_as_mp4=False,
        #     bus_only=True
        # )
        # print(f"Bus-only animation saved to: {bus_only_anim_path}")
        
    except Exception as e:
        print(f"Warning: Could not create fancy animations: {e}")
        print("This might be due to insufficient simulation data or missing dependencies.")

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
    create_fancy_animations(env, config, baseline_save_dir)

    return {
        'sim_result': sim_result
    }

def average_sim_results(results_list):
    """
    Compute average of simulation results across multiple runs.
    Handles numerical values generically; concatenates and summarizes distribution lists.
    """
    if not results_list:
        return {}

    # Collect all keys from first result
    keys = results_list[0]['sim_result'].keys()
    averaged = {}

    for key in keys:
        values = []
        for res in results_list:
            val = res['sim_result'].get(key)
            if isinstance(val, (int, float)):
                values.append(val)

        if values:
            averaged[key] = np.mean(values)

    # Handle distribution metrics (lists) by concatenating and computing combined statistics
    distribution_keys = ['waiting_time_dstr', 'movement_time_dstr', 'travel_time_dstr']

    for dist_key in distribution_keys:
        all_values = []
        for res in results_list:
            dist_list = res['sim_result'].get(dist_key, [])
            if isinstance(dist_list, list):
                all_values.extend(dist_list)

        if all_values:
            dist_array = np.array(all_values)
            # Add combined statistics for this distribution
            averaged[f'{dist_key}_combined_mean'] = np.mean(dist_array)
            averaged[f'{dist_key}_combined_std'] = np.std(dist_array)
            averaged[f'{dist_key}_combined_min'] = np.min(dist_array)
            averaged[f'{dist_key}_combined_max'] = np.max(dist_array)
            averaged[f'{dist_key}_total_passengers'] = len(dist_array)

    return averaged

def print_results(results_list, averaged):
    """
    """
    
    eval_metrics = {
        # Waiting metrics (all times in seconds for consistency)
        'total_wait_time': 'seconds',
        'wait_time_sim_end': 'seconds',
        'sim_end_waiting_passengers_count': 'count',

        # Travel metrics (all times in seconds for consistency)
        'wanting_to_onboard': 'count',
        'total_onboarded_count': 'count',
        'completed_trip_passengers_count': 'count',
        'movement_time': 'seconds',
        'total_travel_time': 'seconds',

        # Per passenger averages
        'avg_wait_time': 'seconds',
        'avg_movement_time': 'seconds',
        'avg_travel_time': 'seconds',

        # Data collected (raw distributions in seconds)
        'waiting_time_dstr': 'distribution',
        'movement_time_dstr': 'distribution',
        'travel_time_dstr': 'distribution',

        # Reward components
        'demand_coverage_potential': '%',
        'demand_coverage_actual': '%',
        'route_overlap_ratio': 'ratio',

        # Route metrics
        'route_length': 'meters',
        'bus_utilization': '%',
        'average_bus_speed': 'm/s',
        'fleet_size': 'count',
        'service_rate': '%',
        'onboard_rate': '%',
        'transfer_rate': '%',
        'route_efficiency': 'passengers/km',
        'node_coverage': '%',

        # Additional metrics for compatibility (reward commented out for baselines)
        # 'episode_final_reward': 'reward',  # Not applicable for heuristic baselines
        'episode_length': 'count',
        'completed_trips': 'count',  # maps to completed_trip_passengers_count
    }

    print("\n=== Individual Run Results ===")
    # Find max key length for alignment
    max_key_len = max(len(key) for key in eval_metrics.keys())

    for i, res in enumerate(results_list, 1):
        print(f"Run {i}:")
        for key in eval_metrics.keys():
            val = res['sim_result'].get(key)
            if val is not None and isinstance(val, (int, float)):
                val_str = f"{val:.2f}" if isinstance(val, float) else f"{val}"
                unit_str = f" ({eval_metrics[key]})"
                print(f"  {key:<{max_key_len}} : {val_str}{unit_str}")
        print("---")

    print("\n=== Averaged Results ===")
    for key in eval_metrics.keys():
        values = []
        for res in results_list:
            val = res['sim_result'].get(key)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)

        if values:
            str_values = [f"{v:.2f}" if isinstance(v, float) else f"{v}" for v in values]
            calc_str = " + ".join(str_values)
            avg_val = averaged.get(key, 0.0)
            avg_str = f"{avg_val:.2f}" if isinstance(avg_val, float) else f"{avg_val}"
            unit_str = f" ({eval_metrics[key]})"
            print(f"  {key:<{max_key_len}} : ({calc_str}) / {len(values)} = {avg_str}{unit_str}")

    # Print combined distribution statistics
    distribution_keys = ['waiting_time_dstr', 'movement_time_dstr', 'travel_time_dstr']
    for dist_key in distribution_keys:
        combined_stats = [k for k in averaged.keys() if k.startswith(f'{dist_key}_combined')]
        if combined_stats:
            print(f"\n  === Combined {dist_key.replace('_', ' ').title()} Statistics ===")
            for stat_key in combined_stats:
                stat_name = stat_key.replace(f'{dist_key}_combined_', '').replace('_', ' ').title()
                stat_value = averaged[stat_key]
                if 'passengers' in stat_key:
                    print(f"  {stat_name:<{max_key_len-2}} : {stat_value}")
                else:
                    print(f"  {stat_name:<{max_key_len-2}} : {stat_value:.2f} seconds")
    
    # Print LaTeX row
    print("\n")
    service = averaged['service_rate']
    onboard = averaged['onboard_rate']
    wait = averaged['avg_wait_time'] / 60 # Minutes
    transfer = averaged['transfer_rate']
    travel = averaged['avg_travel_time'] / 60 # Minutes
    completed = int(averaged['completed_trip_passengers_count'])
    print(f"& ${service:.2f}$ & ${onboard:.2f}$ & ${wait:.2f}$ & ${transfer:.2f}$ & ${travel:.2f}$ & ${completed}$")

def execute_runs(baseline, num_runs, base_seed):
    """
    Execute multiple runs for a baseline.
    """
    results = []
    for run in range(num_runs):
        current_seed = base_seed + run
        set_global_seeds(current_seed)
        print(f"\n=== Run {run+1} (Seed: {current_seed}) ===")
        
        # Create seed-specific directory
        seed_dir = os.path.join(baseline.main_save_dir, f"seed_{current_seed}")
        img_dir = os.path.join(seed_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # Reset env
        state, _ = baseline.env.reset()

        routes = baseline.construct_path(state)
        print(f"Routes: {routes}")

        result = simulate_baseline_routes(baseline.env, baseline.config, routes, img_dir, seed_dir)
        results.append(result)
    
    averaged = average_sim_results(results)
    print_results(results, averaged)
    return results, averaged

#####################################
# Heuristic Baselines: 
#####################################

class RandomBaseline:
    """
    Random Neighbor Baseline.
    """
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir = create_main_save_dir(config)

    def run(self):
        return execute_runs(self, self.num_runs, self.base_seed)

    def construct_path(self, state):
        """
        Multi-route version:
        Algorithm:
        Step 1: Initialize first route (same initialization as RL)
        Step 2: For each route, select random neighbors until max length or no valid moves
        Step 3: Move to next route and repeat
        Step 4: Return all completed routes
        """
        all_routes = []
        current_route_index = 0

        while current_route_index < self.env.NUM_ROUTES:
            print(f"\n=== Building Route {current_route_index + 1} ===")

            # Initialize current route (exactly like RL env initialization)
            current_route = initialize_route(self.env, current_route_index)

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
            all_routes.append(current_route)
            print(f"Route {current_route_index + 1} completed: {current_route}")
            current_route_index += 1

        print(f"\nAll routes completed: {all_routes}")
        return all_routes

class GreedyDemandCoverage:
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir = create_main_save_dir(config)

    def run(self):
        return execute_runs(self, self.num_runs, self.base_seed)

    def construct_path(self, state):
        """
        Algorithm:
        Step 1: Initialization (same initialization as RL), happens in main at reset.
        Step 2: At each step, from valid neighboring nodes
            - Calculate the incremental demand that would be served by adding that node to the path.
            - Consider the combined demand (d_out + d_in) for that node and select the one with the highest.
        Step 3: Repeat step 2 until reaching the max path length.
        """
        current_path = list(self.env.current_path) # Make a copy
        while len(current_path) < self.env.MAX_ROUTE_LENGTH:

            # Get current frontier
            frontier = current_path[-1]

            # Get valid neighbors (adjacent and not already in path)
            path_set = set(current_path)
            valid_neighbors = [n for n in self.env.adj[frontier] if n not in path_set]
            if not valid_neighbors:
                break

            # Convert to indices for demand calculation
            path_indices = np.array([self.env.node_to_idx[node] for node in current_path])

            # For each valid neighbor, compute incremental demand (d_out_path + d_in_path)
            best_score = -np.inf
            best_node = None

            for neighbor in valid_neighbors:
                neigh_idx = self.env.node_to_idx[neighbor]

                # Incremental outgoing: flows from neighbor to current path nodes
                d_out_inc = self.env.od_matrix[neigh_idx, path_indices].sum()

                # Incremental incoming: flows from current path nodes to neighbor
                d_in_inc = self.env.od_matrix[path_indices, neigh_idx].sum()

                score = d_out_inc + d_in_inc

                if score > best_score:
                    best_score = score
                    best_node = neighbor

            current_path.append(best_node)
            print(f"\nRoute so far: {current_path}\n")
        return current_path

class GreedyShortestPath:
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir = create_main_save_dir(config)

    def run(self):
        return execute_runs(self, self.num_runs, self.base_seed)

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
        current_path = list(self.env.current_path)  # Make a copy
        while len(current_path) < self.env.MAX_ROUTE_LENGTH:

            # Get current frontier
            frontier = current_path[-1]

            # Get valid neighbors (adjacent and not already in path)
            path_set = set(current_path)
            valid_neighbors = [n for n in self.env.adj[frontier] if n not in path_set]
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

            current_path.append(best_node)
            print(f"\nRoute so far: {current_path}\n")
        return current_path

class GreedyRewardMaximization:
    def __init__(self, env, config, num_runs, base_seed):
        """
        A baseline for myopic short-term reward maximization.
        """
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir = create_main_save_dir(config)

    def run(self):
        return execute_runs(self, self.num_runs, self.base_seed)

    def construct_path(self, state):
        """
        Algorithm:
        Step 1: Initialization (same initialization as RL), happens in main at reset.
        Step 2: At each step, from valid neighboring nodes
            - Calculate the reward that would be obtained by adding that node to the path.
                - This needs simulating each possible choice at each step.
            - Select the node with the highest reward.
        Step 3: Repeat step 2 until reaching the max path length.
        """
        current_path = list(self.env.current_path)  # Make a copy
        original_path = self.env.current_path[:]  # Save original path
        while len(current_path) < self.env.MAX_ROUTE_LENGTH:

            # Get current frontier
            frontier = current_path[-1]

            # Get valid neighbors (adjacent and not already in path)
            path_set = set(current_path)
            valid_neighbors = [n for n in self.env.adj[frontier] if n not in path_set]
            if not valid_neighbors:
                print(f"No valid neighbors found for frontier: {frontier}")
                break

            best_reward = -np.inf
            best_node = None

            for neighbor in valid_neighbors:
                temp_path = current_path + [neighbor]

                # Temporarily set path and simulate
                self.env.current_path = temp_path
                self.env.world = self.env.build_world(self.env.config.get("network"))
                self.env._apply_action()
                sim_result = self.env._step_until(self.env.horizon, print_metrics=False)
                reward = self.env.compute_reward(sim_result)
                print(f"\nReward obtained from choice of {neighbor} = {reward}\n")
                if reward > best_reward:
                    best_reward = reward
                    best_node = neighbor

            # Restore original path after evaluations
            self.env.current_path = original_path
            current_path.append(best_node)
            print(f"\nRoute so far: {current_path}\n")
        return current_path

#####################################
# Real-world Baseline: 
#####################################

class RealWorldBaseline:
    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir = create_main_save_dir(config)
    
    def run(self):
        return execute_runs(self, self.num_runs, self.base_seed)

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
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
- MAX_PATH_LENGTH
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
    output_path = os.path.join(img_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path)
    print(f"Initial network plot saved to: {output_path}")

def create_path_visualization(env, config, path, img_dir):
    """
    Create network + path overlay visualization.
    """
    output_path = os.path.join(img_dir, f"{config.get('baseline_type', 'unknown')}_final_path.png")
    plot_network_demand_and_path(env.world, path, output_path)
    print(f"Path visualization saved to: {output_path}")

def create_fancy_animations(env, config, baseline_save_dir):
    """
    """
    try:
        # Create fancy animation with all vehicles
        all_vehicles_anim_path = os.path.join(baseline_save_dir, f"{config.get('baseline_type', 'unknown')}_anim_all_vehicles.gif")
        env.world.analyzer.network_fancy(
            animation_speed_inverse=10,
            sample_ratio=1.0,
            interval=5,
            trace_length=5,
            network_font_size=14,
            antialiasing=False,
            file_name=all_vehicles_anim_path,
            save_as_mp4=False
        )
        print(f"All vehicles animation saved to: {all_vehicles_anim_path}")
        
        # Create fancy animation with buses only
        bus_only_anim_path = os.path.join(baseline_save_dir, f"{config.get('baseline_type', 'unknown')}_anim_bus_only.gif")
        env.world.analyzer.network_fancy(
            animation_speed_inverse=10,
            sample_ratio=1.0,
            interval=5,
            trace_length=5,
            network_font_size=14,
            antialiasing=False,
            file_name=bus_only_anim_path,
            save_as_mp4=False,
            bus_only=True
        )
        print(f"Bus-only animation saved to: {bus_only_anim_path}")
        
    except Exception as e:
        print(f"Warning: Could not create fancy animations: {e}")
        print("This might be due to insufficient simulation data or missing dependencies.")

def simulate_baseline_path(env, config, path, img_dir, baseline_save_dir):
    """
    Shared across all baselines.
    Simulate the path to get the performance results.
    Includes visualization and animation generation similar to RL training.
    """
    # Create initial network visualization before simulation
    create_initial_network_plot(env, config, img_dir)

    # Update env's path
    env.current_path = path
    
    # Build world with correct demand allocation
    env.world = env.build_world(config.get("network"))
    
    # Apply buses using env's method
    env._apply_action()
    
    # Run simulation and get metrics
    sim_result = env._step_until(env.horizon, print_metrics=True)

    # Generate visualizations using standalone functions
    create_path_visualization(env, config, path, img_dir)
    create_fancy_animations(env, config, baseline_save_dir)
    
    return {
        'sim_result': sim_result
    }

def average_sim_results(results_list):
    """
    Compute average of simulation results across multiple runs.
    Handles numerical values generically; skips non-numerical (e.g., lists).
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
    
    return averaged

def print_results(results_list, averaged):
    """
    """
    units = {
        'total_wait_time': 'seconds',
        'wait_time_sim_end': 'seconds',
        'sim_end_waiting_passengers_count': 'count',
        'avg_wait_time': 'seconds',
        'wanting_to_onboard': 'count',
        'total_onboarded_count': 'count',
        'completed_trip_passengers_count': 'count',
        'movement_time': 'seconds',
        'total_travel_time': 'seconds',
        'route_length': 'meters',
        'bus_utilization': '%',
        'average_bus_speed': 'm/s',
        'service_rate': '%',
        'onboard_rate': '%',
        'route_efficiency': 'passengers/km'
    }

    print("\n=== Individual Run Results ===")
    # Find max key length for alignment
    max_key_len = 0
    for res in results_list:
        for key in res['sim_result']:
            if isinstance(res['sim_result'][key], (int, float)):
                max_key_len = max(max_key_len, len(key))
    
    for i, res in enumerate(results_list, 1):
        print(f"Run {i}:")
        for key, val in sorted(res['sim_result'].items()):  # Sort keys for consistent order
            if isinstance(val, (int, float)):
                val_str = f"{val:.2f}" if isinstance(val, float) else f"{val}"
                unit_str = f" ({units.get(key, '')})" if units.get(key) else ""
                print(f"  {key:<{max_key_len}} : {val_str}{unit_str}")
        print("---")

    print("\n=== Averaged Results ===")
    for key in sorted(averaged):  # Sort keys for consistent order
        values = []
        for res in results_list:
            val = res['sim_result'].get(key)
            if isinstance(val, (int, float)):
                values.append(val)
        
        if values:
            str_values = []
            for v in values:
                if isinstance(v, float):
                    str_values.append(f"{v:.2f}")
                else:
                    str_values.append(f"{v}")
            
            calc_str = " + ".join(str_values)
            avg_val = averaged[key]
            avg_str = f"{avg_val:.2f}" if isinstance(avg_val, float) else f"{avg_val}"
            unit_str = f" ({units.get(key, '')})" if units.get(key) else ""
            print(f"  {key:<{max_key_len}} : ({calc_str}) / {len(values)} = {avg_str}{unit_str}")
    
    # Print LaTeX row
    print("\n")
    completed = int(averaged.get('completed_trip_passengers_count', 0))
    service = averaged.get('service_rate', 0.0)
    wait = averaged.get('avg_wait_time', 0.0)
    efficiency = averaged.get('route_efficiency', 0.0)
    print(f"& ${completed}$ & ${service:.2f}$ & ${wait:.2f}$ & ${efficiency:.2f}$")

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
        
        path = baseline.construct_path(state)
        print(f"Path: {path}")
        
        result = simulate_baseline_path(baseline.env, baseline.config, path, img_dir, seed_dir)
        results.append(result)
    
    averaged = average_sim_results(results)
    print_results(results, averaged)
    return results, averaged

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
        Algorithm:
        Step 1: Initialization (same initialization as RL), happens in main at reset.
        Step 2: At each step, select a random neighbor from the valid neighbors.
        Step 3: Repeat step 2 until reaching the max path length.
        """
        current_path = list(self.env.current_path)  # Make a copy
        while len(current_path) < self.env.MAX_PATH_LENGTH:

            # Get current frontier
            frontier = current_path[-1]
            
            # Get valid neighbors (adjacent and not already in path)
            path_set = set(current_path)
            valid_neighbors = [n for n in self.env.adj[frontier] if n not in path_set]
            if not valid_neighbors:
                print(f"No valid neighbors found for frontier: {frontier}")
                break

            # Select random neighbor
            next_node = random.choice(valid_neighbors)
            
            current_path.append(next_node)
            print(f"\nRoute so far: {current_path}\n")
        return current_path

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
        while len(current_path) < self.env.MAX_PATH_LENGTH:

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
        while len(current_path) < self.env.MAX_PATH_LENGTH:

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
        while len(current_path) < self.env.MAX_PATH_LENGTH:

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
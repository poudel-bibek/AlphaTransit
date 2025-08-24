"""
Heuristic Baselines: 
- Baselines mostly setup in a way that "to construct the path" we dont need to simulate. 
- However, to get the performance results on the path, we need to simulate.
- Repurposing the RL env setup for the baselines as well.

0. Random baseline (NOT USED RIGHT NOW): 
    - Random selection of next node.
    - Lower bound on performance.

1. Greedy Demand Coverage: 
   - Build route by greedily selecting the nodes that maximize the immediate demand coverage.
   - 

2. # TODO

---------------

These baselines still need to respect constraints such as: 
- MAX_PATH_LENGTH
- SERVICE_FREQUENCY (This will be taken care by the sim bus handler)
- alpha (facor to determine the % of demand allocated to bus)
- For each baseline, we need to form a path and then simulate to get results.

---------------
Additional notes (ignore)
- In case the immediate next node does not add any new demand coverage (because trips are not completed by just adding 1 node)
"""

import os
import numpy as np
from datetime import datetime
from rl.env_utils import plot_network_and_demand, plot_network_demand_and_path


def create_baseline_save_directory(config):
    """
    Create a save directory for baseline results.
    """
    now = datetime.now()
    baseline_save_dir = os.path.join(
        config.get("save_dir", "./baseline_data"), 
        f"{config.get('baseline_type', 'unknown')}_{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}"
    )
    img_dir = os.path.join(baseline_save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"Baseline results will be saved to: {baseline_save_dir}")
    return baseline_save_dir, img_dir

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


class RandomBaseline:
    """
    TODO: Implement later (if required)
    """
    def __init__(self, env):
        self.nodes = env.nodes

    def construct_path(self, state):
        return self.nodes.sample()

    def simulate_path(self, path):
        """
        """
        pass

class GreedyDemandCoverage:
    def __init__(self, env):
        self.env = env
        self.config = env.config
        self.world = env.build_world(env.config.get("network"))
        
        # Create baseline save directory structure
        self.baseline_save_dir, self.img_dir = create_baseline_save_directory(self.config)

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
            
            if best_node is None:
                break
                
            current_path.append(best_node)
        return current_path


    def simulate_path(self, path):
        """
        Simulate the path to get the performance results.
        Includes visualization and animation generation similar to RL training.
        """
        
        # Create initial network visualization before simulation
        create_initial_network_plot(self.env, self.config, self.img_dir)

        # Update env's path
        self.env.current_path = path
        
        # Build world with correct demand allocation
        self.env.world = self.env.build_world(self.env.config.get("network"))
        
        # Apply buses using env's method
        self.env._apply_action()
        
        # Run simulation and get metrics
        sim_result = self.env._step_until(self.env.horizon, print_metrics=True)

        # Calculate reward using env's method
        # reward = self.env.compute_reward(sim_result)

        # Generate visualizations using standalone functions
        create_path_visualization(self.env, self.config, path, self.img_dir)
        create_fancy_animations(self.env, self.config, self.baseline_save_dir)
        
        return {
            'sim_result': sim_result
        }
        
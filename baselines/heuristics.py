"""
Heuristic Baselines for Transit Route Network Design.

- RandomWalk: Uniform random neighbor selection
- DemandCoverage: Demand-proportional node selection
- ShortestPath: Inverse edge length-proportional selection
- RewardMaximization: Greedy immediate reward maximization
- RealWorld: Existing real-world routes
"""

import os
import json
import numpy as np
import random
import torch
from scipy.special import softmax
from rl.env_utils import (
    plot_network_and_demand,
    plot_network_demand_and_path,
    initialize_route,
    write_results_summary,
    save_routes_json,
)
from baselines.utils import (
    set_global_seeds,
    create_main_save_dir,
    simulate_baseline_routes,
    create_initial_network_plot,
    create_path_visualization,
    create_fancy_animations,
    execute_runs,
    print_results,
)

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
    Build route by sampling nodes proportional to their demand coverage contribution.
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
            - Calculate the incremental demand (d_out + d_in) for each candidate node.
            - Normalize scores (z-score) and sample via softmax.
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
                if not valid_neighbors:
                    break

                # Convert to indices for demand calculation
                route_indices = np.array([self.env.node_to_idx[node] for node in current_route])

                # For each valid neighbor, compute incremental demand (d_out + d_in)
                scores = []
                for neighbor in valid_neighbors:
                    neigh_idx = self.env.node_to_idx[neighbor]

                    # Incremental outgoing: flows from neighbor to current path nodes
                    d_out_inc = self.env.od_matrix[neigh_idx, route_indices].sum()

                    # Incremental incoming: flows from current path nodes to neighbor
                    d_in_inc = self.env.od_matrix[route_indices, neigh_idx].sum()

                    scores.append(d_out_inc + d_in_inc)

                # Normalize scores (z-score), then sample via softmax
                scores = np.array(scores)
                scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                probs = softmax(scores)
                next_node = str(np.random.choice(valid_neighbors, p=probs))

                current_route.append(next_node)
                print(f"Route {current_route_index + 1} so far: {current_route}")

            self.env.all_routes.append(current_route)
            print(f"Route {current_route_index + 1} completed: {current_route}")
            current_route_index += 1

        print(f"\nAll routes completed: {self.env.all_routes}")
        return self.env.all_routes
    
class ShortestPath:
    """
    Sample nodes proportional to inverse edge length (shorter edges more likely).
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
            - Calculate the edge length for each candidate node.
            - Normalize negative edge lengths (z-score) and sample via softmax (shorter = higher prob).
        Step 3: Repeat step 2 until reaching the max path length.
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

                # For each valid neighbor, get the edge length
                edge_lengths = []
                for neighbor in valid_neighbors:
                    edge_length = self.env.link_lengths.get((frontier, neighbor), np.inf)
                    edge_lengths.append(edge_length)

                # Normalize negative edge lengths (z-score), then sample via softmax (shorter = higher prob)
                logits = -np.array(edge_lengths)
                logits = (logits - logits.mean()) / (logits.std() + 1e-8)
                probs = softmax(logits)
                next_node = str(np.random.choice(valid_neighbors, p=probs))

                current_route.append(next_node)
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
"""
Pure MCTS Baseline for Transit Route Network Design.

This baseline runs Monte Carlo Tree Search WITHOUT a learned policy-value
network. It uses uniform random priors and rollout-based value estimation
instead of the GNN policy head (P) and value head (V) used by AlphaTransit.

Comparing:
- AlphaTransit (MCTS + learned P + learned V)
- Pure MCTS (MCTS + uniform P + rollout V)
- End-to-End RL (learned P + learned V, no MCTS)

demonstrates how much each component contributes to performance.

Usage:
    python main.py --mode baseline --baseline_type mcts --alpha 0.3
"""

import os
import csv
import math
import random
import time
import numpy as np
from datetime import datetime


class PureMCTS:
    """
    Pure MCTS baseline without learned policy-value network.
    Uses uniform priors and proxy-reward rollout-based value estimation.
    """

    def __init__(self, env, config, num_runs, base_seed):
        from baselines.utils import create_main_save_dir
        self.env = env
        self.config = config
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)

    def run(self):
        from baselines.utils import execute_runs, print_results
        from rl.env_utils import write_results_summary
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir,
                              'eval_results_summary.json')
        return results, aggregated

    def _proxy_rollout_value(self, mcts_state):
        """
        Random playout to terminal state, then compute a fast proxy reward
        based on demand coverage (no full UXsim simulation).
        """
        sim_state = mcts_state.clone()

        # Complete all remaining routes with random actions
        while not sim_state.is_terminal():
            valid = sim_state.get_valid_actions()
            if not valid:
                sim_state = sim_state.force_route_end()
                continue
            action = random.choice(valid)
            sim_state = sim_state.apply_action(action)

        # Compute proxy reward from demand coverage
        self.env.all_routes = [list(r) for r in sim_state.all_routes]
        self.env.current_route = []
        metrics = self.env._get_partial_route_metrics()
        coverage = metrics.get('demand_coverage_potential', 0.0)

        # Simple proxy: coverage as reward (higher = better)
        return coverage

    def _full_rollout_value(self, mcts_state):
        """
        Random playout to terminal state, then run full UXsim simulation.
        Expensive but accurate.
        """
        sim_state = mcts_state.clone()

        while not sim_state.is_terminal():
            valid = sim_state.get_valid_actions()
            if not valid:
                sim_state = sim_state.force_route_end()
                continue
            action = random.choice(valid)
            sim_state = sim_state.apply_action(action)

        reward, _ = self.env.simulate_routes_mcts(sim_state.all_routes)
        return reward

    def construct_path(self, state):
        from rl.mcts_utils import MCTSState, MCTSNode, MCTSTree
        from rl.env_utils import initialize_route

        n_iter = self.config.get("n_iter", 25)
        c_puct = self.config.get("c_puct", 1.0)
        num_routes = self.config.get("num_routes", 16)
        max_len = self.config.get("max_route_length", 14)
        n_nodes = len(self.env.node_to_idx)
        tau = 0.1  # Near-greedy action selection

        # Use proxy rollout by default (fast); set MCTS_FULL_ROLLOUT=1 for full sim
        use_full_rollout = os.environ.get("MCTS_FULL_ROLLOUT", "") == "1"
        rollout_fn = self._full_rollout_value if use_full_rollout else self._proxy_rollout_value

        # CSV logging
        log_path = os.environ.get("MCTS_LOG_CSV", "")
        log_file = None
        log_writer = None
        if log_path:
            log_file = open(log_path, "w", newline="")
            log_writer = csv.DictWriter(log_file, fieldnames=[
                "route_idx", "step", "route_length", "action_node",
                "n_valid_actions", "root_visits", "root_max_visits",
                "root_q_mean", "root_q_max", "step_time_s",
            ])
            log_writer.writeheader()

        all_routes = []
        episode_start = time.time()

        for k in range(num_routes):
            current_route = initialize_route(self.env)
            self.env.all_routes = [list(r) for r in all_routes]

            mcts_state = MCTSState(
                current_route=list(current_route),
                all_routes=[list(r) for r in all_routes],
                current_route_index=k,
                num_routes=num_routes,
                max_route_length=max_len,
                adj=self.env.adj,
                node_to_idx=self.env.node_to_idx,
                idx_to_node=self.env.idx_to_node,
                env=self.env,
            )
            tree = MCTSTree(mcts_state)
            step = 0

            while len(current_route) < max_len:
                step_start = time.time()
                valid = mcts_state.get_valid_actions()
                if not valid:
                    break

                # Expand root with uniform priors + rollout value
                if not tree.root.expanded:
                    priors = {a: 1.0 / len(valid) for a in valid}
                    value = rollout_fn(mcts_state)
                    tree.root.expand(priors, value)

                # Run MCTS simulations
                for _ in range(n_iter):
                    node = tree.root
                    sim_state = mcts_state.clone()
                    path = []

                    # SELECT: walk down tree using PUCT
                    while node.expanded and not sim_state.is_terminal():
                        action = node.select_action(c_puct)
                        path.append((node, action))
                        child = node.get_child(action)
                        sim_state = child.state
                        node = child
                        if not node.expanded:
                            break

                    # EXPAND leaf
                    if not node.expanded and not sim_state.is_terminal():
                        leaf_valid = sim_state.get_valid_actions()
                        if leaf_valid:
                            leaf_priors = {a: 1.0 / len(leaf_valid) for a in leaf_valid}
                            leaf_value = rollout_fn(sim_state)
                            node.expand(leaf_priors, leaf_value)
                        else:
                            leaf_value = rollout_fn(sim_state)
                    elif sim_state.is_terminal():
                        leaf_value = rollout_fn(sim_state)
                    else:
                        leaf_value = node.value

                    # BACKUP
                    for parent, act in reversed(path):
                        parent.update(act, leaf_value)

                # Select action from visit counts
                policy = tree.root.get_visit_count_policy(tau, n_nodes)
                action = max(valid, key=lambda a: policy[a])
                action_node = self.env.idx_to_node[action]

                # Log step
                step_time = time.time() - step_start
                if log_writer:
                    root_visits = sum(tree.root.N.values()) if tree.root.N else 0
                    root_max_v = max(tree.root.N.values()) if tree.root.N else 0
                    q_vals = list(tree.root.Q.values()) if tree.root.Q else [0]
                    log_writer.writerow({
                        "route_idx": k,
                        "step": step,
                        "route_length": len(current_route),
                        "action_node": action_node,
                        "n_valid_actions": len(valid),
                        "root_visits": root_visits,
                        "root_max_visits": root_max_v,
                        "root_q_mean": f"{np.mean(q_vals):.6f}",
                        "root_q_max": f"{max(q_vals):.6f}",
                        "step_time_s": f"{step_time:.2f}",
                    })
                    log_file.flush()

                # Advance
                current_route.append(action_node)
                mcts_state = mcts_state.apply_action(action)
                tree.advance(action)
                step += 1

            all_routes.append(current_route)
            elapsed = time.time() - episode_start
            print(f"  Route {k+1}/{num_routes}: {len(current_route)} nodes, "
                  f"elapsed {elapsed:.0f}s, route={current_route[:4]}...")

        if log_file:
            log_file.close()

        total_time = time.time() - episode_start
        print(f"  Pure MCTS episode complete: {num_routes} routes in {total_time:.0f}s")

        return all_routes

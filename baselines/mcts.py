"""
Pure MCTS Baseline for Transit Route Network Design.

This baseline runs Monte Carlo Tree Search WITHOUT a learned policy-value
network. It uses uniform random priors and rollout-based value estimation
(full UXsim simulation) instead of the GNN policy head (P) and value head (V)
used by AlphaTransit.

Comparing:
- AlphaTransit (MCTS + learned P + learned V)
- Pure MCTS (MCTS + uniform P + rollout V)
- End-to-End RL (learned P + learned V, no MCTS)

demonstrates how much each component contributes to performance.

Usage:
    python main.py --mode baseline --baseline_type mcts --alpha 0.3 --n_iter 100
"""

import os
import csv
import random
import time
import numpy as np
import multiprocessing as mp


def _rollout_worker(args):
    """
    Worker function for parallel rollouts.
    Each worker builds its own UXsim world and runs a full simulation.
    Must be a top-level function for multiprocessing.
    """
    all_routes, config_dict = args
    # Lazy import inside worker to avoid pickling issues
    from rl.env import TransitEnv
    env = TransitEnv(config_dict)
    reward, _ = env.simulate_routes_mcts(all_routes)
    return reward


def _complete_routes_randomly(mcts_state):
    """Random playout from current state to terminal. Returns completed route set."""
    sim_state = mcts_state.clone()
    while not sim_state.is_terminal():
        valid = sim_state.get_valid_actions()
        if not valid:
            sim_state = sim_state.force_route_end()
            continue
        action = random.choice(valid)
        sim_state = sim_state.apply_action(action)
    return [list(r) for r in sim_state.all_routes]


class PureMCTS:
    """
    Pure MCTS baseline without learned policy-value network.
    Uses uniform priors and full UXsim simulation rollouts.
    Rollouts are parallelized across multiple workers.
    """

    def __init__(self, env, config, num_runs, base_seed):
        from baselines.utils import create_main_save_dir
        self.env = env
        self.config = config
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.main_save_dir, self.eval_root_dir = create_main_save_dir(config)
        self.n_workers = config.get("num_mcts_rollout_workers", 8)
        # Serializable config for worker processes (no env/device objects)
        self._config_dict = {k: v for k, v in config.items()
                             if isinstance(v, (int, float, str, bool, type(None)))}

    def run(self):
        from baselines.utils import execute_runs, print_results
        from rl.env_utils import write_results_summary
        results, aggregated = execute_runs(self, self.num_runs, self.base_seed)
        write_results_summary(aggregated, self.num_runs, self.eval_root_dir,
                              'eval_results_summary.json')
        return results, aggregated

    def _batch_rollout(self, leaf_states):
        """
        Run rollouts for multiple leaf states in parallel.
        Returns list of reward values.
        """
        # Complete each leaf state randomly to get terminal route sets
        route_sets = [_complete_routes_randomly(s) for s in leaf_states]

        if len(route_sets) == 1:
            # Single rollout — no need for multiprocessing overhead
            reward, _ = self.env.simulate_routes_mcts(route_sets[0])
            return [reward]

        # Parallel rollouts
        worker_args = [(rs, self._config_dict) for rs in route_sets]
        n_workers = min(self.n_workers, len(route_sets))
        with mp.Pool(n_workers) as pool:
            rewards = pool.map(_rollout_worker, worker_args)
        return rewards

    def _single_rollout(self, mcts_state):
        """Single rollout: random playout + full UXsim simulation."""
        routes = _complete_routes_randomly(mcts_state)
        reward, _ = self.env.simulate_routes_mcts(routes)
        return reward

    def construct_path(self, state):
        from rl.mcts_utils import MCTSState, MCTSNode, MCTSTree
        from rl.env_utils import initialize_route

        n_iter = self.config.get("n_iter", 100)
        c_puct = self.config.get("c_puct", 1.0)
        num_routes = self.config.get("num_routes", 16)
        max_len = self.config.get("max_route_length", 14)
        n_nodes = len(self.env.node_to_idx)
        tau = 0.1  # Near-greedy action selection

        # CSV logging
        log_path = os.environ.get("MCTS_LOG_CSV", "")
        log_file = None
        log_writer = None
        if log_path:
            log_file = open(log_path, "w", newline="")
            log_writer = csv.DictWriter(log_file, fieldnames=[
                "route_idx", "step", "route_length", "action_node",
                "n_valid_actions", "root_visits", "root_max_visits",
                "root_q_mean", "root_q_max", "n_rollouts", "step_time_s",
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
                    value = self._single_rollout(mcts_state)
                    tree.root.expand(priors, value)

                # Collect leaves needing rollouts, then batch-evaluate
                pending_leaves = []  # (node, leaf_state, path) tuples
                completed_sims = []  # sims that hit existing nodes (no rollout needed)

                for sim_idx in range(n_iter):
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

                    if not node.expanded and not sim_state.is_terminal():
                        leaf_valid = sim_state.get_valid_actions()
                        if leaf_valid:
                            # Queue for batch rollout
                            pending_leaves.append((node, sim_state, path, leaf_valid))
                        else:
                            # Dead end — still need rollout for value
                            pending_leaves.append((node, sim_state, path, None))
                    elif node.expanded:
                        # Re-visited an existing node — use its value
                        for parent, act in reversed(path):
                            parent.update(act, node.value)
                    else:
                        # Terminal state
                        pending_leaves.append((node, sim_state, path, None))

                    # Batch rollout when we have enough pending or at end
                    if len(pending_leaves) >= self.n_workers or sim_idx == n_iter - 1:
                        if pending_leaves:
                            leaf_states = [s for _, s, _, _ in pending_leaves]
                            rewards = self._batch_rollout(leaf_states)

                            for (node, sim_state, path, leaf_valid), reward in zip(pending_leaves, rewards):
                                if leaf_valid and not node.expanded:
                                    leaf_priors = {a: 1.0 / len(leaf_valid) for a in leaf_valid}
                                    node.expand(leaf_priors, reward)
                                elif not node.expanded:
                                    node.value = reward
                                    node.expanded = True

                                for parent, act in reversed(path):
                                    parent.update(act, reward)

                            pending_leaves = []

                # Select action from visit counts
                policy = tree.root.get_visit_count_policy(tau, n_nodes)
                action = max(valid, key=lambda a: policy[a])
                action_node = self.env.idx_to_node[action]

                # Log step
                step_time = time.time() - step_start
                n_rollouts_this_step = sum(tree.root.N.values()) if tree.root.N else 0
                if log_writer:
                    q_vals = list(tree.root.Q.values()) if tree.root.Q else [0]
                    log_writer.writerow({
                        "route_idx": k,
                        "step": step,
                        "route_length": len(current_route),
                        "action_node": action_node,
                        "n_valid_actions": len(valid),
                        "root_visits": sum(tree.root.N.values()) if tree.root.N else 0,
                        "root_max_visits": max(tree.root.N.values()) if tree.root.N else 0,
                        "root_q_mean": f"{np.mean(q_vals):.6f}",
                        "root_q_max": f"{max(q_vals):.6f}",
                        "n_rollouts": n_rollouts_this_step,
                        "step_time_s": f"{step_time:.2f}",
                    })
                    log_file.flush()

                print(f"    Route {k+1} step {step}: {action_node}, "
                      f"visits={n_rollouts_this_step}, "
                      f"time={step_time:.1f}s")

                # Advance
                current_route.append(action_node)
                mcts_state = mcts_state.apply_action(action)
                tree.advance(action)
                step += 1

            all_routes.append(current_route)
            elapsed = time.time() - episode_start
            print(f"  Route {k+1}/{num_routes} done: {len(current_route)} nodes, "
                  f"elapsed {elapsed:.0f}s")

        if log_file:
            log_file.close()

        total_time = time.time() - episode_start
        print(f"  Pure MCTS episode complete: {num_routes} routes in {total_time:.0f}s "
              f"({total_time/3600:.1f}h)")

        return all_routes

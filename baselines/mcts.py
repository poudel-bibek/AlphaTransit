"""
Pure MCTS Baseline for Transit Route Network Design.

Ablation baseline that isolates the contribution of tree search from the
learned neural components. Compares against:
- AlphaTransit (MCTS + learned policy P + learned value V)
- End-to-End RL (learned P + learned V, no search)
- Pure MCTS (MCTS + uniform P + rollout V)  <-- this file

Algorithm (per decision step):
1. EXPAND root with uniform priors P(a) = 1/|valid_actions| and a random
   rollout value (complete routes randomly, run full UXsim simulation).
2. For n_iter simulations:
   a. SELECT: walk down tree using PUCT (same formula as AlphaTransit).
   b. EXPAND leaf: uniform priors + full UXsim rollout for value.
   c. BACKUP: propagate reward up the path.
3. Pick action with highest visit count (near-greedy, tau=0.1).
4. Advance the same tree across the full episode; when a route reaches max
   length, MCTSState.apply_action() transitions into the next route state.

Rollouts are parallelized across --num_mcts_rollout_workers processes.
Each rollout creates an independent UXsim world (~8s per rollout on
Bloomington). By default, the baseline matches AlphaTransit's tuned n_iter and
c_puct for the current alpha unless the caller supplies non-default overrides.
Requires deterministic route initialization (e.g. 'transit_center' or
'highest_demand'). Random init makes successor states path-dependent,
invalidating tree statistics.

Note: unlike AlphaTransit's batched MCTS, this baseline does not use virtual
loss to diversify leaves within a rollout batch. That makes the baseline
slightly weaker than an optimized batched pure-MCTS implementation, which is a
conservative choice for ablation.

Usage:
    python main.py --mode baseline --baseline_type mcts --alpha 1.0 --n_iter 100
"""

import os
import csv
import random
import sys
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
        if config.get("route_init") == "random":
            raise ValueError(
                "route_init='random' is incompatible with MCTS. Random starts produce "
                "non-deterministic successor states, invalidating tree statistics. "
                "Use 'transit_center' or 'highest_demand' instead."
            )
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
        from baselines.utils import execute_runs
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

    def _resolve_search_params(self):
        """
        Match AlphaTransit's tuned search settings for the active alpha.

        Explicit CLI values take precedence over the tuned values.
        """
        from config import BEST_PARAMS

        cli_args = sys.argv[1:]
        cli_overrode_n_iter = any(
            arg == "--n_iter" or arg.startswith("--n_iter=")
            for arg in cli_args
        )
        cli_overrode_c_puct = any(
            arg == "--c_puct" or arg.startswith("--c_puct=")
            for arg in cli_args
        )

        config_n_iter = self.config.get("n_iter", 100)
        config_c_puct = self.config.get("c_puct", 1.0)

        alpha = self.config.get("alpha", 0.3)
        at_params = BEST_PARAMS.get("alphatransit", {}).get(alpha, {})

        if cli_overrode_n_iter:
            n_iter = config_n_iter
        else:
            n_iter = at_params.get("n_iter", config_n_iter)

        if cli_overrode_c_puct:
            c_puct = config_c_puct
        else:
            c_puct = at_params.get("c_puct", config_c_puct)

        return n_iter, c_puct

    def construct_path(self, state):
        from rl.mcts_utils import MCTSState, MCTSTree

        # Keep worker-side simulator seeds aligned with the active eval run.
        # env.reset(seed=X) updates env.config["seed"] per run (rl/env.py:583).
        self._config_dict["seed"] = self.env.config.get("seed", 42)

        n_iter, c_puct = self._resolve_search_params()
        num_routes = self.config.get("num_routes", 16)
        max_len = self.config.get("max_route_length", 14)
        n_nodes = len(self.env.node_to_idx)
        tau = 0.1  # Near-greedy action selection

        # CSV logging
        log_path = os.environ.get("MCTS_LOG_CSV", "")
        log_file = None
        log_writer = None
        if log_path:
            write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
            log_file = open(log_path, "a", newline="")
            log_writer = csv.DictWriter(log_file, fieldnames=[
                "route_idx", "step", "route_length", "action_node",
                "n_valid_actions", "root_visits", "root_max_visits",
                "root_q_mean", "root_q_max", "n_rollouts", "step_time_s",
            ])
            if write_header:
                log_writer.writeheader()

        print(f"  Pure MCTS search params: alpha={self.config.get('alpha')}, n_iter={n_iter}, c_puct={c_puct}")

        mcts_state = MCTSState(
            current_route=list(self.env.current_route),
            all_routes=[list(r) for r in self.env.all_routes],
            current_route_index=self.env.current_route_index,
            num_routes=num_routes,
            max_route_length=max_len,
            adj=self.env.adj,
            node_to_idx=self.env.node_to_idx,
            idx_to_node=self.env.idx_to_node,
            env=self.env,
        )
        tree = MCTSTree(mcts_state)
        episode_start = time.time()

        while not mcts_state.is_terminal():
            route_idx = mcts_state.current_route_index
            route_step = max(len(mcts_state.current_route) - 1, 0)
            step_start = time.time()
            valid = mcts_state.get_valid_actions()

            if not valid:
                dead_route = list(mcts_state.current_route)
                mcts_state = mcts_state.force_route_end()
                tree = MCTSTree(mcts_state)
                elapsed = time.time() - episode_start
                print(f"  Route {route_idx+1}/{num_routes} forced end: {len(dead_route)} nodes, "
                      f"elapsed {elapsed:.0f}s")
                continue

            # Expand root with uniform priors + rollout value
            if not tree.root.expanded:
                priors = {a: 1.0 / len(valid) for a in valid}
                value = self._single_rollout(mcts_state)
                tree.root.expand(priors, value)

            # Collect leaves needing rollouts, then batch-evaluate
            pending_leaves = []

            for sim_idx in range(n_iter):
                node = tree.root
                sim_state = mcts_state.clone()
                path = []

                # SELECT: walk down tree using PUCT
                while node.expanded and not sim_state.is_terminal():
                    action = node.select_action(c_puct)
                    if action is None:
                        break  # Dead-end node (no valid actions)
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
                    "route_idx": route_idx,
                    "step": route_step,
                    "route_length": len(mcts_state.current_route),
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

            print(f"    Route {route_idx+1} step {route_step}: {action_node}, "
                  f"visits={n_rollouts_this_step}, "
                  f"time={step_time:.1f}s")

            # Advance the same tree across route boundaries.
            prev_route_idx = mcts_state.current_route_index
            prev_route_len = len(mcts_state.current_route) + 1
            mcts_state = mcts_state.apply_action(action)
            tree.advance(action)

            if mcts_state.current_route_index > prev_route_idx:
                elapsed = time.time() - episode_start
                print(f"  Route {prev_route_idx+1}/{num_routes} done: {prev_route_len} nodes, "
                      f"elapsed {elapsed:.0f}s")

        if log_file:
            log_file.close()

        total_time = time.time() - episode_start
        print(f"  Pure MCTS episode complete: {num_routes} routes in {total_time:.0f}s "
              f"({total_time/3600:.1f}h)")

        self.env.all_routes = [list(r) for r in mcts_state.all_routes]
        return mcts_state.all_routes

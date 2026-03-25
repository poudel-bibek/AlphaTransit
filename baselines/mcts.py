"""
Pure MCTS Baseline for Transit Route Network Design.

This baseline runs Monte Carlo Tree Search WITHOUT a learned policy-value
network. It uses uniform random priors and rollout-based value estimation
instead of the GNN policy head (P) and value head (V) used by AlphaTransit.

This serves as an ablation baseline to isolate the contribution of MCTS
search from the learned neural components. Comparing:
- AlphaTransit (MCTS + learned P + learned V)
- Pure MCTS (MCTS + uniform P + rollout V)
- End-to-End RL (learned P + learned V, no MCTS)

demonstrates how much each component contributes to performance.

Usage:
    python main.py --mode baseline --baseline_type mcts --alpha 0.3
"""

# TODO: Implement PureMCTS baseline
#
# The implementation should:
# 1. Use the same MCTSTree/MCTSNode/MCTSState infrastructure from rl/mcts_utils.py
# 2. Replace the neural network policy with uniform random priors over valid actions
# 3. Replace the neural network value estimate with rollout-based evaluation
#    (random playout to terminal state, use simulation reward as value)
# 4. Use the same MCTS hyperparameters (n_iter, c_puct) as AlphaTransit
# 5. Follow the same baseline interface: __init__(env, config, num_runs, base_seed) + run() + construct_path(state)
# 6. Output routes in the same List[List[str]] format
#
# Key design decisions:
# - Rollout policy: uniform random over valid neighbors (simplest)
# - Value backup: mean return from K random rollouts per leaf
# - Temperature schedule: same as AlphaTransit training
# - Number of simulations per move: same n_iter as AlphaTransit
#
# This will be slow (rollouts require stepping the environment) but provides
# a principled ablation of the neural components.


class PureMCTS:
    """
    Pure MCTS baseline without learned policy-value network.

    Uses uniform priors and rollout-based value estimation.
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

    def construct_path(self, state):
        # TODO: Implement pure MCTS route construction
        # 1. For each route k in range(K):
        #    a. Initialize route at transit center (node 96)
        #    b. While route length < L_max and valid neighbors exist:
        #       - Build MCTS tree from current state
        #       - Run n_iter simulations with uniform priors + random rollouts
        #       - Select action by visit count (same as AlphaTransit)
        #       - Advance state
        #    c. Finalize route
        # 2. Return all K routes as List[List[str]]
        raise NotImplementedError(
            "PureMCTS baseline is not yet implemented. "
            "This is a skeleton for the MCTS ablation experiment."
        )

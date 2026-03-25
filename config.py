"""
Shared configuration for PPO and MCTS training.
"""

import os
import random
import argparse
import numpy as np
import torch
from typing import Any, Dict


# Best hyperparameters found from sweep experiments.
# PPO: from term_ep_1 (alpha=0.3, 18 runs) and term_ep_2 (alpha=1.0, 32 runs)
# MCTS: from 3_faster_general (alpha=0.3, 21 runs) and 4_faster_general (alpha=1.0, 24 runs)
# Best PPO reward mode: terminal_intermediate_delta_no_early_stop (both alphas)
BEST_PARAMS = {
    "ppo": {
        0.3: {
            "lr": 5e-5,
            "anneal_lr": False,
            "K_epochs": 8,
            "num_gat_blocks": 4,
            "batch_size": 256,
            "clip_frac": 0.2,
            "entropy_coef": 0.01,
            "activation": "tanh",
            "ppo_reward_mode": "terminal_intermediate_delta_no_early_stop",
        },
        1.0: {
            "lr": 1e-5,
            "anneal_lr": True,
            "K_epochs": 4,
            "num_gat_blocks": 4,
            "batch_size": 128,
            "clip_frac": 0.1,
            "entropy_coef": 0.02,
            "activation": "tanh",
            "ppo_reward_mode": "terminal_intermediate_delta_no_early_stop",
        },
    },
    "alpha": {
        0.3: {
            "lr": 1e-4,
            "c_puct": 1.5,
            "n_iter": 100,
            "num_gat_blocks": 8,
            "batch_size": 256,
            "buffer_capacity": 50000,
            "activation": "tanh",
            "dirichlet_alpha": 0.3,
            "train_steps_per_iter": 200,
        },
        1.0: {
            "lr": 1e-4,
            "c_puct": 1.0,
            "n_iter": 100,
            "num_gat_blocks": 4,
            "batch_size": 256,
            "buffer_capacity": 50000,
            "activation": "tanh",
            "dirichlet_alpha": 0.3,
            "train_steps_per_iter": 200,
        },
    },
}


def apply_best_params(config: Dict[str, Any], explicitly_set: set = None) -> Dict[str, Any]:
    """
    Override argparse defaults with best sweep params based on algorithm and alpha.
    Only applies if --algorithm and --alpha are set and a match exists in BEST_PARAMS.
    Keys explicitly passed on the CLI are never overridden.
    """
    algo = config.get("algorithm")
    alpha = config.get("alpha")
    if algo in BEST_PARAMS and alpha in BEST_PARAMS[algo]:
        best = BEST_PARAMS[algo][alpha]
        for k, v in best.items():
            if explicitly_set and k in explicitly_set:
                continue  # CLI override takes precedence
            config[k] = v
    return config


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


def get_config() -> Dict[str, Any]:
    """
    A unified config interface for both main and sweep.
    Does NOT set seeds or device here to allow overrides from sweep.

    Applies best sweep params for the given algorithm+alpha, but any
    explicitly passed CLI args take precedence over best params.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    # Track which args were explicitly passed on the command line
    explicitly_set = {action.dest for action in parser._actions
                      if action.dest in args and
                      getattr(args, action.dest) != action.default}
    config = vars(args)
    if config.get("apply_best_params"):
        config = apply_best_params(config, explicitly_set)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for CLI usage.
    Define common RL, system, and sweepable hyperparameters.
    """
    parser = argparse.ArgumentParser(description="RL training/evaluation entrypoint")
    parser.add_argument("--algorithm", choices=["ppo", "alpha"], default=None, help="Learning algorithm (required for train/eval modes)")
    parser.add_argument("--apply_best_params", action="store_true", help="Apply best hyperparameters from sweep experiments for the given algorithm+alpha")

    # Simulation setup:
    parser.add_argument("--network", choices=["sioux_falls", "bloomington",], default="bloomington", help="Network selection")
    parser.add_argument("--mode", choices=["train", "eval", "baseline"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available. Pass --gpu to enable.")
    parser.add_argument("--horizon", type=int, default=10000, help="Simulation horizon")
    parser.add_argument("--delta_t", type=float, default=1, help="Simulation time step")
    parser.add_argument("--delta_n", type=int, default=5, help="Simulation platoon size")
    parser.add_argument("--bus_capacity", type=int, default=40, help="Bus capacity")
    parser.add_argument("--stop_duration", type=int, default=60, help="Stop duration")
    parser.add_argument("--baseline_type", type=str, default="demand_cover", help="Can be random_walk, reward_max, demand_cover, shortest_path, real_world, genetic, evolutionary, neural_evolutionary, mcts")
    parser.add_argument("--num_eval_runs", type=int, default=5, help="Number of evaluation runs")
    parser.add_argument("--eval_seed_offset", type=int, default=2, help="Add offset to starting seed for evaluation outputs")
    parser.add_argument("--save_animations", action="store_true", help="Save animations for evaluation")

    # Genetic Algorithm hyperparameters:
    parser.add_argument("--ga_population", type=int, default=50, help="GA: Population size")
    parser.add_argument("--ga_generations", type=int, default=100, help="GA: Number of generations")
    parser.add_argument("--ga_mutation_rate", type=float, default=0.4, help="GA: Mutation probability")
    parser.add_argument("--ga_crossover_rate", type=float, default=0.8, help="GA: Crossover probability")
    parser.add_argument("--ga_tournament_size", type=int, default=3, help="GA: Tournament selection size")
    parser.add_argument("--ga_elitism", type=int, default=5, help="GA: Number of elite individuals to preserve")
    parser.add_argument("--ga_num_workers", type=int, default=4, help="GA: Number of parallel workers for fitness evaluation (1=sequential)")

    # Learning environment specific:
    parser.add_argument("--service_frequency_mode", type=str, default="max_load", help="Service frequency mode")
    parser.add_argument("--stop_spacing", type=int, default=1, help="Stop spacing. 1 means every node is a stop")
    parser.add_argument("--alpha", type=float, default=0.3, help="Modal split parameter for served O-D pairs")
    parser.add_argument("--ignore_unserved", action="store_true", help="Ignore unserved demand")
    parser.add_argument("--comfort_threshold", type=float, default=1.0, help="Max load factor for service frequency")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius for demand allocation")
    parser.add_argument("--demand_warmup", type=float, default=0.15, help="Fraction of horizon reserved at both start and end")
    parser.add_argument("--route_init", type=str, default="transit_center", help="Route initialization scheme")
    parser.add_argument("--transit_center_node", type=str, default="96", help="Transit center node identifier")
    parser.add_argument("--ppo_reward_mode", type=str, default="terminal_intermediate_delta_no_early_stop",
        choices=["terminal_only", "terminal_intermediate_raw_early_stop", "terminal_intermediate_delta_early_stop", "terminal_intermediate_delta_no_early_stop"],
        help="Reward shaping mode (PPO only; MCTS uses terminal reward targets)")

    # Constraints:
    parser.add_argument("--num_routes", type=int, default=16, help="Number of routes")
    parser.add_argument("--max_route_length", type=int, default=14, help="Maximum route length")
    parser.add_argument("--min_route_length", type=int, default=2, help="Minimum route length")

    # PPO hyperparameters:
    # Training duration: max_steps = 1M env steps (comparable to MCTS with max_iterations=744)
    # Eval frequency is update-count based and depends on episode lengths × ppo_episodes_per_update.
    parser.add_argument("--max_steps", type=int, default=1_000_000, help="PPO: Total training steps")
    parser.add_argument("--ppo_eval_every", type=int, default=5, help="PPO: Evaluate every N updates")
    parser.add_argument("--num_ppo_workers", type=int, default=8, help="PPO: Number of parallel workers")
    parser.add_argument("--ppo_episodes_per_update", type=int, default=8,
                        help="PPO: Episodes to collect before each update (decoupled from worker count)")
    parser.add_argument("--K_epochs", type=int, default=8, help="PPO: Number of epochs per update")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--clip_frac", type=float, default=0.2, help="PPO: Clipping ratio for policy loss")
    parser.add_argument("--vf_clip_param", type=float, default=0.5, help="PPO: Clipping ratio for value loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO: Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="PPO: GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="PPO: Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="PPO: Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_gat_blocks", type=int, default=4, help="Number of GAT attention blocks")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "leaky_relu"], help="Activation function")
    parser.add_argument("--anneal_lr", action="store_true", help="PPO: Anneal learning rate")
    parser.add_argument("--save_policy_ppo", action="store_true", help="PPO: Save policy checkpoints to disk")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="PPO: Minimum learning rate floor when annealing")
    parser.add_argument("--ppo_eval_temperature", type=float, default=0.1, help="PPO: Temperature for near-greedy eval sampling (matches MCTS eval tau)")

    # MCTS hyperparameters:
    # Training duration: 6 workers × ~224 steps/episode = ~1,344 steps/iteration
    #                    max_iterations=744 × 1,344 ≈ 1M steps (comparable to PPO)
    # Eval frequency: 744 iterations / eval_every=3 → ~248 eval points
    parser.add_argument("--n_iter", type=int, default=100, help="MCTS: Simulations per move")
    parser.add_argument("--mcts_batch_size", type=int, default=8, help="MCTS: Leaves to batch per NN forward pass (virtual loss)")
    parser.add_argument("--c_puct", type=float, default=1.0, help="MCTS: PUCT exploration constant")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="MCTS: Dirichlet noise concentration")
    parser.add_argument("--dirichlet_eps", type=float, default=0.25, help="MCTS: Dirichlet noise weight")
    parser.add_argument("--buffer_capacity", type=int, default=50000, help="MCTS: Replay buffer capacity")
    parser.add_argument("--num_mcts_workers", type=int, default=16, help="MCTS: Number of parallel workers")
    parser.add_argument("--episodes_per_iter", type=int, default=16, help="MCTS: Episodes to collect per iteration")
    parser.add_argument("--train_steps_per_iter", type=int, default=200, help="MCTS: Training steps per iteration")
    parser.add_argument("--max_iterations", type=int, default=308, help="MCTS: Max iterations (308 × 16 eps × ~203 steps ≈ 1.0M steps)")
    parser.add_argument("--mcts_eval_every", type=int, default=5, help="MCTS: Evaluate every N iterations")
    parser.add_argument("--temp_schedule", type=str, default="0.7:1.0,0.9:0.7,1.0:0.5", help="MCTS: Temperature schedule as 'progress:tau' pairs (e.g., '0.7:1.0,0.9:0.7,1.0:0.5')")

    # Model:
    parser.add_argument("--concat_heads", action="store_true", help="Concatenate attention heads")

    # WandB:
    parser.add_argument("--wandb_project", type=str, default="transit_design", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="bibek-poudel", help="WandB entity/team name")
    parser.add_argument("--wandb_off", action="store_true", help="Disable WandB logging")

    # Paths:
    parser.add_argument("--save_dir", type=str, default="./training_data", help="Directory to save training data")
    parser.add_argument("--saved_policy_path", type=str, default="./training_data/policies/policy_final.pth", help="Path to saved policy for evaluation")

    return parser

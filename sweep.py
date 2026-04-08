import gc
import torch
import wandb
import argparse
from typing import Any, Dict
from config import get_config, set_global_seeds, apply_best_params, BEST_PARAMS
from ppo import train as ppo_train
from alpha import train as alpha_train


def build_sweep_config_ppo_0_3() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep search space for Alpha = 0.3.

    Key takeaways from PPO sweep (18 runs, by peak eval/episode_terminal_reward):
    - lr 5e-5 (18.4) >> 1e-4 (17.1) >> 3e-4 (15.1) — lower LR better at alpha=0.3
    - K_epochs 8 (18.5) > 4 (16.8)
    - num_gat_blocks 4 (18.6) > 8 (17.2)
    - anneal_lr False (18.7) > True (17.2)
    - batch_size 256 ≈ 128 — minimal differentiation
    - All runs degrade from peak (mean drop 3.7 pts)
    """

    # =====================================================================
    # 1. General Sweep for PPO Alpha 0.3
    # =====================================================================
    # return {
    #     "method": "bayes",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         # Sweep params (3 × 2 × 2 × 2 × 2 = 48 combinations)
    #         "lr": {"values": [5e-5, 1e-4, 3e-4]},
    #         "anneal_lr": {"values": [True, False]},
    #         "K_epochs": {"values": [4, 8]},
    #         "num_gat_blocks": {"values": [4, 8]},
    #         "batch_size": {"values": [128, 256]},
    #         "clip_frac": {"value": 0.2},
    #         "entropy_coef": {"value": 0.01},
    #         "activation": {"value": "tanh"},
    #
    #         # Fixed values (Not sweep params)
    #         "alpha": {"value": 0.3},
    #         "algorithm": {"value": "ppo"},
    #         "gpu": {"value": True},
    #         "num_ppo_workers": {"value": 8},
    #         "ppo_eval_every": {"value": 5},
    #     },
    # }

    # =====================================================================
    # 5. Reward Ablation Sweep for PPO Alpha 0.3
    # Best params are in config.py BEST_PARAMS
    # =====================================================================
    return {
        "method": "grid",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },
        "parameters": {
            "seed": {"values": [42, 123]},
            "ppo_reward_mode": {"values": [
                "terminal_only",
                "terminal_intermediate_raw_early_stop",
                "terminal_intermediate_delta_early_stop",
                "terminal_intermediate_delta_no_early_stop",
            ]},
            "lr": {"value": 5e-5},
            "anneal_lr": {"value": False},
            "K_epochs": {"value": 8},
            "num_gat_blocks": {"value": 4},
            "batch_size": {"value": 256},
            "clip_frac": {"value": 0.2},
            "entropy_coef": {"value": 0.01},
            "activation": {"value": "tanh"},

            # Fixed values (Not sweep params)
            "alpha": {"value": 0.3},
            "algorithm": {"value": "ppo"},
            "gpu": {"value": True},
            "num_ppo_workers": {"value": 8},
            "ppo_episodes_per_update": {"value": 8},
            "ppo_eval_every": {"value": 5},
            "save_policy_ppo": {"value": True},
        },
    }


def build_sweep_config_ppo_1_0() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep search space for Alpha = 1.0.

    Key takeaways from PPO sweep (Feb 13-17, 32 runs, by peak eval/episode_terminal_reward):
    - batch_size 128 (38.2) >> 256 (36.7) — strongest differentiator
    - K_epochs 4 (38.0) > 2 (37.1)
    - lr 1e-5 (38.1) ≈ 3e-5 (38.1) > 5e-6 (36.6) — too low hurts
    - num_gat_blocks 4 (38.1) > 8 (37.6) — mild advantage
    - clip_frac and anneal_lr showed minimal differentiation
    - All runs degrade from peak (mean drop 9 pts) — training unstable late
    """

    # =====================================================================
    # 2. General Sweep for PPO Alpha 1.0
    # =====================================================================
    # return {
    #     "method": "bayes",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         # Sweep params (3 × 2 × 2 × 2 × 2 × 2 = 96 combinations)
    #         "lr": {"values": [5e-6, 1e-5, 3e-5]},
    #         "anneal_lr": {"values": [True, False]},
    #         "K_epochs": {"values": [2, 4]},
    #         "num_gat_blocks": {"values": [4, 8]},
    #         "batch_size": {"values": [128, 256]},
    #         "clip_frac": {"values": [0.1, 0.2]},
    #         "entropy_coef": {"value": 0.02},
    #         "activation": {"value": "tanh"},
    #
    #         # Fixed values (Not sweep params)
    #         "alpha": {"value": 1.0},
    #         "algorithm": {"value": "ppo"},
    #         "gpu": {"value": True},
    #         "num_ppo_workers": {"value": 8},
    #         "ppo_eval_every": {"value": 5},
    #     },
    # }

    # =====================================================================
    # 6. Reward Ablation Sweep for PPO Alpha 1.0
    # Best params are in config.py BEST_PARAMS
    # =====================================================================
    return {
        "method": "grid",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },
        "parameters": {
            "seed": {"values": [42, 123]},
            "ppo_reward_mode": {"values": [
                "terminal_only",
                "terminal_intermediate_raw_early_stop",
                "terminal_intermediate_delta_early_stop",
                "terminal_intermediate_delta_no_early_stop",
            ]},
            "lr": {"value": 1e-5},
            "anneal_lr": {"value": True},
            "K_epochs": {"value": 4},
            "num_gat_blocks": {"value": 4},
            "batch_size": {"value": 128},
            "clip_frac": {"value": 0.1},
            "entropy_coef": {"value": 0.02},
            "activation": {"value": "tanh"},

            # Fixed values (Not sweep params)
            "alpha": {"value": 1.0},
            "algorithm": {"value": "ppo"},
            "gpu": {"value": True},
            "num_ppo_workers": {"value": 8},
            "ppo_episodes_per_update": {"value": 8},
            "ppo_eval_every": {"value": 5},
            "save_policy_ppo": {"value": True},
        },
    }


def build_ppo_sweep_config(alpha: float) -> Dict[str, Any]:
    """
    PPO hyperparameter sweep - selects config based on alpha.
    """
    if alpha == 0.3:
        return build_sweep_config_ppo_0_3()
    elif alpha == 1.0:
        return build_sweep_config_ppo_1_0()
    else:
        raise ValueError(f"No PPO sweep config for alpha={alpha}")


def build_sweep_config_alpha_0_3() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep search space for Alpha = 0.3.

    Key takeaways from MCTS sweep 3 (21 runs, by peak eval/episode_terminal_reward):
    NOTE: These sweeps used the old reward (before hacking fix), but since the reward
    change is minor (service_rate denominator only), these results should still hold.

    - c_puct 1.0 (18.4) > 1.5 (17.6) — c_puct=1.5 has high variance (std=1.55)
    - num_gat_blocks 8 (18.1) > 4 (17.6) — mild advantage
    - train_steps_per_iter 100 (18.4) > 200 (17.8) — but only 5 runs at 100
    - temp_schedule: slow anneal (18.0) ≈ constant (17.8) — minimal differentiation
    - Best run: spring-sweep-6, peak=19.38 (c_puct=1.5, const temp, 8 blocks, 200 steps)
    - Runs are stable: mean drop from peak only 1.15 pts

    Key takeaways from early MCTS sweep (Jan 10, 3 runs, before code optimization for faster training):
    1. lr: Tried 1e-05 and 0.0001. Higher is better — 0.0001 used in 100% of top runs. Impact: +8pp service rate, -11 min wait time.
    2. n_iter: Tried 50 and 100. Higher is better — 100 simulations used in 100% of top runs. Impact:  +8pp service rate improvement.
    3. num_gat_blocks: Tried 4 and 8. Higher is better — 8 blocks used in 100% of top runs. Impact:    2.5x better route efficiency.

    Insights from Jan 29 (API analysis, before code optimization for faster training):
    1. Fast temp schedules kill learning — when temp=0.1, policy_loss drops to ~0.002 (near-zero gradients).
    2. peach-sweep-1 peaked at 39.5% progress (temp=1.0), then flat for remaining 60% of training.
    3. Slower schedules (0.6:1.0,0.85:0.5,1.0:0.1) maintain policy_loss ~0.2-0.3, enabling continued learning.
    """

    # =====================================================================
    # 3. General Sweep for MCTS Alpha 0.3 (completed)
    # =====================================================================
    # return {
    #     "method": "bayes",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         "temp_schedule": {"values": [
    #             "1.0:1.0",
    #             "0.7:1.0,0.9:0.7,1.0:0.5",
    #         ]},
    #         "c_puct": {"values": [1.0, 1.5]},
    #         "train_steps_per_iter": {"values": [100, 200]},
    #         "num_gat_blocks": {"values": [4, 8]},
    #         "n_iter": {"value": 100},
    #         "lr": {"value": 1e-4},
    #         "batch_size": {"value": 256},
    #         "buffer_capacity": {"value": 50000},
    #         "activation": {"value": "tanh"},
    #         "dirichlet_alpha": {"value": 0.3},
    #         "alpha": {"value": 0.3},
    #         "algorithm": {"value": "alphatransit"},
    #         "gpu": {"value": True},
    #         "num_mcts_workers": {"value": 8},
    #         "mcts_eval_every": {"value": 5},
    #     },
    # }

    # =====================================================================
    # 7. n_iter Scaling Sweep for MCTS Alpha 0.3 (completed: oq81en7b)
    # All other params come from BEST_PARAMS via apply_best_params
    # Old (8 blocks): lpg2ifh2 (n=100), j5ns6bh8 (n=200,300,400), 466yjq3q (n=500)
    # =====================================================================
    # return {
    #     "method": "grid",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         "n_iter": {"values": [100, 200, 300, 400, 500]},
    #         "alpha": {"value": 0.3},
    #         "algorithm": {"value": "alphatransit"},
    #         "gpu": {"value": True},
    #         "apply_best_params": {"value": True},
    #         "num_mcts_workers": {"value": 8},
    #     },
    # }

    # =====================================================================
    # 9. episodes_per_iter Scaling Sweep for MCTS Alpha 0.3
    # n_iter=200, fixed 616 iterations (= eps=8 baseline) so all configs
    # get the same number of gradient updates. Total env steps scale with eps.
    # eps=8: ~1M steps, eps=16: ~2M, eps=24: ~3M, eps=32: ~4M
    # =====================================================================
    return {
        "method": "grid",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },
        "parameters": {
            "episodes_per_iter": {"values": [8, 16, 24, 32]},
            "n_iter": {"value": 200},
            "alpha": {"value": 0.3},
            "algorithm": {"value": "alphatransit"},
            "gpu": {"value": True},
            "apply_best_params": {"value": True},
            "num_mcts_workers": {"value": 8},
        },
    }

    # =====================================================================
    # 11. Model Size Sweep for MCTS Alpha 0.3
    # n_iter=200, eps=16 (308 iters, ~1M env steps)
    # Vary num_gat_blocks: 2, 4, 8, 16
    # =====================================================================
    # return {
    #     "method": "grid",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         "num_gat_blocks": {"values": [16, 8, 4, 2]},
    #         "n_iter": {"value": 200},
    #         "episodes_per_iter": {"value": 16},
    #         "alpha": {"value": 0.3},
    #         "algorithm": {"value": "alphatransit"},
    #         "gpu": {"value": True},
    #         "apply_best_params": {"value": True},
    #         "num_mcts_workers": {"value": 8},
    #     },
    # }


def build_sweep_config_alpha_1_0() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep search space for Alpha = 1.0.

    Key takeaways from MCTS sweep 4 (24 runs, by peak eval/episode_terminal_reward):
    NOTE: These sweeps used the old reward (before hacking fix), but since the reward
    change is minor (service_rate denominator only), these results should still hold.

    - c_puct 1.0 (38.0) >> 1.5 (35.3) — strongest differentiator, c_puct=1 dominates
    - temp_schedule: slow anneal (37.4) > constant (36.6) — mild advantage
    - num_gat_blocks 4 (37.3) ≈ 8 (36.9) — minimal differentiation
    - train_steps_per_iter 100 (37.7) > 200 (36.5) — fewer steps per iter better
    - Best run: distinctive-sweep-20, peak=41.01 (c_puct=1, const temp, 4 blocks, 200 steps)
    - Runs are stable: mean drop from peak only 1.04 pts
    - Top 5 runs are ALL c_puct=1.0

    Key takeaways from early MCTS sweep (Jan 22, 5 runs, before code optimization for faster training):
    1. num_gat_blocks: Tried 4 and 8. Lower is better — 4 blocks used in 100% of top 2 runs. Impact: +3.4 reward (+11%), -7.8 min travel time (-20%), -22 buses fleet size (-23%).
    2. c_puct: Tried 1.0 and 1.5. Higher is slightly better — 1.5 used in 50% of top 2 runs. Impact: +0.6 to +1.0 reward (controlled comparison), -1.9 min travel time.

    Insights from Jan 29 (API analysis, before code optimization for faster training):
    1. Fast temp schedules kill learning — when temp=0.1, policy_loss drops to ~0.007 (near-zero gradients).
    2. zany-sweep-1: policy_loss=0.79→0.02→0.0 as temp dropped 1.0→0.5→0.1. Reward plateaued at 35.68.
    3. glorious-sweep-1 (slower schedule 0.6:1.0,0.85:0.5,1.0:0.1): policy_loss stayed ~0.2-0.3, got better peak (36.91).
    """

    # =====================================================================
    # 4. General Sweep for MCTS Alpha 1.0 (completed)
    # =====================================================================
    # return {
    #     "method": "bayes",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         "temp_schedule": {"values": [
    #             "1.0:1.0",
    #             "0.7:1.0,0.9:0.7,1.0:0.5",
    #         ]},
    #         "c_puct": {"values": [1.0, 1.5]},
    #         "train_steps_per_iter": {"values": [100, 200]},
    #         "num_gat_blocks": {"values": [4, 8]},
    #         "n_iter": {"value": 100},
    #         "lr": {"value": 1e-4},
    #         "batch_size": {"value": 256},
    #         "buffer_capacity": {"value": 50000},
    #         "activation": {"value": "tanh"},
    #         "dirichlet_alpha": {"value": 0.3},
    #         "alpha": {"value": 1.0},
    #         "algorithm": {"value": "alphatransit"},
    #         "gpu": {"value": True},
    #         "num_mcts_workers": {"value": 8},
    #         "mcts_eval_every": {"value": 5},
    #     },
    # }

    # =====================================================================
    # 8. n_iter Scaling Sweep for MCTS Alpha 1.0
    # All other params come from BEST_PARAMS via apply_best_params
    # =====================================================================
    # return {
    #     "method": "grid",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         "n_iter": {"values": [100, 200, 300, 400, 500]},
    #         "alpha": {"value": 1.0},
    #         "algorithm": {"value": "alphatransit"},
    #         "gpu": {"value": True},
    #         "apply_best_params": {"value": True},
    #         "num_mcts_workers": {"value": 16},
    #     },
    # }

    # =====================================================================
    # 10. episodes_per_iter Scaling Sweep for MCTS Alpha 1.0
    # n_iter=200, fixed 616 iterations (= eps=8 baseline) so all configs
    # get the same number of gradient updates. Total env steps scale with eps.
    # eps=8: ~1M steps, eps=16: ~2M, eps=24: ~3M, eps=32: ~4M
    # =====================================================================
    return {
        "method": "grid",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },
        "parameters": {
            "episodes_per_iter": {"values": [8, 16, 24, 32]},
            "n_iter": {"value": 200},
            "alpha": {"value": 1.0},
            "algorithm": {"value": "alphatransit"},
            "gpu": {"value": True},
            "apply_best_params": {"value": True},
            "num_mcts_workers": {"value": 16},
        },
    }

    # =====================================================================
    # 12. Model Size Sweep for MCTS Alpha 1.0
    # n_iter=200, eps=16 (308 iters, ~1M env steps)
    # Vary num_gat_blocks: 2, 4, 8, 16
    # =====================================================================
    # return {
    #     "method": "grid",
    #     "metric": {
    #         "name": "eval/episode_terminal_reward",
    #         "goal": "maximize"
    #     },
    #     "parameters": {
    #         "num_gat_blocks": {"values": [16, 8, 4, 2]},
    #         "n_iter": {"value": 200},
    #         "episodes_per_iter": {"value": 16},
    #         "alpha": {"value": 1.0},
    #         "algorithm": {"value": "alphatransit"},
    #         "gpu": {"value": True},
    #         "apply_best_params": {"value": True},
    #         "num_mcts_workers": {"value": 16},
    #     },
    # }


def build_alpha_sweep_config(alpha: float) -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep - selects config based on alpha.
    """
    if alpha == 0.3:
        return build_sweep_config_alpha_0_3()
    elif alpha == 1.0:
        return build_sweep_config_alpha_1_0()
    else:
        raise ValueError(f"No AlphaTransit sweep config for alpha={alpha}")


def get_sweep_config(algorithm: str, alpha: float) -> Dict[str, Any]:
    """
    Get sweep configuration for the specified algorithm and alpha.
    """
    if algorithm == "ppo":
        return build_ppo_sweep_config(alpha)
    elif algorithm == "alphatransit":
        return build_alpha_sweep_config(alpha)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'ppo', 'alphatransit'")

def get_train_fn(algorithm: str):
    """
    Get the training function for the specified algorithm.
    """
    if algorithm == "ppo":
        return ppo_train
    elif algorithm == "alphatransit":
        return alpha_train
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'ppo', 'alphatransit'")

def create_agent_train(algorithm: str):
    """
    Create a training function for wandb.agent to call for each sweep run.
    Returns a closure that captures the algorithm choice.
    """
    train_fn = get_train_fn(algorithm)

    def agent_train() -> None:
        """
        Function called by wandb.agent for each sweep run.
        Initializes WandB run, merges configs, runs training.
        """
        with wandb.init() as run:
            # Get sampled params from sweep
            sampled_params = dict(wandb.config)

            # Get defaults and merge with sampled
            base_config = get_config()
            config = {**base_config, **sampled_params}
            config["algorithm"] = algorithm

            # Apply best params if requested, but sweep params take precedence
            if sampled_params.get("apply_best_params"):
                config = apply_best_params(config, explicitly_set=set(sampled_params.keys()))

            # Auto-compute max_iterations for episodes_per_iter scaling sweeps
            # Fixed at 616 iterations (eps=8 baseline) so all configs get
            # the same number of gradient updates. Env steps scale with eps.
            if 'episodes_per_iter' in sampled_params:
                config['max_iterations'] = 616

            # Set seeds and device
            set_global_seeds(config["seed"])
            device = torch.device("cuda" if config.get("gpu", True) and torch.cuda.is_available() else "cpu")
            config["device"] = device

            # Run training with is_sweep=True (wandb already initialized by sweep context)
            train_fn(config, is_sweep=True)

        # Cleanup between sweep runs
        gc.collect()

    return agent_train

def main() -> None:
    """
    Usage:
        python sweep.py --algorithm ppo
        python sweep.py --algorithm alphatransit
    """
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for transit design RL")
    parser.add_argument("--algorithm", choices=["ppo", "alphatransit"], required=True,
                        help="Algorithm to sweep (required)")
    parser.add_argument("--alpha", type=float, choices=[0.3, 1.0], required=True,
                        help="Modal split alpha value (required)")
    args = parser.parse_args()

    # Get configs
    sweep_config = get_sweep_config(args.algorithm, args.alpha)
    base_config = get_config()

    print(f"Starting {args.algorithm.upper()} alpha={args.alpha} {sweep_config['method']} sweep (Ctrl+C to stop)...")

    # Create and run sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=base_config["wandb_project"],
        entity=base_config["wandb_entity"]
    )

    agent_train_fn = create_agent_train(args.algorithm)
    wandb.agent(sweep_id, function=agent_train_fn)  # No count = runs indefinitely


if __name__ == "__main__":
    main()

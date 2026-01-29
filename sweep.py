import gc
import torch
import wandb
import argparse
from typing import Any, Dict
from config import get_config, set_global_seeds
from ppo import train as ppo_train
from mcts import train as mcts_train


def build_sweep_config_ppo_0_3() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep search space for Alpha = 0.3.

    Key takeaways from PPO sweep in Jan 5:
    Alpha = 0.3 (10 runs)
    1. K_epochs: Tried 2 and 4. Higher is better — optimizer chose 4 in 60% of runs. Impact: +55 reward.
    2. batch_size: Tried 16 and 32. Higher is better — optimizer chose 32 in 80% of runs. Impact: +55 reward.
    3. lr: Tried 1e-05 and 1e-04. Higher is better — optimizer chose 1e-04 in 90% of runs. Impact: +52 reward.

    # Further insights from Jan 10 (27 runs):
    1. activation: Tried leaky_relu and tanh. tanh is better — appears in 54% of top runs. Impact: 4.67 return_std improvement.
    2. update_frequency: Tried 128 and 256. Lower is better — 128 appears in 38% of top runs (but avg is clearly better). Impact: 4.14 return_std improvement.
    3. clip_frac: Tried 0.1 and 0.2. Higher is better — 0.2 appears in 69% of top runs. Impact: 2.22 return_std improvement.

    # Insights from Jan 29 (API analysis):
    1. LR annealing to 1e-6 kills learning — approx_kl drops to 0.00017 (policy barely updates), no clipping occurs.
    2. Need higher LR or anneal=False or higher min_lr floor. Learning is starved with current settings.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },

        # =====================================================================
        # 1. General Sweep for PPO Alpha 0.3
        # =====================================================================
        "parameters": {
            # Sweep params (3 × 2 × 2 × 2 × 2 = 48 combinations, Bayesian samples 32)
            "lr": {"values": [5e-5, 1e-4, 3e-4]},
            "anneal_lr": {"values": [True, False]},
            "K_epochs": {"values": [4, 8]},
            "num_gat_blocks": {"values": [4, 8]},
            "batch_size": {"values": [32, 64]},

            # Fixed params
            "clip_frac": {"value": 0.2},
            "entropy_coef": {"value": 0.01},
            "update_frequency": {"value": 128},
            "activation": {"value": "tanh"},

            # Fixed values (Not sweep params)
            # Training duration: 1M env steps (comparable to MCTS with max_iterations=744)
            # Eval frequency: 7,812 updates / 20 = ~390 eval points
            "alpha": {"value": 0.3}, # ALPHA SETTING.
            "algorithm": {"value": "ppo"},
            "gpu": {"value": True},
            "max_steps": {"value": 750_000},
            "num_ppo_workers": {"value": 8},
            "ppo_eval_every": {"value": 20},
        },
    }


def build_sweep_config_ppo_1_0() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep search space for Alpha = 1.0.

    Key takeaways from PPO sweep in Jan 5:
    Alpha = 1.0 (39 runs)
    1. lr: Tried 1e-05 and 1e-04. Lower is better — optimizer chose 1e-05 in 95% of runs. Impact: +119 reward.
    2. batch_size: Tried 16 and 32. Higher is better — optimizer chose 32 in 95% of runs. Impact: +84 reward.
    3. update_frequency: Tried 128 and 256. Lower is better — optimizer chose 128 in 90% of runs. Impact: +81 reward.

    # Further insights from Jan 10 (124 runs):
    1. Reward function is most strongly driven by coverage metrics. Node and demand coverage are the strongest predictor of reward. Demand Coverage (correlation = 0.822), Node Coverage (correlation = 0.775)
    2. K_epochs: Tried 2 and 4. Lower is better — optimizer chose 2 in 87% of runs. Impact: +47.7 reward
    3. num_gat_blocks: Tried 4 and 8. Lower is better — optimizer chose 4 in 52% of runs. Impact: +39.5 reward
    4. activation: Tried tanh and leaky_relu. tanh is better — optimizer chose tanh in 81% of runs. Impact: +32.6 reward
    5. clip_frac: Tried 0.1 and 0.2. Higher is better — optimizer chose 0.2 in 88% of runs. Impact: +23.4 reward
    6. entropy_coef: Tried 0.01 and 0.02. Negligible difference — optimizer slightly preferred 0.02 (76%). Impact: +2.0 reward

    # Insights from Jan 29 (API analysis):
    1. anneal_lr=False wins — maintains LR at 1e-5, gets 7-10% clipping, higher entropy (0.57-0.66), +20-50 reward over annealed.
    2. anneal_lr=True drops LR to 1e-6 causing KL~0.002, no clipping, lower entropy — policy stops updating effectively.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },

        # =====================================================================
        # 2. General Sweep for PPO Alpha 1.0
        # =====================================================================
        "parameters": {
            # Sweep params (3 × 2 × 2 × 2 × 2 = 48 combinations, Bayesian samples 32)
            "lr": {"values": [5e-6, 1e-5, 3e-5]},
            "anneal_lr": {"values": [True, False]},
            "K_epochs": {"values": [2, 4]},
            "num_gat_blocks": {"values": [4, 8]},
            "batch_size": {"values": [32, 64]},

            # Fixed params
            "clip_frac": {"value": 0.2},
            "entropy_coef": {"value": 0.02},
            "update_frequency": {"value": 128},
            "activation": {"value": "tanh"},

            # Fixed values (Not sweep params)
            # Training duration: 1M env steps (comparable to MCTS with max_iterations=744)
            # Eval frequency: 7,812 updates / 20 = ~390 eval points
            "alpha": {"value": 1.0}, # ALPHA SETTING.
            "algorithm": {"value": "ppo"},
            "gpu": {"value": True},
            "max_steps": {"value": 750_000},
            "num_ppo_workers": {"value": 8},
            "ppo_eval_every": {"value": 20},
        },
    }


def build_ppo_sweep_config() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep - parent function that selects alpha config.
    Currently calls both 0.3 and 1.0 configs (comment out as needed).
    """
    # return build_sweep_config_ppo_0_3()
    return build_sweep_config_ppo_1_0()


def build_sweep_config_mcts_0_3() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep search space for Alpha = 0.3.

    Key takeaways from MCTS sweep in Jan 10:
    Alpha = 0.3 (3 runs)
    1. lr: Tried 1e-05 and 0.0001. Higher is better — 0.0001 used in 100% of top runs. Impact: +8pp service rate, -11 min wait time.
    2. n_iter: Tried 50 and 100. Higher is better — 100 simulations used in 100% of top runs. Impact:  +8pp service rate improvement.
    3. num_gat_blocks: Tried 4 and 8. Higher is better — 8 blocks used in 100% of top runs. Impact:    2.5x better route efficiency.

    # Insights from Jan 29 (API analysis):
    1. Fast temp schedules kill learning — when temp=0.1, policy_loss drops to ~0.002 (near-zero gradients).
    2. peach-sweep-1 peaked at 39.5% progress (temp=1.0), then flat for remaining 60% of training.
    3. Slower schedules (0.6:1.0,0.85:0.5,1.0:0.1) maintain policy_loss ~0.2-0.3, enabling continued learning.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },

        # =====================================================================
        # 3. General Sweep for MCTS Alpha 0.3
        # =====================================================================
        "parameters": {
            # Sweep params (2 × 2 × 2 × 2 = 16 combinations)
            "temp_schedule": {"values": [
                "0.6:1.0,0.85:0.5,1.0:0.1",   # Slower annealing
                "0.7:1.0,0.9:0.5,1.0:0.2",    # Even slower + higher floor
            ]},
            "c_puct": {"values": [1.0, 1.5]},
            "train_steps_per_iter": {"values": [100, 200]},
            "num_gat_blocks": {"values": [4, 8]},

            # Fixed params
            "n_iter": {"value": 400},
            "lr": {"value": 1e-4},
            "batch_size": {"value": 256},
            "buffer_capacity": {"value": 50000},
            "activation": {"value": "tanh"},
            "dirichlet_alpha": {"value": 0.3},

            # Fixed values (Not sweep params)
            "alpha": {"value": 0.3},
            "algorithm": {"value": "mcts"},
            "gpu": {"value": True},
            "max_iterations": {"value": 558},
            "num_mcts_workers": {"value": 8},
            "mcts_eval_every": {"value": 5},
        },

        # =====================================================================
        # 5. n_iter Sweep for MCTS Alpha 0.3
        # =====================================================================
        # "parameters": {
        #     # Sweep param
        #     "n_iter": {"values": [100, 200, 300, 400, 500]},
        #
        #     # TODO: Set after sweeps 1-4 finish
        #     # "temp_schedule": {"value": "..."},
        #     # "c_puct": {"value": ...},
        #     # "train_steps_per_iter": {"value": ...},
        #     # "num_gat_blocks": {"value": ...},
        #
        #     # Fixed params
        #     "lr": {"value": 1e-4},
        #     "batch_size": {"value": 256},
        #     "buffer_capacity": {"value": 50000},
        #     "activation": {"value": "tanh"},
        #     "dirichlet_alpha": {"value": 0.3},
        #
        #     # Fixed values (Not sweep params)
        #     "alpha": {"value": 0.3},
        #     "algorithm": {"value": "mcts"},
        #     "gpu": {"value": True},
        #     "max_iterations": {"value": 558},
        #     "num_mcts_workers": {"value": 8},
        #     "mcts_eval_every": {"value": 5},
        # },
    }


def build_sweep_config_mcts_1_0() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep search space for Alpha = 1.0.

    Key takeaways from MCTS sweep in Jan 22:
    Alpha = 1.0 (5 runs)
    1. num_gat_blocks: Tried 4 and 8. Lower is better — 4 blocks used in 100% of top 2 runs. Impact: +3.4 reward (+11%), -7.8 min travel time (-20%), -22 buses fleet size (-23%).
    2. c_puct: Tried 1.0 and 1.5. Higher is slightly better — 1.5 used in 50% of top 2 runs. Impact: +0.6 to +1.0 reward (controlled comparison), -1.9 min travel time.

    # Insights from Jan 29 (API analysis):
    1. Fast temp schedules kill learning — when temp=0.1, policy_loss drops to ~0.007 (near-zero gradients).
    2. zany-sweep-1: policy_loss=0.79→0.02→0.0 as temp dropped 1.0→0.5→0.1. Reward plateaued at 35.68.
    3. glorious-sweep-1 (slower schedule 0.6:1.0,0.85:0.5,1.0:0.1): policy_loss stayed ~0.2-0.3, got better peak (36.91).
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize"
        },

        # =====================================================================
        # 4. General Sweep for MCTS Alpha 1.0
        # =====================================================================
        "parameters": {
            # Sweep params (2 × 2 × 2 × 2 = 16 combinations)
            "temp_schedule": {"values": [
                "0.6:1.0,0.85:0.5,1.0:0.1",   # Slower annealing
                "0.7:1.0,0.9:0.5,1.0:0.2",    # Even slower + higher floor
            ]},
            "c_puct": {"values": [1.0, 1.5]},
            "train_steps_per_iter": {"values": [100, 200]},
            "num_gat_blocks": {"values": [4, 8]},

            # Fixed params
            "n_iter": {"value": 400},
            "lr": {"value": 1e-4},
            "batch_size": {"value": 256},
            "buffer_capacity": {"value": 50000},
            "activation": {"value": "tanh"},
            "dirichlet_alpha": {"value": 0.3},

            # Fixed values (Not sweep params)
            "alpha": {"value": 1.0},
            "algorithm": {"value": "mcts"},
            "gpu": {"value": True},
            "max_iterations": {"value": 558},
            "num_mcts_workers": {"value": 8},
            "mcts_eval_every": {"value": 5},
        },

        # =====================================================================
        # 6. n_iter Sweep for MCTS Alpha 1.0
        # =====================================================================
        # "parameters": {
        #     # Sweep param
        #     "n_iter": {"values": [100, 200, 300, 400, 500]},
        #
        #     # TODO: Set after sweeps 1-4 finish
        #     # "temp_schedule": {"value": "..."},
        #     # "c_puct": {"value": ...},
        #     # "train_steps_per_iter": {"value": ...},
        #     # "num_gat_blocks": {"value": ...},
        #
        #     # Fixed params
        #     "lr": {"value": 1e-4},
        #     "batch_size": {"value": 256},
        #     "buffer_capacity": {"value": 50000},
        #     "activation": {"value": "tanh"},
        #     "dirichlet_alpha": {"value": 0.3},
        #
        #     # Fixed values (Not sweep params)
        #     "alpha": {"value": 1.0},
        #     "algorithm": {"value": "mcts"},
        #     "gpu": {"value": True},
        #     "max_iterations": {"value": 558},
        #     "num_mcts_workers": {"value": 8},
        #     "mcts_eval_every": {"value": 5},
        # },
    }


def build_mcts_sweep_config() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep - parent function that selects alpha config.
    Currently calls both 0.3 and 1.0 configs (comment out as needed).
    """
    return build_sweep_config_mcts_0_3()
    # return build_sweep_config_mcts_1_0()


def get_sweep_config(algorithm: str) -> Dict[str, Any]:
    """
    Get sweep configuration for the specified algorithm.
    """
    if algorithm == "ppo":
        return build_ppo_sweep_config()
    elif algorithm == "mcts":
        return build_mcts_sweep_config()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'ppo', 'mcts'")

def get_train_fn(algorithm: str):
    """
    Get the training function for the specified algorithm.
    """
    if algorithm == "ppo":
        return ppo_train
    elif algorithm == "mcts":
        return mcts_train
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'ppo', 'mcts'")

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
        python sweep.py --algorithm mcts
    """
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for transit design RL")
    parser.add_argument("--algorithm", choices=["ppo", "mcts"], required=True,
                        help="Algorithm to sweep (required)")
    args = parser.parse_args()

    # Get configs
    sweep_config = get_sweep_config(args.algorithm)
    base_config = get_config()

    print(f"Starting {args.algorithm.upper()} Bayesian sweep (Ctrl+C to stop)...")

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

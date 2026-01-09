import gc
import torch
import wandb
import argparse
from typing import Any, Dict
from config import get_config, set_global_seeds
from ppo import train as ppo_train
from mcts import train as mcts_train

def build_ppo_sweep_config() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep search space.

    Key takeaways from PPO sweep in Jan 5: 
    Alpha = 0.3 (10 runs)
    1. K_epochs: Tried 2 and 4. Higher is better — optimizer chose 4 in 60% of runs. Impact: +55 reward.
    2. batch_size: Tried 16 and 32. Higher is better — optimizer chose 32 in 80% of runs. Impact: +55 reward.
    3. lr: Tried 1e-05 and 1e-04. Higher is better — optimizer chose 1e-04 in 90% of runs. Impact: +52 reward.


    Alpha = 1.0 (39 runs)
    1. lr: Tried 1e-05 and 1e-04. Lower is better — optimizer chose 1e-05 in 95% of runs. Impact: +119 reward.
    2. batch_size: Tried 16 and 32. Higher is better — optimizer chose 32 in 95% of runs. Impact: +84 reward.
    3. update_frequency: Tried 128 and 256. Lower is better — optimizer chose 128 in 90% of runs. Impact: +81 reward.

    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_total_reward", 
            "goal": "maximize"
        },

        # Alpha = 0.3
        "parameters": {
            "clip_frac": {"values": [0.1, 0.2]},
            "entropy_coef": {"values": [0.01, 0.02]},
            "lr": {"values": [1e-4]},
            "K_epochs": {"values": [4]},
            "batch_size": {"values": [32]},
            "update_frequency": {"values": [128, 256]},
            
            # Model architecture (gat_channels/num_heads auto-generated from num_gat_blocks)
            "activation": {"values": ["tanh", "leaky_relu"]},
            "num_gat_blocks": {"values": [4, 8]},

            # Fixed values (Not sweep params)
            "alpha": {"value": 0.3}, # ALPHA SETTING. 
            "algorithm": {"value": "ppo"},
            "gpu": {"value": True},
            "anneal_lr": {"value": True},
            "max_steps": {"value": 120_000},
        },

        # Alpha = 1.0
        # "parameters": {
        #     "clip_frac": {"values": [0.1, 0.2]},
        #     "entropy_coef": {"values": [0.01, 0.02]},
        #     "lr": {"values": [1e-5]},
        #     "K_epochs": {"values": [2, 4]},
        #     "batch_size": {"values": [32]},
        #     "update_frequency": {"values": [128]},
            
        #     # Model architecture (gat_channels/num_heads auto-generated from num_gat_blocks)
        #     "activation": {"values": ["tanh", "leaky_relu"]},
        #     "num_gat_blocks": {"values": [4, 8]},

        #     # Fixed values (Not sweep params)
        #     "alpha": {"value": 1.0}, # ALPHA SETTING. 
        #     "algorithm": {"value": "ppo"},
        #     "gpu": {"value": True},
        #     "anneal_lr": {"value": True},
        #     "max_steps": {"value": 120_000},
        # },

    }

def build_mcts_sweep_config() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep search space.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_total_reward",
            "goal": "maximize"
        },

        # Alpha = 0.3
        "parameters": {
            "lr": {"values": [1e-4, 1e-5]},
            "c_puct": {"values": [1.0, 1.5]},
            "dirichlet_alpha": {"values": [0.3, 0.5]},

            # Model architecture (gat_channels/num_heads auto-generated from num_gat_blocks)
            "activation": {"values": ["tanh", "leaky_relu"]},
            "num_gat_blocks": {"values": [4, 8]},
            
            # How many times to sample batch_size from buffer per iteration.
            # 500 steps × 256 batch = 128,000 samples trained per iteration (~1600 new samples with 8 workers)
            "train_steps_per_iter": {"values": [200, 500]},

            # How many items to sample from buffer per update.
            "batch_size": {"values": [256, 512]},
            
            # How many MCTS simulations to run 
            # Each of the n_iter simulations only:
            # 1. Traverses the tree using PUCT 2. Calls the neural network for P(actions) and V(state)
            # The full UXsim traffic simulation only runs once per completed route 
            "n_iter": {"values": [200, 400]},
            
            # Fixed values (Not sweep params)
            "alpha": {"value": 0.3}, # ALPHA SETTING. 
            "algorithm": {"value": "mcts"},
            "gpu": {"value": True},
            "max_iterations": {"value": 500},
            "num_mcts_workers": {"value": 8}, # Each iteration runs num_mcts_workers episodes in parallel
        },

        # # Alpha = 1.0
        # "parameters": {
        # "lr": {"values": [1e-4, 1e-5]},
        #     "c_puct": {"values": [1.0, 1.5]},
        #     "dirichlet_alpha": {"values": [0.3, 0.5]},
        #     "activation": {"values": ["tanh", "leaky_relu"]},
        #     "num_gat_blocks": {"values": [4, 8]},
        #     "train_steps_per_iter": {"values": [200, 500]},
        #     "batch_size": {"values": [256, 512]},
        #     "n_iter": {"values": [200, 400]},
            
        #     # Fixed values (Not sweep params)
        #     "alpha": {"value": 0.3}, # ALPHA SETTING. 
        #     "algorithm": {"value": "mcts"},
        #     "gpu": {"value": True},
        #     "max_iterations": {"value": 500},
        #     "num_mcts_workers": {"value": 8}, # Each iteration runs num_mcts_workers episodes in parallel
        # },
    }


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


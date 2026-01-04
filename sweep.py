import gc
import torch
import wandb
import argparse
from typing import Any, Dict
from ppo import get_config, set_global_seeds, train as ppo_train_internal
from mcts import mcts_train

def build_ppo_sweep_config() -> Dict[str, Any]:
    """
    PPO hyperparameter sweep search space.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_total_reward", 
            "goal": "maximize"
        },
        "parameters": {
            "clip_frac": {"values": [0.1, 0.2]},
            "entropy_coef": {"values": [0.01, 0.02]},
            "lr": {"values": [1e-5, 1e-4]},
            "K_epochs": {"values": [2, 4]},
            "batch_size": {"values": [16, 32]},
            "update_frequency": {"values": [128, 256]},

            # Fixed values
            "gpu": {"value": True},
            "anneal_lr": {"value": True},
            "max_steps": {"value": 120_000},
        },
    }

def build_mcts_sweep_config() -> Dict[str, Any]:
    """
    MCTS hyperparameter sweep search space.
    TODO: Finish
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/episode_total_reward", 
            "goal": "maximize"
        },
        "parameters": {
            # MCTS hyperparameters (placeholders - adjust when MCTS is implemented)
            # "exploration_constant": {"values": [1.0, 1.414, 2.0]},
            # "num_simulations": {"values": [50, 100, 200]},
            # "max_depth": {"values": [10, 20, 50]},
            
            # Fixed values
            "gpu": {"value": True},
            "max_steps": {"value": 100_000},
        },
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
        return ppo_train_internal
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
            
            # WandB is already initialized by the sweep agent context
            # Set wandb_off=False so train() logs to the active run
            config["wandb_off"] = False
            
            # Run training
            train_fn(config)
        
        # Cleanup between sweep runs
        gc.collect()
    
    return agent_train

def main() -> None:
    """
    Usage:
        python sweep.py --algorithm ppo --count 32
        python sweep.py --algorithm mcts --count 16
    """
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for transit design RL")
    parser.add_argument("--algorithm", choices=["ppo", "mcts"], default="ppo", 
                        help="Algorithm to sweep (default: ppo)")
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
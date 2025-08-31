import torch
import wandb
from main import get_config, train, set_global_seeds
from typing import Any, Dict

def build_sweep_config() -> Dict[str, Any]:
    """
    Define the sweep search space 
    """

    return {
        "method": "bayes",
        
                "metric": {"name": "eval/avg_episode_reward", 
                   "goal": "maximize"},

        "parameters": {
            "lr": {"values": [0.0001, 0.001]}, 
            "batch_size": {"values": [16, 32]},   
            "entropy_coef": {"values": [0.1, 0.2]}, 
            "update_frequency": {"values": [64, 128, 256]},
            
            # "clip_frac": {"values": [0.1, 0.2, 0.3]},
            # "gae_lambda": {"values": [0.9, 0.95, 0.98]},
            # "gamma": {"values": [0.9, 0.95, 0.99]},
            # "K_epochs": {"values": [2, 4, 8]},
            # "value_loss_coef": {"values": [0.25, 0.5, 1]},

            # Reward coefficients (continuous distributions)
            # Sweep betas in smaller sweep with RL coefficients kept constant.
            "beta0": {"distribution": "uniform", "min": 20, "max": 60},  # Service emphasis
            "beta1": {"distribution": "uniform", "min": 10, "max": 40},  # Wait penalty
            "beta2": {"distribution": "uniform", "min": 10, "max": 30},  # Efficiency
            "beta3": {"distribution": "uniform", "min": 10, "max": 40},  # Utilization
        },
    }

def agent_train() -> None:
    """
    Function to be called by wandb.agent for each sweep run.
    Initializes WandB run, merges configs, runs training, logs metrics.
    """
    with wandb.init() as run:
        # Get sampled params from sweep
        sampled_params = dict(wandb.config)
        
        # Get defaults and merge with sampled
        base_config = get_config()
        config = {**base_config, **sampled_params}
        
        set_global_seeds(config["seed"])
        device = torch.device("cuda" if config.get("gpu", True) and torch.cuda.is_available() else "cpu")
        config["device"] = device
        
        train(config)

def main() -> None:
    """
    """
    sweep_config = build_sweep_config()
    base_config = get_config()
    
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project=base_config["wandb_project"], 
        entity=base_config["wandb_entity"]
    )
    wandb.agent(sweep_id, function=agent_train, count=20)  

if __name__ == "__main__":
    main()
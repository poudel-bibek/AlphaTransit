import os
import torch
import random
import argparse
import numpy as np
from rl.env import TransitEnv
from typing import Any, Dict, Optional, Sequence

def train(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Train the transit route design agent to "learn to design routes".

    Note: 
    # Partial Route Simulation (Per-Step Rewards):
    - Step 1: Route [A] → Run simulation with 1-node bus "route" → Get reward R1
    - Step 2: Route [A→B] → Run simulation with 2-node bus route → Get reward R2  
    - Step 3: Route [A→B→C] → Run simulation with 3-node bus route → Get reward R3
    Pro: Agent gets immediate feedback after each node addition
    Cons: 
        - Computationally expensive (compared to route simulation after route completion)
        - Partial routes might connect zero O-D pairs -> zero reward (still useful for learning?)
        - For very large networks/ long-routes, becomes a problem

    Objective: max E[R₁ + γR₂ + γ²R₃ + γ³R₄ + ...]
    Agent learns: "What next node maximizes total discounted reward?"
    """

    train_env = TransitEnv(config)
    episode_rewards = []
    for episode in range(config["max_episodes"]):

        state, info = train_env.reset()
        episode_reward = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        for step in range(config["max_steps_per_episode"]):
            
            # TODO: Sample random action for now (will be replaced with policy later)
            # Sampled action is an index of the node
            action = train_env.action_space.sample() 

            next_state, reward, terminated, truncated, step_info = train_env.step(action)
            
            episode_reward += reward
            
            print(f"\n\nStep {step + 1}: Action {action}, Reward: {reward:.2f}")
            
            # Episode ends when route is complete or constraints violated
            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step + 1} steps. Total reward: {episode_reward:.2f}")
                break
                
            # Update state for next step
            state = next_state
        
        episode_rewards.append(episode_reward)
    
    return {"episode_rewards": episode_rewards, "avg_reward": np.mean(episode_rewards)}

def eval(config: Dict[str, Any]) -> Dict[str, float]:  # noqa: A003
    """
    Evaluate a trained policy
    """
    pass


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


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for CLI usage.
    Define common RL, system, and sweepable hyperparameters.
    """
    parser = argparse.ArgumentParser(description="RL training/evaluation entrypoint")
    # Simulation setup: 
    parser.add_argument("--network", choices=["sioux_falls", "laval", "rivera", "mumford3"], default="sioux_falls", help="Network selection")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available; defaults to CPU when not set or unavailable")
    parser.add_argument("--horizon", type=int, default=3600, help="Simulation horizon")
    parser.add_argument("--delta_t", type=float, default=1, help="Simulation time step")
    parser.add_argument("--delta_n", type=int, default=5, help="Simulation platoon size")
    parser.add_argument("--bus_capacity", type=int, default=40, help="Bus capacity")
    parser.add_argument("--stop_duration", type=int, default=60, help="Stop duration")
    parser.add_argument("--max_episodes", type=int, default=10, help="Maximum number of training episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Maximum steps per episode (safety limit)")

    # Learning environment specific: 
    parser.add_argument("--service_frequency", type=int, default=1, help="Service frequency")
    parser.add_argument("--stop_spacing", type=int, default=1, help="Stop spacing")
    parser.add_argument("--alpha", type=float, default=0.3, help="Modal split parameter for served O-D pairs (proportion taking bus)")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius within each node to consider for demand allocation")
    parser.add_argument("--random_path_init", action="store_true", help="Initialize path randomly (omit flag for False)")
    # Constraints:
    parser.add_argument("--max_path_length", type=int, default=10, help="Maximum path length")
    parser.add_argument("--min_path_length", type=int, default=1, help="Minimum path length")

    # Arguments likely to change during sweeps (tunable hyperparameters)
    parser.add_argument("--total_timesteps", type=int, default=10000, help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Dict[str, Any] = vars(args)

    # Set seeds as the very first thing after parsing CLI
    set_global_seeds(args.seed)

    # Set compute device based on --gpu flag
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    config["device"] = device

    if args.mode == "train":
        train(config)
    else:
        eval(config)


if __name__ == "__main__":
    main()



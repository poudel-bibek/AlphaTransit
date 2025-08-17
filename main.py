import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from rl.env import TransitEnv
from rl.models import GATV2ActorCritic
from rl.ppo import PPO
from torch_geometric.data import Data, Batch
from typing import Any, Dict, Optional, Sequence

def state_to_pyg(state: Dict[str, Any]) -> Data:
    """
    """
    
    x = torch.tensor(state["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(state["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(state["edge_features"], dtype=torch.float32)
    steps_left = torch.tensor(state["steps_left"], dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.steps_left = steps_left
    return data

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

    # action = env.action_space.sample()
    """

    print("Training started... on network: ", config["network"])
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    num_actions = env.action_space.n
    
    print("\tObservation space:")
    for key, space in env.observation_space.items():
        print(f"\t  {key}: {space}")
    print(f"\tAction space: {env.action_space}")

    print(f"\nNetwork details:")
    print(f"\tNumber of nodes: {env.n_nodes}")
    print(f"\tNumber of edges: {env.n_edges}")
    print(f"\tNode list: {env.node_list}")
    print(f"\tIdx to node: {env.idx_to_node}")

    print("\n\tAdjacency list:")
    for node in env.node_list:
        neighbors = env.adj[node]
        print(f"\t  {node}: {', '.join(neighbors)}")
    print(f"\tMax degree: {env.max_degree}, Min degree: {env.min_degree}")

    od_df = pd.DataFrame(env.od_matrix, index=env.node_list, columns=env.node_list)
    with pd.option_context('display.max_rows', 20, 'display.max_columns', 20, 'display.width', 120):
        print("\n\tOD matrix (preview):\n", od_df.round(4))
    print(f"\tMax demand: {env.max_demand}, Min demand: {env.min_demand}")

    # Set policy hyper-param defaults
    policy_kwargs = {
        "num_layers": config.get("num_layers", 3),
        "gat_channels": config.get("gat_channels", [node_feature_dim, 36, 16, 1]),
        "num_heads": config.get("num_heads", [8, 4, 1]),
        "num_edge_features": config.get("num_edge_features", 2),
        "dropout": config.get("dropout", 0.0),
        "global_dim": config.get("global_dim", 1),
        "activation": config.get("activation", "elu"),
        "model_size": config.get("model_size", "medium"),
        "concat": config.get("concat", False),
    }

    # Only node features are supplied at GATv2 init, edge features are injected at each attention layer.
    model = GATV2ActorCritic(num_actions, **policy_kwargs)
    model.to(config["device"])
    param_counts = model.param_count()
    print(f"\nPolicy parameters on device = {config['device']}:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    ppo = PPO(model, **config)
    
    episode_rewards = []
    
    for episode in range(1, config["max_episodes"] + 1):
        print(f"\n=== Episode {episode} ===")
        
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            data = state_to_pyg(state)
            batch = Batch.from_data_list([data]).to(config["device"])
            steps_left = batch.steps_left.to(config["device"])
            valid_indices = torch.tensor(env._get_valid_indices(), dtype=torch.long, device=config["device"])
            
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(batch, steps_left, deterministic=False, valid_indices=valid_indices)
            
            action = action_tensor.cpu().item()
            log_prob = log_prob_tensor.cpu().item()
            value = value_tensor.cpu().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            
            # Store on CPU
            store_data = Data(
                x=data.x.cpu(),
                edge_index=data.edge_index.cpu(),
                edge_attr=data.edge_attr.cpu(),
                steps_left=data.steps_left.cpu()
            )
            
            ppo.memory.store({
                'obs': store_data,
                'action': action,
                'reward': reward,
                'value': value,
                'log_prob': log_prob,
                'done': terminated or truncated,
                'valid_indices': valid_indices.cpu().tolist()
            })
            
            state = next_state
            episode_steps += 1
            
            env.render(f"ep_{episode}_step_{episode_steps}.png")
            
            if terminated or truncated:
                break
        
        # After episode, compute last value if not (terminated or truncated)
        if not (terminated or truncated):
            data = state_to_pyg(state)
            batch = Batch.from_data_list([data]).to(config["device"])
            steps_left = batch.steps_left.to(config["device"])
            valid_indices = torch.tensor(env._get_valid_indices(), dtype=torch.long, device=config["device"])
            
            with torch.no_grad():
                _, _, last_value_tensor = model.act(batch, steps_left, valid_indices=valid_indices)
            last_value = last_value_tensor.cpu().item()

            # Append dummy transition for last value
            ppo.memory.store({
                'obs': store_data,  # reuse last
                'action': 0,  # dummy
                'reward': 0,
                'value': last_value,
                'log_prob': 0,
                'done': True
            })
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode} finished after {episode_steps} steps. Reward: {episode_reward:.2f}")
        
        # Update after each episode
        if len(ppo.memory) > 0:
            stats = ppo.update()
            print("Update stats:", stats)
    
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

    # Learning environment specific: 
    parser.add_argument("--service_frequency", type=int, default=1, help="Service frequency")
    parser.add_argument("--stop_spacing", type=int, default=1, help="Stop spacing")
    parser.add_argument("--alpha", type=float, default=0.3, help="Modal split parameter for served O-D pairs (proportion taking bus)")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius within each node to consider for demand allocation")
    parser.add_argument("--random_path_init", action="store_true", help="Initialize path randomly (omit flag for False)")
    # Constraints:
    parser.add_argument("--max_path_length", type=int, default=10, help="Maximum path length")
    parser.add_argument("--min_path_length", type=int, default=1, help="Minimum path length")

    # PPO params: 
    parser.add_argument("--K_epochs", type=int, default=10, help="Number of PPO epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--clip_frac", type=int, default=0.2, help="PPO clipping ratio")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="linear", help="Learning rate schedule") 

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Dict[str, Any] = vars(args)
    
    # config["max_steps_per_episode"] = config["max_path_length"] # Not using at the moment.
    
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
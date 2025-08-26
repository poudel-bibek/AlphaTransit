import os
import wandb
import torch
import random
import argparse
import numpy as np
import pandas as pd
from rl.ppo import PPO
from typing import Any, Dict
from datetime import datetime
from rl.env import TransitEnv
from rl.models import GATV2ActorCritic
from rl.env_utils import pretty_print_state
from torch_geometric.data import Data, Batch
from rl.env_utils import plot_network_and_demand
from rl.heuristic_baselines import GreedyDemandCoverage

def state_to_pyg(state: Dict[str, Any]) -> Data:
    """
    """
    # Use from_numpy for better performance when possible
    x = torch.from_numpy(state["node_features"]).float()
    edge_index = torch.from_numpy(state["edge_index"]).long() 
    edge_attr = torch.from_numpy(state["edge_features"]).float()
    steps_left = torch.from_numpy(state["steps_left"]).float()
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.steps_left = steps_left
    return data

def perform_ppo_update(ppo: PPO, steps_elapsed: int, anneal_lr: bool) -> None:
    """
    """
    print("\n==================\n")
    print("Memory contents:")
    print(f"\tNumber of transitions: {len(ppo.memory)}")
    print(f"\tRewards: {ppo.memory.rewards}")
    print(f"\tActions: {ppo.memory.actions}")
    print(f"\tLog probs: {ppo.memory.log_probs}")
    print(f"\tValues: {ppo.memory.values}")
    print(f"\tDones: {ppo.memory.dones}")
    print(f"\tEpisode boundaries: {ppo.memory.episode_boundaries}")
    print(f"\tBootstrap values: {ppo.memory.bootstrap_values}")

    if anneal_lr:
        ppo.update_learning_rate(steps_elapsed)

    mem_len = len(ppo.memory)  
    stats = ppo.update() # Also clears the memory.

    print("\n=== PPO Update ===")
    print(f"Step: {steps_elapsed}")
    print(f"Samples: {mem_len}")
    print(f"PG Loss: {stats['pg_loss']:.4f}")
    print(f"Value Loss: {stats['value_loss']:.4f}")
    print(f"Entropy Loss: {stats['entropy_loss']:.4f}")
    print(f"Clipping Frequency: {stats['clipping_frequency']:.4f}")
    print(f"Mean Reward: {stats['mean_reward']:.4f}")
    print(f"Approx KL: {stats['approx_kl']:.4f}")
    print(f"Mean Clip Ratio: {stats['mean_clip_ratio']:.4f}")
    print("==================\n")
    wandb.log({
        "policy_loss": stats['pg_loss'],
        "value_loss": stats['value_loss'],
        "entropy_loss": stats['entropy_loss'],
        "clipping_frequency": stats['clipping_frequency'], # How often the ratio was clipped.
        "mean_reward": stats['mean_reward'], # Average reward over the batch.
        "approx_kl": stats['approx_kl'],
        "mean_clip_ratio": stats['mean_clip_ratio'], # Actual ratio of clipped updates.
    }, step=steps_elapsed)

def train(config: Dict[str, Any]) -> None:
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

    --------------
    Value bootstraping: 
    - Required when an episode ends premeturely (truncated)
    - If terminated naturally, value = 0 (no more rewards coming)
    - If truncated, value = value of the last state. 
        - We preserve the partial solution which has some utility in learning.
        - Truncation conditions: 
            - Gets stuck in a dead-end i.e., had only 1 neighbor, (which is already in the path)   
    """

    print("Training started... on network: ", config["network"])
    
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    num_actions = env.action_space.n
    
    print("\tObservation space:")
    for key, space in env.observation_space.items():
        print(f"\t  {key}: {space}")
    print(f"\tAction space: {env.action_space}")
    
    now = datetime.now()
    training_save_dir = os.path.join(config["save_dir"], f"{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}")
    img_dir = os.path.join(training_save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Build temp world for initial visualization only (not persistent)
    temp_world = env.build_world(config.get("network"))
    output_path = os.path.join(img_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path) # Visualize only after building world

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
    policy_kwargs = get_policy_kwargs(config, node_feature_dim)

    # Only node features are supplied at GATv2 init, edge features are injected at each attention layer.
    model = GATV2ActorCritic(num_actions, **policy_kwargs)
    model.to(config["device"])
    param_counts = model.param_count()

    print(f"\nPolicy parameters on device = {config['device']}:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    ppo = PPO(model, **config)
    policy_dir = os.path.join(training_save_dir, "policies")
    os.makedirs(policy_dir, exist_ok=True)

    episode = 0
    steps_elapsed = 0
    update_count = 0
    episode_rewards = []

    while steps_elapsed < config["total_timesteps"]:
        episode += 1
        print(f"\n=== Episode {episode} ===")
        
        state, _ = env.reset()
        # pretty_print_state(env, state)
        episode_reward, episode_steps = 0, 0
        terminated = False
        bootstrap_value = None  # Initialize outside loop

        while True:
            data = state_to_pyg(state)
            print("\nEpisode data: ")
            print(f"\tData: type: {type(data)}, value: {data}")

            batch = Batch.from_data_list([data]).to(config["device"])
            steps_left = batch.steps_left.to(config["device"])
            valid_indices = torch.tensor([env._get_valid_indices()], dtype=torch.long, device=config["device"])
            print(f"\tValid indices: shape: {valid_indices.shape}, value: {valid_indices}")
            
            if valid_indices.shape[1] == 0: # Episode is truncated.

                # TODO: Penalty set to an arbitrary high value (hard-coded).
                # Make it configurable based on how bad of the truncation was.
                truncation_penalty = -100.0

                _, _, value_tensor = model.act(batch, steps_left, deterministic=False, valid_indices=valid_indices, truncated=True)
                bootstrap_value = value_tensor.cpu().item()

                if len(ppo.memory) > 0:
                    ppo.memory.rewards[-1] += truncation_penalty  # Add penalty to the reward of the action that led here
                episode_reward += truncation_penalty

                print(f"\nEpisode truncated - bootstrap value: {bootstrap_value}\n")
                break

            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(batch, steps_left, deterministic=False, valid_indices=valid_indices)
                print(f"\tAction tensor: shape: {action_tensor.shape}, value: {action_tensor}")
                print(f"\tLog prob tensor: shape: {log_prob_tensor.shape}, value: {log_prob_tensor}")
                print(f"\tValue tensor: shape: {value_tensor.shape}, value: {value_tensor}")
            
            action = action_tensor.cpu().item() 
            next_state, reward, terminated, _ = env.step(action)
            episode_reward += reward
            
            # Store transition
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
                'value': value_tensor.cpu().item(),
                'log_prob': log_prob_tensor.cpu().item(),
                'terminated': terminated,  # Natural end    
                'valid_indices': valid_indices.squeeze(0).cpu().tolist()  # Flatten to list[int] (single env)
            })
                    
            state = next_state
            episode_steps += 1
            
            if terminated:
                bootstrap_value = 0.0
                print(f"\nEpisode terminated naturally - bootstrap value: 0.0\n")
                break

        # only render at the end of the episode.
        env.render(training_save_dir, f"ep_{episode}.png")

        # Mark episode boundary with bootstrap value.
        ppo.memory.mark_episode_end(bootstrap_value)

        steps_elapsed += episode_steps
        episode_rewards.append(episode_reward) # This reward is not logged
        print(f"Episode {episode} finished after {episode_steps} steps. Reward: {episode_reward:.2f}")
        
        # Update PPO when we have enough samples in memory
        # Episodes are not divisible i.e., waits for the full episode to complete before updating.
        # This is harmless and is actually good for GAE calculations.
        if len(ppo.memory) >= config["update_frequency"]:
            perform_ppo_update(ppo, steps_elapsed, config.get("anneal_lr"))
            update_count += 1

            # Save policy after every update.
            policy_path = os.path.join(policy_dir, f"policy_{update_count}.pth")
            torch.save(model.state_dict(), policy_path)

            if update_count % config["eval_every"] == 0:
                eval(config, policy_path, steps_elapsed, training_save_dir)

    # Final update if there's remaining data in memory (Its not that meaningful)
    # if len(ppo.memory) > 0:
    #     perform_ppo_update(ppo, steps_elapsed, config.get("anneal_lr"))
    #     update_count += 1

    avg_reward = np.mean(episode_rewards)
    wandb.log({"avg_episode_reward": avg_reward}, step=steps_elapsed)
    
def eval(config: Dict[str, Any], policy_path: str, steps_elapsed: str, save_dir: str) -> Dict[str, float]: 
    """
    Evaluate a trained policy
    - Load a saved policy
    - Run the policy on the network (take deterministic actions) and get the path
    - Get relevant metrics and log
    - Plot path and call world.analyzer.network_fancy

    Notes: 
    - During eval episode, agent constructs a route path step by step
    - The results are only meaningful at the end of the episode.
    """
    print("Evaluating policy: ", policy_path)
    
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    num_actions = env.action_space.n

    policy_kwargs = get_policy_kwargs(config, node_feature_dim)

    model = GATV2ActorCritic(num_actions, **policy_kwargs)
    model.load_state_dict(torch.load(policy_path))
    model.to(config["device"])
    model.eval() # eval mode

    state, _ = env.reset()
    terminated = False
    episode_reward = 0.0
    
    while not terminated:
        data = state_to_pyg(state)
        batch = Batch.from_data_list([data]).to(config["device"])
        steps_left = batch.steps_left.to(config["device"])
        valid_indices = torch.tensor([env._get_valid_indices()], dtype=torch.long, device=config["device"])
        
        if valid_indices.shape[1] == 0:
            print("No valid actions during evaluation, breaking early.")
            break
        
        with torch.no_grad():
            action_tensor, _, _ = model.act(batch, steps_left, deterministic=True, valid_indices=valid_indices)
 
        action = action_tensor.cpu().item()
        next_state, reward, terminated, info = env.step(action)
        episode_reward += reward
        state = next_state

    # TODO: log more metrics based on sim_result
    sim_result = info['sim_result']
    if not config.get("wandb_off"):
        wandb.log({
            f"eval_episode_reward": episode_reward,
            f"total_passengers_completed_trip": sim_result['total_passengers_completed_trip'],
            f"total_passengers_wanting_to_onboard": sim_result['total_passengers_wanting_to_onboard'],
            f"total_wait_time": sim_result['total_wait_time'],
            f"total_travel_time": sim_result['total_travel_time']
        }, step=steps_elapsed)

    # Plots
    env.render(save_dir, f"eval_{str(steps_elapsed)}.png")
    env.world.analyzer.network_fancy(
        animation_speed_inverse = 10,
        sample_ratio = 1.0,
        interval = 5,
        trace_length = 5,
        network_font_size = 14,
        antialiasing = False,
        file_name = os.path.join(save_dir, f"eval_anim_{str(steps_elapsed)}.gif"),
        save_as_mp4 = False
    )

    env.world.analyzer.network_fancy(
        animation_speed_inverse = 10,
        sample_ratio = 1.0,
        interval = 5,
        trace_length = 5,
        network_font_size = 14,
        antialiasing = False,
        file_name = os.path.join(save_dir, f"eval_anim_{str(steps_elapsed)}_bus_only.gif"),
        save_as_mp4 = False,
        bus_only = True
    )
        

def get_config() -> Dict[str, Any]:
    """
    A unified  config interface for both main and sweep.
    Parses CLI args if provided, otherwise uses defaults.
    Does NOT set seeds or device here to allow overrides from sweep.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    return vars(args)

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

def get_policy_kwargs(config: Dict[str, Any], node_feature_dim: int) -> Dict[str, Any]:
    return {
        "num_layers": config.get("num_layers", 3),
        "gat_channels": config.get("gat_channels", [node_feature_dim, 16, 16, 16]),
        "num_heads": config.get("num_heads", [8, 4, 2]),
        "num_edge_features": config.get("num_edge_features", 2),
        "dropout": config.get("dropout"),
        "global_dim": config.get("global_dim"),
        "activation": config.get("activation"),
        "model_size": config.get("model_size"),
        "concat": config.get("concat"),
    }

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for CLI usage.
    Define common RL, system, and sweepable hyperparameters.
    """
    parser = argparse.ArgumentParser(description="RL training/evaluation entrypoint")
    # Simulation setup: 
    parser.add_argument("--network", choices=["sioux_falls", "laval", "rivera", "mumford3"], default="sioux_falls", help="Network selection")
    parser.add_argument("--mode", choices=["train", "eval", "baseline"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=bool, default=True, help="Use CUDA if available; defaults to True, set to False to force CPU")
    parser.add_argument("--horizon", type=int, default=10000, help="Simulation horizon")
    parser.add_argument("--delta_t", type=float, default=1, help="Simulation time step")
    parser.add_argument("--delta_n", type=int, default=5, help="Simulation platoon size")
    parser.add_argument("--bus_capacity", type=int, default=40, help="Bus capacity")
    parser.add_argument("--stop_duration", type=int, default=60, help="Stop duration")
    parser.add_argument("--update_frequency", type=int, default=64, help="Update PPO when memory has N samples")
    parser.add_argument("--total_timesteps", type=int, default=50000, help="Total training timesteps") # This is not directly related to the simulation horizon.
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N updates to the policy")
    parser.add_argument("--baseline_type", type=str, default="greedy_demand_cover", help="Can be random, greedy, greedy_demand_cover")
    
    # Learning environment specific: 
    parser.add_argument("--service_frequency", type=int, default=6, help="Service frequency. 1 means one bus per hour")
    parser.add_argument("--stop_spacing", type=int, default=1, help="Stop spacing. 1 means every node is a stop")
    parser.add_argument("--alpha", type=float, default=0.3, help="Modal split parameter for served O-D pairs (proportion taking bus)")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius within each node to consider for demand allocation")
    parser.add_argument("--random_path_init", action="store_true", help="Initialize path randomly (omit flag for False)")
    # Constraints:
    parser.add_argument("--max_path_length", type=int, default=10, help="Maximum path length")
    parser.add_argument("--min_path_length", type=int, default=1, help="Minimum path length")

    # PPO params: 
    parser.add_argument("--K_epochs", type=int, default=4, help="Number of PPO epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--clip_frac", type=float, default=0.2, help="PPO clipping ratio for policy loss")
    parser.add_argument("--vf_clip_param", type=float, default=50.0, help="PPO clipping ratio for value loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--anneal_lr", type=bool, default=True, help="Anneal learning rate (default: True)")
    
    parser.add_argument("--model_size", type=str, default="medium", help="Model size")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function")
    parser.add_argument("--concat", type=bool, default=True, help="Concatenate attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--global_dim", type=int, default=1, help="Global dimension (additional feature to the graph-level features)")
    
    # WandB:
    parser.add_argument("--wandb_project", type=str, default="transit_design", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="bibek-poudel", help="WandB entity/team name")
    parser.add_argument("--wandb_off", type=bool, default=False, help="Disable WandB logging")

    parser.add_argument("--save_dir", type=str, default="./training_data", help="Directory to save training data")
    parser.add_argument("--saved_policy_path", type=str, default="./training_data/policies/policy_final.pth", help="Path to saved policy that you want to evaluate")
    # parser.add_argument("--demand_method", choices=["volume", "flow"], default="volume", help="Demand allocation method")
    return parser


def main() -> None:
    """
    """
    config = get_config()

    # Set seeds as the very first thing after parsing CLI
    set_global_seeds(config["seed"])
    
    # Set compute device based on --gpu flag
    device = torch.device("cuda" if (config["gpu"] and torch.cuda.is_available()) else "cpu")
    config["device"] = device

    if config["mode"] == "train":
        if not config.get("wandb_off"):
            wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
        train(config)
        wandb.finish()

    elif config["mode"] == "eval":
        # If performing eval only
        os.makedirs(config["save_dir"], exist_ok=True)
        config["wandb_off"] = True
        policy_path = config["saved_policy_path"]
        eval(config, policy_path, "final", config["save_dir"])

    else: 
        config["wandb_off"] = True
        # means mode is baseline
        env = TransitEnv(config)
        state, _ = env.reset() # Does path initialization at reset. Want it to be same between RL and baseline.
        print(f"Initial path: {env.current_path}")

        if config["baseline_type"] == "greedy_demand_cover":
            baseline = GreedyDemandCoverage(env)
            path = baseline.construct_path(state)
            result = baseline.simulate_path(path)
            print(f"Path: {path}\nSim result: {result}")

        else:
            raise ValueError(f"Invalid baseline type: {config['baseline_type']}")

if __name__ == "__main__":
    main()
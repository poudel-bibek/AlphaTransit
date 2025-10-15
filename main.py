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
from rl.baselines import RandomWalk, DemandCoverage, ShortestPath, RewardMaximization, RealWorld

class CachedPyGConverter:
    """
    Cache static PyG components to avoid recreating tensors every step.
    """
    def __init__(self, device: torch.device):
        """
        The edge index and edge attributes are do not change over the course of an episode.
        To make the sim faster, we cache these tensors.
        """
        self.device = device
        self._cached_edge_index = None
        self._cached_edge_attr = None
        
    def convert(self, state: Dict[str, Any]) -> Data:
        """
        Convert state to PyG format, caching static components.
        """
        # Cache static components on first call
        if self._cached_edge_index is None:
            self._cached_edge_index = torch.from_numpy(state["edge_index"]).long().to(self.device)
            self._cached_edge_attr = torch.from_numpy(state["edge_features"]).float().to(self.device)
        
        # Only create tensors for dynamic components  
        x = torch.from_numpy(state["node_features"]).float().to(self.device)
        route_progress = torch.from_numpy(state["route_progress"]).float().to(self.device)
        
        data = Data(x=x, edge_index=self._cached_edge_index, edge_attr=self._cached_edge_attr)
        data.route_progress = route_progress
        return data

# def state_to_pyg(state: Dict[str, Any]) -> Data:
#     """
#     """
#     # Use from_numpy for better performance when possible
#     x = torch.from_numpy(state["node_features"]).float()
#     edge_index = torch.from_numpy(state["edge_index"]).long() 
#     edge_attr = torch.from_numpy(state["edge_features"]).float()
#     steps_left = torch.from_numpy(state["steps_left"]).float()
    
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#     data.steps_left = steps_left
#     return data

def perform_ppo_update(ppo: PPO, episode: int, steps_elapsed: int, anneal_lr: bool, config: Dict[str, Any]) -> None:
    """
    mean_buffer_reward: average per-step rewards stored in the current memory buffer.
    Is not a true measure of the policy's performance.
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
        ppo.update_learning_rate(episode, config["num_episodes"])

    mem_len = len(ppo.memory)
    stats = ppo.update() # Also clears the memory.

    print("\n=== PPO Update ===")
    print(f"Episode: {episode}, Steps elapsed: {steps_elapsed}")
    print(f"Samples: {mem_len}")
    print(f"PG Loss: {stats['pg_loss']:.4f}")
    print(f"Value Loss: {stats['value_loss']:.4f}")
    print(f"Entropy Loss: {stats['entropy_loss']:.4f}")
    print(f"Clipping Frequency: {stats['clipping_frequency']:.4f}")
    print(f"Mean Buffer Reward: {stats['mean_buffer_reward']:.4f}")
    print(f"Approx KL: {stats['approx_kl']:.4f}")
    print(f"Mean Clip Ratio: {stats['mean_clip_ratio']:.4f}")
    print("==================\n")
    wandb.log({
        "ppo/policy_loss": stats['pg_loss'],
        "ppo/value_loss": stats['value_loss'],
        "ppo/entropy_loss": stats['entropy_loss'],
        "ppo/clipping_frequency": stats['clipping_frequency'], # How often the ratio was clipped.
        "ppo/mean_buffer_reward": stats['mean_buffer_reward'],
        "ppo/approx_kl": stats['approx_kl'],
        "ppo/mean_clip_ratio": stats['mean_clip_ratio'], # Actual ratio of clipped updates.
        "ppo/learning_rate": ppo.optimizer.param_groups[0]['lr'], # Current learning rate after annealing
        "ppo/steps_elapsed": steps_elapsed, # Total steps across all episodes
    }, step=episode)

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

    # Build world for initial visualization
    temp_world = env.build_world(config.get("network"))
    # Load demand data for visualization
    env.load_demand_for_plotting(temp_world)
    output_path = os.path.join(img_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path) # Visualize the network

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

    # PyG converter
    pyg_converter = CachedPyGConverter(config["device"])
    episode, steps_elapsed, update_count = 0, 0, 0

    while episode < config["num_episodes"]:
        episode += 1
        print(f"\n=== Episode {episode} ===")
        
        state, _ = env.reset()
        # pretty_print_state(env, state)
        episode_steps = 0
        terminated = False
        bootstrap_value = None  # Initialize outside loop

        while not terminated:
            data = pyg_converter.convert(state)
            print("\nEpisode data: ")
            print(f"\tData: type: {type(data)}, value: {data}")
            
            batch = Batch.from_data_list([data])  # Data already on device
            valid_indices = torch.tensor([env._get_valid_indices()], dtype=torch.long, device=config["device"])

            print(f"\tValid indices: shape: {valid_indices.shape}, value: {valid_indices}")

            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(batch, 
                                                                       deterministic=False, 
                                                                       valid_indices=valid_indices,
                                                                       truncated=False)
                print(f"\tAction tensor: shape: {action_tensor.shape}, value: {action_tensor}")
                print(f"\tLog prob tensor: shape: {log_prob_tensor.shape}, value: {log_prob_tensor}")
                print(f"\tValue tensor: shape: {value_tensor.shape}, value: {value_tensor}")
            
            action = action_tensor.cpu().item() 
            next_state, reward, terminated, _, sim_result = env.step(action) # Truncation is not used.

            episode_steps += 1
            steps_elapsed += 1

            # Store transition
            store_data = Data(
                x=data.x.cpu(),
                edge_index=data.edge_index.cpu(),
                edge_attr=data.edge_attr.cpu(),
                route_progress=data.route_progress.cpu()
            )
            
            ppo.memory.store({
                'obs': store_data,
                'action': action,
                'reward': reward,
                'value': value_tensor.cpu().item(),
                'log_prob': log_prob_tensor.cpu().item(),
                'terminated': terminated,  # Episode end signal from env
                'valid_indices': valid_indices.squeeze(0).cpu().tolist()  # Flatten to list[int] (single env)
            })
                    
            state = next_state
            
        # Truncation is not possible. Simplified bootstrapping. 
        bootstrap_value = 0.0
        print(f"\nEpisode terminated naturally - bootstrap value: {bootstrap_value}\n")

        # only render at the end of the episode.
        env.render(training_save_dir, f"ep_{episode}.png")

        # Mark episode boundary with bootstrap value.
        ppo.memory.mark_episode_end(bootstrap_value)

        # The reward after the entire route has been built. 
        episode_final_reward = reward 
        # This makes sense for episodes that terminate naturally (with complete routes).
        
        print(f"Episode {episode} finished after {episode_steps} steps. Reward: {episode_final_reward:.2f}")

        wandb.log({
            "episode/episode_final_reward": episode_final_reward,
            "episode/episode_length": episode_steps, # Length of routes
            "episode/steps_elapsed": steps_elapsed, # Total steps across all episodes
            # reward related
            "episode/service_rate": sim_result['service_rate'],
            "episode/demand_coverage_potential": sim_result['demand_coverage_potential'],
            "episode/demand_coverage_actual": sim_result['demand_coverage_actual'],
            "episode/route_overlap_ratio": sim_result['route_overlap_ratio'],
            "episode/transfer_rate": sim_result['transfer_rate'], # Percentage of trips requiring transfers
            "episode/node_coverage": sim_result['node_coverage'], # Percentage of nodes covered by routes
            # performance related
            "episode/onboard_rate": sim_result['onboard_rate'],
            "episode/completed_trips": sim_result['completed_trip_passengers_count'],
            "episode/avg_wait_time": sim_result['avg_wait_time'],
            "episode/avg_travel_time": sim_result['avg_travel_time'],
            "episode/bus_utilization": sim_result['bus_utilization'],
            }, step=episode)
        
        # Update PPO when we have enough samples in memory
        if len(ppo.memory) >= config["update_frequency"]:
            perform_ppo_update(ppo, episode, steps_elapsed, config.get("anneal_lr"), config)
            update_count += 1

            # Save policy after every update.
            policy_path = os.path.join(policy_dir, f"policy_up_{update_count}_ep_{episode}.pth")
            torch.save(model.state_dict(), policy_path)

            if update_count % config["eval_every"] == 0:
                eval(config, policy_path, episode, training_save_dir)
    
def eval(config: Dict[str, Any], policy_path: str, episode: int, save_dir: str) -> Dict[str, float]: 
    """
    Evaluate a trained policy
    - Load a saved policy
    - Run the policy on the network (take deterministic actions) and get the path
    - Get relevant metrics and log
    - Plot path and call world.analyzer.network_fancy

    Notes: 
    - During eval episode, agent constructs a route path step by step
    - The results are only meaningful at the end of the episode.
    -----
    episode_final_reward: this should be a true measure of the policy's performance.
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
    
    pyg_converter = CachedPyGConverter(config["device"])

    state, _ = env.reset()
    terminated = False
    eval_episode_steps = 0
    while not terminated:
        data = pyg_converter.convert(state)
        batch = Batch.from_data_list([data])  # Data already on device
        valid_indices = torch.tensor([env._get_valid_indices()], dtype=torch.long, device=config["device"])
        
        with torch.no_grad():
            action_tensor, _, _ = model.act(batch, 
            deterministic=True, 
            valid_indices=valid_indices,
            truncated=False)
 
        action = action_tensor.cpu().item()
        next_state, reward, terminated, _, sim_result = env.step(action) # Truncation is not used.
        state = next_state
        eval_episode_steps += 1
    episode_final_reward = reward
    if not config.get("wandb_off"):
        # Log
        wandb.log({
            # Final reward (after a full path has been constructed)
            "eval/episode_final_reward": episode_final_reward, # Set to maximize in the sweep.
            "eval/episode_length": eval_episode_steps,

            "eval/service_rate": sim_result['service_rate'],
            "eval/demand_coverage_potential": sim_result['demand_coverage_potential'],
            "eval/demand_coverage_actual": sim_result['demand_coverage_actual'],
            "eval/route_overlap_ratio": sim_result['route_overlap_ratio'],
            "eval/transfer_rate": sim_result['transfer_rate'], # Percentage of trips requiring transfers
            "eval/node_coverage": sim_result['node_coverage'],

            "eval/onboard_rate": sim_result['onboard_rate'],
            "eval/completed_trips": sim_result['completed_trip_passengers_count'],
            "eval/avg_wait_time": sim_result['avg_wait_time'],
            "eval/avg_travel_time": sim_result['avg_travel_time'],
            "eval/bus_utilization": sim_result['bus_utilization'],
        }, step=episode)


    # Plots
    env.render(save_dir, f"eval_{str(episode)}.png")
    env.world.analyzer.network_fancy(
        animation_speed_inverse = 10,
        sample_ratio = 1.0,
        interval = 5,
        trace_length = 5,
        network_font_size = 14,
        antialiasing = False,
        file_name = os.path.join(save_dir, f"eval_anim_{str(episode)}.gif"),
        save_as_mp4 = False,
        bus_only = False # Since our new setup already only contains buses only.
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
    parser.add_argument("--network", choices=["sioux_falls", "bloomington",], default="bloomington", help="Network selection")
    parser.add_argument("--mode", choices=["train", "eval", "baseline"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=bool, default=True, help="Use CUDA if available; defaults to True, set to False to force CPU")
    parser.add_argument("--horizon", type=int, default=10000, help="Simulation horizon") # 10k = 2.7 hours
    parser.add_argument("--delta_t", type=float, default=1, help="Simulation time step") # Increasing delta_t makes simulation faster.
    parser.add_argument("--delta_n", type=int, default=5, help="Simulation platoon size") # Increasing delta_n also makes simulation faster. Does not apply for bus passenger demand.
    parser.add_argument("--bus_capacity", type=int, default=40, help="Bus capacity")
    parser.add_argument("--stop_duration", type=int, default=60, help="Stop duration")
    parser.add_argument("--update_frequency", type=int, default=64, help="Update PPO when memory has N samples")
    parser.add_argument("--num_episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N updates to the policy")
    parser.add_argument("--baseline_type", type=str, default="greedy_demand_cover", help="Can be random_walk, reward_max, demand_cover, shortest_path, real_world")
    parser.add_argument("--num_baseline_runs", type=int, default=10, help="Number of runs (over which we average the results) for the baseline")

    # Learning environment specific: 
    parser.add_argument("--service_frequency_mode", type=str, default="max_load", help="Service frequency mode, e.g., 'fixed' or 'max_load'")
    parser.add_argument("--stop_spacing", type=int, default=1, help="Stop spacing. 1 means every node is a stop")
    parser.add_argument("--alpha", type=float, default=1.0, help="Modal split parameter for served O-D pairs (proportion taking bus)")
    parser.add_argument("--unserved_as_cars", type=bool, default=True, help="Allocate demand that is not served by buses to cars (True) or ignore it (False). When alpha=1.0, unserved demand is always ignored.")
    parser.add_argument("--comfort_threshold", type=float, default=1.0, help="Max load factor allowed per bus when computing service frequency")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius within each node to consider for demand allocation")
    parser.add_argument("--demand_warmup", type=float, default=0.15, help="Fraction of horizon reserved at both start and end with no demand (0.0-0.5)")
    parser.add_argument("--path_init", type=str, default="random", help="Initialize path using various schemes (possible: random, highest demand node, transit center)")
    
    # Constraints:
    parser.add_argument("--num_routes", type=int, default=16, help="Number of routes")
    parser.add_argument("--max_route_length", type=int, default=14, help="Maximum path length")
    parser.add_argument("--min_route_length", type=int, default=1, help="Minimum path length")

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
    parser.add_argument("--global_dim", type=int, default=3, help="Global dimension (additional feature to the graph-level features)")
    
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

    # Baselines have their own seed setting mechanism.
    if config["mode"] == "train" or config["mode"] == "eval":
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
        
        baseline_classes = {
            "random_walk": RandomWalk,
            "demand_cover": DemandCoverage,
            "shortest_path": ShortestPath,
            "reward_max": RewardMaximization,
            "real_world": RealWorld,
        }
        
        if config["baseline_type"] not in baseline_classes:
            raise ValueError(f"Invalid baseline type: {config['baseline_type']}. Available: {list(baseline_classes.keys())}")
        
        BaselineClass = baseline_classes[config["baseline_type"]]
        baseline = BaselineClass(env, config, num_runs=config["num_baseline_runs"], base_seed=config["seed"])
        baseline.run()

if __name__ == "__main__":
    main()

"""
Scripts: 
python main.py --mode=baseline --baseline_type=demand_cover
python main.py --mode=baseline --baseline_type=random_walk
python main.py --mode=baseline --baseline_type=shortest_path
python main.py --mode=baseline --baseline_type=reward_max
python main.py --mode=baseline --baseline_type=real_world

"""
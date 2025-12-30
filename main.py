import os
import wandb
import torch
import random
import argparse
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from rl.env import TransitEnv
from rl.models import GATV2ActorCritic
from torch_geometric.data import Data, Batch
from rl.ppo_agent import PPOAgent
from rl.mcts_agent import MCTSAgent
from rl.env_utils import (
    plot_network_and_demand,
    aggregate_results,
    write_results_summary,
    ensure_eval_results_dir,
    make_seed_output_dir,
    save_routes_json,
    pretty_print_state,
)
from rl.baselines import RandomWalk, DemandCoverage, ShortestPath, RewardMaximization, RealWorld
from rl.parallel_env import ParallelEnvManager, EpisodeResult, Transition, EvalResult


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

def perform_ppo_update(ppo: PPOAgent, episode: int, steps_elapsed: int, anneal_lr: bool, config: Dict[str, Any]) -> None:
    """
    mean_buffer_reward: average per-step rewards stored in the current memory buffer.
    Is not a true measure of the policy's performance.
    """
    # print("\n==================\n")
    # print("Memory contents:")
    # print(f"\tNumber of transitions: {len(ppo.memory)}")
    # print(f"\tRaw rewards: {ppo.memory.raw_rewards}")
    # print(f"\tActions: {ppo.memory.actions}")
    # print(f"\tLog probs: {ppo.memory.log_probs}")
    # print(f"\tValues: {ppo.memory.values}")
    # print(f"\tDones: {ppo.memory.dones}")
    # print(f"\tEpisode boundaries: {ppo.memory.episode_boundaries}")
    # print(f"\tBootstrap values: {ppo.memory.bootstrap_values}")

    if anneal_lr:
        # Anneal LR based on steps progress (not episodes)
        ppo.update_learning_rate(steps_elapsed, config["max_steps"])

    mem_len = len(ppo.memory)
    stats = ppo.update() # Also clears the memory.

    print("\n=== PPO Update ===")
    print(f"Episode: {episode}, Steps elapsed: {steps_elapsed}")
    print(f"Samples: {mem_len}")
    print(f"PG Loss: {stats['pg_loss']:.4f}")
    print(f"Value Loss: {stats['value_loss']:.4f}")
    print(f"Entropy Loss: {stats['entropy_loss']:.4f}")
    print(f"Clipping Frequency: {stats['clipping_frequency']:.4f}")
    print(f"Approx KL: {stats['approx_kl']:.4f}")
    print(f"Mean Clip Ratio: {stats['mean_clip_ratio']:.4f}")
    print("==================\n")
    if not config.get("wandb_off"):
        wandb.log({
            "ppo/policy_loss": stats['pg_loss'],
            "ppo/value_loss": stats['value_loss'],
            "ppo/entropy_loss": stats['entropy_loss'],
            "ppo/clipping_frequency": stats['clipping_frequency'], # How often the ratio was clipped.
            "ppo/approx_kl": stats['approx_kl'],
            "ppo/mean_clip_ratio": stats['mean_clip_ratio'], # Actual ratio of clipped updates.
            "ppo/learning_rate": ppo.optimizer.param_groups[0]['lr'], # Current learning rate after annealing
            "ppo/steps_elapsed": steps_elapsed, # Total steps across all episodes
        }, step=episode)

def train(config: Dict[str, Any]) -> None:
    """

    Note: 
    ------------------------------------------------------------------------------------------------
    Per-Step Rewards (Routes run a simulation only on completion):
    ------------------------------------------------------------------------------------------------
    - Step 1: Route [A] → Get proxy reward R1
    - Step 2: Route [A→B] → Get proxy reward R2  
    - Step 3: Route [A→B→C] if completed → Run full simulation → Get final reward R3
    Pro: Agent gets immediate feedback after each node addition

    ------------------------------------------------------------------------------------------------
    Value bootstraping: 
    - Required when an episode ends premeturely (truncated)
    - If terminated naturally, value = 0 (no more rewards coming)
    - If truncated, value = value of the last state. 
        - We preserve the partial solution which has some utility in learning.
        - Truncation conditions: 
            - Gets stuck in a dead-end i.e., had only 1 neighbor, (which is already in the path)   
    """
    num_workers = config.get("num_workers", 1)
    max_steps = config["max_steps"]
    print(f"Training started on network: {config['network']} with {num_workers} workers for {max_steps:,} steps")
    
    # Reference environment for dimensions and visualization
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    edge_feature_dim = env.N_EDGE_FEATURES
    
    print("\tObservation space:")
    for key, space in env.observation_space.items():
        print(f"\t  {key}: {space}")
    print(f"\tAction space: {env.action_space}")
    
    now = datetime.now()
    training_save_dir = os.path.join(config["save_dir"], f"{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}")
    os.makedirs(training_save_dir, exist_ok=True)

    # Build world for initial visualization
    temp_world = env.build_world(config.get("network"))
    env.load_demand_for_plotting(temp_world)
    output_path = os.path.join(training_save_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path)

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
    config["n_nodes"] = env.n_nodes
    policy_kwargs = get_policy_kwargs(config, node_feature_dim, edge_feature_dim)

    device = torch.device(config["device"])
    model = GATV2ActorCritic(**policy_kwargs).to(device)
    model.apply_orthogonal_init(
        hidden_gain=None,
        bias_const=0.0,
        actor_final_gain=0.01,
        critic_final_gain=1.0,
    )
    model.count_params()

    ppo = PPOAgent(model, **config)
    policy_dir = os.path.join(training_save_dir, "policies")
    os.makedirs(policy_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Parallel Training Architecture (Multiprocessing with Shared Memory):
    # -------------------------------------------------------------------------
    # Main Process (Learner):
    #   - Owns the model and optimizer
    #   - Creates shared memory tensors for model weights (share_memory_())
    #   - Collects episodes from workers (each worker runs one episode per call)
    #   - Processes transitions into PPO memory
    #   - Performs PPO update when update_frequency transitions collected
    #   - Copies updated weights to shared memory
    #
    # Worker Processes (Actors):
    #   - Each has its own TransitEnv instance
    #   - Load weights from shared memory at episode start (no queues!)
    #   - Run episodes and return complete EpisodeResult objects
    #   - Automatically see updated weights on next episode
    # -------------------------------------------------------------------------
    
    max_steps = config["max_steps"]
    
    env_manager = ParallelEnvManager(
        config=config,
        num_workers=num_workers,
    )
    
    episode, steps_elapsed, update_count, policy_path = 0, 0, 0, None
    
    try:
        env_manager.start(model, device, policy_kwargs)
        
        # -------------------------------------------------------------------------
        # Main Training Loop:
        # - Each iteration: collect num_workers episodes (one per worker)
        # - Process transitions into PPO memory
        # - Update policy when update_frequency transitions collected
        # - Stop when steps_elapsed >= max_steps
        # -------------------------------------------------------------------------
        while steps_elapsed < max_steps:
            # Collect num_workers episodes in parallel (one episode per worker)
            episode_results = env_manager.collect_episodes(num_workers)
            
            if not episode_results:
                break
            
            # Process each episode's transitions
            for ep_result in episode_results:
                episode += 1
                
                for trans in ep_result.transitions:
                    # Reconstruct PyG Data object from numpy arrays
                    obs_data = Data(
                        x=torch.from_numpy(trans.obs_x).float(),
                        edge_index=torch.from_numpy(trans.obs_edge_index).long(),
                        edge_attr=torch.from_numpy(trans.obs_edge_attr).float(),
                    )
                    obs_data.route_progress = torch.from_numpy(trans.obs_route_progress).float()
                    
                    # Store in PPO memory buffer (using raw rewards - no normalization)
                    ppo.memory.obs.append(obs_data)
                    ppo.memory.actions.append(trans.action)
                    ppo.memory.raw_rewards.append(trans.raw_reward)
                    ppo.memory.values.append(trans.value)
                    ppo.memory.log_probs.append(trans.log_prob)
                    ppo.memory.dones.append(trans.terminated)
                    ppo.memory.valid_mask.append(torch.from_numpy(trans.valid_mask).bool())
                    steps_elapsed += 1
                
                # Mark episode boundary for GAE computation
                ppo.memory.mark_episode_end(ep_result.bootstrap_value)
                
                # Log episode results
                print(f"Episode {episode} | Steps: {steps_elapsed}/{max_steps} | Reward: {ep_result.episode_reward:.2f}")
                
                sim_result = ep_result.final_sim_result
                if not config.get("wandb_off") and sim_result:
                    served = sim_result.get('completed_passengers', 0) + sim_result.get('ongoing_passengers', 0)
                    wait_seconds = sim_result.get('total_wait_completed', 0) + sim_result.get('total_wait_ongoing', 0)
                    travel_seconds = sim_result.get('total_travel_completed', 0) + sim_result.get('total_travel_ongoing', 0)
                    combined_avg_wait_minutes = (wait_seconds / served) / 60 if served > 0 else 0.0
                    combined_avg_travel_minutes = (travel_seconds / served) / 60 if served > 0 else 0.0
                    
                    wandb.log({
                        "episode/episode_total_reward": ep_result.episode_reward,
                        "episode/episode_length": ep_result.episode_length,
                        "training/steps_elapsed": steps_elapsed,
                        "training/episodes": episode,
                        "episode/demand_coverage_potential": sim_result.get('demand_coverage_potential', 0),
                        "episode/demand_coverage_actual": sim_result.get('demand_coverage_actual', 0),
                        "episode/route_overlap_ratio": sim_result.get('route_overlap_ratio', 0),
                        "episode/node_coverage": sim_result.get('node_coverage', 0),
                        "episode/service_rate": sim_result.get('service_rate', 0),
                        "episode/avg_wait_time": combined_avg_wait_minutes,
                        "episode/avg_travel_time": combined_avg_travel_minutes,
                    }, step=steps_elapsed)
            

            # Perform PPO update when enough transitions collected
            if len(ppo.memory) >= config["update_frequency"]:
                update_count += 1
                perform_ppo_update(ppo, episode, steps_elapsed, config["anneal_lr"], config)
                policy_path = os.path.join(policy_dir, f"policy_up_{update_count}_step_{steps_elapsed}.pth")
                torch.save(model.state_dict(), policy_path)
                
                # Update shared memory weights for workers
                env_manager.update_shared_weights(model)

                print(f"\n--- Running evaluation at episode {episode} ---")
                if config.get("eval_every") > 0 and update_count % config["eval_every"] == 0:
                    eval(config, policy_path, episode, training_save_dir, env_manager)
                
        
        # Final update if any transitions remain
        if len(ppo.memory) > 0:
            update_count += 1
            perform_ppo_update(ppo, episode, steps_elapsed, config["anneal_lr"], config)
            policy_path = os.path.join(policy_dir, f"policy_up_{update_count}_step_{steps_elapsed}.pth")
            torch.save(model.state_dict(), policy_path)
        
    finally:
        # Always stop workers, even if an exception occurred
        env_manager.stop()
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Steps: {steps_elapsed}, Episodes: {episode}, Updates: {update_count}")
    print(f"{'='*60}\n")
        
    
def execute_eval_runs(config: Dict[str, Any], policy_path: str, num_runs: int, base_seed: int, save_dir: str, env_manager: ParallelEnvManager) -> tuple[list, dict]:
    """
    Evaluate a trained policy for multiple runs in parallel.
    Evaluate a trained policy for multiple runs.
    - Load a saved policy
    - Run the policy on the network (take deterministic actions) and get the path
    - Get relevant metrics and log
    - Plot path and call world.analyzer.network_fancy

    Notes: 
    - During eval episode, agent constructs a route path step by step
    - The results are only meaningful at the end of the episode.
    Uses the same process pool as training for parallel evaluation.
    All workers share the same policy weights and run with different seeds.
    
    Args:
        config: Configuration dict
        policy_path: Path to the saved policy
        num_runs: Number of evaluation runs
        base_seed: Starting seed for reproducibility
        save_dir: Directory to save results
        env_manager: ParallelEnvManager (required - no sequential fallback)
    """
    print(f"Evaluating policy: {policy_path} for {num_runs} runs starting at seed: {base_seed}")
    print(f"Using parallel evaluation with {env_manager.num_workers} workers...")
    
    # Load policy weights into shared memory for eval
    state_dict = torch.load(policy_path, map_location="cpu")
    for name, param in state_dict.items():
        env_manager.shared_model_state[name].copy_(param)
    
    # Run parallel evaluations
    eval_results = env_manager.run_parallel_eval(
        num_runs=num_runs,
        base_seed=base_seed,
        seed_offset=config["eval_seed_offset"],
    )
    
    # Convert EvalResult to dict format and save routes
    results = []
    for eval_result in eval_results:
        seed = eval_result.seed
        seed_dir, _ = make_seed_output_dir(save_dir, seed)
        
        # Save routes
        save_routes_json(seed_dir, eval_result.routes)
        
        # Convert to dict format expected by aggregate_results
        result = eval_result.metrics.copy()
        result['routes'] = eval_result.routes
        results.append(result)

    aggregated = aggregate_results(results)
    return results, aggregated

def single_eval_run(config: Dict[str, Any], policy_path: str, save_dir: str, run_number: int) -> dict:
    """
    Execute a single evaluation run.
    """
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    edge_feature_dim = env.N_EDGE_FEATURES
    config["n_nodes"] = env.n_nodes
    policy_kwargs = get_policy_kwargs(config, node_feature_dim, edge_feature_dim)

    model = GATV2ActorCritic(**policy_kwargs)
    state_dict = torch.load(policy_path, map_location=config["device"])
    model.load_state_dict(state_dict)
    model.to(config["device"])
    model.eval() # eval mode
    
    pyg_converter = CachedPyGConverter(config["device"])

    state, _ = env.reset()
    terminated = False
    eval_episode_steps = 0
    episode_total_reward = 0.0
    while not terminated:
        data = pyg_converter.convert(state)
        batch = Batch.from_data_list([data]).to(config["device"])  # Data already on device
        valid_list = env._get_valid_indices()
        
        num_nodes = batch.num_nodes
        valid_mask = torch.zeros(1, num_nodes, dtype=torch.bool, device=config["device"])

        if len(valid_list) > 0:
            for local_idx in valid_list:
                valid_mask[0, local_idx] = True
            # print(f"\tValid mask: shape: {valid_mask.shape}, value: {valid_mask}")

            with torch.no_grad():
                action_tensor, _, _ = model.act(batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                valid_mask=valid_mask,
                stochastic=False) # Deterministic action at test time.
            
        else:
            # When valid indices are empty
            action_tensor = torch.tensor([env.NO_VALID_ACTION], dtype=torch.long, device=config["device"])

        action = action_tensor.cpu().item()
        next_state, reward, terminated, _, sim_result = env.step(action) # Truncation is not used.
        state = next_state
        eval_episode_steps += 1
        episode_total_reward += reward

    # Calculate combined metrics
    served = sim_result['completed_passengers'] + sim_result['ongoing_passengers']
    wait_seconds = sim_result['total_wait_completed'] + sim_result['total_wait_ongoing']
    travel_seconds = sim_result['total_travel_completed'] + sim_result['total_travel_ongoing']

    combined_avg_wait_minutes = (wait_seconds / served) / 60 if served > 0 else 0.0
    combined_avg_travel_minutes = (travel_seconds / served) / 60 if served > 0 else 0.0

    # Generate visualizations for this run
    env.render(save_dir, f"eval_run_{run_number}.png")

    if config["save_animations"]: # Takes almost 40 seconds per run.
        env.world.analyzer.network_fancy(
            animation_speed_inverse=10,
            figsize=11,
            sample_ratio=1.0,
            interval=5,
            trace_length=5,
            network_font_size=11,
            antialiasing=False,
            file_name=os.path.join(save_dir, f"eval_anim_run_{run_number}.gif"),
            save_as_mp4=False,
            bus_only = True # 
        )

    return {
        'episode_total_reward': float(episode_total_reward),
        'episode_length': eval_episode_steps,
        'demand_coverage_potential': sim_result['demand_coverage_potential'],
        'demand_coverage_actual': sim_result['demand_coverage_actual'],
        'route_overlap_ratio': sim_result['route_overlap_ratio'],
        'node_coverage': sim_result['node_coverage'],
        'completed_passengers': sim_result['completed_passengers'],
        'ongoing_passengers': sim_result['ongoing_passengers'],
        'total_onboarded_count': sim_result['total_onboarded_count'],
        'wanting_to_onboard': sim_result['wanting_to_onboard'],
        'service_rate': sim_result['service_rate'],
        'combined_avg_wait_minutes': combined_avg_wait_minutes,
        'transfer_rate': sim_result['transfer_rate'],
        'combined_avg_travel_minutes': combined_avg_travel_minutes,
        'route_efficiency': sim_result['route_efficiency'],
        'fleet_size': sim_result['fleet_size'],
        'bus_utilization': sim_result['bus_utilization'],
        'routes': env.all_routes,
    }

def eval(config: Dict[str, Any], policy_path: str, episode: int, save_dir: str, env_manager: ParallelEnvManager) -> Dict[str, float]:
    """
    Evaluate a trained policy using parallel workers.
    Evaluate a trained policy
    - Load a saved policy
    - Run the policy on the network (take deterministic actions) and get the path
    - Get relevant metrics and log
    - Plot path and call world.analyzer.network_fancy

    Notes:
    - During eval episode, agent constructs a route path step by step
    - The results are only meaningful at the end of the episode.
    Args:
        config: Configuration dict
        policy_path: Path to saved policy
        episode: Current episode number (for logging)
        save_dir: Directory to save results
        env_manager: ParallelEnvManager (required for parallel eval)
    """
    print("Evaluating policy: ", policy_path)

    num_runs = config["num_eval_runs"]
    eval_root_dir = ensure_eval_results_dir(save_dir)
    episode_dir = ensure_eval_results_dir(eval_root_dir, folder_name="", episode=episode)
    starting_seed = config["seed"] 

    # Run parallel evaluations
    results, aggregated = execute_eval_runs(
        config, policy_path, num_runs, starting_seed, episode_dir, env_manager
    )

    # Save summary JSON with statistical information (works for any number of runs)
    write_results_summary(aggregated, num_runs, episode_dir, 'eval_results_summary.json')

    # Log averaged results to wandb (for single run, this is just the single result)
    if not config.get("wandb_off"):
        wandb.log({
            # Total reward accumulated across the episode
            "eval/episode_total_reward": aggregated['episode_total_reward'],
            "eval/episode_length": aggregated['episode_length'],
            "eval/steps_elapsed": aggregated['episode_length'],

            # reward related and others
            "eval/demand_coverage_potential": aggregated['demand_coverage_potential'],
            "eval/demand_coverage_actual": aggregated['demand_coverage_actual'],
            "eval/route_overlap_ratio": aggregated['route_overlap_ratio'],
            "eval/node_coverage": aggregated['node_coverage'],
            "eval/completed_passengers": aggregated['completed_passengers'],
            "eval/ongoing_passengers": aggregated['ongoing_passengers'],
            "eval/total_onboarded_count": aggregated['total_onboarded_count'],
            "eval/wanting_to_onboard": aggregated['wanting_to_onboard'],

            # Performance related
            "eval/service_rate": aggregated['service_rate'],
            "eval/avg_wait_time": aggregated['combined_avg_wait_minutes'],
            "eval/transfer_rate": aggregated['transfer_rate'],
            "eval/avg_travel_time": aggregated['combined_avg_travel_minutes'],
            "eval/route_efficiency": aggregated['route_efficiency'],
            "eval/fleet_size": aggregated['fleet_size'],
            "eval/bus_utilization": aggregated['bus_utilization'],
        }, step=episode)

    return aggregated

def get_config() -> Dict[str, Any]:
    """
    A unified config interface for both main and sweep.
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

def get_policy_kwargs(config: Dict[str, Any], node_feature_dim: int, edge_feature_dim: int) -> Dict[str, Any]:
    """
    Dropout has been disabled. 
    In the parallelized setup, it breaks some core PPO assumptions.
    """
    num_gat_blocks = config.get("num_gat_blocks", 4)
    return {
        "n_node_features": node_feature_dim,
        "proj_out": config.get("proj_out", 64),
        "num_gat_blocks": num_gat_blocks,
        "gat_channels": config.get("gat_channels", [128, 128, 64, 64]),
        "num_heads": config.get("num_heads", [8, 8, 4, 4]),
        "attn_dropout": [0.0] * num_gat_blocks,
        "feat_dropout": [0.0] * num_gat_blocks,
        "actor_head_dropout": 0.0,
        "critic_head_dropout": 0.0,
        "concat": config.get("concat_heads", False),
        "activation": config.get("activation", "tanh"),
        "n_edge_features": edge_feature_dim,
        "actor_head_layers": config.get("actor_head_layers", [256, 128, 64]),
        "critic_head_layers": config.get("critic_head_layers", [256, 128, 64]),
        "critic_readout_type": config.get("critic_readout_type", "sum"),
    }

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for CLI usage.
    Define common RL, system, and sweepable hyperparameters.
    """
    parser = argparse.ArgumentParser(description="RL training/evaluation entrypoint")
    parser.add_argument("--algorithm", choices=["ppo", "mcts"], default="ppo", help="Select learning algorithm")
    # Simulation setup: 
    parser.add_argument("--network", choices=["sioux_falls", "bloomington",], default="bloomington", help="Network selection")
    parser.add_argument("--mode", choices=["train", "eval", "baseline"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available. Pass --gpu to enable.")
    parser.add_argument("--horizon", type=int, default=10000, help="Simulation horizon") # 10k = 2.7 hours
    parser.add_argument("--delta_t", type=float, default=1, help="Simulation time step") # Increasing delta_t makes simulation faster.
    parser.add_argument("--delta_n", type=int, default=5, help="Simulation platoon size") # Increasing delta_n also makes simulation faster. Does not apply for bus passenger demand.
    parser.add_argument("--bus_capacity", type=int, default=40, help="Bus capacity")
    parser.add_argument("--stop_duration", type=int, default=60, help="Stop duration")
    parser.add_argument("--update_frequency", type=int, default=64, help="Update PPO when memory has N samples") 
    parser.add_argument("--max_steps", type=int, default=1_000_000, help="Total training steps (transitions)")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N updates to the policy")
    parser.add_argument("--baseline_type", type=str, default="demand_cover", help="Can be random_walk, reward_max, demand_cover, shortest_path, real_world")
    parser.add_argument("--num_eval_runs", type=int, default=5, help="Number of runs (over which we average the results) for both evaluation and baselines")
    parser.add_argument("--eval_seed_offset", type=int, default=2, help="Add offset to starting seed for evaluation outputs")
    parser.add_argument("--save_animations", action="store_true", help="Save animations for evaluation")
    
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers (1=single worker, >1=parallel)")
    # Learning environment specific: 
    parser.add_argument("--service_frequency_mode", type=str, default="max_load", help="Service frequency mode, e.g., 'fixed' or 'max_load'")
    parser.add_argument("--stop_spacing", type=int, default=1, help="Stop spacing. 1 means every node is a stop")
    parser.add_argument("--alpha", type=float, default=0.3, help="Modal split parameter for served O-D pairs (proportion taking bus)")
    parser.add_argument("--ignore_unserved", action="store_true", help="If this flag is set, demand that is not served by buses is ignored. Otherwise, it is allocated to cars.")
    parser.add_argument("--comfort_threshold", type=float, default=1.0, help="Max load factor allowed per bus when computing service frequency")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius within each node to consider for demand allocation")
    parser.add_argument("--demand_warmup", type=float, default=0.15, help="Fraction of horizon reserved at both start and end with no demand (0.0-0.5)")
    parser.add_argument("--route_init", type=str, default="transit_center", help="Initialize path using various schemes (possible: random, highest demand node, transit_center)")
    parser.add_argument("--transit_center_node", type=str, default="96", help="Node identifier to use when route_init is 'transit_center'")
    
    # Constraints:
    parser.add_argument("--num_routes", type=int, default=16, help="Number of routes")
    parser.add_argument("--max_route_length", type=int, default=14, help="Maximum path length")
    parser.add_argument("--min_route_length", type=int, default=2, help="Minimum path length")

    # PPO params: 
    parser.add_argument("--K_epochs", type=int, default=8, help="Number of PPO epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--clip_frac", type=float, default=0.1, help="PPO clipping ratio for policy loss")
    parser.add_argument("--vf_clip_param", type=float, default=0.5, help="PPO clipping ratio for value loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--anneal_lr", action="store_true", help="Anneal learning rate over training episodes")
    
    # parser.add_argument("--activation", type=str, default="elu", help="Activation function") # elu better for deeper networks?
    parser.add_argument("--concat_heads", action="store_true", help="Concatenate attention heads")
    # WandB:
    parser.add_argument("--wandb_project", type=str, default="transit_design", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="bibek-poudel", help="WandB entity/team name")
    parser.add_argument("--wandb_off", action="store_true", help="Disable WandB logging")

    parser.add_argument("--save_dir", type=str, default="./training_data", help="Directory to save training data")
    parser.add_argument("--saved_policy_path", type=str, default="./training_data/policies/policy_final.pth", help="Path to saved policy that you want to evaluate")
    # parser.add_argument("--demand_method", choices=["volume", "flow"], default="volume", help="Demand allocation method")
    return parser


def main() -> None:
    """
    """
    config = get_config()
    algorithm = config.get("algorithm")
    mode = config["mode"]

    # Baselines can be executed regardless of algorithm selection.
    if mode == "baseline":
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
        
        BaselineClass = baseline_classes[config["baseline_type"]]
        baseline = BaselineClass(env, config, num_runs=config["num_eval_runs"], base_seed=config["seed"])
        baseline.run()
        return

    if mode in {"train", "eval"}:
        set_global_seeds(config["seed"])

    device = torch.device("cuda" if (config["gpu"] and torch.cuda.is_available()) else "cpu")
    config["device"] = device

    if algorithm == "ppo":
        if mode == "train":
            if not config.get("wandb_off"):
                wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
            train(config)
            if not config.get("wandb_off"):
                wandb.finish()
            return

        if mode == "eval":
            os.makedirs(config["save_dir"], exist_ok=True)
            config["wandb_off"] = True
            policy_path = config["saved_policy_path"]
            
            # Create env_manager for parallel evaluation
            env = TransitEnv(config)
            node_feature_dim = env.N_NODE_FEATURES
            edge_feature_dim = env.N_EDGE_FEATURES
            config["n_nodes"] = env.n_nodes
            policy_kwargs = get_policy_kwargs(config, node_feature_dim, edge_feature_dim)
            
            # Load model to get architecture for workers
            device = torch.device(config["device"])
            model = GATV2ActorCritic(**policy_kwargs).to(device)
            state_dict = torch.load(policy_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Create and start env_manager
            num_workers = config.get("num_workers", 4)
            env_manager = ParallelEnvManager(config=config, num_workers=num_workers)
            env_manager.start(model, device, policy_kwargs)
            
            try:
                eval(config, policy_path, "final", config["save_dir"], env_manager)
            finally:
                env_manager.stop()
            
            print(f"Evaluation completed. Results saved to: {config['save_dir']}")
            return


    if algorithm == "mcts":
        env = TransitEnv(config)
        mcts_agent = MCTSAgent(env, config)

        if mode == "train":
            # TODO: Implement MCTS training pipeline in rl/mcts_agent.py
            return

        if mode == "eval":
            # TODO: Implement MCTS evaluation pipeline in rl/mcts_agent.py
            return



if __name__ == "__main__":
    main()

"""
Scripts: 
python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=real_world --save_animations --route_init=transit_center --alpha=0.3

python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=demand_cover --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=shortest_path --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=reward_max --save_animations --route_init=random --alpha=0.3

python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=real_world --save_animations --route_init=transit_center --alpha=1.0

python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=demand_cover --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=shortest_path --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=reward_max --save_animations --route_init=random --alpha=1.0

python main.py --gpu --anneal_lr --gpu

"""

import os
import warnings
import wandb
import torch
import numpy as np
import pandas as pd
from collections import deque
from typing import Any, Dict
from datetime import datetime
from rl.env import TransitEnv
from rl.models import GATV2ActorCritic
from torch_geometric.data import Data
from rl.ppo_agent import PPOAgent
from rl.env_utils import plot_network_and_demand, aggregate_results, write_results_summary, ensure_eval_step_update_dir, make_seed_output_dir, save_routes_json, calculate_combined_metrics
from rl.parallel_env import ParallelEnvManager
from rl.ppo_utils import RunningMeanStd
from config import get_config, set_global_seeds

# Suppress torch-scatter installation warning from PyTorch Geometric
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

def perform_ppo_update(ppo: PPOAgent, episode: int, steps_elapsed: int, update_count: int, anneal_lr: bool, config: Dict[str, Any]) -> None:
    """
    mean_buffer_reward: average per-step rewards stored in the current memory buffer.
    Is not a true measure of the policy's performance.
    """

    if anneal_lr:
        ppo.update_learning_rate(steps_elapsed, config["max_steps"])
    mem_len = len(ppo.memory)
    stats = ppo.update() # Also clears the memory.

    # print("\n=== PPO Update ===")
    # print(f"Episode: {episode}, Steps elapsed: {steps_elapsed}")
    # print(f"Samples: {mem_len}")
    # print(f"PG Loss: {stats['pg_loss']:.4f}")
    # print(f"Value Loss: {stats['value_loss']:.4f}")
    # print(f"Entropy Loss: {stats['entropy_loss']:.4f}")
    # print(f"Clipping Frequency: {stats['clipping_frequency']:.4f}")
    # print(f"Approx KL: {stats['approx_kl']:.4f}")
    # print(f"Mean Clip Ratio: {stats['mean_clip_ratio']:.4f}")
    # print("==================\n")
    if not config.get("wandb_off"):
        wandb.log({
            "ppo/policy_loss": stats['pg_loss'],
            "ppo/value_loss": stats['value_loss'],
            "ppo/entropy_loss": stats['entropy_loss'],
            "ppo/clipping_frequency": stats['clipping_frequency'], # How often the ratio was clipped.
            "ppo/approx_kl": stats['approx_kl'],
            "ppo/mean_clip_ratio": stats['mean_clip_ratio'], # Actual ratio of clipped updates.
            "ppo/learning_rate": ppo.optimizer.param_groups[0]['lr'], # Current learning rate after annealing
            "ppo/update_count": update_count,
        }, step=steps_elapsed)

def train(config: Dict[str, Any], is_sweep: bool = False) -> None:
    """
    Main PPO training function for both standalone and sweep use.

    Args:
        config: Configuration dictionary
        is_sweep: If True, assumes wandb is already initialized by sweep agent

    Parallel Training Architecture (Multiprocessing with Shared Memory):
    Main Process (Learner):
    - Owns the model and optimizer
    - Creates shared memory tensors for model weights 
    - Dispatches one episode-collection command to each worker per PPO round
    - Collects one full-episode trajectory from each worker (barrier sync)
    - Processes transitions (with precomputed advantages/returns) into PPO memory
    - Performs exactly one PPO update per collection round
    - Copies updated weights to shared memory

    Worker Processes (Actors):
    - Each has its own TransitEnv instance
    - Load weights from shared memory at episode start
    - Run one full episode and return one terminal RolloutChunk
    - Receive next collect command only after learner update

    Note:
    - Each PPO update uses one complete trajectory per worker from the same policy version.
    - This removes cross-round staleness caused by asynchronous worker restarts.

    ------------------------------------------------------------------------------------------------
    Per-Step Rewards (Routes run a simulation only on completion):
    ------------------------------------------------------------------------------------------------
    - Step 1: Route [A] → Get proxy reward R1
    - Step 2: Route [A→B] → Get proxy reward R2  
    - Step 3: Route [A→B→C] if completed → Run full simulation → Get final reward R3
    Pro: Agent gets immediate feedback after each node addition

    ------------------------------------------------------------------------------------------------
    Value bootstrapping:
    - Rollouts are full episodes, so bootstrap value is always 0 at trajectory end.
    - GAE credit assignment spans the full episode within each worker trajectory.
    """
    # Initialize wandb for standalone runs (sweep handles its own init)
    if not is_sweep and not config.get("wandb_off"):
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)

    num_workers = config.get("num_ppo_workers")
    max_steps = config["max_steps"]
    print(f"Training started: network={config['network']}, workers={num_workers}, max_steps={max_steps:,}")
    
    # Reference environment for dimensions and visualization
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    edge_feature_dim = env.N_EDGE_FEATURES
    
    # print("\tObservation space:")
    # for key, space in env.observation_space.items():
    #     print(f"\t  {key}: {space}")
    # print(f"\tAction space: {env.action_space}")
    
    now = datetime.now()
    training_save_dir = os.path.join(config["save_dir"], f"{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}")
    os.makedirs(training_save_dir, exist_ok=True)

    # Build world for initial visualization
    temp_world = env.build_world(config.get("network"))
    env.load_demand_for_plotting(temp_world)
    output_path = os.path.join(training_save_dir, f"00_{config.get('network')}_demand_network.png")
    plot_network_and_demand(temp_world, output_path)

    # print(f"\nNetwork details:")
    # print(f"\tNumber of nodes: {env.n_nodes}")
    # print(f"\tNumber of edges: {env.n_edges}")
    # print(f"\tNode list: {env.node_list}")
    # print(f"\tIdx to node: {env.idx_to_node}")

    # print("\n\tAdjacency list:")
    # for node in env.node_list:
    #     neighbors = env.adj[node]
    #     print(f"\t  {node}: {', '.join(neighbors)}")
    # print(f"\tMax degree: {env.max_degree}, Min degree: {env.min_degree}")

    od_df = pd.DataFrame(env.od_matrix, index=env.node_list, columns=env.node_list)
    # with pd.option_context('display.max_rows', 20, 'display.max_columns', 20, 'display.width', 120):
    #     print("\n\tOD matrix (preview):\n", od_df.round(4))
    # print(f"\tMax demand: {env.max_demand}, Min demand: {env.min_demand}")

    # Set policy hyper-param defaults
    config["n_nodes"] = env.n_nodes
    policy_kwargs = get_policy_kwargs_ppo(config, node_feature_dim, edge_feature_dim)

    device = torch.device(config["device"])
    model = GATV2ActorCritic(**policy_kwargs).to(device)
    model.apply_orthogonal_init(
        hidden_gain=None,
        bias_const=0.0,
        actor_final_gain=0.01,
        critic_final_gain=1.0,
    )
    model.count_params()

    # TODO: Uncomment when PyTorch 2.10 is released (adds Python 3.14 support)
    # model = torch.compile(model)

    ppo = PPOAgent(model, **config)
    save_policy = config.get("save_policy_ppo", False)
    policy_dir = os.path.join(training_save_dir, "ppo_policies")
    if save_policy:
        os.makedirs(policy_dir, exist_ok=True)

    max_steps = config["max_steps"]
    env_manager = ParallelEnvManager(config=config, num_workers=num_workers)
    episode_count, steps_elapsed, update_count, policy_path = 0, 0, 0, None
    
    # Rolling buffer for episode metrics - logged at each PPO update for uniform x-axis spacing
    episode_buffer_size = 2 * num_workers
    recent_episode_rewards = deque(maxlen = episode_buffer_size)
    recent_episode_lengths = deque(maxlen = episode_buffer_size)
    recent_episode_metrics = deque(maxlen = episode_buffer_size)  # Full sim_result dicts
    
    # Return normalization using Welford's running std (OpenAI-style).
    # Flow: Learner maintains return_rms and shares reward_scale = return_rms.std with workers.
    #       Workers scale rewards before GAE: scaled_reward = raw_reward / reward_scale.
    #       This keeps critic values and scaled rewards in the same units (consistent GAE).
    #       Learner updates return_rms with incoming returns and pushes new scale to workers.
    # See: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    return_rms = RunningMeanStd()
    
    try:
        env_manager.start(model, policy_kwargs)
        
        # Start all workers collecting episodes
        env_manager.start_collection()
        
        # Log initial state at step 0 so all metrics start at the same x-axis point
        if not config.get("wandb_off"):
            wandb.log({
                "episode/episode_count": 0,
                "ppo/update_count": 0,
            }, step=0)
        
        # Main Training Loop:
        while steps_elapsed < max_steps:
            # 1. Barrier collect: one complete episode from each worker for this PPO round.
            chunks = env_manager.collect_rollouts(expected_workers=num_workers)
            if not chunks:
                break
            
            # 2. Process trajectories and add to memory.
            # Advantages/returns are computed in workers using scaled rewards (worker-side normalization).
            batch_steps = 0
            episode_lengths_this_update = []
            # print(f"[DEBUG] Learner: Processing {len(chunks)} chunks into PPO memory...")
            
            # Recover RAW returns from scaled returns to update running stats in raw units.
            # raw_return = scaled_return * reward_scale_used (since scaled_return = raw_return / scale)
            all_raw_returns = []
            for chunk in chunks:
                raw_returns = [r * chunk.reward_scale_used for r in chunk.returns]
                all_raw_returns.extend(raw_returns)
            
            # Update running stats with RAW returns (not scaled!)
            all_raw_returns_arr = np.array(all_raw_returns, dtype=np.float64)
            old_std = return_rms.std
            return_rms.update(all_raw_returns_arr)
            # print(f"[DEBUG] Return normalization: std {old_std:.4f} -> {return_rms.std:.4f}, raw_returns range=[{all_raw_returns_arr.min():.2f}, {all_raw_returns_arr.max():.2f}]")
            
            # Update shared reward scale for workers.
            # Workers consume this at episode start (full-episode collection mode),
            # so reward normalization remains constant within an episode trajectory.
            env_manager.update_reward_scale(return_rms.std)
            
            # Add transitions to memory (advantages/returns already normalized by workers)
            for chunk in chunks:
                if not chunk.is_terminal:
                    raise RuntimeError("Expected terminal full-episode chunk in PPO barrier collection.")

                episode_lengths_this_update.append(int(chunk.episode_length))
                for idx, transition in enumerate(chunk.transitions):
                    obs_data = Data(
                        x=torch.from_numpy(transition.obs_x).float(),
                        edge_index=torch.from_numpy(transition.obs_edge_index).long(),
                        edge_attr=torch.from_numpy(transition.obs_edge_attr).float(),
                    )
                    obs_data.route_progress = torch.from_numpy(transition.obs_route_progress).float()
                    
                    ppo.memory.obs.append(obs_data)
                    ppo.memory.actions.append(transition.action)
                    ppo.memory.raw_rewards.append(transition.raw_reward)
                    ppo.memory.values.append(transition.value)
                    ppo.memory.log_probs.append(transition.log_prob)
                    ppo.memory.dones.append(transition.terminated)
                    ppo.memory.valid_mask.append(torch.from_numpy(transition.valid_mask).bool())
                    ppo.memory.advantages.append(chunk.advantages[idx])
                    ppo.memory.returns.append(chunk.returns[idx])
                
                batch_steps += len(chunk.transitions)

                # Full-episode barrier collection guarantees terminal chunks here.
                episode_count += 1

                # Buffer episode metrics for aggregated logging at PPO update time
                recent_episode_rewards.append(chunk.episode_reward)
                recent_episode_lengths.append(chunk.episode_length)
                sim_result = chunk.final_sim_result
                if sim_result:
                    recent_episode_metrics.append(sim_result)

            effective_update_frequency = int(np.sum(episode_lengths_this_update))
            mean_episode_len_this_update = float(np.mean(episode_lengths_this_update)) if episode_lengths_this_update else 0.0
            steps_elapsed = min(steps_elapsed + batch_steps, max_steps)
            
            # Progress logging: show update number, steps, episodes at each PPO update
            progress_pct = 100.0 * steps_elapsed / max_steps
            print(
                f"Update {update_count + 1:4d} | Steps: {steps_elapsed:7,}/{max_steps:,} ({progress_pct:5.1f}%) "
                f"| Episodes: {episode_count:4d} | round_steps: {effective_update_frequency:4d} "
                f"| mean_ep_len: {mean_episode_len_this_update:5.1f} | return_std: {return_rms.std:.2f}"
            )
            
            # 3. Perform one PPO update for this synchronized collection round.
            update_count += 1
            perform_ppo_update(ppo, episode_count, steps_elapsed, update_count, config["anneal_lr"], config)
            if save_policy:
                policy_path = os.path.join(policy_dir, f"policy_up_{update_count}_step_{steps_elapsed}.pth")
                torch.save(model.state_dict(), policy_path)

            env_manager.update_shared_weights(model)

            # 4. Start next round only after updated weights are published.
            if steps_elapsed < max_steps:
                env_manager.start_collection()
            
            # 5. Log aggregated episode metrics (mean of recent episodes in buffer)
            if not config.get("wandb_off"):
                episode_log = {
                    "episode/episode_count": episode_count, 
                    "ppo/return_rms_std": return_rms.std,
                    "ppo/effective_update_frequency": effective_update_frequency,
                    "ppo/mean_episode_length_per_update": mean_episode_len_this_update,
                }
                if recent_episode_rewards:
                    episode_log["episode/mean_reward"] = float(np.mean(recent_episode_rewards))
                    episode_log["episode/mean_length"] = float(np.mean(recent_episode_lengths))
                if recent_episode_metrics:
                    ml = list(recent_episode_metrics)
                    wait_times, travel_times = [], []
                    for m in ml:
                        wait_min, travel_min = calculate_combined_metrics(m)
                        wait_times.append(wait_min)
                        travel_times.append(travel_min)
                    episode_log.update({
                        "episode/demand_coverage_potential": np.mean([m['demand_coverage_potential'] for m in ml]),
                        "episode/demand_coverage_actual": np.mean([m['demand_coverage_actual'] for m in ml]),
                        "episode/route_overlap_ratio": np.mean([m['route_overlap_ratio'] for m in ml]),
                        "episode/node_coverage": np.mean([m['node_coverage'] for m in ml]),
                        "episode/service_rate": np.mean([m['service_rate'] for m in ml]),
                        "episode/avg_wait_time": np.mean(wait_times),
                        "episode/avg_travel_time": np.mean(travel_times),
                    })
                wandb.log(episode_log, step=steps_elapsed)

            # Evaluate the policy after a certain number of updates
            if config.get("ppo_eval_every") > 0 and update_count % config["ppo_eval_every"] == 0:
                # print(f"\n--- Running evaluation at update {update_count} (steps {steps_elapsed}) ---")
                eval(config, update_count, steps_elapsed, training_save_dir, env_manager, policy_path=policy_path, model=model)
                
    finally:
        # Always stop workers, even if an exception occurred
        env_manager.stop()
    
    print(f"\nTraining Complete: steps={steps_elapsed:,}, episodes={episode_count}, updates={update_count}")

    # Finish wandb for standalone runs
    if not is_sweep and not config.get("wandb_off"):
        wandb.finish()


def eval(config: Dict[str, Any], update_count: int | str, steps_elapsed: int, save_dir: str, env_manager: ParallelEnvManager, policy_path: str = None, model: torch.nn.Module = None) -> Dict[str, float]:
    """
    Evaluate a trained policy using parallel workers.

    Args:
        config: Configuration dict
        update_count: Policy update number (or 'final' for standalone eval)
        steps_elapsed: Global steps elapsed at the time of evaluation
        save_dir: Directory to save results
        env_manager: ParallelEnvManager for parallel eval
        policy_path: Path to saved policy (used when save_policy_ppo is on)
        model: Model to evaluate (used when save_policy_ppo is off)
    """
    num_runs = config["num_eval_runs"]
    episode_dir = ensure_eval_step_update_dir(save_dir, update=update_count, steps=steps_elapsed, folder_name="eval_results")

    # Run parallel evaluations
    eval_results = env_manager.run_parallel_eval(
        num_runs=num_runs,
        base_seed=config["seed"],
        seed_offset=config["eval_seed_offset"],
        policy_path=policy_path,
        state_dict=model.state_dict() if model is not None and policy_path is None else None,
    )
    
    # Process results: save routes and convert to dict format
    results = []
    for eval_result in eval_results:
        seed_dir, _ = make_seed_output_dir(episode_dir, eval_result.seed)
        save_routes_json(seed_dir, eval_result.routes)
        eval_result.metrics['routes'] = eval_result.routes
        results.append(eval_result.metrics)

    aggregated = aggregate_results(results)
    write_results_summary(aggregated, num_runs, episode_dir, 'eval_results_summary.json')

    # Log to wandb
    if not config.get("wandb_off"):
        wandb.log({
            # Total reward accumulated across the episode
            "eval/episode_total_reward": aggregated['episode_total_reward'],
            # Terminal reward only (for fair comparison with MCTS which uses terminal-only rewards)
            "eval/episode_terminal_reward": aggregated['episode_terminal_reward'],
            "eval/episode_length": aggregated['episode_length'],

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
        }, step=steps_elapsed)

    return aggregated

def get_policy_kwargs_ppo(config: Dict[str, Any], node_feature_dim: int, edge_feature_dim: int) -> Dict[str, Any]:
    """
    Dropout has been disabled.
    In the parallelized setup, it breaks some core PPO assumptions and causes policy mismatch.
    """
    n = config.get("num_gat_blocks", 4)
    half = n // 2
    
    # gat_channels and num_heads must be specified together
    if ("gat_channels" in config) != ("num_heads" in config):
        raise ValueError("gat_channels and num_heads must be specified together")
    
    # Defaults: 4 blocks -> [128,128,64,64], [8,8,4,4]
    #           6 blocks -> [128,128,128,64,64,64], [8,8,8,4,4,4]
    #           8 blocks -> [128,128,128,128,64,64,64,64], [8,8,8,8,4,4,4,4]
    gat_channels = config.get("gat_channels", [128] * half + [64] * (n - half))
    num_heads = config.get("num_heads", [8] * half + [4] * (n - half))


    return {
        "n_node_features": node_feature_dim,
        "proj_out": config.get("proj_out", 64),
        "num_gat_blocks": n,
        "gat_channels": gat_channels,
        "num_heads": num_heads,
        "attn_dropout": [0.0] * n,
        "feat_dropout": [0.0] * n,
        "actor_head_dropout": 0.0,
        "critic_head_dropout": 0.0,
        "concat": config.get("concat_heads", False),
        "activation": config.get("activation", "tanh"),
        "n_edge_features": edge_feature_dim,
        "actor_head_layers": config.get("actor_head_layers", [256, 128, 64]),
        "critic_head_layers": config.get("critic_head_layers", [256, 128, 64]),
        "critic_readout_type": config.get("critic_readout_type", "sum"),
    }

# =============================================================================
# PPO eval entry point. Called from main.py when algorithm == "ppo" and mode == "eval".
# For training, use train() directly.
# =============================================================================

def ppo_eval(config: Dict[str, Any]) -> None:
    """
    Entry point for standalone PPO evaluation mode (without training).
    Sets up environment manager, loads policy, runs parallel evaluation.
    """
    import os
    os.makedirs(config["save_dir"], exist_ok=True)
    config["wandb_off"] = True
    policy_path = config["saved_policy_path"]
    
    # Create env_manager for parallel evaluation
    env = TransitEnv(config)
    node_feature_dim = env.N_NODE_FEATURES
    edge_feature_dim = env.N_EDGE_FEATURES
    config["n_nodes"] = env.n_nodes
    policy_kwargs = get_policy_kwargs_ppo(config, node_feature_dim, edge_feature_dim)
    
    # Load model to get architecture for workers
    device = torch.device(config["device"])
    model = GATV2ActorCritic(**policy_kwargs).to(device)
    state_dict = torch.load(policy_path, map_location=device)
    model.load_state_dict(state_dict)
    # TODO: Uncomment when PyTorch 2.10 is released (adds Python 3.14 support)
    # model = torch.compile(model)

    # Create and start env_manager
    num_workers = config.get("num_ppo_workers")
    env_manager = ParallelEnvManager(config=config, num_workers=num_workers)
    env_manager.start(model, policy_kwargs)
    
    # Use new step/update-keyed saving. For standalone eval, mark update='final' and steps=0.
    eval(config, update_count="final", steps_elapsed=0, save_dir=config["save_dir"], env_manager=env_manager, policy_path=policy_path)
    env_manager.stop()
    
    # print(f"Evaluation completed. Results saved to: {config['save_dir']}")

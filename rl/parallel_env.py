"""
Parallel environment manager for distributed PPO training.

Uses Python multiprocessing with PyTorch shared memory for efficient weight sharing.
- Model tensors are placed in shared memory via tensor.share_memory_()
- Workers read directly from shared memory (no weight queues!)
- Main process updates weights in-place, workers automatically see changes
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from torch_geometric.data import Data, Batch


# Use spawn context for clean process isolation (required for CUDA compatibility)
mp_ctx = mp.get_context('spawn')


@dataclass
class Transition:
    """
    A single environment transition for PPO training.
    Stored as numpy/python types for efficient pickling across processes.
    """
    obs_x: np.ndarray               # Node features [N, D]
    obs_edge_index: np.ndarray      # Edge index [2, E]
    obs_edge_attr: np.ndarray       # Edge features [E, edge_dim]
    obs_route_progress: np.ndarray  # Route progress [num_routes]
    action: int
    raw_reward: float
    value: float
    log_prob: float
    terminated: bool
    valid_mask: np.ndarray          # [1, num_nodes] boolean


@dataclass
class EpisodeResult:
    """
    Complete episode data including all transitions and final metrics.
    """
    worker_id: int
    transitions: List[Transition] = field(default_factory=list)
    bootstrap_value: float = 0.0
    episode_reward: float = 0.0
    episode_length: int = 0
    final_sim_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """
    Result of a single evaluation run.
    Contains metrics and routes (no transitions needed for eval).
    """
    run_id: int
    seed: int
    episode_reward: float = 0.0
    episode_length: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    routes: List[Any] = field(default_factory=list)


def run_single_episode(worker_id: int, config: Dict[str, Any], policy_kwargs: Dict[str, Any], shared_model_state: Dict[str, torch.Tensor]) -> EpisodeResult:
    """
    Run a single episode in a worker process.
    
    Args:
        worker_id: ID of this worker
        config: Environment and training config
        policy_kwargs: Model architecture kwargs (passed from main process)
        shared_model_state: Model weights in shared memory
    
    Weight Sharing (no queues!):
    - shared_model_state contains tensors in shared memory
    - Worker creates local model and loads weights from shared state
    - Main process can update shared_model_state in-place after PPO update
    - Workers read latest weights on next episode
    """
    from rl.env import TransitEnv
    from rl.models import GATV2ActorCritic
    
    # Set different seed for each worker/call
    worker_seed = config["seed"] + worker_id * 1000 + np.random.randint(0, 10000)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    
    device = torch.device("cpu")  # Workers use CPU for inference
    
    # Create environment
    worker_config = dict(config)
    worker_config["seed"] = worker_seed
    env = TransitEnv(worker_config)
    
    # Create local model using policy_kwargs from main process
    model = GATV2ActorCritic(**policy_kwargs).to(device)
    
    # Load weights from shared memory
    model.load_state_dict(shared_model_state)
    # Keep model in train mode during rollouts - dropout should remain active
    # (model.eval() would disable dropout which is not desired for training)
    # We still use torch.no_grad() for efficiency
    
    # Cache for static PyG components
    cached_edge_index = None
    cached_edge_attr = None
    
    def convert_state_to_pyg(state: Dict[str, Any]) -> Data:
        nonlocal cached_edge_index, cached_edge_attr
        
        if cached_edge_index is None:
            cached_edge_index = torch.from_numpy(state["edge_index"]).long().to(device)
            cached_edge_attr = torch.from_numpy(state["edge_features"]).float().to(device)
        
        x = torch.from_numpy(state["node_features"]).float().to(device)
        route_progress = torch.from_numpy(state["route_progress"]).float().to(device)
        
        data = Data(x=x, edge_index=cached_edge_index, edge_attr=cached_edge_attr)
        data.route_progress = route_progress
        return data
    
    # Run episode
    state, _ = env.reset()
    terminated = False
    episode_reward = 0.0
    transitions = []
    sim_result = None
    
    while not terminated:
        data = convert_state_to_pyg(state)
        batch = Batch.from_data_list([data]).to(device)
        
        # Build valid mask
        valid_list = env._get_valid_indices()
        num_nodes = batch.num_nodes
        valid_mask = torch.zeros(1, num_nodes, dtype=torch.bool, device=device)
        
        if len(valid_list) > 0:
            for local_idx in valid_list:
                valid_mask[0, local_idx] = True
            
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    valid_mask=valid_mask, stochastic=True
                )
        else:
            # No valid actions
            with torch.no_grad():
                z = model._get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
                g = model.critic_readout(z, batch.batch)
                value_tensor = model.critic_head(g).squeeze(-1)
            
            action_tensor = torch.tensor([env.NO_VALID_ACTION], dtype=torch.long, device=device)
            log_prob_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)
        
        action = action_tensor.cpu().item()
        next_state, reward, terminated, _, sim_result = env.step(action)
        episode_reward += reward
        
        # Store transition
        transition = Transition(
            obs_x=state["node_features"].copy(),
            obs_edge_index=state["edge_index"].copy(),
            obs_edge_attr=state["edge_features"].copy(),
            obs_route_progress=state["route_progress"].copy(),
            action=action,
            raw_reward=reward,
            value=value_tensor.cpu().item(),
            log_prob=log_prob_tensor.cpu().item(),
            terminated=terminated,
            valid_mask=valid_mask.cpu().numpy(),
        )
        transitions.append(transition)
        state = next_state
    
    # Compute bootstrap value
    if terminated:
        bootstrap_value = 0.0
    else:
        data = convert_state_to_pyg(state)
        batch = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            bootstrap_value = model.get_bootstrap_value(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
    
    return EpisodeResult(
        worker_id=worker_id,
        transitions=transitions,
        bootstrap_value=bootstrap_value,
        episode_reward=episode_reward,
        episode_length=len(transitions),
        final_sim_result=sim_result if sim_result else {},
    )


def run_single_eval(run_id: int, seed: int, config: Dict[str, Any], policy_kwargs: Dict[str, Any], shared_model_state: Dict[str, torch.Tensor]) -> EvalResult:
    """
    Run a single evaluation episode in a worker process.
    
    Key differences from training rollouts:
    - Uses model.eval() to disable dropout (deterministic behavior)
    - Uses stochastic=False for deterministic action selection
    - Uses exact seed provided (not randomized)
    - Returns metrics and routes (no transitions)
    """
    from rl.env import TransitEnv
    from rl.models import GATV2ActorCritic
    import random
    
    # Set the exact seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cpu")
    
    # Create environment with the specific seed
    eval_config = dict(config)
    eval_config["seed"] = seed
    env = TransitEnv(eval_config)
    
    # Create local model
    model = GATV2ActorCritic(**policy_kwargs).to(device)
    model.load_state_dict(shared_model_state)
    # Use eval mode for evaluation - disable dropout for deterministic behavior
    model.eval()
    
    # Cache for static PyG components
    cached_edge_index = None
    cached_edge_attr = None
    
    def convert_state_to_pyg(state: Dict[str, Any]) -> Data:
        nonlocal cached_edge_index, cached_edge_attr
        if cached_edge_index is None:
            cached_edge_index = torch.from_numpy(state["edge_index"]).long().to(device)
            cached_edge_attr = torch.from_numpy(state["edge_features"]).float().to(device)
        x = torch.from_numpy(state["node_features"]).float().to(device)
        route_progress = torch.from_numpy(state["route_progress"]).float().to(device)
        data = Data(x=x, edge_index=cached_edge_index, edge_attr=cached_edge_attr)
        data.route_progress = route_progress
        return data
    
    # Run evaluation episode
    state, _ = env.reset()
    terminated = False
    episode_reward = 0.0
    episode_steps = 0
    sim_result = None
    
    while not terminated:
        data = convert_state_to_pyg(state)
        batch = Batch.from_data_list([data]).to(device)
        
        valid_list = env._get_valid_indices()
        num_nodes = batch.num_nodes
        valid_mask = torch.zeros(1, num_nodes, dtype=torch.bool, device=device)
        
        if len(valid_list) > 0:
            for local_idx in valid_list:
                valid_mask[0, local_idx] = True
            
            with torch.no_grad():
                # DETERMINISTIC action for evaluation
                action_tensor, _, _ = model.act(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    valid_mask=valid_mask, stochastic=False
                )
        else:
            action_tensor = torch.tensor([env.NO_VALID_ACTION], dtype=torch.long, device=device)
        
        action = action_tensor.cpu().item()
        next_state, reward, terminated, _, sim_result = env.step(action)
        episode_reward += reward
        episode_steps += 1
        state = next_state
    
    # Calculate metrics
    served = sim_result['completed_passengers'] + sim_result['ongoing_passengers']
    wait_seconds = sim_result['total_wait_completed'] + sim_result['total_wait_ongoing']
    travel_seconds = sim_result['total_travel_completed'] + sim_result['total_travel_ongoing']
    combined_avg_wait_minutes = (wait_seconds / served) / 60 if served > 0 else 0.0
    combined_avg_travel_minutes = (travel_seconds / served) / 60 if served > 0 else 0.0
    
    metrics = {
        'episode_total_reward': float(episode_reward),
        'episode_length': episode_steps,
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
    }
    
    return EvalResult(
        run_id=run_id,
        seed=seed,
        episode_reward=episode_reward,
        episode_length=episode_steps,
        metrics=metrics,
        routes=env.all_routes,
    )


class ParallelEnvManager:
    """
    Manages parallel episode collection using multiprocessing with shared memory.
    
    Weight Sharing Architecture (no queues!):
    - Main process creates shared memory tensors for model weights
    - Workers load weights from shared memory at start of each episode
    - After PPO update, main writes new weights to shared memory
    - Workers automatically get new weights on next episode
    
    This is cleaner than queue-based weight passing because:
    - No weight serialization/deserialization for updates
    - Single source of truth for model weights
    - Workers just read from shared memory
    """
    
    def __init__(self, config: Dict[str, Any], num_workers: int) -> None:
        self.config = config
        self.num_workers = num_workers
        
        # Policy kwargs (set in start())
        self.policy_kwargs: Optional[Dict[str, Any]] = None
        
        # Shared model state (populated in start())
        self.shared_model_state: Optional[Dict[str, torch.Tensor]] = None
        
        # Process pool
        self.pool: Optional[mp_ctx.Pool] = None
    
    def start(self, model, device: torch.device, policy_kwargs: Dict[str, Any]) -> None:
        """
        Initialize shared memory for model weights and start process pool.
        
        Args:
            model: The trained model (weights will be copied to shared memory)
            device: Device the model is on (unused, workers use CPU)
            policy_kwargs: Model architecture kwargs (passed to workers)
        """
        print(f"Setting up shared memory for model weights...")
        
        # Store policy_kwargs to pass to workers
        self.policy_kwargs = policy_kwargs
        
        # Create shared memory copy of model state
        self.shared_model_state = {}
        for name, param in model.state_dict().items():
            shared_tensor = param.cpu().clone()
            shared_tensor.share_memory_()
            self.shared_model_state[name] = shared_tensor
        
        print(f"Creating process pool with {self.num_workers} workers...")
        self.pool = mp_ctx.Pool(processes=self.num_workers)
        
        print(f"ParallelEnvManager ready (using shared memory for weights)")
    
    def update_shared_weights(self, model) -> None:
        """
        Update shared memory weights from the trained model.
        
        Called after PPO update to push new weights to shared memory.
        Workers will automatically use these weights on their next episode.
        """
        if self.shared_model_state is None:
            raise RuntimeError("Manager not started. Call start() first.")
        
        with torch.no_grad():
            for name, param in model.state_dict().items():
                self.shared_model_state[name].copy_(param.cpu())
    
    def collect_episodes(self, num_episodes: int) -> List[EpisodeResult]:
        """
        Collect a batch of episodes in parallel.
        
        Each worker runs one episode using the current shared weights.
        Returns list of complete EpisodeResult objects.
        """
        if self.pool is None:
            raise RuntimeError("Manager not started. Call start() first.")
        
        # Submit episode tasks to process pool
        tasks = []
        for i in range(num_episodes):
            worker_id = i % self.num_workers
            task = self.pool.apply_async(
                run_single_episode,
                args=(worker_id, self.config, self.policy_kwargs, self.shared_model_state),
            )
            tasks.append(task)
        
        # Collect results
        results = []
        for i, task in enumerate(tasks):
            try:
                result = task.get(timeout=300.0)
                results.append(result)
                print(f"Episode {i+1}/{num_episodes} done: worker={result.worker_id}, reward={result.episode_reward:.2f}, length={result.episode_length}")
            except Exception as e:
                print(f"Episode {i+1} failed with error: {e}")
        
        return results
    
    def run_parallel_eval(self, num_runs: int, base_seed: int, seed_offset: int) -> List[EvalResult]:
        """
        Run multiple evaluation episodes in parallel.
        
        Each worker runs one evaluation with a specific seed.
        All workers share the same policy weights.
        
        Args:
            num_runs: Number of evaluation runs
            base_seed: Starting seed
            seed_offset: Offset between consecutive seeds
            
        Returns:
            List of EvalResult objects (one per run)
        """
        if self.pool is None:
            raise RuntimeError("Manager not started. Call start() first.")
        
        print(f"Running {num_runs} parallel evaluation runs...")
        
        # Submit eval tasks to process pool
        tasks = []
        for run_id in range(num_runs):
            seed = base_seed + (run_id * seed_offset)
            task = self.pool.apply_async(
                run_single_eval,
                args=(run_id, seed, self.config, self.policy_kwargs, self.shared_model_state),
            )
            tasks.append((run_id, seed, task))
        
        # Collect results
        results = []
        for run_id, seed, task in tasks:
            try:
                result = task.get(timeout=300.0)
                results.append(result)
                print(f"Eval run {run_id+1}/{num_runs} (seed={seed}) done: reward={result.episode_reward:.2f}")
            except Exception as e:
                print(f"Eval run {run_id+1} (seed={seed}) failed with error: {e}")
        
        return results
    
    def stop(self) -> None:
        """Shut down the process pool."""
        print("Stopping process pool...")
        
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        
        self.shared_model_state = None
        self.policy_kwargs = None
        print("Process pool stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

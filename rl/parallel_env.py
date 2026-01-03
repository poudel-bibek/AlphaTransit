
"""
Parallel environment manager for distributed PPO training.
Uses Python multiprocessing with shared memory to distribute policy weights efficiently.
"""

import os
import random
import torch
import numpy as np
from rl.env import TransitEnv
from rl.models import GATV2ActorCritic
from dataclasses import dataclass, field
from torch_geometric.data import Data, Batch
from queue import Empty as QueueEmpty

import torch.multiprocessing as mp
mp_ctx = mp.get_context('spawn')

def _cap_worker_threads():
    """
    Limit BLAS/OpenMP and PyTorch intra/inter-op threads to 1 per process.
    Helps prevent massive CPU thread over-subscription when using many workers.
    """
    # Common BLAS/OpenMP env vars
    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[var] = os.environ.get(var, "1") or "1"

    # PyTorch thread limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

@dataclass
class Transition:
    """
    A single environment transition for PPO.
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
class RolloutChunk:
    """
    A chunk of transitions from a worker.
    episode_reward/episode_length are 0 for partial chunks, only populated on terminal.
    advantages/returns are computed per-chunk inside the worker (GAE with bootstrap).
    reward_scale_used: the scale used to normalize rewards for this chunk's GAE computation.
                       Learner uses this to recover raw returns: raw_return = return * scale.
    """
    worker_id: int
    transitions: list = field(default_factory=list)
    is_terminal: bool = False
    episode_reward: float = 0.0
    episode_length: int = 0
    final_sim_result: dict = field(default_factory=dict)
    advantages: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    reward_scale_used: float = 1.0

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
    metrics: dict = field(default_factory=dict)
    routes: list = field(default_factory=list)

class PyGConverter:
    """
    Converts environment state dicts to PyG Data objects.
    Caches static components (edge_index, edge_attr) for efficiency within an episode.
    """
    def __init__(self, device):
        self.device = device
        self.cached_edge_index = None
        self.cached_edge_attr = None
    
    def convert(self, state):
        if self.cached_edge_index is None:
            self.cached_edge_index = torch.from_numpy(state["edge_index"]).long().to(self.device)
            self.cached_edge_attr = torch.from_numpy(state["edge_features"]).float().to(self.device)
        
        x = torch.from_numpy(state["node_features"]).float().to(self.device)
        route_progress = torch.from_numpy(state["route_progress"]).float().to(self.device)
        
        data = Data(x=x, edge_index=self.cached_edge_index, edge_attr=self.cached_edge_attr)
        data.route_progress = route_progress
        return data

def _compute_gae_chunk(rewards, values, dones, gamma, gae_lambda, bootstrap_value):
    """
    Compute GAE advantages/returns for a single contiguous rollout chunk.
    The chunk end is treated as a truncation unless the episode terminated.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        last_advantage = advantages[t]

    returns = advantages + values
    print(f"[DEBUG] GAE chunk: len={len(rewards)}, bootstrap={bootstrap_value:.4f}, adv_range=[{advantages.min():.3f}, {advantages.max():.3f}], ret_range=[{returns.min():.2f}, {returns.max():.2f}], reward_sum={rewards.sum():.2f}")
    return advantages.tolist(), returns.tolist()

def _bootstrap_value(model, converter, state, device):
    """
    Get V(s) for a single state using the critic head.
    """
    data = converter.convert(state)
    batch = Batch.from_data_list([data]).to(device)
    return model.get_bootstrap_value(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

def run_single_eval(env, model, seed, device):
    """
    Internal helper to run evaluation on a given env/model instance.
    Used by rollout_worker_loop to reuse resources.
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model.eval()
    converter = PyGConverter(device)
    
    # Run a single evaluation episode
    state, _ = env.reset(seed=seed)
    terminated = False
    episode_reward = 0.0
    episode_steps = 0
    sim_result = None
    
    while not terminated:
        data = converter.convert(state)
        batch = Batch.from_data_list([data]).to(device)
        
        valid_list = env._get_valid_indices()
        num_nodes = batch.num_nodes
        valid_mask = torch.zeros(1, num_nodes, dtype=torch.bool, device=device)
        
        if len(valid_list) > 0:
            valid_mask[0, valid_list] = True
            
            with torch.no_grad():
                action_tensor, _, _ = model.act( batch.x, 
                                                 batch.edge_index, 
                                                 batch.edge_attr, 
                                                 batch.batch,
                                                 valid_mask=valid_mask, 
                                                 stochastic=False) # DETERMINISTIC action for evaluation
        else:
            # No valid next node found.
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
        run_id = 0, # We must return run_id as 0 here, caller will adjust
        seed = seed,
        episode_reward = episode_reward,
        episode_length = episode_steps,
        metrics = metrics,
        routes = env.all_routes,
    )

def worker(worker_id, config, policy_kwargs, shared_model_state, shared_update_counter, shared_reward_scale, cmd_queue, shared_res_queue, eval_res_queue):
    """
    Asynchronous streaming worker that sends data in chunks during episode execution.
    
    Commands (via cmd_queue):
        {"type": "stop"}      - Terminate worker
        {"type": "eval"}      - Run one eval episode (deterministic)
        {"type": "collect"}   - Run training, streaming chunks of steps_per_worker
    
    Notes:
        - Runs on CPU to avoid CUDA + multiprocessing issues
        - Streaming: sends chunks to shared_res_queue when buffer reaches steps_per_worker OR episode ends
        - Eval results go to eval_res_queue (per-worker) for ordered collection
        - Rewards are scaled by shared_reward_scale before GAE computation for consistent units
    """
    _cap_worker_threads()
    device = torch.device("cpu")
    buffer_size = config.get("steps_per_worker")
    gamma = float(config.get("gamma"))
    gae_lambda = float(config.get("gae_lambda"))
    weight_refresh_interval = config.get("weight_refresh_interval")
    
    # Distinct seeding per worker. Without explicit seeding, workers may lead to correlated experiences.
    base_seed = int(config["seed"]) + (worker_id + 1) * 100003
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    # Single env for both training and eval
    env = TransitEnv(dict(config))

    # Although each worker gets their own policy model, the weights come from a shared memory.
    model = GATV2ActorCritic(**policy_kwargs).to(device)
    converter = PyGConverter(device)

    while True:
        cmd = cmd_queue.get()
        cmd_type = cmd.get("type") if isinstance(cmd, dict) else None

        if cmd_type == "stop":
            break

        elif cmd_type == "eval":
            model.load_state_dict(shared_model_state)
            model.eval()
            result = run_single_eval(env, model, cmd["seed"], device)
            result.run_id = cmd["run_id"]
            eval_res_queue.put([result])

        elif cmd_type == "collect":
            model.load_state_dict(shared_model_state)
            model.eval()
            last_reload_update = shared_update_counter.value  # Track when we last reloaded

            # Minimum clamp for reward_scale to prevent blow-up when std is very small
            # (e.g., early training or near-constant rewards). Without this, division
            # by near-zero scales produces huge advantages that destabilize learning.
            reward_scale = max(shared_reward_scale.value, 1e-4)  
            
            print(f"[DEBUG] Worker {worker_id}: Loaded weights at update_counter={last_reload_update}, reward_scale={reward_scale:.4f}, starting new episode")
            state, _ = env.reset()
            terminated = False
            transitions = []
            episode_reward = 0.0
            episode_steps = 0

            while not terminated:
                data = converter.convert(state)
                batch = Batch.from_data_list([data]).to(device)
                valid_list = env._get_valid_indices()
                num_nodes = batch.num_nodes
                valid_mask = torch.zeros(1, num_nodes, dtype=torch.bool, device=device)

                if len(valid_list) > 0:
                    valid_mask[0, valid_list] = True
                    with torch.no_grad():
                        action_tensor, log_prob_tensor, value_tensor = model.act(batch.x, 
                                                                                batch.edge_index, 
                                                                                batch.edge_attr, 
                                                                                batch.batch, 
                                                                                valid_mask=valid_mask, 
                                                                                stochastic=True)
                else:
                    # No valid next node found.
                    with torch.no_grad():
                        z = model._get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
                        g = model.critic_readout(z, batch.batch)
                        value_tensor = model.critic_head(g).squeeze(-1)

                    action_tensor = torch.tensor([env.NO_VALID_ACTION], dtype=torch.long, device=device)
                    log_prob_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)

                action = action_tensor.cpu().item()
                next_state, reward, terminated, _, sim_result = env.step(action)
                episode_reward += reward

                transition = Transition(
                    obs_x = state["node_features"].copy(),
                    obs_edge_index = state["edge_index"].copy(),
                    obs_edge_attr = state["edge_features"].copy(),
                    obs_route_progress = state["route_progress"].copy(),
                    action = action,
                    raw_reward = reward,
                    value = value_tensor.cpu().item(),
                    log_prob = log_prob_tensor.cpu().item(),
                    terminated = terminated,
                    valid_mask = valid_mask.cpu().numpy())
                    
                transitions.append(transition)
                state = next_state
                episode_steps += 1

                # Stream chunk when buffer is full (mid-episode)
                if len(transitions) >= buffer_size and not terminated:
                    with torch.no_grad():
                        # Bootstrap at chunk boundary to handle rollout truncation.
                        bootstrap_value = _bootstrap_value(model, converter, state, device)

                    # Scale rewards before GAE so advantages/returns are in same units as critic values
                    scaled_rewards = [t.raw_reward / reward_scale for t in transitions]
                    advantages, returns = _compute_gae_chunk( rewards=scaled_rewards,
                                                              values=[t.value for t in transitions],
                                                              dones=[t.terminated for t in transitions],
                                                              gamma=gamma,
                                                              gae_lambda=gae_lambda,
                                                              bootstrap_value=bootstrap_value)

                    chunk = RolloutChunk( worker_id = worker_id,
                                          transitions = list(transitions),
                                          is_terminal = False,
                                          episode_reward = 0.0,   # Partial chunk - not meaningful
                                          episode_length = 0,     # Partial chunk - not meaningful
                                          final_sim_result = {},
                                          advantages = advantages,
                                          returns = returns,
                                          reward_scale_used = reward_scale)

                    print(f"[DEBUG] Worker {worker_id}: Sending MID-EPISODE chunk with {len(chunk.transitions)} transitions, bootstrap_value={bootstrap_value:.4f}, reward_scale={reward_scale:.4f}")
                    shared_res_queue.put(chunk)
                    transitions.clear()
                    
                    # Policy Lag Mitigation: Workers load weights at episode start, but long episodes
                    # (e.g., 224 steps) can span many PPO updates (~35 with update_freq=64). This causes
                    # workers to collect experience with stale policies, violating PPO's on-policy assumption.
                    # Solution: Refresh weights and reward_scale mid-episode every N updates to keep policy lag bounded.
                    current_update = shared_update_counter.value
                    if current_update - last_reload_update >= weight_refresh_interval:
                        reward_scale = max(shared_reward_scale.value, 1e-4)  # Clamp to prevent blow-up
                        print(f"[DEBUG] Worker {worker_id}: Mid-episode weight refresh (update {last_reload_update} -> {current_update}), reward_scale={reward_scale:.4f}")
                        model.load_state_dict(shared_model_state)
                        model.eval()
                        last_reload_update = current_update

            # Episode done - send remaining transitions with episode totals
            if transitions:
                # Terminal chunk: bootstrap is zero by definition.
                # Scale rewards before GAE so advantages/returns are in same units as critic values
                scaled_rewards = [t.raw_reward / reward_scale for t in transitions]
                advantages, returns = _compute_gae_chunk( rewards=scaled_rewards,
                                                          values=[t.value for t in transitions],
                                                          dones=[t.terminated for t in transitions],
                                                          gamma=gamma,
                                                          gae_lambda=gae_lambda,
                                                          bootstrap_value=0.0)

                chunk = RolloutChunk( worker_id = worker_id,
                                      transitions = list(transitions),
                                      is_terminal = True,
                                      episode_reward = episode_reward,
                                      episode_length = episode_steps,
                                      final_sim_result = sim_result if sim_result else {},
                                      advantages = advantages,
                                      returns = returns,
                                      reward_scale_used = reward_scale)

                print(f"[DEBUG] Worker {worker_id}: Sending TERMINAL chunk with {len(chunk.transitions)} transitions, episode_reward={episode_reward:.2f}, episode_length={episode_steps}, reward_scale={reward_scale:.4f}")
                shared_res_queue.put(chunk)

        else:
            # Unknown command type
            # print(f"Unknown command type: {cmd_type}")
            pass

class ParallelEnvManager:
    """
    Manages parallel workers.
    Weight distribution model:
    - The learner creates shared memory tensors for model weights
    - Workers load weights from shared memory at the start of each episode
    - Workers stream RolloutChunks continuously to a shared queue
    - After PPO updates, the learner copies new weights to shared memory
    - Workers naturally pick up new weights on their next episode start
    """
    
    def __init__(self, config, num_workers):
        self.config = config
        self.num_workers = num_workers

        # These are set in start()       
        self.shared_model_state = None
        self.shared_update_counter = None  # Tracks PPO updates for mid-episode weight refresh
        self.shared_reward_scale = None    # Shared reward scale for worker-side normalization

        # Queues
        self._actor_cmd_queues = []
        self._actor_eval_res_queues = []  # Per-worker for ordered eval results
        self._shared_res_queue = None      # Shared for training rollouts
        self._actor_procs = []
        
        # Track active workers (those that have been sent "collect" but not yet finished episode)
        self._active_workers = set()
    
    def start(self, model, policy_kwargs):
        """
        Initialize shared memory for model weights and start persistent rollout workers.
        - model: The trained model (weights will be copied to shared memory)
        - policy_kwargs: Model architecture kwargs (passed to workers)
        """

        # print(f"Setting up shared memory for model weights...")
        
        # Create shared memory copy of model state
        self.shared_model_state = {}
        for name, param in model.state_dict().items():
            shared_tensor = param.cpu().clone()
            shared_tensor.share_memory_()
            self.shared_model_state[name] = shared_tensor
        
        # Shared counter for tracking PPO updates (workers reload weights every weight_refresh_interval updates)
        self.shared_update_counter = mp_ctx.Value('i', 0)
        
        # Shared reward scale for worker-side normalization (initialized to 1.0, updated by learner)
        self.shared_reward_scale = mp_ctx.Value('d', 1.0)

        # Create shared result queue for training rollouts (unbounded).
        # Pontentially this queue could have been made bounded to control policy staleness. 
        # But another mechanism is used to control staleness - weight_refresh_interval - workers reload weights every N updates.
        self._shared_res_queue = mp_ctx.Queue()

        # Start rollout workers 
        for wid in range(self.num_workers):
            cmd_q = mp_ctx.Queue()
            eval_res_q = mp_ctx.Queue()  # Per-worker queue for eval results only
            proc = mp_ctx.Process(
                target=worker,
                args=( wid, 
                       self.config, 
                       policy_kwargs, 
                       self.shared_model_state,
                       self.shared_update_counter,
                       self.shared_reward_scale,
                       cmd_q, 
                       self._shared_res_queue,  # Shared for training
                       eval_res_q,              # Per-worker for eval
                ),
            )
            proc.start()
            self._actor_cmd_queues.append(cmd_q)
            self._actor_eval_res_queues.append(eval_res_q)
            self._actor_procs.append(proc)
    
    def update_shared_weights(self, model):
        """
        Update shared memory weights from the trained model.
        Called after PPO update to push new weights to shared memory.
        """
        if self.shared_model_state is None:
            raise RuntimeError("Manager not started. Call start() first.")
        
        with torch.no_grad():
            for name, param in model.state_dict().items():
                self.shared_model_state[name].copy_(param.cpu())
        
        # Increment update counter so workers know to refresh weights
        with self.shared_update_counter.get_lock():
            self.shared_update_counter.value += 1
            print(f"[DEBUG] Learner: Pushed updated weights to shared memory, update_counter now = {self.shared_update_counter.value}")
    
    def update_reward_scale(self, new_scale: float):
        """
        Update shared reward scale for worker-side normalization.
        Called after updating return_rms to push new scale to workers.
        """
        if self.shared_reward_scale is None:
            raise RuntimeError("Manager not started. Call start() first.")
        
        with self.shared_reward_scale.get_lock():
            old_scale = self.shared_reward_scale.value
            self.shared_reward_scale.value = new_scale
            print(f"[DEBUG] Learner: Updated shared reward_scale {old_scale:.4f} -> {new_scale:.4f}")
    
    def run_parallel_eval(self, num_runs, base_seed, seed_offset, policy_path):
        """
        Run multiple evaluation episodes in parallel.
        
        Each worker runs one evaluation with a specific seed.
        All workers share the same policy weights.
        
        Args:
            num_runs: Number of evaluation runs
            base_seed: Starting seed
            seed_offset: Offset between consecutive seeds
            policy_path: Optional path to policy weights to load before eval
            
        Returns:
            List of EvalResult objects (one per run)
        """

        # load weights from path
        state_dict = torch.load(policy_path, map_location="cpu")
        for name, param in state_dict.items():
            self.shared_model_state[name].copy_(param)
        
        # print(f"Running {num_runs} eval runs with {self.num_workers} workers...")
        
        results = []
        next_run_id = 0
        
        while next_run_id < num_runs:
            # Dispatch to available workers
            active_workers = 0
            for i in range(min(self.num_workers, num_runs - next_run_id)):
                seed = base_seed + (next_run_id * seed_offset)
                self._actor_cmd_queues[i].put({
                    "type": "eval", 
                    "run_id": next_run_id, 
                    "seed": seed
                })
                active_workers += 1
                next_run_id += 1
            
            # Collect results from per-worker eval queues
            for i in range(active_workers):
                worker_res_list = self._actor_eval_res_queues[i].get(timeout=None)
                result = worker_res_list[0]
                results.append(result)
                # print(f"Eval {result.run_id+1}/{num_runs} (seed={result.seed}): reward={result.episode_reward:.2f}")
                    
        return results
    
    def start_collection(self):
        """
        Send collect command to all workers to start/continue episode collection.
        Workers will run one episode each and stream chunks to the shared queue.
        """
        for wid in range(self.num_workers):
            self._actor_cmd_queues[wid].put({"type": "collect"})
            self._active_workers.add(wid)
    
    def collect_rollouts(self, target_transitions):
        """
        Collect rollout chunks from the shared queue until target_transitions reached.
        
        Workers stream chunks continuously. This method drains the queue until we have
        enough transitions for a PPO update. Workers that finish their episodes are
        automatically restarted.
        
        Args:
            target_transitions: Number of transitions to collect before returning
            
        Returns:
            List of RolloutChunk objects, total transitions >= target_transitions
        """
        chunks = []
        total_steps = 0
        
        while total_steps < target_transitions:
            # Block until we get a chunk
            chunk = self._shared_res_queue.get(timeout=None)
            
            steps = len(chunk.transitions)
            total_steps += steps
            chunks.append(chunk)
            
            # If this worker finished its episode, restart it
            if chunk.is_terminal:
                self._actor_cmd_queues[chunk.worker_id].put({"type": "collect"})
        
        print(f"[DEBUG] Learner: Collected {total_steps} transitions from {len(chunks)} chunks (target was {target_transitions})")
        return chunks
    
    def stop(self):
        """
        Shut down workers and clean up resources.
        """
        # print("Stopping workers...")
        for q in self._actor_cmd_queues:
            q.put({"type": "stop"})
        for p in self._actor_procs:
            p.join(timeout=5.0)
        
        self._actor_cmd_queues.clear()
        self._actor_eval_res_queues.clear()
        self._actor_procs.clear()
        self._active_workers.clear()
        self._shared_res_queue = None
        self.shared_model_state = None
        # print("Workers stopped.")

from typing import Any, Dict, List
import torch
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset


class RunningMeanStd:
    """
    Welford's online algorithm for computing running mean and variance.
    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        """Update running stats using batch moments (parallel-friendly Welford)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> float:
        """Return running standard deviation."""
        return np.sqrt(self.var + 1e-8)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize by dividing by running std (preserves sign, only scales magnitude)."""
        return x / self.std


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate samples into mini-batch for PPO
    produces: 
    - Custom atributes like route_progress shaped as [B, Num_routes]
    - actions, log_probs, advantages, returns, values shaped as [B]
    - valid_mask shaped as [B, Max_nodes] boolean
    """

    # 1) Batch the observations
    batch_obs_list = [item['obs'] for item in batch]
    batched_obs = Batch.from_data_list(batch_obs_list)

    # 2) Manually stack the custom attributes
    route_progress_list = [obs.route_progress for obs in batch_obs_list] # list of 1D tensors
    batched_obs.route_progress = torch.stack(route_progress_list, dim=0) # 2D tensor [B, Num_routes]

    # 3) Properly stack valid masks into [B, num_nodes]
    # Each item[valid mask] is a boolean tensor [1, num_nodes]
    valid_masks = []
    for item in batch:
        m = item['valid_mask'].squeeze(0)
        valid_masks.append(m)
    batched_valid_mask = torch.stack(valid_masks, dim=0) # 2D tensor [B, num_nodes]
    
    # 4) Stack the other tensors
    actions = torch.tensor([item['actions'] for item in batch], dtype=torch.long) # [B]
    log_probs = torch.tensor([item['log_probs'] for item in batch], dtype=torch.float32) # [B]
    advantages = torch.tensor([item['advantages'] for item in batch], dtype=torch.float32) # [B]
    returns = torch.tensor([item['returns'] for item in batch], dtype=torch.float32) # [B]
    values = torch.tensor([item['values'] for item in batch], dtype=torch.float32) # [B]
    
    # print(f"\nActions: shape: {actions.shape}, value: {actions}, type: {type(actions)}")
    # print(f"\nLog probs: shape: {log_probs.shape}, value: {log_probs}, type: {type(log_probs)}")
    # print(f"\nAdvantages: shape: {advantages.shape}, value: {advantages}, type: {type(advantages)}")
    # print(f"\nReturns: shape: {returns.shape}, value: {returns}, type: {type(returns)}")
    # print(f"\nValues: shape: {values.shape}, value: {values}, type: {type(values)}")
    # print(f"\nValid mask: shape: {batched_valid_mask.shape}, value: {batched_valid_mask}, type: {type(batched_valid_mask)}")

    print(f"[DEBUG] collate_fn: batched_obs.x={tuple(batched_obs.x.shape)}, actions={tuple(actions.shape)}, log_probs={tuple(log_probs.shape)}, adv_range=[{advantages.min():.3f}, {advantages.max():.3f}], ret_range=[{returns.min():.2f}, {returns.max():.2f}]")

    return {
        'obs': batched_obs,
        'actions': actions,
        'log_probs': log_probs,
        'advantages': advantages,
        'returns': returns,
        'values': values,
        'valid_mask': batched_valid_mask,
    }


class Memory:
    def __init__(self) -> None:
        """
        Initialize empty rollout buffers.
        
        Advantages/returns are precomputed per-chunk in workers (worker-side GAE)
        and normalized in the learner before being stored here.
        """
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.raw_rewards = []
        self.values = []
        self.dones = []  # True for terminated episodes only
        self.valid_mask = []
        self.advantages = [] # Precomputed by workers (worker-side GAE)
        self.returns = []

    def __len__(self) -> int:
        return len(self.obs)

    def store(self, transition: Dict[str, Any]) -> None:
        """
        Add transition to memory.
        """
        self.obs.append(transition['obs'])
        self.actions.append(transition['action'])
        self.raw_rewards.append(transition['raw_reward'])
        self.values.append(transition['value'])
        self.log_probs.append(transition['log_prob'])
        self.dones.append(transition['terminated'])  # Only terminated!
        self.valid_mask.append(transition['valid_mask'])

    def clear(self) -> None:
        """
        Reset all buffers after PPO update.
        """
        self.obs.clear()
        self.actions.clear()
        self.raw_rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.valid_mask.clear()
        self.advantages = []
        self.returns = []

class DatasetClass(Dataset):
    """
    Dataset wrapper for PPO experience replay.
    Handles rollout data with computed advantages/returns.
    """
    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        
    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        return {
            'obs': self.memory.obs[idx],
            'actions': self.memory.actions[idx],
            'log_probs': self.memory.log_probs[idx],
            'advantages': self.memory.advantages[idx],
            'returns': self.memory.returns[idx],
            'values': self.memory.values[idx],
            'valid_mask': self.memory.valid_mask[idx]
        }

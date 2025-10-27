from typing import Any, Dict, List
import torch
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset


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
        m = item['valid_mask']
        valid_masks.append(m)
    batched_valid_mask = torch.stack(valid_masks, dim=0) # 2D tensor [B, num_nodes]

    # 4) Stack the other tensors
    actions = torch.tensor([item['actions'] for item in batch], dtype=torch.long) # [B]
    log_probs = torch.tensor([item['log_probs'] for item in batch], dtype=torch.float32) # [B]
    advantages = torch.tensor([item['advantages'] for item in batch], dtype=torch.float32) # [B]
    returns = torch.tensor([item['returns'] for item in batch], dtype=torch.float32) # [B]
    values = torch.tensor([item['values'] for item in batch], dtype=torch.float32) # [B]

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
        """
        self.obs: List[Any] = []
        self.actions: List[Any] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []  # True for terminated episodes only
        self.valid_mask: List[torch.BoolTensor] = []
        
        # Bootstrap values for each episode (if truncated)
        self.bootstrap_values: List[float] = []  # One per episode
        self.episode_boundaries: List[int] = []  # Indices where episodes end
        self.advantages = None # Will be added during GAE
        self.returns = None

    def __len__(self) -> int:
        return len(self.obs)

    def store(self, transition: Dict[str, Any]) -> None:
        """
        Add transition to memory.
        """
        self.obs.append(transition['obs'])
        self.actions.append(transition['action'])
        self.rewards.append(transition['reward'])
        self.values.append(transition['value'])
        self.log_probs.append(transition['log_prob'])
        self.dones.append(transition['terminated'])  # Only terminated!
        self.valid_mask.append(transition['valid_mask'])
        
    def mark_episode_end(self, bootstrap_value: float) -> None:
        """
        Mark the end of an episode and store bootstrap value if needed.
        """
        self.episode_boundaries.append(len(self.obs) - 1) # 0-indexed
        self.bootstrap_values.append(bootstrap_value)

    def clear(self) -> None:
        """
        Reset all buffers after PPO update.
        """
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.valid_mask.clear()
        self.bootstrap_values.clear()
        self.episode_boundaries.clear()
        self.advantages = None # Will be added during GAE
        self.returns = None

class DatasetClass(Dataset):
    """
    Dataset wrapper for PPO experience replay.
    Handles rollout data with computed advantages.
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


class WelfordNormalizer:
    """
    Online normalizer using Welford's algorithm.
    Efficiently tracks running mean/variance.
    """

    def __init__(self, shape: tuple = ()) -> None:
        """
        Initialize normalizer for given shape.
        
        Args:
            shape: Shape of data to normalize (excluding batch)
        """
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with new batch of data.
        
        Args:
            x: Data batch, first dim is batch size
        """
        batch_size = x.shape[0]
        if batch_size == 0:
            return
        
        # Flatten batch for easier computation
        x = x.reshape(batch_size, -1)
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = batch_size
        
        # Welford's online algorithm
        delta = batch_mean - self.mean.flatten()
        total_count = self.count + batch_count
        
        self.mean = self.mean.flatten() + delta * batch_count / total_count
        
        m_a = self.var.flatten() * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
        
        # Reshape back
        self.mean = self.mean.reshape(self.shape)
        self.var = self.var.reshape(self.shape)

    def normalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Normalize data using running statistics.
        
        Args:
            x: Data to normalize
            epsilon: Small value for numerical stability
        
        Returns:
            Normalized data
        """
        return (x - self.mean) / np.sqrt(self.var + epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse normalization.
        
        Args:
            x: Normalized data
        
        Returns:
            Original scale data
        """
        return x * np.sqrt(self.var) + self.mean

    def state_dict(self) -> Dict[str, Any]:
        """Export statistics for checkpointing."""
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load statistics from checkpoint."""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']
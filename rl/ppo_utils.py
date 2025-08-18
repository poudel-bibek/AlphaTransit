from typing import Any, Dict, List, Optional
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate samples into mini-batch for PPO training.
    Handles both graph observations and standard tensors.
    """
    collated = {
        'obs': Batch.from_data_list([item['obs'] for item in batch]),
        'actions': torch.tensor([item['actions'] for item in batch], dtype=torch.long),
        'log_probs': torch.tensor([item['log_probs'] for item in batch], dtype=torch.float32),
        'advantages': torch.tensor([item['advantages'] for item in batch], dtype=torch.float32),
        'returns': torch.tensor([item['returns'] for item in batch], dtype=torch.float32),
        'values': torch.tensor([item['values'] for item in batch], dtype=torch.float32),
        'valid_indices': [item['valid_indices'] for item in batch]
    }
    return collated

class Memory:
    """
    On-policy memory buffer for PPO rollouts.
    Stores full trajectories until update.
    """

    def __init__(self) -> None:
        """Initialize empty rollout buffers."""
        self.obs: List[Any] = []
        self.actions: List[Any] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        
        # Bootstrap value for GAE computation
        self.bootstrap_value: float = 0.0
        
        # Computed during GAE
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self.valid_indices: List[List[int]] = []

    def store(self, transition: Dict[str, Any]) -> None:
        """
        Add transition to memory.
        
        Args:
            transition: Dict with obs, action, reward, value, log_prob, done
        """
        self.obs.append(transition['obs'])
        self.actions.append(transition['action'])
        self.rewards.append(transition['reward'])
        self.values.append(transition['value'])
        self.log_probs.append(transition['log_prob'])
        self.dones.append(transition['done'])
        self.valid_indices.append(transition.get('valid_indices', []))

    def set_bootstrap_value(self, value: float) -> None:
        """
        Set bootstrap value for GAE computation.
        
        Args:
            value: Bootstrap value for final state (0.0 if terminated, critic value if truncated)
        """
        self.bootstrap_value = value

    def clear(self) -> None:
        """Reset all buffers after PPO update."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.bootstrap_value = 0.0
        self.advantages = None
        self.returns = None
        self.valid_indices.clear()

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.obs)


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
            'valid_indices': self.memory.valid_indices[idx]
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
from typing import Any, Dict, Iterable, Sequence, List
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate samples into mini-batch for PPO training.
    Handles both graph observations and standard tensors.
    """
    # Separate graphs from other data
    graphs = []
    other_data = {key: [] for key in batch[0].keys() if key != 'obs'}
    
    for sample in batch:
        graphs.append(sample['obs'])
        for key in other_data:
            other_data[key].append(sample[key])
    
    # Batch graphs if they're PyG Data objects
    if isinstance(graphs[0], Data):
        batched_obs = Batch.from_data_list(graphs)
    else:
        batched_obs = torch.stack(graphs)
    
    # Stack other tensors
    collated = {'obs': batched_obs}
    for key, values in other_data.items():
        if isinstance(values[0], (int, float, np.number, np.ndarray)):
            collated[key] = torch.tensor(values, dtype=torch.float32)
        else:
            collated[key] = torch.stack(values)
    
    return collated


class DatasetClass(Dataset):
    """
    Dataset wrapper for PPO experience replay.
    Handles rollout data with computed advantages.
    """

    def __init__(self, memory: 'Memory') -> None:
        """
        Create dataset from memory buffer.
        
        Args:
            memory: Memory buffer with rollout data
        """
        self.observations = memory.observations
        self.actions = memory.actions
        self.log_probs = memory.log_probs
        self.values = memory.values
        self.advantages = memory.advantages
        self.returns = memory.returns

    def __len__(self) -> int:
        """Return number of transitions."""
        return len(self.observations)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get single transition by index.
        
        Returns:
            Dictionary with obs, action, advantages, etc.
        """
        return {
            'obs': self.observations[index],
            'actions': self.actions[index],
            'log_probs': self.log_probs[index],
            'values': self.values[index],
            'advantages': self.advantages[index],
            'returns': self.returns[index]
        }


class Memory:
    """
    On-policy memory buffer for PPO rollouts.
    Stores full trajectories until update.
    """

    def __init__(self) -> None:
        """Initialize empty rollout buffers."""
        self.observations: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        
        # Computed during GAE
        self.advantages: np.ndarray = None
        self.returns: np.ndarray = None

    def store(self, transition: Dict[str, Any]) -> None:
        """
        Add transition to memory.
        
        Args:
            transition: Dict with obs, action, reward, value, log_prob, done
        """
        self.observations.append(transition['obs'])
        self.actions.append(transition['action'])
        self.rewards.append(transition['reward'])
        self.values.append(transition['value'])
        self.log_probs.append(transition['log_prob'])
        self.dones.append(transition['done'])

    def clear(self) -> None:
        """Reset all buffers after PPO update."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.observations)


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
"""
MCTS Utility Classes for AlphaTransit

This module contains the core data structures for Monte Carlo Tree Search:
- MCTSState: Lightweight state representation for tree exploration
- MCTSNode: Tree node with PUCT statistics
- MCTSTree: Orchestrates search operations
- ReplayBuffer: FIFO experience replay buffer
- WelfordNormalizer: Online mean/variance tracking for reward normalization
"""

import math
import random
import numpy as np
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from rl.env_utils import initialize_route

@dataclass
class MCTSState:
    """
    Lightweight state representation for MCTS tree exploration.
    Stores route-building state with env reference for initialize_route() calls.
    """
    current_route: List[str]
    all_routes: List[List[str]]
    current_route_index: int
    num_routes: int
    max_route_length: int
    adj: Dict[str, set]  # Reference to env's adjacency (shared, not copied)
    node_to_idx: Dict[str, int]  # Reference to env's node mapping (shared)
    idx_to_node: Dict[int, str]  # Reference to env's reverse mapping (shared)
    env: Any  # Reference to TransitEnv for initialize_route() calls

    def clone(self) -> 'MCTSState':
        """
        Create a shallow copy suitable for tree expansion.
        """
        return MCTSState(
            current_route=list(self.current_route),
            all_routes=[list(r) for r in self.all_routes],
            current_route_index=self.current_route_index,
            num_routes=self.num_routes,
            max_route_length=self.max_route_length,
            adj=self.adj,
            node_to_idx=self.node_to_idx,
            idx_to_node=self.idx_to_node,
            env=self.env,
        )

    def get_valid_actions(self) -> List[int]:
        """
        Get valid action indices from current state.
        """
        if not self.current_route:
            return []
        frontier = self.current_route[-1]
        route_set = set(self.current_route)
        valid_neighbors = self.adj.get(frontier, set()) - route_set
        return [self.node_to_idx[n] for n in valid_neighbors]

    def apply_action(self, action: int) -> 'MCTSState':
        """
        Apply action and return new state.
        Handles route extension and route transitions.
        """
        new_state = self.clone()
        action_node = self.idx_to_node[action]
        new_state.current_route.append(action_node)

        # Check if route is complete (max length reached)
        if len(new_state.current_route) >= self.max_route_length:
            new_state.all_routes.append(list(new_state.current_route))
            new_state.current_route_index += 1
            if new_state.current_route_index < self.num_routes:
                # Sync env state and use initialize_route (same as PPO/env.step)
                self.env.all_routes = [list(r) for r in new_state.all_routes]
                new_state.current_route = initialize_route(self.env)
            else:
                new_state.current_route = []  # Episode done

        return new_state

    def is_terminal(self) -> bool:
        """
        Check if all routes have been constructed.
        """
        return self.current_route_index >= self.num_routes

    def is_route_end(self) -> bool:
        """
        Check if current route has reached max length.
        """
        return len(self.current_route) >= self.max_route_length

    def force_route_end(self) -> 'MCTSState':
        """
        Force current route to end (when no valid actions).
        """
        new_state = self.clone()
        if new_state.current_route:
            new_state.all_routes.append(list(new_state.current_route))
            new_state.current_route_index += 1
            if new_state.current_route_index < self.num_routes:
                # Sync env state and use initialize_route (same as PPO/env.step)
                self.env.all_routes = [list(r) for r in new_state.all_routes]
                new_state.current_route = initialize_route(self.env)
            else:
                new_state.current_route = []
        return new_state


class MCTSNode:
    """
    Tree node with PUCT statistics for AlphaZero-style MCTS.

    Stores:
    - N[a]: visit count for action a
    - W[a]: cumulative value for action a
    - Q[a]: mean value Q = W/N
    - P[a]: prior probability from neural network
    """

    def __init__(self, state: MCTSState, parent: Optional['MCTSNode'] = None, action_from_parent: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[int, 'MCTSNode'] = {}

        # PUCT statistics (initialized on expand)
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.Q: Dict[int, float] = {}
        self.P: Dict[int, float] = {}

        self.expanded = False
        self.value: float = 0.0  # V from neural network

    def expand(self, priors: Dict[int, float], value: float) -> None:
        """
        Expand node with neural network outputs.

        Args:
            priors: Dict mapping action -> prior probability (already masked/normalized)
            value: Value estimate from neural network
        """
        self.expanded = True
        self.value = value

        for action, prob in priors.items():
            self.P[action] = prob
            self.N[action] = 0
            self.W[action] = 0.0
            self.Q[action] = 0.0

    def select_action(self, c_puct: float) -> int:
        """
        Select action using PUCT formula.

        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

        Returns:
            Selected action index
        """
        total_visits = sum(self.N.values())
        sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 1.0

        best_action = None
        best_puct = float('-inf')

        for action in self.P.keys():
            q_value = self.Q[action]
            prior = self.P[action]
            visits = self.N[action]

            # PUCT formula
            exploration = c_puct * prior * sqrt_total / (1 + visits)
            puct = q_value + exploration

            if puct > best_puct:
                best_puct = puct
                best_action = action

        return best_action

    def update(self, action: int, value: float) -> None:
        """
        Backpropagation update for a single action.

        Args:
            action: The action taken
            value: The value to backpropagate
        """
        self.N[action] += 1
        self.W[action] += value
        self.Q[action] = self.W[action] / self.N[action]

    def get_child(self, action: int) -> 'MCTSNode':
        """
        Get or create child node for action.
        """
        if action not in self.children:
            new_state = self.state.apply_action(action)
            self.children[action] = MCTSNode(
                state=new_state,
                parent=self,
                action_from_parent=action,
            )
        return self.children[action]

    def get_visit_count_policy(self, tau: float, n_actions: int) -> np.ndarray:
        """
        Get policy from visit counts with temperature.

        pi(a) = N(a)^(1/tau) / sum_b N(b)^(1/tau)

        Args:
            tau: Temperature (must be > 0). Lower = more greedy, higher = more uniform.
                 Typical range: 0.1 (near-greedy) to 1.0 (proportional to visits).
            n_actions: Total number of possible actions (for output size)

        Returns:
            Policy array of shape [n_actions]
        """
        policy = np.zeros(n_actions, dtype=np.float32)

        if not self.N:
            return policy

        # Temperature-scaled softmax over visit counts
        inv_tau = 1.0 / tau
        visit_counts = np.array([
            (a, self.N[a] ** inv_tau) for a in self.P.keys()
        ], dtype=object)

        total = sum(vc[1] for vc in visit_counts)
        if total > 0:
            for action, count in visit_counts:
                policy[action] = count / total

        return policy


class MCTSTree:
    """
    Orchestrates MCTS search operations.

    Maintains the search tree and provides methods for:
    - Running simulations
    - Advancing the tree (re-rooting after action selection)
    - Getting the improved policy from visit counts
    """

    def __init__(self, initial_state: MCTSState):
        self.root = MCTSNode(state=initial_state)

    def advance(self, action: int) -> None:
        """
        Re-root tree to child corresponding to action.
        Preserves subtree statistics, discards sibling subtrees.
        """
        if action in self.root.children:
            new_root = self.root.children[action]
            new_root.parent = None
            new_root.action_from_parent = None
            self.root = new_root
        else:
            # Child doesn't exist, create new root
            new_state = self.root.state.apply_action(action)
            self.root = MCTSNode(state=new_state)

    def get_policy(self, tau: float, n_actions: int) -> np.ndarray:
        """
        Get visit count policy from root.
        """
        return self.root.get_visit_count_policy(tau, n_actions)


class ReplayBuffer:
    """
    FIFO replay buffer for storing (state, policy, valid_actions, value) tuples.

    When capacity is reached, oldest entries are evicted.
    Stores valid_actions separately from policy to enable proper masking during
    training (mask over all valid actions, not just visited ones).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state_dict: Dict[str, Any], policy: np.ndarray, valid_actions: List[int], value: float) -> None:
        """
        Add a transition to the buffer.

        Args:
            state_dict: State dictionary from env._get_state()
            policy: MCTS-improved policy (visit count distribution)
            valid_actions: List of valid action indices at this state
            value: Normalized terminal reward
        """
        self.buffer.append((state_dict, policy, valid_actions, value))

    def sample(self, batch_size: int) -> List[Tuple[Dict[str, Any], np.ndarray, List[int], float]]:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            List of (state_dict, policy, valid_actions, value) tuples
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class WelfordNormalizer:
    """
    Online mean/variance computation using Welford's algorithm.

    Provides numerically stable incremental updates and normalization.
    """

    def __init__(self, eps: float = 1e-8):
        self.mean: float = 0.0
        self.var: float = 0.0
        self.count: int = 0
        self.eps = eps
        self._m2: float = 0.0  # Sum of squared differences

    def update(self, x: float) -> None:
        """
        Update running statistics with new value.

        Uses Welford's online algorithm for numerical stability.
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2

        # Update variance (using population variance)
        if self.count > 1:
            self.var = self._m2 / self.count
        else:
            self.var = 0.0

    def normalize(self, x: float, clip: float = 3.0) -> float:
        """
        Normalize value using running statistics.

        Args:
            x: Value to normalize
            clip: Clip range for normalized value

        Returns:
            Normalized and clipped value
        """
        std = math.sqrt(self.var) if self.var > 0 else 1.0
        normalized = (x - self.mean) / (std + self.eps)
        return max(-clip, min(clip, normalized))

    @property
    def std(self) -> float:
        """
        Get current standard deviation.
        """
        return math.sqrt(self.var) if self.var > 0 else 0.0

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state for checkpointing.
        """
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            '_m2': self._m2,
            'eps': self.eps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.
        """
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']
        self._m2 = state['_m2']
        self.eps = state.get('eps', 1e-8)


def add_dirichlet_noise(priors: Dict[int, float], alpha: float, eps: float) -> Dict[int, float]:
    """
    Add Dirichlet noise to prior probabilities for exploration.

    P'(a) = (1 - eps) * P(a) + eps * eta(a)
    where eta ~ Dir(alpha)

    Args:
        priors: Original prior probabilities
        alpha: Dirichlet concentration parameter
        eps: Weight for noise (0 = no noise, 1 = pure noise)

    Returns:
        Noisy prior probabilities
    """
    if not priors:
        return priors

    actions = list(priors.keys())
    n = len(actions)

    # Sample Dirichlet noise
    noise = np.random.dirichlet([alpha] * n)

    # Mix with original priors
    noisy_priors = {}
    for i, action in enumerate(actions):
        noisy_priors[action] = (1 - eps) * priors[action] + eps * noise[i]

    return noisy_priors


def parse_temp_schedule(schedule_str: str) -> list:
    """
    Parse temperature schedule string into sorted list of (threshold, tau) tuples.

    Args:
        schedule_str: Format "progress:tau,progress:tau,..." e.g., "0.3:1.0,0.6:0.5,1.0:0.1"

    Returns:
        List of (threshold, tau) tuples sorted by threshold ascending
    """
    pairs = []
    for pair in schedule_str.split(","):
        threshold, tau = pair.split(":")
        pairs.append((float(threshold), float(tau)))
    return sorted(pairs, key=lambda x: x[0])


def get_temperature(progress: float, schedule_str: str = "0.3:1.0,0.6:0.5,1.0:0.1") -> float:
    """
    Get temperature based on training progress and configurable schedule.

    Args:
        progress: Training progress in [0, 1]
        schedule_str: Temperature schedule as "progress:tau" pairs
                      e.g., "0.3:1.0,0.6:0.5,1.0:0.1" means:
                      - progress < 0.3: tau = 1.0
                      - 0.3 <= progress < 0.6: tau = 0.5
                      - progress >= 0.6: tau = 0.1

    Returns:
        Temperature value
    """
    schedule = parse_temp_schedule(schedule_str)

    # Find the appropriate temperature for current progress
    # Schedule is sorted ascending, so we return tau for first threshold > progress
    for threshold, tau in schedule:
        if progress < threshold:
            return tau

    # If progress >= all thresholds, return the last tau
    return schedule[-1][1]

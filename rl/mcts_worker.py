"""
MCTS Worker for Parallel Episode Collection

This module provides standalone functions for running MCTS episodes in worker
processes. All functions are self-contained with no dependency on MCTSAgent
instance state, enabling multiprocessing via starmap.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from torch_geometric.data import Data, Batch

from rl.env import TransitEnv
from rl.models import GATV2ActorCritic
from rl.mcts_utils import MCTSTree, MCTSState, MCTSNode, add_dirichlet_noise


def state_to_pyg_data(state_dict: Dict[str, Any]) -> Data:
    """Convert state dictionary to PyTorch Geometric Data object. (Pure function)"""
    return Data(
        x=torch.from_numpy(state_dict['node_features']).float(),
        edge_index=torch.from_numpy(state_dict['edge_index']).long(),
        edge_attr=torch.from_numpy(state_dict['edge_features']).float(),
    )


def sync_env_from_mcts_state(env: TransitEnv, mcts_state: MCTSState) -> None:
    """Sync environment internal state from MCTSState."""
    env.current_route = list(mcts_state.current_route)
    env.all_routes = [list(r) for r in mcts_state.all_routes]
    env.current_route_index = mcts_state.current_route_index


def get_state_tensor(env: TransitEnv, mcts_state: MCTSState) -> Dict[str, Any]:
    """Get state dictionary by syncing MCTSState to env and calling _get_state()."""
    sync_env_from_mcts_state(env, mcts_state)
    return env._get_state()


@torch.inference_mode()
def network_forward(model, state_dict: Dict[str, Any], valid_actions: List[int], device: str = 'cpu') -> Tuple[Dict[int, float], float]:
    """
    Forward pass through network to get priors and value.

    Standalone version of MCTSAgent._network_forward for worker processes.
    """
    model.eval()

    # Convert to PyG data and batch
    data = state_to_pyg_data(state_dict).to(device)
    batch = Batch.from_data_list([data])

    # Get node embeddings
    z = model._get_node_embeddings(
        batch.x, batch.edge_index, batch.edge_attr
    )

    # Actor: get logits per node
    logits = model.actor_head(z).squeeze(-1)  # [n_nodes]

    # Critic: get value
    g = model.critic_readout(z, batch.batch)
    value = model.critic_head(g).squeeze(-1).item()

    # Mask and softmax for priors
    if not valid_actions:
        return {}, value

    masked_logits = torch.full_like(logits, float('-inf'))
    for a in valid_actions:
        if a < len(logits):
            masked_logits[a] = logits[a]

    probs = F.softmax(masked_logits, dim=0).cpu().numpy()
    priors = {a: float(probs[a]) for a in valid_actions if a < len(probs)}

    return priors, value


def run_mcts_simulations(model, env: TransitEnv, tree: MCTSTree, tau: float, config: Dict[str, Any], add_noise: bool = True, device: str = 'cpu') -> np.ndarray:
    """
    Run MCTS simulations from current tree root.

    Standalone version of MCTSAgent._run_mcts for worker processes.

    Args:
        model: Neural network for value/policy estimation
        env: TransitEnv instance (for state observation generation)
        tree: MCTSTree to search
        tau: Temperature for visit count policy
        config: Config dict with n_iter, c_puct, dirichlet_alpha, dirichlet_eps
        add_noise: Whether to add Dirichlet noise at root
        device: Device for model inference ('cpu' for workers)

    Returns:
        Policy array from visit counts
    """
    n_nodes = env.n_nodes
    n_iter = config.get('n_iter', 100)
    c_puct = config.get('c_puct', 1.5)
    dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
    dirichlet_eps = config.get('dirichlet_eps', 0.25)

    root = tree.root
    root_state = root.state

    # Get valid actions at root
    valid_actions = root_state.get_valid_actions()
    if not valid_actions:
        return np.zeros(n_nodes, dtype=np.float32)

    # Expand root if needed
    if not root.expanded:
        state_dict = get_state_tensor(env, root_state)
        priors, value = network_forward(model, state_dict, valid_actions, device)
        root.expand(priors, value)

    # Add Dirichlet noise at root for exploration
    if add_noise and root.P:
        root.P = add_dirichlet_noise(root.P, dirichlet_alpha, dirichlet_eps)

    # Run simulations
    for _ in range(n_iter):
        node = root
        path: List[Tuple[MCTSNode, int]] = []

        # SELECT: Traverse tree using PUCT
        while node.expanded and not node.state.is_terminal():
            node_valid = node.state.get_valid_actions()
            if not node_valid:
                break

            action = node.select_action(c_puct)
            if action is None:
                break

            path.append((node, action))
            node = node.get_child(action)

        # EXPAND: Expand leaf if not terminal
        if not node.expanded and not node.state.is_terminal():
            leaf_valid = node.state.get_valid_actions()

            if leaf_valid:
                state_dict = get_state_tensor(env, node.state)
                priors, value = network_forward(model, state_dict, leaf_valid, device)
                node.expand(priors, value)
                v = value
            else:
                # No valid actions - evaluate forced-end successor
                next_state = node.state.force_route_end()
                next_state_dict = get_state_tensor(env, next_state)
                next_valid = next_state.get_valid_actions()
                _, v = network_forward(model, next_state_dict, next_valid, device)
        elif node.state.is_terminal():
            # Terminal state: get value estimate from network (no valid actions)
            state_dict = get_state_tensor(env, node.state)
            _, v = network_forward(model, state_dict, [], device)
        else:
            v = node.value

        # BACKPROPAGATE
        for parent_node, action in reversed(path):
            parent_node.update(action, v)

    # Get policy from visit counts
    policy = root.get_visit_count_policy(tau, n_nodes)

    return policy


def _create_mcts_state(env: TransitEnv) -> MCTSState:
    """Create MCTSState from environment."""
    return MCTSState(
        current_route=list(env.current_route),
        all_routes=[list(r) for r in env.all_routes],
        current_route_index=env.current_route_index,
        num_routes=env.NUM_ROUTES,
        max_route_length=env.MAX_ROUTE_LENGTH,
        adj=env.adj,
        node_to_idx=env.node_to_idx,
        idx_to_node=env.idx_to_node,
        env=env,
    )


def run_mcts_episode(model_state_dict: Dict[str, Any], policy_kwargs: Dict[str, Any], config: Dict[str, Any], seed: int, tau: float) -> Tuple[List[Tuple], float, int, List[List[str]]]:
    """
    Run a single MCTS episode in a worker process.

    This function is self-contained and creates all necessary
    objects (env, model, tree) from scratch.

    Args:
        model_state_dict: Serialized model weights
        policy_kwargs: Kwargs for model construction
        config: Full configuration dict
        seed: Random seed for this worker
        tau: Temperature for action selection

    Returns:
        Tuple of (episode_data, raw_reward, episode_length, routes)
    """
    # 1. Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Create environment
    env = TransitEnv(config)

    # 3. Create model and load weights
    device = 'cuda' if (config.get('gpu', False) and torch.cuda.is_available()) else 'cpu'
    model = GATV2ActorCritic(**policy_kwargs)
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)

    # 4. Run episode
    state_dict, _ = env.reset(seed=seed)
    mcts_state = _create_mcts_state(env)
    tree = MCTSTree(mcts_state)

    episode_data = []
    actions_taken = []

    while not mcts_state.is_terminal():
        valid_actions = mcts_state.get_valid_actions()

        if not valid_actions:
            # No valid actions - force route end
            actions_taken.append(env.NO_VALID_ACTION)
            mcts_state = mcts_state.force_route_end()
            tree = MCTSTree(mcts_state)
            continue

        # Get current state observation (need to sync mcts_state → env)
        current_state_dict = get_state_tensor(env, mcts_state)

        # Run MCTS and get policy
        policy = run_mcts_simulations(model, env, tree, tau, config, add_noise=True, device=device)

        # Record state and policy
        episode_data.append((
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in current_state_dict.items()},
            policy.copy(),
            list(valid_actions)
        ))

        # Sample action from policy
        valid_policy = policy[valid_actions]
        if valid_policy.sum() > 0:
            valid_policy = valid_policy / valid_policy.sum()
            action = np.random.choice(valid_actions, p=valid_policy)
        else:
            action = np.random.choice(valid_actions)

        actions_taken.append(action)
        mcts_state = mcts_state.apply_action(action)
        tree.advance(action)

    # 5. Replay actions to get terminal reward
    state_dict, _ = env.reset(seed=seed)
    terminal_reward = 0.0

    for action in actions_taken:
        state_dict, reward, terminated, _, info = env.step(action)
        terminal_reward = reward  # Keep updating - final one is terminal reward

        if terminated:
            break

    # 6. Return results (as tuple for pickling simplicity)
    return (
        episode_data,
        terminal_reward,
        len(actions_taken),
        [list(r) for r in env.all_routes],
    )

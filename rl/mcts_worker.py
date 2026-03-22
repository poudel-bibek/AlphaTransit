"""
MCTS Worker for Parallel Episode Collection

This module provides standalone functions for running MCTS episodes in worker
processes. All functions are self-contained with no dependency on MCTSAgent
instance state, enabling multiprocessing.

Key optimizations:
1. Persistent workers (mcts_worker_loop): env created once per worker
   process, not per episode.
2. Centralized inference service: workers ship leaf states to one shared
   model process instead of contending on the GPU with per-worker models.
3. Batched leaf evaluation (run_mcts_simulations): collects K leaves per batch
   using virtual loss to diversify PUCT selections, evaluates all K in a single
   NN forward pass, then rolls back virtual loss and backpropagates real values.
   Reduces NN calls from n_iter to ~n_iter/K per move.
"""

import os
import random
import numpy as np
import torch
from typing import Any, Dict, List, Tuple

from rl.env import TransitEnv
from rl.mcts_utils import MCTSTree, MCTSState, MCTSNode, add_dirichlet_noise


def _cap_worker_threads():
    """Limit BLAS/OpenMP and PyTorch threads to 1 per worker process."""
    for var in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ]:
        os.environ[var] = os.environ.get(var, "1") or "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def sync_env_from_mcts_state(env: TransitEnv, mcts_state: MCTSState) -> None:
    """Sync environment internal state from MCTSState."""
    env.current_route = mcts_state.current_route
    env.all_routes = mcts_state.all_routes
    env.current_route_index = mcts_state.current_route_index


def get_state_tensor(env: TransitEnv, mcts_state: MCTSState) -> Dict[str, Any]:
    """Get state dictionary by syncing MCTSState to env and calling _get_state()."""
    sync_env_from_mcts_state(env, mcts_state)
    return env._get_state()


class RemoteInferenceClient:
    """Blocking RPC client used by a single MCTS worker."""

    def __init__(self, worker_id: int, policy_version: int, request_queue: Any, response_queue: Any):
        self.worker_id = worker_id
        self.policy_version = policy_version
        self.request_queue = request_queue
        self.response_queue = response_queue

    def infer_batch(self, payloads: List[Dict[str, Any]]) -> List[Tuple[Dict[int, float], float]]:
        if not payloads:
            return []

        self.request_queue.put(
            {
                "worker_id": self.worker_id,
                "policy_version": self.policy_version,
                "payloads": payloads,
            }
        )
        response = self.response_queue.get()
        if response.get("error"):
            raise RuntimeError(response["error"])
        if int(response["policy_version"]) != self.policy_version:
            raise RuntimeError(
                f"Worker {self.worker_id} received stale inference response for "
                f"policy_version={response['policy_version']}, expected={self.policy_version}"
            )
        return response["outputs"]

    def infer_single(
        self,
        state_key: Any,
        state_dict: Dict[str, Any],
        valid_actions: List[int],
    ) -> Tuple[Dict[int, float], float]:
        return self.infer_batch(
            [
                {
                    "state_key": state_key,
                    "state_dict": state_dict,
                    "valid_actions": valid_actions,
                }
            ]
        )[0]


def _apply_virtual_loss(path: List[Tuple[MCTSNode, int]], vloss: int = 1) -> None:
    """
    Apply virtual loss along a selection path to discourage re-selection.

    When collecting K leaves in a batch, we need each selection to explore
    a different part of the tree. After selecting leaf i, we temporarily
    penalize the path taken by adding a fake visit with negative value.
    This makes the Q-values along this path appear worse, so the next
    PUCT selection (for leaf i+1) will prefer a different branch.

    The virtual loss is rolled back in Phase 3 before real backpropagation.
    """
    for node, action in path:
        node.N[action] += vloss   # Fake visit count
        node.W[action] -= vloss   # Negative value to lower Q
        node.Q[action] = node.W[action] / node.N[action]


def _rollback_virtual_loss(path: List[Tuple[MCTSNode, int]], vloss: int = 1) -> None:
    """
    Remove virtual loss along a selection path before real backpropagation.

    Undoes the fake visit and negative value added by _apply_virtual_loss,
    restoring the tree statistics to their true state. After rollback,
    the normal update() call applies the real value from the NN evaluation.
    """
    for node, action in path:
        node.N[action] -= vloss
        node.W[action] += vloss
        if node.N[action] > 0:
            node.Q[action] = node.W[action] / node.N[action]
        else:
            node.Q[action] = 0.0


def run_mcts_simulations(
    env: TransitEnv,
    tree: MCTSTree,
    tau: float,
    config: Dict[str, Any],
    inference_client: RemoteInferenceClient,
    add_noise: bool = True,
) -> np.ndarray:
    """
    Run MCTS simulations with batched leaf evaluation and virtual loss.

    Collects K leaves per batch using virtual loss to diversify selections,
    evaluates them in a single NN forward pass, then rolls back virtual loss
    and backpropagates real values. Single-threaded, so tree mutation is safe.

    Args:
        env: TransitEnv instance (for state observation generation)
        tree: MCTSTree to search
        tau: Temperature for visit count policy
        config: Config dict with n_iter, c_puct, mcts_batch_size, etc.
        inference_client: RPC client for centralized model inference
        add_noise: Whether to add Dirichlet noise at root

    Returns:
        Policy array from visit counts
    """
    n_nodes = env.n_nodes
    n_iter = config.get('n_iter', 100)
    leaf_batch_size = config.get('mcts_batch_size', 8)
    c_puct = config.get('c_puct', 1.5)
    dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
    dirichlet_eps = config.get('dirichlet_eps', 0.25)

    root = tree.root

    # Get valid actions at root
    valid_actions = root.state.get_valid_actions()
    if not valid_actions:
        return np.zeros(n_nodes, dtype=np.float32)

    # Expand root if needed (single forward pass)
    if not root.expanded:
        state_dict = get_state_tensor(env, root.state)
        priors, value = inference_client.infer_single(
            state_key=root.state.cache_key(),
            state_dict=state_dict,
            valid_actions=valid_actions,
        )
        root.expand(priors, value)

    # Add Dirichlet noise at root for exploration
    if add_noise and root.P:
        root.P = add_dirichlet_noise(root.P, dirichlet_alpha, dirichlet_eps)

    # =========================================================================
    # Batched simulation loop
    #
    # Instead of the standard sequential loop (select one leaf, expand it,
    # backprop, repeat), we process K leaves at a time:
    #
    #   Phase 1 - SELECT:  Run K PUCT selections sequentially. After each,
    #                       apply virtual loss on the selected path so the
    #                       next selection diverges to a different leaf.
    #   Phase 2 - EXPAND:  Collect all K leaf states, evaluate them in a
    #                       single batched NN forward pass.
    #   Phase 3 - BACKUP:  Roll back virtual loss on all K paths, expand
    #                       the leaf nodes with real priors, and backprop
    #                       real values up each path.
    #
    # This is all single-threaded within one worker, so tree mutation
    # (N/W/Q dicts, children dict) is safe without locking.
    # =========================================================================
    sims_done = 0
    while sims_done < n_iter:
        current_batch = min(leaf_batch_size, n_iter - sims_done)

        # --- Phase 1: SELECT K leaves, applying virtual loss after each ---
        # Each entry: (leaf_type, path, node, extra_info)
        #   leaf_type: 'expand'   - normal unexpanded leaf, needs NN priors + value
        #              'force_end' - no valid actions, evaluate the forced successor
        #              'terminal'  - all routes built, just need NN value
        #              'existing'  - already expanded (edge case), use stored value
        pending = []

        for _ in range(current_batch):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            # PUCT selection down the tree. Virtual loss from earlier selections
            # in this batch lowers Q-values on already-chosen paths, steering
            # this selection toward a different leaf.
            while node.expanded and not node.state.is_terminal():
                node_valid = node.state.get_valid_actions()
                if not node_valid:
                    break
                action = node.select_action(c_puct)
                if action is None:
                    break
                path.append((node, action))
                node = node.get_child(action)

            # Classify the leaf we landed on
            if not node.expanded and not node.state.is_terminal():
                leaf_valid = node.state.get_valid_actions()
                if leaf_valid:
                    pending.append(('expand', path, node, leaf_valid))
                else:
                    # Dead end mid-episode: evaluate "what if this route ends here
                    # and the next route starts from the transit center?"
                    next_state = node.state.force_route_end()
                    next_valid = next_state.get_valid_actions()
                    pending.append(('force_end', path, node, (next_state, next_valid)))
            elif node.state.is_terminal():
                pending.append(('terminal', path, node, None))
            else:
                pending.append(('existing', path, node, None))

            # Apply virtual loss so the next selection in this batch diverges.
            # This is rolled back in Phase 3 before real backpropagation.
            _apply_virtual_loss(path)

        # --- Phase 2: Batch NN forward for all leaves needing evaluation ---
        # Collect (state_dict, valid_actions) for each leaf that needs the NN.
        # 'existing' leaves skip the NN since they already have a stored value.
        nn_entries = []  # (index_in_pending, state_key, state_dict, valid_actions)
        for i, (leaf_type, path, node, info) in enumerate(pending):
            if leaf_type == 'expand':
                # get_state_tensor syncs mcts_state to env then calls env._get_state().
                # Each call produces independent numpy arrays, so collecting multiple
                # states before the batch forward is safe.
                sd = get_state_tensor(env, node.state)
                nn_entries.append((i, node.state.cache_key(), sd, info))
            elif leaf_type == 'force_end':
                next_state, next_valid = info
                sd = get_state_tensor(env, next_state)
                nn_entries.append((i, next_state.cache_key(), sd, next_valid))
            elif leaf_type == 'terminal':
                sd = get_state_tensor(env, node.state)
                nn_entries.append((i, node.state.cache_key(), sd, []))

        # Single batched forward pass for all K leaves (the key speedup)
        nn_results = {}
        if nn_entries:
            payloads = [
                {
                    "state_key": state_key,
                    "state_dict": state_dict,
                    "valid_actions": valid_actions,
                }
                for _, state_key, state_dict, valid_actions in nn_entries
            ]
            batch_out = inference_client.infer_batch(payloads)
            for j, (idx, _, _, _) in enumerate(nn_entries):
                nn_results[idx] = batch_out[j]

        # --- Phase 3: Rollback virtual loss, expand nodes, backpropagate ---
        for i, (leaf_type, path, node, info) in enumerate(pending):
            # Undo the fake visits/values from Phase 1
            _rollback_virtual_loss(path)

            if leaf_type == 'expand':
                priors, value = nn_results[i]
                # Guard: if two sims in this batch reached the same unexpanded node
                # (rare, since virtual loss steers them apart), only expand once.
                if not node.expanded:
                    node.expand(priors, value)
                v = value
            elif leaf_type == 'force_end':
                _, v = nn_results[i]
            elif leaf_type == 'terminal':
                _, v = nn_results[i]
            else:  # existing (selection stopped at an expanded non-terminal node)
                v = node.value

            # Standard MCTS backpropagation with the real value
            for parent_node, action in reversed(path):
                parent_node.update(action, v)

        sims_done += current_batch

    return root.get_visit_count_policy(tau, n_nodes)


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


def _run_episode_on_env(env, config, seed, tau, inference_client: RemoteInferenceClient):
    """
    Run a single MCTS episode using an already-initialized env.

    Called by mcts_worker_loop for each "collect" command.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset(seed=seed)
    mcts_state = _create_mcts_state(env)
    tree = MCTSTree(mcts_state)

    episode_data = []
    actions_taken = []

    while not mcts_state.is_terminal():
        valid_actions = mcts_state.get_valid_actions()

        if not valid_actions:
            actions_taken.append(env.NO_VALID_ACTION)
            mcts_state = mcts_state.force_route_end()
            tree = MCTSTree(mcts_state)
            continue

        current_state_dict = get_state_tensor(env, mcts_state)
        policy = run_mcts_simulations(
            env=env,
            tree=tree,
            tau=tau,
            config=config,
            inference_client=inference_client,
            add_noise=True,
        )

        episode_data.append((
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in current_state_dict.items()},
            policy.copy(),
            list(valid_actions)
        ))

        valid_policy = policy[valid_actions]
        if valid_policy.sum() > 0:
            valid_policy = valid_policy / valid_policy.sum()
            action = np.random.choice(valid_actions, p=valid_policy)
        else:
            action = np.random.choice(valid_actions)

        actions_taken.append(action)
        mcts_state = mcts_state.apply_action(action)
        tree.advance(action)

    terminal_reward, _ = env.simulate_routes_mcts(mcts_state.all_routes)

    return (
        episode_data,
        terminal_reward,
        len(actions_taken),
        [list(r) for r in env.all_routes],
    )


def mcts_worker_loop(worker_id, config, cmd_queue, result_queue, inference_request_queue, inference_response_queue):
    """
    Persistent MCTS worker process.

    Unlike the old Pool+starmap approach where each episode created a fresh
    TransitEnv, this creates the environment once at startup and reuses it
    across all episodes. Neural-network inference is delegated to the
    centralized inference service.

    Commands (via cmd_queue):
        {"type": "collect", "tau": float, "seed": int}  - Run one self-play episode
        {"type": "stop"}                                  - Terminate worker
    """
    _cap_worker_threads()
    # Create env ONCE (the whole point of persistent workers).
    # Previously this was recreated every episode, adding CSV parsing and
    # environment initialization overhead per iteration.
    env = TransitEnv(config)

    while True:
        cmd = cmd_queue.get()
        cmd_type = cmd.get("type") if isinstance(cmd, dict) else None

        if cmd_type == "stop":
            break

        elif cmd_type == "collect":
            tau = cmd["tau"]
            seed = cmd["seed"]
            policy_version = cmd["policy_version"]
            inference_client = RemoteInferenceClient(
                worker_id=worker_id,
                policy_version=policy_version,
                request_queue=inference_request_queue,
                response_queue=inference_response_queue,
            )
            result = _run_episode_on_env(
                env=env,
                config=config,
                seed=seed,
                tau=tau,
                inference_client=inference_client,
            )
            result_queue.put(result)

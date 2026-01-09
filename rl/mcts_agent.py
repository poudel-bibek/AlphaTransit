"""
MCTSAgent: AlphaTransit Implementation for Transit Route Network Design

This module implements the AlphaTransit algorithm, which combines Monte Carlo
Tree Search (MCTS) with neural network guidance for transit route design.

Key components:
- MCTS with PUCT selection for action selection
- Neural network (GATv2ActorCritic) for policy priors and value estimation
- Dirichlet noise for exploration during self-play
- Replay buffer with terminal-only rewards
- Welford normalization for reward stability
- Parallel episode collection via num_mcts_workers (replaces episodes_per_iter)
"""

import gc
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import wandb
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from torch_geometric.data import Data, Batch

# Use get_context (not set_start_method) - can be called multiple times safely
# Following PPO's pattern in rl/parallel_env.py
mp_ctx = mp.get_context('spawn')

from rl.models import GATV2ActorCritic
from rl.env_utils import plot_network_and_demand, aggregate_results, write_results_summary, ensure_eval_step_update_dir, make_seed_output_dir, save_routes_json
from rl.mcts_utils import MCTSState, MCTSNode, MCTSTree, ReplayBuffer, WelfordNormalizer, add_dirichlet_noise, get_temperature


class MCTSAgent:
    """
    Monte Carlo Tree Search Agent for AlphaTransit.

    Combines MCTS with neural network guidance to learn transit route design.
    Uses terminal-only rewards with Welford normalization.
    """

    def __init__(self, env: Any, config: Dict[str, Any], policy_kwargs: Dict[str, Any]) -> None:
        """
        Initialize MCTS agent.

        Args:
            env: TransitEnv instance
            config: Configuration dictionary containing:
                - MCTS hyperparameters (n_iter, c_puct, dirichlet_alpha, etc.)
                - Training hyperparameters (lr, batch_size, etc.)
            policy_kwargs: Model architecture kwargs from get_policy_kwargs_mcts
        """
        self.env = env
        self.config = dict(config)
        self.policy_kwargs = policy_kwargs  # Store for passing to workers
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        # MCTS hyperparameters
        self.n_iter = config['n_iter']
        self.c_puct = config['c_puct']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.dirichlet_eps = config['dirichlet_eps']

        # Training hyperparameters
        self.buffer_capacity = config['buffer_capacity']
        self.train_steps_per_iter = config['train_steps_per_iter']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.max_iterations = config['max_iterations']

        # Environment parameters
        self.n_nodes = env.n_nodes
        self.n_actions = env.n_nodes + 1  # +1 for NO_VALID_ACTION
        self.num_routes = env.NUM_ROUTES
        self.max_route_length = env.MAX_ROUTE_LENGTH

        # Initialize model
        self.model = GATV2ActorCritic(**policy_kwargs).to(self.device)
        self.model.apply_orthogonal_init()

        # TODO: Uncomment when PyTorch 2.10 is released (adds Python 3.14 support)
        # self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Replay buffer and normalizer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        self.reward_normalizer = WelfordNormalizer()

        # Training state
        # total_env_steps: Total environment interactions (actions taken across all episodes).
        # This is the key metric for sample efficiency comparison with PPO, which also
        # measures progress in environment steps. Enables apples-to-apples comparison
        # of how many environment interactions each algorithm needs to reach a given
        # performance level.
        self.total_env_steps = 0
        self.total_episodes = 0

        # Evaluation config
        self.eval_every = config.get('eval_every', 10)
        self.num_eval_runs = config.get('num_eval_runs', 5)
        self.seed = config.get('seed', 42)
        self.eval_seed_offset = config.get('eval_seed_offset', 2)

        # Create timestamped save directory (like PPO)
        now = datetime.now()
        base_save_dir = config.get('save_dir', './training_data')
        self.training_save_dir = Path(base_save_dir) / f"{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}"
        self.training_save_dir.mkdir(parents=True, exist_ok=True)

        # Policy directory for model weights
        self.policy_dir = self.training_save_dir / "mcts_policies"
        self.policy_dir.mkdir(parents=True, exist_ok=True)

        # Parallel workers configuration
        self.num_workers = config.get('num_mcts_workers', 4)
        self.pool = mp_ctx.Pool(self.num_workers)

        # Save initial network visualization
        self._save_network_visualization()

    def _save_network_visualization(self) -> None:
        """Save initial network and demand visualization."""
        temp_world = self.env.build_world(self.config.get("network"))
        self.env.load_demand_for_plotting(temp_world)
        output_path = self.training_save_dir / f"00_{self.config.get('network')}_demand_network.png"
        plot_network_and_demand(temp_world, str(output_path))

    def _create_mcts_state(self) -> MCTSState:
        """Create MCTSState from current environment state."""
        return MCTSState(
            current_route=list(self.env.current_route),
            all_routes=[list(r) for r in self.env.all_routes],
            current_route_index=self.env.current_route_index,
            num_routes=self.num_routes,
            max_route_length=self.max_route_length,
            adj=self.env.adj,
            node_to_idx=self.env.node_to_idx,
            idx_to_node=self.env.idx_to_node,
            env=self.env,
        )

    def _sync_env_from_mcts_state(self, mcts_state: MCTSState) -> None:
        """Temporarily sync environment internal state from MCTSState."""
        self.env.current_route = list(mcts_state.current_route)
        self.env.all_routes = [list(r) for r in mcts_state.all_routes]
        self.env.current_route_index = mcts_state.current_route_index

    def _get_state_tensor(self, mcts_state: MCTSState) -> Dict[str, Any]:
        """
        Get state dictionary by syncing MCTSState to env and calling _get_state().

        This reconstructs the full observation from the lightweight MCTSState.
        """
        self._sync_env_from_mcts_state(mcts_state)
        return self.env._get_state()

    def _state_to_pyg_data(self, state_dict: Dict[str, Any]) -> Data:
        """Convert state dictionary to PyTorch Geometric Data object."""
        return Data(
            x=torch.from_numpy(state_dict['node_features']).float(),
            edge_index=torch.from_numpy(state_dict['edge_index']).long(),
            edge_attr=torch.from_numpy(state_dict['edge_features']).float(),
        )

    def _get_valid_mask(self, valid_actions: List[int]) -> torch.Tensor:
        """Create valid action mask tensor."""
        mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        for a in valid_actions:
            if a < self.n_nodes:
                mask[a] = True
        return mask

    @torch.no_grad()
    def _network_forward(self, state_dict: Dict[str, Any], valid_actions: List[int]) -> Tuple[Dict[int, float], float]:
        """
        Forward pass through network to get priors and value.

        Args:
            state_dict: State dictionary from env._get_state()
            valid_actions: List of valid action indices

        Returns:
            priors: Dict mapping action -> prior probability
            value: Value estimate (scalar)
        """
        self.model.eval()

        # Convert to PyG data and batch
        data = self._state_to_pyg_data(state_dict).to(self.device)
        batch = Batch.from_data_list([data])

        # Get node embeddings
        z = self.model._get_node_embeddings(
            batch.x, batch.edge_index, batch.edge_attr
        )

        # Actor: get logits per node
        logits = self.model.actor_head(z).squeeze(-1)  # [n_nodes]

        # Critic: get value
        g = self.model.critic_readout(z, batch.batch)
        value = self.model.critic_head(g).squeeze(-1).item()

        # Mask and softmax for priors
        if not valid_actions:
            return {}, value

        # Create masked logits
        masked_logits = torch.full_like(logits, float('-inf'))
        for a in valid_actions:
            if a < len(logits):
                masked_logits[a] = logits[a]

        # Softmax to get probabilities
        probs = F.softmax(masked_logits, dim=0).cpu().numpy()

        # Build prior dict
        priors = {a: float(probs[a]) for a in valid_actions if a < len(probs)}

        return priors, value

    def _run_mcts(self, tree: MCTSTree, tau: float, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS simulations from current tree root.

        Args:
            tree: MCTSTree to search
            tau: Temperature for visit count policy (passed from caller for consistency)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Policy array from visit counts
        """
        root = tree.root
        root_state = root.state

        # Get valid actions at root
        valid_actions = root_state.get_valid_actions()
        if not valid_actions:
            # No valid actions - return empty policy
            return np.zeros(self.n_nodes, dtype=np.float32)

        # Expand root if needed
        if not root.expanded:
            state_dict = self._get_state_tensor(root_state)
            priors, value = self._network_forward(state_dict, valid_actions)
            root.expand(priors, value)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.P:
            root.P = add_dirichlet_noise(
                root.P, self.dirichlet_alpha, self.dirichlet_eps
            )

        # Run simulations
        for _ in range(self.n_iter):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            # SELECT: Traverse tree using PUCT
            while node.expanded and not node.state.is_terminal():
                # Check for valid actions at this node
                node_valid = node.state.get_valid_actions()
                if not node_valid:
                    # No valid actions - force route end
                    break

                action = node.select_action(self.c_puct)
                if action is None:
                    break

                path.append((node, action))
                node = node.get_child(action)

            # EXPAND: Expand leaf if not terminal
            if not node.expanded and not node.state.is_terminal():
                leaf_valid = node.state.get_valid_actions()

                if leaf_valid:
                    state_dict = self._get_state_tensor(node.state)
                    priors, value = self._network_forward(state_dict, leaf_valid)
                    node.expand(priors, value)
                    v = value
                else:
                    # No valid actions mid-episode: evaluate the forced-end successor.
                    # A route ending early isn't necessarily bad - the value depends on
                    # what comes next. We evaluate: "If this route ends here, what's the
                    # value of starting the next route from the transit center?" This
                    # removes bias against exploration paths that lead to shorter routes.
                    next_state = node.state.force_route_end()
                    next_state_dict = self._get_state_tensor(next_state)
                    next_valid = next_state.get_valid_actions()
                    _, v = self._network_forward(next_state_dict, next_valid)
            elif node.state.is_terminal():
                # Terminal state: get value estimate from network (no valid actions)
                state_dict = self._get_state_tensor(node.state)
                _, v = self._network_forward(state_dict, [])
            else:
                v = node.value

            # BACKPROPAGATE
            for parent_node, action in reversed(path):
                parent_node.update(action, v)

        # Get policy from visit counts using passed temperature
        policy = root.get_visit_count_policy(tau, self.n_nodes)

        return policy

    def _collect_episodes(self, tau: float, iteration: int) -> List[Tuple]:
        """
        Collect episodes from parallel workers.

        Args:
            tau: Temperature for action selection
            iteration: Current training iteration (for seed generation)

        Returns:
            List of (episode_data, raw_reward, episode_length, routes) tuples
        """
        from rl.mcts_worker import run_mcts_episode

        # Generate unique seeds for each worker
        base_seed = self.seed + iteration * self.num_workers

        # Serialize model weights to CPU for workers
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Prepare worker inputs
        worker_inputs = [
            (cpu_state_dict, self.policy_kwargs, self.config, base_seed + i, tau)
            for i in range(self.num_workers)
        ]

        # Run episodes in parallel
        print(f"  Collecting...", end="", flush=True)
        results = self.pool.starmap(run_mcts_episode, worker_inputs)
        print(" done")
        return results

    def _cleanup(self) -> None:
        """Clean up pool at end of training."""
        self.pool.close()
        self.pool.join()

    def _train_step(self) -> Dict[str, float]:
        """
        Perform one training step on a batch from replay buffer.

        Returns:
            Dict of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.model.train()

        # Sample batch: (state_dict, policy, valid_actions, value)
        batch_data = self.replay_buffer.sample(self.batch_size)

        # Prepare batch
        data_list = []
        target_policies = []
        target_values = []
        valid_masks = []

        for state_dict, policy, valid_actions, value in batch_data:
            data = self._state_to_pyg_data(state_dict)
            data_list.append(data)
            target_policies.append(torch.from_numpy(policy).float())
            target_values.append(value)

            # Build mask from valid_actions (not from policy > 0).
            # This ensures the network learns over ALL valid actions, including
            # valid-but-unvisited ones where MCTS assigned zero probability.
            mask = torch.zeros(self.n_nodes, dtype=torch.bool)
            for a in valid_actions:
                if a < self.n_nodes:
                    mask[a] = True
            valid_masks.append(mask)

        # Batch data
        batch = Batch.from_data_list(data_list).to(self.device)
        target_policies = torch.stack(target_policies).to(self.device)
        target_values = torch.tensor(target_values, dtype=torch.float32).to(self.device)
        valid_masks = torch.stack(valid_masks).to(self.device)

        # Forward pass
        z = self.model._get_node_embeddings(
            batch.x, batch.edge_index, batch.edge_attr
        )
        logits = self.model.actor_head(z).squeeze(-1)
        g = self.model.critic_readout(z, batch.batch)
        values = self.model.critic_head(g).squeeze(-1)

        # Reshape logits per graph
        ptr = self.model._get_ptr(batch.batch)
        num_graphs = batch.num_graphs

        policy_losses = []
        for i in range(num_graphs):
            start, end = ptr[i], ptr[i + 1]
            graph_logits = logits[start:end]

            # Mask invalid actions (actions outside valid set)
            mask_i = valid_masks[i, :len(graph_logits)]
            masked_logits = graph_logits.clone()
            masked_logits[~mask_i] = float('-inf')

            # Log softmax for cross-entropy (over valid actions only)
            log_probs = F.log_softmax(masked_logits, dim=0)

            # Cross-entropy with target policy.
            # Only sum where target_p > 0 to avoid NaN from 0 * -inf (IEEE-754).
            # This is mathematically equivalent since 0 * log(p) = 0, but IEEE
            # gives NaN for 0 * -inf which can poison training silently.
            target_p = target_policies[i, :len(graph_logits)]
            nonzero_mask = target_p > 0
            if nonzero_mask.any():
                policy_loss = -torch.sum(target_p[nonzero_mask] * log_probs[nonzero_mask])
            else:
                # Edge case: all zeros in target (shouldn't happen normally)
                policy_loss = torch.tensor(0.0, device=self.device)
            policy_losses.append(policy_loss)

        policy_loss = torch.stack(policy_losses).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
        }

    def train(self) -> None:
        """
        Main training loop.

        Alternates between self-play data generation and network optimization.
        Uses parallel workers to collect episodes concurrently.
        """
        print(f"Starting MCTS training for {self.max_iterations} iterations")
        print(f"  Parallel workers: {self.num_workers}")
        print(f"  Train steps per iteration: {self.train_steps_per_iter}")
        print(f"  MCTS simulations per move: {self.n_iter}")
        print(f"  Device: {self.device}")

        best_reward = float('-inf')

        try:
            for iteration in range(1, self.max_iterations + 1):
                # Temperature is computed once per iteration from iteration-based progress,
                # ensuring consistency between self-play action selection and logging.
                progress = iteration / self.max_iterations
                tau = get_temperature(progress)

                # Parallel self-play phase
                results = self._collect_episodes(tau, iteration)

                # Aggregate results: first update Welford with all raw rewards
                episode_rewards = []
                for episode_data, raw_reward, length, routes in results:
                    self.reward_normalizer.update(raw_reward)
                    episode_rewards.append(raw_reward)

                # Then add normalized data to buffer
                for episode_data, raw_reward, length, routes in results:
                    normalized_reward = self.reward_normalizer.normalize(raw_reward)
                    for state_dict, policy, valid_actions in episode_data:
                        self.replay_buffer.add(state_dict, policy, valid_actions, normalized_reward)

                # Track environment steps and episodes
                total_steps = sum(length for _, _, length, _ in results)
                self.total_env_steps += total_steps
                self.total_episodes += len(results)

                # Memory cleanup after processing worker results
                del results
                gc.collect()
                torch.cuda.empty_cache()

                # Training phase
                train_metrics = []
                for _ in range(self.train_steps_per_iter):
                    metrics = self._train_step()
                    if metrics:
                        train_metrics.append(metrics)

                # Logging
                avg_reward = np.mean(episode_rewards)
                if train_metrics:
                    avg_policy_loss = np.mean([m['policy_loss'] for m in train_metrics])
                    avg_value_loss = np.mean([m['value_loss'] for m in train_metrics])
                else:
                    avg_policy_loss = avg_value_loss = 0.0

                print(f"Iter {iteration:4d} | "
                      f"Reward: {avg_reward:8.2f} | "
                      f"Policy Loss: {avg_policy_loss:.4f} | "
                      f"Value Loss: {avg_value_loss:.4f} | "
                      f"Buffer: {len(self.replay_buffer):6d} | "
                      f"Tau: {tau:.2f}")

                # WandB logging
                if not self.config.get("wandb_off"):
                    wandb.log({
                        "mcts/policy_loss": avg_policy_loss,
                        "mcts/value_loss": avg_value_loss,
                        "mcts/total_loss": avg_policy_loss + avg_value_loss,
                        "mcts/avg_reward": avg_reward,
                        "mcts/best_reward": best_reward if avg_reward <= best_reward else avg_reward,
                        "mcts/buffer_size": len(self.replay_buffer),
                        "mcts/temperature": tau,
                        "mcts/iteration": iteration,
                        "mcts/progress": progress,
                    }, step=self.total_env_steps)

                # Save policy and track best
                policy_path = self._save_policy(iteration, self.total_env_steps)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    # Save best policy separately
                    best_path = self.policy_dir / "policy_best.pth"
                    torch.save(self.model.state_dict(), best_path)

                # Periodic evaluation (like PPO)
                if self.eval_every > 0 and iteration % self.eval_every == 0:
                    self._run_evaluation(policy_path, iteration)

            # Final save
            final_path = self.policy_dir / "policy_final.pth"
            torch.save(self.model.state_dict(), final_path)
            print(f"Training complete. Best reward: {best_reward:.2f}")
            print(f"Results saved to: {self.training_save_dir}")

        finally:
            self._cleanup()  # Always clean up pool

    def _save_policy(self, update: int, steps: int) -> str:
        """
        Save model weights (consistent with PPO naming).
        Returns path to saved policy.
        """
        filename = f"policy_up_{update}_step_{steps}.pth"
        path = self.policy_dir / filename
        torch.save(self.model.state_dict(), path)
        return str(path)

    def _load_policy(self, path: str) -> None:
        """Load model weights from policy file."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _run_single_eval_episode(self, seed: int) -> Dict[str, Any]:
        """
        Run a single evaluation episode with given seed.
        Returns metrics dict with routes.
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.model.eval()
        eval_tau = 0.1  # Near-greedy (matches training minimum)

        state_dict, _ = self.env.reset(seed=seed)
        mcts_state = self._create_mcts_state()
        tree = MCTSTree(mcts_state)
        actions = []

        while not mcts_state.is_terminal():
            valid_actions = mcts_state.get_valid_actions()
            if not valid_actions:
                # Record NO_VALID_ACTION to keep MCTSState and real env in sync
                # (see run_mcts_episode in mcts_worker.py for detailed explanation)
                actions.append(self.env.NO_VALID_ACTION)
                mcts_state = mcts_state.force_route_end()
                tree = MCTSTree(mcts_state)
                continue

            policy = self._run_mcts(tree, tau=eval_tau, add_noise=False)
            valid_policy = policy[valid_actions]
            if valid_policy.sum() > 0:
                action = valid_actions[np.argmax(valid_policy)]
            else:
                action = valid_actions[0]

            actions.append(action)
            mcts_state = mcts_state.apply_action(action)
            tree.advance(action)

        # Replay to get actual reward and metrics
        state_dict, _ = self.env.reset(seed=seed)
        for action in actions:
            state_dict, reward, terminated, _, info = self.env.step(action)
            if terminated:
                break

        # Get simulation results
        # NOTE: env.step() returns sim_result directly as info (not wrapped like baselines)
        sim_result = info
        metrics = {
            'episode_total_reward': reward,
            'episode_length': len(actions),
            'routes': [list(r) for r in self.env.all_routes],
            'seed': seed,
            **sim_result
        }
        return metrics

    def _run_evaluation(self, policy_path: str, iteration: int) -> Dict[str, float]:
        """
        Run evaluation during training (like PPO's eval function).
        Uses same directory structure and utilities as PPO.
        """
        episode_dir = ensure_eval_step_update_dir(
            str(self.training_save_dir),
            update=iteration,
            steps=self.total_env_steps,
            folder_name="eval_results"
        )

        results = []
        for i in range(self.num_eval_runs):
            seed = self.seed + self.eval_seed_offset + i
            metrics = self._run_single_eval_episode(seed)

            # Save routes for this seed
            seed_dir, _ = make_seed_output_dir(episode_dir, seed)
            save_routes_json(seed_dir, metrics['routes'])
            results.append(metrics)

        # Aggregate and save summary
        aggregated = aggregate_results(results)
        write_results_summary(aggregated, self.num_eval_runs, episode_dir, 'eval_results_summary.json')

        # Log to wandb
        if not self.config.get("wandb_off"):
            wandb.log({
                "eval/episode_total_reward": aggregated['episode_total_reward'],
                "eval/episode_length": aggregated['episode_length'],
                "eval/demand_coverage_potential": aggregated['demand_coverage_potential'],
                "eval/demand_coverage_actual": aggregated['demand_coverage_actual'],
                "eval/route_overlap_ratio": aggregated['route_overlap_ratio'],
                "eval/node_coverage": aggregated['node_coverage'],
                "eval/service_rate": aggregated['service_rate'],
                "eval/iteration": iteration,
            }, step=self.total_env_steps)

        return aggregated

    def evaluate(self, policy_path: str, save_dir: str) -> Dict[str, Any]:
        """
        Standalone evaluation entry point (like ppo_eval).

        Args:
            policy_path: Path to saved policy weights
            save_dir: Directory to save results
        """
        # Load policy
        if policy_path and os.path.exists(policy_path):
            self._load_policy(policy_path)
            print(f"Loaded policy from {policy_path}")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create eval directory
        episode_dir = ensure_eval_step_update_dir(
            str(save_path),
            update="final",
            steps=0,
            folder_name="eval_results"
        )

        results = []
        for i in range(self.num_eval_runs):
            seed = self.seed + self.eval_seed_offset + i
            metrics = self._run_single_eval_episode(seed)

            # Save routes for this seed
            seed_dir, _ = make_seed_output_dir(episode_dir, seed)
            save_routes_json(seed_dir, metrics['routes'])
            results.append(metrics)

            print(f"Episode {i + 1}/{self.num_eval_runs}: Reward = {metrics['episode_total_reward']:.2f}")

        # Aggregate and save summary
        aggregated = aggregate_results(results)
        write_results_summary(aggregated, self.num_eval_runs, episode_dir, 'eval_results_summary.json')

        print(f"\nEvaluation Results ({self.num_eval_runs} episodes):")
        print(f"  Mean Reward: {aggregated.get('episode_total_reward', 0):.2f}")
        print(f"  Results saved to: {episode_dir}")

        return aggregated

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict
import numpy as np
from torch.utils.data import DataLoader
from rl.ppo_utils import DatasetClass, Memory, collate_fn


class PPOAgent:
    """
    Proximal Policy Optimization
    """

    def __init__(self, model: nn.Module, **kwargs) -> None:
        """
        """
        self.model = model
        self.device = kwargs.get('device')

        # PPO hyperparameters
        # Note: gamma and gae_lambda are handled by workers (worker-side GAE)
        self.clip_frac = kwargs.get('clip_frac')
        self.K_epochs = kwargs.get('K_epochs')
        self.batch_size = kwargs.get('batch_size')
        self.value_loss_coef = kwargs.get('value_loss_coef')
        self.entropy_coef = kwargs.get('entropy_coef')
        self.max_grad_norm = kwargs.get('max_grad_norm')
        self.vf_clip_param = kwargs.get('vf_clip_param')
        self.lr = kwargs.get('lr')
        self.anneal_lr = kwargs.get('anneal_lr')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = kwargs.get('device')
        self.memory = Memory()

    def store_transition(self, transition: Dict[str, Any]) -> None:
        """Store the transition in rollout memory."""
        self.memory.store(transition)

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        Returns a Dict of training stats.
        
        Expects advantages/returns to be precomputed by workers (worker-side GAE).
        For the choice between KL Divergence vs. Clipping, we use clipping.
        """
        # IMPORTANT: Keep model in eval mode during updates!
        # Rollouts compute log_prob_old with dropout OFF (eval mode).
        # If we enable dropout here (train mode), log_prob_new would differ
        # even with identical weights, breaking the PPO ratio assumption.
        # PPO regularization comes from: entropy bonus, clipping, and value clipping.
        self.model.eval()
        # print(f"[DEBUG] PPO Update: Starting with {len(self.memory)} transitions, K_epochs={self.K_epochs}, batch_size={self.batch_size}")

        # Validate that workers have precomputed advantages/returns
        if len(self.memory.advantages) != len(self.memory) or len(self.memory.returns) != len(self.memory):
            raise ValueError(
                f"Advantages/returns must be precomputed by workers. "
                f"Got {len(self.memory.advantages)} advantages and {len(self.memory.returns)} returns "
                f"for {len(self.memory)} transitions."
            )

        # Normalize advantages ONCE per rollout (before mini-batching), which is standard practice.
        # This reduces variance compared to per-mini-batch normalization.
        if len(self.memory.advantages) > 0:
            adv = torch.tensor(self.memory.advantages, dtype=torch.float32)
            # print(f"[DEBUG] Adv normalization: raw adv shape={adv.shape}, mean={adv.mean():.4f}, std={adv.std():.4f}, min={adv.min():.4f}, max={adv.max():.4f}")
            adv = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)
            # print(f"[DEBUG] Adv normalization: normalized mean={adv.mean():.6f}, std={adv.std():.4f}")
            self.memory.advantages = adv.numpy()
        
        dataset = DatasetClass(self.memory)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,shuffle=True, collate_fn=collate_fn )
        
        # Training stats
        pg_losses, value_losses, entropy_losses = [], [], []
        clipping_frequencies = []
        approx_kls, clip_ratios = [], []
        
        # PPO epochs
        for _ in range(self.K_epochs):
            for batch_data in dataloader:

                obs = batch_data['obs'].to(self.device)
                # print(f"Type of obs: {type(obs)}, value: {obs}")
                actions = batch_data['actions'].to(self.device)
                old_log_probs = batch_data['log_probs'].to(self.device)
                advantages = batch_data['advantages'].to(self.device)
                returns = batch_data['returns'].to(self.device)
                valid_mask = batch_data['valid_mask'].to(self.device)
                
                # print(f"[DEBUG] Mini-batch shapes: obs.x={tuple(obs.x.shape)}, obs.edge_index={tuple(obs.edge_index.shape)}, actions={tuple(actions.shape)}, advantages={tuple(advantages.shape)}, returns={tuple(returns.shape)}")
                
                # print({
                #     "x": tuple(obs.x.shape),
                #     "edge_index": tuple(obs.edge_index.shape),
                #     "edge_attr": tuple(getattr(obs, "edge_attr", torch.empty(0)).shape),
                #     "batch_vec": tuple(obs.batch.shape),
                #     "valid_mask": tuple(valid_mask.shape),
                #     "actions": tuple(actions.shape),
                # })

                # Get current policy outputs
                log_probs, entropy, values = self.model.evaluate(obs.x,          # Node features [sum_n_i, D]
                                                            obs.edge_index,      # Edge index [2, sum_e_i]
                                                            obs.edge_attr,       # Edge features [E, edge_dim]
                                                            obs.batch,
                                                            valid_mask,          # Valid mask [batch_size, max_nodes]
                                                            actions)             # Actions [batch_size]


                # Policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs)
                # print(f"[DEBUG] Ratio stats: shape={tuple(ratio.shape)}, mean={ratio.mean():.4f}, min={ratio.min():.4f}, max={ratio.max():.4f}, old_lp_mean={old_log_probs.mean():.4f}, new_lp_mean={log_probs.mean():.4f}")
                surrogate_loss1 = -advantages * ratio
                surrogate_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_frac, 1 + self.clip_frac)
                surrogate_loss = torch.max(surrogate_loss1, surrogate_loss2).mean()
                
                # Value loss (clipped)
                old_values = batch_data['values'].to(self.device)
                values_clipped = old_values + torch.clamp(values - old_values, -self.vf_clip_param, self.vf_clip_param)
                value_loss_unclipped = (values - returns) ** 2
                value_loss_clipped = (values_clipped - returns) ** 2

                # Value loss is scaled by value loss coeff as well as 0.5 separately
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = entropy.mean()
                approx_kls.append((old_log_probs - log_probs).mean().item())
                clip_ratios.append(ratio.mean().item())

                # Total loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss # Entropy term negated
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track stats
                pg_losses.append(surrogate_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clipping_frequencies.append(((ratio - 1).abs() > self.clip_frac).float().mean().item())
        
        # Clear memory after update
        self.memory.clear()

        # Some explicit memory management
        gc.collect() # Force garbage collection to reclaim memory
        # After the K_epochs loop ends
        del dataset

        # These are average losses over the batch.
        # print(f"[DEBUG] PPO Update: Completed. pg_loss={np.mean(pg_losses):.4f}, value_loss={np.mean(value_losses):.4f}, approx_kl={np.mean(approx_kls):.4f}")
        return {
            'pg_loss': np.mean(pg_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clipping_frequency': np.mean(clipping_frequencies),
            'approx_kl': np.mean(approx_kls),
            'mean_clip_ratio': np.mean(clip_ratios)
        }

    def compute_gae(self) -> None:
        """
        Generalized Advantage Estimation
        NOTE: THIS IS NOT USED RIGHT NOW. 
            - GAE calculation has moved to worker side. 
            - This would have been useful in learner side GAE.
        
        The setup has:
        - a buffer that collects samples till update() is called after a threshold size.
        - rollout truncation at chunk boundaries if enabled.
        However, advantage calculations cannot cross boundaries; we must reset at each episode/chunk end.
        Hence we make use of: 
        - episode_boundaries: indices where episodes/chunks end
        - bootstrap_values: values of the last state of each episode/chunk

        Note (parallel rollouts):
        - This is a fallback path when worker-side GAE is not used.
        - episode_boundaries/bootstrap_values must be populated for each chunk/episode end.
        """

        with torch.no_grad():
            # Get potentially multi-episode data.
            # Use RAW rewards - reward normalization violates PPO assumptions
            # Advantages are normalized once per rollout instead (see update() method)
            all_rewards = torch.tensor(self.memory.raw_rewards, dtype=torch.float32)
            all_values = torch.tensor(self.memory.values, dtype=torch.float32)
            all_dones = torch.tensor(self.memory.dones, dtype=torch.float32)
            all_advantages = torch.zeros_like(all_rewards)

            # Process each episode separately to respect boundaries (GAE resets for each episode)
            episode_start = 0
            for ep_idx, episode_end in enumerate(self.memory.episode_boundaries):
                # Get episode slice
                rewards = all_rewards[episode_start:episode_end + 1]
                values = all_values[episode_start:episode_end + 1]
                dones = all_dones[episode_start:episode_end + 1]

                # Get bootstrap for this episode 
                bootstrap_value = self.memory.bootstrap_values[ep_idx]
                
                # Initialize GAE for this episode
                advantages = torch.zeros_like(rewards)

                # Initialize the running (last) advantage to 0. This is used in the backward recursion.
                # It represents the accumulated discounted future advantages.
                last_advantage = 0
                
                # Reverse iteration for GAE: We compute advantages starting from the end of the trajectory
                # and work backwards. This allows us to efficiently accumulate the discounted sum
                # without recomputing sums for each timestep (O(T) time complexity).
                for step in reversed(range(len(rewards))):
                    
                    # For truncated episodes, we DON'T want to mask the next_value (because we want to use the bootstrap value)
                    # For terminated episodes, next_value is already 0
                    next_non_terminal = 1.0 - dones[step]  # Only mask if terminated

                    # Determine the next state's value estimate (V(s_{t+1})).
                    # For the last step (step == len(rewards) - 1), use the bootstrap value if the episode
                    # was truncated, which estimates potential future rewards beyond the truncation point
                    # If terminated naturally, bootstrap_value should be 0.
                    # For earlier steps, use the critic's value estimate from the next timestep.
                    if step == len(rewards) - 1:
                        next_value = bootstrap_value # For final step: use bootstrap. This value is 0.0 if terminated.
                        # This this is 0.0, this automatically acts as a mask for the next_non_terminal. But still keeping next_non_terminal because is used in the last equation.
                    else:
                        next_value = values[step + 1]

                    # Calculate the TD error (delta) for this timestep. This is Equation (11) from the GAE paper: δ_t = r_{t} + γ * V(s_{t+1}) - V(s_t)
                    # Where:
                    #   - r_t (rewards[step]) is the reward received after action at step t
                    #   - V(s_{t+1}) (next_value) is the value of the next state
                    #   - V(s_t) (values[step]) is the value of the current state
                    #   - The next_non_terminal mask ensures that if the step is done (terminal), we don't add γ * V(s_{t+1}), as there are no future rewards.
                    delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
                    
                    # Compute the GAE for this timestep. This is the recursive form of: Â_t = δ_t + (γ λ) * Â_{t+1} (with masking for done states)
                    # Where Â_{t+1} is the 'last_advantage' from the previous iteration
                    # Unfolding this recursion gives the full sum: Â_t = δ_t + (γ λ) δ_{t+1} + (γ λ)^2 δ_{t+2} + ...
                    # The next_non_terminal mask prevents propagating advantages across episode boundaries.
                    advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

                    # Update running advantage
                    last_advantage = advantages[step]

                # Store to global
                all_advantages[episode_start:episode_end + 1] = advantages

                # Next episode
                episode_start = episode_end + 1

            # Compute returns = advantages + values
            all_returns = all_advantages + all_values

            # Store
            self.memory.advantages = all_advantages.numpy()
            self.memory.returns = all_returns.numpy()

    def update_learning_rate(self, current_step: int, total_steps: int) -> None:
        """
        Update learning rate according to a linear schedule based on training steps.
        """
        if total_steps <= 0:
            return

        progress = current_step / total_steps
        new_lr = self.lr * (1 - progress)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
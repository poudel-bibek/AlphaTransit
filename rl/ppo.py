from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from rl.ppo_utils import DatasetClass, Memory, collate_fn


class PPO:
    def __init__(self, model: nn.Module, **kwargs) -> None:
        """
        """
        self.model = model
        self.device = kwargs.get('device')

        # PPO hyperparameters
        self.clip_frac = kwargs.get('clip_frac')
        self.K_epochs = kwargs.get('K_epochs')
        self.batch_size = kwargs.get('batch_size')
        self.value_loss_coef = kwargs.get('value_loss_coef')
        self.entropy_coef = kwargs.get('entropy_coef')
        self.max_grad_norm = kwargs.get('max_grad_norm')
        self.gae_lambda = kwargs.get('gae_lambda')
        self.gamma = kwargs.get('gamma')
        self.current_timestep = 0
        
        # Learning rate settings
        self.lr = kwargs.get('lr')
        self.lr_schedule = kwargs.get('lr_schedule')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = kwargs.get('device')

        # Memory buffer
        self.memory = Memory()

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        Returns a Dict of training stats
        - Includes GAE
        - For the choice between KL Divergence vs. Clipping, we use clipping.
        """

        self.compute_gae()
        
        dataset = DatasetClass(self.memory)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,shuffle=True, collate_fn=collate_fn )
        
        # Training stats
        pg_losses, value_losses, entropy_losses = [], [], []
        clip_fractions = []
        
        # PPO epochs
        for _ in range(self.K_epochs):
            for batch_data in dataloader:

                obs = batch_data['obs'].to(self.device)
                actions = batch_data['actions'].to(self.device)
                old_log_probs = batch_data['log_probs'].to(self.device)
                advantages = batch_data['advantages'].to(self.device)
                returns = batch_data['returns'].to(self.device)
                
                # Retrieve stored valid_indices
                valid_indices = torch.tensor(batch_data['valid_indices'], dtype=torch.long, device=self.device)
                
                # Get current policy outputs
                log_probs, entropy, values = self.model.evaluate(obs, actions, steps_left=obs.steps_left, valid_indices=valid_indices)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_frac, 1 + self.clip_frac)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss (clipped)
                values_clipped = batch_data['values'].to(self.device) + torch.clamp(
                    values - batch_data['values'].to(self.device), -self.clip_frac, self.clip_frac
                )
                value_loss1 = (values - returns) ** 2
                value_loss2 = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = pg_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track stats
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(((ratio - 1).abs() > self.clip_ratio).float().mean().item())
        
        # Clear memory after update
        self.memory.clear()
        
        return {
            'pg_loss': np.mean(pg_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions)
        }

    def compute_gae(self) -> None:
        """
        Compute Generalized Advantage Estimation for trajectories.
        Updates advantages and returns in memory.

        """

        with torch.no_grad():
            rewards = torch.tensor(self.memory.rewards, dtype=torch.float32)
            values = torch.tensor(self.memory.values, dtype=torch.float32)
            dones = torch.tensor(self.memory.dones, dtype=torch.float32)
            
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            
            # Reverse iteration for GAE
            for step in reversed(range(len(rewards))):

                if step == len(rewards) - 1:
                    # Use bootstrap value for final state
                    next_value = self.memory.bootstrap_value
                else:
                    next_value = values[step + 1]
                
                # For each step, we calculate the TD error (delta). Equation 12 in the paper. delta = r + γV(s') - V(s)
                delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
                
                # Equation 11 in the paper. GAE(t) = δ(t) + (γλ)δ(t+1) + (γλ)²δ(t+2) + ...
                advantages[step] = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * last_advantage

                last_advantage = advantages[step]
            
            returns = advantages + values
            
            # Store computed values
            self.memory.advantages = advantages.numpy()
            self.memory.returns = returns.numpy()

    def update_learning_rate(self) -> None:
        """
        Update learning rate according to schedule.
        Supports linear and constant schedules.
        """
        if self.lr_schedule == 'linear':
            progress = self.current_timestep / self.total_timesteps
            new_lr = self.lr * (1 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        # Constant schedule does nothing
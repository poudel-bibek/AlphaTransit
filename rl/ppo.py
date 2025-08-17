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
        
        # PPO hyperparameters
        self.clip_ratio = kwargs.get('clip_ratio')
        self.ppo_epochs = kwargs.get('ppo_epochs')
        self.mini_batch_size = kwargs.get('mini_batch_size')
        self.value_loss_coef = kwargs.get('value_loss_coef')
        self.entropy_coef = kwargs.get('entropy_coef')
        self.max_grad_norm = kwargs.get('max_grad_norm')
        self.gae_lambda = kwargs.get('gae_lambda')
        self.gamma = kwargs.get('gamma')
        self.current_timestep = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = kwargs.get('device')

        # Learning rate settings
        self.lr = kwargs.get('learning_rate')
        self.lr_schedule = kwargs.get('lr_schedule')

        # Memory buffer
        self.memory = Memory()

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        
        Returns:
            Dictionary of training statistics
        """
        
        self.compute_gae()
        
        dataset = DatasetClass(self.memory)
        dataloader = DataLoader(dataset, batch_size=self.mini_batch_size,shuffle=True, collate_fn=collate_fn )
        
        # Training stats
        pg_losses, value_losses, entropy_losses = [], [], []
        clip_fractions = []
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            for batch in dataloader:
                obs = batch['obs'].to(self.model.device)
                actions = batch['actions'].to(self.model.device)
                old_log_probs = batch['log_probs'].to(self.model.device)
                advantages = batch['advantages'].to(self.model.device)
                returns = batch['returns'].to(self.model.device)
                
                # Get current policy outputs
                log_probs, entropy, values = self.model.evaluate(obs, actions, steps_left=obs.steps_left)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss (clipped)
                values_clipped = batch['values'].to(self.model.device) + torch.clamp(
                    values - batch['values'].to(self.model.device), -self.clip_ratio, self.clip_ratio
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
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0  # Assuming terminal state
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
                last_advantage = advantages[t]
            
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
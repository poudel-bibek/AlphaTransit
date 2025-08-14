from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch


class GATV2ActorCritic(nn.Module):
    """
    Actor-Critic with GATv2 backbone for graph-based RL.
    Shared GATv2 features with separate actor/critic heads.
    """

    def __init__(
        self, 
        node_feature_dim: int, 
        num_actions: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize GATv2 backbone and actor-critic heads.
        
        Args:
            node_feature_dim: Input node feature dimension
            num_actions: Number of possible nodes to select
            hidden_dim: Hidden dimension for GATv2 layers
            num_heads: Number of attention heads
            num_layers: Number of GATv2 layers
            dropout: Dropout probability
        """
        super().__init__()
        self.num_actions = num_actions
        
        # Build GATv2 backbone
        self.gat_layers = nn.ModuleList()
        in_dim = node_feature_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim // num_heads if i < num_layers - 1 else hidden_dim
            self.gat_layers.append(
                GATv2Conv(
                    in_dim if i == 0 else hidden_dim,
                    out_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=i < num_layers - 1
                )
            )
            in_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        
        # Actor head for node selection
        self.actor_net = nn.Sequential(
            self.layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)
        )
        
        # Critic head for value estimation
        self.critic_net = nn.Sequential(
            self.layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

        
    def _apply_action_mask(self,) -> None:
        """
        """

        MASK_VALUE = -1e10 # large negative 
        pass
    
    def layer_init(self, layer: nn.Linear, std: float = 1.0) -> nn.Linear:
        """
        Initialize layer with orthogonal weights and zero bias.
        Standard PPO initialization scheme.
        """
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, 0.0)
        return layer

    def forward(self, graph_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GATv2 backbone.
        Returns node-level features and batch assignment.
        """
        x, edge_index = graph_batch.x, graph_batch.edge_index
        
        # Pass through GATv2 layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        return x, graph_batch.batch

    def readout_layer(self, graph_batch: Batch) -> torch.Tensor:
        """
        Aggregate node embeddings into graph-level features.
        Flexible pooling - can easily swap to max, sum, or attention pooling.
        """
        node_features, batch = self.forward(graph_batch)
        # Can easily change to global_max_pool, global_add_pool, etc.
        graph_features = global_mean_pool(node_features, batch)
        return graph_features

    def actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map features to action logits for node selection.
        Returns unnormalized logits over candidate nodes.
        """
        return self.actor_net(features)

    def critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map features to state-value estimates.
        Returns scalar value per graph.
        """
        return self.critic_net(features).squeeze(-1)

    def act(self, observations: Batch, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select next node to add to route.
        Samples stochastically during training, can be deterministic for eval.
        
        Returns:
            actions: Selected node indices
            log_probs: Log probabilities of selections
            values: Value estimates
        """
        features = self.readout_layer(observations)
        values = self.critic(features)
        logits = self.actor(features)
        
        # Mask invalid nodes if provided
        if hasattr(observations, 'action_mask'):
            logits = logits.masked_fill(~observations.action_mask, -1e8)
        
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = logits.argmax(-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs, values

    def evaluate(self, observations: Batch, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-probs, entropy, and values for PPO loss.
        
        Returns:
            log_probs: Log probabilities of selected nodes
            entropy: Policy entropy for exploration
            values: Value estimates
        """
        features = self.readout_layer(observations)
        values = self.critic(features)
        logits = self.actor(features)
        
        # Mask invalid nodes if provided
        if hasattr(observations, 'action_mask'):
            logits = logits.masked_fill(~observations.action_mask, -1e8)
        
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values
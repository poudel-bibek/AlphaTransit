from typing import Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch


class GATV2ActorCritic(nn.Module):
    """
    Actor-Critic with shared GATv2 features with separate actor/critic heads.
    - GATv2 processes graph-structured data (nodes, edges) to produce node embeddings, 
      which are then aggregated into a graph-level features in readout layer, where we have a fixed sized vector.
    - The steps_left is a scalar that applies to the whole graph (global feature), which we concatentate after the graph-level features..
    """

    def __init__( self, num_actions: int, **kwargs) -> None:
        """
        GATv2 backbone and actor-critic heads.
        
        Args:
            node_feature_dim: used as in_channels in GATv2Conv layers
            num_actions: Number of total nodes in the network
            **kwargs: 
                num_layers: Number of GATv2 layers (int)
                gat_channels: Hidden dimension for GATv2 layers (list: [n_node_features, hidden_channels 1, hidden_channels 2, ..., output_dimension]) 
                    - The first element is the number of node features.
                    - The last element is the output dimension.
                num_heads: Number of attention heads for each GATv2 layer (list: [num_heads 1, num_heads 2, ...])
                num_edge_features: edge features (int)
                dropout: Dropout probability (float)
                global_dim: size of steps_left (int)
                activation: activation function (str: "elu", "tanh", "leaky_relu", "relu")
                model_size: choice for model size (str: "small", "medium")

        Notes: 
        - No shared MLP layers between actor and critic.
        - A custom layer init is used for GATv2 layers.
        
        # TODO: 
        - Batch norm, layer norm, etc.
        """
        super().__init__()
        self.num_actions = num_actions
        self.num_layers = kwargs.get("num_layers")
        self.gat_channels = kwargs.get("gat_channels")
        
        # Why heads = 1 in the last layer?
        # Multi-head attention is used in the earlier layers to capture different aspects of the graph and the final layer consolidates this info.
        self.num_heads = kwargs.get("num_heads")
        self.num_edge_features = kwargs.get("num_edge_features")
        self.dropout = kwargs.get("dropout")
        self.global_dim = kwargs.get("global_dim") 
        
        self.activation = kwargs.get("activation")
        if self.activation == "elu":
            self.activation = nn.ELU()
        elif self.activation == "tanh":
            self.activation = nn.Tanh()
        elif self.activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif self.activation == "relu":
            self.activation = nn.ReLU()
        
        model_size = kwargs.get("model_size")
        if model_size == "small":
            actor_sizes = [128, 64]
            critic_sizes = [128, 64]
        elif model_size == "medium":
            actor_sizes = [256, 128, 64]
            critic_sizes = [256, 128, 64]

        self.concat = kwargs.get("concat")

        # Build GATv2 layers
        # For a 2 layer setup:
        # First layer should output [num_nodes, hidden_gat_dims[0]*num_heads[0]] (concat = False)
        # First layer should output [num_nodes, hidden_gat_dims[0]] (concat = True)
        # concat = True by default: the output from different attn heads are concatenated to the output of size hidden_gat_dims[0]*num_heads[0]
        # When concat = False, the output is the average of the attn heads.
        # Concat reduces dimensionality, but loses too much information.

        # Second layer should output [num_nodes, hidden_gat_dims[1]*num_heads[1]] (concat = False)
        # Second layer should output [num_nodes, hidden_gat_dims[1]] (concat = True)
        gat_layers = []
        for i in range(self.num_layers):
            if self.concat:
                in_dim = self.gat_channels[0] if i == 0 else self.gat_channels[i]*self.num_heads[i-1]
            else:
                in_dim = self.gat_channels[i]

            out_dim = self.gat_channels[i+1]
            
            conv = GATv2Conv(
                in_channels = in_dim,
                out_channels = out_dim,
                heads=self.num_heads[i],
                dropout=self.dropout,
                concat=self.concat,
                edge_dim=self.num_edge_features 
            )
            conv = self.gat_layer_init(conv)  # Apply custom init
            gat_layers.append(conv)

        self.gat_layers = nn.ModuleList(gat_layers)  # Use ModuleList to properly register modules

        if self.concat:
            mlp_input = self.gat_channels[-1]*self.num_heads[-1] + self.global_dim
        else:
            mlp_input = self.gat_channels[-1] + self.global_dim

        # Actor 
        actor_layers = []
        actor_input = mlp_input
        for j in range(len(actor_sizes)):
            actor_layers.append(self.layer_init(nn.Linear(actor_input, actor_sizes[j])))
            # Add layer norm, batch norm, dropout, etc.
            # actor_layers.append(nn.LayerNorm(actor_sizes[j]))
            actor_layers.append(self.activation)
            actor_input = actor_sizes[j]

        # Actor output layer with smaller gain (make actor start with close to uniform distribution i.e. not too confident on any particular action)
        actor_layers.append(self.layer_init(nn.Linear(actor_sizes[-1], self.num_actions), std=0.01))
        self.actor_net = nn.Sequential(*actor_layers)  # Changed from self.actor

        # Critic 
        critic_layers = []
        critic_input = mlp_input
        for k in range(len(critic_sizes)):
            critic_layers.append(self.layer_init(nn.Linear(critic_input, critic_sizes[k])))
            # Add layer norm, batch norm, dropout, etc.
            # critic_layers.append(nn.LayerNorm(critic_sizes[k]))
            critic_layers.append(self.activation)
            critic_input = critic_sizes[k]

        # Critic output layer with gain 1 (slightly smaller than sqrt(2))
        critic_layers.append(self.layer_init(nn.Linear(critic_sizes[-1], 1), std=1.0))
        self.critic_net = nn.Sequential(*critic_layers)  # Changed from self.critic
    
    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        """
        Orthogoal initialization of weights and Constant initialization of biases.
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def gat_layer_init(self, conv: GATv2Conv, std: float = np.sqrt(2), bias_const: float = 0.0) -> GATv2Conv:
        """
        Custom orthogonal initialization for GATv2Conv internal weights.
        """
        # Initialize left and right linear projections
        nn.init.orthogonal_(conv.lin_l.weight, std)
        nn.init.constant_(conv.lin_l.bias, bias_const)
        
        nn.init.orthogonal_(conv.lin_r.weight, std)
        nn.init.constant_(conv.lin_r.bias, bias_const)
        
        # Initialize attention weights - treat as matrix
        nn.init.orthogonal_(conv.att, std)
        nn.init.constant_(conv.bias, bias_const)
        
        return conv

    def forward(self, graph_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GAT layers to produce node embeddings.
        
        Args:
            graph_batch: PyG Batch containing node features, edge_index, edge_attr
        
        Returns:
            node_embeddings: Final node representations [num_nodes, final_dim]
            batch: Batch assignment for nodes (for pooling)
        """
        x = graph_batch.x  # Node features
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        
        # Pass through GAT layers
        for i, conv in enumerate(self.gat_layers):
            x = conv(x, edge_index, edge_attr=edge_attr) # Inject edge features at each GATv2 layer
            
            # Apply activation except for last layer
            if i < len(self.gat_layers) - 1:
                x = self.activation(x)
        
        return x, batch
        

    def readout_layer(self, graph_batch: Batch, steps_left: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate node embeddings into graph-level features.
        Flexible pooling - can easily swap to max, sum, or attention pooling.
        """
        node_features, batch = self.forward(graph_batch)
        # Can easily change to global_max_pool, global_add_pool, etc.
        graph_features = global_mean_pool(node_features, batch)
        
        # Concatenate global features (steps_left)
        print(f"graph_features.shape: {graph_features.shape}, steps_left.shape: {steps_left.shape}")
        graph_features = torch.cat([graph_features, steps_left.view(-1, 1)], dim=1)
        
        return graph_features

    def act(self, graph_batch: Batch, steps_left: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select next node to add to route.
        Samples stochastically during training, can be deterministic for eval.
        
        Returns:
            actions: Selected node indices
            log_probs: Log probabilities of selections
            values: Value estimates
        """
        features = self.readout_layer(graph_batch, steps_left)
        logits = self.actor_net(features)
        values = self.critic_net(features).squeeze(-1)
        
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = logits.argmax(-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs, values

    def evaluate(self, graph_batch: Batch, actions: torch.Tensor, steps_left: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-probs, entropy, and values for PPO loss.
        
        Returns:
            log_probs: Log probabilities of selected nodes
            entropy: Policy entropy for exploration
            values: Value estimates
        """
        features = self.readout_layer(graph_batch, steps_left)
        logits = self.actor_net(features)
        values = self.critic_net(features).squeeze(-1)
        
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values

    def param_count(self):
        """
        """
        shared_gat_params = sum(p.numel() for layer in self.gat_layers for p in layer.parameters() if p.requires_grad)
        actor_params = sum(p.numel() for p in self.actor_net.parameters() if p.requires_grad)  # Changed to actor_net
        critic_params = sum(p.numel() for p in self.critic_net.parameters() if p.requires_grad)  # Changed to critic_net
        total_params = shared_gat_params + actor_params + critic_params

        return {
            "shared_gat": shared_gat_params,
            "actor": actor_params,
            "critic": critic_params,
            "Grand total": total_params
        }
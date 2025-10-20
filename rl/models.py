from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch


class GATV2ActorCritic(nn.Module):
    """
    Actor-Critic with shared GATv2 features with separate actor/critic heads.
    - GATv2 processes graph-structured data (nodes, edges) to produce node embeddings, 
      which are then aggregated into a graph-level features in readout layer, where we have a fixed sized vector.
    - The route_progress is a vector that applies to the whole graph (global feature), which we concatentate after the graph-level features..
    
    To enable stacking without issues like vanishing gradients, we adopt a number of things for good "hygiene":
    1. Residual connections. 
    2. Layer normalization.
    3. Dropout.
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
                global_dim: size of route_progress (float array)
                activation: activation function (str: "elu", "tanh", "leaky_relu", "relu")

        Notes: 
        - No shared MLP layers between actor and critic.
        - A custom layer init is used for GATv2 layers.
        """
        super().__init__()
        self.num_actions = num_actions  # n_nodes + 1 (for the NO_VALID_ACTION)
        self.n_nodes = kwargs.get("n_nodes")
        self.num_layers = kwargs.get("num_layers")
        self.gat_channels = kwargs.get("gat_channels") # [in_dim, h1, ..., hL]
        # Why heads = 1 in the last layer?
        # Multi-head attention is used in the earlier layers to capture different aspects of the graph and the final layer consolidates this info.
        self.num_heads = kwargs.get("num_heads") # [heads1, ..., headsL]
        self.num_edge_features = kwargs.get("num_edge_features")
        self.dropout = kwargs.get("dropout")
        self.global_dim = kwargs.get("global_dim") 
        self.concat = kwargs.get("concat")

        act = kwargs.get("activation")
        if act == "elu":
            self.activation = nn.ELU()
        elif act == "tanh":
            self.activation = nn.Tanh()
        elif act == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif act == "relu":
            self.activation = nn.ReLU()
        
        # Only include the hidden dimensions.
        actor_sizes = [512, 256] # [128, 64] 
        critic_sizes = [256, 64] # [128, 64] 

        # Build GATv2 layers with residuals, LayerNorm, and dropout.
        # E.g., For a 2 layer setup:
        # First layer should output [num_nodes, hidden_gat_dims[0]*num_heads[0]] (concat = False)
        # First layer should output [num_nodes, hidden_gat_dims[0]] (concat = True)
        # concat = True by default: the output from different attn heads are concatenated to the output of size hidden_gat_dims[0]*num_heads[0]
        # When concat = False, the output is the average of the attn heads.
        # Concat reduces dimensionality, but loses too much information.
        # Second layer should output [num_nodes, hidden_gat_dims[1]*num_heads[1]] (concat = False)
        # Second layer should output [num_nodes, hidden_gat_dims[1]] (concat = True)
        
        gat_layers = []
        res_projs = []
        norms = []

        for i in range(self.num_layers):
            if i == 0:
                # First layer always takes original node features
                in_dim = self.gat_channels[0]
            else:
                # Subsequent layers depend on concat setting
                if self.concat:
                    # Previous layer output: out_channels * num_heads
                    in_dim = self.gat_channels[i] * self.num_heads[i-1]
                else:
                    # Previous layer output: out_channels (averaged across heads)
                    in_dim = self.gat_channels[i]
            
            # Output dim per head and effective out dim after concat or mean
            out_dim = self.gat_channels[i+1]
            out_eff = out_dim * self.num_heads[i] if self.concat else out_dim
            
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

            # Residual projection if shapes differ.
            if in_dim != out_eff:
                res_projs.append(self.layer_init(nn.Linear(in_dim, out_eff)))
            else: 
                res_projs.append(nn.Identity())
            
            # Layer norm on the residual sum
            norms.append(nn.LayerNorm(out_eff))
            
        self.gat_layers = nn.ModuleList(gat_layers)  # Use ModuleList to properly register modules
        self.res_projs = nn.ModuleList(res_projs)
        self.norms = nn.ModuleList(norms)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Actor 
        if self.concat:
            actor_input = self.gat_channels[-1]*self.num_heads[-1]*self.n_nodes + self.global_dim
        else:
            actor_input = self.gat_channels[-1]*self.n_nodes + self.global_dim
        
        actor_layers = []
        for hs in actor_sizes:
            actor_layers.append(self.layer_init(nn.Linear(actor_input, hs)))
            actor_layers.append(self.activation)
            actor_input = hs

        # Actor output layer with smaller gain (make actor start with close to uniform distribution i.e. not too confident on any particular action)
        actor_layers.append(self.layer_init(nn.Linear(actor_sizes[-1], self.num_actions), std=0.01))
        self.actor_net = nn.Sequential(*actor_layers)  # Changed from self.actor

        # Critic 
        if self.concat:
            critic_input = self.gat_channels[-1] * self.num_heads[-1] + self.global_dim
        else:
            critic_input = self.gat_channels[-1] + self.global_dim
        
        critic_layers = []
        for hs in critic_sizes:
            critic_layers.append(self.layer_init(nn.Linear(critic_input, hs)))
            critic_layers.append(self.activation)
            critic_input = hs

        # Critic output layer with gain 1 (slightly smaller than sqrt(2))
        critic_layers.append(self.layer_init(nn.Linear(critic_sizes[-1], 1), std=1.0))
        self.critic_net = nn.Sequential(*critic_layers)  # Changed from self.critic

    def _mask_logits(self, logits: torch.Tensor, valid_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Mask invalid actions by setting their logits to -inf.
        Handles batches: valid_indices [batch, max_valid] padded with -1.
        Ignores -1 during masking.

        If an empty valid_indices is passed, everything will be masked.
        So, don't pass empty valid_indices.
        """
        if valid_indices is None:
            return logits

        MASK_VALUE = -1e9  # Large negative instead of -inf to avoid NaNs
        mask = torch.zeros_like(logits, dtype=torch.bool)

        batch_size = logits.shape[0]
        for b in range(batch_size):
            valid = valid_indices[b]
            valid = valid[valid >= 0]  # Ignore padding
            if len(valid) > 0:
                mask[b, valid] = True

        # print("[DEBUG] Original logits:\n", logits)
        # print("[DEBUG] Valid indices:\n", valid_indices)
        # print("[DEBUG] Mask (True for valid):\n", mask)
        masked = logits.masked_fill(~mask, MASK_VALUE)
        # print("[DEBUG] Masked logits:\n", masked)
        return masked

    def gat_forward(self, graph_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GAT layers
        batch: Batch assignment for nodes (for pooling)
        """
        x = graph_batch.x  # Node features
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr

        # print(f"[DEBUG] Node features before GAT layers: {x.shape}")
        # Pass through GAT layers
        for i, conv in enumerate(self.gat_layers):
            identity = x 
            x = conv(x, edge_index, edge_attr=edge_attr)  # Inject edge features at each GATv2 layer
            # print(f"[DEBUG] After GAT layer {i}, node features:\n{x.shape}")
            # Apply activation except for last layer
            if i < len(self.gat_layers) - 1:
                x = self.activation(x)
            x = self.dropout_layer(x)
            x = x + self.res_projs[i](identity)
            x = self.norms[i](x)
        return x

    def _compute_dist_and_value(self, 
                                graph_batch: Batch, 
                                valid_indices: Optional[torch.Tensor]
                                ) -> Tuple[Categorical, torch.Tensor]:
        """
        Returns a dummy action (-1) if episode is truncated. Still need to return a value in this case.
        """
        # Compute GAT features once
        node_features = self.gat_forward(graph_batch)
        # print(f"[DEBUG] Node features - has NaN: {torch.isnan(node_features).any()}, shape: {node_features.shape}")
        
        # Use readout functions with shared node features
        actor_features = self.actor_readout(node_features, graph_batch)
        # print(f"[DEBUG] Actor features - has NaN: {torch.isnan(actor_features).any()}, shape: {actor_features.shape}")
        
        logits = self.actor_net(actor_features)
        # print(f"[DEBUG] Logits - has NaN: {torch.isnan(logits).any()}, shape: {logits.shape}")
        
        critic_features = self.critic_readout(node_features, graph_batch)
        values = self.critic_net(critic_features).squeeze(-1)
        
        masked_logits = self._mask_logits(logits, valid_indices)
        # print(f"[DEBUG] Masked logits - has NaN: {torch.isnan(masked_logits).any()}, shape: {masked_logits.shape}")
        dist = Categorical(logits=masked_logits)
        # print("[DEBUG] Distribution probs:\n", dist.probs)
        return dist, values

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
        Potential safety issue: Direct access to conv.att (3D tensor) for orthogonal init may be inappropriate for attention weights.
        Safer alternative: Avoid touching GAT internals at init (use PyG default).
        """
        # Safer approach: Only touch clearly linear 2D weights and biases by name
        for name, param in conv.named_parameters(recurse=False):
            if "lin" in name and param.dim() >= 2:
                print(f"Initializing {name}: shape={param.shape}")
                nn.init.orthogonal_(param, std)
            elif "bias" in name and param is not None and param.dim() == 1:
                print(f"Initializing {name}: shape={param.shape}")
                nn.init.constant_(param, bias_const)
        return conv

    def actor_readout(self, node_features: torch.Tensor, graph_batch: Batch) -> torch.Tensor:
        """
        Readout: Aggregate node embeddings into graph-level features i.e., a way to get a fixed sized vector for the graph.
        However, in this case we don't have a variable sized input (current routes are embedded as node features) 

        Approaches: 
            1. Global mean pooling: 
            - Average the node features across all nodes in the graph.
            - Example: node 1 features: [0.1, 0.2, 0.3], node 2: [0.4, 0.5, 0.6], node 3: [0.7, 0.8, 0.9]
            - After pooling: Global node features = [0.4, 0.5, 0.6]
            - Disadvantage: Severe loss of information.

            2. Alternative: Global sort pooling:
            - Sort the nodes according to their mean activation then take the top k nodes.
            
            3. Alternative: No pooling in Actor + Global mean pooling in Critic (Currently used).
            - Just concatenate all node features from the last GATv2 layer.
            - Justification: 
                - Fundamentally, the task for the policy actor is `next node selection`.
                - Global mean pooling reduces information too much, and forces the MLP to reconstruct node-specific probabilities from an average.
                - This is likely suboptimal for node-selection task.
                - If we remove pooling, actor logits would be computed directly from each node's embedding. 
                - This avoids the averaging loss, allowing better differentiation between nodes/actions.
                - At the same time, perhaps the critic benefits from pooling, as value estimation is inherently graph-level.
                - route_progress is a global feature (not broadcasted to each node); maybe more important to the critic.
            - Cons: 
                - Changing node order changes the concatenated vector (Same graph, different node order = different results). 
                    - However, for a given graph, the node order is fixed.
                - Parameter count is higher because the MLP at input is larger.
                    - No big deal.
                - Probably the most prominent con of this approach is that this is not scalable to large networks.
                    - When the number of nodes is large, the parameter count for actor MLP head is too high.

            4. Alternative: Validity-Weighted Attention Readout
            
        """
        # 1. Global mean pooling
        # graph_features = global_mean_pool(node_features, batch)
        # graph_features = torch.cat([graph_features, route_progress.view(-1, 1)], dim=1) # Concatenate global features (route_progress)
        
        # 3. No pooling.
        # Flatten all node features per graph
        batch_size = len(torch.unique(graph_batch.batch))  # Number of graphs in batch
        emb_dim = node_features.shape[1]
        num_nodes = node_features.shape[0] // batch_size  # Assumes fixed num_nodes per graph
        # Reshape to [batch_size, num_nodes * emb_dim]
        graph_features = node_features.view(batch_size, num_nodes * emb_dim)
        print(f"[DEBUG] Actor graph features shape: {graph_features.shape}")

        route_progress = graph_batch.route_progress
        # Route progress shape handling:
        # - Single observation: [NUM_ROUTES] -> add batch dim -> [1, NUM_ROUTES]
        # - Batched observations: [batch_size, NUM_ROUTES] (already correct)
        if route_progress.dim() == 1:
            route_progress = route_progress.unsqueeze(0)
        # print(f"[DEBUG] Route progress shape: {route_progress.shape}")

        # Concatenate global features (route_progress)
        graph_features = torch.cat([graph_features, route_progress], dim=1)
        # print(f"[DEBUG] Actor graph features shape after concatenation: {graph_features.shape}")

        return graph_features

    def critic_readout(self, node_features: torch.Tensor, graph_batch: Batch) -> torch.Tensor:
        """
        Critic readout: Global mean pooling for graph-level value estimation.
        """
        # 1. Global mean pooling 
        graph_features = global_mean_pool(node_features, graph_batch.batch)
        
        # print(f"[DEBUG] Critic pooled features shape: {graph_features.shape}")
        
        route_progress = graph_batch.route_progress
        # Route progress shape handling:
        # - Single observation: [NUM_ROUTES] -> add batch dim -> [1, NUM_ROUTES]
        # - Batched observations: [batch_size, NUM_ROUTES] (already correct)
        if route_progress.dim() == 1:
            route_progress = route_progress.unsqueeze(0)
        # print(f"[DEBUG] Route progress shape: {route_progress.shape}")

        # Concatenate global features (route_progress)
        graph_features = torch.cat([graph_features, route_progress], dim=1)
        # print(f"[DEBUG] Critic graph features shape after concatenation: {graph_features.shape}")

        return graph_features
    
    def act(self, graph_batch: Batch, 
            deterministic: bool = False, 
            valid_indices: Optional[torch.Tensor] = None, 
            truncated: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select next node to add to route.
        Samples stochastically during training, can be deterministic for eval.
        Returns:
            actions: Selected node indices
            log_probs: Log probabilities of selections
            values: Value estimates
        """
        
        # # Handle truncation
        # if truncated:
        #     node_features = self.gat_forward(graph_batch)
        #     critic_features = self.critic_readout(node_features, graph_batch)
        #     values = self.critic_net(critic_features).squeeze(-1)
        #     dummy_action = torch.tensor([-1], device=graph_batch.x.device)
        #     dummy_log_prob = torch.tensor([0.0], device=graph_batch.x.device)
        #     return dummy_action, dummy_log_prob, values
        
        dist, values = self._compute_dist_and_value(graph_batch, valid_indices)
        
        if deterministic:
            actions = dist.logits.argmax(-1)
            print(f"\n\tAction (Deterministic): {actions}\n")
        else:
            actions = dist.sample()
            print(f"\n\tAction (Stochastic): {actions}\n")
        
        log_probs = dist.log_prob(actions)
        # print(f"[DEBUG] Log probs: {log_probs}")
        return actions, log_probs, values
    
    def evaluate(self, graph_batch: Batch, actions: torch.Tensor, valid_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-probs, entropy, and values for PPO loss.
        
        Returns:
            log_probs: Log probabilities of selected nodes
            entropy: Policy entropy for exploration
            values: Value estimates
        """
        
        dist, values = self._compute_dist_and_value(graph_batch, valid_indices)
        
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
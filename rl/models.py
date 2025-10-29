import math
import random
import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical

def count_module(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "elu": return nn.ELU()
    if name == "leaky_relu": return nn.LeakyReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    raise ValueError(f"Unknown activation '{name}'")

def _gain_from_activation(name: str) -> float:
    """
    Calculate the gain for the activation function.
    - For ReLU, the gain is sqrt(2).
    - For LeakyReLU, the gain is sqrt(2 / (1 + negative_slope^2)).
    - For Tanh, the gain is 5/3.
    - For GELU and ELU, the gain is 1.0.
    """
    name = name.lower()
    if name == "relu":
        return nn.init.calculate_gain("relu")
    if name == "leaky_relu":
        return nn.init.calculate_gain("leaky_relu", 0.01)
    if name == "tanh":
        return nn.init.calculate_gain("tanh")
    if name in {"gelu", "elu"}:
        return 1.0
    return 1.0

class GATBlock(nn.Module):
    def __init__(self, **kwargs):
        """
        - Using two separate dropouts.
            - First inside GATv2Conv (randomly drop attention coefficients)
            - Second after each GATv2Conv block (randomly drop connections between GAT blocks)
        """
        super().__init__()
        
        self.concat = kwargs["concat"]
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.heads = kwargs["heads"]

        self.conv = GATv2Conv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,         # Per head
            heads = self.heads,
            dropout = kwargs["attn_dropout"],         # First dropout
            concat = self.concat,                     # False means output dim = hidden channels, if True, output dim = hidden channels * heads
            edge_dim = kwargs["edge_dim"],            # Edge features
            add_self_loops = False,                   # In our case, edges have no self-loops.
        )

        self.norm = nn.LayerNorm(self.in_channels)    # pre norm on input width
        self.feat_dropout = nn.Dropout(kwargs["feat_dropout"]) # Second dropout
        self.activation = kwargs["activation"]

        # Based on the concat setting, we will either project the output 
        self._eff_out = self.out_channels * self.heads if self.concat else self.out_channels
        self.residual = nn.Identity() if self.in_channels == self._eff_out else nn.Linear(self.in_channels, self._eff_out)
    
    def eff_out(self):
        """
        Function name is eff_out, variable name is _eff_out.
        """
        return self._eff_out

    def forward(self, x, edge_index, edge_attr):
        """
        Each GAT block will have a residual skip connection. 
        Pre-normalization:
        - Normalize the input of the block then conv, activation, dropout
        - Project the output to the effective output dimension then then add as residual.
        """
        xn = self.norm(x)
        h = self.conv(xn, edge_index, edge_attr)
        h = self.activation(h)   # Activation goes after normalization, before dropout
        h = self.feat_dropout(h)
        return self.residual(x) + h # residual added 

class GATV2ActorCritic(nn.Module):
    """
    Actor-Critic with shared GATv2 backbone that produces node embeddings.
    - Actor is permutation equivariant (decisions per node, output = logits per node).
    - Critic is permutation invariant (decisions per graph, output = one scalar value per graph).
    - Size of policy network is independent of the size of the graph (n_nodes).
    - Before GAT blocks: Project node features
    """
    def __init__(self, **kwargs):
        super().__init__()

        # Projection
        self.n_node_features = kwargs['n_node_features']
        self.proj_out = kwargs['proj_out']

        # GAT blocks
        self.num_gat_blocks = kwargs['num_gat_blocks']
        self.gat_channels = kwargs['gat_channels']
        self.num_heads = kwargs['num_heads']
        self.n_edge_features = kwargs['n_edge_features']
        self.activation_name = kwargs['activation']
        self.activation = make_activation(self.activation_name)
        self.attn_dropout = kwargs['attn_dropout']
        self.feat_dropout = kwargs['feat_dropout']
        self.actor_head_dropout = kwargs['actor_head_dropout']
        self.critic_head_dropout = kwargs['critic_head_dropout']
        self.concat = kwargs['concat']

        # A/ C
        self.actor_head_layers = kwargs['actor_head_layers']
        self.critic_head_layers = kwargs['critic_head_layers']

        # Sanity checks
        assert len(self.gat_channels) == self.num_gat_blocks
        assert len(self.num_heads) == self.num_gat_blocks
        assert len(self.attn_dropout) == self.num_gat_blocks
        assert len(self.feat_dropout) == self.num_gat_blocks

        # Layers
        # Node features projection
        self.proj = nn.Linear(self.n_node_features, self.proj_out)

        # GAT blocks
        blocks = []
        in_dim = self.proj_out
        for i in range(self.num_gat_blocks):
            
            block = GATBlock(
                in_channels = in_dim,
                out_channels = self.gat_channels[i],
                heads = self.num_heads[i],
                attn_dropout = self.attn_dropout[i],
                feat_dropout = self.feat_dropout[i],
                activation = self.activation,
                edge_dim = self.n_edge_features,
                concat = self.concat,
            )
            blocks.append(block)
            # Next block input is going to be based on the effective output dimension
            in_dim = block.eff_out()

        self.gat_blocks = nn.ModuleList(blocks)
        # The output dim after layers (GAT blocks) going to be in_dim.
        self.backbone_out = in_dim

        # Actor head
        self.actor_head = self._make_mlp(self.actor_head_layers, 1, self.actor_head_dropout)

        # Critic head
        self.critic_head = self._make_mlp(self.critic_head_layers, 1, self.critic_head_dropout)

    def _make_mlp(self, layers: List[int], out_dim: int, dropout: float) -> nn.Sequential:
        """
        Make a MLP with the given layers and dropout.
        """
        dims = [self.backbone_out] + list(layers) + [out_dim]
        mlp = []
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp.append(self.activation)
                if dropout > 0:
                    mlp.append(nn.Dropout(dropout))
        return nn.Sequential(*mlp)

    def _get_node_embeddings(self, x, edge_index, edge_attr):
        """
        Get the node embeddings from the GAT blocks.
        """
        # Input projection.
        x = self.proj(x) # [n_nodes, proj_out]
        # GAT stack 
        for block in self.gat_blocks:
            x = block(x, edge_index, edge_attr) # Edge attributes injected
        return x 
 
    def _get_ptr(self, batch_tensor):
        """
        Get the pointer indices for the batch tensor.
        """
        num_graphs = batch_tensor.max().item() + 1
        ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch_tensor.device)
        ptr[1:] = torch.cumsum(torch.bincount(batch_tensor), dim=0)
        return ptr

    def act(self, x, edge_index, edge_attr, batch, valid_mask, stochastic: bool = True):
        """
       valid_mask (NOT optional) is a 2D bool tensor i.e., for each graph in the batch e.g., 2 graphs of 4 nodes each, [[False, True, True, False], [False, True, False, True]]
       - Mask out nodes that are not valid.
       - How batch works in PyG: 
        - If graph 0 has [A, B, C] nodes, and graph 1 has [D, E, F] nodes
        - Then PyG flattens them to a single graph with [A, B, C, D, E, F] nodes.
        - Then batch will be [0, 0, 0, 1, 1, 1] with nodes A-F indexed as 0 to 5
        - If you are allowed to pick [B, C] from graph 0 (local 1,2) and [D, F] from graph 1 (local 0,2), then valid_indices would be[[False, True, True], [True, False, True]].
       The batch input is not the object but the batch tensor i.e., [0, 0, 0, 1, 1, 1...]
       Build a categorical distribution (per graph) over the valid indices.
       When stochastic is True, sample from the distribution otherwise return argmax
       Returns: 
       - actions: tensor [batch_size] of local node indices selected as actions
       - log_probs: tensor [batch_size] of log probabilities of the selected actions
       - values: tensor [batch_size] i.e., one value per graph.
       """

        z = self._get_node_embeddings(x, edge_index, edge_attr)
        logits = self.actor_head(z).squeeze(-1)
        g = self.critic_readout(z, batch)
        values = self.critic_head(g).squeeze(-1)
        num_graphs = batch.max().item() + 1
        ptr = self._get_ptr(batch)

        if valid_mask.dim() != 2 or valid_mask.shape[0] != num_graphs:
            raise ValueError("valid_mask must be 2D bool tensor of shape [batch_size, max_nodes]")

        actions_list = []
        log_probs_list = []

        for i in range(num_graphs):
            start, end = ptr[i], ptr[i+1]
            
            # Number of nodes in the current graph.
            num_nodes_i = end - start
            valid_mask_i = valid_mask[i, :num_nodes_i] # For the i-th graph, get the valid mask.

            # Sanity
            if not valid_mask_i.any():
                raise ValueError(f"No valid nodes in graph {i}")

            # Logits for the current graph.
            graph_logits = logits[start:end]
            masked_logits = graph_logits.clone()
            masked_logits[~valid_mask_i] = -float('inf')

            # print(f"\nGraph {i} masked logits: \n{masked_logits}")
            # Distribution has to be built over all local indices in the current graph but logits for invalid ones are masked to -inf.
            # If we were to build a dist over only valid nodes (for example 3 valid nodes), then it would be over indices 0, 1, 2.
            # This is a problem because although the valid nodes could be index 10 , 11, 12 in the graph the dist would have indices 0, 1, 2.
            dist = Categorical(logits=masked_logits)
        
            if stochastic:
                local_idx = dist.sample()
            else:
                local_idx = torch.argmax(masked_logits) # Select the node index with the highest logit.
                # local_idx = dist.probs.argmax() # Select the node index with the highest probability. This is fine but slower and can NaN if all invalid.

            log_prob = dist.log_prob(local_idx)      # scalar value
            actions_list.append(local_idx)
            log_probs_list.append(log_prob)

        actions = torch.stack(actions_list)
        log_probs = torch.stack(log_probs_list)
        return actions, log_probs, values
    
    def critic_readout(self, z, batch):
        """
        Various pooling strategies can be used (pools over all nodes in a graph batch (which may contain multiple graphs)).
        Currently used: Global mean pooling i.e., average the node embeddings across all nodes in the graph.
        # TODO: Experiment with others.
        """
        return global_mean_pool(z, batch) # [batch_size, D]   

    def evaluate(self, x, edge_index, edge_attr, batch, valid_mask, actions):
        """
        The input actions are local indices into the current graph i.e., a 2D tensor of shape [batch_size, 1], 1 corresponding to local index of node_chosen_for_graph_i.
        valid_mask is a 2D bool tensor i.e., for each graph in the batch e.g., 2 graphs of 4 nodes each, [[False, True, True, False], [False, True, False, True]]
        Values (scalar values) are one value per graph.
        Returns: log_probs, entropies, values
        """
        z = self._get_node_embeddings(x, edge_index, edge_attr) 
        logits = self.actor_head(z).squeeze(-1)
        num_graphs = batch.max().item() + 1
        ptr = self._get_ptr(batch)
        
        if valid_mask.dim() != 2 or valid_mask.shape[0] != num_graphs:
            raise ValueError("valid_mask must be 2D bool tensor of shape [batch_size, max_nodes]")

        g = self.critic_readout(z, batch)
        values = self.critic_head(g).squeeze(-1)

        log_probs_list = []
        entropies_list = []

        for i in range(num_graphs):
            start, end = ptr[i], ptr[i+1]

            # Number of nodes in the current graph.
            num_nodes_i = end - start
            valid_mask_i = valid_mask[i, :num_nodes_i] # For the i-th graph, get the valid mask.

            # Handle forced-end steps where no valid actions existed.
            if not valid_mask_i.any():
                local_idx = actions[i]
                if int(local_idx.item()) != num_nodes_i:
                    raise ValueError(f"Received action {local_idx.item()} despite empty valid mask for graph {i}")

                zero = logits.new_zeros(())
                log_probs_list.append(zero)
                entropies_list.append(zero)
                continue

            # Logits for the current graph.
            graph_logits = logits[start:end]
            masked_logits = graph_logits.clone()
            masked_logits[~valid_mask_i] = -float('inf')

            # print(f"\nGraph {i} masked logits: \n{masked_logits}")

            # Sanity
            if masked_logits.numel() == 0:
                raise ValueError(f"No valid logits in graph {i}")

            # Distribution over valid local indices in the current graph.
            dist = Categorical(logits=masked_logits)
            local_idx = actions[i] # The action that was chosen for the i-th graph (local index).

            # Sanity 
            if int(local_idx.item()) >= num_nodes_i or not valid_mask_i[local_idx.item()]:
                raise ValueError(f"Invalid action {local_idx.item()} in graph {i}")

            log_prob = dist.log_prob(local_idx)
            entropy = dist.entropy()
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)

        log_probs = torch.stack(log_probs_list)
        entropies = torch.stack(entropies_list)
        return log_probs, entropies, values

    @torch.no_grad()
    def apply_orthogonal_init(self, hidden_gain: Optional[float] = None,
                              bias_const: float = 0.0,
                              actor_final_gain: float = 0.01,
                              critic_final_gain: float = 1.0):
        """
        Orthogonal initialization of weights and Constant initialization of biases.
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

        LayerNorm gets weight 1 and bias 0.
        Final actor and critic layers use their own gains.
        """
        if hidden_gain is None:
            hidden_gain = _gain_from_activation(self.activation_name)
        else:
            hidden_gain = hidden_gain
        
        # Pass 1: initialize every Linear and LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=hidden_gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias_const)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Pass 2: override the very last Linear of each head
        def _reset_last_linear(seq: nn.Sequential, gain: float):
            for layer in reversed(seq):
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, bias_const)
                    break

        _reset_last_linear(self.actor_head, actor_final_gain)
        _reset_last_linear(self.critic_head, critic_final_gain)
        
    
    def count_params(self):
        proj_params = count_module(self.proj)
        block_params = sum(count_module(b) for b in self.gat_blocks)
        actor_params = count_module(self.actor_head)
        critic_params = count_module(self.critic_head)
        total = proj_params + block_params + actor_params + critic_params
        lines = ["Parameter breakdown", "--------------------"]
        lines.append(f"Projection: {proj_params:,}")
        lines.append(f"GAT blocks total: {block_params:,}")
        lines.append(f"Actor head: {actor_params:,}")
        lines.append(f"Critic head: {critic_params:,}")
        lines.append(f"Total trainable: {total:,}")
        print("\n".join(lines))
        print("--------------------")

"""
if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(
        n_node_features=12,
        proj_out=32,
        num_gat_blocks=3,
        gat_channels=[32, 16, 8],
        num_heads=[16, 8, 4],
        attn_dropout=[0.1, 0.1, 0.1],
        feat_dropout=[0.1, 0.1, 0.1],
        actor_head_dropout=0.2,
        critic_head_dropout=0.2,
        concat=False,
        activation="leaky_relu",
        n_edge_features=2,
        actor_head_layers=[64, 32],
        critic_head_layers=[64, 32],
    )

    print("Initializing model...")
    model = GATV2ActorCritic(**cfg).to(device)
    model.apply_orthogonal_init(
        hidden_gain=None,         # auto from activation
        bias_const=0.0,
        actor_final_gain=0.01,    # small gain to keep initial policy high-entropy
        critic_final_gain=1.0,
    )
    model.count_params()

    # Graph 0: 10 nodes, fully connected for simplicity (edges between all pairs)
    num_nodes0 = 10
    x0 = torch.randn(num_nodes0, cfg["n_node_features"])
    ei0 = torch.combinations(torch.arange(num_nodes0), r=2).t()  # All pairs
    ei0 = torch.cat([ei0, ei0.flip(0)], dim=1)  # Undirected
    ea0 = torch.randn(ei0.size(1), cfg["n_edge_features"])
    g0 = Data(x=x0, edge_index=ei0, edge_attr=ea0)

    # Graph 1: 15 nodes, fully connected
    num_nodes1 = 15
    x1 = torch.randn(num_nodes1, cfg["n_node_features"])
    ei1 = torch.combinations(torch.arange(num_nodes1), r=2).t()  # All pairs
    ei1 = torch.cat([ei1, ei1.flip(0)], dim=1)  # Undirected
    ea1 = torch.randn(ei1.size(1), cfg["n_edge_features"])
    g1 = Data(x=x1, edge_index=ei1, edge_attr=ea1)

    # Batch the graphs
    batch = Batch.from_data_list([g0, g1]).to(device)
    print(f"\nBatch created with {batch.num_graphs} graphs:")
    print(f"Graph 0: {g0.num_nodes} nodes, global indices [0 to {g0.num_nodes-1}]")
    print(f"Graph 1: {g1.num_nodes} nodes, global indices [{g0.num_nodes} to {g0.num_nodes + g1.num_nodes - 1}]")
    print(f"Total nodes in batch: {batch.num_nodes}")

    # Define valid local indices for each graph
    valid_mask_local = [
        [2, 4, 5, 7, 9],  # For graph 0 (local indices)
        [0, 3, 6, 8, 10, 12, 14]  # For graph 1
    ]
    max_num_nodes = max(g0.num_nodes, g1.num_nodes)  # 15
    valid_mask = torch.zeros(2, max_num_nodes, dtype=torch.bool, device=device)
    for i, locals in enumerate(valid_mask_local):
        print(f"Graph {i} valid local indices: {locals}")
        for idx in locals:
            valid_mask[i, idx] = True

    # Test act() - stochastic sampling
    print("\nTesting act() with stochastic=True:")
    actions, log_probs, values = model.act(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, valid_mask, stochastic=True
    )
    for i in range(batch.num_graphs):
        global_idx = actions[i].item() + (0 if i == 0 else g0.num_nodes)
        print(f"Graph {i}: Selected local action {actions[i].item()} (global {global_idx}), log_prob {log_probs[i].item():.4f}, value {values[i].item():.4f}")

    # Test act() - deterministic (argmax)
    print("\nTesting act() with stochastic=False (deterministic):")
    actions_det, log_probs_det, values_det = model.act(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, valid_mask, stochastic=False
    )
    for i in range(batch.num_graphs):
        global_idx = actions_det[i].item() + (0 if i == 0 else g0.num_nodes)
        print(f"Graph {i}: Selected local action {actions_det[i].item()} (global {global_idx}), log_prob {log_probs_det[i].item():.4f}, value {values_det[i].item():.4f}")

    # Simulate past actions (local indices, assuming they were taken and are valid)
    past_actions = torch.tensor([4, 6], dtype=torch.long, device=device)  # Local 4 for graph0, 6 for graph1
    print("\nTesting evaluate() with past actions (local): {past_actions.tolist()}")
    for i, act in enumerate(past_actions):
        global_idx = act.item() + (0 if i == 0 else g0.num_nodes)
        print(f"Graph {i}: Past local action {act.item()} (global {global_idx})")

    log_probs_eval, entropies, values_eval = model.evaluate(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, valid_mask, past_actions
    )
    for i in range(batch.num_graphs):
        print(f"Graph {i}: Eval log_prob {log_probs_eval[i].item():.4f}, entropy {entropies[i].item():.4f}, value {values_eval[i].item():.4f}")

    print("\n" + "="*30)
    print("Testing act() with empty valid_mask")
    print("="*30)

    # All graphs have no valid indices
    print("\nAll graphs have no valid mask")
    valid_indices_empty_all = torch.zeros(batch.num_graphs, max_num_nodes, dtype=torch.bool, device=device)
    model.act(batch.x, batch.edge_index, batch.edge_attr, batch.batch, valid_indices_empty_all, stochastic=True)

"""
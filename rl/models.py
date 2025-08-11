from typing import Any


class GATV2ActorCritic:
    """
    Actor-Critic architecture with a GATv2 backbone for graphs.
    Provides shared features with separate actor and critic heads.
    Only interfaces are declared; implementation is omitted.
    """

    def __init__(self, node_feature_dim: int, num_actions: int) -> None:
        """
        Initialize model metadata and placeholder components.
        Save dimensions for backbone and head construction later.
        No heavy allocations occur in this skeleton.
        """
        pass

    def layer_init(self, layer: Any) -> Any:
        """
        Apply the default initialization scheme to a layer.
        Helps reproduce standard PPO initialization conventions.
        Returns the prepared layer after initialization.
        """
        pass

    def readout_layer(self, graph_batch: Any) -> Any:
        """
        Aggregate node embeddings into graph-level representations.
        Bridges GAT backbone outputs to policy/value heads.
        Returns a placeholder output in this skeleton.
        """
        pass

    def actor(self, features: Any) -> Any:
        """
        Map shared features to action distribution parameters.
        Supports discrete or continuous action spaces conceptually.
        Returns un-normalized logits or distribution parameters.
        """
        pass

    def critic(self, features: Any) -> Any:
        """
        Map shared features to a scalar state-value estimate.
        Outputs one value per environment instance in practice.
        Returns a placeholder until implemented.
        """
        pass

    def act(self, observations: Any, deterministic: bool = False) -> Any:
        """
        Produce actions given observations, optionally deterministically.
        Returns actions, log-probabilities, and value estimates.
        Logic is deferred to a later implementation phase.
        """
        pass

    def evaluate(self, observations: Any, actions: Any) -> Any:
        """
        Evaluate log-probs, entropy, and values for PPO updates.
        Used during optimization to compute surrogate losses.
        Implementation is intentionally omitted in the skeleton.
        """
        pass



import warnings
import wandb
from typing import Any, Dict
from rl.env import TransitEnv
from rl.mcts_agent import MCTSAgent

# Suppress torch-scatter installation warning from PyTorch Geometric
warnings.filterwarnings("ignore", message=".*torch-scatter.*")


def get_policy_kwargs_alpha(config: Dict[str, Any], node_feature_dim: int, edge_feature_dim: int) -> Dict[str, Any]:
    """
    Get model kwargs for AlphaTransit. Matches structure of get_policy_kwargs_ppo.
    """
    n = config.get("num_gat_blocks", 4)
    half = n // 2
    
    # gat_channels and num_heads must be specified together
    if ("gat_channels" in config) != ("num_heads" in config):
        raise ValueError("gat_channels and num_heads must be specified together")
    
    # Defaults: 4 blocks -> [128,128,64,64], [8,8,4,4]
    #           6 blocks -> [128,128,128,64,64,64], [8,8,8,4,4,4]
    #           8 blocks -> [128,128,128,128,64,64,64,64], [8,8,8,8,4,4,4,4]
    gat_channels = config.get("gat_channels", [128] * half + [64] * (n - half))
    num_heads = config.get("num_heads", [8] * half + [4] * (n - half))

    return {
        "n_node_features": node_feature_dim,
        "proj_out": config.get("proj_out", 64),
        "num_gat_blocks": n,
        "gat_channels": gat_channels,
        "num_heads": num_heads,
        "attn_dropout": [0.0] * n,
        "feat_dropout": [0.0] * n,
        "actor_head_dropout": 0.0,
        "critic_head_dropout": 0.0,
        "concat": config.get("concat_heads", False),
        "activation": config.get("activation", "tanh"),
        "n_edge_features": edge_feature_dim,
        "actor_head_layers": config.get("actor_head_layers", [256, 128, 64]),
        "critic_head_layers": config.get("critic_head_layers", [256, 128, 64]),
        "critic_readout_type": config.get("critic_readout_type", "sum"),
    }


def train(config: Dict[str, Any], is_sweep: bool = False) -> None:
    """
    Main AlphaTransit training function for both standalone and sweep use.

    Args:
        config: Configuration dictionary
        is_sweep: If True, assumes wandb is already initialized by sweep agent
    """
    # IMPORTANT: route_init="random" is DISABLED for MCTS.
    # Reason: MCTSState.apply_action() and force_route_end() call initialize_route()
    # which consumes shared RNG when random, making transitions stochastic and
    # path-dependent. Tree statistics become invalid (same state + action → different
    # successors across simulations). Only "transit_center" or "highest_demand" work.
    if config.get("route_init") == "random":
        raise ValueError(
            "route_init='random' is incompatible with MCTS. "
            "Use 'transit_center' or 'highest_demand' instead."
        )

    # Initialize wandb for standalone runs (sweep handles its own init)
    if not is_sweep and not config.get("wandb_off"):
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)

    env = TransitEnv(config)
    policy_kwargs = get_policy_kwargs_alpha(config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES)
    mcts_agent = MCTSAgent(env, config, policy_kwargs, spawn_workers=True)
    mcts_agent.train()

    # Finish wandb for standalone runs
    if not is_sweep and not config.get("wandb_off"):
        wandb.finish()


# =============================================================================
# Alpha eval entry point. Called from main.py when algorithm == "alphatransit" and mode == "eval".
# For training, use train() directly.
# =============================================================================

def alpha_eval(config: Dict[str, Any]) -> None:
    """
    Entry point for standalone AlphaTransit evaluation mode (like ppo_eval).
    Requires --gpu (uses parallel workers + inference server).
    """
    if config.get("route_init") == "random":
        raise ValueError(
            "route_init='random' is incompatible with MCTS. "
            "Use 'transit_center' or 'highest_demand' instead."
        )

    config["wandb_off"] = True

    env = TransitEnv(config)
    policy_kwargs = get_policy_kwargs_alpha(config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES)
    mcts_agent = MCTSAgent(env, config, policy_kwargs, spawn_workers=True)
    from datetime import datetime
    ts = datetime.now().strftime('%b_%d_%H_%M_%S')
    eval_save_dir = f"{config['save_dir']}_{ts}"
    mcts_agent.evaluate(policy_path=config.get("saved_policy_path", ""), save_dir=eval_save_dir)

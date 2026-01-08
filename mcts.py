"""
AlphaTransit: MCTS-Based Policy Learning for TRNDP
================================================================================

FULL VERSION (Two-Part Algorithm)
================================================================================

Algorithm 1: AlphaTransit for TRNDP -- Phase 1: Self-Play
---------------------------------------------------------
Input: Graph G=(V,E), OD matrix D, directed edges I, edge features Z,
       routes K, max length L_max, MCTS simulations N_iter, exploration c_puct,
       noise (alpha_dir, epsilon)
Output: Replay buffer D with tuples (s, pi, z_tilde)

Initialize: Buffer D <- empty, Welford statistics (mu, sigma, n_w) <- (0, 0, 0)

FOR training iteration = 1, 2, ...
    tau <- TempSchedule(progress)                    # 1.0 -> 0.5 -> 0.1
    FOR episode e = 1 to N_ep
        Pi <- empty; V_cmp <- empty; E <- []; T <- InitTree()
        FOR k = 1 to K
            r_k <- [transit_center]; V_cur <- {r_k[0]}
            WHILE |r_k| < L_max
                C_t <- valid one-hop neighbors of frontier
                IF C_t = empty THEN break                # no feasible extension

                X_t <- FormState(V_cur, V_cmp)
                s_t <- (X_t, I, Z)

                # Expand root if needed
                IF NOT T.root.expanded THEN
                    (P, V) <- f_theta(s_t)
                    P <- MaskNorm(P, C_t)
                    T.root.Expand(P, V)
                ENDIF

                eta ~ Dir(alpha_dir)
                P <- (1 - epsilon) * P + epsilon * eta   # Dirichlet noise

                # MCTS simulations
                FOR i = 1 to N_iter
                    node <- T.root; path <- []
                    WHILE node.expanded AND NOT IsTerminal(node)
                        a <- argmax_a' [Q + c_puct * P * sqrt(sum_b N_b) / (1 + N_a')]
                        path.Append((node, a))
                        node <- node.Child(a)
                    ENDWHILE

                    IF NOT node.expanded AND NOT IsTerminal(node) THEN
                        (P', V') <- f_theta(node.state)
                        P' <- MaskNorm(P', C)
                        node.Expand(P', V')
                        v <- V'
                    ELSE
                        v <- node.V
                    ENDIF

                    FOR (n, a) in Reversed(path)
                        N_{n,a} += 1
                        W_{n,a} += v
                        Q_{n,a} <- W_{n,a} / N_{n,a}
                    ENDFOR
                ENDFOR

                pi_t <- Softmax(N_{T.root}^{1/tau})      # visit count policy
                E.Append((s_t, pi_t))
                a_t ~ pi_t
                Append a_t to r_k
                V_cur <- nodes in r_k
                T.Advance(a_t)                           # re-root tree to child
            ENDWHILE
            V_cmp <- V_cmp UNION V_cur
            Pi <- Pi UNION {r_k}
        ENDFOR

        z <- R_final(Pi)                                 # traffic simulation
        (mu, sigma, n_w) <- WelfordUpdate(mu, sigma, n_w, z)
        z_tilde <- Clip((z - mu) / (sigma + 1e-8), -3, 3)

        FOR (s_i, pi_i) in E
            D.Add((s_i, pi_i, z_tilde))                  # FIFO if |D| > B_max
        ENDFOR
    ENDFOR
    # Proceed to Phase 2: Network Optimization
ENDFOR


Algorithm 2: AlphaTransit for TRNDP -- Phase 2: Training
--------------------------------------------------------
Input: Replay buffer D, network f_theta, training steps N_steps,
       batch size B, learning rate alpha
Output: Updated parameters theta

FOR step = 1 to N_steps
    Sample minibatch B <- Sample(D, B)
    L <- 0
    FOR (s, pi, z_tilde) in B
        (P_theta, V_theta) <- f_theta(s)
        P_theta <- MaskNorm(P_theta, C)
        L_policy <- -sum_a pi(a) * log(P_theta(a))       # cross-entropy
        L_value <- (z_tilde - V_theta)^2                 # MSE
        L <- L + L_policy + L_value
    ENDFOR
    theta <- theta - alpha * grad_theta(L / |B|)
ENDFOR
RETURN theta


================================================================================
HYPERPARAMETERS
================================================================================
- N_iter = 100          (MCTS simulations per move)
- c_puct = 1.5          (exploration constant)
- alpha_dir = 0.3       (Dirichlet noise concentration)
- epsilon = 0.25        (Dirichlet noise weight)
- Buffer capacity = 100,000 (FIFO eviction)
- N_ep = 2              (episodes per iteration)
- N_steps = 500         (training steps per iteration)
- Batch size B = 128
- Learning rate = 5e-5
- Temperature schedule: tau = 1.0 (progress < 0.3), 0.5 (0.3-0.6), 0.1 (> 0.6)
- Evaluation: tau -> 0 (greedy), no Dirichlet noise


================================================================================
SUPPORTING DEFINITIONS
================================================================================
- MaskNorm(P, C): Set P(a) = 0 for a not in C, renormalize so sum = 1
- WelfordUpdate(mu, sigma, n, x): Online mean/variance update
- FormState(V_cur, V_cmp): Construct node features encoding current/completed routes
- IsTerminal(s): True when all K routes constructed
- T.Advance(a): Re-root tree to child node for action a, discard siblings
- TempSchedule(progress): Returns tau based on training progress (steps/max_steps)

"""

import warnings
import wandb
from typing import Any, Dict
from rl.env import TransitEnv
from rl.mcts_agent import MCTSAgent

# Suppress torch-scatter installation warning from PyTorch Geometric
warnings.filterwarnings("ignore", message=".*torch-scatter.*")


def get_policy_kwargs_mcts(config: Dict[str, Any], node_feature_dim: int, edge_feature_dim: int) -> Dict[str, Any]:
    """
    Get model kwargs for MCTS. Matches structure of get_policy_kwargs_ppo.
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


def mcts_train(config: Dict[str, Any]) -> None:
    """
    Entry point for MCTS training mode.
    Initializes wandb if enabled, runs training, and cleans up.
    """
    if not config.get("wandb_off"):
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)

    env = TransitEnv(config)
    policy_kwargs = get_policy_kwargs_mcts(config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES)
    mcts_agent = MCTSAgent(env, config, policy_kwargs)
    mcts_agent.train()

    if not config.get("wandb_off"):
        wandb.finish()


def mcts_eval(config: Dict[str, Any]) -> None:
    """
    Entry point for standalone MCTS evaluation mode (like ppo_eval).
    """
    import os
    os.makedirs(config["save_dir"], exist_ok=True)
    config["wandb_off"] = True

    env = TransitEnv(config)
    policy_kwargs = get_policy_kwargs_mcts(config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES)
    mcts_agent = MCTSAgent(env, config, policy_kwargs)
    mcts_agent.evaluate(
        policy_path=config.get("saved_policy_path", ""),
        save_dir=config["save_dir"]
    )

r"""
AlphaTransit: MCTS-Based Policy Learning for TRNDP
================================================================================

FULL VERSION (Two-Part Algorithm)
================================================================================

Algorithm 1: AlphaTransit for TRNDP -- Phase 1: Self-Play
---------------------------------------------------------
\textbf{Input:} Graph $G=(V,E)$, OD matrix $D$, directed edges $I$, edge features $Z$,
       routes $K$, max length $L_{\max}$, MCTS simulations $N_{\text{iter}}$,
       exploration $c_{\text{puct}}$, noise $(\alpha_{\text{dir}}, \epsilon)$, workers $W$
\textbf{Output:} Replay buffer $\mathcal{D}$ with tuples $(s, \pi, \tilde{z})$

\textbf{Initialize:} $\mathcal{D} \gets \emptyset$, Welford statistics $(\mu, \sigma^2, n) \gets (0, 0, 0)$

\FOR{iteration $= 1, 2, \ldots$}
    $\tau \gets \text{TemperatureSchedule}(\text{progress})$

    \tcp{Run $W$ episodes in parallel, each with unique seed}
    \FORALL{worker $w = 1, \ldots, W$ \textbf{in parallel}}
        $\Pi_w \gets \emptyset$; $V_{\text{cmp}} \gets \emptyset$; $\mathcal{E}_w \gets \emptyset$; $\mathcal{T} \gets \text{InitTree}()$
        \FOR{$k = 1$ \TO $K$}
            $r_k \gets [\text{transit\_center}]$; $V_{\text{cur}} \gets \{r_k[0]\}$
            \WHILE{$|r_k| < L_{\max}$}
                $\mathcal{C}_t \gets$ valid one-hop neighbors of frontier
                \IF{$\mathcal{C}_t = \emptyset$}
                    \textbf{break}  \tcp{no feasible extension}
                \ENDIF

                $X_t \gets \text{FormState}(V_{\text{cur}}, V_{\text{cmp}})$
                $s_t \gets (X_t, I, Z)$

                \tcp{Expand root if needed}
                \IF{$\mathcal{T}.\text{root}$ not expanded}
                    $(P, v) \gets f_\theta(s_t)$
                    $P \gets \text{MaskNorm}(P, \mathcal{C}_t)$
                    Expand $\mathcal{T}.\text{root}$ with $(P, v)$
                \ENDIF

                $\eta \sim \text{Dir}(\alpha_{\text{dir}})$
                $P \gets (1 - \epsilon) P + \epsilon \eta$  \tcp{Dirichlet noise}

                \tcp{MCTS simulations}
                \FOR{$i = 1$ \TO $N_{\text{iter}}$}
                    $\text{node} \gets \mathcal{T}.\text{root}$; $\text{path} \gets []$
                    \WHILE{node expanded \AND node not terminal}
                        $a \gets \argmax_{a'} \left[ Q_{a'} + c_{\text{puct}} \cdot P_{a'} \cdot \frac{\sqrt{\sum_b N_b}}{1 + N_{a'}} \right]$
                        Add $(\text{node}, a)$ to path
                        $\text{node} \gets \text{Child}(\text{node}, a)$
                    \ENDWHILE

                    \IF{node not expanded \AND node not terminal}
                        $(P', v') \gets f_\theta(\text{node.state})$
                        $P' \gets \text{MaskNorm}(P', \mathcal{C})$
                        Expand node with $(P', v')$
                        $v \gets v'$
                    \ELSE
                        $v \gets \text{node}.V$
                    \ENDIF

                    \FOR{$(n, a)$ in reversed(path)}
                        $N_{n,a} \gets N_{n,a} + 1$
                        $W_{n,a} \gets W_{n,a} + v$
                        $Q_{n,a} \gets W_{n,a} / N_{n,a}$
                    \ENDFOR
                \ENDFOR

                $\pi_t \gets \text{Softmax}(N_{\mathcal{T}.\text{root}}^{1/\tau})$  \tcp{visit count policy}
                $\mathcal{E}_w \gets \mathcal{E}_w \cup \{(s_t, \pi_t)\}$
                $a_t \sim \pi_t$
                Add $a_t$ to $r_k$
                $V_{\text{cur}} \gets$ nodes in $r_k$
                Re-root $\mathcal{T}$ to child of $a_t$
            \ENDWHILE
            $V_{\text{cmp}} \gets V_{\text{cmp}} \cup V_{\text{cur}}$
            $\Pi_w \gets \Pi_w \cup \{r_k\}$
        \ENDFOR
        $z_w \gets R_{\text{final}}(\Pi_w)$  \tcp{traffic simulation}
    \ENDFOR

    \tcp{Aggregate results from all workers}
    \FOR{$w = 1$ \TO $W$}
        $(\mu, \sigma^2, n) \gets \text{WelfordUpdate}(\mu, \sigma^2, n, z_w)$
        \FOR{$(s_i, \pi_i) \in \mathcal{E}_w$}
            $\mathcal{D} \gets \mathcal{D} \cup \{(s_i, \pi_i, z_w)\}$  \tcp{Store raw reward; FIFO if $|\mathcal{D}| > B_{\max}$}
        \ENDFOR
    \ENDFOR
    \tcp{Proceed to Phase 2: Network Optimization}
\ENDFOR


Algorithm 2: AlphaTransit for TRNDP -- Phase 2: Training
--------------------------------------------------------
\textbf{Input:} Replay buffer $\mathcal{D}$, network $f_\theta$, training steps $N_{\text{steps}}$,
       batch size $B$, learning rate $\alpha$
\textbf{Output:} Updated parameters $\theta$

\FOR{step $= 1$ \TO $N_{\text{steps}}$}
    Sample minibatch $\mathcal{B} \gets \text{Sample}(\mathcal{D}, B)$
    $\mathcal{L} \gets 0$
    \FOR{$(s, \pi, z) \in \mathcal{B}$}
        $\tilde{z} \gets \text{clip}\left(\frac{z - \mu}{\sigma + \epsilon}, -3, 3\right)$  \tcp{Normalize with current stats}
        $(P_\theta, V_\theta) \gets f_\theta(s)$
        $P_\theta \gets \text{MaskNorm}(P_\theta, \mathcal{C})$
        $\mathcal{L}_{\text{policy}} \gets -\sum_a \pi(a) \log P_\theta(a)$  \tcp{cross-entropy}
        $\mathcal{L}_{\text{value}} \gets (\tilde{z} - V_\theta)^2$  \tcp{MSE}
        $\mathcal{L} \gets \mathcal{L} + \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$
    \ENDFOR
    $\theta \gets \theta - \alpha \nabla_\theta (\mathcal{L} / |\mathcal{B}|)$
\ENDFOR
\RETURN $\theta$


================================================================================
HYPERPARAMETERS
================================================================================
- $N_{\text{iter}} = 400$       (MCTS simulations per move)
- $c_{\text{puct}} = 1.5$       (exploration constant)
- $\alpha_{\text{dir}} = 0.3$   (Dirichlet noise concentration)
- $\epsilon = 0.25$             (Dirichlet noise weight)
- Buffer capacity $B_{\max} = 100{,}000$ (FIFO eviction)
- $W = 8$                       (parallel workers, episodes per iteration)
- $N_{\text{steps}} = 500$      (training steps per iteration)
- Batch size $B = 256$
- Learning rate $\alpha = 5 \times 10^{-5}$
- Temperature schedule: $\tau = 1.0$ (progress $< 0.3$), $0.5$ ($0.3$-$0.6$), $0.1$ ($> 0.6$)
- Evaluation: $\tau = 0.1$ (near-greedy), no Dirichlet noise


================================================================================
SUPPORTING DEFINITIONS
================================================================================
- $\text{MaskNorm}(P, \mathcal{C})$: Set $P(a) = 0$ for $a \notin \mathcal{C}$, renormalize so $\sum_a P(a) = 1$
- $\text{WelfordUpdate}(\mu, \sigma^2, n, x)$: Online mean/variance update
- $\text{FormState}(V_{\text{cur}}, V_{\text{cmp}})$: Construct node features encoding current/completed routes
- Terminal state: All $K$ routes constructed
- Re-root tree: Move root to child node for action $a$, discard siblings
- $\text{TemperatureSchedule}(\text{progress})$: Returns $\tau$ based on training progress

"""

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
    """
    import os
    os.makedirs(config["save_dir"], exist_ok=True)
    config["wandb_off"] = True

    env = TransitEnv(config)
    policy_kwargs = get_policy_kwargs_alpha(config, env.N_NODE_FEATURES, env.N_EDGE_FEATURES)
    mcts_agent = MCTSAgent(env, config, policy_kwargs, spawn_workers=False)
    mcts_agent.evaluate(
        policy_path=config.get("saved_policy_path", ""),
        save_dir=config["save_dir"]
    )

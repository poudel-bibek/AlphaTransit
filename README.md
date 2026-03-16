# AlphaTransit: Learning to Design City-Scale Transit Routes

This repository implements deep reinforcement learning algorithms for the **Transit Route Network Design Problem (TRNDP)**. We build on [UXsim](https://github.com/toruseo/UXsim), a mesoscopic traffic simulator, adding graph-based policies, transit-specific extensions (bus dispatching, passenger boarding/alighting, transfers), and standardized evaluation tooling.

<p align="center">
  <img src="assets/real_world_anim_all_vehicles.gif" alt="Transit simulation visualization" width="640">
</p>

## Problem Overview

TRNDP involves designing a set of transit routes within a road network to optimize passenger service while respecting operational constraints. The problem is NP-hard with a combinatorial search space (e.g., ~10^115 configurations for 16 routes of length 14 on the Bloomington network).

**Key challenges:**
- Deceptive optimization landscape where greedy choices create bottlenecks
- Sparse, delayed rewards (quality only visible after full simulation)
- Multi-objective trade-offs between passenger experience and operator costs

## Algorithms

### AlphaTransit

Our primary algorithm combines Monte Carlo Tree Search (MCTS) with neural network guidance, inspired by AlphaZero:
- **PUCT selection** for balancing exploration and exploitation
- **GATv2 policy network** providing action priors and value estimates
- **Dirichlet noise** at root for exploration during self-play
- **Terminal-only rewards** with Welford normalization for stability
- **Parallel episode collection** with configurable workers (default: 8)

### PPO (End-to-End Reinforcement Learning)

Graph-attention actor-critic trained with Proximal Policy Optimization:
- **Parallel environment workers** for efficient sample collection
- **Intermediate delta reward shaping** during route construction + terminal simulation reward
- **Episode-level Generalized Advantage Estimation** for variance reduction
- Same network architecture as AlphaTransit for fair comparison

### Baselines

| Baseline | Description |
|----------|-------------|
| **Random Walk** | Uniform random neighbor selection |
| **Demand Coverage** | Sample proportional to incremental demand (z-score normalized, softmax) |
| **Shortest Path** | Sample proportional to inverse edge length |
| **Reward Maximization** | Greedy immediate reward maximization |
| **Genetic Algorithm** | Population-based metaheuristic with route-exchange crossover and path-regeneration mutation |
| **Real World** | Existing Bloomington Transit routes (16 routes) |

All methods use identical reward functions and simulation parameters for fair comparison.

## Installation

### Requirements

- **Python 3.14+** (required for free-threading multiprocessing improvements)
- **CUDA-capable GPU** (optional, for faster training)
- **uv** package manager (recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone and setup
git clone <repository-url>
cd AlphaTransit
uv sync
source .venv/bin/activate
```

## Quick Start

### Train with best hyperparameters

Pass `--apply_best_params` to automatically use the best hyperparameters found from sweep experiments for the given algorithm and alpha. Explicit CLI args still override.

```bash
# PPO with best params (alpha=0.3, ~5 hours)
python main.py --algorithm ppo --gpu --alpha=0.3 --apply_best_params

# PPO with best params (alpha=1.0, ~3 hours)
python main.py --algorithm ppo --gpu --alpha=1.0 --apply_best_params

# AlphaTransit with best params (alpha=0.3, ~24-30 hours)
python main.py --algorithm mcts --gpu --alpha=0.3 --apply_best_params

# AlphaTransit with best params (alpha=1.0, ~18-24 hours)
python main.py --algorithm mcts --gpu --alpha=1.0 --apply_best_params

# Override a specific param while keeping the rest
python main.py --algorithm ppo --gpu --alpha=0.3 --apply_best_params --lr=0.001
```

### Run baselines

```bash
# Heuristic baselines (alpha=0.3)
python main.py --mode=baseline --baseline_type=random_walk    --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=demand_cover   --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=shortest_path  --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=reward_max     --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=real_world     --route_init=transit_center --alpha=0.3

# Heuristic baselines (alpha=1.0)
python main.py --mode=baseline --baseline_type=random_walk    --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=demand_cover   --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=shortest_path  --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=reward_max     --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=real_world     --route_init=transit_center --alpha=1.0

# Genetic Algorithm
python main.py --mode=baseline --baseline_type=genetic --alpha=0.3 --ga_population=50 --ga_generations=100 --ga_num_workers=4
python main.py --mode=baseline --baseline_type=genetic --alpha=1.0 --ga_population=50 --ga_generations=100 --ga_num_workers=4
```

### Evaluate a saved policy

```bash
# AlphaTransit evaluation
python main.py --algorithm mcts --mode=eval --saved_policy_path=training_data/<timestamp>/mcts_policies/policy_final.pth

# PPO evaluation
python main.py --algorithm ppo --mode=eval --saved_policy_path=training_data/<timestamp>/ppo_policies/policy_final.pth
```

### Hyperparameter sweeps (requires WandB)

```bash
python sweep.py --algorithm ppo
python sweep.py --algorithm mcts
```

### Reward ablation experiments

```bash
# Alpha 0.3
python main.py --algorithm ppo --gpu --alpha=0.3 --apply_best_params --ppo_reward_mode=terminal_only
python main.py --algorithm ppo --gpu --alpha=0.3 --apply_best_params --ppo_reward_mode=terminal_intermediate_raw_early_stop
python main.py --algorithm ppo --gpu --alpha=0.3 --apply_best_params --ppo_reward_mode=terminal_intermediate_delta_early_stop
python main.py --algorithm ppo --gpu --alpha=0.3 --apply_best_params --ppo_reward_mode=terminal_intermediate_delta_no_early_stop

# Alpha 1.0
python main.py --algorithm ppo --gpu --alpha=1.0 --apply_best_params --ppo_reward_mode=terminal_only
python main.py --algorithm ppo --gpu --alpha=1.0 --apply_best_params --ppo_reward_mode=terminal_intermediate_raw_early_stop
python main.py --algorithm ppo --gpu --alpha=1.0 --apply_best_params --ppo_reward_mode=terminal_intermediate_delta_early_stop
python main.py --algorithm ppo --gpu --alpha=1.0 --apply_best_params --ppo_reward_mode=terminal_intermediate_delta_no_early_stop
```

## Key Hyperparameters

Best hyperparameters are stored in `config.py` (`BEST_PARAMS` dict) and applied automatically with `--apply_best_params`. They were found via Bayesian/grid sweeps on WandB. Below are the defaults (which match alpha=0.3 PPO best params):

### AlphaTransit

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_iter` | 100 | MCTS simulations per decision |
| `--c_puct` | 1.0 | PUCT exploration constant |
| `--temp_schedule` | "1.0:1.0" | Temperature schedule (progress:tau pairs) |
| `--num_gat_blocks` | 4 | GATv2 attention blocks |
| `--train_steps_per_iter` | 200 | Network training steps per MCTS iteration |
| `--buffer_capacity` | 50000 | Replay buffer capacity |
| `--max_iterations` | 680 | Training iterations (~1M steps with 8 workers) |
| `--num_mcts_workers` | 8 | Parallel episode collection workers |

### PPO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_steps` | 1,000,000 | Total training environment steps |
| `--lr` | 5e-5 | Learning rate |
| `--K_epochs` | 8 | PPO epochs per update |
| `--batch_size` | 256 | Mini-batch size |
| `--clip_frac` | 0.2 | PPO clipping ratio |
| `--num_gat_blocks` | 4 | GATv2 attention blocks |
| `--activation` | tanh | Activation function |
| `--ppo_reward_mode` | delta_no_early_stop | Reward shaping mode |
| `--num_ppo_workers` | 8 | Parallel environment workers |

### Genetic Algorithm

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ga_population` | 50 | Population size |
| `--ga_generations` | 100 | Number of generations |
| `--ga_mutation_rate` | 0.4 | Mutation probability |
| `--ga_crossover_rate` | 0.8 | Crossover probability |
| `--ga_num_workers` | 4 | Parallel fitness evaluation workers |

### Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.3 | Modal split (fraction of demand using transit) |
| `--num_routes` | 16 | Number of routes to design |
| `--max_route_length` | 14 | Maximum nodes per route |
| `--route_init` | "transit_center" | Route initialization ("transit_center" or "random") |

## Project Structure

```
AlphaTransit/
├── main.py                 # CLI entry point
├── config.py               # Configuration, argument parsing, and best hyperparameters
├── mcts.py                 # AlphaTransit training/evaluation
├── ppo.py                  # PPO training/evaluation
├── sweep.py                # WandB hyperparameter sweeps
│
├── rl/
│   ├── env.py              # TransitEnv (Gymnasium wrapper for UXsim)
│   ├── models.py           # GATv2ActorCritic network architecture
│   ├── mcts_agent.py       # AlphaTransit agent implementation
│   ├── mcts_utils.py       # MCTS tree, replay buffer, Welford normalizer
│   ├── mcts_worker.py      # Parallel episode collection for AlphaTransit
│   ├── ppo_agent.py        # PPO agent implementation
│   ├── ppo_utils.py        # PPO memory buffer, normalizers
│   ├── parallel_env.py     # Parallel environment workers for PPO
│   ├── baselines.py        # Heuristic and GA baselines
│   └── env_utils.py        # Plotting, aggregation, seed utilities
│
├── networks/
│   ├── bloomington/        # Bloomington, IN network data
│   │   ├── bloomington_nodes_standard.csv
│   │   ├── bloomington_links_standard.csv
│   │   ├── bloomington_demand_standard.csv
│   │   └── bloomington_existing_routes.json
│   └── sioux_falls/        # Sioux Falls network (for testing)
│
├── uxsim/                  # UXsim simulator (extended with BusHandler)
├── plots/                  # Visualization and analysis scripts
└── training_data/          # Training outputs, checkpoints, sweep data
```

## Network Architecture

**GATv2ActorCritic** (shared between AlphaTransit and PPO):

- **Input projection**: Node features (16-dim) → hidden dimension
- **4 GATv2 blocks**: Channels [128, 128, 64, 64], heads [8, 8, 4, 4], with residual connections and LayerNorm
- **Actor head**: MLP producing per-node logits (permutation equivariant)
- **Critic head**: Global pooling → MLP producing scalar value (permutation invariant)

The architecture is size-independent: the same model works on networks of any size.

## State Representation

Each node has 16 features:
- **Static (0-4)**: Normalized coordinates, degree, total outgoing/incoming demand
- **Dynamic (5-12)**: Route-aware demand features (local and global, for current and completed routes)
- **Flags (13-15)**: In current route, in completed routes (fraction), valid next action

## Reward Function

Terminal reward combining:
- **Demand coverage**: Fraction of total demand reachable by designed routes
- **Service rate**: Fraction of total demand successfully served (fixed denominator)
- **Travel time penalty**: Average passenger journey duration
- **Route overlap penalty**: Discourages redundant route segments
- **Fleet cost**: Penalizes excessive bus deployment
- **Bus utilization**: Rewards efficient use of fleet

## Simulation

- **Engine**: UXsim mesoscopic simulator with Newell's car-following model
- **Time step**: dt = 1 second, platoon size dn = 5
- **Horizon**: 10,000 steps (~2.7 hours)
- **Bus parameters**: 40 passenger capacity, 60s dwell time per stop
- **Frequency setting**: Max-load rule normalized for route overlaps

## Data

The Bloomington network includes:
- **143 nodes, 243 edges** (topologically accurate road network)
- **Origin-destination demand** derived from LEHD/LODES census data
- **16 existing bus routes** from Bloomington Transit GTFS feed

## Logging

Training logs to Weights & Biases by default:

```bash
# With WandB
python main.py --algorithm mcts --wandb_project=transit --wandb_entity=my-team

# Without WandB
python main.py --algorithm mcts --wandb_off
```

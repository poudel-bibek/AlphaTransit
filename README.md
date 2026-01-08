
# Learning to Design City-scale Transit Routes using Reinforcement Learning

This repository builds on top of [UXsim](https://github.com/toruseo/UXsim) to study network-level transit design with deep reinforcement learning. We retain the complete UXsim simulator while adding graph-based policies, evaluation tooling, and standardized transit demand data.

<img src="assets/real_world_anim_all_vehicles.gif" alt="Transit visualization" width="640">

## Data

The pre-processed Bloomington network dataset (network, demand, transit routes) are under `networks/bloomington/`:

- network
  - `bloomington_nodes_standard.csv`
  - `bloomington_links_standard.csv`
- demand: `bloomington_demand_standard.csv`
- existing routes: `bloomington_existing_routes.json`


## Highlights

- Built on UXsim's mesoscopic traffic engine for fast, numerically stable simulations.
- Two learning algorithms:
  - **PPO**: Graph-attention PPO agent with parallel environment workers
  - **MCTS (AlphaTransit)**: AlphaZero-style MCTS with neural network guidance
- Baseline heuristics (`random_walk`, `demand_cover`, `shortest_path`, `reward_max`, `real_world`) for comparison.
- Cached PyG data pipelines, demand visualizations, and rich WandB logging.
- Hyperparameter sweep support via `sweep.py`.

## Requirements

- **Python 3.14+** (required for free-threading multiprocessing improvements)
- **uv** package manager (recommended)
- CUDA-capable GPU (optional, for faster training)

## Installation

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Clone and set up the project**:
   ```bash
   git clone <repository-url>
   cd Transit_Design
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Option 2: Manual Setup

1. **Install Python 3.14** from the [official website](https://www.python.org/downloads/release/python-3142/).

2. **Create a virtual environment**:
   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Verify installation with a heuristic baseline

```bash
python main.py --mode=baseline --baseline_type=demand_cover --network=bloomington --num_eval_runs=1 --save_animations
```

This runs a single evaluation on the Bloomington network, generates 16 candidate routes (the default), and writes animation gif to `training_data/`.

### 2. Train an RL policy

**PPO Training:**
```bash
# Single process training
python main.py --algorithm ppo --gpu --anneal_lr

# Parallel training with 8 workers (default)
python main.py --algorithm ppo --gpu --anneal_lr --num_workers=8
```

**MCTS (AlphaTransit) Training:**
```bash
python main.py --algorithm mcts --gpu
```

Training artifacts are saved to `training_data/<timestamp>/` with policy checkpoints under `ppo_policies/` or `mcts_policies/`.

### 3. Evaluate a saved policy

```bash
# Evaluate PPO policy
python main.py --algorithm ppo --mode=eval --saved_policy_path=training_data/<timestamp>/ppo_policies/policy_final.pth

# Evaluate MCTS policy
python main.py --algorithm mcts --mode=eval --saved_policy_path=training_data/<timestamp>/mcts_policies/policy_final.pth
```

### 4. Run hyperparameter sweeps

```bash
# PPO sweep (requires WandB)
python sweep.py --algorithm ppo

# MCTS sweep
python sweep.py --algorithm mcts
```

### 5. Optional: Log to Weights & Biases

```bash
python main.py --algorithm ppo --wandb_project=transit_design --wandb_entity=my-team
```

Disable logging anytime with `--wandb_off`.

## Algorithms

### PPO (Proximal Policy Optimization)

Graph-attention actor-critic that proposes routes node-by-node. Uses parallel environment workers for efficient sample collection.

Key hyperparameters:
- `--lr`: Learning rate (default: 5e-5)
- `--K_epochs`: PPO epochs per update (default: 4)
- `--clip_frac`: PPO clipping ratio (default: 0.1)
- `--num_workers`: Parallel workers (default: 8)
- `--max_steps`: Total training steps (default: 1M)

### MCTS (AlphaTransit)

AlphaZero-style Monte Carlo Tree Search with neural network guidance. Uses PUCT selection, Dirichlet noise for exploration, and terminal-only rewards with Welford normalization.

Key hyperparameters:
- `--n_iter`: MCTS simulations per move (default: 100)
- `--c_puct`: PUCT exploration constant (default: 1.5)
- `--dirichlet_alpha`: Dirichlet noise concentration (default: 0.3)
- `--episodes_per_iter`: Self-play episodes per iteration (default: 2)
- `--max_iterations`: Training iterations (default: 500)

## Project Structure

```
├── main.py                # CLI entry point for train/eval/baseline
├── config.py              # Shared configuration and argument parsing
├── ppo.py                 # PPO training loop and entry points
├── mcts.py                # MCTS training loop and entry points
├── sweep.py               # WandB hyperparameter sweep runner
├── rl/
│   ├── env.py             # TransitEnv: wraps UXsim worlds and bus operations
│   ├── models.py          # GATv2 actor-critic network
│   ├── ppo_agent.py       # PPO agent implementation
│   ├── ppo_utils.py       # PPO memory buffer, normalizers
│   ├── mcts_agent.py      # MCTS agent (AlphaTransit) implementation
│   ├── mcts_utils.py      # MCTS tree, replay buffer, utilities
│   ├── parallel_env.py    # Multi-process workers for PPO
│   ├── env_utils.py       # Plotting, result aggregation, seed helpers
│   └── baselines.py       # Heuristic route design baselines
├── networks/              # Network datasets (Bloomington, Sioux Falls)
├── plots/                 # Visualization utilities
└── uxsim/                 # Core UXsim simulator (upstream code)
```

## UXsim Integration

- `TransitEnv` wraps UXsim `World` objects so the agent receives node/edge tensors plus passenger progress indicators.
- Cached tensors keep UXsim's static topology on GPU while per-step demand features stream in.
- UXsim analyzers generate the PNG network snapshots and JSON summaries found in `training_data/`.

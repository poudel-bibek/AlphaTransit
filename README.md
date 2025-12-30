
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
- Graph-attention PPO agent that proposes complete route sets and scores coverage, wait time, and overlap.
- Baseline heuristics (`random_walk`, `demand_cover`, `shortest_path`, `reward_max`, `real_world`) for comparison.
- Cached PyG data pipelines, demand visualizations, and rich WANDB logging.

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

### 2. Train a new RL policy

```bash
python main.py --gpu --anneal_lr
```

Training defaults run on `bloomington` with `--mode=train` and `--num_episodes=2000` (~1 000 000 simulator steps). Adjust `--num_episodes` if you want a shorter run (e.g., `--num_episodes=50` ≈ 25 000 steps).

Training artifacts land in `training_data/<timestamp>/` with policy checkpoints under `policies/` and summary CSVs in `results/`.

### 3. Evaluate a saved policy

```bash
python main.py --mode=eval --network=bloomington --saved_policy_path=training_data/policies/policy_up_10_ep_50.pth --num_eval_runs=5 --save_dir=eval_runs
```

Aggregated metrics (such as mean wait time, transfer rate, coverage) are saved to `eval_runs/summary.json`.

### 4. Optional: Log to Weights & Biases

```bash
python main.py --mode=train --wandb_project=transit_design --wandb_entity=my-team
```

Disable logging anytime with `--wandb_off`.

## Project Structure

```
├── main.py                # CLI for train/eval/baseline workflows
├── rl/
│   ├── env.py             # TransitEnv: wraps UXsim worlds and bus operations
│   ├── models.py          # GATv2 actor-critic definitions
│   ├── ppo_agent.py       # PPO implementation and rollout memory
│   ├── env_utils.py       # Plotting, result aggregation, seed helpers
│   └── baselines.py       # Heuristic route design baselines
├── plots/
│   ├── plot_results.py    # Utilities for comparing training/eval outputs
│   └── plot_networks.py   # Network plots
└── uxsim/                 # Core UXsim simulator (upstream code)
```

## UXsim Integration

- `TransitEnv` wraps UXsim `World` objects so the agent receives node/edge tensors plus passenger progress indicators.
- Cached tensors keep UXsim's static topology on GPU while per-step demand features stream in.
- UXsim analyzers generate the PNG network snapshots and JSON summaries found in `training_data/`.

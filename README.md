
# Learning to Design City-scale Transit Routes using Reinforcement Learning

This repository builds on top of [UXsim](https://github.com/toruseo/UXsim) to study network-level transit design with deep reinforcement learning. We retain the complete UXsim simulator while adding graph-based policies, evaluation tooling, and standardized transit demand data.

## Highlights

- Built on UXsim’s mesoscopic traffic engine for fast, numerically stable simulations.
- Graph-attention PPO agent that proposes complete route sets and scores coverage, wait time, and overlap.
- Baseline heuristics (`random_walk`, `demand_cover`, `shortest_path`, `reward_max`, `real_world`) for comparison.
- Cached PyG data pipelines, demand visualizations, and rich WANDB logging.

## Installation

- Create an isolated environment (optional):
  ```
  conda create -n transit python=3.12 -y
  conda activate transit
  ```
- Install the project in editable mode:
  ```
  pip install -e .
  ```

## Quick Start

### 1. Environment setup

```
conda activate transit
```

`pip install -e .` installs UXsim and the RL tooling in editable mode so you can iterate on the environment and policies together.

### 2. Verify installation with a heuristic baseline

```
python main.py --mode=baseline --baseline_type=demand_cover --network=bloomington --num_eval_runs=1 --save_animations
```

This runs a single evaluation on the Bloomington network, generates 16 candidate routes (the default), and writes animation gif to `training_data/`.

### 3. Train a new RL policy

```
python main.py --gpu --anneal_lr
```

Training defaults run on `bloomington` with `--mode=train` and `--num_episodes=2000` (~1 000 000 simulator steps). Adjust `--num_episodes` if you want a shorter run (e.g., `--num_episodes=50` ≈ 25 000 steps).

Training artifacts land in `training_data/<timestamp>/` with policy checkpoints under `policies/` and summary CSVs in `results/`.

### 4. Evaluate a saved policy

```
python main.py --mode=eval --network=bloomington --saved_policy_path=training_data/policies/policy_up_10_ep_50.pth --num_eval_runs=5 --save_dir=eval_runs
```

Aggregated metrics (such as mean wait time, transfer rate, coverage) are saved to `eval_runs/summary.json`.

### 5. Optional: Log to Weights & Biases

```
python main.py --mode=train --wandb_project=transit_design --wandb_entity=my-team
```

Disable logging anytime with `--wandb_off`.

## Project Structure

```
├── main.py                # CLI for train/eval/baseline workflows
├── rl/
│   ├── env.py             # TransitEnv: wraps UXsim worlds and bus operations
│   ├── models.py          # GATv2 actor-critic definitions
│   ├── ppo.py             # PPO implementation and rollout memory
│   ├── env_utils.py       # Plotting, result aggregation, seed helpers
│   └── baselines.py       # Heuristic route design baselines
├── networks/              # Processed network + demand CSVs
├── plot_results.py        # Utilities for comparing training/eval outputs
└── uxsim/                 # Core UXsim simulator (upstream code)
```

The Bloomington dataset under `networks/bloomington/` ships with pre-processed files—`bloomington_nodes_standard.csv`, `bloomington_links_standard.csv`, and `bloomington_demand_standard.csv`—ready for immediate training and evaluation.

## UXsim Integration

- `TransitEnv` wraps UXsim `World` objects so the agent receives node/edge tensors plus passenger progress indicators.
- Cached tensors keep UXsim’s static topology on GPU while per-step demand features stream in.
- UXsim analyzers generate the PNG network snapshots and JSON summaries found in `training_data/`.

## Dataset & Visualization Utilities

- `plot_results.py` converts evaluation JSON outputs into ridership charts and travel-time histograms.
- Routes, passenger counts, and wait times are saved per seed inside `training_data/<timestamp>/seed_<id>/`.
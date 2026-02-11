#!/usr/bin/env bash
# =============================================================================
# scripts.sh — Command reference for AlphaTransit
#
# This file is NOT meant to be executed as a whole.
# Copy-paste individual commands as needed.
# =============================================================================

set -euo pipefail
echo "This file is a command reference — copy-paste individual commands." && exit 0

# =============================================================================
# 1. Baselines — Alpha 0.3
# =============================================================================

# --- transit_center init ---
python main.py --mode=baseline --baseline_type=random_walk    --save_animations --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=demand_cover   --save_animations --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=shortest_path  --save_animations --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=reward_max     --save_animations --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=real_world     --save_animations --route_init=transit_center --alpha=0.3

# --- random init ---
python main.py --mode=baseline --baseline_type=random_walk    --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=demand_cover   --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=shortest_path  --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=reward_max     --save_animations --route_init=random --alpha=0.3

# =============================================================================
# 2. Baselines — Alpha 1.0
# =============================================================================

# --- transit_center init ---
python main.py --mode=baseline --baseline_type=random_walk    --save_animations --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=demand_cover   --save_animations --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=shortest_path  --save_animations --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=reward_max     --save_animations --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=real_world     --save_animations --route_init=transit_center --alpha=1.0

# --- random init ---
python main.py --mode=baseline --baseline_type=random_walk    --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=demand_cover   --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=shortest_path  --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=reward_max     --save_animations --route_init=random --alpha=1.0

# =============================================================================
# 3. Genetic Algorithm
# =============================================================================

python main.py --mode=baseline --baseline_type=genetic --alpha=0.3 --ga_population=50 --ga_generations=100 --ga_num_workers=4
python main.py --mode=baseline --baseline_type=genetic --alpha=1.0 --ga_population=50 --ga_generations=100 --ga_num_workers=4

# =============================================================================
# 4. PPO Training
# =============================================================================

python main.py --algorithm ppo --gpu --alpha=0.3
python main.py --algorithm ppo --gpu --alpha=1.0

# =============================================================================
# 5. MCTS Training
# =============================================================================

# Alpha 0.3
python main.py --algorithm mcts --gpu --alpha=0.3

# Alpha 1.0
python main.py --algorithm mcts --gpu --alpha=1.0

# =============================================================================
# 6. Hyperparameter Sweeps
# =============================================================================

python sweep.py --algorithm ppo
python sweep.py --algorithm mcts

# =============================================================================
# 7. Reward Ablation Experiments (Experiment 1)
#
# 4 reward modes x 2 alpha values = 8 runs
# Uses best-known hyperparams per alpha from sweep insights.
# =============================================================================

# --- Alpha 0.3 ---
python main.py --algorithm ppo --gpu --alpha=0.3 --reward_mode=terminal_only
python main.py --algorithm ppo --gpu --alpha=0.3 --reward_mode=terminal_intermediate_raw_early_stop
python main.py --algorithm ppo --gpu --alpha=0.3 --reward_mode=terminal_intermediate_delta_early_stop
python main.py --algorithm ppo --gpu --alpha=0.3 --reward_mode=terminal_intermediate_delta_no_early_stop

# --- Alpha 1.0 ---
python main.py --algorithm ppo --gpu --alpha=1.0 --reward_mode=terminal_only
python main.py --algorithm ppo --gpu --alpha=1.0 --reward_mode=terminal_intermediate_raw_early_stop
python main.py --algorithm ppo --gpu --alpha=1.0 --reward_mode=terminal_intermediate_delta_early_stop
python main.py --algorithm ppo --gpu --alpha=1.0 --reward_mode=terminal_intermediate_delta_no_early_stop

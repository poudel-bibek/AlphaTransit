#!/usr/bin/env bash
# ============================================================
# NeurIPS experiment runs (JK connections + temp annealing)
# Each command should be run in a separate terminal.
# Baselines are unaffected (no NN) — no need to re-run.
# ============================================================

# --- 1. PPO Reward Ablation Sweeps (saves policies) ---
# 8 runs per alpha (4 reward modes × 2 seeds), ~5h each

# Alpha 0.3
python sweep.py --algorithm ppo --alpha 0.3

# Alpha 1.0
python sweep.py --algorithm ppo --alpha 1.0

# --- 2. MCTS n_iter Scaling Sweeps ---
# 5 runs per alpha (n_iter = 100, 200, 300, 400, 500)
# NOTE: Uncomment sweeps 7/8 and comment out 9/10 in sweep.py first

# Alpha 0.3
python sweep.py --algorithm mcts --alpha 0.3

# Alpha 1.0
python sweep.py --algorithm mcts --alpha 1.0

# --- 3. MCTS episodes_per_iter Scaling Sweeps ---
# 3 runs per alpha (eps = 8, 16, 24 at n_iter=200, ~1M env steps)
# NOTE: Uncomment sweeps 9/10 and comment out 7/8 in sweep.py first

# Alpha 0.3
python sweep.py --algorithm mcts --alpha 0.3

# Alpha 1.0
python sweep.py --algorithm mcts --alpha 1.0

from typing import Any, Dict, Optional, Sequence
import argparse
import os
import random

import numpy as np
import torch


def train(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run the full training loop over many updates and timesteps.
    Handles data collection, optimization, and periodic evaluation.
    Implementation will be added in subsequent steps.
    """
    pass


def eval(config: Dict[str, Any]) -> Dict[str, float]:  # noqa: A003
    """
    Evaluate a trained policy over fixed episodes.
    Computes metrics and optionally records trajectories.
    Details are left to the implementation.
    """
    pass


def set_global_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for CLI usage.
    Define common RL, system, and sweepable hyperparameters.
    """
    parser = argparse.ArgumentParser(description="RL training/evaluation entrypoint")
    # Arguments unlikely to change during sweeps (static)
    parser.add_argument("--network", choices=["sioux_falls", "laval", "rivera", "mumford3"], default="sioux_falls", help="Network selection")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available; defaults to CPU when not set or unavailable")
    parser.add_argument("--horizon", type=int, default=3600, help="Simulation horizon")
    parser.add_argument("--delta_t", type=float, default=1, help="Simulation time step")
    parser.add_argument("--delta_n", type=int, default=5, help="Simulation platoon size")

    # Arguments likely to change during sweeps (tunable hyperparameters)
    parser.add_argument("--total_timesteps", type=int, default=10000, help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Parse arguments and dispatch to train or eval routines.
    Serves as the CLI entry point for single-run experiments.
    Avoids W&B sweep-specific logic by design.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config: Dict[str, Any] = vars(args)

    # Set seeds as the very first thing after parsing CLI
    set_global_seeds(args.seed)

    # Set compute device based on --gpu flag
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    config["device"] = device

    if args.mode == "train":
        train(config)
    else:
        eval(config)


if __name__ == "__main__":
    main()



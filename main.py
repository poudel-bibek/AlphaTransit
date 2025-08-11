from typing import Any, Dict, Optional, Sequence
import argparse
import math
import random


class Application:
    """
    Container for high-level training orchestration.
    Holds shared resources such as envs, models, and config.
    Only the interface is defined here.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Prepare application-level state and dependency wiring.
        Accept configuration that will be forwarded to components.
        No heavy setup is performed in this skeleton.
        """
        pass


def train(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run the full training loop over many updates and timesteps.
    Handles data collection, optimization, and periodic evaluation.
    Implementation will be added in subsequent steps.
    """
    print("[train] Starting training with config:")
    for key, value in sorted(config.items()):
        print(f"  - {key}: {value}")

    # Dummy training step to illustrate flow
    _dummy_training_iterations = max(1, int(config.get("total_timesteps", 1000) // 1000))
    print(f"[train] Running {_dummy_training_iterations} dummy iterations...")

    # Call eval to produce the target metric that sweeps will track
    metrics = eval(config)
    print("[train] Completed. Metrics:", metrics)
    return metrics


def eval(config: Dict[str, Any]) -> Dict[str, float]:  # noqa: A003
    """
    Evaluate a trained policy over fixed episodes.
    Computes metrics and optionally records trajectories.
    Details are left to the implementation.
    """
    seed = int(config.get("seed", 0))
    lr = float(config.get("lr", 3e-4))
    num_envs = int(config.get("num_envs", 4))
    clip_coef = float(config.get("clip_coef", 0.2))

    random.seed(seed)

    # A synthetic metric shaped by hyperparameters (purely illustrative)
    # Lower is better, to match goal: "minimize"
    stability = max(0.0, 1.0 - clip_coef)
    complexity = math.log10(max(lr, 1e-9)) + 9.0  # 0..something positive
    scale = 100.0 / max(1, int(math.sqrt(max(1, num_envs))))
    noise = random.uniform(-1.0, 1.0)

    avg_ped_arrival = max(0.0, scale * (1.0 + complexity) * (1.0 + 0.5 * stability) + noise)
    metrics = {"evals/avg_ped_arrival": float(avg_ped_arrival)}
    print(f"[eval] evals/avg_ped_arrival = {metrics['evals/avg_ped_arrival']:.4f}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for CLI usage.
    Define common RL and system arguments in one place.
    No arguments are registered in the skeleton.
    """
    parser = argparse.ArgumentParser(description="RL training/evaluation entrypoint")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=10000, help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of vectorized environments")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Compute device selection")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index if using GPU")
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

    if args.mode == "train":
        train(config)
    else:
        eval(config)


if __name__ == "__main__":
    main()



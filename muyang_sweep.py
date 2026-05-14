from __future__ import annotations

import argparse
import gc
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import wandb

from alpha import train as alpha_train
from config import BEST_PARAMS, build_arg_parser, normalize_config, set_global_seeds
from rl.parallel_env import _cap_worker_threads


DEFAULT_ALPHA_VALUES = [round(i / 10, 1) for i in range(1, 11)]
MUYANG_N_ITER = 200
MUYANG_EPISODES_PER_ITER = 16
MUYANG_MAX_ITERATIONS = 308
MUYANG_NUM_MCTS_WORKERS = 8

TABLE_COLUMNS = [
    "alpha",
    "lr",
    "c_puct",
    "n_iter",
    "episodes_per_iter",
    "max_iterations",
    "num_mcts_workers",
    "num_gat_blocks",
    "batch_size",
    "buffer_capacity",
    "activation",
    "dirichlet_alpha",
    "train_steps_per_iter",
]


def parse_alpha_values(value: str) -> list[float]:
    """Parse comma-separated alpha values and normalize to one decimal place."""
    alphas = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        alpha = round(float(item), 1)
        if alpha <= 0.0 or alpha > 1.0:
            raise argparse.ArgumentTypeError(
                f"alpha values must be in (0, 1], got {alpha}"
            )
        alphas.append(alpha)

    if not alphas:
        raise argparse.ArgumentTypeError("at least one alpha value is required")

    return alphas


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.1f}".replace(".", "_")


def default_training_config() -> dict[str, Any]:
    """
    Start from the same argparse defaults as main.py, then force AlphaTransit train mode.
    """
    parser = build_arg_parser()
    config = vars(parser.parse_args([]))
    config.update(
        {
            "algorithm": "alphatransit",
            "mode": "train",
            "gpu": True,
            "wandb_off": True,
            "apply_best_params": True,
        }
    )
    return normalize_config(config)


def build_run_config(
    *,
    alpha: float,
    base_config: dict[str, Any],
    best_params: dict[str, Any],
    save_root: Path,
    seed: int,
    gpu: bool,
    wandb_off: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    network: str,
    best_alpha_source: float,
) -> dict[str, Any]:
    """
    Build one AlphaTransit run config.

    The selected best-param row is copied into every config before alpha is set,
    so alpha is the only model/objective parameter that changes across runs.
    """
    config = deepcopy(base_config)
    config.update(best_params)
    config.update(
        {
            "alpha": alpha,
            "algorithm": "alphatransit",
            "mode": "train",
            "n_iter": MUYANG_N_ITER,
            "episodes_per_iter": MUYANG_EPISODES_PER_ITER,
            "max_iterations": MUYANG_MAX_ITERATIONS,
            "num_mcts_workers": MUYANG_NUM_MCTS_WORKERS,
            "network": network,
            "seed": seed,
            "gpu": gpu,
            "wandb_off": wandb_off,
            "save_dir": str(save_root / f"alpha_{alpha_slug(alpha)}"),
            "muyang_best_alpha_source": best_alpha_source,
        }
    )

    if wandb_project is not None:
        config["wandb_project"] = wandb_project
    if wandb_entity is not None:
        config["wandb_entity"] = wandb_entity

    return normalize_config(config)


def format_markdown_table(configs: list[dict[str, Any]]) -> str:
    rows = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |",
    ]
    for config in configs:
        values = []
        for column in TABLE_COLUMNS:
            value = config[column]
            if isinstance(value, float):
                value = f"{value:g}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_wandb_sweep_config(
    *,
    alpha_values: list[float],
    fixed_config: dict[str, Any],
    best_alpha_source: float,
) -> dict[str, Any]:
    """
    Build a real WandB grid sweep config.

    Alpha is the only parameter with multiple values; all other entries are
    fixed so the run set appears as a single 10-run sweep in WandB.
    """
    fixed_parameters = {
        column: {"value": fixed_config[column]}
        for column in TABLE_COLUMNS
        if column != "alpha"
    }
    fixed_parameters.update(
        {
            "algorithm": {"value": "alphatransit"},
            "mode": {"value": "train"},
            "network": {"value": fixed_config["network"]},
            "seed": {"value": fixed_config["seed"]},
            "route_init": {"value": fixed_config["route_init"]},
            "transit_center_node": {"value": fixed_config["transit_center_node"]},
            "num_routes": {"value": fixed_config["num_routes"]},
            "max_route_length": {"value": fixed_config["max_route_length"]},
            "min_route_length": {"value": fixed_config["min_route_length"]},
            "mcts_batch_size": {"value": fixed_config["mcts_batch_size"]},
            "mcts_eval_every": {"value": fixed_config["mcts_eval_every"]},
            "num_eval_runs": {"value": fixed_config["num_eval_runs"]},
            "temp_schedule": {"value": fixed_config["temp_schedule"]},
            "muyang_best_alpha_source": {"value": best_alpha_source},
        }
    )

    return {
        "name": "muyang_sweep",
        "method": "grid",
        "metric": {
            "name": "eval/episode_terminal_reward",
            "goal": "maximize",
        },
        "parameters": {
            "alpha": {"values": alpha_values},
            **fixed_parameters,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train AlphaTransit policies across alpha values while holding the "
            "selected best-parameter row fixed."
        )
    )
    parser.add_argument(
        "--alpha-values",
        type=parse_alpha_values,
        default=DEFAULT_ALPHA_VALUES,
        help="Comma-separated alpha values. Default: 0.1,0.2,...,1.0",
    )
    parser.add_argument(
        "--best-alpha-source",
        type=float,
        choices=sorted(BEST_PARAMS["alphatransit"].keys()),
        default=0.3,
        help=(
            "Which AlphaTransit BEST_PARAMS row to pin for every run. "
            "Default: 0.3"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--network",
        choices=["bloomington", "sioux_falls"],
        default="bloomington",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./training_data/muyang_sweep"),
        help="Root directory for per-alpha training outputs.",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        default=True,
        help="Use CUDA for AlphaTransit training. Enabled by default.",
    )
    parser.add_argument(
        "--no-gpu",
        dest="gpu",
        action="store_false",
        help="Disable CUDA. AlphaTransit MCTS training normally requires CUDA.",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb_off",
        action="store_false",
        default=False,
        help="Enable WandB logging.",
    )
    wandb_group.add_argument(
        "--wandb-off",
        dest="wandb_off",
        action="store_true",
        help="Disable WandB logging.",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--resume-sweep-id",
        type=str,
        default=None,
        help=(
            "Join an existing WandB sweep instead of creating a new one. "
            "Accepts either a bare sweep ID or entity/project/sweep_id."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Number of sweep runs for this agent. Defaults to the number of "
            "alpha values. Use 0 to run indefinitely."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the parameter table and sweep config without contacting WandB.",
    )
    return parser


def create_sweep_agent_train(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    best_params: dict[str, Any],
) -> Any:
    def agent_train() -> None:
        with wandb.init() as run:
            sampled = dict(wandb.config)
            alpha = round(float(sampled["alpha"]), 1)
            config = build_run_config(
                alpha=alpha,
                base_config=base_config,
                best_params=best_params,
                save_root=args.save_dir,
                seed=args.seed,
                gpu=args.gpu,
                wandb_off=False,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                network=args.network,
                best_alpha_source=args.best_alpha_source,
            )
            config.update(sampled)
            config["alpha"] = alpha
            config["save_dir"] = str(args.save_dir / f"alpha_{alpha_slug(alpha)}")
            config["wandb_off"] = False
            config["device"] = torch.device("cuda")

            run.name = f"muyang_alpha_{alpha_slug(alpha)}"
            wandb.config.update(
                {
                    "save_dir": config["save_dir"],
                    "wandb_off": False,
                    "device": str(config["device"]),
                },
                allow_val_change=True,
            )

            print(
                f"\nTraining AlphaTransit WandB sweep run for alpha={alpha:g}; "
                f"outputs under {config['save_dir']}"
            )
            set_global_seeds(config["seed"])
            alpha_train(config, is_sweep=True)

        gc.collect()
        torch.cuda.empty_cache()

    return agent_train


def main() -> None:
    args = build_parser().parse_args()

    if args.wandb_off and not args.dry_run:
        raise ValueError("WandB sweep registration requires WandB enabled.")
    if args.gpu and not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("AlphaTransit MCTS training requires --gpu with CUDA available.")
    if not args.gpu and not args.dry_run:
        raise RuntimeError("AlphaTransit MCTS training is expected to run with --gpu.")

    _cap_worker_threads()

    best_params = dict(BEST_PARAMS["alphatransit"][args.best_alpha_source])
    base_config = default_training_config()
    configs = [
        build_run_config(
            alpha=alpha,
            base_config=base_config,
            best_params=best_params,
            save_root=args.save_dir,
            seed=args.seed,
            gpu=args.gpu,
            wandb_off=args.wandb_off,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            network=args.network,
            best_alpha_source=args.best_alpha_source,
        )
        for alpha in args.alpha_values
    ]

    print(
        f"Using AlphaTransit BEST_PARAMS source alpha={args.best_alpha_source:g}; "
        "only alpha changes across WandB sweep runs."
    )
    print(format_markdown_table(configs))

    sweep_config = build_wandb_sweep_config(
        alpha_values=args.alpha_values,
        fixed_config=configs[0],
        best_alpha_source=args.best_alpha_source,
    )

    if args.dry_run:
        print("\nWandB sweep config summary:")
        print(f"method: {sweep_config['method']}")
        print(f"metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
        print(f"alpha values: {sweep_config['parameters']['alpha']['values']}")
        return

    project = args.wandb_project or base_config["wandb_project"]
    entity = args.wandb_entity or base_config.get("wandb_entity")

    if args.resume_sweep_id:
        if "/" in args.resume_sweep_id:
            sweep_id = args.resume_sweep_id
        else:
            if not entity:
                raise ValueError("Bare --resume-sweep-id requires --wandb-entity or WANDB_ENTITY")
            sweep_id = f"{entity}/{project}/{args.resume_sweep_id}"
        print(f"\nJoining existing WandB sweep: {sweep_id}")
    else:
        print(f"\nCreating WandB grid sweep in project={project!r}, entity={entity!r}")
        sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
        print(f"Created WandB sweep: {sweep_id}")

    agent_count = len(args.alpha_values) if args.count is None else args.count
    if agent_count == 0:
        agent_count = None

    wandb.agent(
        sweep_id,
        function=create_sweep_agent_train(
            args=args,
            base_config=base_config,
            best_params=best_params,
        ),
        count=agent_count,
    )


if __name__ == "__main__":
    main()

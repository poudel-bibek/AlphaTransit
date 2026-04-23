from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import FIGURES_DIR
    from . import analyze_routes, networks, routes, training
except ImportError:
    from common import FIGURES_DIR
    import analyze_routes
    import networks
    import routes
    import training


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified entry point for plot generation.")
    subparsers = parser.add_subparsers(dest="command")

    route_grid = subparsers.add_parser("route-grid", help="Build the route comparison figure.")
    route_grid.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    route_grid.add_argument("--max-cols", type=int, default=5)
    route_grid.add_argument("--output", type=Path)

    route_gif = subparsers.add_parser("route-gif", help="Build the route cycle GIF.")
    route_gif.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    route_gif.add_argument("--methods", type=_parse_csv_list)
    route_gif.add_argument("--no-tiles", action="store_true")
    route_gif.add_argument("--output", type=Path)

    deceptive = subparsers.add_parser("deceptive-gif", help="Build the deceptive overlap GIF.")
    deceptive.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    deceptive.add_argument("--method", default="Demand Cover")
    deceptive.add_argument("--counts", type=_parse_int_list, default=[1, 4, 8, 12, 16])
    deceptive.add_argument("--no-tiles", action="store_true")
    deceptive.add_argument("--output", type=Path)

    analyze = subparsers.add_parser("analyze-routes", help="Run structural route analysis.")
    analyze.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    analyze.add_argument("--csv", type=Path)

    network_panel = subparsers.add_parser("network-panel", help="Build the Bloomington network panel.")
    network_panel.add_argument("--output", type=Path)

    bloomington = subparsers.add_parser("bloomington-map", help="Build the Bloomington network/demand/routes figure.")
    bloomington.add_argument("--output", type=Path)

    laval = subparsers.add_parser("laval-network", help="Build the Laval network figure.")
    laval.add_argument("--output", type=Path)

    networks_all = subparsers.add_parser("networks-all", help="Build the full network figure bundle.")
    networks_all.add_argument("--output-dir", type=Path, default=FIGURES_DIR)

    training_overview = subparsers.add_parser("training-overview", help="Build the PPO ablation and training comparison figure.")
    training_overview.add_argument("--output", type=Path)

    training_curves = subparsers.add_parser("training-curves", help="Build AlphaTransit vs PPO training curves.")
    training_curves.add_argument("--output", type=Path)

    rl_ablation = subparsers.add_parser("rl-ablation", help="Build the PPO reward ablation figure.")
    rl_ablation.add_argument("--output", type=Path)

    mcts_scaling = subparsers.add_parser("mcts-scaling", help="Build the MCTS scaling and wall-clock figure.")
    mcts_scaling.add_argument("--output", type=Path)

    wall_clock = subparsers.add_parser("wall-clock", help="Build the wall-clock summary plot.")
    wall_clock.add_argument("--alpha", type=float, choices=[0.3, 1.0], default=1.0)
    wall_clock.add_argument("--output", type=Path)

    pareto = subparsers.add_parser("pareto", help="Build the Pareto comparison figure.")
    pareto.add_argument("--output", type=Path)

    sweep = subparsers.add_parser("sweep-convergence", help="Build the sweep convergence figure.")
    sweep.add_argument("--output", type=Path)

    scaling = subparsers.add_parser("scaling-laws", help="Build the scaling-laws figure.")
    scaling.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    scaling.add_argument("--output", type=Path)

    training_all = subparsers.add_parser("training-all", help="Build the default training/results figure bundle.")
    training_all.add_argument("--output-dir", type=Path, default=FIGURES_DIR)

    experiment_cmd = subparsers.add_parser("experiment", help="Run the wall-clock experiment.")
    experiment_cmd.add_argument("--alpha", type=float, choices=[0.3, 1.0])
    experiment_cmd.add_argument("--n-iter", type=int, choices=[100, 200, 300, 400, 500])
    experiment_cmd.add_argument("--method", choices=["pure_mcts", "alphatransit"])
    experiment_cmd.add_argument("--workers", type=int, default=8)
    experiment_cmd.add_argument("--resume", action="store_true")
    experiment_cmd.add_argument("--dry-run", action="store_true")

    all_cmd = subparsers.add_parser("all", help="Build the default network, route, and training bundles.")
    all_cmd.add_argument("--output-dir", type=Path, default=FIGURES_DIR)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "route-grid":
        output = args.output or (FIGURES_DIR / f"routes_alpha_{args.alpha}.pdf")
        routes.build_route_figure(args.alpha, output, max_cols=args.max_cols)
        return

    if args.command == "route-gif":
        output = args.output or (FIGURES_DIR / f"route_cycle_alpha_{args.alpha}.gif")
        routes.build_route_cycle_gif(
            output,
            alpha_key=args.alpha,
            method_names=args.methods,
            use_tiles=not args.no_tiles,
        )
        return

    if args.command == "deceptive-gif":
        output = args.output or (
            FIGURES_DIR
            / f"deceptive_landscape_alpha_{args.alpha}_{_slugify(args.method)}.gif"
        )
        routes.build_deceptive_landscape_gif(
            output,
            alpha_key=args.alpha,
            method_name=args.method,
            key_route_counts=args.counts,
            use_tiles=not args.no_tiles,
        )
        return

    if args.command == "analyze-routes":
        forwarded = ["--alpha", args.alpha]
        if args.csv:
            forwarded.extend(["--csv", str(args.csv)])
        analyze_routes.main(forwarded)
        return

    if args.command == "network-panel":
        networks.plot_network_panel(args.output or (FIGURES_DIR / "network_panel.pdf"))
        return

    if args.command == "bloomington-map":
        networks.plot_bloomington_map(args.output or (FIGURES_DIR / "bloomington_map.pdf"))
        return

    if args.command == "laval-network":
        networks.plot_laval_network(args.output or (FIGURES_DIR / "laval_network.pdf"))
        return

    if args.command == "networks-all":
        networks.generate_all(args.output_dir)
        return

    if args.command == "training-overview":
        training.plot_reward_ablation_and_training(
            args.output or (FIGURES_DIR / "ppo_reward_ablation_and_training.pdf")
        )
        return

    if args.command == "training-curves":
        training.plot_training_curves(args.output or (FIGURES_DIR / "training_curves.pdf"))
        return

    if args.command == "rl-ablation":
        training.plot_rl_ablation(args.output or (FIGURES_DIR / "rl_reward_ablation.pdf"))
        return

    if args.command == "mcts-scaling":
        training.plot_mcts_scaling_and_wallclock(
            args.output or (FIGURES_DIR / "mcts_scaling_and_wallclock.pdf")
        )
        return

    if args.command == "wall-clock":
        alpha_tag = str(args.alpha).replace(".", "_")
        training.plot_wall_clock_alpha(
            args.output or (FIGURES_DIR / f"wall_clock_alpha_{alpha_tag}.pdf"),
            alpha_value=args.alpha,
        )
        return

    if args.command == "pareto":
        training.plot_pareto(args.output or (FIGURES_DIR / "pareto_analysis.pdf"))
        return

    if args.command == "sweep-convergence":
        training.plot_sweep_convergence(args.output or (FIGURES_DIR / "sweep_convergence.pdf"))
        return

    if args.command == "scaling-laws":
        output = args.output or (FIGURES_DIR / f"scaling_laws_alpha_{args.alpha}.pdf")
        training.plot_scaling_laws(output, alpha_key=args.alpha)
        return

    if args.command == "training-all":
        training.generate_default(args.output_dir)
        return

    if args.command == "experiment":
        try:
            from . import experiment
        except ImportError:
            import experiment
        forwarded: list[str] = []
        if args.alpha is not None:
            forwarded.extend(["--alpha", str(args.alpha)])
        if args.n_iter is not None:
            forwarded.extend(["--n-iter", str(args.n_iter)])
        if args.method:
            forwarded.extend(["--method", args.method])
        if args.workers != 8:
            forwarded.extend(["--workers", str(args.workers)])
        if args.resume:
            forwarded.append("--resume")
        if args.dry_run:
            forwarded.append("--dry-run")
        experiment.main(forwarded)
        return

    if args.command == "all":
        output_dir = args.output_dir
        networks.generate_all(output_dir)
        training.generate_default(output_dir)
        routes.build_route_figure("0_3", output_dir / "routes_alpha_0_3.pdf")
        routes.build_route_figure("1_0", output_dir / "routes_alpha_1_0.pdf")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

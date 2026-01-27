"""
Route plotting utilities.

Best checkpoints used by default:
- PPO
  - alpha 0.3: sweep glamorous-sweep-1, eval_up_7700_step_992642
  - alpha 1.0: sweep lyric-sweep-1, eval_up_280_step_37348
- MCTS (AlphaTransit)
  - alpha 0.3: sweep glad-sweep-1, eval_up_345_step_378717
  - alpha 1.0: sweep glorious-sweep-1, eval_up_408_step_469476
- Genetic Algorithm
  - alpha 0.3: current_best (Gen 94/100)
  - alpha 1.0: current_best (Gen 81/100)
- Real-world baseline: networks/bloomington/bloomington_existing_routes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = REPO_ROOT / "networks"
RESULTS_DIR = REPO_ROOT / "training_data" / "ICML_results"
DEFAULT_OUTPUT = REPO_ROOT / "plots" / "routes_composite.png"

METHODS = (
    ("Real-world", "real_world", "#374151"),
    ("Genetic Algorithm", "genetic", "#fb7185"),
    ("End-to-End RL", "PPO", "#1d4ed8"),
    ("AlphaTransit", "INCOMPLETE_MCTS", "#22c55e"),
)

ROUTE_LINE_WIDTH = 1.35
ROUTE_ALPHA = 0.6
ROUTE_SHADOW_WIDTH = 3.6
ROUTE_SHADOW_ALPHA = 0.12
MARGIN_RATIO = 0.03
BASE_EDGE_COLOR = "#d1d5db"
BASE_EDGE_ALPHA = 0.45
BASE_EDGE_WIDTH = 0.9
PIN_IMAGE_PATH = REPO_ROOT / "plots" / "google_map_pin_256.png"
PIN_IMAGE = plt.imread(PIN_IMAGE_PATH) if PIN_IMAGE_PATH.exists() else None
PIN_ZOOM = 0.1

BEST_CHECKPOINTS = {
    "0_3": {
        "real_world": REPO_ROOT / "networks" / "bloomington" / "bloomington_existing_routes.json",
        "genetic": RESULTS_DIR / "0_3" / "genetic_Jan_25_14_30_25" / "current_best",
        "PPO": RESULTS_DIR
        / "0_3"
        / "PPO_Jan_23_11_56_38"
        / "eval_results"
        / "eval_up_7700_step_992642",
        "INCOMPLETE_MCTS": RESULTS_DIR
        / "0_3"
        / "INCOMPLETE_MCTS_Jan_23_12_22_20"
        / "eval_results"
        / "eval_up_345_step_378717",
    },
    "1_0": {
        "real_world": REPO_ROOT / "networks" / "bloomington" / "bloomington_existing_routes.json",
        "genetic": RESULTS_DIR / "1_0" / "genetic_Jan_25_14_28_17" / "current_best",
        "PPO": RESULTS_DIR
        / "1_0"
        / "PPO_Jan_23_12_02_18"
        / "eval_results"
        / "eval_up_280_step_37348",
        "INCOMPLETE_MCTS": RESULTS_DIR
        / "1_0"
        / "INCOMPLETE_MCTS_Jan_23_12_09_24"
        / "eval_results"
        / "eval_up_408_step_469476",
    },
}

ORDINAL_LABELS = ["first", "second", "third", "fourth", "fifth"]
SEED_SETS = {
    "real_world": [0, 0, 0, 0, 0],
    "genetic": [42, 44, 46, 48, 50],
    "PPO": [42, 44, 46, 48, 50],
    "INCOMPLETE_MCTS": [44, 45, 46, 47, 48],
}


def apply_label_suffix(path: Path, label: str) -> Path:
    suffix = f"_{label}"
    if path.stem.endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def seed_count() -> int:
    count = len(SEED_SETS["PPO"])
    for method_key, seeds in SEED_SETS.items():
        if len(seeds) != count:
            raise ValueError(f"Seed list length mismatch for {method_key}")
    return count


def seed_for_method(method_key: str, seed_index: int) -> int:
    if method_key not in SEED_SETS:
        raise ValueError(f"Unsupported method key for seeds: {method_key}")
    seeds = SEED_SETS[method_key]
    if seed_index < 0 or seed_index >= len(seeds):
        raise ValueError(f"Seed index out of range for {method_key}: {seed_index}")
    return seeds[seed_index]


def load_nodes(network: str) -> pd.DataFrame:
    nodes_path = NETWORKS_DIR / network / f"{network}_nodes_standard.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing network nodes file: {nodes_path}")
    return pd.read_csv(nodes_path, dtype={"name": str})


def load_links(network: str) -> pd.DataFrame:
    links_path = NETWORKS_DIR / network / f"{network}_links_standard.csv"
    if not links_path.exists():
        raise FileNotFoundError(f"Missing network links file: {links_path}")
    return pd.read_csv(links_path, dtype={"start": str, "end": str})


def load_routes(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    routes = []
    if isinstance(payload, dict):
        for _, nodes in sorted(payload.items()):
            if len(nodes) >= 2:
                routes.append([str(node) for node in nodes])
    elif isinstance(payload, list):
        for item in payload:
            nodes = None
            if isinstance(item, dict) and "nodes" in item:
                nodes = item["nodes"]
            elif isinstance(item, list):
                nodes = item
            if nodes is None or len(nodes) < 2:
                continue
            routes.append([str(node) for node in nodes])
    else:
        raise ValueError(f"Unsupported route format in {path}")
    if not routes:
        raise ValueError(f"No routes found in {path}")
    return routes


def alpha_to_folder(alpha: str) -> str:
    alpha = alpha.strip()
    if alpha in {"0.3", "0.30", "0_3"}:
        return "0_3"
    if alpha in {"1.0", "1", "1_0"}:
        return "1_0"
    raise ValueError(f"Unsupported alpha: {alpha}")


def resolve_routes_path(alpha_dir: str, method_key: str, seed: int, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"Override path does not exist: {override}")
        return override

    if alpha_dir not in BEST_CHECKPOINTS:
        raise ValueError(f"Unsupported alpha directory: {alpha_dir}")
    if method_key not in BEST_CHECKPOINTS[alpha_dir]:
        raise ValueError(f"Unsupported method key: {method_key}")

    base_path = BEST_CHECKPOINTS[alpha_dir][method_key]
    if method_key == "real_world":
        return base_path
    return base_path / f"seed_{seed}" / "designed_routes.json"


def compute_bounds(
    all_routes: Iterable[List[List[str]]],
    coords: Dict[str, Dict[str, float]],
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for routes in all_routes:
        for route in routes:
            for node in route:
                if node not in coords:
                    raise KeyError(f"Node {node} missing from network geometry")
                point = coords[node]
                xs.append(point["x"])
                ys.append(point["y"])
    if not xs or not ys:
        raise ValueError("No route coordinates found for bounds.")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin_x = max((x_max - x_min) * MARGIN_RATIO, 1.0)
    margin_y = max((y_max - y_min) * MARGIN_RATIO, 1.0)
    return x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y


def draw_pin(ax: plt.Axes, coords: Dict[str, Dict[str, float]], node_id: str | None) -> None:
    if node_id is None or PIN_IMAGE is None:
        return
    point = coords.get(node_id)
    if point is None:
        return
    pin_artist = AnnotationBbox(
        OffsetImage(PIN_IMAGE, zoom=PIN_ZOOM, resample=True),
        (point["x"], point["y"]),
        frameon=False,
        box_alignment=(0.5, 0.0),
        pad=0,
        zorder=10,
    )
    ax.add_artist(pin_artist)


def draw_basemap(ax: plt.Axes, links: pd.DataFrame, coords: Dict[str, Dict[str, float]]) -> None:
    for _, row in links.iterrows():
        start = coords.get(row["start"])
        end = coords.get(row["end"])
        if start is None or end is None:
            continue
        ax.plot(
            [start["x"], end["x"]],
            [start["y"], end["y"]],
            color=BASE_EDGE_COLOR,
            alpha=BASE_EDGE_ALPHA,
            linewidth=BASE_EDGE_WIDTH,
            solid_capstyle="round",
            zorder=0,
        )


def draw_routes(
    ax: plt.Axes,
    routes: Iterable[Sequence[str]],
    coords: Dict[str, Dict[str, float]],
    color: str,
    show_stops: bool,
) -> None:
    for route in routes:
        xs, ys = [], []
        for node in route:
            point = coords.get(node)
            if point is None:
                raise KeyError(f"Node {node} missing from network geometry")
            xs.append(point["x"])
            ys.append(point["y"])
        if len(xs) < 2:
            continue
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=ROUTE_SHADOW_WIDTH,
            alpha=ROUTE_SHADOW_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=ROUTE_LINE_WIDTH,
            alpha=ROUTE_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
        if show_stops:
            ax.scatter(xs, ys, s=8, color=color, alpha=0.5, linewidths=0, zorder=3)


def build_composite_grid(
    network: str,
    alphas: Sequence[str],
    seeds: Dict[str, int],
    output_path: Path,
    overrides: Dict[str, Path | None],
    show_stops: bool,
    pin_node: str | None,
    dpi: int,
) -> Path:
    nodes = load_nodes(network)
    links = load_links(network)
    coords = nodes.set_index("name")[["x", "y"]].to_dict("index")

    rows: List[List[Tuple[str, List[List[str]], str]]] = []
    for alpha in alphas:
        alpha_dir = alpha_to_folder(alpha)
        method_routes: List[Tuple[str, List[List[str]], str]] = []
        for label, key, color in METHODS:
            seed = seeds.get(key)
            if seed is None:
                raise ValueError(f"Missing seed for method: {key}")
            routes_path = resolve_routes_path(alpha_dir, key, seed, overrides.get(key))
            if not routes_path.exists():
                raise FileNotFoundError(f"Missing routes for {label} (alpha={alpha}): {routes_path}")
            routes = load_routes(routes_path)
            method_routes.append((label, routes, color))
        rows.append(method_routes)

    bounds = compute_bounds([routes for row in rows for _, routes, _ in row], coords)

    n_rows = len(rows)
    n_cols = len(METHODS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.8 * n_rows),
        constrained_layout=True,
    )

    if n_rows == 1 and n_cols == 1:
        axes_grid = [[axes]]
    elif n_rows == 1:
        axes_grid = [list(axes)]
    elif n_cols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    for row_idx, method_routes in enumerate(rows):
        for col_idx, (label, routes, color) in enumerate(method_routes):
            ax = axes_grid[row_idx][col_idx]
            ax.set_facecolor("#ffffff")
            draw_basemap(ax, links, coords)
            draw_routes(ax, routes, coords, color=color, show_stops=show_stops)
            draw_pin(ax, coords, pin_node)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.text(
                0.5,
                -0.06,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=16,
                color="#111827",
                weight="semibold",
                clip_on=False,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot designed routes in a grid")
    parser.add_argument("--network", default="bloomington", help="Network name")
    parser.add_argument(
        "--alpha",
        default="1.0",
        help="Alpha value (0.3 or 1.0). Use --alphas to plot multiple rows.",
    )
    parser.add_argument(
        "--alphas",
        help="Comma-separated alpha values to plot as rows, e.g. 0.3,1.0",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=1,
        help="Seed index (1-5) for per-method seed lists",
    )
    parser.add_argument(
        "--cycle-seeds",
        action="store_true",
        help="Render all seed indices (first through fifth)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--show-stops", action="store_true", help="Draw stop markers")
    parser.add_argument("--pin-node", default="96", help="Transit center node id for pin")
    parser.add_argument("--no-pin", action="store_true", help="Disable transit center pin")
    parser.add_argument("--real-world", type=Path, help="Override real-world routes JSON path")
    parser.add_argument("--genetic", type=Path, help="Override genetic routes JSON path")
    parser.add_argument("--ppo", type=Path, help="Override PPO routes JSON path")
    parser.add_argument("--mcts", type=Path, help="Override AlphaTransit routes JSON path")
    args = parser.parse_args()

    overrides = {
        "real_world": args.real_world,
        "genetic": args.genetic,
        "PPO": args.ppo,
        "INCOMPLETE_MCTS": args.mcts,
    }

    if args.alphas:
        alphas = [value.strip() for value in args.alphas.split(",") if value.strip()]
    else:
        alphas = [args.alpha]

    pin_node = None if args.no_pin else str(args.pin_node)

    total_seeds = seed_count()
    if args.cycle_seeds:
        seed_indices = list(range(1, total_seeds + 1))
    else:
        seed_indices = [args.seed_index]

    for seed_index in seed_indices:
        if seed_index < 1 or seed_index > total_seeds:
            raise ValueError(f"Seed index must be between 1 and {total_seeds}")
        ordinal_index = seed_index - 1
        seeds = {key: seed_for_method(key, ordinal_index) for _, key, _ in METHODS}
        if args.cycle_seeds:
            label = ORDINAL_LABELS[ordinal_index] if ordinal_index < len(ORDINAL_LABELS) else f"seed_{seed_index}"
            output_path = apply_label_suffix(args.output, label)
        else:
            output_path = args.output
        build_composite_grid(
            network=args.network,
            alphas=alphas,
            seeds=seeds,
            output_path=output_path,
            overrides=overrides,
            show_stops=args.show_stops,
            pin_node=pin_node,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

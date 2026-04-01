"""
Legacy: network visualization and route topology grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

sns.set_theme(style="white", context="talk")

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = REPO_ROOT / "networks"
DEFAULT_DESIGN_DIR = REPO_ROOT / "plots" / "networks_to_plot"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots"

FIGURE_SIZE = (13.8 * 1.15, 8.5 * 1.15)
BASE_EDGE_COLOR = "#b9c5d7"
BASE_EDGE_ALPHA = 0.92
BASE_EDGE_WIDTH = 1.32
NODE_BASE_COLOR = "#94a3b8"
NODE_BASE_SIZE = 20
ROUTE_NODE_SIZE = 60
ROUTE_NODE_EDGE_COLOR = "#f8fafc"
ROUTE_NODE_EDGE_WIDTH = 1.4
ROUTE_LINE_WIDTH = 2.1
ROUTE_LINE_ALPHA = 0.96
OVERLAY_ALPHA = 0.9
OVERLAY_LINE_WIDTH = 2.1
MARGIN_RATIO = 0.03

PIN_IMAGE_PATH = Path(__file__).resolve().parent / "google_map_pin_256.png"
PIN_IMAGE = plt.imread(PIN_IMAGE_PATH)
PIN_ZOOM = 0.14




def load_network_geometry(network_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load standardized node and link tables for a network.
    Return data frames with geometry and connectivity.
    Keep node identifiers as strings for route joins.
    """
    base_dir = NETWORKS_DIR / network_name
    nodes_path = base_dir / f"{network_name}_nodes_standard.csv"
    links_path = base_dir / f"{network_name}_links_standard.csv"
    if not nodes_path.exists() or not links_path.exists():
        raise FileNotFoundError(f"Cannot locate network assets under {base_dir}")

    nodes = pd.read_csv(nodes_path, dtype={"name": str})
    links = pd.read_csv(links_path, dtype={"start": str, "end": str})
    return nodes, links


def load_routes_from_json(path: Path) -> List[List[str]]:
    """
    Load ordered routes from JSON.
    Return a list of node-id sequences per route.
    Keep identifiers as strings for plotting.
    """
    with path.open("r", encoding="utf-8") as fp:
        payload: Dict[str, Sequence[str]] = json.load(fp)

    routes = []
    for _, nodes in sorted(payload.items()):
        if len(nodes) >= 2:
            routes.append([str(node) for node in nodes])
    return routes


def _compute_bounds(nodes: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Determine padded plot bounds.
    Keep a small margin so annotations never clip.
    """
    x_min, x_max = nodes["x"].min(), nodes["x"].max()
    y_min, y_max = nodes["y"].min(), nodes["y"].max()
    margin_x = max((x_max - x_min) * MARGIN_RATIO, 1.0)
    margin_y = max((y_max - y_min) * MARGIN_RATIO, 1.0)
    return x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y


def _draw_base_network(ax: plt.Axes, nodes: pd.DataFrame, links: pd.DataFrame) -> None:
    """
    Render the underlying street graph.
    Plot light edges first, then faint nodes for depth.
    """
    coords = nodes.set_index("name")[["x", "y"]].to_dict("index")
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
            zorder=1,
        )

    ax.scatter(nodes["x"], nodes["y"], s=NODE_BASE_SIZE, color=NODE_BASE_COLOR, alpha=0.55, linewidths=0, zorder=2)


def _draw_routes(
    ax: plt.Axes,
    routes: Iterable[Sequence[str]],
    coords: Dict[str, Dict[str, float]],
    palette: Sequence[Tuple[float, float, float]],
    line_width: float = ROUTE_LINE_WIDTH,
    line_alpha: float = ROUTE_LINE_ALPHA,
    node_size: float = ROUTE_NODE_SIZE,
    shadow_color: str = "#f1f5f9",
) -> None:
    """
    Overlay colored routes on top of the base network.
    Use smooth strokes with glowing node markers.
    """
    for idx, route in enumerate(routes):
        if len(route) < 2:
            continue
        color = palette[idx % len(palette)]
        xs, ys = [], []
        for node in route:
            point = coords.get(node)
            if point is None:
                raise KeyError(f"Node {node} missing from geometry")
            xs.append(point["x"])
            ys.append(point["y"])
        ax.plot(
            xs,
            ys,
            color=shadow_color,
            linewidth=line_width + 2.2,
            alpha=0.55,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2.8 + idx * 0.02,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=line_width,
            alpha=line_alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3 + idx * 0.02,
        )
        ax.scatter(
            xs,
            ys,
            s=node_size,
            color=color,
            edgecolors=ROUTE_NODE_EDGE_COLOR,
            linewidths=ROUTE_NODE_EDGE_WIDTH,
            zorder=4 + idx * 0.02,
            alpha=0.98,
        )
        # No additional start node styling to keep markers uniform.


def plot_single_network(
    routes: List[List[str]],
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    title: str,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """
    Plot a single transit design on the base network.
    Highlight supplied routes with vivid gradients.
    Save the finished figure to output_path.
    """
    coords = nodes.set_index("name")[["x", "y"]].to_dict("index")
    palette = sns.color_palette("mako", max(len(routes), 10))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _draw_base_network(ax, nodes, links)
    _draw_routes(ax, routes, coords, palette, shadow_color="#e2e8f0")

    x_min, x_max, y_min, y_max = _compute_bounds(nodes)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, color="#0f172a", fontsize=18, pad=6, weight="bold", y=0.975)

    if "96" in coords:
        center = coords["96"]
        pin_artist = AnnotationBbox(
            OffsetImage(PIN_IMAGE, zoom=PIN_ZOOM, resample=True),
            (center["x"], center["y"]),
            frameon=False,
            box_alignment=(0.5, 0.0),
            pad=0,
            zorder=12,
        )
        ax.add_artist(pin_artist)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overlay_networks(
    primary_routes: List[List[str]],
    secondary_routes: List[List[str]],
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    labels: Tuple[str, str],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """
    Plot two route sets side by side on the same base network layout.
    Left column corresponds to labels[0], right to labels[1].
    """
    coords = nodes.set_index("name")[["x", "y"]].to_dict("index")
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0] * 1.4, FIGURE_SIZE[1]), constrained_layout=False)
    fig.subplots_adjust(wspace=0.04)

    x_min, x_max, y_min, y_max = _compute_bounds(nodes)

    real_palette = sns.color_palette("mako", max(len(primary_routes), 8))
    rl_palette = sns.color_palette("viridis", max(len(secondary_routes), 8))
    configs = [
        (primary_routes, real_palette, labels[0], axes[0]),
        (secondary_routes, rl_palette, labels[1], axes[1]),
    ]

    for routes, palette, title, ax in configs:
        ax.set_facecolor("#ffffff")
        _draw_base_network(ax, nodes, links)
        _draw_routes(
            ax,
            routes,
            coords,
            palette,
            line_width=OVERLAY_LINE_WIDTH,
            line_alpha=OVERLAY_ALPHA,
            node_size=ROUTE_NODE_SIZE * 0.9,
            shadow_color="#e2e8f0",
        )

        if "96" in coords:
            center = coords["96"]
            pin_artist = AnnotationBbox(
                OffsetImage(PIN_IMAGE, zoom=PIN_ZOOM, resample=True),
                (center["x"], center["y"]),
                frameon=False,
                box_alignment=(0.5, 0.0),
                pad=0,
                zorder=12,
            )
            ax.add_artist(pin_artist)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5,
            0.98,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=22,
            color="#0f172a",
            fontweight="semibold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def main() -> None:
    """
    Generate single and overlay plots from design JSON files.
    Pair *_rl and *_real files automatically.
    Save all figures under the requested output directory.
    """
    parser = argparse.ArgumentParser(description="Plot transit network designs")
    parser.add_argument("--network", default="bloomington", help="Network name to load")
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=DEFAULT_DESIGN_DIR,
        help="Directory containing design JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots will be written",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for exports")
    args = parser.parse_args()

    nodes, links = load_network_geometry(args.network)
    design_dir = args.design_dir
    output_dir = args.output_dir
    design_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    design_files = list(design_dir.glob("*.json"))
    if not design_files:
        raise FileNotFoundError(f"No JSON designs found in {design_dir}")

    rl_files: Dict[str, Path] = {}
    real_files: Dict[str, Path] = {}
    for path in design_files:
        stem = path.stem
        if stem.endswith("_rl"):
            rl_files[stem[:-3]] = path
        elif stem.endswith("_real"):
            real_files[stem[:-5]] = path

    handled_prefixes = set()
    for base_key, rl_path in rl_files.items():
        routes_rl = load_routes_from_json(rl_path)
        plot_single_network(
            routes_rl,
            nodes,
            links,
            title=f"{base_key.replace('_', ' ').title()} – RL Design",
            output_path=output_dir / f"{base_key}_rl.png",
            dpi=args.dpi,
        )
        if base_key in real_files:
            routes_real = load_routes_from_json(real_files[base_key])
            plot_single_network(
                routes_real,
                nodes,
                links,
                title=f"{base_key.replace('_', ' ').title()} – Real Design",
                output_path=output_dir / f"{base_key}_real.png",
                dpi=args.dpi,
            )
            plot_overlay_networks(
                primary_routes=routes_real,
                secondary_routes=routes_rl,
                nodes=nodes,
                links=links,
                labels=("Real-world", "RL (Ours)"),
                output_path=output_dir / f"{base_key}_compare.png",
                dpi=args.dpi,
            )
            handled_prefixes.add(base_key)
        else:
            handled_prefixes.add(base_key)

    for base_key, path in real_files.items():
        if base_key in handled_prefixes:
            continue
        routes = load_routes_from_json(path)
        plot_single_network(
            routes,
            nodes,
            links,
            title=f"{base_key.replace('_', ' ').title()} – Real Design",
            output_path=output_dir / f"{base_key}_real.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

"""
Route plotting utilities.

Best checkpoints used by default:
- PPO
  - alpha 0.3: sweep glamorous-sweep-1, eval_up_7700_step_992642
  - alpha 1.0: sweep lyric-sweep-1, eval_up_280_step_37348
- MCTS (AlphaTransit)
  - For this route plot and training curves: alpha 0.3: sweep glad-sweep-1, eval_up_345_step_378717
  - For Table: sweep peach-sweep-1, eval_up_243_step_359104 (INCOMPLETE_MCTS_JAN_26)
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
    ("Real-World", "real_world", "#374151"),
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

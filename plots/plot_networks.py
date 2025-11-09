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


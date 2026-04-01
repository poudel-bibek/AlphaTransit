"""
Route topology comparison figures for NeurIPS paper.

Generates two PDFs (one per alpha value) showing designed transit route
topologies from configurable methods. Each panel overlays colored routes
on the Bloomington street network. Grid layout adapts to number of methods.

Outputs:
    - plots/routes_alpha0_3.pdf
    - plots/routes_alpha1_0.pdf

Usage:
    python plots/figure_routes.py
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
FS = 20
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': FS,
    'axes.labelsize': FS + 2,
    'axes.titlesize': FS + 2,
    'xtick.labelsize': FS - 1,
    'ytick.labelsize': FS - 1,
    'legend.fontsize': FS - 2,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.5,
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.4,
    'grid.linestyle': ':',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = REPO_ROOT / "networks"
RESULTS_DIR = REPO_ROOT / "training_data" / "ICML_results"
OUTPUT_DIR = REPO_ROOT / "plots"

TRANSIT_CENTER_NODE = "96"

# ---------------------------------------------------------------------------
# Configurable method list — comment out methods to exclude them.
# Each dict: name (display label), color, path_0_3, path_1_0.
# Paths point to the designed_routes.json for the BEST seed.
# ---------------------------------------------------------------------------
EXISTING_ROUTES = NETWORKS_DIR / "bloomington" / "bloomington_existing_routes.json"

METHODS = [
    {
        'name': 'Real-World',
        'color': '#374151',
        'path_0_3': EXISTING_ROUTES,
        'path_1_0': EXISTING_ROUTES,
    },
    {
        'name': 'Random Walk',
        'color': '#6366f1',
        'path_0_3': RESULTS_DIR / "0_3" / "random_walk_Jan_25_09_09_12" / "seed_50" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "random_walk_Jan_25_09_30_32" / "seed_44" / "designed_routes.json",
    },
    {
        'name': 'Demand Cover',
        'color': '#0ea5e9',
        'path_0_3': RESULTS_DIR / "0_3" / "demand_cover_Jan_25_09_10_56" / "seed_50" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "demand_cover_Jan_25_09_32_30" / "seed_50" / "designed_routes.json",
    },
    {
        'name': 'Shortest Path',
        'color': '#f59e0b',
        'path_0_3': RESULTS_DIR / "0_3" / "shortest_path_Jan_25_09_14_31" / "seed_44" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "shortest_path_Jan_25_10_03_34" / "seed_44" / "designed_routes.json",
    },
    {
        'name': 'Genetic Algorithm',
        'color': '#fb7185',
        'path_0_3': RESULTS_DIR / "0_3" / "genetic_Jan_25_14_30_25" / "current_best" / "seed_42" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "genetic_Jan_25_14_28_17" / "current_best" / "seed_42" / "designed_routes.json",
    },
    {
        'name': 'Bee Colony',
        'color': '#a855f7',
        'path_0_3': RESULTS_DIR / "0_3" / "bee_colony_Mar_28_08_46_46" / "seed_44" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "bee_colony_Mar_28_08_47_04" / "seed_50" / "designed_routes.json",
    },
    {
        'name': 'Neural Evolutionary',
        'color': '#14b8a6',
        'path_0_3': RESULTS_DIR / "0_3" / "neural_evolutionary_Mar_28_08_46_47" / "seed_50" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "neural_evolutionary_Mar_28_08_47_05" / "seed_56" / "designed_routes.json",
    },
    {
        'name': 'Pure MCTS',
        'color': '#ef4444',
        'path_0_3': RESULTS_DIR / "0_3" / "pure_mcts_Mar_29_00_05_00" / "seed_42" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "pure_mcts_Mar_28_17_00_00" / "seed_44" / "designed_routes.json",
    },
    {
        'name': 'End-to-End RL',
        'color': '#1d4ed8',
        'path_0_3': RESULTS_DIR / "0_3" / "PPO_Jan_23_11_56_38" / "eval_results" / "eval_up_7700_step_992642" / "seed_48" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "PPO_Jan_23_12_02_18" / "eval_results" / "eval_up_280_step_37348" / "seed_44" / "designed_routes.json",
    },
    {
        'name': 'AlphaTransit',
        'color': '#22c55e',
        'path_0_3': RESULTS_DIR / "0_3" / "MCTS_Jan_26_18_30_46" / "eval_results" / "eval_up_345_step_492447" / "seed_45" / "designed_routes.json",
        'path_1_0': RESULTS_DIR / "1_0" / "INCOMPLETE_MCTS_Jan_23_12_09_24" / "eval_results" / "eval_up_408_step_469476" / "seed_47" / "designed_routes.json",
    },
]

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------
ROUTE_LINE_WIDTH = 1.35
ROUTE_ALPHA = 0.6
ROUTE_SHADOW_WIDTH = 3.6
ROUTE_SHADOW_ALPHA = 0.12
BASE_EDGE_COLOR = "#d1d5db"
BASE_EDGE_ALPHA = 0.45
BASE_EDGE_WIDTH = 0.9
MARGIN_RATIO = 0.03


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------
def _grid_layout(n: int, max_cols: int) -> Tuple[int, int]:
    """Return (nrows, ncols) for n panels under a configurable column cap."""
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_nodes(network: str = "bloomington") -> pd.DataFrame:
    """Load node geometry table."""
    path = NETWORKS_DIR / network / f"{network}_nodes_standard.csv"
    return pd.read_csv(path, dtype={"name": str})


def load_links(network: str = "bloomington") -> pd.DataFrame:
    """Load link (edge) table."""
    path = NETWORKS_DIR / network / f"{network}_links_standard.csv"
    return pd.read_csv(path, dtype={"start": str, "end": str})


def load_routes(path: Path) -> List[List[str]]:
    """Load transit routes from JSON as lists of node-id strings.

    Handles dict format (designed routes) and list-of-dicts format
    (existing real-world routes).
    """
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    routes: List[List[str]] = []
    if isinstance(payload, dict):
        for _, nodes in sorted(payload.items()):
            if len(nodes) >= 2:
                routes.append([str(n) for n in nodes])
    elif isinstance(payload, list):
        for item in payload:
            nodes = None
            if isinstance(item, dict) and "nodes" in item:
                nodes = item["nodes"]
            elif isinstance(item, list):
                nodes = item
            if nodes is not None and len(nodes) >= 2:
                routes.append([str(n) for n in nodes])
    return routes


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_basemap(ax: plt.Axes, links: pd.DataFrame,
                 coords: Dict[str, Dict[str, float]]) -> None:
    """Draw the street network as light gray edges."""
    for _, row in links.iterrows():
        start = coords.get(row["start"])
        end = coords.get(row["end"])
        if start is None or end is None:
            continue
        ax.plot(
            [start["x"], end["x"]], [start["y"], end["y"]],
            color=BASE_EDGE_COLOR, alpha=BASE_EDGE_ALPHA,
            linewidth=BASE_EDGE_WIDTH, solid_capstyle="round", zorder=0,
        )


def draw_routes(ax: plt.Axes, routes: Sequence[Sequence[str]],
                coords: Dict[str, Dict[str, float]], color: str) -> None:
    """Overlay colored transit routes with shadow effect."""
    for route in routes:
        xs, ys = [], []
        for node in route:
            pt = coords.get(node)
            if pt is None:
                continue
            xs.append(pt["x"])
            ys.append(pt["y"])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color=color, linewidth=ROUTE_SHADOW_WIDTH,
                alpha=ROUTE_SHADOW_ALPHA, solid_capstyle="round",
                solid_joinstyle="round", zorder=1)
        ax.plot(xs, ys, color=color, linewidth=ROUTE_LINE_WIDTH,
                alpha=ROUTE_ALPHA, solid_capstyle="round",
                solid_joinstyle="round", zorder=2)


def draw_transit_center(ax: plt.Axes,
                        coords: Dict[str, Dict[str, float]]) -> None:
    """Mark the transit center (node 96) with a red pin."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    if TRANSIT_CENTER_NODE not in coords:
        return
    pin_path = Path(__file__).resolve().parent / 'pin_red.png'
    if not pin_path.exists():
        return
    pin_img = plt.imread(pin_path).copy()
    if pin_img.shape[2] == 4:
        pin_img[:, :, 3] = pin_img[:, :, 3] * 0.9
    else:
        import numpy as np
        alpha_ch = np.ones((*pin_img.shape[:2], 1), dtype=pin_img.dtype) * 0.9
        pin_img = np.concatenate([pin_img, alpha_ch], axis=2)
    pt = coords[TRANSIT_CENTER_NODE]
    ab = AnnotationBbox(
        OffsetImage(pin_img, zoom=0.162, resample=True),
        (pt["x"], pt["y"]), frameon=False,
        box_alignment=(0.5, 0.0), pad=0, zorder=10)
    ax.add_artist(ab)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------
def build_route_figure(alpha_key: str, output_path: Path,
                       max_cols: int = 5) -> None:
    """Build a multi-panel figure comparing route topologies at one alpha.

    Grid layout adapts to the number of active methods in METHODS.
    """
    active = [m for m in METHODS]
    n = len(active)
    if n == 0:
        print("  No methods configured, skipping.")
        return

    nodes = load_nodes()
    links = load_links()
    coords = nodes.set_index("name")[["x", "y"]].to_dict("index")

    path_key = f"path_{alpha_key}"

    # Load routes for each method
    all_xs, all_ys = [], []
    method_data: List[Tuple[str, List[List[str]], str]] = []
    for m in active:
        rpath = m[path_key]
        if not rpath.exists():
            raise FileNotFoundError(f"Missing routes: {rpath}")
        routes = load_routes(rpath)
        method_data.append((m['name'], routes, m['color']))
        for route in routes:
            for node in route:
                pt = coords.get(node)
                if pt:
                    all_xs.append(pt["x"])
                    all_ys.append(pt["y"])

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    mx = max((x_max - x_min) * MARGIN_RATIO, 1.0)
    my = max((y_max - y_min) * MARGIN_RATIO, 1.0)
    bounds = (x_min - mx, x_max + mx, y_min - my, y_max + my)

    nrows, ncols = _grid_layout(n, max_cols)
    panel_w, panel_h = 3.8, 3.2
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows

    # Use twice as many GridSpec columns so incomplete rows can be centered
    # while each panel keeps the same width as full-row panels.
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = GridSpec(nrows, 2 * ncols, figure=fig)
    all_axes = []

    remaining = n
    for row in range(nrows):
        row_count = min(ncols, remaining)
        start_slot = ncols - row_count
        for col in range(row_count):
            slot0 = start_slot + 2 * col
            all_axes.append(fig.add_subplot(gs[row, slot0:slot0 + 2]))
        remaining -= row_count

    for i, (label, routes, color) in enumerate(method_data):
        ax = all_axes[i]
        ax.set_facecolor("#FFFFFF")
        draw_basemap(ax, links, coords)
        draw_routes(ax, routes, coords, color)
        draw_transit_center(ax, coords)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.text(0.5, -0.04, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=FS,
                color="#111827", weight="semibold", clip_on=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="#FFFFFF", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    """Generate route topology PDFs for both alpha values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cols', type=int, default=5,
                        help='Maximum number of panels per row.')
    args = parser.parse_args()
    if args.max_cols < 1:
        raise ValueError('--max-cols must be at least 1')

    print("Figure: Route topology comparison")
    for alpha_key, filename in [("0_3", "routes_alpha0_3.pdf"),
                                ("1_0", "routes_alpha1_0.pdf")]:
        build_route_figure(alpha_key, OUTPUT_DIR / filename,
                           max_cols=args.max_cols)
    print("Done.")


if __name__ == "__main__":
    main()

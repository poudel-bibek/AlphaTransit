from __future__ import annotations

import functools
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

try:
    from .common import ASSETS_DIR, NETWORKS_DIR, NEURIPS_RESULTS_DIR, ROUTE_COLORS, apply_plot_style, maybe_to_web_mercator, save_figure
except ImportError:
    from common import ASSETS_DIR, NETWORKS_DIR, NEURIPS_RESULTS_DIR, ROUTE_COLORS, apply_plot_style, maybe_to_web_mercator, save_figure


FS = 20
TRANSIT_CENTER_NODE = "96"
ROUTE_LINE_WIDTH = 1.35
ROUTE_ALPHA = 0.6
ROUTE_SHADOW_WIDTH = 3.6
ROUTE_SHADOW_ALPHA = 0.12
BASE_EDGE_COLOR = "#d1d5db"
BASE_EDGE_ALPHA = 0.45
BASE_EDGE_WIDTH = 0.9
MARGIN_RATIO = 0.03

EXISTING_ROUTES = NETWORKS_DIR / "bloomington" / "bloomington_existing_routes.json"
CoordMap = Dict[str, tuple[float, float]]
ALPHA_RESULTS_DIRS = {
    "0_3": NEURIPS_RESULTS_DIR / "0_3",
    "1_0": NEURIPS_RESULTS_DIR / "1_0",
}


def _latest_matching_dir(parent: Path, prefix: str) -> Path:
    matches = sorted(
        (path for path in parent.glob(f"{prefix}*") if path.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No directory matching {prefix}* under {parent}")
    return matches[-1]


def _resolve_seed_route(base_dir: Path, seed: str = "seed_42") -> Path:
    direct = base_dir / seed / "designed_routes.json"
    if direct.exists():
        return direct

    matches = sorted(base_dir.rglob(f"{seed}/designed_routes.json"))
    if not matches:
        raise FileNotFoundError(
            f"No designed_routes.json found for {seed} under {base_dir}"
        )
    return matches[-1]


def _resolve_baseline_route(
    alpha_key: str,
    prefix: str,
    *,
    seed: str = "seed_42",
    extra_parts: Sequence[str] = (),
) -> Path:
    method_dir = _latest_matching_dir(ALPHA_RESULTS_DIRS[alpha_key], prefix)
    return _resolve_seed_route(method_dir.joinpath(*extra_parts), seed)


def _resolve_alphatransit_route(alpha_key: str, seed: str = "seed_42") -> Path:
    niter_dir = "nips_7_n_iter" if alpha_key == "0_3" else "nips_8_n_iter"
    base_dir = ALPHA_RESULTS_DIRS[alpha_key] / "alphatransit" / niter_dir / "n_iter_500"
    return _resolve_seed_route(base_dir, seed)

@functools.cache
def get_methods() -> List[Dict[str, object]]:
    return [
        {
            "name": "Real-World",
            "color": "#374151",
            "path_0_3": EXISTING_ROUTES,
            "path_1_0": EXISTING_ROUTES,
        },
        {
            "name": "Random Walk",
            "color": "#6366f1",
            "path_0_3": _resolve_baseline_route("0_3", "random_walk_"),
            "path_1_0": _resolve_baseline_route("1_0", "random_walk_"),
        },
        {
            "name": "Demand Cover",
            "color": "#0ea5e9",
            "path_0_3": _resolve_baseline_route("0_3", "demand_cover_"),
            "path_1_0": _resolve_baseline_route("1_0", "demand_cover_"),
        },
        {
            "name": "Shortest Path",
            "color": "#f59e0b",
            "path_0_3": _resolve_baseline_route("0_3", "shortest_path_"),
            "path_1_0": _resolve_baseline_route("1_0", "shortest_path_"),
        },
        {
            "name": "Genetic Algorithm",
            "color": "#fb7185",
            "path_0_3": _resolve_baseline_route("0_3", "genetic_", extra_parts=("current_best",)),
            "path_1_0": _resolve_baseline_route("1_0", "genetic_", extra_parts=("current_best",)),
        },
        {
            "name": "Bee Colony",
            "color": "#a855f7",
            "path_0_3": _resolve_baseline_route("0_3", "bee_colony_"),
            "path_1_0": _resolve_baseline_route("1_0", "bee_colony_"),
        },
        {
            "name": "Neural Evolutionary",
            "color": "#14b8a6",
            "path_0_3": _resolve_baseline_route("0_3", "neural_evolutionary_"),
            "path_1_0": _resolve_baseline_route("1_0", "neural_evolutionary_"),
        },
        {
            "name": "Pure MCTS",
            "color": "#ef4444",
            "path_0_3": _resolve_baseline_route("0_3", "mcts_pure_"),
            "path_1_0": _resolve_baseline_route("1_0", "mcts_pure_"),
        },
        {
            "name": "End-to-End RL",
            "color": "#1d4ed8",
            "path_0_3": _resolve_baseline_route("0_3", "end_to_end_rl_", extra_parts=("train_seed_42",)),
            "path_1_0": _resolve_baseline_route("1_0", "end_to_end_rl_", extra_parts=("train_seed_42",)),
        },
        {
            "name": "AlphaTransit",
            "color": "#22c55e",
            "path_0_3": _resolve_alphatransit_route("0_3"),
            "path_1_0": _resolve_alphatransit_route("1_0"),
        },
    ]


def load_nodes(network: str = "bloomington") -> pd.DataFrame:
    path = NETWORKS_DIR / network / f"{network}_nodes_standard.csv"
    return pd.read_csv(path, dtype={"name": str})


def load_links(network: str = "bloomington") -> pd.DataFrame:
    path = NETWORKS_DIR / network / f"{network}_links_standard.csv"
    return pd.read_csv(path, dtype={"start": str, "end": str})


def load_routes(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    route_sets: List[List[str]] = []
    if isinstance(payload, dict):
        for _, nodes in sorted(payload.items()):
            if len(nodes) >= 2:
                route_sets.append([str(node) for node in nodes])
    elif isinstance(payload, list):
        for item in payload:
            nodes = item.get("nodes") if isinstance(item, dict) else item
            if nodes is not None and len(nodes) >= 2:
                route_sets.append([str(node) for node in nodes])
    return route_sets


def get_method_config(method_name: str) -> Dict[str, object]:
    for method in get_methods():
        if method["name"] == method_name:
            return method
    raise KeyError(f"Unknown route method: {method_name}")


def get_method_routes(method_name: str, alpha_key: str) -> List[List[str]]:
    method = get_method_config(method_name)
    path = method[f"path_{alpha_key}"]
    if not path.exists():
        raise FileNotFoundError(f"Missing route file for {method_name} ({alpha_key}): {path}")
    return load_routes(path)


def _grid_layout(n: int, max_cols: int) -> Tuple[int, int]:
    ncols = min(n, max_cols)
    return math.ceil(n / ncols), ncols


def draw_basemap(
    ax: plt.Axes,
    links: pd.DataFrame,
    coords: CoordMap,
    *,
    color: str = BASE_EDGE_COLOR,
    alpha: float = BASE_EDGE_ALPHA,
    linewidth: float = BASE_EDGE_WIDTH,
    zorder: int = 0,
) -> None:
    segments = [
        ((start[0], start[1]), (end[0], end[1]))
        for start, end in (
            (coords.get(row["start"]), coords.get(row["end"]))
            for _, row in links.iterrows()
        )
        if start is not None and end is not None
    ]
    if segments:
        ax.add_collection(LineCollection(segments, colors=color, alpha=alpha, linewidths=linewidth, capstyle="round", zorder=zorder))


def draw_routes(
    ax: plt.Axes,
    routes: Sequence[Sequence[str]],
    coords: CoordMap,
    color: str,
    *,
    use_palette: bool = False,
    line_width: float = ROUTE_LINE_WIDTH,
    alpha: float = ROUTE_ALPHA,
) -> None:
    for idx, route in enumerate(routes):
        xs, ys = [], []
        for node in route:
            pt = coords.get(node)
            if pt is None:
                continue
            xs.append(pt[0])
            ys.append(pt[1])
        if len(xs) < 2:
            continue
        route_color = ROUTE_COLORS[idx % len(ROUTE_COLORS)] if use_palette else color
        ax.plot(
            xs,
            ys,
            color=route_color,
            linewidth=ROUTE_SHADOW_WIDTH if not use_palette else line_width,
            alpha=ROUTE_SHADOW_ALPHA if not use_palette else alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
        ax.plot(
            xs,
            ys,
            color=route_color,
            linewidth=line_width,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2 if not use_palette else 3,
        )


def draw_transit_center(ax: plt.Axes, coords: CoordMap, *, zoom: float = 0.1458, node_id: str = TRANSIT_CENTER_NODE, pin: str = "pin_red.png") -> None:
    if node_id not in coords:
        return
    pin_path = ASSETS_DIR / pin
    if not pin_path.exists():
        return
    pin_img = plt.imread(pin_path).copy()
    if pin_img.shape[2] == 4:
        pin_img[:, :, 3] = pin_img[:, :, 3] * 0.9
    pt = coords[node_id]
    artist = AnnotationBbox(
        OffsetImage(pin_img, zoom=zoom, resample=True),
        pt,
        frameon=False,
        box_alignment=(0.5, 0.0),
        pad=0,
        zorder=10,
    )
    ax.add_artist(artist)


def build_route_figure(alpha_key: str, output_path: Path, max_cols: int = 5, fs: int = FS) -> None:
    apply_plot_style(fs)
    active = list(get_methods())
    if not active:
        return

    nodes = load_nodes()
    links = load_links()
    coords = _coords_dict(nodes)
    path_key = f"path_{alpha_key}"

    all_xs, all_ys = [], []
    method_data: List[Tuple[str, List[List[str]], str]] = []
    for method in active:
        rpath = method[path_key]
        if not rpath.exists():
            raise FileNotFoundError(f"Missing routes: {rpath}")
        routes = load_routes(rpath)
        method_data.append((method["name"], routes, method["color"]))
        for route in routes:
            for node in route:
                pt = coords.get(node)
                if pt is not None:
                    all_xs.append(pt[0])
                    all_ys.append(pt[1])

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    mx = max((x_max - x_min) * MARGIN_RATIO, 1.0)
    my = max((y_max - y_min) * MARGIN_RATIO, 1.0)
    bounds = (x_min - mx, x_max + mx, y_min - my, y_max + my)

    nrows, ncols = _grid_layout(len(method_data), max_cols)
    fig = plt.figure(figsize=(3.8 * ncols, 3.2 * nrows), constrained_layout=True)
    gs = GridSpec(nrows, 2 * ncols, figure=fig)
    axes = []
    remaining = len(method_data)
    for row in range(nrows):
        row_count = min(ncols, remaining)
        start_slot = ncols - row_count
        for col in range(row_count):
            slot = start_slot + 2 * col
            axes.append(fig.add_subplot(gs[row, slot : slot + 2]))
        remaining -= row_count

    for ax, (label, routes, color) in zip(axes, method_data):
        display_label = "Reinforcement Learning" if label == "End-to-End RL" else label
        ax.set_facecolor("#FFFFFF")
        draw_basemap(ax, links, coords)
        draw_routes(ax, routes, coords, color)
        draw_transit_center(ax, coords)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.text(
            0.5,
            -0.04,
            display_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fs,
            color="#111827",
            weight="semibold",
            clip_on=False,
        )

    save_figure(fig, output_path, facecolor="#FFFFFF", pad_inches=0.02)
    plt.close(fig)


def _coords_dict(nodes: pd.DataFrame) -> CoordMap:
    return {
        str(row["name"]): (row["x"], row["y"])
        for _, row in nodes.iterrows()
    }


def _routes_frame(
    links: pd.DataFrame,
    coords: CoordMap,
    routes: Sequence[Sequence[str]],
    *,
    use_tiles: bool,
) -> Image.Image:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    plot_coords = coords
    if use_tiles:
        try:
            import contextily as ctx

            plot_coords = maybe_to_web_mercator(coords)
            for _, row in links.iterrows():
                start = plot_coords.get(row["start"])
                end = plot_coords.get(row["end"])
                if start is None or end is None:
                    continue
                ax.plot([start[0], end[0]], [start[1], end[1]], color="#555555", alpha=0.4, linewidth=0.8, solid_capstyle="round", zorder=1)
            xs = [c[0] for c in plot_coords.values()]
            ys = [c[1] for c in plot_coords.values()]
            xpad = (max(xs) - min(xs)) * 0.03
            ypad = (max(ys) - min(ys)) * 0.03
            ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
            ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        except ImportError:
            use_tiles = False

    if not use_tiles:
        draw_basemap(ax, links, coords)
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        xpad = (max(xs) - min(xs)) * 0.03
        ypad = (max(ys) - min(ys)) * 0.03
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    draw_routes(
        ax,
        routes,
        plot_coords,
        "#FFFFFF",
        use_palette=True,
        line_width=2.8,
        alpha=0.92,
    )
    draw_transit_center(ax, plot_coords, zoom=0.13)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("#CCCCCC")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def build_route_cycle_gif(output_path: Path, alpha_key: str = "0_3", method_names: Iterable[str] | None = None, *, use_tiles: bool = True) -> None:
    apply_plot_style(16, use_tex=False)
    links = load_links()
    coords = _coords_dict(load_nodes())
    names = list(method_names) if method_names is not None else [
        "Random Walk",
        "Demand Cover",
        "Genetic Algorithm",
        "End-to-End RL",
        "Pure MCTS",
    ]
    frames = []
    for name in names:
        frames.append(_routes_frame(links, coords, get_method_routes(name, alpha_key), use_tiles=use_tiles))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=1200, loop=0)
    print(f"Saved {output_path} ({len(frames)} frames)")


def _edge_usage(routes: Sequence[Sequence[str]]) -> Counter:
    usage: Counter = Counter()
    for route in routes:
        for idx in range(len(route) - 1):
            edge = tuple(sorted([route[idx], route[idx + 1]]))
            usage[edge] += 1
    return usage


def _deceptive_frame(
    links: pd.DataFrame,
    coords: CoordMap,
    routes: Sequence[Sequence[str]],
    *,
    n_route: int,
    total_routes: int,
    use_tiles: bool,
) -> Image.Image:
    plot_coords = maybe_to_web_mercator(coords) if use_tiles else coords
    fig, ax = plt.subplots(1, 1, figsize=(9, 7.5))
    covered = _edge_usage(routes)

    for _, row in links.iterrows():
        edge_key = tuple(sorted([row["start"], row["end"]]))
        if edge_key in covered:
            continue
        p1, p2 = plot_coords.get(row["start"]), plot_coords.get(row["end"])
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d1d5db", alpha=0.3, linewidth=0.5, solid_capstyle="round", zorder=0)

    cmap = plt.cm.YlOrRd
    for edge, count in covered.items():
        p1, p2 = plot_coords.get(edge[0]), plot_coords.get(edge[1])
        if not p1 or not p2:
            continue
        t = min(count / 6.0, 1.0)
        color = cmap(0.15 + 0.85 * t)
        lw = 1.8 + 3.5 * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.9, linewidth=lw, solid_capstyle="round", zorder=2 + count)

    xs = [c[0] for c in plot_coords.values()]
    ys = [c[1] for c in plot_coords.values()]
    xpad = (max(xs) - min(xs)) * 0.03
    ypad = (max(ys) - min(ys)) * 0.03
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    if use_tiles:
        try:
            import contextily as ctx

            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.4)
        except ImportError:
            pass

    draw_transit_center(ax, plot_coords, zoom=0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("#CCCCCC")

    ax.text(0.98, 0.02, f"{n_route}/{total_routes} routes", transform=ax.transAxes, ha="right", va="bottom", fontsize=13, color="#888888")
    if n_route > 1:
        scalar = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=6))
        scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=ax, shrink=0.5, pad=0.01, aspect=25)
        cbar.set_ticks([1, 2, 3, 4, 5, 6])
        cbar.ax.tick_params(labelsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def build_deceptive_landscape_gif(
    output_path: Path,
    *,
    alpha_key: str = "0_3",
    method_name: str = "Demand Cover",
    key_route_counts: Sequence[int] = (1, 4, 8, 12, 16),
    use_tiles: bool = True,
) -> None:
    apply_plot_style(14, use_tex=False)
    links = load_links()
    coords = _coords_dict(load_nodes())
    routes = get_method_routes(method_name, alpha_key)
    frames = []
    total = len(routes)
    for n in key_route_counts:
        if n <= total:
            frames.append(_deceptive_frame(links, coords, routes[:n], n_route=n, total_routes=total, use_tiles=use_tiles))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [1500, 1200, 1200, 1200, 2500][: len(frames)]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Saved {output_path} ({len(frames)} frames)")

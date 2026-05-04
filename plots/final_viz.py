from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from PIL import Image

try:
    from .common import ASSETS_DIR, ROUTE_COLORS, apply_plot_style
    from .routes import (
        draw_basemap,
        draw_routes,
        get_method_routes,
        get_methods,
        load_links,
        load_nodes,
    )
except ImportError:
    from common import ASSETS_DIR, ROUTE_COLORS, apply_plot_style
    from routes import (
        draw_basemap,
        draw_routes,
        get_method_routes,
        get_methods,
        load_links,
        load_nodes,
    )


README_ASSETS_DIR = ASSETS_DIR.parent.parent / "assets"
README_ROUTE_DESIGNS_GIF = README_ASSETS_DIR / "readme_route_designs.gif"
README_VEHICLE_PLATOONS_GIF = README_ASSETS_DIR / "readme_vehicle_platoons.gif"

MAX_README_GIF_BYTES = 8 * 1024 * 1024
FRAME_SIZE = (640, 520)
BACKGROUND = "#F5F5F2"
ROUTE_BACKGROUND = "#FFFFFF"
BASE_EDGE = "#D4D8DD"
BASE_EDGE_DARK = "#A8B0BA"
ROUTE_LABEL_COLOR = "#111111"
NEURIPS_RESULTS_DIR = ASSETS_DIR.parent.parent / "training_data" / "NeurIPS_results"
METRIC_METHOD_PREFIXES = {
    "Real-World": ("real_world_", ()),
    "Random Walk": ("random_walk_", ()),
    "Demand Cover": ("demand_cover_", ()),
    "Shortest Path": ("shortest_path_", ()),
    "Genetic Algorithm": ("genetic_", ("current_best",)),
    "Bee Colony": ("bee_colony_", ()),
    "Neural Evolutionary": ("neural_evolutionary_", ()),
    "Pure MCTS": ("mcts_pure_", ()),
    "End-to-End RL": ("end_to_end_rl_", ("train_seed_42",)),
}
CoordMap = dict[str, tuple[float, float]]


def _coords_dict() -> CoordMap:
    nodes = load_nodes()
    return {
        str(row["name"]): (float(row["x"]), float(row["y"]))
        for _, row in nodes.iterrows()
    }


def _bounds(coords: CoordMap, margin: float = 0.055) -> tuple[float, float, float, float]:
    xs = np.asarray([pt[0] for pt in coords.values()], dtype=float)
    ys = np.asarray([pt[1] for pt in coords.values()], dtype=float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    span = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half = span * (0.5 + margin)
    return cx - half, cx + half, cy - half, cy + half


def _setup_axis(ax: plt.Axes, bounds: tuple[float, float, float, float], *, background: str = BACKGROUND) -> None:
    ax.set_facecolor(background)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _new_figure(width_px: int, height_px: int, *, background: str = BACKGROUND) -> tuple[plt.Figure, plt.Axes]:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(background)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig, ax


def _figure_to_image(fig: plt.Figure, *, background: str = BACKGROUND) -> Image.Image:
    buf = io.BytesIO()
    canvas_bbox = Bbox.from_bounds(0, 0, fig.get_figwidth(), fig.get_figheight())
    fig.savefig(
        buf,
        format="png",
        facecolor=background,
        dpi=fig.dpi,
        bbox_inches=canvas_bbox,
        pad_inches=0,
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_readme_style() -> None:
    apply_plot_style(14, use_tex=False, background=BACKGROUND)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        }
    )


def _draw_base(ax: plt.Axes, coords: CoordMap, *, darker: bool = False) -> None:
    links = load_links()
    draw_basemap(
        ax,
        links,
        coords,
        color=BASE_EDGE_DARK if darker else BASE_EDGE,
        alpha=0.72 if darker else 0.48,
        linewidth=0.95 if darker else 0.64,
        zorder=1,
    )
    xy = np.asarray(list(coords.values()), dtype=float)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=3.2 if darker else 2.0,
        color="#FFFFFF",
        edgecolors="#AEB6C2" if darker else "#B8BEC6",
        linewidths=0.22 if darker else 0.16,
        alpha=0.72 if darker else 0.55,
        zorder=2,
    )


def _route_design_bounds(
    coords: CoordMap,
    route_sets: Sequence[Sequence[Sequence[str]]],
) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for routes in route_sets:
        for route in routes:
            for node in route:
                pt = coords.get(node)
                if pt is None:
                    continue
                xs.append(pt[0])
                ys.append(pt[1])
    if not xs:
        return _bounds(coords)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    mx = max((x_max - x_min) * 0.03, 1.0)
    my = max((y_max - y_min) * 0.03, 1.0)
    return x_min - mx, x_max + mx, y_min - my, y_max + my


def _route_design_label(label: str) -> str:
    return "Reinforcement Learning" if label == "End-to-End RL" else label


def _latest_matching_dir(parent: Path, prefix: str) -> Path:
    matches = sorted(
        (path for path in parent.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No directory matching {prefix}* under {parent}")
    return matches[-1]


def _resolve_metric_summary_path(method_name: str, alpha_key: str, route_path: Path) -> Path:
    for parent in route_path.parents:
        direct = parent / "eval_results_summary.json"
        if direct.is_file():
            return direct

    if method_name == "AlphaTransit":
        niter_dir = "nips_7_n_iter" if alpha_key == "0_3" else "nips_8_n_iter"
        base = NEURIPS_RESULTS_DIR / alpha_key / "alphatransit" / niter_dir / "n_iter_500"
        matches = sorted(base.glob("*/eval_results_summary.json"), key=lambda path: path.stat().st_mtime)
        if matches:
            return matches[-1]

    if method_name in METRIC_METHOD_PREFIXES:
        prefix, extra = METRIC_METHOD_PREFIXES[method_name]
        base = _latest_matching_dir(NEURIPS_RESULTS_DIR / alpha_key, prefix).joinpath(*extra)
        direct = base / "eval_results_summary.json"
        if direct.is_file():
            return direct

    raise FileNotFoundError(f"No eval summary found for {method_name} ({alpha_key})")


def _load_design_metrics(method_name: str, alpha_key: str, route_path: Path) -> tuple[float, float] | None:
    try:
        summary_path = _resolve_metric_summary_path(method_name, alpha_key, route_path)
    except FileNotFoundError:
        return None

    with summary_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    results = payload.get("results", payload)
    return (
        float(results["service_rate"]["avg"] * 100.0),
        float(results["fleet_size"]["avg"]),
    )


def _add_suptitle(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        1.035,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=22,
        family="DejaVu Serif",
        color=ROUTE_LABEL_COLOR,
        weight="semibold",
        clip_on=False,
    )


def _render_route_frame(
    coords: CoordMap,
    bounds: tuple[float, float, float, float],
    *,
    label: str,
    color: str,
    routes: Sequence[Sequence[str]] | None,
    metrics: tuple[float, float] | None,
    width_px: int,
    height_px: int,
) -> Image.Image:
    fig, ax = _new_figure(width_px, height_px, background=ROUTE_BACKGROUND)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.14)
    _setup_axis(ax, bounds, background=ROUTE_BACKGROUND)
    draw_basemap(ax, load_links(), coords)
    _add_suptitle(ax, "Transit Design")
    _draw_metric_panel(ax, metrics)

    if routes is None:
        display_label = "Bloomington Network"
    else:
        draw_routes(
            ax,
            routes,
            coords,
            color,
        )
        display_label = _route_design_label(label)

    ax.text(
        0.5,
        -0.04,
        display_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=20,
        color=ROUTE_LABEL_COLOR,
        weight="normal",
        clip_on=False,
    )
    return _figure_to_image(fig, background=ROUTE_BACKGROUND)


def _draw_metric_panel(ax: plt.Axes, metrics: tuple[float, float] | None) -> None:
    from matplotlib.patches import Rectangle

    x0, y0, width, height = 0.585, 0.775, 0.385, 0.165
    panel = Rectangle(
        (x0, y0),
        width,
        height,
        transform=ax.transAxes,
        facecolor="#FFFFFF",
        edgecolor="#D1D5DB",
        linewidth=0.75,
        alpha=0.94,
        zorder=50,
    )
    ax.add_patch(panel)
    service_text = "--" if metrics is None else f"{metrics[0]:5.1f}%"
    fleet_text = "--" if metrics is None else f"{metrics[1]:5.0f}"
    rows = (
        ("Service rate", service_text, y0 + height * 0.64),
        ("Fleet size", fleet_text, y0 + height * 0.31),
    )
    for label, value, y in rows:
        ax.text(
            x0 + 0.028,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.8,
            color="#4B5563",
            zorder=51,
        )
        ax.text(
            x0 + width - 0.028,
            y,
            value,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9.8,
            family="DejaVu Sans Mono",
            color="#111827",
            zorder=51,
        )



def _resize_frames(frames: Sequence[Image.Image], scale: float) -> list[Image.Image]:
    resized = []
    for frame in frames:
        w, h = frame.size
        size = (max(320, int(w * scale)), max(260, int(h * scale)))
        resized.append(frame.resize(size, Image.Resampling.LANCZOS))
    return resized


def _save_gif_under_limit(
    frames: Sequence[Image.Image],
    output_path: Path,
    *,
    durations: int | Sequence[int],
    max_bytes: int,
    colors: int = 96,
) -> Path:
    if not frames:
        raise ValueError("Cannot save a GIF with no frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    working_frames = [frame.copy() for frame in frames]

    for _ in range(6):
        palette_frames = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
            for frame in working_frames
        ]
        palette_frames[0].save(
            tmp_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
        )
        size = tmp_path.stat().st_size
        if size <= max_bytes:
            tmp_path.replace(output_path)
            print(f"Saved {output_path} ({size / (1024 * 1024):.2f} MB)")
            return output_path

        tmp_path.unlink(missing_ok=True)
        working_frames = _resize_frames(working_frames, 0.86)

    raise RuntimeError(
        f"Could not keep {output_path} under {max_bytes / (1024 * 1024):.1f} MB. "
        "Reduce frame_count, frame_size, or colors."
    )


def _alpha_value(alpha_key: str) -> float:
    return 0.3 if alpha_key == "0_3" else 1.0


def _simulation_config(
    *,
    alpha_key: str,
    horizon: int,
    seed: int,
    num_routes: int,
) -> dict:
    from config import build_arg_parser, normalize_config

    parser = build_arg_parser()
    config = vars(parser.parse_args([]))
    config.update(
        {
            "network": "bloomington",
            "mode": "baseline",
            "baseline_type": "real_world",
            "seed": seed,
            "horizon": horizon,
            "alpha": _alpha_value(alpha_key),
            "num_routes": num_routes,
            "transit_center_node": "96",
            "save_animations": False,
            "wandb_off": True,
        }
    )
    return normalize_config(config)


def _simulate_vehicle_logs(
    *,
    alpha_key: str,
    method_name: str,
    horizon: int,
    seed: int,
):
    from config import set_global_seeds
    from rl.env import TransitEnv

    routes = get_method_routes(method_name, alpha_key)
    config = _simulation_config(
        alpha_key=alpha_key,
        horizon=horizon,
        seed=seed,
        num_routes=len(routes),
    )
    set_global_seeds(seed)

    env = TransitEnv(config)
    env.all_routes = routes
    env.current_route = []
    env.current_route_index = len(routes)
    env.is_baseline = True
    env.world = env.build_world("bloomington")
    env._apply_action()
    env.world.exec_simulation(until_t=horizon)

    logs = env.world.analyzer.gps_like_log_to_pandas()
    if logs.empty:
        raise RuntimeError("Simulation produced no vehicle GPS logs")
    bus_name_to_route = {
        str(veh.name): _route_index_from_bus_name(str(veh.name))
        for veh in env.world.VEHICLES.values()
        if getattr(veh, "mode", None) == "bus"
    }
    bus_name_to_route = {
        name: route_idx
        for name, route_idx in bus_name_to_route.items()
        if route_idx is not None
    }
    if not bus_name_to_route:
        raise RuntimeError("Simulation produced no bus vehicles")

    logs = logs.copy()
    logs["name"] = logs["name"].astype(str)
    logs = logs[logs["name"].isin(bus_name_to_route)].copy()
    if logs.empty:
        raise RuntimeError("Simulation produced no drawable bus GPS logs")
    logs["route_idx"] = logs["name"].map(bus_name_to_route).astype(int)
    for col in ("t", "x", "y", "v"):
        logs[col] = logs[col].astype(float)
    logs = logs.replace([np.inf, -np.inf], np.nan).dropna(subset=["t", "x", "y", "v"])
    logs = logs[logs["v"] >= 0].copy()
    if logs.empty:
        raise RuntimeError("Simulation produced no drawable moving vehicle logs")
    return logs


def _route_index_from_bus_name(name: str) -> int | None:
    match = re.match(r"bus_route_(\d+)(?:_freq_\d+)?$", name)
    return int(match.group(1)) if match else None


def _route_color(route_idx: int) -> str:
    return ROUTE_COLORS[route_idx % len(ROUTE_COLORS)]


def _select_vehicle_names(logs, max_vehicles: int) -> set[str]:
    moving = logs[logs["v"] > 0.05].groupby("name").size()
    counts = logs.groupby("name").size()
    scores = counts.astype(float).mul(0.01).add(moving.astype(float), fill_value=0.0)
    return set(str(name) for name in scores.sort_values(ascending=False).head(max_vehicles).index)


def _frame_times(logs, frame_count: int) -> np.ndarray:
    moving = logs[logs["v"] > 0.05]
    source = moving if not moving.empty else logs
    start = float(source["t"].quantile(0.05))
    end = float(source["t"].quantile(0.95))
    if end <= start:
        start, end = float(logs["t"].min()), float(logs["t"].max())
    return np.linspace(start, end, frame_count)


def _render_simulation_frame(
    coords: CoordMap,
    bounds: tuple[float, float, float, float],
    *,
    routes: Sequence[Sequence[str]],
    logs,
    frame_time: float,
    trail_seconds: float,
    width_px: int,
    height_px: int,
) -> Image.Image:
    fig, ax = _new_figure(width_px, height_px)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
    _setup_axis(ax, bounds)
    _draw_base(ax, coords, darker=True)
    _add_suptitle(ax, "Simulation")

    for route_idx, route in enumerate(routes):
        draw_routes(
            ax,
            [route],
            coords,
            _route_color(route_idx),
            line_width=1.25,
            alpha=0.30,
        )

    window = logs[(logs["t"] >= frame_time - trail_seconds) & (logs["t"] <= frame_time)]
    if window.empty:
        return _figure_to_image(fig)

    current_rows = []
    for _, vehicle_log in window.sort_values("t").groupby("name", sort=False):
        vehicle_log = vehicle_log.tail(36)
        current = vehicle_log.iloc[-1]
        current_rows.append(current)
        if len(vehicle_log) < 2:
            continue
        color = _route_color(int(current["route_idx"]))
        ax.plot(
            vehicle_log["x"].to_numpy(),
            vehicle_log["y"].to_numpy(),
            color=color,
            linewidth=1.55,
            alpha=0.42,
            solid_capstyle="round",
            zorder=12,
        )

    if current_rows:
        xs = np.asarray([float(row["x"]) for row in current_rows])
        ys = np.asarray([float(row["y"]) for row in current_rows])
        colors = [_route_color(int(row["route_idx"])) for row in current_rows]
        ax.scatter(xs, ys, s=112, color=colors, alpha=0.13, linewidths=0, zorder=13)
        ax.scatter(
            xs,
            ys,
            s=24,
            color=colors,
            edgecolors="#FFFFFF",
            linewidths=0.45,
            alpha=0.96,
            zorder=14,
        )

    return _figure_to_image(fig)


def build_route_designs_gif(
    output_path: Path = README_ROUTE_DESIGNS_GIF,
    *,
    alpha_key: str = "0_3",
    method_names: Iterable[str] | None = None,
    include_intro: bool = True,
    frame_size: tuple[int, int] = FRAME_SIZE,
    max_bytes: int = MAX_README_GIF_BYTES,
) -> Path:
    """
    Build the README route-design GIF.

    The GIF starts from the Bloomington network canvas, then cycles through the
    selected route designs with the method name rendered below each frame.
    Defaults use all 10 methods from the alpha=0.3 route-design figure.
    """
    _apply_readme_style()
    coords = _coords_dict()
    width_px, height_px = frame_size

    selected = list(method_names) if method_names is not None else [str(m["name"]) for m in get_methods()]
    method_lookup = {str(m["name"]): m for m in get_methods()}
    method_data = [
        (
            name,
            str(method_lookup[name]["color"]),
            get_method_routes(name, alpha_key),
            _load_design_metrics(name, alpha_key, Path(method_lookup[name][f"path_{alpha_key}"])),
        )
        for name in selected
    ]
    bounds = _route_design_bounds(coords, [routes for _, _, routes, _ in method_data])

    frames: list[Image.Image] = []
    durations: list[int] = []
    if include_intro:
        frames.append(
            _render_route_frame(
                coords,
                bounds,
                label="Bloomington Network",
                color="#4A90D9",
                routes=None,
                metrics=None,
                width_px=width_px,
                height_px=height_px,
            )
        )
        durations.append(950)

    for name, color, routes, metrics in method_data:
        frames.append(
            _render_route_frame(
                coords,
                bounds,
                label=name,
                color=color,
                routes=routes,
                metrics=metrics,
                width_px=width_px,
                height_px=height_px,
            )
        )
        durations.append(1050)

    return _save_gif_under_limit(frames, output_path, durations=durations, max_bytes=max_bytes)


def build_vehicle_platoon_gif(
    output_path: Path = README_VEHICLE_PLATOONS_GIF,
    *,
    alpha_key: str = "0_3",
    method_name: str = "Real-World",
    simulation_horizon: int = 4200,
    seed: int = 42,
    max_vehicles: int = 110,
    trail_seconds: float = 110.0,
    frame_count: int = 96,
    frame_duration_ms: int = 115,
    frame_size: tuple[int, int] = FRAME_SIZE,
    max_bytes: int = MAX_README_GIF_BYTES,
) -> Path:
    """
    Build the README vehicle GIF from actual UXsim simulation logs.

    The previous README GIF was generated from UXsim vehicle animation after
    baseline simulation. This uses the same simulation source, but renders the
    resulting GPS-like vehicle logs in the paper plot style.
    """
    _apply_readme_style()
    coords = _coords_dict()
    bounds = _bounds(coords)
    width_px, height_px = frame_size
    routes = get_method_routes(method_name, alpha_key)
    logs = _simulate_vehicle_logs(
        alpha_key=alpha_key,
        method_name=method_name,
        horizon=simulation_horizon,
        seed=seed,
    )
    selected_names = _select_vehicle_names(logs, max_vehicles)
    logs = logs[logs["name"].astype(str).isin(selected_names)].copy()
    times = _frame_times(logs, frame_count)

    frames = [
        _render_simulation_frame(
            coords,
            bounds,
            routes=routes,
            logs=logs,
            frame_time=float(frame_time),
            trail_seconds=trail_seconds,
            width_px=width_px,
            height_px=height_px,
        )
        for frame_time in times
    ]
    return _save_gif_under_limit(frames, output_path, durations=frame_duration_ms, max_bytes=max_bytes, colors=112)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build README-ready final visualization GIFs.")
    parser.add_argument("--target", choices=["all", "routes", "platoons"], default="all")
    parser.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    parser.add_argument("--max-mb", type=float, default=MAX_README_GIF_BYTES / (1024 * 1024))
    parser.add_argument("--simulation-horizon", type=int, default=4200)
    args = parser.parse_args(argv)

    max_bytes = int(args.max_mb * 1024 * 1024)
    if args.target in {"all", "routes"}:
        build_route_designs_gif(alpha_key=args.alpha, max_bytes=max_bytes)
    if args.target in {"all", "platoons"}:
        build_vehicle_platoon_gif(
            alpha_key=args.alpha,
            simulation_horizon=args.simulation_horizon,
            max_bytes=max_bytes,
        )


if __name__ == "__main__":
    _main()

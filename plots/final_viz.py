from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "alphatransit_speed",
    ["#E8F4FD", "#4A90D9", "#7B68EE", "#DA70D6", "#FF00FF"],
)
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
        alpha=0.58 if darker else 0.48,
        linewidth=0.72 if darker else 0.64,
        zorder=1,
    )
    xy = np.asarray(list(coords.values()), dtype=float)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=3.2 if darker else 2.0,
        color="#FFFFFF",
        edgecolors="#B8BEC6",
        linewidths=0.16,
        alpha=0.55,
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


def _render_route_frame(
    coords: CoordMap,
    bounds: tuple[float, float, float, float],
    *,
    label: str,
    color: str,
    routes: Sequence[Sequence[str]] | None,
    width_px: int,
    height_px: int,
) -> Image.Image:
    fig, ax = _new_figure(width_px, height_px, background=ROUTE_BACKGROUND)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.14)
    _setup_axis(ax, bounds, background=ROUTE_BACKGROUND)
    draw_basemap(ax, load_links(), coords)

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
        weight="semibold",
        clip_on=False,
    )
    return _figure_to_image(fig, background=ROUTE_BACKGROUND)



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
    logs = logs.copy()
    for col in ("t", "x", "y", "v"):
        logs[col] = logs[col].astype(float)
    logs = logs.replace([np.inf, -np.inf], np.nan).dropna(subset=["t", "x", "y", "v"])
    logs = logs[logs["v"] >= 0].copy()
    if logs.empty:
        raise RuntimeError("Simulation produced no drawable moving vehicle logs")
    return logs


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
    logs,
    frame_time: float,
    trail_seconds: float,
    speed_norm: Normalize,
    width_px: int,
    height_px: int,
) -> Image.Image:
    fig, ax = _new_figure(width_px, height_px)
    _setup_axis(ax, bounds)
    _draw_base(ax, coords)

    window = logs[(logs["t"] >= frame_time - trail_seconds) & (logs["t"] <= frame_time)]
    if window.empty:
        return _figure_to_image(fig)

    current_rows = []
    for _, vehicle_log in window.sort_values("t").groupby("name", sort=False):
        vehicle_log = vehicle_log.tail(10)
        current = vehicle_log.iloc[-1]
        current_rows.append(current)
        if len(vehicle_log) < 2:
            continue
        color = SPEED_CMAP(speed_norm(float(current["v"])))
        ax.plot(
            vehicle_log["x"].to_numpy(),
            vehicle_log["y"].to_numpy(),
            color=color,
            linewidth=1.25,
            alpha=0.28,
            solid_capstyle="round",
            zorder=8,
        )

    if current_rows:
        xs = np.asarray([float(row["x"]) for row in current_rows])
        ys = np.asarray([float(row["y"]) for row in current_rows])
        speeds = np.asarray([float(row["v"]) for row in current_rows])
        colors = SPEED_CMAP(speed_norm(speeds))
        ax.scatter(xs, ys, s=92, color=colors, alpha=0.12, linewidths=0, zorder=9)
        ax.scatter(
            xs,
            ys,
            s=18,
            color=colors,
            edgecolors="#FFFFFF",
            linewidths=0.35,
            alpha=0.92,
            zorder=10,
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
        (name, str(method_lookup[name]["color"]), get_method_routes(name, alpha_key))
        for name in selected
    ]
    bounds = _route_design_bounds(coords, [routes for _, _, routes in method_data])

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
                width_px=width_px,
                height_px=height_px,
            )
        )
        durations.append(950)

    for name, color, routes in method_data:
        frames.append(
            _render_route_frame(
                coords,
                bounds,
                label=name,
                color=color,
                routes=routes,
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
    simulation_horizon: int = 1800,
    seed: int = 42,
    max_vehicles: int = 110,
    trail_seconds: float = 60.0,
    frame_count: int = 28,
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
    logs = _simulate_vehicle_logs(
        alpha_key=alpha_key,
        method_name=method_name,
        horizon=simulation_horizon,
        seed=seed,
    )
    selected_names = _select_vehicle_names(logs, max_vehicles)
    logs = logs[logs["name"].astype(str).isin(selected_names)].copy()
    times = _frame_times(logs, frame_count)
    moving = logs[logs["v"] > 0.05]
    vmax = float((moving if not moving.empty else logs)["v"].quantile(0.98))
    speed_norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0), clip=True)

    frames = [
        _render_simulation_frame(
            coords,
            bounds,
            logs=logs,
            frame_time=float(frame_time),
            trail_seconds=trail_seconds,
            speed_norm=speed_norm,
            width_px=width_px,
            height_px=height_px,
        )
        for frame_time in times
    ]
    return _save_gif_under_limit(frames, output_path, durations=90, max_bytes=max_bytes, colors=112)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build README-ready final visualization GIFs.")
    parser.add_argument("--target", choices=["all", "routes", "platoons"], default="all")
    parser.add_argument("--alpha", choices=["0_3", "1_0"], default="0_3")
    parser.add_argument("--max-mb", type=float, default=MAX_README_GIF_BYTES / (1024 * 1024))
    parser.add_argument("--simulation-horizon", type=int, default=1800)
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

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.transforms import Bbox
from PIL import Image

try:
    from .common import ASSETS_DIR, ROUTE_COLORS, apply_plot_style
    from .routes import (
        draw_basemap,
        draw_routes,
        draw_transit_center,
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
        draw_transit_center,
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
BASE_EDGE = "#D4D8DD"
BASE_EDGE_DARK = "#A8B0BA"
TEXT = "#151515"
PLATOON_COLORS = (
    "#00BFFF",
    "#4A90D9",
    "#7B68EE",
    "#DA70D6",
    "#FF00FF",
    "#22C55E",
    "#F97316",
    "#E11D48",
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


def _setup_axis(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _new_figure(width_px: int, height_px: int) -> tuple[plt.Figure, plt.Axes]:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(BACKGROUND)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig, ax


def _figure_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    canvas_bbox = Bbox.from_bounds(0, 0, fig.get_figwidth(), fig.get_figheight())
    fig.savefig(
        buf,
        format="png",
        facecolor=BACKGROUND,
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


def _title_chip(ax: plt.Axes, label: str, *, color: str, subtitle: str | None = None) -> None:
    text = label if subtitle is None else f"{label}\n{subtitle}"
    ax.text(
        0.5,
        0.965,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=17 if subtitle is None else 15,
        fontweight="semibold",
        color=TEXT,
        linespacing=1.15,
        bbox=dict(
            boxstyle="round,pad=0.38,rounding_size=0.08",
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=1.4,
            alpha=0.94,
        ),
        zorder=50,
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
    fig, ax = _new_figure(width_px, height_px)
    _setup_axis(ax, bounds)
    _draw_base(ax, coords, darker=routes is None)

    if routes is None:
        draw_transit_center(ax, coords, zoom=0.105, pin="pin_blue.png")
        _title_chip(ax, "Bloomington Network", color="#4A90D9", subtitle="route-design canvas")
    else:
        draw_routes(
            ax,
            routes,
            coords,
            color,
            use_palette=True,
            line_width=2.2,
            alpha=0.93,
        )
        draw_transit_center(ax, coords, zoom=0.105, pin="pin_blue.png")
        _title_chip(ax, label, color=color)

    return _figure_to_image(fig)


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


def _polyline(route: Sequence[str], coords: CoordMap) -> np.ndarray:
    points = [coords[node] for node in route if node in coords]
    return np.asarray(points, dtype=float)


def _polyline_lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.asarray([], dtype=float), 0.0
    deltas = np.diff(points, axis=0)
    seg_lengths = np.sqrt((deltas * deltas).sum(axis=1))
    total = float(seg_lengths.sum())
    return seg_lengths, total


def _sample_polyline(points: np.ndarray, fraction: float) -> tuple[float, float]:
    seg_lengths, total = _polyline_lengths(points)
    if total <= 0:
        return float(points[0, 0]), float(points[0, 1])

    target = (fraction % 1.0) * total
    cumulative = 0.0
    for idx, seg_len in enumerate(seg_lengths):
        if cumulative + seg_len >= target:
            local = (target - cumulative) / max(seg_len, 1e-9)
            pt = points[idx] + local * (points[idx + 1] - points[idx])
            return float(pt[0]), float(pt[1])
        cumulative += seg_len
    return float(points[-1, 0]), float(points[-1, 1])


def _route_segments(polylines: Sequence[np.ndarray]) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for points in polylines:
        if len(points) < 2:
            continue
        for idx in range(len(points) - 1):
            segments.append(points[idx : idx + 2])
    return segments


def _render_platoon_frame(
    coords: CoordMap,
    bounds: tuple[float, float, float, float],
    polylines: Sequence[np.ndarray],
    *,
    frame_idx: int,
    frame_count: int,
    width_px: int,
    height_px: int,
) -> Image.Image:
    fig, ax = _new_figure(width_px, height_px)
    _setup_axis(ax, bounds)
    _draw_base(ax, coords)

    segments = _route_segments(polylines)
    if segments:
        route_colors = [PLATOON_COLORS[idx % len(PLATOON_COLORS)] for idx in range(len(segments))]
        ax.add_collection(
            LineCollection(
                segments,
                colors=route_colors,
                linewidths=1.7,
                alpha=0.22,
                capstyle="round",
                joinstyle="round",
                zorder=4,
            )
        )

    phase = frame_idx / frame_count
    for route_idx, points in enumerate(polylines):
        if len(points) < 2:
            continue
        color = PLATOON_COLORS[route_idx % len(PLATOON_COLORS)]
        base = phase * (0.36 + 0.025 * (route_idx % 4)) + route_idx * 0.071
        for vehicle_idx in range(4):
            progress = base - vehicle_idx * 0.022
            x, y = _sample_polyline(points, progress)
            trail_alpha = max(0.16, 0.52 - vehicle_idx * 0.10)
            ax.scatter([x], [y], s=145 - vehicle_idx * 22, color=color, alpha=0.12, linewidths=0, zorder=10)
            ax.scatter([x], [y], s=44 - vehicle_idx * 5, color="#FFFFFF", alpha=trail_alpha, linewidths=0, zorder=11)
            ax.scatter(
                [x],
                [y],
                s=21 - vehicle_idx * 2,
                color=color,
                edgecolors="#FFFFFF",
                linewidths=0.55,
                alpha=0.95 - vehicle_idx * 0.09,
                zorder=12,
            )

    draw_transit_center(ax, coords, zoom=0.095, pin="pin_blue.png")
    _title_chip(ax, "Bloomington Platoon Flow", color="#DA70D6")
    return _figure_to_image(fig)


def build_route_designs_gif(
    output_path: Path = README_ROUTE_DESIGNS_GIF,
    *,
    alpha_key: str = "1_0",
    method_names: Iterable[str] | None = None,
    include_intro: bool = True,
    frame_size: tuple[int, int] = FRAME_SIZE,
    max_bytes: int = MAX_README_GIF_BYTES,
) -> Path:
    """
    Build the README route-design GIF.

    The GIF starts from the Bloomington network canvas, then cycles through the
    selected route designs with the method name rendered on each frame. Defaults
    use all 10 methods from plots.routes.get_methods().
    """
    _apply_readme_style()
    coords = _coords_dict()
    bounds = _bounds(coords)
    width_px, height_px = frame_size

    selected = list(method_names) if method_names is not None else [str(m["name"]) for m in get_methods()]
    method_lookup = {str(m["name"]): m for m in get_methods()}

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

    for name in selected:
        method = method_lookup[name]
        frames.append(
            _render_route_frame(
                coords,
                bounds,
                label=name,
                color=str(method["color"]),
                routes=get_method_routes(name, alpha_key),
                width_px=width_px,
                height_px=height_px,
            )
        )
        durations.append(1050)

    return _save_gif_under_limit(frames, output_path, durations=durations, max_bytes=max_bytes)


def build_vehicle_platoon_gif(
    output_path: Path = README_VEHICLE_PLATOONS_GIF,
    *,
    alpha_key: str = "1_0",
    method_name: str = "AlphaTransit",
    route_limit: int = 9,
    frame_count: int = 28,
    frame_size: tuple[int, int] = FRAME_SIZE,
    max_bytes: int = MAX_README_GIF_BYTES,
) -> Path:
    """
    Build the README vehicle-platoon GIF.

    Vehicles are rendered as moving platoons over the selected Bloomington route
    design. The defaults stay short and compact enough for a normal Git upload.
    """
    _apply_readme_style()
    coords = _coords_dict()
    bounds = _bounds(coords)
    width_px, height_px = frame_size
    routes = get_method_routes(method_name, alpha_key)[:route_limit]
    polylines = [points for route in routes if len(points := _polyline(route, coords)) >= 2]
    if not polylines:
        raise ValueError(f"No drawable routes found for {method_name} alpha={alpha_key}")

    frames = [
        _render_platoon_frame(
            coords,
            bounds,
            polylines,
            frame_idx=idx,
            frame_count=frame_count,
            width_px=width_px,
            height_px=height_px,
        )
        for idx in range(frame_count)
    ]
    return _save_gif_under_limit(frames, output_path, durations=90, max_bytes=max_bytes, colors=112)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build README-ready final visualization GIFs.")
    parser.add_argument("--target", choices=["all", "routes", "platoons"], default="all")
    parser.add_argument("--alpha", choices=["0_3", "1_0"], default="1_0")
    parser.add_argument("--max-mb", type=float, default=MAX_README_GIF_BYTES / (1024 * 1024))
    args = parser.parse_args(argv)

    max_bytes = int(args.max_mb * 1024 * 1024)
    if args.target in {"all", "routes"}:
        build_route_designs_gif(alpha_key=args.alpha, max_bytes=max_bytes)
    if args.target in {"all", "platoons"}:
        build_vehicle_platoon_gif(alpha_key=args.alpha, max_bytes=max_bytes)


if __name__ == "__main__":
    _main()

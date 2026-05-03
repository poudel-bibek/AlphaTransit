from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

try:
    from .common import (
        NEURIPS_RESULTS_DIR,
        NETWORKS_DIR,
        ASSETS_DIR,
        apply_plot_style,
        format_steps_axis,
        load_results_summary,
        read_wandb_scan_csv,
        rolling_mean_std,
        save_figure,
    )
    from .training import (
        MCTS_BLOCK_HISTORY_DIRS,
        MCTS_EP_HISTORY_DIRS,
        MCTS_NITER_HISTORY_DIRS,
        MODE_ORDER,
        MODE_STYLES,
        NITER_COLORS,
        PARETO_ABBREV,
        PARETO_STYLES,
        WALL_CLOCK_COLORS,
        _extract_ppo_mode,
        _load_mcts_metric,
        _load_ppo_runs,
        get_pareto_summary_paths,
    )
    from .routes import (
        TRANSIT_CENTER_NODE,
        _coords_dict,
        build_route_figure as build_route_design_figure,
        draw_basemap,
        draw_transit_center,
        load_links,
        load_nodes,
        load_routes,
    )
except ImportError:
    from common import (
        NEURIPS_RESULTS_DIR,
        NETWORKS_DIR,
        ASSETS_DIR,
        apply_plot_style,
        format_steps_axis,
        load_results_summary,
        read_wandb_scan_csv,
        rolling_mean_std,
        save_figure,
    )
    from training import (
        MCTS_BLOCK_HISTORY_DIRS,
        MCTS_EP_HISTORY_DIRS,
        MCTS_NITER_HISTORY_DIRS,
        MODE_ORDER,
        MODE_STYLES,
        NITER_COLORS,
        PARETO_ABBREV,
        PARETO_STYLES,
        WALL_CLOCK_COLORS,
        _extract_ppo_mode,
        _load_mcts_metric,
        _load_ppo_runs,
        get_pareto_summary_paths,
    )
    from routes import (
        TRANSIT_CENTER_NODE,
        _coords_dict,
        build_route_figure as build_route_design_figure,
        draw_basemap,
        draw_transit_center,
        load_links,
        load_nodes,
        load_routes,
    )

try:
    import contextily as _ctx

    _HAS_CTX = True
except ImportError:
    _ctx = None
    _HAS_CTX = False

try:
    from scipy.stats import gaussian_kde as _gaussian_kde

    _HAS_KDE = True
except ImportError:
    _gaussian_kde = None
    _HAS_KDE = False

try:
    from pyproj import Transformer as _Transformer

    _HAS_PYPROJ = True
except ImportError:
    _Transformer = None
    _HAS_PYPROJ = False


METHOD_COMPARISON_FS = 18
COMBINED_SCALING_FS = 18
SCALING_OVERVIEW_FS = 20
LEARNING_OVERVIEW_FS = 20
ROUTE_DESIGN_FS = 21
LEARNING_SMOOTHING_WINDOW = 8
LEARNING_DATA_STEP_LIMIT = 1_020_000
LEARNING_AXIS_STEP_LIMIT = 1_040_000
ROUTE_DESIGN_ALPHA_KEYS = ("0_3", "1_0")
BACKGROUND = "#FFFFFF"
BLOOMINGTON_MAP_BACKGROUND = "#F7F8FA"
TICK_TEXT_COLOR = "#333333"
TICK_TEXT_WEIGHT = "medium"
LEGEND_TEXT_COLOR = "#111111"
ANNOTATION_OFFSETS = [
    (0, 10),
    (0, -12),
    (10, 0),
    (-10, 0),
    (12, 10),
    (-12, 10),
    (12, -10),
    (-12, -10),
    (16, 4),
    (-16, 4),
]
MANUAL_LABEL_OFFSETS = {
    ("0_3", "headline", "Demand Cover"): (12, 10),
    ("0_3", "headline", "Random Walk"): (12, -10),
    ("0_3", "headline", "Real-World"): (-10, 0),
    ("0_3", "headline", "Bee Colony"): (-10, 0),
    ("0_3", "headline", "Neural Evol."): (-10, 0),
    ("0_3", "headline", "Pure MCTS"): (12, 10),
    ("0_3", "headline", "RL"): (-12, 10),
    ("0_3", "headline", "Shortest Path"): (8, 8),
    ("0_3", "passenger", "Random Walk"): (12, 10),
    ("0_3", "passenger", "Shortest Path"): (-12, 10),
    ("0_3", "passenger", "Demand Cover"): (-12, 10),
    ("0_3", "passenger", "Pure MCTS"): (-12, 10),
    ("0_3", "passenger", "RL"): (0, 12),
    ("0_3", "passenger", "AlphaTransit"): (-10, 0),
    ("0_3", "passenger", "Genetic Alg."): (0, -12),
    ("0_3", "passenger", "Neural Evol."): (10, 0),
    ("0_3", "operator", "Demand Cover"): (12, 10),
    ("0_3", "operator", "Random Walk"): (-12, -10),
    ("0_3", "operator", "Shortest Path"): (12, 10),
    ("0_3", "operator", "Pure MCTS"): (12, -10),
    ("0_3", "operator", "RL"): (-12, 10),
    ("0_3", "operator", "Genetic Alg."): (0, -12),
    ("0_3", "operator", "Neural Evol."): (-10, 0),
    ("0_3", "operator", "AlphaTransit"): (-10, 0),
    ("1_0", "headline", "Random Walk"): (-12, 10),
    ("1_0", "headline", "Demand Cover"): (12, 10),
    ("1_0", "headline", "Shortest Path"): (12, 10),
    ("1_0", "headline", "Pure MCTS"): (12, 10),
    ("1_0", "headline", "Neural Evol."): (-10, 0),
    ("1_0", "headline", "Bee Colony"): (-10, 0),
    ("1_0", "headline", "RL"): (12, 10),
    ("1_0", "passenger", "Pure MCTS"): (12, -10),
    ("1_0", "passenger", "Demand Cover"): (12, 10),
    ("1_0", "passenger", "Shortest Path"): (12, -10),
    ("1_0", "passenger", "Random Walk"): (12, 10),
    ("1_0", "passenger", "RL"): (-12, 10),
    ("1_0", "passenger", "Bee Colony"): (0, -12),
    ("1_0", "operator", "Random Walk"): (12, 10),
    ("1_0", "operator", "Shortest Path"): (12, 10),
    ("1_0", "operator", "Demand Cover"): (-12, -10),
    ("1_0", "operator", "Pure MCTS"): (-12, -10),
    ("1_0", "operator", "Bee Colony"): (-10, 0),
    ("1_0", "operator", "Neural Evol."): (-10, 0),
    ("1_0", "operator", "AlphaTransit"): (-10, 0),
    ("1_0", "operator", "Genetic Alg."): (-10, 0),
    ("1_0", "operator", "RL"): (12, 10),
}
LEGEND_METHOD_ORDER = [
    "Real-World",
    "Random Walk",
    "Demand Cover",
    "Shortest Path",
    "Genetic Alg.",
    "Pure MCTS",
    "Neural Evol.",
    "RL",
    "Bee Colony",
    "AlphaTransit",
]
METHOD_DISPLAY_NAMES = {
    "Genetic Alg.": "Genetic Algorithm",
    "Neural Evol.": "Neural Evolutionary",
    "RL": "End-to-End RL",
}
METHOD_COMPARISON_ALPHATRANSIT_OVERRIDES = {
    "1_0": NEURIPS_RESULTS_DIR
    / "1_0"
    / "alphatransit"
    / "nips_8_n_iter"
    / "n_iter_500",
}
LEARNING_MODE_LABELS = {
    "terminal_only": "Terminal",
    "terminal_intermediate_raw_early_stop": "Raw + ES",
    "terminal_intermediate_delta_early_stop": r"$\Delta$ + ES",
    "terminal_intermediate_delta_no_early_stop": r"$\Delta$ + No ES",
}
LEARNING_MODE_COLORS = {
    "terminal_only": "#4A90D9",
    "terminal_intermediate_raw_early_stop": "#E84393",
    "terminal_intermediate_delta_early_stop": "#F16913",
    "terminal_intermediate_delta_no_early_stop": "#1B9E77",
}
LEARNING_MODE_LINESTYLES = {
    "terminal_only": "-",
    "terminal_intermediate_raw_early_stop": "--",
    "terminal_intermediate_delta_early_stop": ":",
    "terminal_intermediate_delta_no_early_stop": "-.",
}
SCALING_BACKGROUND = "#F5F5F2"
SCALING_ALPHA_STYLES = {
    "1_0": {"color": "#3B73D1", "label": r"$\alpha=1.0$"},
    "0_3": {"color": "#3DA08D", "label": r"$\alpha=0.3$"},
}
SCALING_FAMILY_SPECS = {
    "search": {
        "title": r"\textbf{Search Depth}",
        "x_label": r"MCTS sims",
        "variant_prefix": "n_iter",
        "values": [100, 200, 300, 400, 500],
    },
    "data": {
        "title": r"\textbf{Data Diversity}",
        "x_label": r"episodes / iter",
        "variant_prefix": "ep_per_iter",
        "values": [8, 16, 24, 32],
    },
    "model": {
        "title": r"\textbf{Policy Network Size}",
        "x_label": r"GAT blocks",
        "variant_prefix": "num_gat_blocks",
        "values": [2, 4, 8, 16],
    },
}
SWEEP_BEHAVIOR_STYLES = {
    "search": {
        "title": r"\textbf{Search Sweep}",
        "tradeoff_title": r"\textbf{Search: Quality vs Compute}",
        "title_color": "#1E4E9C",
        "legend_fmt": r"$n_{\mathrm{iter}}={value}$",
        "point_fmt": r"$n={value}$",
        "colors": ["#BFDFFF", "#6FB3FF", "#2F7CFF", "#1454C8", "#08306B"],
    },
    "data": {
        "title": r"\textbf{Data Sweep}",
        "tradeoff_title": r"\textbf{Data: Quality vs Compute}",
        "title_color": "#1C8A4B",
        "legend_fmt": r"$\mathrm{ep/iter}={value}$",
        "point_fmt": r"$\mathrm{ep}={value}$",
        "colors": ["#8CD17D", "#4DAF4A", "#238B45", "#00441B"],
    },
    "model": {
        "title": r"\textbf{Model Sweep}",
        "tradeoff_title": r"\textbf{Model: Quality vs Compute}",
        "title_color": "#D96B00",
        "legend_fmt": r"$\mathrm{blocks}={value}$",
        "point_fmt": r"$b={value}$",
        "colors": ["#FFE08A", "#FDB04E", "#F16913", "#8C2D04"],
    },
}
SWEEP_TRAINING_REWARD_METRIC = "mcts/avg_reward"
SWEEP_TRAINING_REWARD_LABEL = "Avg Reward"
SWEEP_STEP_SCALE = 1e6
SWEEP_STEP_SCALE_LABEL = r"$\times 10^{6}$"
SWEEP_RUNTIME_SCALE = 1e5
SWEEP_RUNTIME_SCALE_LABEL = r"$\times 10^{5}$"
SWEEP_LINESTYLES = ["solid", "dashed", "dotted", "dashdot", (0, (5, 2, 1, 2))]


def _resolve_summary_path(path: Path) -> Path:
    if path.is_file():
        return path

    direct = path / "eval_results_summary.json"
    if direct.is_file():
        return direct

    matches = sorted(path.glob("*/eval_results_summary.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No eval_results_summary.json found under {path}")
    return matches[-1]


def _metric_stats(metric: dict, scale: float = 1.0) -> tuple[float, float]:
    if "data" in metric:
        values = np.asarray(metric["data"], dtype=float) * scale
        if values.size:
            std = values.std(ddof=1) if values.size > 1 else 0.0
            return float(values.mean()), float(std)
    return float(metric["avg"] * scale), float(metric["std"] * scale)


def _combined_stats(first: dict, second: dict) -> tuple[float, float]:
    values = np.asarray(first["data"], dtype=float) + np.asarray(second["data"], dtype=float)
    std = values.std(ddof=1) if values.size > 1 else 0.0
    return float(values.mean()), float(std)


def _load_method_points(alpha_key: str) -> dict[str, dict[str, tuple[float, float]]]:
    points: dict[str, dict[str, tuple[float, float]]] = {}
    summary_paths = dict(get_pareto_summary_paths(alpha_key))
    alphatransit_override = METHOD_COMPARISON_ALPHATRANSIT_OVERRIDES.get(alpha_key)
    if alphatransit_override is not None:
        summary_paths["AlphaTransit"] = alphatransit_override

    for method, path in summary_paths.items():
        results = load_results_summary(_resolve_summary_path(path))
        points[method] = {
            "service_rate": _metric_stats(results["service_rate"], scale=100.0),
            "fleet_size": _metric_stats(results["fleet_size"]),
            "journey_time": _metric_stats(results["combined_avg_travel_minutes"]),
            "transfer_rate": _metric_stats(results["transfer_rate"]),
            "bus_utilization": _metric_stats(results["bus_utilization"]),
            "route_efficiency": _metric_stats(results["route_efficiency"]),
        }

    return points


def _five_integer_ticks_and_limits(vmin: float, vmax: float) -> tuple[np.ndarray, float, float]:
    if np.isclose(vmin, vmax):
        center = int(round(vmin))
        step = max(1, int(round(max(abs(vmin), 1.0) * 0.08)))
        ticks = np.array([center + step * offset for offset in (-2, -1, 0, 1, 2)], dtype=float)
        pad = max(abs(center) * 0.05, 0.5)
        return ticks, float(ticks[0] - pad), float(ticks[-1] + pad)

    step = max(1, int(np.ceil((vmax - vmin) / 4.0)))
    tick_min = int(np.floor(vmin))
    while tick_min + 4 * step < vmax:
        step += 1
    ticks = tick_min + step * np.arange(5, dtype=float)

    axis_min = ticks[0] - max(abs(vmin) * 0.05, 0.5)
    axis_max = ticks[-1] + max(abs(vmax) * 0.05, 0.5)
    return ticks, float(axis_min), float(axis_max)


def _pad_axis_bounds(vmin: float, vmax: float, *, min_pad: float = 0.35, frac: float = 0.03) -> tuple[float, float]:
    span = max(vmax - vmin, 1.0)
    pad = max(span * frac, min_pad)
    return float(vmin - pad), float(vmax + pad)


def _bold_label(text: str) -> str:
    return rf"\textbf{{{text}}}"


def _style_tick_labels(ax: plt.Axes, *, axis: str = "both") -> None:
    ax.tick_params(axis=axis, colors=TICK_TEXT_COLOR)
    labels = []
    if axis in {"x", "both"}:
        labels.extend(ax.get_xticklabels())
    if axis in {"y", "both"}:
        labels.extend(ax.get_yticklabels())
    for label in labels:
        label.set_color(TICK_TEXT_COLOR)
        label.set_fontweight(TICK_TEXT_WEIGHT)


def _style_legend_text(legend) -> None:
    for text in legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)
        text.set_fontweight(TICK_TEXT_WEIGHT)


def _scaling_history_dir(alpha_key: str, family_key: str) -> Path:
    history_dirs = {
        "search": MCTS_NITER_HISTORY_DIRS,
        "data": MCTS_EP_HISTORY_DIRS,
        "model": MCTS_BLOCK_HISTORY_DIRS,
    }
    return history_dirs[family_key][alpha_key]


def _find_unique_history_csv(history_dir: Path, prefix: str) -> Path:
    matches = sorted(history_dir.glob(f"{prefix}_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV for {prefix} under {history_dir}, found {len(matches)}")
    return matches[0]


def _topk_reward_stats(csv_path: Path, top_k: int = 5) -> tuple[float, float]:
    history = read_wandb_scan_csv(csv_path)
    history = history.dropna(subset=[SWEEP_TRAINING_REWARD_METRIC]).copy()
    if history.empty:
        raise ValueError(f"No {SWEEP_TRAINING_REWARD_METRIC} rewards in {csv_path}")
    rewards = history[SWEEP_TRAINING_REWARD_METRIC].to_numpy(dtype=float)
    k = min(top_k, len(rewards))
    top = np.sort(rewards)[-k:]
    std = top.std(ddof=1) if k > 1 else 0.0
    return float(top.mean()), float(std)


def _load_scaling_series(alpha_key: str, family_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = SCALING_FAMILY_SPECS[family_key]
    base_dir = _scaling_history_dir(alpha_key, family_key)
    xs: list[float] = []
    means: list[float] = []
    stds: list[float] = []

    for value in spec["values"]:
        try:
            csv_path = _find_unique_history_csv(base_dir, f"{spec['variant_prefix']}_{value}")
            mean, std = _topk_reward_stats(csv_path)
        except (FileNotFoundError, ValueError):
            continue
        xs.append(float(value))
        means.append(mean)
        stds.append(std)

    return (
        np.asarray(xs, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(stds, dtype=float),
    )


def _load_sweep_behavior_runs(alpha_key: str, family_key: str) -> list[dict[str, float | np.ndarray | str]]:
    spec = SCALING_FAMILY_SPECS[family_key]
    style = SWEEP_BEHAVIOR_STYLES[family_key]
    base_dir = _scaling_history_dir(alpha_key, family_key)
    runs: list[dict[str, float | np.ndarray | str]] = []

    for idx, value in enumerate(spec["values"]):
        try:
            csv_path = _find_unique_history_csv(base_dir, f"{spec['variant_prefix']}_{value}")
        except FileNotFoundError:
            continue

        history = read_wandb_scan_csv(csv_path)
        if SWEEP_TRAINING_REWARD_METRIC not in history.columns:
            continue
        history = history.dropna(subset=["_step", SWEEP_TRAINING_REWARD_METRIC, "_runtime"]).copy()
        if history.empty:
            continue

        steps = history["_step"].to_numpy(dtype=float)
        rewards = history[SWEEP_TRAINING_REWARD_METRIC].to_numpy(dtype=float)
        smooth_reward, smooth_std = rolling_mean_std(rewards, window=min(7, len(rewards)))
        top_mean, top_std = _topk_reward_stats(csv_path)
        runtime_seconds = float(history["_runtime"].max())

        runs.append(
            {
                "value": float(value),
                "legend_label": style["legend_fmt"].replace("{value}", str(value)),
                "point_label": style["point_fmt"].replace("{value}", str(value)),
                "steps": steps,
                "reward": rewards,
                "reward_smooth": smooth_reward,
                "reward_smooth_std": smooth_std,
                "runtime_seconds": runtime_seconds,
                "top_mean": top_mean,
                "top_std": top_std,
                "color": style["colors"][min(idx, len(style["colors"]) - 1)],
            }
        )

    return runs


def _draw_search_depth_legend(
    ax: plt.Axes,
    handles: list[mlines.Line2D],
    labels: list[str],
    *,
    fontsize: float,
    edgecolor: str,
) -> None:
    if len(handles) == 5:
        # Matplotlib fills multi-column legends column-by-column. This order
        # renders as top row 100/200/300 and bottom row 400/500.
        order = [0, 3, 1, 4, 2]
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]

    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        ncol=min(3, len(handles)),
        frameon=True,
        fancybox=False,
        edgecolor=edgecolor,
        fontsize=fontsize,
        handlelength=1.5,
        columnspacing=0.8,
        borderpad=0.3,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("#FFFFFF")
    _style_legend_text(legend)


def _save_figure_with_optional_png(
    fig: plt.Figure,
    output_path: Path,
    *,
    facecolor: str,
) -> None:
    suffix = output_path.suffix.lower()
    pdf_path = output_path if suffix == ".pdf" else output_path.with_suffix(".pdf")
    save_figure(fig, pdf_path, facecolor=facecolor)

    # PNG export is disabled for the final figures; PDFs are the paper artifact.
    # png_path = pdf_path.with_suffix(".png")
    # converter = shutil.which("pdftocairo")
    # if converter is not None:
    #     png_base = png_path.with_suffix("")
    #     png_path.parent.mkdir(parents=True, exist_ok=True)
    #     subprocess.run(
    #         [converter, "-png", "-singlefile", str(pdf_path), str(png_base)],
    #         check=True,
    #         capture_output=True,
    #         text=True,
    #     )
    #     print(f"Saved {png_path}")


def plot_method_comparison_triptych(
    output_path: Path = NEURIPS_RESULTS_DIR / "final_comparison_alpha_1_0.pdf",
    *,
    alpha_key: str = "1_0",
    show_titles: bool = True,
) -> Path:
    """
    Build the final three-panel method-comparison figure from local eval summaries.

    The panels intentionally separate the story into:
    - Service Rate vs Fleet Size: headline coverage-vs-cost comparison
    - Passenger Burden: Journey Time versus Transfer Rate
    - Operator Quality: Bus Utilization versus Route Efficiency

    Each point is a method-level mean over the local evaluation seeds for the
    requested objective weight alpha; horizontal and vertical error bars show
    seed-level standard deviations. The green "Optimal" corner marks the
    preferred direction for each tradeoff, not an estimated optimum. Method
    annotations use compact abbreviations while the legend carries full names.

    This function is the public entrypoint for the current final comparison plot.
    Additional final-paper plots can live alongside it in this module as separate
    top-level functions, each responsible for saving its own output file.
    """
    fs = METHOD_COMPARISON_FS
    apply_plot_style(fs, background=BACKGROUND)
    points = _load_method_points(alpha_key)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8))

    def label_text(method: str) -> str:
        label = PARETO_ABBREV.get(method, method)
        return r"\textbf{" + label + "}" if method == "AlphaTransit" else label

    def place_labels(ax: plt.Axes, panel_points: list[tuple[str, float, float]], panel_key: str) -> None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        placed_boxes: list[tuple[float, float, float, float]] = []

        ordered = sorted(
            panel_points,
            key=lambda item: PARETO_STYLES[item[0]]["zorder"],
            reverse=True,
        )

        def overlap(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> bool:
            ax0, ay0, ax1, ay1 = box_a
            bx0, by0, bx1, by1 = box_b
            return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

        def candidate_box(label: str, x: float, y: float, dx: int, dy: int) -> tuple[float, float, float, float]:
            width = max(0.06 * x_range, 0.018 * x_range * len(label))
            height = 0.09 * y_range
            x_shift = 0.025 * x_range if dx > 0 else -0.025 * x_range if dx < 0 else 0.0
            y_shift = 0.04 * y_range if dy > 0 else -0.04 * y_range if dy < 0 else 0.0

            if dx >= 0:
                x0 = x + x_shift
                x1 = x0 + width
            else:
                x1 = x + x_shift
                x0 = x1 - width

            if dy >= 0:
                y0 = y + y_shift
                y1 = y0 + height
            else:
                y1 = y + y_shift
                y0 = y1 - height

            return (x0, y0, x1, y1)

        for method, x, y in ordered:
            style = PARETO_STYLES[method]
            text = label_text(method)
            manual_offset = MANUAL_LABEL_OFFSETS.get((alpha_key, panel_key, method))
            best_offset = manual_offset if manual_offset is not None else ANNOTATION_OFFSETS[0]
            best_score = None

            if manual_offset is None:
                for dx, dy in ANNOTATION_OFFSETS:
                    box = candidate_box(text, x, y, dx, dy)
                    bounds_penalty = (
                        int(box[0] < x_min)
                        + int(box[2] > x_max)
                        + int(box[1] < y_min)
                        + int(box[3] > y_max)
                    )
                    overlap_penalty = sum(1 for placed in placed_boxes if overlap(box, placed))
                    score = (bounds_penalty, overlap_penalty, abs(dy) + abs(dx))

                    if best_score is None or score < best_score:
                        best_offset = (dx, dy)
                        best_score = score

            dx, dy = best_offset
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
            ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fs - 1,
                color="#111111",
                ha=ha,
                va=va,
            )
            placed_boxes.append(candidate_box(text, x, y, dx, dy))

    def draw_panel(
        ax: plt.Axes,
        *,
        panel_key: str,
        x_key: str,
        y_key: str,
        title: str,
        x_label: str,
        y_label: str,
    ) -> list[tuple[str, float, float]]:
        xs: list[float] = []
        ys: list[float] = []
        x_lows: list[float] = []
        x_highs: list[float] = []
        y_lows: list[float] = []
        y_highs: list[float] = []
        panel_points: list[tuple[str, float, float]] = []

        def add_best_region_overlay() -> None:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_pad = 0.015 * x_range
            y_pad = 0.015 * y_range
            x_span = 0.412 * x_range
            y_span = 0.412 * y_range

            if panel_key == "headline":
                x_grid = np.linspace(x_min + x_pad, x_min + x_pad + x_span, 14)
                y_grid = np.linspace(y_max - y_pad - y_span, y_max - y_pad, 14)
            elif panel_key == "passenger":
                x_grid = np.linspace(x_min + x_pad, x_min + x_pad + x_span, 14)
                y_grid = np.linspace(y_min + y_pad, y_min + y_pad + y_span, 14)
            else:
                x_grid = np.linspace(x_max - x_pad - x_span, x_max - x_pad, 14)
                y_grid = np.linspace(y_max - y_pad - y_span, y_max - y_pad, 14)

            xx, yy = np.meshgrid(x_grid, y_grid)
            x_norm = (xx - x_grid.min()) / (x_grid.max() - x_grid.min())
            y_norm = (yy - y_grid.min()) / (y_grid.max() - y_grid.min())

            if panel_key == "headline":
                color = "#7BC96F"
                score = (1.0 - x_norm) * y_norm
                text_xy = (x_min + 0.026 * x_range, y_max - 0.022 * y_range)
                text_ha, text_va = "left", "top"
            elif panel_key == "passenger":
                color = "#7BC96F"
                score = (1.0 - x_norm) * (1.0 - y_norm)
                text_xy = (x_min + 0.026 * x_range, y_min + 0.022 * y_range)
                text_ha, text_va = "left", "bottom"
            else:
                color = "#7BC96F"
                score = x_norm * y_norm
                text_xy = (x_max - 0.026 * x_range, y_max - 0.022 * y_range)
                text_ha, text_va = "right", "top"

            score = np.clip(score, 0.0, 1.0) ** 1.35
            rgba = np.tile(np.array(to_rgba(color)), (score.size, 1))
            rgba[:, 3] = 0.11 + 0.40 * score.ravel()

            ax.scatter(
                xx.ravel(),
                yy.ravel(),
                s=(5.5 + 28.0 * score).ravel(),
                c=rgba,
                linewidths=0.0,
                marker="o",
                zorder=-2,
                clip_on=True,
            )
            for dx, dy, alpha in [(-0.8, 0.0, 0.22), (0.8, 0.0, 0.22), (0.0, -0.8, 0.22), (0.0, 0.8, 0.22)]:
                ax.annotate(
                    "Optimal",
                    text_xy,
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=fs,
                    color="#F7FFF4",
                    ha=text_ha,
                    va=text_va,
                    alpha=alpha,
                    zorder=-1,
                )
            ax.text(
                *text_xy,
                "Optimal",
                fontsize=fs,
                color="#1F2937",
                ha=text_ha,
                va=text_va,
                alpha=0.98,
                zorder=-1,
            )

        for method, metrics in points.items():
            style = PARETO_STYLES[method]
            x_avg, x_std = metrics[x_key]
            y_avg, y_std = metrics[y_key]
            xs.append(x_avg)
            ys.append(y_avg)
            x_lows.append(x_avg - x_std)
            x_highs.append(x_avg + x_std)
            y_lows.append(y_avg - y_std)
            y_highs.append(y_avg + y_std)
            panel_points.append((method, x_avg, y_avg))

            ax.errorbar(
                x_avg,
                y_avg,
                xerr=x_std,
                yerr=y_std,
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=3,
                alpha=0.55,
                zorder=style["zorder"] - 1,
            )
            ax.scatter(
                x_avg,
                y_avg,
                color=style["color"],
                marker=style["marker"],
                s=(np.sqrt(style["size"]) + 1) ** 2,
                zorder=style["zorder"],
                edgecolors="white",
                linewidth=0.5,
            )

        if title:
            ax.set_title(title, fontsize=fs + 3, color="#111111")
        ax.set_xlabel(_bold_label(x_label))
        ax.set_ylabel(_bold_label(y_label))
        ax.xaxis.label.set_color("#111111")
        ax.yaxis.label.set_color("#111111")
        ax.tick_params(axis="both", colors=TICK_TEXT_COLOR)
        ax.grid(True, zorder=0)

        if xs and ys:
            x_low = min(x_lows)
            x_high = max(x_highs)
            y_low = min(y_lows)
            y_high = max(y_highs)
            x_ticks, x_left, x_right = _five_integer_ticks_and_limits(x_low, x_high)
            y_ticks, y_bottom, y_top = _five_integer_ticks_and_limits(y_low, y_high)
            if alpha_key == "0_3" and panel_key == "headline":
                x_left = 19.5
            if alpha_key == "1_0" and panel_key == "headline":
                x_ticks = np.array([45, 120, 195, 270, 345], dtype=float)
                x_left = 42.75
                x_right = 361.5
            if alpha_key == "0_3" and panel_key == "passenger":
                x_ticks = np.array([30, 36, 42, 48, 54], dtype=float)
                x_left = 28.5
                x_right = 54.5
                y_top = 94.0
            if alpha_key == "0_3" and panel_key == "operator":
                x_right = 27.5
            if alpha_key == "1_0" and panel_key == "operator":
                x_ticks = np.array([15, 25, 35, 45, 55], dtype=float)
                y_ticks = np.array([20, 45, 70, 95, 120], dtype=float)
                x_left = 12.0
                x_right = 55.0
                y_top = 126.0
            x_padded_low, x_padded_high = _pad_axis_bounds(x_low, x_high)
            y_padded_low, y_padded_high = _pad_axis_bounds(y_low, y_high)
            x_left = min(x_left, x_padded_low)
            x_right = max(x_right, x_padded_high)
            y_bottom = min(y_bottom, y_padded_low)
            y_top = max(y_top, y_padded_high)
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_bottom, y_top)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
            _style_tick_labels(ax)
            add_best_region_overlay()

        return panel_points

    panel_points = []

    panel_points.append(draw_panel(
        axes[0],
        panel_key="headline",
        x_key="fleet_size",
        y_key="service_rate",
        title=r"\textbf{Passenger vs Operator}" if show_titles else "",
        x_label=r"Fleet Size",
        y_label=r"Service Rate (\%)",
    ))
    panel_points.append(draw_panel(
        axes[1],
        panel_key="passenger",
        x_key="journey_time",
        y_key="transfer_rate",
        title=r"\textbf{Passenger Burden}" if show_titles else "",
        x_label=r"Journey Time (min)",
        y_label=r"Transfer Rate (\%)",
    ))
    panel_points.append(draw_panel(
        axes[2],
        panel_key="operator",
        x_key="bus_utilization",
        y_key="route_efficiency",
        title=r"\textbf{Operator Efficiency}" if show_titles else "",
        x_label=r"Bus Utilization (\%)",
        y_label=r"Route Efficiency",
    ))

    legend_methods = [method for method in LEGEND_METHOD_ORDER if method in points]
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=PARETO_STYLES[method]["color"],
            marker=PARETO_STYLES[method]["marker"],
            linestyle="None",
            markersize=np.sqrt(PARETO_STYLES[method]["size"]) + 1,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=f"{METHOD_DISPLAY_NAMES.get(method, method)} ({PARETO_ABBREV[method]})",
        )
        for method in legend_methods
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 0.018),
        fontsize=fs + 1,
        columnspacing=1.2,
        handletextpad=0.22,
        labelspacing=0.5,
        handleheight=0.9,
        borderpad=0.3,
    )
    _style_legend_text(legend)
    fig.subplots_adjust(wspace=0.28, bottom=0.27)
    for ax, points_for_axis, panel_key in zip(axes, panel_points, ["headline", "passenger", "operator"]):
        place_labels(ax, points_for_axis, panel_key)
    _save_figure_with_optional_png(fig, output_path, facecolor=BACKGROUND)
    plt.close(fig)
    return output_path


def plot_scaling_behavior_overview(
    output_path: Path = NEURIPS_RESULTS_DIR / "final_scaling_behavior_alpha_1_0.pdf",
    *,
    alpha_key: str = "1_0",
) -> Path:
    """
    Build one composite scaling figure for a single alpha setting.

    Top row: MCTS reward dynamics. Bottom row: final quality-vs-compute
    tradeoffs with top-5 reward uncertainty. Columns correspond to Search,
    Data Diversity, and Policy Network Size.
    Search depth uses n_iter {100, 200, 300, 400, 500}; data diversity uses
    ep_per_iter {8, 16, 24, 32}; policy network size uses num_gat_blocks
    {2, 4, 8, 16}. Top panels plot smoothed W&B mcts/avg_reward against
    environment step. Bottom panels plot top-5 mean reward against compute
    budget, where compute budget is max W&B _runtime scaled by 1e5 seconds.
    Error bars on bottom points show the standard deviation of the top-5 reward
    samples used for that point.
    """
    fs = SCALING_OVERVIEW_FS
    apply_plot_style(fs, background=SCALING_BACKGROUND)
    family_order = list(SCALING_FAMILY_SPECS.keys())
    family_runs = {
        family_key: _load_sweep_behavior_runs(alpha_key, family_key)
        for family_key in family_order
    }
    if not any(family_runs.values()):
        raise ValueError(f"No scaling overview data found for alpha={alpha_key}.")

    def compact_value_label(family_key: str, value: float) -> str:
        if family_key == "model":
            return f"{int(value)}x"
        return rf"${int(value)}$"

    def reward_axis_config(
        values: list[float],
        stds: list[float] | None = None,
        *,
        tick_count: int = 5,
        tight: bool = False,
    ) -> tuple[np.ndarray, float, float, str]:
        if not values:
            raise ValueError("Cannot configure reward axis without data.")
        if stds is None:
            stds = [0.0] * len(values)
        lows = np.asarray(values, dtype=float) - np.asarray(stds, dtype=float)
        highs = np.asarray(values, dtype=float) + np.asarray(stds, dtype=float)
        vmin = float(lows.min())
        vmax = float(highs.max())
        if tight:
            span = max(vmax - vmin, 1.0)
            pad = max(span * 0.22, 0.45)
            bottom = float(vmin - pad)
            top = float(vmax + pad)
            ticks = np.linspace(bottom, top, tick_count)
        elif tick_count == 5:
            ticks, bottom, top = _five_integer_ticks_and_limits(vmin, vmax)
        else:
            span = max(vmax - vmin, 1.0)
            step = max(1, int(np.ceil(span / max(tick_count - 1, 1))))
            tick_min = int(np.floor(vmin))
            while tick_min + (tick_count - 1) * step < vmax:
                step += 1
            ticks = tick_min + step * np.arange(tick_count, dtype=float)
            bottom = float(ticks[0] - max(abs(vmin) * 0.05, 0.5))
            top = float(ticks[-1] + max(abs(vmax) * 0.05, 0.5))
        tick_step = float(np.min(np.diff(ticks))) if len(ticks) > 1 else top - bottom
        fmt = "decimal" if ((top - bottom) <= 7.0 and (max(abs(bottom), abs(top)) < 10.0 or tick_step < 2.0)) else "integer"
        return ticks, bottom, top, fmt

    def apply_reward_axis(ax: plt.Axes, ticks: np.ndarray, bottom: float, top: float, fmt: str) -> None:
        ax.set_ylim(bottom, top)
        ax.set_yticks(ticks)
        if fmt == "decimal":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
        _style_tick_labels(ax, axis="y")

    def apply_compute_axis(ax: plt.Axes, scaled_values: np.ndarray, tick_count: int = 4) -> None:
        values = np.asarray(scaled_values, dtype=float)
        if values.size == 0:
            left, right = 0.0, 1.0
            ticks = np.linspace(left, right, tick_count)
        else:
            vmin = float(values.min())
            vmax = float(values.max())
            span = max(vmax - vmin, 0.1)
            pad = 0.07 * span
            left = max(0.0, vmin - pad)
            right = vmax + pad
            ticks = np.linspace(vmin, vmax, tick_count)
        ax.set_xlim(left, right)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
        _style_tick_labels(ax, axis="x")

    top_values: list[float] = []
    for runs in family_runs.values():
        for run in runs:
            smooth_reward = np.asarray(run["reward_smooth"], dtype=float)
            smooth_std = np.asarray(run["reward_smooth_std"], dtype=float)
            top_values.extend((smooth_reward - smooth_std).tolist())
            top_values.extend((smooth_reward + smooth_std).tolist())
    top_ticks, top_bottom, top_top, top_fmt = reward_axis_config(top_values)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.8, 6.55),
        gridspec_kw={"height_ratios": [1.69, 1.0]},
    )

    for col, family_key in enumerate(family_order):
        spec = SCALING_FAMILY_SPECS[family_key]
        family_style = SWEEP_BEHAVIOR_STYLES[family_key]
        runs = family_runs[family_key]
        train_ax = axes[0, col]
        tradeoff_ax = axes[1, col]
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                color=str(run["color"]),
                marker="o",
                linestyle=SWEEP_LINESTYLES[line_idx % len(SWEEP_LINESTYLES)],
                linewidth=1.8,
                markersize=9.0,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=compact_value_label(family_key, float(run["value"])),
            )
            for line_idx, run in enumerate(runs)
        ]

        for ax in (train_ax, tradeoff_ax):
            ax.set_facecolor(BACKGROUND)
            ax.tick_params(axis="both", colors=TICK_TEXT_COLOR)
            ax.grid(True, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CCCCCC")
            ax.spines["bottom"].set_color("#CCCCCC")

        train_ax.set_title(spec["title"], fontsize=fs + 1, color="#111111")

        max_step = 0.0
        for line_idx, run in enumerate(runs):
            steps = np.asarray(run["steps"], dtype=float)
            reward = np.asarray(run["reward_smooth"], dtype=float)
            reward_std = np.asarray(run["reward_smooth_std"], dtype=float)
            color = str(run["color"])
            linestyle = SWEEP_LINESTYLES[line_idx % len(SWEEP_LINESTYLES)]
            max_step = max(max_step, float(steps.max()))
            train_ax.fill_between(
                steps,
                reward - reward_std,
                reward + reward_std,
                color=color,
                alpha=0.10,
                linewidth=0.0,
                zorder=1,
            )
            train_ax.plot(
                steps,
                reward,
                color=color,
                linestyle=linestyle,
                linewidth=1.85,
                solid_capstyle="round",
                zorder=3,
            )

        if max_step:
            train_ax.set_xlim(0.0, max_step * 1.01)
        train_ax.set_xlabel(_bold_label("Environment Steps"), labelpad=1)
        train_ax.xaxis.label.set_color("#111111")
        apply_reward_axis(train_ax, top_ticks, top_bottom, top_top, top_fmt)
        train_ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"{0 if abs(value) < 0.5 else value / SWEEP_STEP_SCALE:g}"
            )
        )
        _style_tick_labels(train_ax, axis="x")
        train_ax.xaxis.get_offset_text().set_visible(False)
        train_ax.annotate(
            SWEEP_STEP_SCALE_LABEL,
            xy=(1.003, 0.015),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=fs - 5,
            color=TICK_TEXT_COLOR,
            fontweight=TICK_TEXT_WEIGHT,
            annotation_clip=False,
            zorder=0,
        )
        if col == 0:
            train_ax.set_ylabel(_bold_label(SWEEP_TRAINING_REWARD_LABEL))
            train_ax.yaxis.label.set_color("#111111")
        if legend_handles:
            legend_labels = [handle.get_label() for handle in legend_handles]
            if family_key == "search":
                _draw_search_depth_legend(
                    train_ax,
                    legend_handles,
                    legend_labels,
                    fontsize=fs - 2,
                    edgecolor="#CCCCCC",
                )
            else:
                legend = train_ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc="lower center",
                    ncol=len(legend_handles),
                    frameon=True,
                    fancybox=False,
                    edgecolor="#CCCCCC",
                    fontsize=fs - 2,
                    columnspacing=1.0,
                    handlelength=1.5,
                    handletextpad=0.4,
                    labelspacing=0.45,
                    handleheight=0.9,
                    borderpad=0.3,
                )
                legend.get_frame().set_facecolor("#FFFFFF")
                _style_legend_text(legend)

        runtime_seconds = np.asarray([float(run["runtime_seconds"]) for run in runs], dtype=float)
        runtimes = runtime_seconds / SWEEP_RUNTIME_SCALE
        qualities = np.asarray([float(run["top_mean"]) for run in runs], dtype=float)
        quality_stds = np.asarray([float(run["top_std"]) for run in runs], dtype=float)
        colors = [str(run["color"]) for run in runs]

        if len(runs):
            order = np.argsort(runtimes)
            connected_x = runtimes[order]
            connected_y = qualities[order]
            sizes = np.full(len(runs), 155.0)

            tradeoff_ax.plot(
                connected_x,
                connected_y,
                color="#777777",
                linewidth=1.25,
                linestyle=(0, (4, 4)),
                alpha=0.70,
                zorder=1,
            )
            for runtime, quality, quality_std, color in zip(runtimes, qualities, quality_stds, colors):
                tradeoff_ax.errorbar(
                    runtime,
                    quality,
                    yerr=quality_std,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.70,
                    zorder=2,
                )
            tradeoff_ax.scatter(
                runtimes,
                qualities,
                s=sizes,
                c=colors,
                edgecolors="white",
                linewidths=0.9,
                alpha=0.96,
                zorder=3,
            )

            best_idx = int(np.argmax(qualities))
            tradeoff_ax.annotate(
                "Best",
                (float(runtimes[best_idx]), float(qualities[best_idx] + quality_stds[best_idx])),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fs - 1,
                color=TICK_TEXT_COLOR,
                fontweight=TICK_TEXT_WEIGHT,
            )
        bottom_ticks, bottom_bottom, bottom_top, bottom_fmt = reward_axis_config(
            qualities.tolist(),
            quality_stds.tolist(),
            tick_count=4,
            tight=True,
        )
        apply_compute_axis(tradeoff_ax, runtimes)
        tradeoff_ax.set_xlabel(_bold_label("Compute Budget (s)"))
        tradeoff_ax.xaxis.label.set_color("#111111")
        tradeoff_ax.xaxis.get_offset_text().set_visible(False)
        tradeoff_ax.annotate(
            SWEEP_RUNTIME_SCALE_LABEL,
            xy=(1.003, 0.045),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=fs - 5,
            color=TICK_TEXT_COLOR,
            fontweight=TICK_TEXT_WEIGHT,
            annotation_clip=False,
            zorder=0,
        )
        apply_reward_axis(tradeoff_ax, bottom_ticks, bottom_bottom, bottom_top, bottom_fmt)
        if col == 0:
            tradeoff_ax.set_ylabel(_bold_label("Top 5 Reward"))
            tradeoff_ax.yaxis.label.set_color("#111111")

    fig.subplots_adjust(left=0.078, right=0.985, top=0.920, bottom=0.130, wspace=0.30, hspace=0.50)

    _save_figure_with_optional_png(fig, output_path, facecolor=BACKGROUND)
    plt.close(fig)
    return output_path


def _style_learning_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(axis="both", colors=TICK_TEXT_COLOR)
    ax.grid(True, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")


def _set_learning_step_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(np.linspace(0.0, 1_000_000.0, 5))
    format_steps_axis(ax)
    _style_tick_labels(ax, axis="x")


def _add_learning_line_axis_buffer(ax: plt.Axes, *, top_scale: float = 1.08) -> None:
    ax.set_xlim(0.0, LEARNING_AXIS_STEP_LIMIT)
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(y_bottom, y_bottom + (y_top - y_bottom) * top_scale)


def _set_nice_learning_y_ticks(ax: plt.Axes, *, y_top: float | None = None) -> None:
    bottom, top = ax.get_ylim()
    if y_top is not None:
        top = y_top
    span = max(top - bottom, 1.0)
    step = _round_up_to_nice(max(span / 4.0, 1.0))
    tick_bottom = np.floor(bottom / step) * step
    tick_top = y_top if y_top is not None else np.ceil(top / step) * step
    ticks = np.arange(tick_bottom, tick_top + 0.5 * step, step)
    if y_top is not None and not np.isclose(ticks[-1], y_top):
        ticks = np.append(ticks[ticks < y_top], y_top)
    # Keep the outermost labeled ticks slightly inside the frame, matching the
    # visual breathing room used in the Search Time panel.
    tick_span = max(tick_top - tick_bottom, step)
    bottom_pad = max(0.04 * tick_span, 0.5)
    top_pad = max(0.03 * tick_span, 0.35)
    ax.set_yticks(ticks)
    ax.set_ylim(tick_bottom - bottom_pad, tick_top + top_pad)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(round(value))}" if float(value).is_integer() else f"{value:g}")
    )
    _style_tick_labels(ax, axis="y")


def _set_fixed_learning_y_range(ax: plt.Axes, bottom: float, top: float, tick_count: int = 5) -> None:
    ticks = np.linspace(bottom, top, tick_count)
    _set_learning_y_ticks_with_padding(ax, ticks)


def _set_learning_y_ticks_with_padding(
    ax: plt.Axes,
    ticks: Sequence[float],
    *,
    top_pad_frac: float = 0.03,
) -> None:
    ticks = np.asarray(ticks, dtype=float)
    if ticks.size == 0:
        return
    bottom = float(ticks[0])
    top = float(ticks[-1])
    span = max(top - bottom, 1.0)
    bottom_pad = max(0.04 * span, 0.5)
    top_pad = max(top_pad_frac * span, 0.35)
    ax.set_ylim(bottom - bottom_pad, top + top_pad)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(round(value))}" if float(value).is_integer() else f"{value:g}")
    )
    _style_tick_labels(ax, axis="y")


def _plot_smoothed_line(
    ax: plt.Axes,
    steps: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    window: int = 10,
    shade_scale: float = 1.0,
) -> None:
    if len(steps) == 0 or len(values) == 0:
        return
    order = np.argsort(steps)
    steps = np.asarray(steps, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    if len(steps) > 240:
        grid = np.linspace(float(steps[0]), float(steps[-1]), 240)
        values = np.interp(grid, steps, values)
        steps = grid
    mean, std = rolling_mean_std(values, window=min(window, len(values)))
    ax.plot(
        steps,
        mean,
        color=color,
        linewidth=1.9,
        linestyle=linestyle,
        label=label,
        solid_capstyle="round",
        zorder=3,
    )
    ax.fill_between(
        steps,
        mean - shade_scale * std,
        mean + shade_scale * std,
        color=color,
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )


_LEARNING_LEGEND_KW = dict(
    frameon=True, fancybox=False, edgecolor="#CCCCCC",
    columnspacing=0.9, handlelength=1.25, handletextpad=0.4, labelspacing=0.35, borderpad=0.28,
)


def _learning_panel_legend(ax: plt.Axes, *, fs: int, ncol: int = 1, loc: str = "lower right", handles=None, labels=None) -> None:
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(handles=handles, labels=labels, loc=loc, ncol=ncol, fontsize=fs - 2, **_LEARNING_LEGEND_KW)
    legend.get_frame().set_facecolor("#FFFFFF")
    _style_legend_text(legend)


def _finalize_learning_panel(
    ax: plt.Axes,
    *,
    title: str,
    fs: int,
    top_scale: float = 1.08,
    y_top: float | None = None,
) -> None:
    ax.set_title(_bold_label(title), fontsize=fs + 1, color="#111111", pad=10)
    ax.set_xlabel(_bold_label("Environment Steps"))
    ax.set_ylabel(_bold_label("Reward"))
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    _add_learning_line_axis_buffer(ax, top_scale=top_scale)
    if y_top is not None:
        y_bottom, _ = ax.get_ylim()
        ax.set_ylim(y_bottom, y_top)
    _set_learning_step_ticks(ax)
    _set_nice_learning_y_ticks(ax, y_top=y_top)


def _plot_final_rl_ablation_panel(ax: plt.Axes, alpha_key: str, *, fs: int) -> None:
    runs = _load_ppo_runs(alpha_key)
    mode_data: dict[str, list[dict[str, list[float]]]] = {}
    for run in runs:
        mode = run.get(
            "ppo_reward_mode",
            run.get("config", {}).get("ppo_reward_mode", "?"),
        )
        mode_data.setdefault(mode, []).append(
            {
                "history": run["history"],
                "steps": run.get("steps", list(range(len(run["history"])))),
            }
        )

    for mode in MODE_ORDER:
        entries = mode_data.get(mode, [])
        if not entries:
            continue
        min_len = min(len(entry["history"]) for entry in entries)
        trimmed = np.asarray([entry["history"][:min_len] for entry in entries], dtype=float)
        steps = np.asarray(entries[0]["steps"][:min_len], dtype=float)
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)
        if len(mean_curve) >= 3:
            mean_curve, _ = rolling_mean_std(mean_curve, window=min(LEARNING_SMOOTHING_WINDOW, len(mean_curve)))
            std_curve, _ = rolling_mean_std(std_curve, window=min(LEARNING_SMOOTHING_WINDOW, len(std_curve)))
        style = MODE_STYLES[mode]
        color = LEARNING_MODE_COLORS.get(mode, style["color"])
        ax.plot(
            steps,
            mean_curve,
            color=color,
            linewidth=1.75,
            linestyle=LEARNING_MODE_LINESTYLES.get(mode, "-"),
            label=LEARNING_MODE_LABELS.get(mode, style["label"]),
            zorder=3,
        )
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )

    if alpha_key == "1_0":
        _finalize_learning_panel(ax, title="End-to-End RL Reward Shaping", fs=fs, top_scale=1.025)
        _set_learning_y_ticks_with_padding(ax, [-10.0, 0.0, 10.0, 20.0, 30.0], top_pad_frac=0.05)
    else:
        _finalize_learning_panel(ax, title="End-to-End RL Reward Shaping", fs=fs, y_top=0.0)


def _plot_final_training_comparison_panel(ax: plt.Axes, alpha_key: str, *, fs: int) -> None:
    metric_key = "eval/episode_terminal_reward"

    at_steps, at_values = _load_mcts_metric(alpha_key, 500, metric_key)
    if len(at_steps):
        mask = at_steps <= LEARNING_DATA_STEP_LIMIT
        _plot_smoothed_line(
            ax,
            at_steps[mask],
            at_values[mask],
            color=WALL_CLOCK_COLORS["alphatransit"],
            label="AlphaTransit",
            linestyle="-",
            window=LEARNING_SMOOTHING_WINDOW,
        )

    rl_steps, rl_values, _ = _extract_ppo_mode(alpha_key)
    if len(rl_steps):
        _plot_smoothed_line(
            ax,
            rl_steps,
            rl_values,
            color="#E84393",
            label="End-to-End RL",
            linestyle="--",
            window=LEARNING_SMOOTHING_WINDOW,
        )

    _finalize_learning_panel(ax, title="AlphaTransit vs End-to-End RL", fs=fs, top_scale=1.025)
    if alpha_key == "1_0":
        _set_learning_y_ticks_with_padding(ax, [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0], top_pad_frac=0.05)
    else:
        _set_fixed_learning_y_range(ax, -13.0, 3.0)


def _compact_seconds_label(value: float, _: int) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:g}K"
    return f"{value:g}"


def _round_up_to_nice(value: float) -> float:
    if value <= 0:
        return 1.0
    n = int(np.floor(np.log10(value)))
    base = value / (10.0 ** n)
    nice = 1.0 if base <= 1 else 2.0 if base <= 2 else 5.0 if base <= 5 else 10.0
    return nice * (10.0 ** n)


def _plot_final_wall_clock_panel(ax: plt.Axes, alpha_key: str, *, fs: int) -> None:
    alpha_value = 0.3 if alpha_key == "0_3" else 1.0
    n_iters = [100, 200, 300, 400, 500]
    log_dir = NEURIPS_RESULTS_DIR / "wall_clock_scaling"
    time_colors = {
        "Pure MCTS": "#E84393",
        "AlphaTransit": WALL_CLOCK_COLORS["alphatransit"],
    }

    rows: list[dict[str, float | int | str]] = []
    for method, path in [
        ("Pure MCTS", log_dir / f"pure_mcts_alpha_{alpha_key}_log.csv"),
        ("AlphaTransit", log_dir / f"alphatransit_alpha_{alpha_key}_log.csv"),
    ]:
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if data.empty:
            continue
        data = data[np.isclose(data["alpha"].astype(float), alpha_value)]
        for n_iter, group in data.groupby("n_iter"):
            route_lengths = [
                int(route_group["route_length"].max()) + 1
                for _, route_group in group.groupby("route_idx")
            ]
            if len(route_lengths) != 5 or any(length != 14 for length in route_lengths):
                continue
            values = group["step_time_s"].astype(float).to_numpy()
            if values.size == 0:
                continue
            mean = float(values.mean())
            rows.append(
                {
                    "method": method,
                    "n_iter": int(n_iter),
                    "mean": mean,
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
            )

    plot_df = pd.DataFrame(rows)

    def plot_method(target_ax: plt.Axes, method: str, color: str, zorder: int) -> None:
        if plot_df.empty:
            return
        method_df = plot_df[plot_df["method"] == method].sort_values("n_iter")
        if method_df.empty:
            return
        means = method_df["mean"].to_numpy(dtype=float)
        stds = method_df["std"].to_numpy(dtype=float)
        lower = np.minimum(stds, np.maximum(means - 1e-6, 1e-6))
        upper = stds
        target_ax.errorbar(
            method_df["n_iter"].to_numpy(dtype=int),
            means,
            yerr=np.vstack([lower, upper]),
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=4.0,
            capthick=1.2,
            markersize=6.5,
            label=method,
            zorder=zorder,
        )

    if plot_df.empty:
        ax.set_title(_bold_label("Search Time"), fontsize=fs + 1, color="#111111")
        ax.set_xlabel(_bold_label("MCTS Simulations"))
        ax.set_ylabel(_bold_label("Seconds / Decision"))
        return

    ax.set_axis_off()
    # Tight gap between bands so the broken-axis indicators read as a pair.
    upper_ax = ax.inset_axes([0.0, 0.50, 1.0, 0.50])
    lower_ax = ax.inset_axes([0.0, 0.00, 1.0, 0.48], sharex=upper_ax)

    for band_ax in (upper_ax, lower_ax):
        _style_learning_axis(band_ax)
        band_ax.set_yscale("log")
        band_ax.set_xticks(n_iters)
        band_ax.set_xlim(70, 530)
        band_ax.grid(True, which="major", axis="both", zorder=0)
        band_ax.grid(False, which="minor", axis="y")
        band_ax.tick_params(axis="y", which="major", length=7.0, width=0.85, color=TICK_TEXT_COLOR)
        # Drop minor tick marks: they're unlabeled stubs that crowd the major ticks.
        band_ax.tick_params(axis="y", which="minor", left=False, right=False)

    plot_method(upper_ax, "Pure MCTS", time_colors["Pure MCTS"], 5)
    plot_method(lower_ax, "AlphaTransit", time_colors["AlphaTransit"], 5)

    # Upper band (log): per-alpha ticks so the data fits inside the band.
    # α=0.3 Pure MCTS spans 700-5150 → [500, 2K, 8K]; α=1.0 spans 138-947 → [100, 500, 2K].
    # Lower band (linear): hard-coded [0, 4, 8] starting at 0. Both get a small buffer
    # on every edge so labels don't crowd the panel boundary or the broken-axis indicators.
    upper_ticks = [100.0, 500.0, 2000.0] if alpha_key == "1_0" else [500.0, 2000.0, 8000.0]
    upper_ax.set_yscale("log")
    upper_ax.set_ylim(upper_ticks[0] / 1.85, upper_ticks[-1] * 1.15)
    upper_ax.set_yticks(upper_ticks)
    upper_ax.yaxis.set_minor_formatter(NullFormatter())
    upper_ax.yaxis.set_major_formatter(FuncFormatter(_compact_seconds_label))
    _style_tick_labels(upper_ax, axis="y")

    lower_ticks = [0.0, 4.0, 8.0]
    lower_ax.set_yscale("linear")
    lower_ax.set_ylim(-lower_ticks[-1] * 0.10, lower_ticks[-1] * 1.20)
    lower_ax.set_yticks(lower_ticks)
    lower_ax.yaxis.set_minor_formatter(NullFormatter())
    lower_ax.yaxis.set_major_formatter(FuncFormatter(_compact_seconds_label))
    _style_tick_labels(lower_ax, axis="both")

    upper_ax.spines["bottom"].set_visible(False)
    lower_ax.spines["top"].set_visible(False)
    upper_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    break_kwargs = dict(color="#666666", linewidth=0.9, clip_on=False)
    x_center = 0.0  # so the y-axis spine bisects each diagonal break mark
    upper_ax.plot(
        (x_center - 0.015, x_center + 0.015),
        (-0.018, 0.018),
        transform=upper_ax.transAxes,
        **break_kwargs,
    )
    lower_ax.plot(
        (x_center - 0.015, x_center + 0.015),
        (0.982, 1.018),
        transform=lower_ax.transAxes,
        **break_kwargs,
    )

    upper_ax.set_title(_bold_label("Search Time"), fontsize=fs + 1, color="#111111", pad=10)
    lower_ax.set_xlabel(_bold_label("MCTS Simulations"))
    lower_ax.xaxis.label.set_color("#111111")
    ax.text(
        -0.17,
        0.50,
        _bold_label("Seconds / Decision"),
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs + 2,
        color="#111111",
    )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=time_colors[label],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=7.2,
            label=label,
        )
        for label in ("AlphaTransit", "Pure MCTS")
    ]
    legend = lower_ax.legend(
        handles=legend_handles,
        labels=["AlphaTransit", "Pure MCTS"],
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=fs - 2,
        columnspacing=0.9,
        handlelength=1.25,
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.28,
    )
    legend.get_frame().set_facecolor("#FFFFFF")
    _style_legend_text(legend)


def plot_learning_overview(
    output_path: Path = NEURIPS_RESULTS_DIR / "final_learning_overview_alpha_1_0.pdf",
    *,
    alpha_key: str = "1_0",
) -> Path:
    """
    Build the final RL/AlphaTransit training and search-time overview for one alpha.

    Left panel ("End-to-End RL Reward Shaping"): PPO reward-ablation curves
    grouped by reward mode. Curves average all available runs for each mode and
    use the PPO evaluation reward histories loaded for the requested alpha.

    Middle panel ("AlphaTransit vs End-to-End RL"): AlphaTransit (500 MCTS
    simulations per decision) versus the selected End-to-End RL curve. Both
    curves are smoothed over training steps, clipped to 1,020,000 steps, and
    plotted with a small right x-axis buffer.

    Right panel ("Search Time"): per-decision wall-clock timing for Pure MCTS
    and AlphaTransit over n_iter in {100, 200, 300, 400, 500}. Curves summarize
    a CPU-only, one-worker Bloomington benchmark over five fixed-length routes
    (14 stops each); points show mean decision time and error bars show one
    standard deviation across route-construction decisions.
    """
    fs = LEARNING_OVERVIEW_FS
    apply_plot_style(fs, background=BACKGROUND)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))

    for ax in axes:
        _style_learning_axis(ax)

    _plot_final_rl_ablation_panel(axes[0], alpha_key, fs=fs)
    _plot_final_training_comparison_panel(axes[1], alpha_key, fs=fs)
    _plot_final_wall_clock_panel(axes[2], alpha_key, fs=fs)

    # Search Time attaches its own legend to lower_ax inside the panel function;
    # the outer ax is set_axis_off'd, so only the two reward panels are handled here.
    for ax, ncol in zip((axes[0], axes[1]), (2, 1)):
        _learning_panel_legend(ax, fs=fs, ncol=ncol)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.16, wspace=0.30)
    _save_figure_with_optional_png(fig, output_path, facecolor=BACKGROUND)
    plt.close(fig)
    return output_path


# ============================================================================
# Overview maps: Bloomington (street + demand) + Laval network
# ============================================================================
OVERVIEW_MAP_BACKGROUND = "#FFFFFF"
_DEMAND_COLOR_ORIGIN = "#16A6FF"
_DEMAND_COLOR_DEST = "#F0447A"
_TOP_ORIGIN_COLOR = "#1677FF"
_TOP_DEST_COLOR = "#D92D5C"
_BLOOMINGTON_HUB_NODE = "96"
_LAVAL_HUB_NODE = "542"
_BASEMAP_TILE_ZOOM_BLOOMINGTON = 14
_BASEMAP_TILE_ZOOM_LAVAL = 13
_BLOOMINGTON_CRS = "EPSG:32616"
# Laval graph overlay: source coords are EPSG:3347 meters that the legacy
# converter mistakenly multiplied by 0.3048 (claiming feet→meters). Undo the
# scale, then reproject to EPSG:3857 for exact placement on the basemap.
_LAVAL_SOURCE_CRS = "EPSG:3347"
_LAVAL_SOURCE_UNSCALE = 1.0 / 0.3048
_LAVAL_BASEMAP_PAD_FRAC = 0.08  # fraction of graph extent to pad basemap bbox
_NODE_DOT_CMAP = "YlOrBr"
_BLOOMINGTON_NODE_COLOR = "#EC7014"  # deeper orange (YlOrBr step past mid)
_LAVAL_NODE_COLOR = "#FF7A00"  # neon orange, brighter than Bloomington
_HUB_PIN_ZOOM = 0.056
_LEGEND_PIN_ZOOM = 0.0375
_NETWORK_EDGE_ALPHA = 0.32
_DENSITY_ORIGIN = dict(
    bw_method=0.09,
    levels=[0.08, 0.16, 0.28, 0.42, 0.58, 0.76, 0.92],
    fill_alphas=[0.035, 0.032, 0.028, 0.023, 0.018, 0.013, 0.009],
    line_alpha=0.68,
)
_DENSITY_DEST = dict(
    bw_method=0.13,
    levels=[0.035, 0.07, 0.12, 0.20, 0.32, 0.48, 0.66, 0.84],
    fill_alphas=[0.055, 0.050, 0.043, 0.036, 0.028, 0.020, 0.014, 0.010],
    line_alpha=0.76,
)
_PANEL_LEGEND_KW = dict(
    frameon=True, fancybox=False, edgecolor="#D6DBE3",
    handlelength=1.4, handletextpad=0.55, labelspacing=0.35, borderpad=0.35,
)


class _RedPinLegendHandle:
    pass


class _RedPinLegendHandler(HandlerBase):
    def __init__(self, pin_image, zoom: float = 0.05):
        super().__init__()
        self.pin_image = pin_image
        self.zoom = zoom

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artist = AnnotationBbox(
            OffsetImage(self.pin_image, zoom=self.zoom, resample=True),
            (xdescent + width / 2.0, ydescent + height / 2.0),
            xycoords=trans,
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
        )
        return [artist]


def _pin_image(filename: str) -> np.ndarray | None:
    pin_path = ASSETS_DIR / filename
    if not pin_path.exists():
        return None
    pin_img = plt.imread(pin_path).copy()
    if pin_img.shape[2] == 4:
        pin_img[:, :, 3] = pin_img[:, :, 3] * 0.9
    return pin_img


def _transit_center_legend_handle(pin_filename: str = "pin_blue.png", fallback_color: str = "#1A73E8") -> tuple[object, dict]:
    pin_img = _pin_image(pin_filename)
    if pin_img is None:
        return (
            mlines.Line2D([], [], marker="o", linestyle="None", color=fallback_color, markeredgecolor="#FFFFFF", markersize=8.0),
            {},
        )
    handle = _RedPinLegendHandle()
    return handle, {_RedPinLegendHandle: _RedPinLegendHandler(pin_img, zoom=_LEGEND_PIN_ZOOM)}


def _bbox_from_coords(coords: dict) -> tuple[float, float, float, float]:
    xs = np.fromiter((c[0] for c in coords.values()), dtype=float, count=len(coords))
    ys = np.fromiter((c[1] for c in coords.values()), dtype=float, count=len(coords))
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _aggregate_node_demand(demand: pd.DataFrame) -> pd.Series:
    return (
        demand.groupby("orig")["volume"].sum()
        .add(demand.groupby("dest")["volume"].sum(), fill_value=0.0)
    )


def _bloomington_data() -> dict:
    nodes = load_nodes("bloomington")
    links = load_links("bloomington")
    demand = pd.read_csv(
        NETWORKS_DIR / "bloomington" / "bloomington_demand_standard.csv",
        dtype={"orig": str, "dest": str},
    )
    coords = _coords_dict(nodes)
    origin_demand = demand.groupby("orig")["volume"].sum()
    dest_demand = demand.groupby("dest")["volume"].sum()
    degree = pd.concat([links["start"].astype(str), links["end"].astype(str)]).value_counts()
    return {
        "links": links,
        "coords": coords,
        "origin_demand": origin_demand,
        "dest_demand": dest_demand,
        "top_origin": str(origin_demand.idxmax()),
        "top_dest": str(dest_demand.idxmax()),
        "node_demand": _aggregate_node_demand(demand),
        "degree": degree,
        "bbox": _bbox_from_coords(coords),
        "hub_node": _BLOOMINGTON_HUB_NODE,
        "basemap_crs": _BLOOMINGTON_CRS,
        "basemap_zoom": _BASEMAP_TILE_ZOOM_BLOOMINGTON,
        "basemap_rot_deg": 0.0,
    }


def _laval_data() -> dict:
    """Reproject Laval graph through EPSG:3347 → EPSG:3857; basemap bbox auto-fits the graph."""
    if not _HAS_PYPROJ:
        return {"bbox": None, "basemap_crs": None, "basemap_zoom": _BASEMAP_TILE_ZOOM_LAVAL, "basemap_rot_deg": 0.0}
    nodes = load_nodes("laval")
    links = load_links("laval")
    raw_coords = _coords_dict(nodes)
    src_to_wm = _Transformer.from_crs(_LAVAL_SOURCE_CRS, "EPSG:3857", always_xy=True)
    coords = {
        nid: src_to_wm.transform(sx * _LAVAL_SOURCE_UNSCALE, sy * _LAVAL_SOURCE_UNSCALE)
        for nid, (sx, sy) in raw_coords.items()
    }
    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 * (1.0 + _LAVAL_BASEMAP_PAD_FRAC)
    degree = pd.concat([links["start"].astype(str), links["end"].astype(str)]).value_counts()
    return {
        "bbox": (cx - half, cx + half, cy - half, cy + half),
        "basemap_crs": "EPSG:3857",
        "basemap_zoom": _BASEMAP_TILE_ZOOM_LAVAL,
        "basemap_rot_deg": 0.0,
        "links": links,
        "coords": coords,
        "degree": degree,
        "node_demand": {},
        "hub_node": _LAVAL_HUB_NODE,
    }


def _setup_map_axis(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    pad_frac: float = 0.02,
) -> None:
    xmin, xmax, ymin, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half = max(xmax - xmin, ymax - ymin) / 2.0
    pad = max(half * pad_frac, 1.0)
    half += pad
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(0.6)


def _set_km_ticks(ax: plt.Axes, source_crs: str | None, fs: int) -> None:
    """Tick labels as ground km offset from panel center, integer step picked by panel size."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    cx, cy = (xlim[0] + xlim[1]) / 2.0, (ylim[0] + ylim[1]) / 2.0
    half_units = (xlim[1] - xlim[0]) / 2.0

    unit_to_m = 1.0
    if source_crs == "EPSG:3857" and _HAS_PYPROJ:
        to_lonlat = _Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        _, lat_center = to_lonlat.transform(cx, cy)
        unit_to_m = float(np.cos(np.radians(lat_center)))

    half_km = half_units * unit_to_m / 1000.0
    step_km = 5 if half_km > 8 else 2 if half_km > 4 else 1
    n_steps = int(half_km // step_km)
    km_ticks = list(range(-n_steps * step_km, (n_steps + 1) * step_km, step_km))
    units_per_km = 1000.0 / unit_to_m

    ax.set_xticks([cx + k * units_per_km for k in km_ticks])
    ax.set_yticks([cy + k * units_per_km for k in km_ticks])
    ax.set_xticklabels([f"{k}" for k in km_ticks])
    ax.set_yticklabels([f"{k}" for k in km_ticks])
    ax.tick_params(axis="both", which="major", labelsize=fs - 1, colors=TICK_TEXT_COLOR, length=3, width=0.5)
    _style_tick_labels(ax)
    ax.set_xlabel(_bold_label("Easting (km)"), fontsize=fs + 2, color="#111111", labelpad=4)
    ax.set_ylabel(_bold_label("Northing (km)"), fontsize=fs + 2, color="#111111", labelpad=4)


def _boost_basemap_saturation(arr: np.ndarray, sat: float = 1.45, contrast: float = 1.08) -> np.ndarray:
    """Push saturation and contrast slightly so muted tiles read better."""
    a = np.asarray(arr).astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    rgb = a[..., :3]
    luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
    rgb = luma + (rgb - luma) * sat
    rgb = (rgb - 0.5) * contrast + 0.5
    rgb = np.clip(rgb, 0.0, 1.0)
    if a.shape[-1] == 4:
        return np.concatenate([rgb, a[..., 3:4]], axis=-1)
    return rgb


def _add_osm_basemap(ax: plt.Axes, *, crs: str | None, zoom: int, rotate_deg: float = 0.0) -> None:
    if not _HAS_CTX or crs is None:
        return
    try:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if rotate_deg == 0.0:
            xpad = (xlim[1] - xlim[0]) * 0.15
            ypad = (ylim[1] - ylim[0]) * 0.15
            ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
            ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
            _ctx.add_basemap(ax, source=_ctx.providers.CartoDB.VoyagerNoLabels, alpha=1.0, attribution=False, crs=crs, reset_extent=False, zoom=zoom)
            img_obj = ax.images[-1]
            img_obj.set_data(_boost_basemap_saturation(img_obj.get_array()))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            xpad = (xlim[1] - xlim[0]) * 0.5
            ypad = (ylim[1] - ylim[0]) * 0.5
            west, east = xlim[0] - xpad, xlim[1] + xpad
            south, north = ylim[0] - ypad, ylim[1] + ypad
            img, extent = _ctx.bounds2img(west, south, east, north, zoom=zoom, source=_ctx.providers.CartoDB.VoyagerNoLabels, ll=False)
            img = _boost_basemap_saturation(img)
            cx = (xlim[0] + xlim[1]) / 2.0
            cy = (ylim[0] + ylim[1]) / 2.0
            trans = Affine2D().rotate_deg_around(cx, cy, rotate_deg) + ax.transData
            ax.imshow(img, extent=extent, interpolation="bilinear", origin="upper", zorder=0, transform=trans)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    except Exception as exc:
        print(f"Skipping basemap: {exc}")


def _draw_demand_density(
    ax: plt.Axes,
    data: dict,
    demand_by_node: pd.Series,
    *,
    color: str,
    zorder: int,
    bw_method: float,
    levels: list[float],
    fill_alphas: list[float],
    line_alpha: float,
) -> None:
    if not _HAS_KDE:
        return
    coords = data["coords"]
    node_ids = [str(n) for n in demand_by_node.index if str(n) in coords and demand_by_node[n] > 0]
    if len(node_ids) < 3:
        return
    xs = np.asarray([coords[n][0] for n in node_ids], dtype=float)
    ys = np.asarray([coords[n][1] for n in node_ids], dtype=float)
    weights = np.asarray([float(demand_by_node[n]) for n in node_ids], dtype=float)
    xmin, xmax, ymin, ymax = data["bbox"]
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin - xpad, xmax + xpad, 220),
        np.linspace(ymin - ypad, ymax + ypad, 220),
    )
    kde = _gaussian_kde(np.vstack([xs, ys]), weights=weights, bw_method=bw_method)
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    if density.max() <= 0:
        return
    density /= density.max()
    for level, alpha in zip(levels, fill_alphas):
        ax.contourf(grid_x, grid_y, density, levels=[level, 1.05], colors=[color], alpha=alpha, zorder=zorder)
    line_widths = np.linspace(1.9, 0.55, len(levels) - 1)
    ax.contour(grid_x, grid_y, density, levels=levels[1:], colors=[color], linewidths=line_widths, alpha=line_alpha, zorder=zorder + 1)


def _panel_legend(ax: plt.Axes, handles, labels, fs: int, *, loc: str = "upper right", handler_map: dict | None = None) -> None:
    leg = ax.legend(handles=handles, labels=labels, fontsize=fs - 1, loc=loc, handler_map=handler_map, **_PANEL_LEGEND_KW)
    leg.get_frame().set_facecolor("#FFFFFF")
    _style_legend_text(leg)


def _plot_network_panel(
    ax: plt.Axes,
    data: dict,
    fs: int,
    *,
    title: str,
    vmin: float,
    vmax: float,
    legend_loc: str = "upper left",
    pad_frac: float = 0.02,
    with_overlay: bool = True,
    cmap=_NODE_DOT_CMAP,
    solid_color: str | None = None,
):
    """Draw a network panel with demand-colored nodes and transit-center pin. Returns the scatter handle (or None)."""
    _setup_map_axis(ax, data["bbox"], pad_frac=pad_frac)
    _add_osm_basemap(ax, crs=data["basemap_crs"], zoom=data["basemap_zoom"], rotate_deg=data.get("basemap_rot_deg", 0.0))
    _set_km_ticks(ax, data["basemap_crs"], fs)
    ax.set_title(_bold_label(title), fontsize=fs + 1, color="#111111", pad=10)
    if not with_overlay:
        return None

    coords = data["coords"]
    draw_basemap(ax, data["links"], coords, color="#3F4654", linewidth=0.55, alpha=_NETWORK_EDGE_ALPHA, zorder=2)

    node_ids = list(coords.keys())
    nx = np.asarray([coords[n][0] for n in node_ids])
    ny = np.asarray([coords[n][1] for n in node_ids])
    if solid_color is None:
        degrees = np.asarray([data["degree"].get(n, 1) for n in node_ids], dtype=float)
        demands = np.asarray([data["node_demand"].get(n, 0.0) for n in node_ids], dtype=float)
        log_demands = np.log1p(demands)
        deg_range = max(degrees.max() - degrees.min(), 1.0)
        sizes = 3.0 + (degrees - degrees.min()) / deg_range * 40.0
        sc = ax.scatter(nx, ny, s=sizes, c=log_demands, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="none", alpha=0.85, zorder=3)
        legend_marker_color = "#D95F0E"
    else:
        sc = ax.scatter(nx, ny, s=14.0, color=solid_color, edgecolors="none", alpha=0.9, zorder=3)
        legend_marker_color = solid_color

    draw_transit_center(ax, coords, zoom=_HUB_PIN_ZOOM, node_id=data["hub_node"], pin="pin_blue.png")

    pin_handle, pin_handler_map = _transit_center_legend_handle()
    edge_handle = mlines.Line2D([], [], color="#3F4654", linewidth=2.0, alpha=0.7)
    node_handle = mlines.Line2D([], [], marker="o", linestyle="None", color=legend_marker_color, markeredgecolor="#FFFFFF", markersize=8)
    _panel_legend(
        ax,
        handles=[edge_handle, node_handle, pin_handle],
        labels=[f"{len(data['links'])} edges", f"{len(coords)} nodes", "Transit center"],
        fs=fs,
        loc=legend_loc,
        handler_map=pin_handler_map,
    )
    return sc


def _plot_bloomington_demand(ax: plt.Axes, data: dict, fs: int, *, pad_frac: float = 0.07) -> None:
    _setup_map_axis(ax, data["bbox"], pad_frac=pad_frac)
    _add_osm_basemap(ax, crs=data["basemap_crs"], zoom=data["basemap_zoom"], rotate_deg=data.get("basemap_rot_deg", 0.0))
    _set_km_ticks(ax, data["basemap_crs"], fs)
    draw_basemap(ax, data["links"], data["coords"], color="#AEB8C4", linewidth=0.46, alpha=0.35, zorder=2)
    _draw_demand_density(ax, data, data["origin_demand"], color=_DEMAND_COLOR_ORIGIN, zorder=3, **_DENSITY_ORIGIN)
    _draw_demand_density(ax, data, data["dest_demand"], color=_DEMAND_COLOR_DEST, zorder=7, **_DENSITY_DEST)
    for node_id, marker, color, size in (
        (data["top_origin"], "^", _TOP_ORIGIN_COLOR, 170.0),
        (data["top_dest"], "*", _TOP_DEST_COLOR, 265.0),
    ):
        if node_id in data["coords"]:
            x, y = data["coords"][node_id]
            ax.scatter([x], [y], s=size, marker=marker, color=color, edgecolors="#FFFFFF", linewidths=1.2, zorder=25)
    ax.set_title(_bold_label("Bloomington Demand"), fontsize=fs + 1, color="#111111", pad=10)
    _panel_legend(
        ax,
        handles=[
            mlines.Line2D([], [], color=_DEMAND_COLOR_ORIGIN, linewidth=2.4),
            mlines.Line2D([], [], color=_DEMAND_COLOR_DEST, linewidth=2.4),
        ],
        labels=["Origins", "Destinations"],
        fs=fs,
    )


def plot_overview_maps(output_path: Path = NEURIPS_RESULTS_DIR / "final_overview_maps.pdf") -> Path:
    """Three-panel benchmark overview: Bloomington network, Bloomington demand, Laval basemap."""
    fs = LEARNING_OVERVIEW_FS
    apply_plot_style(fs, background=OVERVIEW_MAP_BACKGROUND)
    bloomington = _bloomington_data()
    laval = _laval_data()

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.4))
    _plot_network_panel(axes[0], bloomington, fs, title="Bloomington Network", vmin=0.0, vmax=1.0, legend_loc="upper right", pad_frac=0.07, solid_color=_BLOOMINGTON_NODE_COLOR)
    _plot_bloomington_demand(axes[1], bloomington, fs)
    _plot_network_panel(axes[2], laval, fs, title="Laval Network", vmin=0.0, vmax=1.0, solid_color=_LAVAL_NODE_COLOR)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.92, bottom=0.10, wspace=0.20)

    save_figure(fig, output_path, facecolor=OVERVIEW_MAP_BACKGROUND)
    plt.close(fig)
    return output_path


def plot_route_designs(
    output_dir: Path = NEURIPS_RESULTS_DIR,
    *,
    alpha_keys: Sequence[str] = ROUTE_DESIGN_ALPHA_KEYS,
    max_cols: int = 5,
) -> list[Path]:
    """
    Build one selected-route-design comparison grid per alpha setting.

    This final-plots entrypoint delegates to plots.routes.build_route_figure so
    route-file resolution and drawing stay centralized. The route module loads
    Bloomington network nodes/links, draws the gray street basemap, overlays each
    method's designed_routes.json, and marks the transit center. Algorithmic
    methods use the configured seed_42 route files; Real-World uses
    networks/bloomington/bloomington_existing_routes.json. AlphaTransit uses the
    n_iter=500 selected route design for each requested alpha.
    Typography for this final figure is controlled here via ROUTE_DESIGN_FS.

    The default bundle writes one file per alpha:
    routes_alpha_0_3.pdf and routes_alpha_1_0.pdf.
    """
    fs = ROUTE_DESIGN_FS
    output_paths: list[Path] = []
    for alpha_key in alpha_keys:
        output_path = output_dir / f"routes_alpha_{alpha_key}.pdf"
        build_route_design_figure(alpha_key, output_path, max_cols=max_cols, fs=fs)
        output_paths.append(output_path)
    return output_paths


PLOT_BUILDERS = {
    "method-comparison": plot_method_comparison_triptych,
    "scaling-overview": plot_scaling_behavior_overview,
    "learning-overview": plot_learning_overview,
    "overview-maps": plot_overview_maps,
    "route-designs": plot_route_designs,
}


def generate_default(
    output_dir: Path = NEURIPS_RESULTS_DIR,
    *,
    alpha_key: str = "1_0",
) -> None:
    plot_method_comparison_triptych(
        output_dir / f"final_comparison_alpha_{alpha_key}.pdf",
        alpha_key=alpha_key,
    )
    for scaling_alpha_key in ("1_0", "0_3"):
        plot_scaling_behavior_overview(
            output_dir / f"final_scaling_behavior_alpha_{scaling_alpha_key}.pdf",
            alpha_key=scaling_alpha_key,
        )
        plot_learning_overview(
            output_dir / f"final_learning_overview_alpha_{scaling_alpha_key}.pdf",
            alpha_key=scaling_alpha_key,
        )
    plot_overview_maps(output_dir / "final_overview_maps.pdf")
    plot_route_designs(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final paper plots built from local results.")
    subparsers = parser.add_subparsers(dest="command")

    method_comparison = subparsers.add_parser(
        "method-comparison",
        help="Build the three-panel service, passenger, and operator comparison figure.",
    )
    method_comparison.add_argument("--alpha", choices=["0_3", "1_0"], default="1_0")
    method_comparison.add_argument("--output", type=Path)

    scaling_overview = subparsers.add_parser(
        "scaling-overview",
        help="Build the combined scaling and sweep overview figure for one alpha setting.",
    )
    scaling_overview.add_argument("--alpha", choices=["0_3", "1_0"], default="1_0")
    scaling_overview.add_argument("--output", type=Path)

    learning_overview = subparsers.add_parser(
        "learning-overview",
        help="Build the final RL, AlphaTransit, and wall-clock overview figure for one alpha setting.",
    )
    learning_overview.add_argument("--alpha", choices=["0_3", "1_0"], default="1_0")
    learning_overview.add_argument("--output", type=Path)

    route_designs = subparsers.add_parser(
        "route-designs",
        help="Build one designed-route comparison grid per alpha setting.",
    )
    route_designs.add_argument(
        "--alphas",
        choices=["0_3", "1_0"],
        nargs="+",
        default=list(ROUTE_DESIGN_ALPHA_KEYS),
    )
    route_designs.add_argument("--max-cols", type=int, default=5)
    route_designs.add_argument("--output-dir", type=Path, default=NEURIPS_RESULTS_DIR)

    overview_maps = subparsers.add_parser(
        "overview-maps",
        help="Build the three-panel benchmark overview (Bloomington street + demand + Laval network).",
    )
    overview_maps.add_argument("--output", type=Path)

    bundle = subparsers.add_parser(
        "all",
        help="Build the default final-paper figure bundle.",
    )
    bundle.add_argument("--alpha", choices=["0_3", "1_0"], default="1_0")
    bundle.add_argument("--output-dir", type=Path, default=NEURIPS_RESULTS_DIR)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "method-comparison":
        output = args.output or (NEURIPS_RESULTS_DIR / f"final_comparison_alpha_{args.alpha}.pdf")
        plot_method_comparison_triptych(output, alpha_key=args.alpha)
        return

    if args.command == "scaling-overview":
        output = args.output or (NEURIPS_RESULTS_DIR / f"final_scaling_behavior_alpha_{args.alpha}.pdf")
        plot_scaling_behavior_overview(output, alpha_key=args.alpha)
        return

    if args.command == "learning-overview":
        output = args.output or (NEURIPS_RESULTS_DIR / f"final_learning_overview_alpha_{args.alpha}.pdf")
        plot_learning_overview(output, alpha_key=args.alpha)
        return

    if args.command == "route-designs":
        plot_route_designs(args.output_dir, alpha_keys=tuple(args.alphas), max_cols=args.max_cols)
        return

    if args.command == "overview-maps":
        output = args.output or (NEURIPS_RESULTS_DIR / "final_overview_maps.pdf")
        plot_overview_maps(output)
        return

    if args.command == "all":
        generate_default(args.output_dir, alpha_key=args.alpha)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

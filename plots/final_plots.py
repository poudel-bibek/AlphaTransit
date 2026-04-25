from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np

try:
    from .common import (
        NEURIPS_RESULTS_DIR,
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
        PARETO_ABBREV,
        PARETO_STYLES,
        TRAINING_STYLES,
        WALL_CLOCK_COLORS,
        _extract_ppo_mode,
        _load_mcts_metric,
        _load_ppo_runs,
        _load_wall_clock_scaling,
        get_pareto_summary_paths,
    )
    from .routes import build_route_figure as build_route_design_figure
except ImportError:
    from common import (
        NEURIPS_RESULTS_DIR,
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
        PARETO_ABBREV,
        PARETO_STYLES,
        TRAINING_STYLES,
        WALL_CLOCK_COLORS,
        _extract_ppo_mode,
        _load_mcts_metric,
        _load_ppo_runs,
        _load_wall_clock_scaling,
        get_pareto_summary_paths,
    )
    from routes import build_route_figure as build_route_design_figure


METHOD_COMPARISON_FS = 18
COMBINED_SCALING_FS = 18
SWEEP_BEHAVIOR_FS = 18
SCALING_OVERVIEW_FS = 20
LEARNING_OVERVIEW_FS = 20
ROUTE_DESIGN_FS = 21
LEARNING_SMOOTHING_WINDOW = 8
LEARNING_DATA_STEP_LIMIT = 1_020_000
LEARNING_AXIS_STEP_LIMIT = 1_040_000
ROUTE_DESIGN_ALPHA_KEYS = ("0_3", "1_0")
BACKGROUND = "#FFFFFF"
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
    ("0_3", "passenger", "RL"): (12, 10),
    ("0_3", "passenger", "AlphaTransit"): (-10, 0),
    ("0_3", "passenger", "Genetic Alg."): (10, 0),
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
    "RL": "Reinforcement Learning",
}
LEARNING_MODE_LABELS = {
    "terminal_only": "Terminal",
    "terminal_intermediate_raw_early_stop": "Raw + ES",
    "terminal_intermediate_delta_early_stop": r"$\Delta$ + ES",
    "terminal_intermediate_delta_no_early_stop": r"$\Delta$ + No ES",
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

    for method, path in get_pareto_summary_paths(alpha_key).items():
        results = load_results_summary(_resolve_summary_path(path))
        points[method] = {
            "service_rate": _metric_stats(results["service_rate"], scale=100.0),
            "fleet_size": _metric_stats(results["fleet_size"]),
            "passenger_time": _combined_stats(
                results["combined_avg_wait_minutes"],
                results["combined_avg_travel_minutes"],
            ),
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
    for text in legend.get_texts():
        text.set_color("#111111")


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
) -> Path:
    """
    Build the final three-panel method-comparison figure from local eval summaries.

    The panels intentionally separate the story into:
    - Service Rate vs Fleet Size: headline coverage-vs-cost comparison
    - Passenger Burden: Wait + Travel versus Transfer Rate
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
                fontsize=fs - 2,
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
                s=style["size"],
                zorder=style["zorder"],
                edgecolors="white",
                linewidth=0.5,
            )

        if title:
            ax.set_title(title, fontsize=fs + 1, color="#111111")
        ax.set_xlabel(_bold_label(x_label))
        ax.set_ylabel(_bold_label(y_label))
        ax.xaxis.label.set_color("#111111")
        ax.yaxis.label.set_color("#111111")
        ax.tick_params(axis="both", colors="#666666")
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
            if alpha_key == "1_0" and panel_key == "passenger":
                x_right = 72.0
            if alpha_key == "0_3" and panel_key == "passenger":
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
            add_best_region_overlay()

        return panel_points

    panel_points = []

    panel_points.append(draw_panel(
        axes[0],
        panel_key="headline",
        x_key="fleet_size",
        y_key="service_rate",
        title=r"\textbf{Coverage vs Resource}",
        x_label=r"Fleet Size",
        y_label=r"Service Rate (\%)",
    ))
    panel_points.append(draw_panel(
        axes[1],
        panel_key="passenger",
        x_key="passenger_time",
        y_key="transfer_rate",
        title=r"\textbf{Passenger Burden}",
        x_label=r"Wait + Travel Time (min)",
        y_label=r"Transfer Rate (\%)",
    ))
    panel_points.append(draw_panel(
        axes[2],
        panel_key="operator",
        x_key="bus_utilization",
        y_key="route_efficiency",
        title=r"\textbf{Operator Efficiency}",
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
            markersize=np.sqrt(PARETO_STYLES[method]["size"]),
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
    for text in legend.get_texts():
        text.set_color("#111111")
    fig.subplots_adjust(wspace=0.28, bottom=0.27)
    for ax, points_for_axis, panel_key in zip(axes, panel_points, ["headline", "passenger", "operator"]):
        place_labels(ax, points_for_axis, panel_key)
    _save_figure_with_optional_png(fig, output_path, facecolor=BACKGROUND)
    plt.close(fig)
    return output_path


def plot_combined_scaling_summary(
    output_path: Path = NEURIPS_RESULTS_DIR / "final_scaling_summary.pdf",
) -> Path:
    """
    Build a 2x3 AlphaTransit scaling figure with one alpha per row.

    Columns correspond to the three scaling axes; rows correspond to alpha settings.
    Each panel plots top-5 evaluation reward for the available runs on that axis.
    Search varies MCTS simulations per decision in {100, 200, 300, 400, 500};
    data diversity varies episodes per policy-update iteration in {8, 16, 24, 32};
    policy network size varies GAT blocks in {2, 4, 8, 16}. Each point summarizes
    the top-5 evaluation rewards from the corresponding W&B history CSV.
    """
    fs = COMBINED_SCALING_FS
    apply_plot_style(fs, background=BACKGROUND)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 6.2), sharex="col", sharey="row")
    row_order = ["1_0", "0_3"]

    family_series: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for family_key in SCALING_FAMILY_SPECS:
        family_series[family_key] = {}
        for alpha_key in SCALING_ALPHA_STYLES:
            series = _load_scaling_series(alpha_key, family_key)
            family_series[family_key][alpha_key] = series

    if not any(len(series[1]) for family in family_series.values() for series in family.values()):
        raise ValueError("No scaling-summary data found.")

    def row_axis_config(alpha_key: str) -> tuple[np.ndarray, float, float, str]:
        means: list[float] = []
        stds: list[float] = []
        for family_key in SCALING_FAMILY_SPECS:
            _, family_means, family_stds = family_series[family_key][alpha_key]
            means.extend(family_means.tolist())
            stds.extend(family_stds.tolist())
        if not means:
            raise ValueError(f"No scaling data for alpha={alpha_key}")

        lows = np.asarray([mean - std for mean, std in zip(means, stds)], dtype=float)
        highs = np.asarray([mean + std for mean, std in zip(means, stds)], dtype=float)
        vmin = float(lows.min())
        vmax = float(highs.max())

        if alpha_key == "1_0":
            tick_min = 3.0 * np.floor(vmin / 3.0)
            tick_max = 3.0 * np.floor(vmax / 3.0)
            if tick_max <= tick_min:
                tick_max = tick_min + 3.0
            ticks = np.arange(tick_min, tick_max + 0.1, 3.0)
            lower = min(vmin - 0.55, ticks[0] - 0.55)
            upper = max(vmax + 0.55, ticks[-1] + 0.55)
            fmt = "int"
        else:
            tick_min = np.floor(vmin) + 0.5
            if tick_min - 0.2 > vmin:
                tick_min -= 1.0
            tick_max = np.ceil(vmax - 0.5) + 0.5
            if tick_max + 0.2 < vmax:
                tick_max += 1.0
            ticks = np.arange(tick_min, tick_max + 0.1, 1.0)
            lower = min(vmin - 0.22, ticks[0] - 0.22)
            upper = max(vmax + 0.22, ticks[-1] + 0.22)
            fmt = "half"
        return ticks, float(lower), float(upper), fmt

    legend_handles = []
    for alpha_key, style in SCALING_ALPHA_STYLES.items():
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=style["color"],
                marker="o",
                linewidth=2.6,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=style["label"],
            )
        )

    row_configs = {alpha_key: row_axis_config(alpha_key) for alpha_key in row_order}

    for row, alpha_key in enumerate(row_order):
        style = SCALING_ALPHA_STYLES[alpha_key]
        y_ticks, y_bottom, y_top, y_fmt = row_configs[alpha_key]

        for col, (family_key, spec) in enumerate(SCALING_FAMILY_SPECS.items()):
            ax = axes[row, col]
            ax.set_facecolor("#FBFBF9")
            xs, means, stds = family_series[family_key][alpha_key]
            if len(xs):
                ax.fill_between(
                    xs,
                    means - stds,
                    means + stds,
                    color=style["color"],
                    alpha=0.10,
                    linewidth=0.0,
                    zorder=1,
                )
                ax.plot(
                    xs,
                    means,
                    color=style["color"],
                    linewidth=2.6,
                    marker="o",
                    markersize=7.6,
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    solid_capstyle="round",
                    zorder=3,
                )
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    fmt="none",
                    ecolor=style["color"],
                    elinewidth=1.05,
                    capsize=3,
                    alpha=0.85,
                    zorder=2,
                )

                best_idx = int(np.argmax(means))
                ax.annotate(
                    "best",
                    (xs[best_idx], means[best_idx] + stds[best_idx]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fs - 5,
                    color="#666666",
                    fontweight="bold",
                )

            if row == 0:
                ax.set_title(spec["title"], fontsize=fs + 1, color="#1F2937")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(_bold_label(spec["x_label"]))
                ax.xaxis.label.set_color("#111111")

            ax.tick_params(axis="both", colors="#666666")
            ax.grid(True, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CCCCCC")
            ax.spines["bottom"].set_color("#CCCCCC")

            ax.set_xticks(spec["values"])
            x_values = spec["values"]
            x_pad = (max(x_values) - min(x_values)) * 0.06 if len(x_values) > 1 else 1.0
            ax.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad)
            ax.set_ylim(y_bottom, y_top)
            ax.set_yticks(y_ticks)
            if y_fmt == "int":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))

            if col > 0:
                ax.tick_params(labelleft=False)

        axes[row, 0].set_ylabel(_bold_label("Top-5 Avg Reward"))
        axes[row, 0].yaxis.label.set_color("#111111")

    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 0.028),
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.6,
        borderpad=0.3,
    )
    for text in legend.get_texts():
        text.set_color("#1F2937")

    fig.subplots_adjust(left=0.085, right=0.99, top=0.87, bottom=0.18, wspace=0.18, hspace=0.24)

    for row, alpha_key in enumerate(row_order):
        bbox = axes[row, 0].get_position()
        row_center = (bbox.y0 + bbox.y1) / 2.0
        style = SCALING_ALPHA_STYLES[alpha_key]
        fig.text(
            0.036,
            row_center,
            style["label"],
            rotation=90,
            va="center",
            ha="center",
            fontsize=fs - 1,
            color=style["color"],
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#FFFFFF",
                "edgecolor": style["color"],
                "linewidth": 1.3,
            },
        )

    for ax in axes.ravel():
        bbox = ax.get_position()
        panel = FancyBboxPatch(
            (bbox.x0 - 0.006, bbox.y0 - 0.014),
            bbox.width + 0.012,
            bbox.height + 0.028,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            transform=fig.transFigure,
            facecolor="#FBFBF9",
            edgecolor="#D6DBE3",
            linewidth=1.2,
            zorder=-10,
        )
        fig.add_artist(panel)

    _save_figure_with_optional_png(fig, output_path, facecolor=SCALING_BACKGROUND)
    plt.close(fig)
    return output_path


def plot_sweep_behavior_dashboard(
    output_path: Path = NEURIPS_RESULTS_DIR / "final_sweep_behavior_alpha_1_0.pdf",
    *,
    alpha_key: str = "1_0",
) -> Path:
    """
    Build a 2x3 sweep-behavior dashboard for one alpha setting.

    Top row: MCTS reward dynamics for each sweep family.
    Bottom row: top-5 eval reward versus runtime, with a dashed Pareto-like frontier.
    The sweep families match SCALING_FAMILY_SPECS: search depth is n_iter
    {100, 200, 300, 400, 500}, data diversity is ep_per_iter {8, 16, 24, 32},
    and policy network size is num_gat_blocks {2, 4, 8, 16}. Training curves use
    the W&B mcts/avg_reward metric, smoothed with rolling mean/std; runtime is
    taken from the maximum W&B _runtime value for each run.
    """
    fs = SWEEP_BEHAVIOR_FS
    apply_plot_style(fs, background=BACKGROUND)
    fig, axes = plt.subplots(2, 3, figsize=(16.6, 8.6), gridspec_kw={"height_ratios": [1.15, 1.0]})

    family_runs = {
        family_key: _load_sweep_behavior_runs(alpha_key, family_key)
        for family_key in SCALING_FAMILY_SPECS
    }
    if not any(family_runs.values()):
        raise ValueError(f"No sweep behavior data found for alpha={alpha_key}")

    top_rewards: list[float] = []
    bottom_rewards: list[float] = []
    for runs in family_runs.values():
        for run in runs:
            smooth_reward = np.asarray(run["reward_smooth"], dtype=float)
            smooth_std = np.asarray(run["reward_smooth_std"], dtype=float)
            top_rewards.extend((smooth_reward - smooth_std).tolist())
            top_rewards.extend((smooth_reward + smooth_std).tolist())
            bottom_rewards.append(float(run["top_mean"]))

    top_ticks, top_bottom, top_top = _five_integer_ticks_and_limits(min(top_rewards), max(top_rewards))
    bottom_ticks, bottom_bottom, bottom_top = _five_integer_ticks_and_limits(min(bottom_rewards), max(bottom_rewards))

    for col, family_key in enumerate(SCALING_FAMILY_SPECS):
        style = SWEEP_BEHAVIOR_STYLES[family_key]
        runs = family_runs[family_key]
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]

        top_ax.set_facecolor(SCALING_BACKGROUND)
        bottom_ax.set_facecolor(SCALING_BACKGROUND)

        legend_handles: list[mlines.Line2D] = []
        legend_labels: list[str] = []

        for line_idx, run in enumerate(runs):
            steps = np.asarray(run["steps"], dtype=float)
            smooth_reward = np.asarray(run["reward_smooth"], dtype=float)
            smooth_std = np.asarray(run["reward_smooth_std"], dtype=float)
            color = str(run["color"])
            linestyle = SWEEP_LINESTYLES[line_idx % len(SWEEP_LINESTYLES)]

            top_ax.fill_between(
                steps,
                smooth_reward - smooth_std,
                smooth_reward + smooth_std,
                color=color,
                alpha=0.10,
                linewidth=0.0,
                zorder=1,
            )
            top_ax.plot(
                steps,
                smooth_reward,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=3,
            )
            legend_handles.append(mlines.Line2D([], [], color=color, linestyle=linestyle, linewidth=2.0))
            legend_labels.append(str(run["legend_label"]))

        top_ax.set_title(style["title"], color=style["title_color"], fontsize=fs + 1)
        top_ax.set_xlabel(_bold_label("Training Step"))
        top_ax.xaxis.label.set_color("#111111")
        top_ax.tick_params(axis="both", colors="#666666")
        top_ax.grid(True, zorder=0)
        top_ax.set_ylim(top_bottom, top_top)
        top_ax.set_yticks(top_ticks)
        top_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
        if col == 0:
            top_ax.set_ylabel(_bold_label(SWEEP_TRAINING_REWARD_LABEL))
            top_ax.yaxis.label.set_color("#111111")
        else:
            top_ax.tick_params(labelleft=False)
        format_steps_axis(top_ax)

        if runs:
            max_step = max(float(np.asarray(run["steps"], dtype=float).max()) for run in runs)
            top_ax.set_xlim(0.0, max_step * 1.03)
            if family_key == "search":
                _draw_search_depth_legend(
                    top_ax,
                    legend_handles,
                    legend_labels,
                    fontsize=fs - 6,
                    edgecolor="#D5D8DE",
                )
            else:
                top_ax.legend(
                    legend_handles,
                    legend_labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.14),
                    ncol=min(3, len(legend_labels)),
                    frameon=True,
                    fancybox=False,
                    edgecolor="#D5D8DE",
                    fontsize=fs - 7,
                    handlelength=1.6,
                    columnspacing=0.9,
                    borderpad=0.28,
                    handletextpad=0.5,
                )

        runtimes = np.asarray([float(run["runtime_seconds"]) for run in runs], dtype=float)
        qualities = np.asarray([float(run["top_mean"]) for run in runs], dtype=float)
        colors = [str(run["color"]) for run in runs]

        if len(runs):
            sizes = np.linspace(210, 390, len(runs))
            order = np.argsort(runtimes)
            frontier_x = runtimes[order]
            frontier_y = np.maximum.accumulate(qualities[order])

            bottom_ax.plot(
                frontier_x,
                frontier_y,
                color="#666666",
                linewidth=1.4,
                linestyle=(0, (4, 4)),
                zorder=1,
            )
            bottom_ax.scatter(
                runtimes,
                qualities,
                s=sizes,
                c=colors,
                edgecolors="white",
                linewidths=0.9,
                alpha=0.96,
                zorder=3,
            )

            for run, size in zip(runs, sizes):
                bottom_ax.annotate(
                    str(run["point_label"]),
                    (float(run["runtime_seconds"]), float(run["top_mean"])),
                    xytext=(0, -14),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs - 6,
                    color="#333333",
                )

            if len(frontier_x) >= 2:
                anchor_idx = min(1, len(frontier_x) - 1)
                bottom_ax.annotate(
                    "better frontier",
                    xy=(frontier_x[anchor_idx], frontier_y[anchor_idx]),
                    xytext=(-4, 28),
                    textcoords="offset points",
                    fontsize=fs - 5,
                    color="#555555",
                    ha="right",
                    va="bottom",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#777777",
                        "lw": 1.0,
                        "shrinkA": 0,
                        "shrinkB": 4,
                    },
                )

        bottom_ax.set_title(style["tradeoff_title"], color=style["title_color"], fontsize=fs)
        bottom_ax.set_xlabel(_bold_label("Compute Cost (s)"))
        bottom_ax.xaxis.label.set_color("#111111")
        bottom_ax.tick_params(axis="both", colors="#666666")
        bottom_ax.grid(True, zorder=0)
        bottom_ax.set_xscale("log")
        bottom_ax.set_ylim(bottom_bottom, bottom_top)
        bottom_ax.set_yticks(bottom_ticks)
        bottom_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
        if col == 0:
            bottom_ax.set_ylabel(_bold_label("Top-5 Avg Reward"))
            bottom_ax.yaxis.label.set_color("#111111")
        else:
            bottom_ax.tick_params(labelleft=False)

        for ax in (top_ax, bottom_ax):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CCCCCC")
            ax.spines["bottom"].set_color("#CCCCCC")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.10, wspace=0.26, hspace=0.34)
    _save_figure_with_optional_png(fig, output_path, facecolor=SCALING_BACKGROUND)
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
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip("."))
        )

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
            ax.tick_params(axis="both", colors="#666666")
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
        train_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e5:g}"))
        train_ax.xaxis.get_offset_text().set_visible(False)
        train_ax.annotate(
            r"$\times 10^{5}$",
            xy=(1.003, 0.015),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=fs - 5,
            color="#666666",
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
                    fontsize=fs - 1,
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
                    fontsize=fs - 1,
                    columnspacing=1.0,
                    handlelength=1.5,
                    handletextpad=0.4,
                    labelspacing=0.45,
                    handleheight=0.9,
                    borderpad=0.3,
                )
                legend.get_frame().set_facecolor("#FFFFFF")
                for text in legend.get_texts():
                    text.set_color("#111111")

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
                linestyle="-",
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
                color="#666666",
                fontweight="bold",
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
            color="#666666",
            annotation_clip=False,
            zorder=0,
        )
        apply_reward_axis(tradeoff_ax, bottom_ticks, bottom_bottom, bottom_top, bottom_fmt)
        if col == 0:
            tradeoff_ax.set_ylabel(_bold_label("Top 5 Reward"))
            tradeoff_ax.yaxis.label.set_color("#111111")

    fig.subplots_adjust(left=0.078, right=0.985, top=0.920, bottom=0.130, wspace=0.25, hspace=0.44)

    _save_figure_with_optional_png(fig, output_path, facecolor=BACKGROUND)
    plt.close(fig)
    return output_path


def _style_learning_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(axis="both", colors="#666666")
    ax.grid(True, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")


def _set_learning_step_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(np.linspace(0.0, 1_000_000.0, 5))
    format_steps_axis(ax)


def _add_learning_line_axis_buffer(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, LEARNING_AXIS_STEP_LIMIT)
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(y_bottom, y_bottom + (y_top - y_bottom) * 1.08)


def _set_five_y_ticks(ax: plt.Axes) -> None:
    bottom, top = ax.get_ylim()
    span = abs(top - bottom)
    tick_bottom = bottom + 0.05 * span
    tick_top = top - 0.05 * span
    ticks = np.linspace(tick_bottom, tick_top, 5)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.1f}" if span < 12 else f"{int(round(value))}")
    )


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
        ax.plot(
            steps,
            mean_curve,
            color=style["color"],
            linewidth=1.75,
            label=LEARNING_MODE_LABELS.get(mode, style["label"]),
            zorder=3,
        )
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=style["color"],
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )

    ax.set_title(_bold_label("RL Reward Design"), fontsize=fs + 1, color="#111111")
    ax.set_xlabel(_bold_label("Environment Steps"))
    ax.set_ylabel(_bold_label("Reward"))
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    _add_learning_line_axis_buffer(ax)
    _set_learning_step_ticks(ax)
    _set_five_y_ticks(ax)


def _plot_final_training_comparison_panel(ax: plt.Axes, alpha_key: str, *, fs: int) -> None:
    metric_key = "eval/episode_terminal_reward"
    at_style_key = "alphatransit_0.3" if alpha_key == "0_3" else "alphatransit_1.0"
    rl_style_key = "ppo_0.3" if alpha_key == "0_3" else "ppo_1.0"

    at_steps, at_values = _load_mcts_metric(alpha_key, 500, metric_key)
    if len(at_steps):
        mask = at_steps <= LEARNING_DATA_STEP_LIMIT
        _plot_smoothed_line(
            ax,
            at_steps[mask],
            at_values[mask],
            color=TRAINING_STYLES[at_style_key]["color"],
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
            color=TRAINING_STYLES[rl_style_key]["color"],
            label="Reinforcement Learning",
            linestyle="--",
            window=LEARNING_SMOOTHING_WINDOW,
        )

    ax.set_title(_bold_label("Training Progress"), fontsize=fs + 1, color="#111111")
    ax.set_xlabel(_bold_label("Environment Steps"))
    ax.set_ylabel(_bold_label("Reward"))
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    _add_learning_line_axis_buffer(ax)
    _set_learning_step_ticks(ax)
    _set_five_y_ticks(ax)


def _plot_final_wall_clock_panel(ax: plt.Axes, alpha_key: str, *, fs: int) -> None:
    alpha_value = 0.3 if alpha_key == "0_3" else 1.0
    df = _load_wall_clock_scaling()
    df = df[np.isclose(df["alpha"].astype(float), alpha_value)]
    if df.empty:
        raise ValueError(f"No wall-clock rows found for alpha={alpha_value}.")

    n_iters = [100, 200, 300, 400, 500]
    x = np.arange(len(n_iters), dtype=float)
    width = 0.34
    pure_times: list[float] = []
    alpha_times: list[float] = []
    pure_stds: list[float] = []
    alpha_stds: list[float] = []
    for n_iter in n_iters:
        pure_row = df[(df["method"] == "pure_mcts") & (df["n_iter"].astype(int) == n_iter)]
        alpha_row = df[(df["method"] == "alphatransit") & (df["n_iter"].astype(int) == n_iter)]
        pure_times.append(float(pure_row["total_seconds"].iloc[0]) / 60.0 if len(pure_row) else np.nan)
        alpha_times.append(float(alpha_row["total_seconds"].iloc[0]) / 60.0 if len(alpha_row) else np.nan)
        pure_stds.append(float(pure_row["std_seconds"].iloc[0]) / 60.0 if len(pure_row) and "std_seconds" in pure_row else np.nan)
        alpha_stds.append(float(alpha_row["std_seconds"].iloc[0]) / 60.0 if len(alpha_row) and "std_seconds" in alpha_row else np.nan)

    ax.bar(
        x - width / 2,
        pure_times,
        width,
        yerr=pure_stds,
        color=WALL_CLOCK_COLORS["pure_mcts"],
        error_kw={"ecolor": "#111111", "elinewidth": 1.05, "capsize": 4.0, "capthick": 1.05},
        alpha=0.88,
        label="MCTS",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        alpha_times,
        width,
        yerr=alpha_stds,
        color=WALL_CLOCK_COLORS["alphatransit"],
        error_kw={"ecolor": "#111111", "elinewidth": 1.05, "capsize": 4.0, "capthick": 1.05},
        alpha=0.88,
        label="AlphaTransit",
        zorder=3,
    )
    max_time = np.nanmax(
        np.asarray(pure_times + alpha_times, dtype=float)
        + np.nan_to_num(np.asarray(pure_stds + alpha_stds, dtype=float), nan=0.0)
    )
    if np.isfinite(max_time):
        ax.set_ylim(0.0, max_time * 1.18)
    _set_five_y_ticks(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n_iter) for n_iter in n_iters])
    ax.set_title(_bold_label("Search Runtime"), fontsize=fs + 1, color="#111111")
    ax.set_xlabel(_bold_label("MCTS Simulations"))
    ax.set_ylabel(_bold_label("Time / Route (min)"))
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    ax.grid(True, axis="y", zorder=0)


def plot_learning_overview(
    output_path: Path = NEURIPS_RESULTS_DIR / "final_learning_overview_alpha_1_0.pdf",
    *,
    alpha_key: str = "1_0",
) -> Path:
    """
    Build the final RL/AlphaTransit training and wall-clock overview for one alpha.

    Left panel: PPO/RL reward ablation curves grouped by reward mode; curves
    average all available runs for each mode and use the PPO evaluation reward
    histories loaded for the requested alpha.

    Middle panel: wall-clock route-construction timing for MCTS and
    AlphaTransit over n_iter in {100, 200, 300, 400, 500}. Bars summarize a
    CPU-only, one-worker benchmark on Bloomington. Each method constructs five
    fixed-length routes (14 stops each); bars show mean route-construction time
    over the five routes and error bars show route-level standard deviation.

    Right panel: AlphaTransit versus end-to-end RL training. AlphaTransit uses
    the search-depth run with 500 MCTS simulations per decision via
    _load_mcts_metric(alpha_key, 500, "eval/episode_terminal_reward"). The RL
    curve comes from _extract_ppo_mode(alpha_key). Both curves are smoothed over
    training steps, clipped to 1,020,000 steps, and plotted with a small right
    x-axis buffer.
    """
    fs = LEARNING_OVERVIEW_FS
    apply_plot_style(fs, background=BACKGROUND)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))

    for ax in axes:
        _style_learning_axis(ax)

    _plot_final_rl_ablation_panel(axes[0], alpha_key, fs=fs)
    _plot_final_wall_clock_panel(axes[1], alpha_key, fs=fs)
    _plot_final_training_comparison_panel(axes[2], alpha_key, fs=fs)

    legend_locs = ["lower right", "upper left", "lower right"]
    legend_cols = [2, 1, 1]
    for ax, legend_loc, legend_ncol in zip(axes, legend_locs, legend_cols):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles=handles,
                labels=labels,
                loc=legend_loc,
                ncol=legend_ncol,
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
            for text in legend.get_texts():
                text.set_color("#111111")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.16, wspace=0.30)
    _save_figure_with_optional_png(fig, output_path, facecolor=BACKGROUND)
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
    "combined-scaling": plot_combined_scaling_summary,
    "sweep-behavior": plot_sweep_behavior_dashboard,
    "scaling-overview": plot_scaling_behavior_overview,
    "learning-overview": plot_learning_overview,
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

    combined_scaling = subparsers.add_parser(
        "combined-scaling",
        help="Build the three-panel alpha-overlay scaling summary figure.",
    )
    combined_scaling.add_argument("--output", type=Path)

    sweep_behavior = subparsers.add_parser(
        "sweep-behavior",
        help="Build the 2x3 sweep behavior and quality-vs-compute dashboard.",
    )
    sweep_behavior.add_argument("--alpha", choices=["0_3", "1_0"], default="1_0")
    sweep_behavior.add_argument("--output", type=Path)

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

    if args.command == "combined-scaling":
        output = args.output or (NEURIPS_RESULTS_DIR / "final_scaling_summary.pdf")
        plot_combined_scaling_summary(output)
        return

    if args.command == "sweep-behavior":
        output = args.output or (NEURIPS_RESULTS_DIR / f"final_sweep_behavior_alpha_{args.alpha}.pdf")
        plot_sweep_behavior_dashboard(output, alpha_key=args.alpha)
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

    if args.command == "all":
        generate_default(args.output_dir, alpha_key=args.alpha)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

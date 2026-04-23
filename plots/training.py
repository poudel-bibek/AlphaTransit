from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .common import (
        FIGURE_DATA_DIR,
        FIGURES_DIR,
        NEURIPS_RESULTS_DIR,
        NEURIPS_SWEEPS_DIR,
        apply_plot_style,
        format_steps_axis,
        load_json,
        load_results_summary,
        read_wandb_scan_csv,
        rolling_mean_std,
        save_figure,
    )
except ImportError:
    from common import (
        FIGURE_DATA_DIR,
        FIGURES_DIR,
        NEURIPS_RESULTS_DIR,
        NEURIPS_SWEEPS_DIR,
        apply_plot_style,
        format_steps_axis,
        load_json,
        load_results_summary,
        read_wandb_scan_csv,
        rolling_mean_std,
        save_figure,
    )


FS = 20
SMOOTH_WINDOW = 10
RIGHT_SMOOTH_WINDOW = 20

PPO_ABLATION_RUNS = {
    "0_3": NEURIPS_SWEEPS_DIR / "nips_5_ppo_rew_abl_alpha_0_3" / "wandb_data" / "all_runs.json",
    "1_0": NEURIPS_SWEEPS_DIR / "nips_6_ppo_rew_abl_alpha_1_0" / "wandb_data" / "all_runs.json",
}
MCTS_NITER_HISTORY_DIRS = {
    "0_3": NEURIPS_SWEEPS_DIR / "nips_7_mcts_n_iter_alpha_0_3" / "wandb_data" / "wandb_scan_history",
    "1_0": NEURIPS_SWEEPS_DIR / "nips_8_mcts_n_iter_alpha_1_0" / "wandb_data" / "wandb_scan_history",
}
MCTS_EP_HISTORY_DIRS = {
    "0_3": NEURIPS_SWEEPS_DIR / "nips_9_mcts_ep_per_iter_alpha_0_3" / "wandb_data" / "wandb_scan_history",
    "1_0": NEURIPS_SWEEPS_DIR / "nips_10_mcts_ep_per_iter_alpha_1_0" / "wandb_data" / "wandb_scan_history",
}
MCTS_BLOCK_HISTORY_DIRS = {
    "0_3": NEURIPS_SWEEPS_DIR / "nips_11_mcts_model_size_alpha_0_3" / "wandb_data" / "wandb_scan_history",
    "1_0": NEURIPS_SWEEPS_DIR / "nips_12_mcts_model_size_alpha_1_0" / "wandb_data" / "wandb_scan_history",
}
WALL_CLOCK_RESULTS_CSV = FIGURES_DIR / "wall_clock_scaling" / "results.csv"
SWEEP_CONVERGENCE_EXPORTS = {
    "0.3": FIGURE_DATA_DIR / "sweep_alpha0.3_history.csv",
    "1.0": FIGURE_DATA_DIR / "sweep_alpha1.0_history.csv",
}
ALPHA_RESULTS_DIRS = {
    "0_3": NEURIPS_RESULTS_DIR / "0_3",
    "1_0": NEURIPS_RESULTS_DIR / "1_0",
}

MODE_STYLES = {
    "terminal_only": {"color": "#4A90D9", "label": "Terminal Only"},
    "terminal_intermediate_raw_early_stop": {"color": "#E84393", "label": "Raw + Early Stop"},
    "terminal_intermediate_delta_early_stop": {"color": "#9B59B6", "label": "Delta + Early Stop"},
    "terminal_intermediate_delta_no_early_stop": {"color": "#2ECC71", "label": "Delta (No Early Stop)"},
}
MODE_ORDER = [
    "terminal_only",
    "terminal_intermediate_raw_early_stop",
    "terminal_intermediate_delta_early_stop",
    "terminal_intermediate_delta_no_early_stop",
]
TRAINING_STYLES = {
    "alphatransit_0.3": {"color": "#4A90D9", "label": r"AlphaTransit $\alpha{=}0.3$", "ls": "-"},
    "alphatransit_1.0": {"color": "#2ECC71", "label": r"AlphaTransit $\alpha{=}1.0$", "ls": "-"},
    "ppo_0.3": {"color": "#E84393", "label": r"End-to-End RL $\alpha{=}0.3$", "ls": "--"},
    "ppo_1.0": {"color": "#9B59B6", "label": r"End-to-End RL $\alpha{=}1.0$", "ls": "--"},
}
NITER_COLORS = {100: "#A8D8EA", 200: "#4A90D9", 300: "#7B68EE", 400: "#9B59B6", 500: "#E84393"}
WALL_CLOCK_COLORS = {"pure_mcts": "#E84393", "alphatransit": "#4A90D9"}


def _latest_matching_dir(parent: Path, prefix: str) -> Path:
    matches = sorted(
        (path for path in parent.glob(f"{prefix}*") if path.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No directory matching {prefix}* under {parent}")
    return matches[-1]


def _resolve_summary_base(
    alpha_key: str,
    prefix: str,
    *extra_parts: str,
) -> Path:
    return _latest_matching_dir(ALPHA_RESULTS_DIRS[alpha_key], prefix).joinpath(
        *extra_parts
    )

@functools.cache
def get_pareto_summary_paths(alpha_key: str) -> Dict[str, Path]:
    niter_dir = "nips_7_n_iter" if alpha_key == "0_3" else "nips_8_n_iter"
    return {
        "Real-World": _resolve_summary_base(alpha_key, "real_world_"),
        "Random Walk": _resolve_summary_base(alpha_key, "random_walk_"),
        "Demand Cover": _resolve_summary_base(alpha_key, "demand_cover_"),
        "Shortest Path": _resolve_summary_base(alpha_key, "shortest_path_"),
        "Genetic Alg.": _resolve_summary_base(alpha_key, "genetic_", "current_best"),
        "Bee Colony": _resolve_summary_base(alpha_key, "bee_colony_"),
        "Neural Evol.": _resolve_summary_base(alpha_key, "neural_evolutionary_"),
        "Pure MCTS": _resolve_summary_base(alpha_key, "mcts_pure_"),
        "RL": _resolve_summary_base(alpha_key, "end_to_end_rl_", "train_seed_123"),
        "AlphaTransit": NEURIPS_RESULTS_DIR / alpha_key / "alphatransit" / niter_dir / "n_iter_500",
    }
PARETO_STYLES = {
    "AlphaTransit": {"color": "#2ECC71", "marker": "*", "size": 350, "zorder": 10},
    "RL": {"color": "#7B68EE", "marker": "s", "size": 130, "zorder": 5},
    "Pure MCTS": {"color": "#E84393", "marker": "D", "size": 130, "zorder": 5},
    "Neural Evol.": {"color": "#3498DB", "marker": "^", "size": 150, "zorder": 5},
    "Bee Colony": {"color": "#E74C3C", "marker": "v", "size": 150, "zorder": 5},
    "Genetic Alg.": {"color": "#9B59B6", "marker": "p", "size": 150, "zorder": 5},
    "Real-World": {"color": "#555555", "marker": "X", "size": 130, "zorder": 3},
    "Random Walk": {"color": "#95A5A6", "marker": "h", "size": 140, "zorder": 3},
    "Demand Cover": {"color": "#1ABC9C", "marker": "o", "size": 120, "zorder": 3},
    "Shortest Path": {"color": "#F39C12", "marker": "P", "size": 130, "zorder": 3},
}
PARETO_ABBREV = {
    "AlphaTransit": "AT",
    "RL": "RL",
    "Pure MCTS": "MCTS",
    "Neural Evol.": "NE",
    "Bee Colony": "BC",
    "Genetic Alg.": "GA",
    "Real-World": "Real",
    "Random Walk": "RW",
    "Demand Cover": "DC",
    "Shortest Path": "SP",
}


def _find_unique_csv(directory: Path, prefix: str) -> Path:
    matches = sorted(directory.glob(f"{prefix}_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV for {prefix} in {directory}, found {len(matches)}")
    return matches[0]


def _load_history_csv(path: Path, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    df = read_wandb_scan_csv(path)
    df = df.dropna(subset=[metric_key]).copy()
    return df["_step"].to_numpy(dtype=float), df[metric_key].to_numpy(dtype=float)


def _smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    return rolling_mean_std(values, window)


def _resample_and_smooth(steps: np.ndarray, values: np.ndarray, *, window: int = SMOOTH_WINDOW, n_grid: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(steps) < 2:
        return steps, values, np.zeros_like(values)
    grid = np.linspace(steps[0], steps[-1], n_grid)
    resampled = np.interp(grid, steps, values)
    mean, std = rolling_mean_std(resampled, window)
    return grid, mean, std


def _load_ppo_runs(alpha_key: str) -> list[dict]:
    return load_json(PPO_ABLATION_RUNS[alpha_key])


def _extract_ppo_mode(alpha_key: str, mode: str = "terminal_intermediate_delta_no_early_stop") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = _load_ppo_runs(alpha_key)
    entries = []
    for run in runs:
        selected_mode = run.get(
            "ppo_reward_mode",
            run.get("config", {}).get("ppo_reward_mode", "?"),
        )
        if selected_mode == mode:
            entries.append(
                {
                    "history": run["history"],
                    "steps": run.get("steps", list(range(len(run["history"])))),
                }
            )
    if not entries:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(entry["history"]) for entry in entries)
    trimmed = np.array([entry["history"][:min_len] for entry in entries], dtype=float)
    steps = np.array(entries[0]["steps"][:min_len], dtype=float)
    return steps, trimmed.mean(axis=0), trimmed.std(axis=0)


def _load_mcts_metric(alpha_key: str, n_iter: int, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    csv_path = _find_unique_csv(MCTS_NITER_HISTORY_DIRS[alpha_key], f"n_iter_{n_iter}")
    return _load_history_csv(csv_path, metric_key)


def _plot_single_curve(ax: plt.Axes, steps: np.ndarray, values: np.ndarray, style: dict, target_density: int = 120, window: int = SMOOTH_WINDOW) -> None:
    if len(steps) > target_density * 2:
        grid = np.linspace(steps[0], steps[-1], target_density)
        values = np.interp(grid, steps, values)
        steps = grid
    mean, std = _smooth(values, window)
    ax.plot(steps, mean, color=style["color"], linewidth=2.0, label=style["label"], linestyle=style["ls"])
    ax.fill_between(steps, mean - std, mean + std, alpha=0.20, color=style["color"])


def plot_reward_ablation_panel(ax: plt.Axes, alpha_key: str, title: str) -> None:
    runs = _load_ppo_runs(alpha_key)
    mode_data = {}
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
        style = MODE_STYLES[mode]
        min_len = min(len(entry["history"]) for entry in entries)
        trimmed = np.array([entry["history"][:min_len] for entry in entries], dtype=float)
        steps = np.array(entries[0]["steps"][:min_len], dtype=float)
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)
        if len(mean_curve) >= SMOOTH_WINDOW:
            mean_curve, _ = _smooth(mean_curve, SMOOTH_WINDOW)
            std_curve, _ = _smooth(std_curve, SMOOTH_WINDOW)
        ax.plot(steps, mean_curve, label=style["label"], color=style["color"], linewidth=1.8)
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=style["color"])

    ax.set_title(title, fontsize=FS + 1)
    ax.set_xlabel(r"Environment Steps")
    ax.set_ylabel(r"Eval Reward")
    format_steps_axis(ax)
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)


def plot_training_panel(ax: plt.Axes) -> None:
    metric_key = "eval/episode_terminal_reward"
    for alpha_key, style_key in [("0_3", "alphatransit_0.3"), ("1_0", "alphatransit_1.0")]:
        steps, values = _load_mcts_metric(alpha_key, 500, metric_key)
        if len(steps) == 0:
            continue
        mask = steps <= 1_000_000
        _plot_single_curve(ax, steps[mask], values[mask], TRAINING_STYLES[style_key], window=RIGHT_SMOOTH_WINDOW)

    for alpha_key, style_key in [("0_3", "ppo_0.3"), ("1_0", "ppo_1.0")]:
        steps, mean_curve, _ = _extract_ppo_mode(alpha_key)
        if len(steps) == 0:
            continue
        _plot_single_curve(ax, steps, mean_curve, TRAINING_STYLES[style_key], window=RIGHT_SMOOTH_WINDOW)

    ax.set_title(r"AlphaTransit vs End-to-End RL", fontsize=FS + 1)
    ax.set_xlabel(r"Environment Steps")
    ax.set_ylabel(r"Eval Reward")
    format_steps_axis(ax)
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)


def plot_reward_ablation_and_training(output_path: Path = FIGURES_DIR / "ppo_reward_ablation_and_training.pdf") -> None:
    apply_plot_style(FS)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    plot_reward_ablation_panel(axes[0], "0_3", r"End-to-End RL $\alpha=0.3$")
    plot_reward_ablation_panel(axes[1], "1_0", r"End-to-End RL $\alpha=1.0$")
    plot_training_panel(axes[2])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            bbox_to_anchor=(0.35, 0.08),
            fontsize=FS - 1,
        )

    right_center = (axes[2].get_position().x0 + axes[2].get_position().x1) / 2
    legend_handles = [
        mlines.Line2D([], [], color="none", label=r"\textbf{$\alpha{=}0.3$}"),
        mlines.Line2D([], [], color="none", label=r"\textbf{$\alpha{=}1.0$}"),
        mlines.Line2D([], [], color="#4A90D9", ls="-", lw=2.0, label="AlphaTransit"),
        mlines.Line2D([], [], color="#2ECC71", ls="-", lw=2.0, label="AlphaTransit"),
        mlines.Line2D([], [], color="#E84393", ls="--", lw=2.0, label="End-to-End RL"),
        mlines.Line2D([], [], color="#9B59B6", ls="--", lw=2.0, label="End-to-End RL"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(right_center, 0.08),
        fontsize=FS - 1,
        handlelength=1.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(wspace=0.28, bottom=0.20)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_training_curves(output_path: Path = FIGURES_DIR / "training_curves.pdf") -> None:
    apply_plot_style(FS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, alpha_key, alpha_label in zip(axes, ["0_3", "1_0"], ["0.3", "1.0"]):
        mcts_steps, mcts_values = _load_mcts_metric(alpha_key, 500, "eval/episode_terminal_reward")
        ppo_steps, ppo_values, _ = _extract_ppo_mode(alpha_key)
        if len(mcts_steps):
            _plot_single_curve(
                ax,
                mcts_steps,
                mcts_values,
                {
                    "color": "#4A90D9" if alpha_key == "0_3" else "#2ECC71",
                    "label": "AlphaTransit",
                    "ls": "-",
                },
                window=RIGHT_SMOOTH_WINDOW,
            )
        if len(ppo_steps):
            _plot_single_curve(
                ax,
                ppo_steps,
                ppo_values,
                {
                    "color": "#E84393" if alpha_key == "0_3" else "#9B59B6",
                    "label": "End-to-End RL",
                    "ls": "-",
                },
                window=RIGHT_SMOOTH_WINDOW,
            )
        ax.set_title(rf"$\alpha={alpha_label}$")
        ax.set_xlabel(r"Environment Steps")
        ax.set_ylabel(r"Eval Reward")
        format_steps_axis(ax)
        ax.grid(True)
        ax.set_xlim(-20_000, 1_020_000)
        ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCCCCC")
    fig.subplots_adjust(wspace=0.2)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_rl_ablation(output_path: Path = FIGURES_DIR / "rl_reward_ablation.pdf") -> None:
    apply_plot_style(FS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    plot_reward_ablation_panel(axes[0], "0_3", r"$\alpha=0.3$")
    plot_reward_ablation_panel(axes[1], "1_0", r"$\alpha=1.0$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.06), fontsize=FS - 2)
    fig.subplots_adjust(wspace=0.2, bottom=0.18)
    save_figure(fig, output_path)
    plt.close(fig)


def _load_all_mcts_runs(alpha_key: str) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    history_dir = MCTS_NITER_HISTORY_DIRS[alpha_key]
    runs = {}
    for csv_path in sorted(history_dir.glob("n_iter_*.csv")):
        try:
            n_iter = int(csv_path.stem.split("_")[2])
        except (IndexError, ValueError):
            continue
        runs[n_iter] = _load_history_csv(csv_path, "mcts/total_loss")
    return runs


def _plot_mcts_scaling_panel(ax: plt.Axes, alpha_key: str, title: str) -> None:
    for n_iter, (steps, values) in sorted(_load_all_mcts_runs(alpha_key).items()):
        mask = steps <= 1_000_000
        resampled_steps, mean_curve, std_curve = _resample_and_smooth(
            steps[mask],
            values[mask],
            window=20,
        )
        color = NITER_COLORS.get(n_iter, "#333333")
        ax.plot(
            resampled_steps,
            mean_curve,
            color=color,
            linewidth=1.8,
            label=rf"$n_{{\mathrm{{iter}}}}={n_iter}$",
            alpha=0.85,
        )
        ax.fill_between(
            resampled_steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.12,
        )

    ax.axvline(x=700_000, color="#AAAAAA", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(x=900_000, color="#AAAAAA", linestyle=":", linewidth=1.0, alpha=0.7)
    yhi = ax.get_ylim()[1]
    ax.text(340_000, yhi * 0.95, r"$\tau=1.0$", fontsize=FS - 2, color="#999999", va="top", ha="center")
    ax.text(715_000, yhi * 0.95, r"$\tau\!\to\!0.7$", fontsize=FS - 2, color="#999999", va="top")
    ax.text(915_000, yhi * 0.95, r"$\tau\!\to\!0.5$", fontsize=FS - 2, color="#999999", va="top")
    ax.set_title(title, fontsize=FS + 1)
    ax.set_xlabel(r"Environment Steps")
    ax.set_ylabel(r"Total Loss")
    format_steps_axis(ax)
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)


def _load_wall_clock_scaling() -> pd.DataFrame:
    if not WALL_CLOCK_RESULTS_CSV.exists():
        raise FileNotFoundError(
            "Wall-clock scaling data not found: "
            f"{WALL_CLOCK_RESULTS_CSV}\nRun: python plots/main.py experiment"
        )
    return pd.read_csv(WALL_CLOCK_RESULTS_CSV)


def _plot_wall_clock_scaling_panel(ax: plt.Axes, alpha_value: float) -> None:
    df = _load_wall_clock_scaling()
    df = df[df["alpha"] == alpha_value]
    n_iters = sorted(df["n_iter"].unique())
    x = np.arange(len(n_iters))
    width = 0.32
    pure_times = []
    alpha_times = []
    for n_iter in n_iters:
        pure_row = df[(df["method"] == "pure_mcts") & (df["n_iter"] == n_iter)]
        alpha_row = df[(df["method"] == "alphatransit") & (df["n_iter"] == n_iter)]
        pure_times.append(pure_row["total_seconds"].values[0] / 60 if len(pure_row) else 0)
        alpha_times.append(alpha_row["total_seconds"].values[0] / 60 if len(alpha_row) else 0)

    bars1 = ax.bar(
        x - width / 2,
        pure_times,
        width,
        color=WALL_CLOCK_COLORS["pure_mcts"],
        alpha=0.85,
        label="Pure MCTS",
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        alpha_times,
        width,
        color=WALL_CLOCK_COLORS["alphatransit"],
        alpha=0.85,
        label="AlphaTransit",
        zorder=3,
    )
    for bar, value in zip(bars1, pure_times):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=FS - 3,
                color="#333333",
            )
    for bar, value in zip(bars2, alpha_times):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=FS - 3,
                color="#333333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([str(n_iter) for n_iter in n_iters], fontsize=FS - 1)
    ax.set_xlabel(r"$n_{\mathrm{iter}}$")
    ax.set_ylabel(r"Time per Route (minutes)")
    ax.grid(True, axis="y", zorder=0)
    if pure_times:
        ax.set_ylim(0, max(pure_times) * 1.25)


def plot_mcts_scaling_and_wallclock(output_path: Path = FIGURES_DIR / "mcts_scaling_and_wallclock.pdf") -> None:
    apply_plot_style(FS)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    _plot_mcts_scaling_panel(axes[0], "0_3", r"$\alpha=0.3$")
    _plot_mcts_scaling_panel(axes[1], "1_0", r"$\alpha=1.0$")
    _plot_wall_clock_scaling_panel(axes[2], 0.3)

    seen = {}
    for ax in axes[:2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen[label] = handle
    if seen:
        sorted_items = sorted(seen.items(), key=lambda item: item[0])
        left_mid_center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        fig.legend(
            [handle for _, handle in sorted_items],
            [label for label, _ in sorted_items],
            loc="upper center",
            ncol=5,
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            bbox_to_anchor=(left_mid_center, 0.08),
            fontsize=FS - 2,
        )
    handles_wc, labels_wc = axes[2].get_legend_handles_labels()
    if handles_wc:
        right_center = (axes[2].get_position().x0 + axes[2].get_position().x1) / 2
        fig.legend(
            handles_wc,
            labels_wc,
            loc="upper center",
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            bbox_to_anchor=(right_center, 0.08),
            fontsize=FS - 2,
        )
    fig.subplots_adjust(wspace=0.28, bottom=0.20)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_wall_clock_alpha(output_path: Path = FIGURES_DIR / "wall_clock_alpha1_0.pdf", alpha_value: float = 1.0) -> None:
    apply_plot_style(FS)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    _plot_wall_clock_scaling_panel(ax, alpha_value)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=FS - 2)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def _resolve_summary_path(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "eval_results_summary.json"
    if direct.is_file():
        return direct
    matches = sorted(
        path.glob("*/eval_results_summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No eval_results_summary.json found under {path}")
    return matches[-1]


def _pareto_point(summary_path: Path) -> tuple[float, float, float, float]:
    results = load_results_summary(summary_path)
    return (
        float(results["service_rate"]["avg"] * 100.0),
        float(results["service_rate"]["std"] * 100.0),
        float(results["fleet_size"]["avg"]),
        float(results["fleet_size"]["std"]),
    )


def _load_pareto_data(alpha_key: str) -> Dict[str, tuple[float, float, float, float]]:
    return {
        method: _pareto_point(_resolve_summary_path(path))
        for method, path in get_pareto_summary_paths(alpha_key).items()
    }


def plot_pareto(output_path: Path = FIGURES_DIR / "pareto_analysis.pdf") -> None:
    apply_plot_style(FS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, alpha_key, title in zip(axes, ["0_3", "1_0"], [r"Low Demand", r"High Demand"]):
        points = _load_pareto_data(alpha_key)
        ax.set_title(title, fontsize=FS + 1)
        for method, (sr_avg, sr_std, fs_avg, fs_std) in points.items():
            style = PARETO_STYLES[method]
            label = PARETO_ABBREV.get(method, method)
            ax.errorbar(
                fs_avg,
                sr_avg,
                xerr=fs_std,
                yerr=sr_std,
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=3,
                alpha=0.55,
                zorder=style["zorder"] - 1,
            )
            ax.scatter(
                fs_avg,
                sr_avg,
                color=style["color"],
                marker=style["marker"],
                s=style["size"],
                zorder=style["zorder"],
                edgecolors="white",
                linewidth=0.5,
            )
            ax.annotate(
                r"\textbf{" + label + "}" if method == "AlphaTransit" else label,
                (fs_avg, sr_avg),
                xytext=(0, -12 if method != "Random Walk" else 12),
                textcoords="offset points",
                fontsize=FS - 4,
                color=style["color"],
                ha="center",
                va="top" if method != "Random Walk" else "bottom",
            )
        ax.set_xlabel(r"Fleet Size")
        ax.set_ylabel(r"Service Rate (\%)")
        ax.grid(True, zorder=0)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - (xlim[1] - xlim[0]) * 0.08, xlim[1] + (xlim[1] - xlim[0]) * 0.08)
        ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.05, ylim[1] + (ylim[1] - ylim[0]) * 0.05)
    fig.subplots_adjust(wspace=0.25)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_sweep_convergence(output_path: Path = FIGURES_DIR / "sweep_convergence.pdf") -> None:
    apply_plot_style(12)

    def load_sweep(alpha_label: str) -> pd.DataFrame:
        path = SWEEP_CONVERGENCE_EXPORTS[alpha_label]
        if not path.exists():
            raise FileNotFoundError(f"Sweep convergence export missing: {path}")
        hist = pd.read_csv(path)
        hist = hist.dropna(subset=["eval/episode_terminal_reward", "_step"]).copy()
        hist["_step"] = hist["_step"].astype(int)
        return hist

    def plot_panel(ax: plt.Axes, hist: pd.DataFrame, alpha_label: str) -> None:
        runs = hist.groupby("run_name")
        best_run_name = hist.groupby("run_name")["eval/episode_terminal_reward"].max().idxmax()
        for name, group in runs:
            if name == best_run_name:
                continue
            group = group.sort_values("_step")
            mean_curve, _ = _smooth(group["eval/episode_terminal_reward"].to_numpy(dtype=float), 10)
            ax.plot(group["_step"], mean_curve, color="#4A90D9", alpha=0.35, linewidth=1.0, zorder=1)
        best = runs.get_group(best_run_name).sort_values("_step")
        best_mean, _ = _smooth(best["eval/episode_terminal_reward"].to_numpy(dtype=float), 10)
        ax.plot(best["_step"], best_mean, color="#2ECC71", linewidth=2.0, zorder=3, label="Selected configuration")
        ax.set_title(
            rf"$\alpha={alpha_label}$ ({hist['run_name'].nunique()} configurations)",
            fontsize=13,
        )
        ax.set_xlabel(r"Training Step")
        ax.set_ylabel(r"Terminal Reward")
        format_steps_axis(ax)
        ax.set_xlim(-20_000, 1_020_000)
        ax.legend(loc="lower right", framealpha=0.9, edgecolor="#CCCCCC", fancybox=False)
        ax.grid(True, zorder=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_panel(axes[0], load_sweep("0.3"), "0.3")
    plot_panel(axes[1], load_sweep("1.0"), "1.0")
    fig.subplots_adjust(wspace=0.2)
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_family(
    ax: plt.Axes,
    directory: Path,
    prefix: str,
    metric_key: str,
    color_lookup: Dict[int, str],
    label_fmt: str,
) -> None:
    curves = []
    for csv_path in sorted(directory.glob(f"{prefix}_*.csv")):
        try:
            value = int(csv_path.stem.split("_")[2])
        except (IndexError, ValueError):
            continue
        steps, values = _load_history_csv(csv_path, metric_key)
        if len(steps) == 0:
            continue
        mask = steps <= 1_000_000
        curves.append((value, steps[mask], values[mask]))

    for value, steps, values in sorted(curves):
        mean_curve, _ = _smooth(values, 15)
        color = color_lookup.get(value, "#333333")
        ax.plot(steps, mean_curve, color=color, linewidth=1.2, alpha=0.85, label=label_fmt.format(value))

    ax.set_xlabel(r"Environment Steps")
    ax.set_ylabel(r"Total Loss")
    format_steps_axis(ax)
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)


def plot_scaling_laws(output_path: Path = FIGURES_DIR / "scaling_laws.pdf", alpha_key: str = "0_3") -> None:
    apply_plot_style(FS)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    _plot_family(
        axes[0],
        MCTS_NITER_HISTORY_DIRS[alpha_key],
        "n_iter",
        "mcts/total_loss",
        NITER_COLORS,
        r"$n_{{\mathrm{{iter}}}}={}$",
    )
    _plot_family(
        axes[1],
        MCTS_EP_HISTORY_DIRS[alpha_key],
        "ep_per_iter",
        "mcts/total_loss",
        {8: "#A8D8EA", 16: "#4A90D9", 24: "#7B68EE", 32: "#E84393"},
        r"$\mathrm{{eps}}={}$",
    )
    _plot_family(
        axes[2],
        MCTS_BLOCK_HISTORY_DIRS[alpha_key],
        "num_gat_blocks",
        "mcts/total_loss",
        {2: "#A8D8EA", 4: "#4A90D9", 8: "#7B68EE", 16: "#E84393"},
        r"$\mathrm{{blocks}}={}$",
    )
    axes[0].set_title(r"MCTS search scaling", fontsize=FS + 1)
    axes[1].set_title(r"Data scaling", fontsize=FS + 1)
    axes[2].set_title(r"Model scaling", fontsize=FS + 1)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=FS - 3)
    fig.subplots_adjust(wspace=0.25)
    save_figure(fig, output_path)
    plt.close(fig)


def generate_default(output_dir: Path = FIGURES_DIR) -> None:
    plot_reward_ablation_and_training(output_dir / "ppo_reward_ablation_and_training.pdf")
    plot_mcts_scaling_and_wallclock(output_dir / "mcts_scaling_and_wallclock.pdf")
    plot_wall_clock_alpha(output_dir / "wall_clock_alpha1_0.pdf", 1.0)
    plot_pareto(output_dir / "pareto_analysis.pdf")

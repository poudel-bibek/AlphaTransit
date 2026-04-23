from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
FIGURES_DIR = ROOT / "figures"
FIGURE_DATA_DIR = ROOT / "figure_data"
ASSETS_DIR = PLOTS_DIR / "assets"
NETWORKS_DIR = ROOT / "networks"
TRAINING_DATA_DIR = ROOT / "training_data"
NEURIPS_RESULTS_DIR = TRAINING_DATA_DIR / "NeurIPS_results"
NEURIPS_SWEEPS_DIR = NEURIPS_RESULTS_DIR / "NeurIPS_sweeps"

ROUTE_COLORS = [
    "#e6194B",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#000075",
]


def apply_plot_style(font_size: int = 20, use_tex: bool = True, background: str = "#FFFFFF") -> None:
    """Apply the shared plot style used across active plotting modules."""
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": font_size,
            "axes.labelsize": font_size + 2,
            "axes.titlesize": font_size + 2,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 2,
            "figure.facecolor": background,
            "axes.facecolor": background,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.5,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.4,
            "grid.linestyle": ":",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_figure(fig: plt.Figure, output_path: Path, *, facecolor: str | None = None, pad_inches: float = 0.15) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        facecolor=facecolor or fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    print(f"Saved {output_path}")


def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(values)
    mean = series.rolling(window=window, min_periods=1).mean().to_numpy()
    std = series.rolling(window=window, min_periods=2).std().fillna(0).to_numpy()
    return mean, std


def format_steps_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: rf"{v/1e6:.1f}M" if v >= 1e6 else rf"{v/1e3:.0f}K")
    )


def read_wandb_scan_csv(path: Path, step_col: str = "_step") -> pd.DataFrame:
    """Load a W&B export CSV, keeping rows sorted by step and dropping duplicate steps."""
    df = pd.read_csv(path)
    if step_col not in df.columns:
        raise KeyError(f"{path} is missing required step column {step_col!r}")
    df = df.dropna(subset=[step_col]).copy()
    df[step_col] = df[step_col].astype(float)
    df = df.sort_values(step_col)
    return df.drop_duplicates(subset=step_col, keep="first")


def maybe_to_web_mercator(coords: Dict[str, tuple[float, float]]) -> Dict[str, tuple[float, float]]:
    """Convert UTM Bloomington coordinates to Web Mercator when pyproj is available."""
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:32616", "EPSG:3857", always_xy=True)
        return {
            node_id: transformer.transform(x, y)
            for node_id, (x, y) in coords.items()
        }
    except ImportError:
        return coords


def load_results_summary(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    if "results" not in payload:
        raise KeyError(f"{path} does not match the expected eval_results_summary.json shape")
    return payload["results"]

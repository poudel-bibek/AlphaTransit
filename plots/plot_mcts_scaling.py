"""
Plot MCTS n_iter scaling results.

Usage:
    python plots/plot_mcts_scaling.py

Outputs (in training_data/NeurIPS_results/{alpha}/nips_{sweep}_mcts_n_iter_alpha_{alpha}/):
    - n_iter_scaling_training_curves.png   — eval reward + total loss side by side
    - n_iter_scaling_best_reward.png       — line+dot chart of peak rewards per n_iter
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Style setup (from plots.md)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': '#F5F5F2',
    'axes.facecolor': '#F5F5F2',
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

# Dual-hue inspired palette for n_iter values (ice-blue -> magenta)
COLORS = {
    100: '#4A90D9',
    200: '#7B68EE',
    300: '#9B59B6',
    400: '#DA70D6',
    500: '#E84393',
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
N_ITERS = [100, 200, 300, 400, 500]

SWEEPS = {
    '0_3': ROOT / 'training_data' / 'NeurIPS_results' / '0_3' / 'nips_7_mcts_n_iter_alpha_0_3',
    '1_0': ROOT / 'training_data' / 'NeurIPS_results' / '1_0' / 'nips_8_mcts_n_iter_alpha_1_0',
}

SAVE_PDF = False  # Set to True to also save PDF versions


def _csv_dir(sweep_dir: Path) -> Path:
    return sweep_dir / 'wandb_data' / 'wandb_scan_history'


def load_csv(sweep_dir: Path, n_iter: int) -> pd.DataFrame:
    csv_dir = _csv_dir(sweep_dir)
    candidates = [f for f in os.listdir(csv_dir) if f.startswith(f'n_iter_{n_iter}_') and f.endswith('.csv')]
    if not candidates:
        raise FileNotFoundError(f'No CSV for n_iter={n_iter} in {csv_dir}')
    return pd.read_csv(csv_dir / candidates[0])


def has_data(sweep_dir: Path) -> bool:
    csv_dir = _csv_dir(sweep_dir)
    return csv_dir.exists() and any(f.endswith('.csv') for f in os.listdir(csv_dir))


def _plot_curves_on_ax(ax, sweep_dir, metric_key, smooth_window=5):
    for n_iter in N_ITERS:
        try:
            df = load_csv(sweep_dir, n_iter)
        except FileNotFoundError:
            continue
        col_df = df[df[metric_key].notna()].copy().sort_values('_step')
        steps = col_df['_step'].values
        values = col_df[metric_key].values

        if smooth_window > 1 and len(values) >= smooth_window:
            series = pd.Series(values)
            smoothed = series.rolling(window=smooth_window, min_periods=1).mean().values
            std = series.rolling(window=smooth_window, min_periods=1).std().fillna(0).values
            ax.fill_between(steps, smoothed - 0.5 * std, smoothed + 0.5 * std,
                            color=COLORS[n_iter], alpha=0.15)
            ax.plot(steps, smoothed, color=COLORS[n_iter], linewidth=1.8,
                    label=rf'$n = {n_iter}$')
        else:
            ax.plot(steps, values, color=COLORS[n_iter], linewidth=1.8,
                    label=rf'$n = {n_iter}$')


def _style_ax(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: rf'{x/1e6:.1f}M' if x >= 1e6 else rf'{x/1e3:.0f}K'))
    ax.grid(True)


def _save(fig, out_path: Path):
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    if SAVE_PDF:
        fig.savefig(out_path.with_suffix('.pdf'))
        print(f'Saved {out_path.with_suffix(".pdf")}')


def plot_training_curves(alpha: str, sweep_dir: Path, smooth_window: int = 5):
    if not has_data(sweep_dir):
        print(f'No data for alpha={alpha}, skipping training curves.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _plot_curves_on_ax(axes[0], sweep_dir, 'eval/episode_terminal_reward', smooth_window)
    axes[0].set_xlabel(r'Environment Steps')
    axes[0].set_ylabel(r'Eval Terminal Reward')
    _style_ax(axes[0])

    _plot_curves_on_ax(axes[1], sweep_dir, 'mcts/total_loss', smooth_window)
    axes[1].set_xlabel(r'Environment Steps')
    axes[1].set_ylabel(r'Total Loss')
    _style_ax(axes[1])

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
              frameon=True, fancybox=False, edgecolor='#CCCCCC',
              bbox_to_anchor=(0.5, -0.04))

    fig.subplots_adjust(wspace=0.25, bottom=0.15)

    out = ROOT / 'training_data' / 'NeurIPS_results' / f'n_iter_scaling_training_curves_alpha_{alpha}.png'
    _save(fig, out)
    plt.close(fig)


def plot_best_reward(alpha: str, sweep_dir: Path):
    if not has_data(sweep_dir):
        print(f'No data for alpha={alpha}, skipping best reward.')
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    peaks = []
    available = []
    for n_iter in N_ITERS:
        try:
            df = load_csv(sweep_dir, n_iter)
            peak = df['eval/episode_terminal_reward'].max()
            peaks.append(peak)
            available.append(n_iter)
        except FileNotFoundError:
            continue

    ax.plot(available, peaks, color='#555555', linewidth=1.2, alpha=0.6, zorder=5)
    for n_iter, peak in zip(available, peaks):
        ax.scatter(n_iter, peak, color=COLORS[n_iter], s=80, zorder=10,
                   edgecolors='white', linewidth=0.8)
        ax.text(n_iter, peak + 0.5, rf'{peak:.1f}', ha='center', va='bottom',
                fontsize=10, color='#333333')

    ax.set_xlabel(r'MCTS Simulations per Move ($n\_iter$)')
    ax.set_ylabel(r'Peak Eval Terminal Reward')
    ax.set_xticks(N_ITERS)
    ax.set_xticklabels([rf'${n}$' for n in N_ITERS])
    ax.set_ylim(min(peaks) - 3, max(peaks) + 4)
    ax.grid(True)

    out = ROOT / 'training_data' / 'NeurIPS_results' / f'n_iter_scaling_best_reward_alpha_{alpha}.png'
    _save(fig, out)
    plt.close(fig)


if __name__ == '__main__':
    for alpha, sweep_dir in SWEEPS.items():
        plot_training_curves(alpha, sweep_dir)
        plot_best_reward(alpha, sweep_dir)

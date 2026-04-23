"""
AlphaTransit scaling behavior (alpha=0.3): three axes of scaling.

  LEFT:   MCTS search scaling (n_iter = 100..500) — real data from sweep 7
  MIDDLE: Data scaling (episodes_per_iter = 8..32) — dummy data (placeholder)
  RIGHT:  Model scaling (GAT blocks = 2..16) — dummy data (placeholder)

All panels plot mcts/total_loss (lower is better).

Usage:
    python plots/figure_scaling_laws.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style (from plots.md, white background per existing figures)
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

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(OUTDIR, 'data')

SMOOTH_WINDOW = 15
LINE_WIDTH = 1.2
LINE_ALPHA = 0.85

# ---------------------------------------------------------------------------
# Color palettes — consistent with figure_training_dynamics.py
# ---------------------------------------------------------------------------
NITER_COLORS = {
    100: '#A8D8EA',
    200: '#4A90D9',
    300: '#7B68EE',
    400: '#9B59B6',
    500: '#E84393',
}

EP_COLORS = {
    8:  '#A8D8EA',
    16: '#4A90D9',
    24: '#7B68EE',
    32: '#E84393',
}

MODEL_COLORS = {
    2:  '#A8D8EA',
    4:  '#4A90D9',
    8:  '#7B68EE',
    16: '#E84393',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smooth(vals, window=SMOOTH_WINDOW):
    s = pd.Series(vals)
    mean = s.rolling(window=window, min_periods=1).mean().values
    std = s.rolling(window=window, min_periods=2).std().fillna(0).values
    return mean, std


# ---------------------------------------------------------------------------
# LEFT: MCTS search scaling (n_iter) — real data
# ---------------------------------------------------------------------------
def plot_search_scaling(ax):
    """Plot total loss vs env steps for n_iter = 100..500."""
    metric = 'mcts/total_loss'

    for n_iter, color in sorted(NITER_COLORS.items()):
        matches = [f for f in os.listdir(DATADIR)
                   if f.startswith(f'n_iter_{n_iter}_') and f.endswith('.csv')]
        if not matches:
            continue

        df = pd.read_csv(os.path.join(DATADIR, matches[0]))
        df = df.dropna(subset=[metric, '_step']).sort_values('_step')
        df = df.drop_duplicates(subset='_step', keep='first')
        steps = df['_step'].values
        vals = df[metric].values

        mask = steps <= 1_000_000
        steps, vals = steps[mask], vals[mask]

        ax.plot(steps, vals, color=color, linewidth=LINE_WIDTH,
                label=rf'$n_{{\mathrm{{iter}}}}={n_iter}$', alpha=LINE_ALPHA)

    # Temperature schedule transitions
    ax.axvline(x=700_000, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axvline(x=900_000, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7)
    yhi = ax.get_ylim()[1]
    ax.text(340_000, yhi * 0.97, r'$\tau=1.0$',
            fontsize=FS - 2, color='#999999', va='top', ha='center')
    ax.text(715_000, yhi * 0.97, r'$\tau\!\to\!0.7$',
            fontsize=FS - 2, color='#999999', va='top')
    ax.text(915_000, yhi * 0.97, r'$\tau\!\to\!0.5$',
            fontsize=FS - 2, color='#999999', va='top')

    ax.set_title(r'Search Depth', fontsize=FS + 1)
    ax.set_xlabel(r'Environment Steps')
    ax.set_ylabel(r'Total Loss')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.set_xlim(-20_000, 1_020_000)
    ax.grid(True, zorder=0)


# ---------------------------------------------------------------------------
# MIDDLE: ep_per_iter scaling — dummy data
# ---------------------------------------------------------------------------
def plot_data_scaling(ax):
    """Plot total loss vs policy iterations for eps = 8..32."""
    metric = 'mcts/total_loss'

    for eps, color in sorted(EP_COLORS.items()):
        path = os.path.join(DATADIR, f'ep_per_iter_{eps}.csv')
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        iters = df['iteration'].values
        vals = df[metric].values

        mean, std_vals = smooth(vals)
        ax.plot(iters, mean, color=color, linewidth=LINE_WIDTH,
                label=rf'$E={eps}$', alpha=LINE_ALPHA)
        ax.fill_between(iters, mean - std_vals, mean + std_vals,
                        color=color, alpha=0.12)

    ax.set_title(r'Data', fontsize=FS + 1)
    ax.set_xlabel(r'Policy Iterations')
    ax.set_ylabel(r'Total Loss')
    ax.grid(True, zorder=0)


# ---------------------------------------------------------------------------
# RIGHT: model size scaling — dummy data
# ---------------------------------------------------------------------------
def plot_model_scaling(ax):
    """Plot eval reward vs policy iterations for GAT blocks = 2..16."""
    metric = 'eval/episode_terminal_reward'

    for nblocks, color in sorted(MODEL_COLORS.items()):
        path = os.path.join(DATADIR, f'model_size_{nblocks}.csv')
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        iters = df['iteration'].values
        vals = df[metric].values

        mean, std_vals = smooth(vals)
        ax.plot(iters, mean, color=color, linewidth=LINE_WIDTH,
                label=rf'$B={nblocks}$', alpha=LINE_ALPHA)
        ax.fill_between(iters, mean - std_vals, mean + std_vals,
                        color=color, alpha=0.12)

    ax.set_title(r'Model Size', fontsize=FS + 1)
    ax.set_xlabel(r'Policy Iterations')
    ax.set_ylabel(r'Eval Reward')
    ax.grid(True, zorder=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig, (ax_l, ax_m, ax_r) = plt.subplots(1, 3, figsize=(20, 5.5))

    plot_search_scaling(ax_l)
    plot_data_scaling(ax_m)
    plot_model_scaling(ax_r)

    # Bottom legend for each panel, centered under its axes
    fig.subplots_adjust(wspace=0.28, bottom=0.28)

    for ax, ncol in [(ax_l, 2), (ax_m, 2), (ax_r, 2)]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper center', ncol=ncol,
                      bbox_to_anchor=(0.5, -0.18), frameon=True, fancybox=False,
                      edgecolor='#CCCCCC', fontsize=FS - 3)

    outpath = os.path.join(OUTDIR, 'scaling_laws')
    fig.savefig(outpath + '.pdf')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()

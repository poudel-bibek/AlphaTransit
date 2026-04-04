"""
Bayesian sweep visualization: reward convergence across all configurations.
Two panels: alpha=0.3 (LEFT), alpha=1.0 (RIGHT).
Shows systematic exploration, not cherry-picking.

Data: pulled from wandb sweeps uu2tfzr9 (alpha=0.3) and k8jqkdyw (alpha=1.0).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------------
# Style from plots.md
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

# Colors consistent with existing plots
BEST_COLOR  = '#2ECC71'   # AlphaTransit green
OTHER_COLOR = '#4A90D9'   # blue family
OTHER_ALPHA = 0.35
BEST_LW     = 2.0
OTHER_LW    = 1.0

SMOOTH_WINDOW = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sweep(alpha_label):
    hist = pd.read_csv(os.path.join(OUTDIR, f'sweep_alpha{alpha_label}_history.csv'))
    hist = hist.dropna(subset=['eval/episode_terminal_reward', '_step'])
    hist['_step'] = hist['_step'].astype(int)
    return hist


def smooth(vals, window=SMOOTH_WINDOW):
    return pd.Series(vals).rolling(window=window, min_periods=1).mean().values


def plot_panel(ax, hist, alpha_label):
    runs = hist.groupby('run_name')
    n_runs = hist['run_name'].nunique()

    # Find the best run by max terminal reward achieved
    best_run_name = (
        hist.groupby('run_name')['eval/episode_terminal_reward']
        .max()
        .idxmax()
    )

    # Plot all non-best runs first
    for name, grp in runs:
        if name == best_run_name:
            continue
        grp = grp.sort_values('_step')
        ax.plot(
            grp['_step'],
            smooth(grp['eval/episode_terminal_reward']),
            color=OTHER_COLOR,
            alpha=OTHER_ALPHA,
            linewidth=OTHER_LW,
            zorder=1,
        )

    # Plot best run on top
    best = runs.get_group(best_run_name).sort_values('_step')
    ax.plot(
        best['_step'],
        smooth(best['eval/episode_terminal_reward']),
        color=BEST_COLOR,
        linewidth=BEST_LW,
        zorder=3,
        label='Selected configuration',
    )

    ax.set_title(rf'$\alpha={alpha_label}$ ({n_runs} configurations)', fontsize=13)
    ax.set_xlabel(r'Training Step')
    ax.set_ylabel(r'Terminal Reward')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.set_xlim(-20_000, 1_020_000)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#CCCCCC',
              fancybox=False)
    ax.grid(True, zorder=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

hist03 = load_sweep('0.3')
hist10 = load_sweep('1.0')

plot_panel(ax_left,  hist03, '0.3')
plot_panel(ax_right, hist10, '1.0')

fig.subplots_adjust(wspace=0.2)

outpath = os.path.join(OUTDIR, 'sweep_convergence')
fig.savefig(outpath + '.pdf')
print(f'Saved {outpath}.pdf')
plt.close(fig)

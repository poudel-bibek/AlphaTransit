"""
AlphaTransit vs Reinforcement Learning training curves.

  LEFT:  alpha = 0.3
  RIGHT: alpha = 1.0

Self-contained in plots/data/training_curves/.

Usage:
    python plots/figure_training_curves.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Style — matches figure_training_dynamics.py exactly
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

OUTDIR = Path(__file__).resolve().parent
DATADIR = OUTDIR / 'data' / 'training_curves'

SMOOTH_WINDOW = 6
LINE_WIDTH = 2.0
FILL_ALPHA = 0.20

STYLES = {
    'alphatransit': {'color': '#4A90D9', 'label': 'AlphaTransit', 'ls': '-'},
    'rl':           {'color': '#E84393', 'label': 'End-to-End RL', 'ls': '-'},
}


def smooth(values, window=SMOOTH_WINDOW):
    s = pd.Series(values)
    mean = s.rolling(window=window, min_periods=1).mean().values
    std = s.rolling(window=window, min_periods=2).std().fillna(0).values
    return mean, std


def load_mcts_reward(alpha_tag):
    """Load AlphaTransit (MCTS n_iter=500) eval reward curve."""
    path = DATADIR / f'mcts_niter500_alpha_{alpha_tag}.json'
    with open(path) as f:
        rows = json.load(f)

    metric = 'eval/episode_terminal_reward'
    steps, vals = [], []
    for row in rows:
        s, v = row.get('_step'), row.get(metric)
        if s is not None and v is not None:
            steps.append(s)
            vals.append(v)
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    vals = np.array(vals)[order]

    _, idx = np.unique(steps, return_index=True)
    steps, vals = steps[idx], vals[idx]

    mask = steps <= 1_000_000
    return steps[mask], vals[mask]


def load_ppo_reward(alpha_tag):
    """Load PPO (delta_no_early_stop) reward curve, averaged across seeds."""
    path = DATADIR / f'ppo_abl_alpha_{alpha_tag}.json'
    with open(path) as f:
        runs = json.load(f)

    mode = 'terminal_intermediate_delta_no_early_stop'
    entries = []
    for run in runs:
        m = run.get('ppo_reward_mode', run.get('config', {}).get('ppo_reward_mode', '?'))
        if m == mode:
            entries.append({
                'history': run['history'],
                'steps': run.get('steps', list(range(len(run['history'])))),
            })
    if not entries:
        return np.array([]), np.array([])

    min_len = min(len(e['history']) for e in entries)
    trimmed = np.array([e['history'][:min_len] for e in entries])
    x = np.array(entries[0]['steps'][:min_len])
    mean_curve = trimmed.mean(axis=0)
    return x, mean_curve


def plot_curve(ax, steps, vals, style, target_density=120):
    """Resample dense data, smooth, plot with shaded std band."""
    if len(steps) > target_density * 2:
        grid = np.linspace(steps[0], steps[-1], target_density)
        vals = np.interp(grid, steps, vals)
        steps = grid
    mean, std = smooth(vals)
    ax.plot(steps, mean, color=style['color'], linewidth=LINE_WIDTH,
            label=style['label'], linestyle=style['ls'])
    ax.fill_between(steps, mean - std, mean + std,
                    alpha=FILL_ALPHA, color=style['color'])


def plot_panel(ax, alpha_tag):
    """Plot AlphaTransit vs RL for one alpha value."""
    steps_at, vals_at = load_mcts_reward(alpha_tag)
    if len(steps_at) > 0:
        plot_curve(ax, steps_at, vals_at, STYLES['alphatransit'])

    steps_rl, vals_rl = load_ppo_reward(alpha_tag)
    if len(steps_rl) > 0:
        plot_curve(ax, steps_rl, vals_rl, STYLES['rl'])

    ax.set_xlabel(r'Environment Steps')
    ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
    ax.set_xticklabels([r'0', r'250K', r'500K', r'750K', r'1.0M'])
    ax.grid(True, zorder=0)
    ax.set_xlim(-20_000, 1_020_000)


def main():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    plot_panel(ax_l, '0_3')
    plot_panel(ax_r, '1_0')

    ax_l.set_ylabel(r'Reward')
    ax_l.set_title(r'Low Demand', fontsize=FS + 1)
    ax_r.set_title(r'High Demand', fontsize=FS + 1)

    # Shared bottom legend
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               frameon=True, fancybox=False, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, 0.06), fontsize=FS - 1)

    fig.subplots_adjust(wspace=0.22, bottom=0.22)

    outpath = OUTDIR / '4'
    fig.savefig(str(outpath) + '.pdf', facecolor='#FFFFFF')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()

"""
Reinforcement Learning reward mode ablation.

  LEFT:  alpha = 0.3
  RIGHT: alpha = 1.0

Four reward modes compared: Terminal Only, Raw + Early Stop,
Delta + Early Stop, Delta (No Early Stop).

Self-contained in plots/data/training_curves/.

Usage:
    python plots/figure_rl_ablation.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Style — matches figure_training_dynamics.py and plots.md
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

SMOOTH_WINDOW = 10
LINE_WIDTH = 1.8
FILL_ALPHA = 0.15

MODE_STYLES = {
    'terminal_only': {
        'color': '#4A90D9',
        'label': 'Terminal Only',
    },
    'terminal_intermediate_raw_early_stop': {
        'color': '#E84393',
        'label': 'Raw + Early Stop',
    },
    'terminal_intermediate_delta_early_stop': {
        'color': '#9B59B6',
        'label': 'Delta + Early Stop',
    },
    'terminal_intermediate_delta_no_early_stop': {
        'color': '#2ECC71',
        'label': 'Delta (No Early Stop)',
    },
}

MODE_ORDER = [
    'terminal_only',
    'terminal_intermediate_raw_early_stop',
    'terminal_intermediate_delta_early_stop',
    'terminal_intermediate_delta_no_early_stop',
]


def smooth(values, window=SMOOTH_WINDOW):
    s = pd.Series(values)
    mean = s.rolling(window=window, min_periods=1).mean().values
    std = s.rolling(window=window, min_periods=1).std().fillna(0).values
    return mean, std


def plot_panel(ax, alpha_tag, title, modes_filter=None):
    """Plot reward ablation curves for one alpha value.

    modes_filter: optional iterable of mode keys to include. If None, all
    modes in MODE_ORDER are plotted.
    """
    path = DATADIR / f'ppo_abl_alpha_{alpha_tag}.json'
    with open(path) as f:
        runs = json.load(f)

    mode_data = {}
    for run in runs:
        mode = run.get('ppo_reward_mode',
                       run.get('config', {}).get('ppo_reward_mode', '?'))
        mode_data.setdefault(mode, []).append({
            'history': run['history'],
            'steps': run.get('steps', list(range(len(run['history'])))),
        })

    for mode in MODE_ORDER:
        if modes_filter is not None and mode not in modes_filter:
            continue
        entries = mode_data.get(mode, [])
        if not entries:
            continue

        style = MODE_STYLES[mode]
        min_len = min(len(e['history']) for e in entries)
        trimmed = np.array([e['history'][:min_len] for e in entries])
        x = np.array(entries[0]['steps'][:min_len])
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)

        if len(mean_curve) >= SMOOTH_WINDOW:
            mean_curve, _ = smooth(mean_curve)
            std_curve, _ = smooth(std_curve)

        ax.plot(x, mean_curve, label=style['label'],
                color=style['color'], linewidth=LINE_WIDTH)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=FILL_ALPHA, color=style['color'])

    ax.set_title(title, fontsize=FS + 1)
    ax.set_xlabel(r'Environment Steps')
    ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
    ax.set_xticklabels([r'0', r'250K', r'500K', r'750K', r'1.0M'])
    ax.grid(True, zorder=0)
    ax.set_xlim(-20_000, 1_020_000)


def _save_two_panel(outpath, modes_filter=None, ylims=None):
    """Render a 2-panel (Low / High Transit Demand) figure.

    modes_filter: optional list of reward modes to include (default: all).
    ylims: optional dict {'0_3': (ymin, ymax), '1_0': (ymin, ymax)} to force
           identical axis limits across variants.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_panel(ax_l, '0_3', r'Low Demand', modes_filter=modes_filter)
    plot_panel(ax_r, '1_0', r'High Demand', modes_filter=modes_filter)

    ax_l.set_ylabel(r'Reward')
    if ylims is not None:
        ax_l.set_ylim(*ylims['0_3'])
        ax_r.set_ylim(*ylims['1_0'])

    handles, labels = ax_l.get_legend_handles_labels()
    ncol = min(2, max(1, len(handles)))
    fig.legend(handles, labels, loc='upper center', ncol=ncol,
               frameon=True, fancybox=False, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, 0.06), fontsize=FS - 1)
    fig.subplots_adjust(wspace=0.22, bottom=0.22)

    fig.savefig(str(outpath), facecolor='#FFFFFF')
    plt.close(fig)


def main():
    # First render a hidden full-mode figure just to capture the reference y-limits
    # so that 5_1 (Terminal Only) shares the exact same axes as 5_2 (all 4 modes).
    fig_ref, (ax_ref_l, ax_ref_r) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_panel(ax_ref_l, '0_3', r'Low Demand')
    plot_panel(ax_ref_r, '1_0', r'High Demand')
    ylims = {'0_3': ax_ref_l.get_ylim(), '1_0': ax_ref_r.get_ylim()}
    plt.close(fig_ref)

    # 5_1: 2-panel Low + High Transit Demand, TERMINAL ONLY curve
    _save_two_panel(OUTDIR / '5_1.pdf',
                    modes_filter=['terminal_only'], ylims=ylims)
    # 5_2: 2-panel Low + High Transit Demand, all 4 reward modes
    _save_two_panel(OUTDIR / '5_2.pdf', ylims=ylims)

    print(f'Saved 5_1.pdf, 5_2.pdf')


if __name__ == '__main__':
    main()

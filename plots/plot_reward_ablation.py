"""
Plot PPO reward ablation results.

Usage:
    python plots/plot_reward_ablation.py --outdir PATH --sweeps 0_3:/path/to/all_runs.json

Outputs:
    - reward_ablation_alpha_{alpha}.png
"""
import json
import argparse
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

# Dual-hue inspired palette for reward modes
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

# Plot order
MODE_ORDER = [
    'terminal_only',
    'terminal_intermediate_raw_early_stop',
    'terminal_intermediate_delta_early_stop',
    'terminal_intermediate_delta_no_early_stop',
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

SAVE_PDF = False  # Set to True to also save PDF versions


def _save(fig, out_path: Path):
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    if SAVE_PDF:
        fig.savefig(out_path.with_suffix('.pdf'))
        print(f'Saved {out_path.with_suffix(".pdf")}')


def plot_ablation(alpha: str, data_path: Path, outdir: Path):
    if not data_path.exists():
        print(f'No data for alpha={alpha}, skipping.')
        return

    with open(data_path) as f:
        runs = json.load(f)

    # Group histories by reward mode
    mode_histories = {}
    for run in runs:
        mode = run.get('ppo_reward_mode', run.get('config', {}).get('ppo_reward_mode', '?'))
        mode_histories.setdefault(mode, []).append(run['history'])

    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in MODE_ORDER:
        histories = mode_histories.get(mode, [])
        if not histories:
            continue

        style = MODE_STYLES[mode]
        min_len = min(len(h) for h in histories)
        trimmed = np.array([h[:min_len] for h in histories])
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)

        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, label=style['label'], color=style['color'], linewidth=1.8)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=style['color'])

    ax.set_xlabel(r'Eval Step')
    ax.set_ylabel(r'Eval Terminal Reward')
    ax.grid(True)

    # Legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
              frameon=True, fancybox=False, edgecolor='#CCCCCC',
              bbox_to_anchor=(0.5, -0.06))

    fig.subplots_adjust(bottom=0.18)

    out = outdir / f'reward_ablation_alpha_{alpha}.png'
    _save(fig, out)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot PPO reward ablation results')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save plots')
    parser.add_argument('--sweeps', type=str, nargs='+', required=True,
                        help='alpha:data_json pairs, e.g. 0_3:/path/to/all_runs.json')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise FileNotFoundError(f'Output directory does not exist: {outdir}')

    for entry in args.sweeps:
        alpha, data_path = entry.split(':', 1)
        plot_ablation(alpha, Path(data_path), outdir)

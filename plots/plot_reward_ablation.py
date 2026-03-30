"""
Plot PPO reward ablation results — both alphas side by side.

Usage:
    python plots/plot_reward_ablation.py --outdir PATH --sweep_0_3 PATH --sweep_1_0 PATH

At least one sweep path must be provided. Missing sweeps show a placeholder.
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

SAVE_PDF = False  # Set to True to also save PDF versions
SMOOTH_WINDOW = 10


def _save(fig, out_path: Path):
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(pdf_path)
    print(f'Saved {pdf_path}')


def _plot_ablation_on_ax(ax, data_path: Path):
    """Plot reward ablation curves on a given axes. Returns True if data was plotted."""
    if data_path is None or not data_path.exists():
        ax.text(0.5, 0.5, r'\textit{Pending}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=13, color='#999999')
        ax.set_xlabel(r'Environment Steps')
        ax.set_ylabel(r'Eval Terminal Reward')
        ax.grid(True)
        return False

    with open(data_path) as f:
        runs = json.load(f)

    mode_data = {}
    for run in runs:
        mode = run.get('ppo_reward_mode', run.get('config', {}).get('ppo_reward_mode', '?'))
        mode_data.setdefault(mode, []).append({
            'history': run['history'],
            'steps': run.get('steps', list(range(len(run['history'])))),
        })

    for mode in MODE_ORDER:
        entries = mode_data.get(mode, [])
        if not entries:
            continue

        style = MODE_STYLES[mode]
        min_len = min(len(e['history']) for e in entries)
        trimmed = np.array([e['history'][:min_len] for e in entries])
        # Use env steps from first run as x-axis
        x = np.array(entries[0]['steps'][:min_len])
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)

        # Smooth
        if SMOOTH_WINDOW > 1 and len(mean_curve) >= SMOOTH_WINDOW:
            import pandas as pd
            mean_series = pd.Series(mean_curve)
            std_series = pd.Series(std_curve)
            mean_curve = mean_series.rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values
            std_curve = std_series.rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values

        ax.plot(x, mean_curve, label=style['label'], color=style['color'], linewidth=1.8)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=style['color'])

    ax.set_xlabel(r'Environment Steps')
    ax.set_ylabel(r'Eval Terminal Reward')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.grid(True)
    return True


def plot_ablation(outdir: Path, sweep_0_3: Path = None, sweep_1_0: Path = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _plot_ablation_on_ax(axes[0], sweep_0_3)
    _plot_ablation_on_ax(axes[1], sweep_1_0)

    # Shared legend at bottom
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4,
                  frameon=True, fancybox=False, edgecolor='#CCCCCC',
                  bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(wspace=0.25, bottom=0.18)

    out = outdir / 'reward_ablation.png'
    _save(fig, out)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot PPO reward ablation results')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save plots')
    parser.add_argument('--sweep_0_3', type=str, default=None, help='Path to all_runs.json for alpha=0.3')
    parser.add_argument('--sweep_1_0', type=str, default=None, help='Path to all_runs.json for alpha=1.0')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise FileNotFoundError(f'Output directory does not exist: {outdir}')

    sweep_0_3 = Path(args.sweep_0_3) if args.sweep_0_3 else None
    sweep_1_0 = Path(args.sweep_1_0) if args.sweep_1_0 else None

    plot_ablation(outdir, sweep_0_3, sweep_1_0)

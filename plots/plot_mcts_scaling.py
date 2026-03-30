"""
Plot MCTS n_iter scaling: effect of n_iter (100-500) on AlphaTransit training.
Layout: side by side, alpha=0.3 (left) and alpha=1.0 (right).

Data sources (wandb runs, 1 per n_iter per alpha):
  alpha=1.0: sweep r5zy3487
  alpha=0.3: sweeps lpg2ifh2 + j5ns6bh8 + 466yjq3q (n_iter=500 is partial ~0.20M)
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

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
    'legend.fontsize': 10,
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MCTS scaling')
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Colors: gradient from light to saturated for n_iter 100..500
# ---------------------------------------------------------------------------
NITER_COLORS = {
    100: '#A8D8EA',
    200: '#4A90D9',
    300: '#7B68EE',
    400: '#9B59B6',
    500: '#E84393',
}

# ---------------------------------------------------------------------------
# Run ID -> n_iter for alpha=1.0 (filenames lack niter tag)
# ---------------------------------------------------------------------------
RUN_ID_TO_NITER = {
    'irh8ti2d': 100,
    'y94tkhmv': 200,
    'tw2b5jna': 300,
    'aqsv1e8s': 400,
    '4wq54mnu': 500,
}

# Smoothing window
SMOOTH_WINDOW = 20

# Only MCTS metrics for now
METRICS = [
    # ('mcts/avg_reward',  r'Average Reward',  'mcts_scaling_avg_reward'),
    ('mcts/total_loss',  r'Total Loss',      'mcts_scaling_total_loss'),
    # ('mcts/value_loss',  r'Value Loss',      'mcts_scaling_value_loss'),
    # ('mcts/policy_loss', r'Policy Loss',     'mcts_scaling_policy_loss'),
    # ('mcts/best_reward', r'Best Reward',     'mcts_scaling_best_reward'),
    # ('eval/service_rate',            r'Service Rate',    'mcts_scaling_service_rate'),
    # ('eval/episode_terminal_reward', r'Terminal Reward',  'mcts_scaling_terminal_reward'),
]


# ============================= HELPERS =====================================

def load_all_runs(alpha_dir):
    """Load every JSON in alpha_dir. Returns {n_iter: list_of_row_dicts}."""
    runs = {}
    for fpath in sorted(glob.glob(os.path.join(alpha_dir, '*.json'))):
        fname = os.path.basename(fpath)

        # resolve n_iter from filename or run_id lookup
        n_iter = None
        if 'niter' in fname:
            try:
                n_iter = int(fname.split('niter')[1].split('.')[0])
            except ValueError:
                pass
        if n_iter is None:
            run_id = fname.split('_')[0]
            n_iter = RUN_ID_TO_NITER.get(run_id)
        if n_iter is None:
            print(f'  SKIP (unknown n_iter): {fname}')
            continue

        print(f'  Loading {fname} (n_iter={n_iter})...')
        with open(fpath) as f:
            history = json.load(f)
        if history:
            runs[n_iter] = history
    return runs


def extract_metric(history, metric_key):
    """
    Extract (steps, values) for one metric, dropping None rows.
    Returns sorted numpy arrays.
    """
    steps, vals = [], []
    for row in history:
        s = row.get('_step')
        v = row.get(metric_key)
        if s is not None and v is not None:
            steps.append(s)
            vals.append(v)
    if steps:
        order = np.argsort(steps)
        return np.array(steps)[order], np.array(vals)[order]
    return np.array([]), np.array([])




def resample_and_smooth(steps, vals, window=SMOOTH_WINDOW, n_grid=2000):
    """
    Resample onto a uniform grid, then smooth.
    This ensures ALL runs get identical treatment regardless of logging density.
    Returns (grid_steps, smoothed_mean, smoothed_std).
    """
    import pandas as pd

    if len(steps) < 2:
        return steps, vals, np.zeros_like(vals)

    # uniform grid from start to end of this run's data
    grid = np.linspace(steps[0], steps[-1], n_grid)

    # interpolate raw values onto the grid
    resampled = np.interp(grid, steps, vals)

    # smooth on the uniform grid
    s = pd.Series(resampled)
    mean = s.rolling(window=window, min_periods=1).mean().values
    std = s.rolling(window=window, min_periods=2).std().fillna(0).values

    return grid, mean, std


# ============================= LOAD DATA ===================================
print('Loading data...')
DATA = {}
for alpha in ['0_3', '1_0']:
    print(f'\n  alpha={alpha}:')
    DATA[alpha] = load_all_runs(os.path.join(BASE, alpha))
print('\nData loaded.\n')


# ============================= PLOTTING ====================================

def plot_metric(metric_key, ylabel, filename):
    """One figure: two panels (alpha=0.3, alpha=1.0), one line per n_iter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (alpha, alpha_label) in enumerate([
        ('0_3', r'$\alpha=0.3$'),
        ('1_0', r'$\alpha=1.0$'),
    ]):
        ax = axes[col]
        ax.set_title(alpha_label, fontsize=13)
        runs = DATA[alpha]

        if not runs:
            ax.text(0.5, 0.5, r'\textit{No data}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=13, color='#999999')
            ax.set_xlabel(r'Training Step')
            ax.set_ylabel(ylabel)
            ax.grid(True)
            continue

        for n_iter in sorted(runs.keys()):
            # extract raw metric, clip to 1M steps
            steps, vals = extract_metric(runs[n_iter], metric_key)
            if len(steps) == 0:
                continue
            mask = steps <= 1_000_000
            steps, vals = steps[mask], vals[mask]

            # resample to uniform grid then smooth — identical treatment for all
            color = NITER_COLORS.get(n_iter, '#333333')
            steps, mean, std = resample_and_smooth(steps, vals)

            # plot line + shaded band
            ax.plot(steps, mean, color=color, linewidth=1.8,
                    label=rf'$n_{{\mathrm{{iter}}}}={n_iter}$', alpha=0.85)
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.12)

        # temperature schedule transitions at 70% and 90% of training
        # temperature schedule: tau=1.0 -> 0.7 at 70% -> 0.5 at 90%
        ax.axvline(x=700_000, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.axvline(x=900_000, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7)
        yhi = ax.get_ylim()[1]
        ax.text(340_000, yhi * 0.95, r'$\tau=1.0$',
                fontsize=13, color='#999999', va='top', ha='center')
        ax.text(715_000, yhi * 0.95, r'$\tau\!\to\!0.7$',
                fontsize=13, color='#999999', va='top')
        ax.text(915_000, yhi * 0.95, r'$\tau\!\to\!0.5$',
                fontsize=13, color='#999999', va='top')

        ax.set_xlabel(r'Training Step')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
        ax.grid(True)
        ax.set_xlim(-20_000, 1_020_000)

    # shared legend from both panels
    seen = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen[label] = handle
    sorted_items = sorted(seen.items(), key=lambda x: x[0])
    if sorted_items:
        fig.legend([v for _, v in sorted_items],
                   [k for k, _ in sorted_items],
                   loc='lower center', ncol=5,
                   frameon=True, fancybox=False, edgecolor='#CCCCCC',
                   bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(wspace=0.2, bottom=0.18)

    outpath = os.path.join(OUTDIR, filename + '.pdf')
    fig.savefig(outpath)
    print(f'Saved {outpath}')
    plt.close(fig)


for metric_key, ylabel, filename in METRICS:
    print(f'Plotting {metric_key}...')
    plot_metric(metric_key, ylabel, filename)

print('\nAll done!')

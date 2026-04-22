"""
Training dynamics and ablation figures.

Generates three PDFs:
  1. ppo_reward_ablation_and_training.pdf — reward mode ablation + AlphaTransit vs End-to-End RL
  2. mcts_scaling_and_wallclock.pdf — n_iter scaling loss + wall-clock comparison
  3. pareto_analysis.pdf — service rate vs fleet size scatter

Usage:
    python plots/figure_training_dynamics.py
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Style (from plots.md, with white background per user feedback)
# ---------------------------------------------------------------------------
FS = 20  # Base font size — adjust this single value to scale all text
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
NEURIPS_DATA = ROOT / 'training_data' / 'For NeurIPS'
ABLATION_0_3 = NEURIPS_DATA / 'nips_5_ppo_rew_abl_alpha_0_3' / 'wandb_data' / 'all_runs.json'
ABLATION_1_0 = NEURIPS_DATA / 'nips_6_ppo_rew_abl_alpha_1_0' / 'wandb_data' / 'all_runs.json'

# MCTS training data (from rebuttal work — wandb JSON files)
MCTS_DATA = ROOT.parent / 'ICML' / 'Reviewer 4' / 'MCTS scaling'

OUTDIR = ROOT / 'plots'

# ---------------------------------------------------------------------------
# Reward ablation styles
# ---------------------------------------------------------------------------
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

# RIGHT panel styles
# Color = method, linestyle = alpha
# End-to-End RL uses green (#2ECC71) matching "Delta (No Early Stop)" in LEFT/CENTER
# AlphaTransit uses a distinct orange/amber
RIGHT_STYLES = {
    'alphatransit_0.3': {'color': '#4A90D9', 'label': r'AlphaTransit $\alpha{=}0.3$', 'ls': '-'},
    'alphatransit_1.0': {'color': '#2ECC71', 'label': r'AlphaTransit $\alpha{=}1.0$', 'ls': '-'},
    'ppo_0.3': {'color': '#E84393', 'label': r'End-to-End RL $\alpha{=}0.3$', 'ls': '--'},
    'ppo_1.0': {'color': '#9B59B6', 'label': r'End-to-End RL $\alpha{=}1.0$', 'ls': '--'},
}

SMOOTH_WINDOW = 10
RIGHT_SMOOTH_WINDOW = 20  # Moderate smoothing for RIGHT panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smooth(values, window=SMOOTH_WINDOW):
    s = pd.Series(values)
    mean = s.rolling(window=window, min_periods=1).mean().values
    std = s.rolling(window=window, min_periods=1).std().fillna(0).values
    return mean, std


def plot_ablation_panel(ax, data_path, title):
    """Plot reward ablation curves on one axes."""
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
        x = np.array(entries[0]['steps'][:min_len])
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)

        # Smooth
        if len(mean_curve) >= SMOOTH_WINDOW:
            mean_curve, _ = smooth(mean_curve)
            std_curve, _ = smooth(std_curve)

        ax.plot(x, mean_curve, label=style['label'], color=style['color'], linewidth=1.8)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=style['color'])

    ax.set_title(title, fontsize=FS + 1)
    ax.set_xlabel(r'Environment Steps')
    ax.set_ylabel(r'Eval Reward')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)


def load_mcts_run(alpha_dir, n_iter):
    """Load MCTS wandb JSON for a specific n_iter."""
    RUN_ID_TO_NITER = {
        'irh8ti2d': 100, 'y94tkhmv': 200, 'tw2b5jna': 300,
        'aqsv1e8s': 400, '4wq54mnu': 500,
    }
    for fpath in sorted(glob.glob(os.path.join(alpha_dir, '*.json'))):
        fname = os.path.basename(fpath)
        nit = None
        if 'niter' in fname:
            try:
                nit = int(fname.split('niter')[1].split('.')[0])
            except ValueError:
                pass
        if nit is None:
            run_id = fname.split('_')[0]
            nit = RUN_ID_TO_NITER.get(run_id)
        if nit == n_iter:
            with open(fpath) as f:
                return json.load(f)
    return None


def extract_metric(history, key):
    """Extract (steps, values) sorted, dropping None."""
    steps, vals = [], []
    for row in history:
        s, v = row.get('_step'), row.get(key)
        if s is not None and v is not None:
            steps.append(s)
            vals.append(v)
    if steps:
        order = np.argsort(steps)
        return np.array(steps)[order], np.array(vals)[order]
    return np.array([]), np.array([])


def extract_ppo_mode(all_runs_path, mode='terminal_intermediate_delta_no_early_stop'):
    """Extract PPO runs for a specific reward mode, average across seeds."""
    with open(all_runs_path) as f:
        runs = json.load(f)
    entries = []
    for run in runs:
        m = run.get('ppo_reward_mode', run.get('config', {}).get('ppo_reward_mode', '?'))
        if m == mode:
            entries.append({
                'history': run['history'],
                'steps': run.get('steps', list(range(len(run['history'])))),
            })
    if not entries:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(e['history']) for e in entries)
    trimmed = np.array([e['history'][:min_len] for e in entries])
    x = np.array(entries[0]['steps'][:min_len])
    mean_curve = trimmed.mean(axis=0)
    std_curve = trimmed.std(axis=0)
    return x, mean_curve, std_curve


def resample_and_smooth(steps, vals, window=SMOOTH_WINDOW, n_grid=2000):
    """Resample onto a uniform grid then smooth. All data gets the same treatment."""
    if len(steps) < 2:
        return steps, vals, np.zeros_like(vals)
    grid = np.linspace(steps[0], steps[-1], n_grid)
    vals = np.interp(grid, steps, vals)
    s = pd.Series(vals)
    mean = s.rolling(window=window, min_periods=1).mean().values
    std = s.rolling(window=window, min_periods=2).std().fillna(0).values
    return grid, mean, std


def _plot_single_curve(ax, steps, vals, style, target_density=120):
    """Plot a single smoothed curve with shaded std band.
    Resamples dense data to target_density points, then applies
    the same smooth() as LEFT/CENTER panels."""
    if len(steps) > target_density * 2:
        grid = np.linspace(steps[0], steps[-1], target_density)
        vals = np.interp(grid, steps, vals)
        steps = grid
    mean, std = smooth(vals)
    ax.plot(steps, mean, color=style['color'], linewidth=2.0,
            label=style['label'], linestyle=style['ls'])
    ax.fill_between(steps, mean - std, mean + std,
                    alpha=0.20, color=style['color'])


def plot_training_panel(ax):
    """RIGHT panel: AlphaTransit (n_iter=500) vs End-to-End RL at both alphas."""
    metric_key = 'eval/episode_terminal_reward'

    # AlphaTransit at n_iter=500 for both alphas
    for alpha, alpha_dir_name, style_key in [
        ('0.3', '0_3', 'alphatransit_0.3'),
        ('1.0', '1_0', 'alphatransit_1.0'),
    ]:
        mcts_dir = str(MCTS_DATA / alpha_dir_name)
        history = load_mcts_run(mcts_dir, 500)
        if history is None:
            continue
        steps, vals = extract_metric(history, metric_key)
        if len(steps) == 0:
            continue
        mask = steps <= 1_000_000
        steps, vals = steps[mask], vals[mask]
        _plot_single_curve(ax, steps, vals, RIGHT_STYLES[style_key])

    # End-to-End RL (delta_no_early_stop) at both alphas
    for alpha, data_path, style_key in [
        ('0.3', ABLATION_0_3, 'ppo_0.3'),
        ('1.0', ABLATION_1_0, 'ppo_1.0'),
    ]:
        x, mean_curve, std_curve = extract_ppo_mode(str(data_path))
        if len(x) == 0:
            continue
        _plot_single_curve(ax, x, mean_curve, RIGHT_STYLES[style_key])

    ax.set_title(r'AlphaTransit vs End-to-End RL', fontsize=FS + 1)
    ax.set_xlabel(r'Environment Steps')
    ax.set_ylabel(r'Eval Reward')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)
    # Legend handled externally


# ---------------------------------------------------------------------------
# MCTS scaling + wall clock figure (3-panel)
# ---------------------------------------------------------------------------
NITER_COLORS = {
    100: '#A8D8EA',
    200: '#4A90D9',
    300: '#7B68EE',
    400: '#9B59B6',
    500: '#E84393',
}

WALL_CLOCK_DIR = ROOT.parent / 'Reviewer 4' / 'wall_clock'
COLOR_MCTS = '#E84393'
COLOR_ALPHA = '#4A90D9'


def _load_all_mcts_runs(alpha_dir):
    """Load every JSON in alpha_dir. Returns {n_iter: list_of_row_dicts}."""
    RUN_ID_TO_NITER = {
        'irh8ti2d': 100, 'y94tkhmv': 200, 'tw2b5jna': 300,
        'aqsv1e8s': 400, '4wq54mnu': 500,
    }
    runs = {}
    for fpath in sorted(glob.glob(os.path.join(alpha_dir, '*.json'))):
        fname = os.path.basename(fpath)
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
            continue
        with open(fpath) as f:
            runs[n_iter] = json.load(f)
    return runs


def _get_per_route_times(csv_path):
    """Get per-route wall-clock times (minutes) from Pure MCTS log CSV."""
    df = pd.read_csv(csv_path)
    breaks = df.index[(df['route_idx'] == 0) & (df['step'] == 0)].tolist()
    breaks.append(len(df))
    route_times = []
    for i in range(len(breaks) - 1):
        seed_df = df.iloc[breaks[i]:breaks[i + 1]]
        for ridx in range(16):
            route_df = seed_df[seed_df['route_idx'] == ridx]
            if len(route_df) > 0:
                route_times.append(route_df['step_time_s'].sum() / 60)
    return np.array(route_times)


def _plot_mcts_scaling_panel(ax, alpha_dir, title):
    """Plot MCTS n_iter scaling (total_loss) on one axes."""
    runs = _load_all_mcts_runs(alpha_dir)
    metric_key = 'mcts/total_loss'

    for n_iter in sorted(runs.keys()):
        steps, vals = extract_metric(runs[n_iter], metric_key)
        if len(steps) == 0:
            continue
        mask = steps <= 1_000_000
        steps, vals = steps[mask], vals[mask]

        color = NITER_COLORS.get(n_iter, '#333333')
        steps, mean, std = resample_and_smooth(steps, vals, window=20)

        ax.plot(steps, mean, color=color, linewidth=1.8,
                label=rf'$n_{{\mathrm{{iter}}}}={n_iter}$', alpha=0.85)
        ax.fill_between(steps, mean - std, mean + std,
                        color=color, alpha=0.12)

    # Temperature schedule transitions
    ax.axvline(x=700_000, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axvline(x=900_000, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7)
    yhi = ax.get_ylim()[1]
    ax.text(340_000, yhi * 0.95, r'$\tau=1.0$',
            fontsize=FS - 2, color='#999999', va='top', ha='center')
    ax.text(715_000, yhi * 0.95, r'$\tau\!\to\!0.7$',
            fontsize=FS - 2, color='#999999', va='top')
    ax.text(915_000, yhi * 0.95, r'$\tau\!\to\!0.5$',
            fontsize=FS - 2, color='#999999', va='top')

    ax.set_title(title, fontsize=FS + 1)
    ax.set_xlabel(r'Environment Steps')
    ax.set_ylabel(r'Total Loss')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.grid(True)
    ax.set_xlim(-20_000, 1_020_000)


WALL_CLOCK_SCALING_CSV = OUTDIR / 'wall_clock_scaling' / 'results.csv'


def _load_wall_clock_scaling():
    """Load wall-clock scaling data from experiment CSV.

    Returns a pandas DataFrame with columns:
    method, alpha, n_iter, seed, total_seconds, route_length, num_steps.
    Raises FileNotFoundError if results.csv is missing.
    """
    if not WALL_CLOCK_SCALING_CSV.exists():
        raise FileNotFoundError(
            f"Wall-clock scaling data not found: {WALL_CLOCK_SCALING_CSV}\n"
            f"Run: python plots/experiment_wall_clock.py"
        )
    return pd.read_csv(WALL_CLOCK_SCALING_CSV)


def _plot_wall_clock_scaling_panel(ax, alpha):
    """Grouped bar chart: Pure MCTS vs AlphaTransit wall-clock time across n_iter values.

    Reads from plots/wall_clock_scaling/results.csv. Plots 5 groups
    (n_iter=100..500) with 2 bars each (Pure MCTS, AlphaTransit).
    """
    df = _load_wall_clock_scaling()
    df = df[df['alpha'] == alpha]

    n_iters = sorted(df['n_iter'].unique())
    x = np.arange(len(n_iters))
    width = 0.32

    mcts_times = []
    at_times = []
    for n in n_iters:
        mcts_row = df[(df['method'] == 'pure_mcts') & (df['n_iter'] == n)]
        at_row = df[(df['method'] == 'alphatransit') & (df['n_iter'] == n)]
        mcts_times.append(mcts_row['total_seconds'].values[0] / 60 if len(mcts_row) else 0)
        at_times.append(at_row['total_seconds'].values[0] / 60 if len(at_row) else 0)

    bars1 = ax.bar(x - width / 2, mcts_times, width,
                   color=COLOR_MCTS, alpha=0.85, label='Pure MCTS', zorder=3)
    bars2 = ax.bar(x + width / 2, at_times, width,
                   color=COLOR_ALPHA, alpha=0.85, label='AlphaTransit', zorder=3)

    for bar, val in zip(bars1, mcts_times):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=FS - 3, color='#333333')

    for bar, val in zip(bars2, at_times):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=FS - 3, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_iters], fontsize=FS - 1)
    ax.set_xlabel(r'$n_{\mathrm{iter}}$')
    ax.set_ylabel(r'Time per Route (minutes)')
    ax.grid(True, axis='y', zorder=0)
    if mcts_times:
        ax.set_ylim(0, max(mcts_times) * 1.25)


def plot_mcts_scaling_and_wallclock():
    """3-panel figure: MCTS n_iter scaling (two alphas) + wall-clock comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # LEFT: MCTS scaling alpha=0.3
    _plot_mcts_scaling_panel(axes[0], str(MCTS_DATA / '0_3'), r'$\alpha=0.3$')

    # MIDDLE: MCTS scaling alpha=1.0
    _plot_mcts_scaling_panel(axes[1], str(MCTS_DATA / '1_0'), r'$\alpha=1.0$')

    # RIGHT: Wall clock scaling (alpha=0.3 only)
    _plot_wall_clock_scaling_panel(axes[2], alpha=0.3)

    # Shared legend for n_iter at bottom, spanning LEFT+MIDDLE panels
    seen = {}
    for ax in axes[:2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen[label] = handle
    sorted_items = sorted(seen.items(), key=lambda x: x[0])
    if sorted_items:
        left_mid_center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        fig.legend([v for _, v in sorted_items],
                   [k for k, _ in sorted_items],
                   loc='upper center', ncol=5,
                   frameon=True, fancybox=False, edgecolor='#CCCCCC',
                   bbox_to_anchor=(left_mid_center, 0.08), fontsize=FS - 2)

    # External legend for RIGHT panel (wall clock)
    handles_wc, labels_wc = axes[2].get_legend_handles_labels()
    if handles_wc:
        right_center = (axes[2].get_position().x0 + axes[2].get_position().x1) / 2
        fig.legend(handles_wc, labels_wc, loc='upper center', ncol=2,
                   frameon=True, fancybox=False, edgecolor='#CCCCCC',
                   bbox_to_anchor=(right_center, 0.08), fontsize=FS - 2)

    fig.subplots_adjust(wspace=0.28, bottom=0.20)

    outpath = OUTDIR / 'mcts_scaling_and_wallclock'
    fig.savefig(str(outpath) + '.pdf')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


def plot_wall_clock_alpha1_0():
    """Standalone wall-clock scaling figure for alpha=1.0 (appendix).

    Single-panel grouped bar chart showing Pure MCTS vs AlphaTransit
    wall-clock time across n_iter values at alpha=1.0.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    _plot_wall_clock_scaling_panel(ax, alpha=1.0)

    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='#CCCCCC', fontsize=FS - 2)

    fig.tight_layout()
    outpath = OUTDIR / 'wall_clock_alpha1_0'
    fig.savefig(str(outpath) + '.pdf')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto analysis: Service Rate vs Fleet Size
# ---------------------------------------------------------------------------
PARETO_DATA = {
    '0.3': {
        'Real-World':    (42.91, 89),
        'Random Walk':   (40.91, 37.20),
        'Demand Cover':  (39.72, 37.80),
        'Shortest Path': (39.69, 20.60),
        'Genetic Alg.':  (53.39, 67),
        'Bee Colony':    (39.83, 94),
        'Neural Evol.':  (47.85, 101),
        'Pure MCTS':     (50.44, 76),
        'End-to-End RL': (55.86, 46),
        'AlphaTransit':  (61.34, 55),
    },
    '1.0': {
        'Real-World':    (58.78, 281),
        'Random Walk':   (60.39, 113.40),
        'Demand Cover':  (65.50, 128.20),
        'Shortest Path': (59.55, 48.60),
        'Genetic Alg.':  (81.05, 202),
        'Bee Colony':    (64.74, 301),
        'Neural Evol.':  (70.51, 320),
        'Pure MCTS':     (72.04, 185),
        'End-to-End RL': (66.10, 193),
        'AlphaTransit':  (85.50, 167),
    },
}

PARETO_STYLES = {
    'AlphaTransit':  {'color': '#2ECC71', 'marker': '*', 'size': 350, 'zorder': 10},
    'End-to-End RL': {'color': '#7B68EE', 'marker': 's', 'size': 130, 'zorder': 5},
    'Pure MCTS':     {'color': '#E84393', 'marker': 'D', 'size': 130, 'zorder': 5},
    'Neural Evol.':  {'color': '#3498DB', 'marker': '^', 'size': 150, 'zorder': 5},
    'Bee Colony':    {'color': '#E74C3C', 'marker': 'v', 'size': 150, 'zorder': 5},
    'Genetic Alg.':  {'color': '#9B59B6', 'marker': 'p', 'size': 150, 'zorder': 5},
    'Real-World':    {'color': '#555555', 'marker': 'X', 'size': 130, 'zorder': 3},
    'Random Walk':   {'color': '#95A5A6', 'marker': 'h', 'size': 140, 'zorder': 3},
    'Demand Cover':  {'color': '#1ABC9C', 'marker': 'o', 'size': 120, 'zorder': 3},
    'Shortest Path': {'color': '#F39C12', 'marker': 'P', 'size': 130, 'zorder': 3},
}

PARETO_OFFSETS = {
    '0.3': {
        'AlphaTransit':  (0, -12),
        'End-to-End RL': (0, -12),
        'Pure MCTS':     (0, -12),
        'Neural Evol.':  (0, -12),
        'Bee Colony':    (0, -12),
        'Genetic Alg.':  (0, -12),
        'Real-World':    (0, 12),
        'Random Walk':   (0, 12),
        'Demand Cover':  (0, -12),
        'Shortest Path': (0, 12),
    },
    '1.0': {
        'AlphaTransit':  (0, -12),
        'Genetic Alg.':  (0, -12),
        'Pure MCTS':     (0, 12),
        'Neural Evol.':  (0, -12),
        'End-to-End RL': (0, 12),
        'Demand Cover':  (0, -12),
        'Bee Colony':    (0, -12),
        'Random Walk':   (0, 12),
        'Shortest Path': (0, -12),
        'Real-World':    (0, -12),
    },
}

PARETO_LABEL_FS = FS - 4


def plot_pareto():
    """Scatter plot: Service Rate vs Fleet Size at both alphas."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, (alpha, alpha_label) in enumerate([
        ('0.3', r'Low Demand'),
        ('1.0', r'High Demand'),
    ]):
        ax = axes[col]
        ax.set_title(alpha_label, fontsize=FS + 1)

        for method, (sr, fleet) in PARETO_DATA[alpha].items():
            style = PARETO_STYLES[method]
            ax.scatter(fleet, sr,
                       color=style['color'], marker=style['marker'],
                       s=style['size'], zorder=style['zorder'],
                       edgecolors='white', linewidth=0.5)

            dx, dy = PARETO_OFFSETS[alpha].get(method, (10, 0))
            label_text = r'\textbf{' + method + '}' if method == 'AlphaTransit' else method
            if dx > 0:
                ha = 'left'
            elif dx < 0:
                ha = 'right'
            else:
                ha = 'center'
            va = 'center' if abs(dy) <= 2 else ('bottom' if dy > 0 else 'top')
            ax.annotate(label_text, (fleet, sr), xytext=(dx, dy),
                        textcoords='offset points', fontsize=PARETO_LABEL_FS,
                        color=style['color'], ha=ha, va=va)

        ax.set_xlabel(r'Fleet Size')
        ax.set_ylabel(r'Service Rate (\%)')
        ax.grid(True, zorder=0)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xpad = (xlim[1] - xlim[0]) * 0.08
        ypad = (ylim[1] - ylim[0]) * 0.05
        ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
        ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)

    fig.subplots_adjust(wspace=0.25)

    outpath = OUTDIR / 'pareto_analysis'
    fig.savefig(str(outpath) + '.pdf')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def main():
    # Figure 1: PPO reward ablation + AlphaTransit vs End-to-End RL
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # LEFT: PPO reward ablation alpha=0.3
    plot_ablation_panel(axes[0], str(ABLATION_0_3), r'End-to-End RL $\alpha=0.3$')

    # CENTER: PPO reward ablation alpha=1.0
    plot_ablation_panel(axes[1], str(ABLATION_1_0), r'End-to-End RL $\alpha=1.0$')

    # RIGHT: AlphaTransit vs PPO training curves
    plot_training_panel(axes[2])

    # Position legends just below the x-axis labels
    # LEFT/CENTER legend centered under first two panels
    handles_abl, labels_abl = axes[0].get_legend_handles_labels()
    if handles_abl:
        fig.legend(handles_abl, labels_abl, loc='upper center', ncol=2,
                   frameon=True, fancybox=False, edgecolor='#CCCCCC',
                   bbox_to_anchor=(0.35, 0.08), fontsize=FS - 1)

    # RIGHT legend: grouped by alpha headers, centered under third panel
    right_center = (axes[2].get_position().x0 + axes[2].get_position().x1) / 2
    import matplotlib.lines as mlines
    # Row 1: alpha headers (invisible handles)
    hdr_03 = mlines.Line2D([], [], color='none', label=r'\textbf{$\alpha{=}0.3$}')
    hdr_10 = mlines.Line2D([], [], color='none', label=r'\textbf{$\alpha{=}1.0$}')
    # Row 2: AlphaTransit (solid lines, matching colors)
    at_03 = mlines.Line2D([], [], color='#4A90D9', ls='-', lw=2.0, label='AlphaTransit')
    at_10 = mlines.Line2D([], [], color='#2ECC71', ls='-', lw=2.0, label='AlphaTransit')
    # Row 3: End-to-End RL (dashed lines, matching colors)
    e2e_03 = mlines.Line2D([], [], color='#E84393', ls='--', lw=2.0, label='End-to-End RL')
    e2e_10 = mlines.Line2D([], [], color='#9B59B6', ls='--', lw=2.0, label='End-to-End RL')
    # ncol=2 fills row-first: [hdr_03, hdr_10], [at_03, at_10], [e2e_03, e2e_10]
    leg_handles = [hdr_03, hdr_10, at_03, at_10, e2e_03, e2e_10]
    leg_labels = [h.get_label() for h in leg_handles]
    fig.legend(leg_handles, leg_labels, loc='upper center', ncol=2,
               frameon=True, fancybox=False, edgecolor='#CCCCCC',
               bbox_to_anchor=(right_center, 0.08), fontsize=FS - 1,
               handlelength=1.5, columnspacing=1.2, handletextpad=0.5)

    fig.subplots_adjust(wspace=0.28, bottom=0.20)

    outpath = OUTDIR / 'ppo_reward_ablation_and_training'
    fig.savefig(str(outpath) + '.pdf')
    # fig.savefig(str(outpath) + '.png')  # dvipng not available
    print(f'Saved {outpath}.pdf')
    plt.close(fig)

    # Figure 2: MCTS scaling + wall clock (alpha=0.3 in RIGHT panel)
    plot_mcts_scaling_and_wallclock()

    # Figure 3: Wall clock alpha=1.0 (appendix)
    plot_wall_clock_alpha1_0()

    # Figure 4: Pareto analysis
    plot_pareto()


if __name__ == '__main__':
    main()

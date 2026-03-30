"""
Plot average route length over training for AlphaTransit (MCTS) and PPO.
Shows that agents learn to design routes shorter than L_max=14.

Layout: 2 columns (left: alpha=0.3, right: alpha=1.0)
Each panel shows one MCTS and one PPO run (best config from paper).

Selected runs (1 per algo per alpha, matching paper's BEST_PARAMS):
  PPO  alpha=0.3: imvwgy8o (reward_mode=terminal_intermediate_delta_no_early_stop)
  PPO  alpha=1.0: du13apih (reward_mode=terminal_intermediate_delta_no_early_stop)
  MCTS alpha=0.3: vywy1t0j (n_iter=100, matching BEST_PARAMS)
  MCTS alpha=1.0: ksy2e7q0 (n_iter=100, matching BEST_PARAMS)
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

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

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'route_lengths')
OUTDIR = os.path.dirname(os.path.abspath(__file__))

ALGO_STYLES = {
    'mcts': {
        'color': '#4A90D9',
        'label': 'AlphaTransit',
    },
    'ppo': {
        'color': '#E84393',
        'label': 'End-to-End RL',
    },
}

L_MAX = 14       # maximum route length constraint
NUM_ROUTES = 16  # routes per episode
SMOOTH_WINDOW = 5


def load_run(algo, alpha):
    """Load the single selected run for this algo/alpha combination."""
    dirpath = os.path.join(BASE, algo, alpha)
    files = sorted(glob.glob(os.path.join(dirpath, '*.json')))

    if not files:
        print(f"  WARNING: No data for {algo}/{alpha}")
        return None, None

    fpath = files[0]
    print(f"  Loading {os.path.basename(fpath)}")

    with open(fpath) as f:
        history = json.load(f)

    steps = []
    avg_lengths = []
    for row in history:
        step = row.get('_step')
        if step is None:
            continue

        # episode/mean_length (PPO) or eval/episode_length (MCTS):
        # total steps across all 16 routes. Divide by NUM_ROUTES for per-route avg.
        # PPO logs edges (not nodes), so add +1 per route for initial node.
        ep_len = row.get('episode/mean_length') or row.get('eval/episode_length')
        if ep_len is not None:
            avg_route_len = ep_len / NUM_ROUTES
            if algo == 'ppo':
                avg_route_len += 1  # PPO logs edges; +1 for initial node
            steps.append(step)
            avg_lengths.append(avg_route_len)

    if not steps:
        print(f"  WARNING: No route length data in {fpath}")
        return None, None

    return np.array(steps), np.array(avg_lengths)


def _plot_alpha_on_ax(ax, alpha):
    """Plot both algos for one alpha on the given axes."""
    for algo in ['mcts', 'ppo']:
        print(f"Processing {algo} {alpha}...")
        steps, avg_len = load_run(algo, alpha)
        if steps is None:
            continue

        # Clip to 1M steps
        mask = steps <= 1_000_000
        steps, avg_len = steps[mask], avg_len[mask]

        style = ALGO_STYLES[algo]

        # Compute rolling mean + std for shaded region
        if SMOOTH_WINDOW > 1 and len(avg_len) >= SMOOTH_WINDOW:
            import pandas as pd
            raw = pd.Series(avg_len)
            smooth_mean = raw.rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values
            smooth_std = raw.rolling(window=SMOOTH_WINDOW, min_periods=1).std().fillna(0).values
        else:
            smooth_mean = avg_len
            smooth_std = np.zeros_like(avg_len)

        ax.plot(steps, smooth_mean, color=style['color'],
                linewidth=1.8, label=style['label'], alpha=0.85)
        ax.fill_between(steps, smooth_mean - smooth_std, smooth_mean + smooth_std,
                        color=style['color'], alpha=0.15)

    # L_max reference line
    ax.axhline(y=L_MAX, color='#999999', linestyle='--', linewidth=1.2,
               label=r'$L_{\max}=14$')

    ax.set_xlabel(r'Training Step')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: rf'{v/1e6:.1f}M' if v >= 1e6 else rf'{v/1e3:.0f}K'))
    ax.grid(True)
    ax.set_ylim(bottom=8, top=L_MAX + 0.5)
    ax.set_xlim(-20_000, 1_020_000)  # 2% padding on each side


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for col, (alpha, alpha_label) in enumerate([
    ('0_3', r'$\alpha=0.3$'),
    ('1_0', r'$\alpha=1.0$'),
]):
    ax = axes[col]
    ax.set_title(alpha_label, fontsize=13)
    _plot_alpha_on_ax(ax, alpha)
    ax.set_ylabel(r'Average Route Length')

# Shared legend at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3,
           frameon=True, fancybox=False, edgecolor='#CCCCCC',
           bbox_to_anchor=(0.5, -0.02))

fig.subplots_adjust(wspace=0.15, bottom=0.18)

outpath = os.path.join(OUTDIR, 'route_lengths_over_training')
# fig.savefig(outpath + '.png')
fig.savefig(outpath + '.pdf')
print(f"\nSaved to {outpath}.pdf")
plt.close(fig)

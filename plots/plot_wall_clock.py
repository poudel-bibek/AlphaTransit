"""
Bar chart: wall-clock time per route for Pure MCTS vs AlphaTransit.
Single figure with grouped bars, both alphas, error bars from per-route variance.

Data:
  Pure MCTS: per-route times from CSV logs (16 routes/seed, 2 seeds each)
  AlphaTransit: total wandb _runtime / (n_episodes * 16 routes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    'xtick.labelsize': 12,
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
WALL_DIR = os.path.join(OUTDIR, 'wall_clock')

COLOR_MCTS = '#E84393'
COLOR_ALPHA = '#4A90D9'


def get_per_route_times(csv_path):
    """Get per-route wall-clock times (minutes) from Pure MCTS log CSV."""
    df = pd.read_csv(csv_path)
    # seed boundaries
    breaks = df.index[(df['route_idx'] == 0) & (df['step'] == 0)].tolist()
    breaks.append(len(df))

    route_times = []
    for i in range(len(breaks) - 1):
        seed_df = df.iloc[breaks[i]:breaks[i + 1]]
        for ridx in range(16):
            route_df = seed_df[seed_df['route_idx'] == ridx]
            if len(route_df) > 0:
                route_times.append(route_df['step_time_s'].sum() / 60)  # minutes
    return np.array(route_times)


# --- Compute per-route data ---
mcts_03 = get_per_route_times(os.path.join(WALL_DIR, 'pure_mcts_alpha0.3_log.csv'))
mcts_10 = get_per_route_times(os.path.join(WALL_DIR, 'pure_mcts_alpha1.0_log.csv'))

# AlphaTransit: total_runtime / (308 episodes * 16 routes)
at_03_min = 15.4 * 60 / (308 * 16)  # ~0.19 min per route
at_10_min = 9.5 * 60 / (308 * 16)   # ~0.12 min per route

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 5))

x = np.array([0.35, 0.85])
width = 0.18

# Pure MCTS bars with error bars (per-route std)
mcts_means = [mcts_03.mean(), mcts_10.mean()]
mcts_stds = [mcts_03.std(), mcts_10.std()]
bars1 = ax.bar(x - width/2, mcts_means, width, yerr=mcts_stds,
               color=COLOR_MCTS, alpha=0.85, label='Pure MCTS',
               capsize=5, error_kw={'linewidth': 1.5}, zorder=3)

# AlphaTransit bars
at_means = [at_03_min, at_10_min]
bars2 = ax.bar(x + width/2, at_means, width,
               color=COLOR_ALPHA, alpha=0.85, label='AlphaTransit', zorder=3)

# Add value labels on bars
for bar, val, std in zip(bars1, mcts_means, mcts_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=13, color='#333333')

for bar, val in zip(bars2, at_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}', ha='center', va='bottom', fontsize=13, color='#333333')

ax.set_xlim(0, 1.2)
ax.set_xticks(x)
ax.set_xticklabels([r'$\alpha=0.3$', r'$\alpha=1.0$'])
ax.tick_params(axis='x', direction='out', bottom=True, top=False)
ax.set_ylabel(r'Time per Route (minutes)')
ax.grid(True, axis='y', zorder=0)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, max(mcts_means) * 1.4)

fig.tight_layout()

outpath = os.path.join(OUTDIR, 'wall_clock_comparison')
fig.savefig(outpath + '.pdf')
fig.savefig(outpath + '.png', dpi=300)
print(f'Saved {outpath}.pdf and .png')
plt.close(fig)

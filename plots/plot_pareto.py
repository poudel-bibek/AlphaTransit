"""
Service Rate vs Fleet Size scatter/Pareto plot.
Shows AlphaTransit achieves a better operating point: higher service rate
with moderate fleet size, not cell-by-cell domination.

Layout: side by side, alpha=0.3 (left) and alpha=1.0 (right).
Each method is a labeled point. New baselines (BCO, NEA, Pure MCTS) highlighted.
"""

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

import os
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data from the results table (service_rate %, fleet_size)
# ---------------------------------------------------------------------------
data = {
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

# Method styling
# All distinct markers, no repeated triangles, all distinct colors
# Star renders larger than others at same size value, so reduce it
STYLES = {
    'AlphaTransit':  {'color': '#2ECC71', 'marker': '*', 'size': 200, 'zorder': 10},
    'End-to-End RL': {'color': '#7B68EE', 'marker': 's', 'size': 80,  'zorder': 5},
    'Pure MCTS':     {'color': '#E84393', 'marker': 'D', 'size': 80,  'zorder': 5},
    'Neural Evol.':  {'color': '#3498DB', 'marker': '^', 'size': 100, 'zorder': 5},
    'Bee Colony':    {'color': '#E74C3C', 'marker': 'v', 'size': 100, 'zorder': 5},
    'Genetic Alg.':  {'color': '#9B59B6', 'marker': 'p', 'size': 100, 'zorder': 5},
    'Real-World':    {'color': '#555555', 'marker': 'X', 'size': 80,  'zorder': 3},
    'Random Walk':   {'color': '#95A5A6', 'marker': 'h', 'size': 90,  'zorder': 3},
    'Demand Cover':  {'color': '#1ABC9C', 'marker': 'o', 'size': 70,  'zorder': 3},
    'Shortest Path': {'color': '#F39C12', 'marker': 'P', 'size': 80,  'zorder': 3},
}

# Label offsets in points (dx, dy) — carefully positioned to avoid symbol overlap
# Positive dx = right, positive dy = up
OFFSETS = {
    '0.3': {
        # horizontally or vertically aligned with symbol
        'AlphaTransit':  (0, -9),    # below, h-centered
        'End-to-End RL': (0, -9),    # below, h-centered
        'Pure MCTS':     (0, -9),    # below, h-centered
        'Neural Evol.':  (0, -9),    # below, h-centered
        'Bee Colony':    (0, -9),    # below, h-centered
        'Genetic Alg.':  (0, -9),    # below, h-centered
        'Real-World':    (0, 9),     # above, h-centered
        'Random Walk':   (0, 9),     # above, h-centered
        'Demand Cover':  (0, -8),    # below, h-centered
        'Shortest Path': (0, 8),     # above, h-centered
    },
    '1.0': {
        # horizontally or vertically aligned with symbol
        'AlphaTransit':  (0, -9),    # below, h-centered
        'Genetic Alg.':  (0, -9),    # below, h-centered
        'Pure MCTS':     (0, 9),     # above, h-centered
        'Neural Evol.':  (0, -9),    # below, h-centered
        'End-to-End RL': (0, 9),     # above, h-centered
        'Demand Cover':  (0, -9),    # below, h-centered
        'Bee Colony':    (0, -9),    # below, h-centered
        'Random Walk':   (0, 9),     # above, h-centered
        'Shortest Path': (0, -9),    # below, h-centered
        'Real-World':    (0, -9),    # below, h-centered
    },
}

LABEL_FONTSIZE = 10


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for col, (alpha, alpha_label) in enumerate([
    ('0.3', r'$\alpha=0.3$'),
    ('1.0', r'$\alpha=1.0$'),
]):
    ax = axes[col]
    ax.set_title(alpha_label, fontsize=13)

    for method, (sr, fleet) in data[alpha].items():
        style = STYLES[method]
        ax.scatter(fleet, sr,
                   color=style['color'], marker=style['marker'],
                   s=style['size'], zorder=style['zorder'],
                   edgecolors='white', linewidth=0.5)

        # label — offset in points, centered to symbol where possible
        dx, dy = OFFSETS[alpha].get(method, (8, 0))
        label_text = r'\textbf{' + method + '}' if method == 'AlphaTransit' else method
        fontweight = 'normal'
        # horizontal: left-align if pushing right, right-align if pushing left
        if dx > 0:
            ha = 'left'
        elif dx < 0:
            ha = 'right'
        else:
            ha = 'center'
        # vertical: center when dx is dominant, else align based on dy
        va = 'center' if abs(dy) <= 2 else ('bottom' if dy > 0 else 'top')
        ax.annotate(label_text, (fleet, sr), xytext=(dx, dy),
                    textcoords='offset points', fontsize=LABEL_FONTSIZE,
                    fontweight=fontweight, color=style['color'],
                    ha=ha, va=va)

    ax.set_xlabel(r'Fleet Size')
    ax.set_ylabel(r'Service Rate (\%)')
    ax.grid(True, zorder=0)
    # pad axes so labels don't clip
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xpad = (xlim[1] - xlim[0]) * 0.08
    ypad = (ylim[1] - ylim[0]) * 0.05
    ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
    ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)

fig.subplots_adjust(wspace=0.2)

outpath = os.path.join(OUTDIR, 'pareto_service_fleet')
fig.savefig(outpath + '.pdf')
print(f'Saved {outpath}.pdf')
plt.close(fig)

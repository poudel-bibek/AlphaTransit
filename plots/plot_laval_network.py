"""
Laval network visualization: 632 nodes, 1971 links.
Highlights hub node 542 (highest degree + centrality).
Shows node degree via size, demand via color intensity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ---------------------------------------------------------------------------
# Style from plots.md
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
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
DATA_DIR = os.path.join(os.path.dirname(OUTDIR), 'networks', 'laval')
RAW_DIR = os.path.dirname(OUTDIR)  # generalization/ folder for raw Mumford files

HUB_NODE = 542

# --- Load data ---
nodes = pd.read_csv(os.path.join(DATA_DIR, 'laval_nodes_standard.csv'))
links = pd.read_csv(os.path.join(DATA_DIR, 'laval_links_standard.csv'))
demand = pd.read_csv(os.path.join(DATA_DIR, 'laval_demand_standard.csv'))

# Build coordinate lookup
coords = {row['name']: (row['x'], row['y']) for _, row in nodes.iterrows()}

# Compute degree per node
degree = {}
for _, row in links.iterrows():
    s, e = row['start'], row['end']
    degree[s] = degree.get(s, 0) + 1
    degree[e] = degree.get(e, 0) + 1

# Compute total demand per node (outgoing + incoming)
node_demand = {}
for _, row in demand.iterrows():
    o, d = row['orig'], row['dest']
    node_demand[o] = node_demand.get(o, 0) + row['volume']
    node_demand[d] = node_demand.get(d, 0) + row['volume']

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 8))

# Draw links
for _, row in links.iterrows():
    s, e = row['start'], row['end']
    if s in coords and e in coords:
        x0, y0 = coords[s]
        x1, y1 = coords[e]
        ax.plot([x0, x1], [y0, y1], color='#CCCCCC', linewidth=0.3, zorder=1)

# Prepare node arrays
node_names = list(coords.keys())
xs = np.array([coords[n][0] for n in node_names])
ys = np.array([coords[n][1] for n in node_names])
degs = np.array([degree.get(n, 1) for n in node_names])
demands = np.array([node_demand.get(n, 0) for n in node_names])

# Size by degree (scaled)
sizes = 3 + (degs - degs.min()) / (degs.max() - degs.min()) * 40

# Color by demand (log scale for better visibility)
log_demands = np.log1p(demands)
norm = mcolors.Normalize(vmin=log_demands.min(), vmax=log_demands.max())

# Draw all nodes
scatter = ax.scatter(xs, ys, s=sizes, c=log_demands, cmap='YlOrRd',
                     norm=norm, edgecolors='none', alpha=0.8, zorder=2)

# Highlight hub node with distinct diamond marker
hub_x, hub_y = coords[HUB_NODE]
ax.scatter([hub_x], [hub_y], s=80, c='#1A1A1A', edgecolors='white',
           linewidths=0.8, zorder=4, marker='D')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label(r'Demand (log scale)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Axis labels
ax.set_xlabel(r'Easting (m)')
ax.set_ylabel(r'Northing (m)')

# Clean up axes
ax.set_aspect('equal')
ax.tick_params(axis='both', which='both', bottom=True, left=True)

# Legend in top left
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#1A1A1A',
           markeredgecolor='white', markersize=8, label=r'Transit center (node 542)'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
          fontsize=10, edgecolor='#CCCCCC')


fig.tight_layout()

outpath = os.path.join(OUTDIR, 'laval_network')
fig.savefig(outpath + '.pdf')
print(f'Saved {outpath}.pdf')
plt.close(fig)

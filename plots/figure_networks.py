"""
Network visualizations (Bloomington and Laval).

Generates two figures:
  1. bloomington_map.pdf — 3-panel Bloomington network overview:
     (a) Road network graph (143 nodes, 243 edges)
     (b) Trip demand (origin/destination bubble sizes)
     (c) Existing 16 transit routes
  2. laval_network.pdf — Laval network (632 nodes, 1971 links):
     Node size by degree, color by demand volume, hub node 542 highlighted.

Data sources:
  - networks/bloomington/bloomington_{nodes,links,demand}_standard.csv
  - networks/bloomington/bloomington_existing_routes.json
  - networks/laval/laval_{nodes,links,demand}_standard.csv

Usage:
    cd AlphaTransit && python plots/figure_networks.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
FS = 14  # Base font size
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
OUTDIR = ROOT / 'plots'

PIN_PATH = OUTDIR / 'google_map_pin_256.png'
PIN_IMAGE = plt.imread(PIN_PATH) if PIN_PATH.exists() else None
PIN_ZOOM = 0.12

# Route color palette: 16 distinct colors
ROUTE_COLORS = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075',
]


# ===========================================================================
# Shared helpers
# ===========================================================================

def _draw_edges(ax, links_df, coords, alpha=0.5, color='#999999', lw=0.6):
    """Draw network edges as line segments between node coordinates.

    Iterates over all rows in links_df and plots each as a line from the
    start node to the end node using the provided coordinate lookup.
    """
    for _, row in links_df.iterrows():
        s, e = str(row['start']), str(row['end'])
        if s in coords and e in coords:
            x0, y0 = coords[s]
            x1, y1 = coords[e]
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                    alpha=alpha, zorder=1, solid_capstyle='round')


def _draw_pin(ax, coords, node_id):
    """Place a Google Maps-style pin icon at the given node.

    Uses an AnnotationBbox with the pin PNG anchored at the node's
    coordinates. No-op if the pin image is missing or node_id is invalid.
    """
    if PIN_IMAGE is None or node_id not in coords:
        return
    cx, cy = coords[node_id]
    pin = AnnotationBbox(
        OffsetImage(PIN_IMAGE, zoom=PIN_ZOOM, resample=True),
        (cx, cy), frameon=False, box_alignment=(0.5, 0.0),
        pad=0, zorder=12)
    ax.add_artist(pin)


def _set_bounds(ax, coords):
    """Set equal-aspect axis limits with 3% padding, hide ticks, style spines.

    Computes tight bounds around all node coordinates then adds a small
    margin so markers at the edges do not clip.
    """
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    xpad = (max(xs) - min(xs)) * 0.03
    ypad = (max(ys) - min(ys)) * 0.03
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color('#CCCCCC')


# ===========================================================================
# Bloomington: 3-panel figure
# ===========================================================================

def _load_bloomington():
    """Load Bloomington network data: nodes, links, demand, and routes.

    Returns a dict with keys: nodes, links, demand, routes, coords,
    origin_demand, dest_demand, top_origin, top_dest, transit_center.
    """
    net_dir = ROOT / 'networks' / 'bloomington'
    nodes = pd.read_csv(net_dir / 'bloomington_nodes_standard.csv', dtype={'name': str})
    links = pd.read_csv(net_dir / 'bloomington_links_standard.csv', dtype={'start': str, 'end': str})
    demand = pd.read_csv(net_dir / 'bloomington_demand_standard.csv', dtype={'orig': str, 'dest': str})
    with open(net_dir / 'bloomington_existing_routes.json') as f:
        routes = json.load(f)
    coords = {str(row['name']): (row['x'], row['y']) for _, row in nodes.iterrows()}
    origin_demand = demand.groupby('orig')['volume'].sum()
    dest_demand = demand.groupby('dest')['volume'].sum()
    return {
        'nodes': nodes, 'links': links, 'demand': demand, 'routes': routes,
        'coords': coords, 'origin_demand': origin_demand, 'dest_demand': dest_demand,
        'top_origin': origin_demand.idxmax(), 'top_dest': dest_demand.idxmax(),
        'transit_center': '96',
    }


def _bloom_panel_network(ax, d):
    """Road network graph over satellite map background.

    Plots 143 nodes as blue dots and 243 edges as colored lines
    on top of OpenStreetMap tiles via contextily.
    """
    # Black network structure with sky-blue node fill.
    edge_color = '#111111'
    node_color = '#87CEEB'
    node_edge = '#4285F4'
    _draw_edges(ax, d['links'], d['coords'], alpha=0.82, color=edge_color, lw=1.05)
    xs = [d['coords'][n][0] for n in d['coords']]
    ys = [d['coords'][n][1] for n in d['coords']]
    ax.scatter(xs, ys, s=24, color=node_color, alpha=0.95,
               edgecolors=node_edge, linewidths=0.40, zorder=2)
    _set_bounds(ax, d['coords'])
    legend_els = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=node_color, markeredgecolor=node_edge,
               markeredgewidth=0.8, markersize=6.5,
               label=f"{len(d['coords'])} nodes"),
        Line2D([0], [0], color=edge_color, linewidth=1.4,
               label=f"{len(d['links'])} edges"),
    ]
    ax.legend(handles=legend_els, loc='upper right', fontsize=FS - 3,
              framealpha=0.9, edgecolor='#CCCCCC', handletextpad=0.5)


def _bloom_panel_demand(ax, d):
    """Trip demand as KDE contours over satellite background.

    Origin density shown as blue filled contours, destination density as
    red filled contours, computed via scipy gaussian_kde.
    """
    from scipy.stats import gaussian_kde

    all_x = [d['coords'][n][0] for n in d['coords']]
    all_y = [d['coords'][n][1] for n in d['coords']]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    X, Y = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, 200),
        np.linspace(y_min - pad_y, y_max + pad_y, 200),
    )
    grid_pts = np.vstack([X.ravel(), Y.ravel()])

    # Origins KDE
    ox = [d['coords'][n][0] for n in d['coords'] if d['origin_demand'].get(n, 0) > 0]
    oy = [d['coords'][n][1] for n in d['coords'] if d['origin_demand'].get(n, 0) > 0]
    ow = [d['origin_demand'][n] for n in d['coords'] if d['origin_demand'].get(n, 0) > 0]
    levels = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    # Keep the KDE visible, but lighter so the demand markers remain legible.
    fill_opacities = [0.04, 0.03, 0.022, 0.015, 0.011, 0.008, 0.0045, 0.0025]

    if len(ox) > 2:
        kde_o = gaussian_kde(np.vstack([ox, oy]), weights=ow, bw_method=0.08)
        Z_o = kde_o(grid_pts).reshape(X.shape)
        Z_o_norm = Z_o / Z_o.max() if Z_o.max() > 0 else Z_o
        lw_o = [2.25, 1.8, 1.35, 1.08, 0.9, 0.72, 0.54, 0.36]
        ax.contour(X, Y, Z_o_norm, levels=levels,
                   colors=['#00BFFF'], linewidths=lw_o, alpha=0.58, zorder=4)
        for i, lvl in enumerate(levels):
            ax.contourf(X, Y, Z_o_norm, levels=[lvl, 10.0],
                        colors=['#00BFFF'], alpha=fill_opacities[i], zorder=i + 2)

    # Destinations KDE
    dx = [d['coords'][n][0] for n in d['coords'] if d['dest_demand'].get(n, 0) > 0]
    dy = [d['coords'][n][1] for n in d['coords'] if d['dest_demand'].get(n, 0) > 0]
    dw = [d['dest_demand'][n] for n in d['coords'] if d['dest_demand'].get(n, 0) > 0]
    if len(dx) > 2:
        kde_d = gaussian_kde(np.vstack([dx, dy]), weights=dw, bw_method=0.08)
        Z_d = kde_d(grid_pts).reshape(X.shape)
        Z_d_norm = Z_d / Z_d.max() if Z_d.max() > 0 else Z_d
        lw_d = [2.0, 1.6, 1.2, 0.96, 0.8, 0.64, 0.48, 0.32]
        ax.contour(X, Y, Z_d_norm, levels=levels,
                   colors=['#FF4400'], linewidths=lw_d, alpha=0.58, zorder=10)
        for i, lvl in enumerate(levels):
            ax.contourf(X, Y, Z_d_norm, levels=[lvl, 10.0],
                        colors=['#FF4400'], alpha=fill_opacities[i], zorder=i + 10)

    # Use geometric markers here so the marked location is the marker center.
    if d['top_origin'] in d['coords']:
        ox, oy = d['coords'][d['top_origin']]
        ax.scatter(ox, oy, s=170, marker='D', c='#4285F4',
                   edgecolors='white', linewidths=1.6, zorder=20)
    if d['top_dest'] in d['coords']:
        ddx, ddy = d['coords'][d['top_dest']]
        ax.scatter(ddx, ddy, s=340, marker='*', c='#EA4335',
                   edgecolors='white', linewidths=1.0, zorder=21)

    _set_bounds(ax, d['coords'])
    legend_els = [
        Line2D([0], [0], color='#00BFFF', linewidth=3, alpha=0.8, label='Origins'),
        Line2D([0], [0], color='#FF4400', linewidth=3, alpha=0.8, label='Destinations'),
    ]
    ax.legend(handles=legend_els, loc='upper right', fontsize=FS - 3,
              framealpha=0.9, edgecolor='#CCCCCC', handletextpad=0.4)


def _bloom_panel_routes(ax, d):
    """Existing 16 Bloomington transit routes over satellite background.

    Each route drawn as a colored line with shadow for depth.
    Route short-names placed near the end of each route with white
    background bbox for readability.
    """
    for idx, route in enumerate(d['routes']):
        route_nodes = [str(n) for n in route['nodes']]
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        xs, ys = [], []
        for n in route_nodes:
            if n in d['coords']:
                xs.append(d['coords'][n][0])
                ys.append(d['coords'][n][1])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color='white', linewidth=4.5, alpha=0.7,
                solid_capstyle='round', solid_joinstyle='round', zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2.5, alpha=1.0,
                solid_capstyle='round', solid_joinstyle='round', zorder=3)
        # Label near route end (95%) with white bbox
        label_idx = min(int(len(xs) * 0.95), len(xs) - 1)
        short = route.get('short_name', route['name'])
        ax.text(xs[label_idx], ys[label_idx], short,
                fontsize=FS - 3, fontweight='bold', color='black',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.9, edgecolor=color, linewidth=1.5),
                zorder=6)
    # Red pin at transit center
    _draw_pin(ax, d['coords'], d['transit_center'])
    _set_bounds(ax, d['coords'])


def _draw_pin(ax, coords, node_id):
    """Place a red pin at the given node."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    pin_path = Path(__file__).resolve().parent / 'pin_red.png'
    if not pin_path.exists() or node_id not in coords:
        return
    pin_img = plt.imread(pin_path)
    cx, cy = coords[node_id]
    ab = AnnotationBbox(
        OffsetImage(pin_img, zoom=0.132, resample=True),
        (cx, cy), frameon=False,
        box_alignment=(0.5, 0.0), pad=0, zorder=20)
    ax.add_artist(ab)


def plot_bloomington():
    """Generate the 3-panel Bloomington map figure with satellite background.

    Creates a wide (20x6) figure with panels for the road network,
    demand distribution, and existing transit routes, each with
    OpenStreetMap tile background. Saves to plots/bloomington_map.pdf.
    """
    import contextily as ctx
    from pyproj import Transformer

    d = _load_bloomington()

    # Convert UTM Zone 16N → Web Mercator for contextily
    transformer = Transformer.from_crs("EPSG:32616", "EPSG:3857", always_xy=True)
    coords_3857 = {}
    for node_id, (x, y) in d['coords'].items():
        mx, my = transformer.transform(x, y)
        coords_3857[node_id] = (mx, my)
    d_wm = {**d, 'coords': coords_3857}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    _bloom_panel_network(axes[0], d_wm)
    _bloom_panel_demand(axes[1], d_wm)
    _bloom_panel_routes(axes[2], d_wm)

    # Add map tiles to each panel
    for ax in axes:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.05)
    outpath = OUTDIR / 'bloomington_map'
    fig.savefig(str(outpath) + '.pdf')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


# ===========================================================================
# Laval: single-panel network figure
# ===========================================================================

def _load_laval():
    """Load Laval network data: nodes, links, demand, degree, and coordinates.

    Returns a dict with keys: nodes, links, coords, degree, node_demand,
    hub_node.
    """
    net_dir = ROOT / 'networks' / 'laval'
    nodes = pd.read_csv(net_dir / 'laval_nodes_standard.csv')
    links = pd.read_csv(net_dir / 'laval_links_standard.csv')
    demand = pd.read_csv(net_dir / 'laval_demand_standard.csv')
    coords = {row['name']: (row['x'], row['y']) for _, row in nodes.iterrows()}
    degree = {}
    for _, row in links.iterrows():
        s, e = row['start'], row['end']
        degree[s] = degree.get(s, 0) + 1
        degree[e] = degree.get(e, 0) + 1
    node_demand = {}
    for _, row in demand.iterrows():
        o, d = row['orig'], row['dest']
        node_demand[o] = node_demand.get(o, 0) + row['volume']
        node_demand[d] = node_demand.get(d, 0) + row['volume']
    return {
        'nodes': nodes, 'links': links, 'coords': coords,
        'degree': degree, 'node_demand': node_demand, 'hub_node': 542,
    }


def plot_laval():
    """Generate the Laval network visualization and save as PDF.

    Single-panel figure (10x8): 632 nodes colored by demand (YlOrRd, log scale),
    sized by degree. Hub node 542 highlighted with a diamond marker.
    Colorbar shows demand intensity. Saves to plots/laval_network.pdf.
    """
    d = _load_laval()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw links
    for _, row in d['links'].iterrows():
        s, e = row['start'], row['end']
        if s in d['coords'] and e in d['coords']:
            x0, y0 = d['coords'][s]
            x1, y1 = d['coords'][e]
            ax.plot([x0, x1], [y0, y1], color='#CCCCCC', linewidth=0.3, zorder=1)

    # Prepare node arrays
    node_names = list(d['coords'].keys())
    xs = np.array([d['coords'][n][0] for n in node_names])
    ys = np.array([d['coords'][n][1] for n in node_names])
    degs = np.array([d['degree'].get(n, 1) for n in node_names])
    demands = np.array([d['node_demand'].get(n, 0) for n in node_names])

    # Size by degree, color by demand (log scale)
    sizes = 3 + (degs - degs.min()) / (degs.max() - degs.min()) * 40
    log_demands = np.log1p(demands)
    norm = mcolors.Normalize(vmin=log_demands.min(), vmax=log_demands.max())

    scatter = ax.scatter(xs, ys, s=sizes, c=log_demands, cmap='YlOrRd',
                         norm=norm, edgecolors='none', alpha=0.8, zorder=2)

    # Highlight hub node
    hub_x, hub_y = d['coords'][d['hub_node']]
    ax.scatter([hub_x], [hub_y], s=80, c='#1A1A1A', edgecolors='white',
               linewidths=0.8, zorder=4, marker='D')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(r'Demand (log scale)', fontsize=FS - 2)
    cbar.ax.tick_params(labelsize=FS - 4)

    ax.set_xlabel(r'Easting (m)')
    ax.set_ylabel(r'Northing (m)')
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=True, left=True)

    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#1A1A1A',
               markeredgecolor='white', markersize=8,
               label=rf'Transit center (node {d["hub_node"]})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
              fontsize=FS - 4, edgecolor='#CCCCCC')

    fig.tight_layout()
    outpath = OUTDIR / 'laval_network'
    fig.savefig(str(outpath) + '.pdf')
    print(f'Saved {outpath}.pdf')
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Generate all network visualization figures.

    Calls plot_bloomington() and plot_laval() to produce both PDFs.
    """
    plot_bloomington()
    plot_laval()


if __name__ == '__main__':
    main()

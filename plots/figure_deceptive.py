"""
Deceptive landscape phenomenon: routes accumulate on the same corridors.
GIF builds up route-by-route, showing edge-usage heatmap intensifying
as each individually 'reasonable' route piles onto the same area.

Usage:
    python plots/figure_deceptive.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from collections import Counter
from pathlib import Path
from PIL import Image
import io

FS = 14
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': FS,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'figure.dpi': 150,
})

OUTDIR = Path(__file__).resolve().parent
DATADIR = OUTDIR / 'data' / 'network'
ASSETS_DIR = OUTDIR / 'assets'


def load_network():
    nodes = pd.read_csv(DATADIR / 'bloomington_nodes_standard.csv', dtype={'name': str})
    links = pd.read_csv(DATADIR / 'bloomington_links_standard.csv', dtype={'start': str, 'end': str})
    coords = {str(row['name']): (row['x'], row['y']) for _, row in nodes.iterrows()}
    return links, coords


def load_routes(filename):
    with open(DATADIR / filename) as f:
        data = json.load(f)
    routes = []
    if isinstance(data, dict):
        for _, route_nodes in sorted(data.items()):
            routes.append([str(n) for n in route_nodes])
    return routes


def get_edge_usage(routes):
    usage = Counter()
    for route in routes:
        for i in range(len(route) - 1):
            edge = tuple(sorted([route[i], route[i+1]]))
            usage[edge] += 1
    return usage


def to_webmercator(coords):
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:32616", "EPSG:3857", always_xy=True)
        return {nid: transformer.transform(x, y) for nid, (x, y) in coords.items()}
    except ImportError:
        return coords


def render_frame(links, coords_wm, routes_so_far, n_route, total_routes,
                 xlim, ylim, use_tiles, tile_cache):
    """Render one frame with routes_so_far overlaid as edge heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7.5))

    edge_usage = get_edge_usage(routes_so_far)
    covered_edges = set(edge_usage.keys())

    # Uncovered network edges
    for _, row in links.iterrows():
        s, e = row['start'], row['end']
        edge_key = tuple(sorted([s, e]))
        if edge_key in covered_edges:
            continue
        p1, p2 = coords_wm.get(s), coords_wm.get(e)
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='#d1d5db', alpha=0.3, linewidth=0.5,
                    solid_capstyle='round', zorder=0)

    # Covered edges — heatmap by usage
    cmap = plt.cm.YlOrRd
    for edge, count in edge_usage.items():
        n1, n2 = edge
        p1, p2 = coords_wm.get(n1), coords_wm.get(n2)
        if not p1 or not p2:
            continue
        t = min(count / 6.0, 1.0)
        color = cmap(0.15 + 0.85 * t)
        lw = 1.8 + 3.5 * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, alpha=0.9, linewidth=lw,
                solid_capstyle='round', zorder=2 + count)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if use_tiles:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.4)

    # Pin
    pin_path = ASSETS_DIR / 'pin_red.png'
    if pin_path.exists() and '96' in coords_wm:
        pin_img = plt.imread(pin_path)
        cx, cy = coords_wm['96']
        ab = AnnotationBbox(
            OffsetImage(pin_img, zoom=0.1, resample=True),
            (cx, cy), frameon=False,
            box_alignment=(0.5, 0.0), pad=0, zorder=20)
        ax.add_artist(ab)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color('#CCCCCC')

    # Route counter
    ax.text(0.98, 0.02, f'{n_route}/{total_routes} routes',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FS - 1, color='#888888')

    # Colorbar
    if n_route > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=mcolors.Normalize(vmin=1, vmax=6))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01, aspect=25)
        cbar.set_ticks([1, 2, 3, 4, 5, 6])
        cbar.ax.tick_params(labelsize=FS - 4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#FFFFFF',
                bbox_inches='tight', pad_inches=0.05, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    links, coords = load_network()

    try:
        import contextily, pyproj
        use_tiles = True
        print("Using OSM tiles")
    except ImportError:
        use_tiles = False

    coords_wm = to_webmercator(coords)

    # Compute bounds once
    xs = [c[0] for c in coords_wm.values()]
    ys = [c[1] for c in coords_wm.values()]
    xpad = (max(xs) - min(xs)) * 0.03
    ypad = (max(ys) - min(ys)) * 0.03
    xlim = (min(xs) - xpad, max(xs) + xpad)
    ylim = (min(ys) - ypad, max(ys) + ypad)

    routes = load_routes('routes_demand_cover.json')
    total = len(routes)

    frames = []
    # Key frames: after route 1, 4, 8, 12, 16
    key_indices = [1, 4, 8, 12, 16]

    for n in key_indices:
        if n > total:
            break
        print(f"  Rendering {n}/{total} routes...")
        frame = render_frame(links, coords_wm, routes[:n], n, total,
                            xlim, ylim, use_tiles, None)
        frames.append(frame)

    # Save GIF
    outpath = OUTDIR / '3.gif'
    durations = [1500, 1200, 1200, 1200, 2500]  # Linger on first and last
    frames[0].save(
        outpath, save_all=True, append_images=frames[1:],
        duration=durations[:len(frames)], loop=0,
    )
    print(f"Saved {outpath} ({len(frames)} frames)")

    # Also save final frame as PNG
    frames[-1].save(OUTDIR / '3.png')
    print(f"Saved {OUTDIR / '3.png'}")


if __name__ == '__main__':
    main()

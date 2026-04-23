"""
Animated GIF: 5 different transit route designs on Bloomington map background.
Cycles through methods with smooth transitions. Self-contained in plots/.

Usage:
    python plots/figure_route_gif.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pathlib import Path
from PIL import Image
import io

FS = 16
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

# Route color palettes — each method gets its own accent color
METHOD_COLORS = {
    'Random Walk':   '#6366f1',
    'Demand Cover':  '#0ea5e9',
    'Genetic Alg.':  '#fb7185',
    'End-to-End RL': '#1d4ed8',
    'Pure MCTS':     '#ef4444',
}

# Per-route colors (16 distinct)
ROUTE_COLORS = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075',
]

METHODS = [
    ('Random Walk',   'routes_random_walk.json'),
    ('Demand Cover',  'routes_demand_cover.json'),
    ('Genetic Alg.',  'routes_genetic.json'),
    ('End-to-End RL', 'routes_ppo.json'),
    ('Pure MCTS',     'routes_pure_mcts.json'),
]

ROUTE_LW = 2.8
ROUTE_ALPHA = 0.92
SHADOW_LW = 0
SHADOW_ALPHA = 0
EDGE_COLOR = '#999999'
EDGE_ALPHA = 0.35
EDGE_LW = 0.7


def load_network():
    nodes = pd.read_csv(DATADIR / 'bloomington_nodes_standard.csv', dtype={'name': str})
    links = pd.read_csv(DATADIR / 'bloomington_links_standard.csv', dtype={'start': str, 'end': str})
    coords = {str(row['name']): (row['x'], row['y']) for _, row in nodes.iterrows()}
    return links, coords


def load_routes(filename):
    path = DATADIR / filename
    with open(path) as f:
        data = json.load(f)
    routes = []
    if isinstance(data, dict):
        for _, nodes in sorted(data.items()):
            routes.append([str(n) for n in nodes])
    return routes


def draw_basemap(ax, links, coords):
    """Draw street network as light edges."""
    for _, row in links.iterrows():
        s = coords.get(row['start'])
        e = coords.get(row['end'])
        if s and e:
            ax.plot([s[0], e[0]], [s[1], e[1]],
                    color=EDGE_COLOR, alpha=EDGE_ALPHA, linewidth=EDGE_LW,
                    solid_capstyle='round', zorder=0)


def draw_basemap_with_tiles(ax, links, coords):
    """Draw street network with OpenStreetMap background."""
    try:
        import contextily as ctx
        from pyproj import Transformer

        # Convert UTM Zone 16N → Web Mercator
        transformer = Transformer.from_crs("EPSG:32616", "EPSG:3857", always_xy=True)
        coords_wm = {}
        for nid, (x, y) in coords.items():
            mx, my = transformer.transform(x, y)
            coords_wm[nid] = (mx, my)

        # Draw edges in web mercator
        for _, row in links.iterrows():
            s = coords_wm.get(row['start'])
            e = coords_wm.get(row['end'])
            if s and e:
                ax.plot([s[0], e[0]], [s[1], e[1]],
                        color='#555555', alpha=0.4, linewidth=0.8,
                        solid_capstyle='round', zorder=1)

        # Set bounds
        xs = [c[0] for c in coords_wm.values()]
        ys = [c[1] for c in coords_wm.values()]
        xpad = (max(xs) - min(xs)) * 0.03
        ypad = (max(ys) - min(ys)) * 0.03
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        return coords_wm
    except ImportError:
        # Fallback to plain basemap
        draw_basemap(ax, links, coords)
        return coords


def draw_routes(ax, routes, coords, method_color=None):
    """Draw all routes with per-route colors + shadow."""
    for idx, route in enumerate(routes):
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        xs, ys = [], []
        for node in route:
            pt = coords.get(node)
            if pt:
                xs.append(pt[0])
                ys.append(pt[1])
        if len(xs) < 2:
            continue
        # Route
        ax.plot(xs, ys, color=color, linewidth=ROUTE_LW,
                alpha=ROUTE_ALPHA, solid_capstyle='round',
                solid_joinstyle='round', zorder=3)


def draw_pin(ax, coords, node_id='96'):
    """Draw red pin at transit center."""
    pin_path = ASSETS_DIR / 'pin_red.png'
    if not pin_path.exists() or node_id not in coords:
        return
    pin_img = plt.imread(pin_path)
    cx, cy = coords[node_id]
    ab = AnnotationBbox(
        OffsetImage(pin_img, zoom=0.13, resample=True),
        (cx, cy), frameon=False,
        box_alignment=(0.5, 0.0), pad=0, zorder=20)
    ax.add_artist(ab)


def render_frame(links, coords, method_name, routes, use_tiles=True):
    """Render one frame and return as PIL Image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    if use_tiles:
        coords_plot = draw_basemap_with_tiles(ax, links, coords)
    else:
        draw_basemap(ax, links, coords)
        coords_plot = coords
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        xpad = (max(xs) - min(xs)) * 0.03
        ypad = (max(ys) - min(ys)) * 0.03
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    draw_routes(ax, routes, coords_plot)
    draw_pin(ax, coords_plot)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color('#CCCCCC')

    # No labels — just the routes

    # Render to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#FFFFFF',
                bbox_inches='tight', pad_inches=0.05, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    links, coords = load_network()

    # Try with tiles first, fall back to plain
    try:
        import contextily, pyproj
        use_tiles = True
        print("Using OpenStreetMap background")
    except ImportError:
        use_tiles = False
        print("contextily/pyproj not available, using plain basemap")

    frames = []
    for method_name, filename in METHODS:
        print(f"  Rendering {method_name}...")
        routes = load_routes(filename)
        frame = render_frame(links, coords, method_name, routes, use_tiles=use_tiles)
        frames.append(frame)

    # Save GIF — each frame shown for 2 seconds
    outpath = OUTDIR / '2.gif'
    frames[0].save(
        outpath,
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0,
    )
    print(f"Saved {outpath} ({len(frames)} frames)")

    # Also save first frame as 2.png
    frames[0].save(OUTDIR / '2.png')
    print(f"Saved {OUTDIR / '2.png'}")


if __name__ == '__main__':
    main()

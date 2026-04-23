"""
Standalone Bloomington street network plot.

Usage:
    python plots/figure_network_panel.py
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FS = 20
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': FS,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = REPO_ROOT / "networks" / "bloomington"
OUTDIR = Path(__file__).resolve().parent

BASE_EDGE_COLOR = "#d1d5db"
BASE_EDGE_ALPHA = 0.45
BASE_EDGE_WIDTH = 0.9
MARGIN_RATIO = 0.03


def load_data():
    nodes = pd.read_csv(NETWORKS_DIR / "bloomington_nodes_standard.csv", dtype={"name": str})
    links = pd.read_csv(NETWORKS_DIR / "bloomington_links_standard.csv", dtype={"start": str, "end": str})
    coords = nodes.set_index("name")[["x", "y"]].to_dict("index")
    return nodes, links, coords


def draw_basemap(ax, links, coords):
    for _, row in links.iterrows():
        start = coords.get(row["start"])
        end = coords.get(row["end"])
        if start is None or end is None:
            continue
        ax.plot(
            [start["x"], end["x"]], [start["y"], end["y"]],
            color=BASE_EDGE_COLOR, alpha=BASE_EDGE_ALPHA,
            linewidth=BASE_EDGE_WIDTH, solid_capstyle="round", zorder=0,
        )


def main():
    nodes, links, coords = load_data()

    all_xs = [c["x"] for c in coords.values()]
    all_ys = [c["y"] for c in coords.values()]
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    mx = max((x_max - x_min) * MARGIN_RATIO, 1.0)
    my = max((y_max - y_min) * MARGIN_RATIO, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    draw_basemap(ax, links, coords)

    ax.set_xlim(x_min - mx, x_max + mx)
    ax.set_ylim(y_min - my, y_max + my)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    outpath = OUTDIR / "1"
    fig.savefig(str(outpath) + ".png", facecolor="#FFFFFF")
    fig.savefig(str(outpath) + ".pdf", facecolor="#FFFFFF")
    print(f"Saved {outpath}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

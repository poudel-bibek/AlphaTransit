from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    from .common import ASSETS_DIR, FIGURES_DIR, NETWORKS_DIR, ROUTE_COLORS, apply_plot_style, maybe_to_web_mercator, save_figure
except ImportError:
    from common import ASSETS_DIR, FIGURES_DIR, NETWORKS_DIR, ROUTE_COLORS, apply_plot_style, maybe_to_web_mercator, save_figure


FS = 14


def _pin_image() -> np.ndarray | None:
    pin_path = ASSETS_DIR / "pin_red.png"
    return plt.imread(pin_path) if pin_path.exists() else None


def _draw_edges(ax: plt.Axes, links_df: pd.DataFrame, coords: dict, alpha: float = 0.5, color: str = "#999999", lw: float = 0.6) -> None:
    for _, row in links_df.iterrows():
        start = str(row["start"])
        end = str(row["end"])
        if start not in coords or end not in coords:
            continue
        x0, y0 = coords[start]
        x1, y1 = coords[end]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha, zorder=1, solid_capstyle="round")


def _draw_pin(ax: plt.Axes, coords: dict, node_id: str, zoom: float = 0.12) -> None:
    pin = _pin_image()
    if pin is None or node_id not in coords:
        return
    cx, cy = coords[node_id]
    artist = AnnotationBbox(
        OffsetImage(pin, zoom=zoom, resample=True),
        (cx, cy),
        frameon=False,
        box_alignment=(0.5, 0.0),
        pad=0,
        zorder=20,
    )
    ax.add_artist(artist)


def _set_bounds(ax: plt.Axes, coords: dict) -> None:
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    xpad = (max(xs) - min(xs)) * 0.03
    ypad = (max(ys) - min(ys)) * 0.03
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("#CCCCCC")


def load_bloomington() -> dict:
    net_dir = NETWORKS_DIR / "bloomington"
    nodes = pd.read_csv(net_dir / "bloomington_nodes_standard.csv", dtype={"name": str})
    links = pd.read_csv(net_dir / "bloomington_links_standard.csv", dtype={"start": str, "end": str})
    demand = pd.read_csv(net_dir / "bloomington_demand_standard.csv", dtype={"orig": str, "dest": str})
    with open(net_dir / "bloomington_existing_routes.json", "r", encoding="utf-8") as fp:
        routes = json.load(fp)
    coords = {str(row["name"]): (row["x"], row["y"]) for _, row in nodes.iterrows()}
    origin_demand = demand.groupby("orig")["volume"].sum()
    dest_demand = demand.groupby("dest")["volume"].sum()
    return {
        "nodes": nodes,
        "links": links,
        "demand": demand,
        "routes": routes,
        "coords": coords,
        "origin_demand": origin_demand,
        "dest_demand": dest_demand,
        "top_origin": origin_demand.idxmax(),
        "top_dest": dest_demand.idxmax(),
        "transit_center": "96",
    }


def plot_network_panel(output_path: Path = FIGURES_DIR / "network_panel.pdf") -> None:
    apply_plot_style(FS)
    data = load_bloomington()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    _draw_edges(ax, data["links"], data["coords"], alpha=0.45, color="#d1d5db", lw=0.9)
    _set_bounds(ax, data["coords"])
    ax.axis("off")
    save_figure(fig, output_path, facecolor="#FFFFFF")
    plt.close(fig)


def _plot_bloomington_network(ax: plt.Axes, data: dict) -> None:
    edge_color = "#111111"
    node_color = "#87CEEB"
    node_edge = "#4285F4"
    _draw_edges(ax, data["links"], data["coords"], alpha=0.82, color=edge_color, lw=1.05)
    xs = [data["coords"][node][0] for node in data["coords"]]
    ys = [data["coords"][node][1] for node in data["coords"]]
    ax.scatter(xs, ys, s=24, color=node_color, alpha=0.95, edgecolors=node_edge, linewidths=0.40, zorder=2)
    _set_bounds(ax, data["coords"])
    legend = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=node_color, markeredgecolor=node_edge, markeredgewidth=0.8, markersize=6.5, label=f"{len(data['coords'])} nodes"),
        Line2D([0], [0], color=edge_color, linewidth=1.4, label=f"{len(data['links'])} edges"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=FS - 3, framealpha=0.9, edgecolor="#CCCCCC", handletextpad=0.5)


def _plot_bloomington_demand(ax: plt.Axes, data: dict) -> None:
    from scipy.stats import gaussian_kde

    xs_all = [data["coords"][node][0] for node in data["coords"]]
    ys_all = [data["coords"][node][1] for node in data["coords"]]
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    grid_x, grid_y = np.meshgrid(np.linspace(x_min - pad_x, x_max + pad_x, 200), np.linspace(y_min - pad_y, y_max + pad_y, 200))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])

    levels = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    fill_opacities = [0.04, 0.03, 0.022, 0.015, 0.011, 0.008, 0.0045, 0.0025]

    origin_nodes = [node for node in data["coords"] if data["origin_demand"].get(node, 0) > 0]
    if len(origin_nodes) > 2:
        ox = [data["coords"][node][0] for node in origin_nodes]
        oy = [data["coords"][node][1] for node in origin_nodes]
        ow = [data["origin_demand"][node] for node in origin_nodes]
        origin_kde = gaussian_kde(np.vstack([ox, oy]), weights=ow, bw_method=0.08)
        origin_grid = origin_kde(grid_points).reshape(grid_x.shape)
        origin_grid = origin_grid / origin_grid.max() if origin_grid.max() > 0 else origin_grid
        ax.contour(grid_x, grid_y, origin_grid, levels=levels, colors=["#00BFFF"], linewidths=[2.25, 1.8, 1.35, 1.08, 0.9, 0.72, 0.54, 0.36], alpha=0.58, zorder=4)
        for idx, level in enumerate(levels):
            ax.contourf(grid_x, grid_y, origin_grid, levels=[level, 10.0], colors=["#00BFFF"], alpha=fill_opacities[idx], zorder=idx + 2)

    destination_nodes = [node for node in data["coords"] if data["dest_demand"].get(node, 0) > 0]
    if len(destination_nodes) > 2:
        dx = [data["coords"][node][0] for node in destination_nodes]
        dy = [data["coords"][node][1] for node in destination_nodes]
        dw = [data["dest_demand"][node] for node in destination_nodes]
        destination_kde = gaussian_kde(np.vstack([dx, dy]), weights=dw, bw_method=0.08)
        destination_grid = destination_kde(grid_points).reshape(grid_x.shape)
        destination_grid = destination_grid / destination_grid.max() if destination_grid.max() > 0 else destination_grid
        ax.contour(grid_x, grid_y, destination_grid, levels=levels, colors=["#FF4400"], linewidths=[2.0, 1.6, 1.2, 0.96, 0.8, 0.64, 0.48, 0.32], alpha=0.58, zorder=10)
        for idx, level in enumerate(levels):
            ax.contourf(grid_x, grid_y, destination_grid, levels=[level, 10.0], colors=["#FF4400"], alpha=fill_opacities[idx], zorder=idx + 10)

    if data["top_origin"] in data["coords"]:
        ox, oy = data["coords"][data["top_origin"]]
        ax.scatter(ox, oy, s=170, marker="D", c="#4285F4", edgecolors="white", linewidths=1.6, zorder=20)
    if data["top_dest"] in data["coords"]:
        dx, dy = data["coords"][data["top_dest"]]
        ax.scatter(dx, dy, s=340, marker="*", c="#EA4335", edgecolors="white", linewidths=1.0, zorder=21)

    _set_bounds(ax, data["coords"])
    legend = [
        Line2D([0], [0], color="#00BFFF", linewidth=3, alpha=0.8, label="Origins"),
        Line2D([0], [0], color="#FF4400", linewidth=3, alpha=0.8, label="Destinations"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=FS - 3, framealpha=0.9, edgecolor="#CCCCCC", handletextpad=0.4)


def _plot_bloomington_routes(ax: plt.Axes, data: dict) -> None:
    for idx, route in enumerate(data["routes"]):
        route_nodes = [str(node) for node in route["nodes"]]
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        xs, ys = [], []
        for node in route_nodes:
            if node not in data["coords"]:
                continue
            x, y = data["coords"][node]
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color="white", linewidth=4.5, alpha=0.7, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2.5, alpha=1.0, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        label_idx = min(int(len(xs) * 0.95), len(xs) - 1)
        short_name = route.get("short_name", route["name"])
        ax.text(
            xs[label_idx],
            ys[label_idx],
            short_name,
            fontsize=FS - 3,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=color, linewidth=1.5),
            zorder=6,
        )
    _draw_pin(ax, data["coords"], data["transit_center"], zoom=0.132)
    _set_bounds(ax, data["coords"])


def plot_bloomington_map(output_path: Path = FIGURES_DIR / "bloomington_map.pdf") -> None:
    apply_plot_style(FS)
    try:
        import contextily as ctx
    except ImportError as exc:
        raise RuntimeError("Bloomington map plotting requires contextily") from exc

    data = load_bloomington()
    wm_data = {**data, "coords": maybe_to_web_mercator(data["coords"])}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _plot_bloomington_network(axes[0], wm_data)
    _plot_bloomington_demand(axes[1], wm_data)
    _plot_bloomington_routes(axes[2], wm_data)

    for ax in axes:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.05)
    save_figure(fig, output_path)
    plt.close(fig)


def load_laval() -> dict:
    net_dir = NETWORKS_DIR / "laval"
    nodes = pd.read_csv(net_dir / "laval_nodes_standard.csv", dtype={"name": str})
    links = pd.read_csv(net_dir / "laval_links_standard.csv", dtype={"start": str, "end": str})
    demand = pd.read_csv(net_dir / "laval_demand_standard.csv", dtype={"orig": str, "dest": str})
    coords = {str(row["name"]): (row["x"], row["y"]) for _, row in nodes.iterrows()}

    degree = {}
    for _, row in links.iterrows():
        start = str(row["start"])
        end = str(row["end"])
        degree[start] = degree.get(start, 0) + 1
        degree[end] = degree.get(end, 0) + 1

    node_demand = {}
    for _, row in demand.iterrows():
        origin = str(row["orig"])
        destination = str(row["dest"])
        node_demand[origin] = node_demand.get(origin, 0) + row["volume"]
        node_demand[destination] = node_demand.get(destination, 0) + row["volume"]

    return {
        "nodes": nodes,
        "links": links,
        "coords": coords,
        "degree": degree,
        "node_demand": node_demand,
        "hub_node": "542",
    }


def plot_laval_network(output_path: Path = FIGURES_DIR / "laval_network.pdf") -> None:
    apply_plot_style(FS)
    data = load_laval()
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in data["links"].iterrows():
        start = row["start"]
        end = row["end"]
        if start in data["coords"] and end in data["coords"]:
            x0, y0 = data["coords"][start]
            x1, y1 = data["coords"][end]
            ax.plot([x0, x1], [y0, y1], color="#CCCCCC", linewidth=0.3, zorder=1)

    node_names = list(data["coords"].keys())
    xs = np.array([data["coords"][name][0] for name in node_names])
    ys = np.array([data["coords"][name][1] for name in node_names])
    degrees = np.array([data["degree"].get(name, 1) for name in node_names])
    demands = np.array([data["node_demand"].get(name, 0) for name in node_names])

    sizes = 3 + (degrees - degrees.min()) / (degrees.max() - degrees.min()) * 40
    log_demands = np.log1p(demands)
    scatter = ax.scatter(xs, ys, s=sizes, c=log_demands, cmap="YlOrRd", norm=mcolors.Normalize(vmin=log_demands.min(), vmax=log_demands.max()), edgecolors="none", alpha=0.8, zorder=2)

    hub_x, hub_y = data["coords"][data["hub_node"]]
    ax.scatter([hub_x], [hub_y], s=80, c="#1A1A1A", edgecolors="white", linewidths=0.8, zorder=4, marker="D")

    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    colorbar.set_label(r"Demand (log scale)", fontsize=FS - 2)
    colorbar.ax.tick_params(labelsize=FS - 4)

    ax.set_xlabel(r"Easting (m)")
    ax.set_ylabel(r"Northing (m)")
    ax.set_aspect("equal")
    legend = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#1A1A1A", markeredgecolor="white", markersize=8, label=rf"Transit center (node {data['hub_node']})"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.9, fontsize=FS - 4, edgecolor="#CCCCCC")

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def generate_all(output_dir: Path = FIGURES_DIR) -> None:
    plot_bloomington_map(output_dir / "bloomington_map.pdf")
    plot_laval_network(output_dir / "laval_network.pdf")
    plot_network_panel(output_dir / "network_panel.pdf")

from typing import Dict, Tuple, Optional
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_network_and_demand(world, output_path: Optional[str] = None) -> None:
    """
    Plot network + OD demand in one panel.
    Line width scales with demand; repeated O→D entries are summed.
    Saves to `{world.name}_demand_network.png` if no path given.
    """

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # 1) Prepare node coordinates for scaling and later layers
    node_xs = [n.x for n in world.NODES]
    node_ys = [n.y for n in world.NODES]

    # 3) Aggregate demand volumes by O-D
    #    world.demand_info['adddemand'] records every call made by scenario building,
    #    e.g., adddemand(o="A", d="B", volume=10) or adddemand(o="A", d="B", flow=0.01, t_start=0, t_end=3600).
    #    We SUM repeats to produce a single combined volume per (origin, destination).
    #    Example:
    #      - adddemand(A,B, volume=10)
    #      - adddemand(A,B, volume=5)
    #      => aggregated(A,B) = 15
    #      - If volume not provided but flow and time window are, we convert to volume ≈ flow * (t_end - t_start).
    od_to_volume: Dict[Tuple[str, str], float] = defaultdict(float)
    entries = world.demand_info.get("adddemand", [])
    for e in entries:
        o = str(e.get("orig")); d = str(e.get("dest"))
        ts = e.get("t_start"); te = e.get("t_end")
        flow = e.get("flow", -1); vol = e.get("volume", -1)

        # Prefer explicit volume; otherwise derive from flow over the time window.
        value = 0.0
        if isinstance(vol, (int, float)) and vol not in (-1, None):
            value = float(vol)
        elif isinstance(flow, (int, float)) and flow not in (-1, None) and isinstance(ts, (int, float)) and isinstance(te, (int, float)):
            value = max(0.0, float(flow) * (float(te) - float(ts)))

        if o != d and value > 0:
            od_to_volume[(o, d)] += value

    # 4) Arrowhead scaling (CRITICAL)
    #    Matplotlib's ax.arrow sizes head_width/head_length in DATA UNITS, not pixels.
    #    If coordinates are latitude/longitude (range ~10-100), fixed sizes like 6/10 overflow,
    #    producing a giant filled blob. We therefore scale arrowhead sizes to the plot extent.
    #    NOTE: As requested, demand arrow thickness is FIXED (not proportional to demand).
    max_val = max(od_to_volume.values()) if od_to_volume else 0.0  # kept for completeness, not used for width now
    if world.NODES:
        xs = node_xs; ys = node_ys
        x_range = (max(xs) - min(xs)) or 1.0
        y_range = (max(ys) - min(ys)) or 1.0
        extent = max(x_range, y_range)
    else:
        extent = 1.0
    head_w = 0.015 * extent   # smaller heads to reduce clutter
    head_l = 0.02 * extent

    # 2) Demand arrows FIRST, light and thin, behind network layers
    for (o, d), vol in od_to_volume.items():
        no = world.get_node(o); nd = world.get_node(d)
        dx, dy = nd.x - no.x, nd.y - no.y
        sx, sy = no.x + 0.1 * dx, no.y + 0.1 * dy
        ex, ey = no.x + 0.9 * dx, no.y + 0.9 * dy

        # Fixed linewidth for demand arrows (no per-demand scaling)
        lw = 0.6
        ax.arrow(sx, sy, ex - sx, ey - sy,
                 head_width=head_w, head_length=head_l,
                 fc="#2E86AB", ec="#2E86AB", alpha=0.12,
                 length_includes_head=True, lw=lw, zorder=1)

    for link in world.LINKS:
        x1, y1 = link.start_node.x, link.start_node.y
        x2, y2 = link.end_node.x, link.end_node.y
        ax.plot([x1, x2], [y1, y2], c="#000000", lw=2.4, solid_capstyle="round", zorder=3)

    ax.scatter(node_xs, node_ys, s=320, c="#F5F6FA", edgecolors="#000000", linewidths=1.5, zorder=4)
    for n in world.NODES:
        ax.text(n.x, n.y, str(n.name), ha="center", va="center", fontsize=10, color="#000000", zorder=5)

    # Smaller main title (as figure title) and a subtitle on the axes
    fig.suptitle(f"{world.name}: Network + Demand", fontsize=14)
    ax.set_title(f"Nodes: {len(world.NODES)}   Links: {len(world.LINKS)}", fontsize=11, color="#4C566A")
    ax.set_aspect("equal")
    # Clean axes: no ticks, labels, or grid lines
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    # Default filename if not provided
    if output_path is None:
        output_path = f"{world.name}_demand_network.png"

    plt.savefig(output_path)
    plt.close(fig)


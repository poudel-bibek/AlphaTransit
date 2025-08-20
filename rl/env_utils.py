from typing import Dict, Tuple, Optional, Any, List
import pandas as pd
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def pretty_print_state(self: Any, state: Dict[str, Any], max_nodes: Optional[int] = None, max_edges: Optional[int] = None) -> None:
    """
    Print observation fields in a readable format.
    Shows shapes, samples, and mappings.
    """
    # Node and edge counts
    num_nodes_obs = state["node_features"].shape[0]
    num_edges_obs = state["edge_index"].shape[1]
    resolved_max_nodes = num_nodes_obs if max_nodes is None else max_nodes
    resolved_max_edges = num_edges_obs if max_edges is None else max_edges

    # Steps
    steps_left = float(state["steps_left"][0])
    steps_taken = len(self.current_path) - 1
    print(f"\n=== State Summary ===")
    print(f"Current path: {self.current_path}")
    print(f"Steps taken / max: {steps_taken} / {self.MAX_PATH_LENGTH} (steps_left_norm={steps_left:.3f})")

    # Node features
    feature_names = [
        "x_norm", "y_norm", "degree_norm", "d_out_norm", "d_in_norm",
        "d_out_path_norm", "d_in_path_norm", "in_path_flag", "is_valid_next"
    ]
    nf_df = pd.DataFrame(state["node_features"], index=self.node_list, columns=feature_names)
    with pd.option_context('display.max_rows', resolved_max_nodes, 'display.max_columns', 8, 'display.width', 120):
        print("\nNode features (sample):\n", nf_df.round(4))

    # Edges
    edge_index = state["edge_index"]
    edge_features = state["edge_features"]
    num_edges = edge_index.shape[1]
    print(f"\nEdges: {num_edges}")
    preview = min(resolved_max_edges, num_edges)
    rows = []
    for e in range(preview):
        src_idx = int(edge_index[0, e])
        dst_idx = int(edge_index[1, e])
        src = self.idx_to_node[src_idx]
        dst = self.idx_to_node[dst_idx]
        length_n, speed_n = [float(x) for x in edge_features[e]]
        rows.append({
            "src": src,
            "dst": dst,
            "length_norm": round(length_n, 4),
            "speed_norm": round(speed_n, 4),
        })
    ef_df = pd.DataFrame(rows)
    print("Edge samples:\n", ef_df)

def plot_network_and_demand(world, output_loc: str) -> None:
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

    plt.savefig(output_loc)
    plt.close(fig)


def plot_network_demand_and_path(world, current_path: List[str], output_loc: str) -> None:
    """
    Plot network + OD demand + path in one panel.
    Line width scales with demand; repeated O→D entries are summed.
    Current path highlighted in salmon red with proper layering.
    """

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # 1) Prepare node coordinates for scaling and later layers
    node_xs = [n.x for n in world.NODES]
    node_ys = [n.y for n in world.NODES]

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

    # 3) Network links (black)
    for link in world.LINKS:
        x1, y1 = link.start_node.x, link.start_node.y
        x2, y2 = link.end_node.x, link.end_node.y
        ax.plot([x1, x2], [y1, y2], c="#000000", lw=2.4, solid_capstyle="round", zorder=3)

    # 4) Plot regular nodes first
    ax.scatter(node_xs, node_ys, s=320, c="#F5F6FA", edgecolors="#000000", linewidths=1.5, zorder=4)
    
    # 5) Highlight current path if provided
    if current_path and len(current_path) > 1:
        # Draw path connections with thick salmon red lines
        for i in range(len(current_path) - 1):
            node1 = world.get_node(current_path[i])
            node2 = world.get_node(current_path[i + 1])
            if node1 and node2:
                ax.plot([node1.x, node2.x], [node1.y, node2.y], 
                       c="#FA8072", lw=4.5, solid_capstyle="round", zorder=6, alpha=0.8)
    
    # 6) Highlight path nodes with salmon red color
    if current_path:
        path_xs = []
        path_ys = []
        for node_name in current_path:
            node = world.get_node(node_name)
            if node:
                path_xs.append(node.x)
                path_ys.append(node.y)
        
        if path_xs:  # Only plot if we found valid nodes
            ax.scatter(path_xs, path_ys, s=380, c="#FA8072", edgecolors="#000000", 
                      linewidths=2.0, zorder=7, alpha=0.9)

    # 7) Add node labels on top of everything
    for n in world.NODES:
        ax.text(n.x, n.y, str(n.name), ha="center", va="center", fontsize=10, color="#000000", zorder=8)

    # Main title and path information
    fig.suptitle(f"{world.name}: Network + Demand", fontsize=16)
    
    # Add path text below the main title
    path_str = " → ".join(current_path) if current_path else "No path selected"
    fig.text(0.5, 0.92, f"Current Path: {path_str}", ha='center', fontsize=12, 
             color="#E74C3C" if current_path else "#666666", weight='bold')
    
    ax.set_title(f"Nodes: {len(world.NODES)}   Links: {len(world.LINKS)}", fontsize=11, color="#4C566A")
    ax.set_aspect("equal")
    # Clean axes: no ticks, labels, or grid lines
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    plt.savefig(output_loc)
    plt.close(fig)
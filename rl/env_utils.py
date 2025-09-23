import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Optional, Any, List, Tuple

def normalize_coordinates(world):
    """
    Normalize node coordinates to [0, 1] range.
    """
    nodes = list(world.NODES)
    node_xs = [n.x for n in nodes]
    node_ys = [n.y for n in nodes]

    x_min, x_max = min(node_xs), max(node_xs)
    y_min, y_max = min(node_ys), max(node_ys)

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Create normalized coordinate mappings
    node_to_norm_x = {node: (node.x - x_min) / x_range for node in nodes}
    node_to_norm_y = {node: (node.y - y_min) / y_range for node in nodes}

    return node_to_norm_x, node_to_norm_y

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
    route_progress = float(state["route_progress"][0])  # Current route progress
    steps_taken = len(self.current_route) - 1
    print(f"\n=== State Summary ===")
    print(f"Current route: {self.current_route}")
    print(f"Steps taken / max: {steps_taken} / {self.MAX_ROUTE_LENGTH} (route_progress={route_progress:.3f})")

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
    """

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Normalize coordinates using helper function
    node_to_norm_x, node_to_norm_y = normalize_coordinates(world)

    # Aggregate demand volumes
    od_to_volume = defaultdict(float)
    for e in world.demand_info.get("adddemand", []):
        o, d = str(e.get("orig")), str(e.get("dest"))
        vol = e.get("volume", 0)
        if vol and o != d:
            od_to_volume[(o, d)] += vol

    # Plot demand arrows
    for (o, d), vol in od_to_volume.items():
        no = world.get_node(o); nd = world.get_node(d)
        if no and nd:
            dx, dy = node_to_norm_x[nd] - node_to_norm_x[no], node_to_norm_y[nd] - node_to_norm_y[no]
            ax.arrow(node_to_norm_x[no] + 0.1*dx, node_to_norm_y[no] + 0.1*dy, 0.8*dx, 0.8*dy,
                    head_width=0.015, head_length=0.02,  # Scaled for [0,1] coordinates
                    fc="#2E86AB", ec="#2E86AB", alpha=0.2, lw=1.0, zorder=0)

    # Plot network links and nodes
    for link in world.LINKS:
        start_norm_x = node_to_norm_x[link.start_node]
        start_norm_y = node_to_norm_y[link.start_node]
        end_norm_x = node_to_norm_x[link.end_node]
        end_norm_y = node_to_norm_y[link.end_node]
        ax.plot([start_norm_x, end_norm_x], [start_norm_y, end_norm_y],
               c="#000000", lw=2.0, zorder=1)

    # Create scatter plot coordinates from normalized mappings
    node_xs_norm = [node_to_norm_x[n] for n in world.NODES]
    node_ys_norm = [node_to_norm_y[n] for n in world.NODES]
    ax.scatter(node_xs_norm, node_ys_norm, s=320, c="white", edgecolors="#000000", linewidths=2.0, zorder=2)
    for n in world.NODES:
        ax.text(node_to_norm_x[n], node_to_norm_y[n], str(n.name), ha="center", va="center", fontsize=10, color="#000000", zorder=4)

    fig.suptitle(f"{world.name}: Network + Demand", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_loc, dpi=150)
    plt.close(fig)


def plot_network_demand_and_path(world, routes: List[List[str]], output_loc: str, plot_demand: bool = False) -> None:
    """
    Plot network + OD demand + multiple routes.
    """

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Normalize coordinates using helper function
    node_to_norm_x, node_to_norm_y = normalize_coordinates(world)

    # Aggregate demand volumes
    od_to_volume = defaultdict(float)
    for e in world.demand_info.get("adddemand", []):
        o, d = str(e.get("orig")), str(e.get("dest"))
        vol = e.get("volume", 0)
        if vol and o != d:
            od_to_volume[(o, d)] += vol

    # Plot demand arrows (scaled for normalized coordinates)
    if plot_demand:
        for (o, d), vol in od_to_volume.items():
            no = world.get_node(o); nd = world.get_node(d)
            if no and nd:
                dx, dy = node_to_norm_x[nd] - node_to_norm_x[no], node_to_norm_y[nd] - node_to_norm_y[no]
                ax.arrow(node_to_norm_x[no] + 0.1*dx, node_to_norm_y[no] + 0.1*dy, 0.8*dx, 0.8*dy,
                        head_width=0.015, head_length=0.02,  # Scaled for [0,1] coordinates
                        fc="#2E86AB", ec="#2E86AB", alpha=0.2, lw=0.8, zorder=0)

    # Plot network links and nodes
    for link in world.LINKS:
        start_norm_x = node_to_norm_x[link.start_node]
        start_norm_y = node_to_norm_y[link.start_node]
        end_norm_x = node_to_norm_x[link.end_node]
        end_norm_y = node_to_norm_y[link.end_node]
        ax.plot([start_norm_x, end_norm_x], [start_norm_y, end_norm_y],
               c="#000000", lw=1.5, zorder=1)

    # Create scatter plot coordinates from normalized mappings
    node_xs_norm = [node_to_norm_x[n] for n in world.NODES]
    node_ys_norm = [node_to_norm_y[n] for n in world.NODES]

    # Define route colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Default black edges
    edge_colors = ["#000000"] * len(world.NODES)
    node_list = list(world.NODES)

    # Color starting node borders
    for i, route in enumerate(routes):
        if route:
            start_name = route[0]
            start_node = world.get_node(start_name)
            if start_node in node_list:
                idx = node_list.index(start_node)
                edge_colors[idx] = colors[i % len(colors)]

    ax.scatter(node_xs_norm, node_ys_norm, s=320, c="white", edgecolors=edge_colors, linewidths=2.0, zorder=2)

    # Add node labels
    for n in world.NODES:
        ax.text(node_to_norm_x[n], node_to_norm_y[n], str(n.name), ha="center", va="center", fontsize=10, color="#000000", zorder=4)

    # Two-pass route plotting system
    # Pass 1: Identify routes per link (bi-directional)
    link_to_routes = defaultdict(list)  # (start_node, end_node) -> list of (route_index, direction) tuples
    for i, route in enumerate(routes):
        if len(route) < 2:
            continue

        for j in range(len(route) - 1):
            start = world.get_node(route[j])
            end = world.get_node(route[j+1])
            link_to_routes[(start, end)].append(i)

    # Pass 2: Plot routes with perpendicular offsets
    for i, route in enumerate(routes):
        if len(route) < 2:
            continue

        color = colors[i % len(colors)]
        for j in range(len(route) - 1):
            start = world.get_node(route[j])
            end = world.get_node(route[j+1])

            # Get all routes using this link or its reverse
            forward_routes = link_to_routes[(start, end)]
            reverse_routes = link_to_routes[(end, start)]
            all_routes_on_link = set(forward_routes + reverse_routes)
            num_routes = len(all_routes_on_link)

            # Assign local offset index (0 to num_routes-1)
            local_index = sorted(all_routes_on_link).index(i)

            # Calculate direction vector
            dx = node_to_norm_x[end] - node_to_norm_x[start]
            dy = node_to_norm_y[end] - node_to_norm_y[start]
            length = (dx**2 + dy**2)**0.5

            # Standardize direction for consistent side: always from "lower" to "higher" node name
            if start.name > end.name:
                # Reverse direction for calculation
                dx = -dx
                dy = -dy

            # Perpendicular unit vector (rotate 90 degrees) - consistent direction
            perp_dx = -dy / length
            perp_dy = dx / length

            # Offset magnitude - always non-zero, increasing for multiple
            base_mag = 0.0065
            offset_mag = base_mag * (local_index + 1)  # Always positive offset, no zero

            offset_x_start = offset_mag * perp_dx
            offset_y_start = offset_mag * perp_dy
            offset_x_end = offset_mag * perp_dx
            offset_y_end = offset_mag * perp_dy

            # If we reversed, apply to correct points
            if start.name > end.name:
                offset_x_start, offset_x_end = offset_x_end, offset_x_start
                offset_y_start, offset_y_end = offset_y_end, offset_y_start

            ax.plot([node_to_norm_x[start] + offset_x_start, node_to_norm_x[end] + offset_x_end],
                    [node_to_norm_y[start] + offset_y_start, node_to_norm_y[end] + offset_y_end],
                    color=color, linewidth=1.5, alpha=0.9, zorder=1)

    # Add route information as a legend
    handles = []
    labels = []
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        route_text = f"Route {i}: {' → '.join(route)}"
        handle = plt.Line2D([0], [0], color=color, linewidth=2, label=route_text)
        handles.append(handle)
        labels.append(route_text)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=1,
                fontsize=10, frameon=False, handlelength=2.5)

    ax.set_title(f"{world.name}: Routes", fontsize=16, pad=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig(output_loc, bbox_inches='tight', dpi=150)
    plt.close(fig)
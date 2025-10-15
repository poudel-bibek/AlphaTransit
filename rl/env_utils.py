import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from typing import Any, Dict, Optional, List, Tuple, Sequence


def initialize_route(env: Any, avoid_completed_routes: bool = False) -> List[str]:
    """
    Select a route starting node according to the initialization strategy.
    - For Bloomington transit, the transit center node is 96.
    Args:
        env: TransitEnv instance supplying demand data and configuration flags.
        avoid_completed_routes: Whether to avoid nodes already used in completed routes.

    Returns:
        A single-element list containing the chosen starting node name (as string).
    """
    
    # Gather candidate nodes from demand dataframe
    all_nodes = list(env.demand_df_cached["orig"].unique())
    if avoid_completed_routes:
        # Flatten self.all_routes into a set of used nodes 
        completed_nodes = set(node for route in env.all_routes for node in route)
        choice_nodes = [node for node in all_nodes if node not in completed_nodes]
    else:
        choice_nodes = all_nodes

    strategy = env.path_init 
    if strategy not in {"random", "highest_demand", "transit_center"}:
        raise ValueError(f"Invalid path initialization strategy: {strategy}")

    elif strategy == "random":
        choice = random.choice(choice_nodes)
        print(f"Initializing route randomly at node: {choice}")
        return [choice]

    elif strategy == "highest_demand":
        # Rank all nodes by highest demand emanating from them (total volume leaving each node)
        demand_df_grouped = env.demand_df_cached.groupby("orig").sum(numeric_only=True).reset_index()
        demand_df_ranked = demand_df_grouped.sort_values("volume", ascending=False)
        for _, row in demand_df_ranked.iterrows():
            candidate_node = row["orig"]
            if candidate_node in choice_nodes:
                print(f"Initializing route at highest available demand node: {candidate_node}")
                return [candidate_node]

    elif strategy == "transit_center":
        print(f"Initializing route at transit center node: 96")
        return [96] # NOTE: This is a hardcoded value for Bloomington transit.

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
    fig, ax = plt.subplots(1, 1, figsize=(13.2, 9.6))

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


def plot_network_demand_and_path(world, routes: List[List[str]], output_loc: str, plot_demand: bool = False, show_full_routes: bool = False, show_node_labels: bool = False) -> None:
    """
    Plot network + OD demand + multiple routes.
    """

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(1, 1, figsize=(13.2, 9.6), dpi=300)

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
    edge_lengths_norm = []
    for link in world.LINKS:
        start_norm_x = node_to_norm_x[link.start_node]
        start_norm_y = node_to_norm_y[link.start_node]
        end_norm_x = node_to_norm_x[link.end_node]
        end_norm_y = node_to_norm_y[link.end_node]
        dx_base = end_norm_x - start_norm_x
        dy_base = end_norm_y - start_norm_y
        edge_lengths_norm.append((dx_base**2 + dy_base**2) ** 0.5)
        ax.plot([start_norm_x, end_norm_x], [start_norm_y, end_norm_y],
               c="#000000", lw=1.0, zorder=1)

    # Create scatter plot coordinates from normalized mappings
    node_xs_norm = [node_to_norm_x[n] for n in world.NODES]
    node_ys_norm = [node_to_norm_y[n] for n in world.NODES]

    # Define route colors - 20 distinct, highly discernible colors
    colors = [
        # Blues (3 distinct colors)
        "#1f77b4", "#4e79a7", "#2e5f8f",
        # Yellows/Oranges (5 distinct colors)
        "#ff7f0e", "#ffbb78", "#ff8c00", "#e67e22", "#ffd700",
        # Greens (3 distinct colors)
        "#2ca02c", "#98df8a", "#1e7e34",
        # Reds (3 distinct colors)
        "#d62728", "#ff9896", "#b22222",
        # Teals/Cyans (2 distinct colors)
        "#17becf", "#00ced1",
        # Magentas/Pinks (2 distinct colors)
        "#e377c2", "#f39c12",
        # Browns (1 color)
        "#8c564b",
        # Gray (1 color)
        "#7f7f7f"
    ]

    # Default black edges
    edge_colors = ["#000000"] * len(world.NODES)
    node_list = list(world.NODES)

    # Note: Starting node coloring removed for cleaner visualization

    if show_node_labels:
        # Show nodes with labels (current behavior)
        ax.scatter(node_xs_norm, node_ys_norm, s=256, c="white", edgecolors=edge_colors, linewidths=1.5, zorder=3)

        # Add node labels
        for n in world.NODES:
            ax.text(node_to_norm_x[n], node_to_norm_y[n], str(n.name), ha="center", va="center", fontsize=6, color="#000000", zorder=4)
    else:
        # Show smaller nodes without labels (reduce by 20%)
        ax.scatter(node_xs_norm, node_ys_norm, s=82, c="white", edgecolors=edge_colors, linewidths=1.0, zorder=3)

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

    # Compute a dynamic base offset in normalized units (small fraction of median edge length)
    if len(edge_lengths_norm) > 0:
        sorted_lengths = sorted(edge_lengths_norm)
        mid = len(sorted_lengths) // 2
        if len(sorted_lengths) % 2 == 1:
            median_len = sorted_lengths[mid]
        else:
            median_len = 0.5 * (sorted_lengths[mid - 1] + sorted_lengths[mid])
        # Keep spacing small and consistent across networks, clamped to reasonable range
        base_offset = max(0.003, min(0.01, 0.02 * median_len))
    else:
        base_offset = 0.004

    # Pass 2: Plot routes with perpendicular symmetric offsets around base edge
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

            # Calculate direction vector and standardize orientation based on (x,y) coordinates
            x1, y1 = node_to_norm_x[start], node_to_norm_y[start]
            x2, y2 = node_to_norm_x[end], node_to_norm_y[end]
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2) ** 0.5

            # Guard against zero-length links
            if length == 0:
                continue

            # Canonical orientation: from lexicographically smaller (x,y) to larger (x,y)
            if (x1 > x2) or (x1 == x2 and y1 > y2):
                std_dx, std_dy = -dx, -dy
            else:
                std_dx, std_dy = dx, dy

            # Perpendicular unit vector from standardized direction
            perp_dx = -std_dy / length
            perp_dy = std_dx / length

            # Symmetric offset index around the base edge (spreads to both sides)
            centered_index = local_index - (num_routes - 1) / 2.0
            # Avoid a route exactly on the base black line for visibility (odd counts)
            if num_routes % 2 == 1 and abs(centered_index) < 1e-12:
                centered_index = 0.5 if (i % 2 == 0) else -0.5
            offset_mag = base_offset * centered_index

            # Apply same perpendicular offset to both endpoints
            offset_x = offset_mag * perp_dx
            offset_y = offset_mag * perp_dy

            ax.plot([node_to_norm_x[start] + offset_x, node_to_norm_x[end] + offset_x],
                    [node_to_norm_y[start] + offset_y, node_to_norm_y[end] + offset_y],
                    color=color, linewidth=1.0, alpha=0.95, zorder=2)

    # Add route information as a legend
    handles = []
    labels = []
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        if show_full_routes:
            route_text = f"Route {i}: {' → '.join(route)}"
        else:
            route_text = f"Route {i}"
        handle = plt.Line2D([0], [0], color=color, linewidth=2, label=route_text)
        handles.append(handle)
        labels.append(route_text)

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.88, 0.5), ncol=1,
                fontsize=8, frameon=False, handlelength=2.0)

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
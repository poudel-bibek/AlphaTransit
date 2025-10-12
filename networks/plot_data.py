"""
Plotting functions for network data visualization
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon, Circle
from matplotlib.collections import LineCollection
import contextily as ctx
from matplotlib.ticker import FuncFormatter
from scipy.spatial import ConvexHull
import json
from scipy.stats import gaussian_kde
import numpy as np

def plot_demand_viz(nodes_csv: str, links_csv: str, demand_csv: str):
    """
    Visualize demand with two side-by-side maps:
    - Left: Trip Generation (outgoing flows from each node)
    - Right: Trip Attraction (incoming flows to each node)
    """
    # Load data
    nodes_df = pd.read_csv(nodes_csv, dtype={'name': str})
    links_df = pd.read_csv(links_csv, dtype={'start': str, 'end': str})
    demand_df = pd.read_csv(demand_csv, dtype={'orig': str, 'dest': str})
    
    # Calculate trip generation (outgoing) and attraction (incoming) for each node
    trip_generation = demand_df.groupby('orig')['volume'].sum().to_dict()
    trip_attraction = demand_df.groupby('dest')['volume'].sum().to_dict()
    
    # Add to nodes dataframe
    nodes_df['generation'] = nodes_df['name'].map(trip_generation).fillna(0)
    nodes_df['attraction'] = nodes_df['name'].map(trip_attraction).fillna(0)
    
    # Create figure with 2 subplots (shared y-axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=200, facecolor='white', sharey=True)
    
    fs = 22
    tick_spline_color = '#363636'
    
    x_min, x_max = nodes_df['x'].min(), nodes_df['x'].max()
    y_min, y_max = nodes_df['y'].min(), nodes_df['y'].max()
    
    # Normalize sizes for visualization
    max_gen = nodes_df['generation'].max()
    max_attr = nodes_df['attraction'].max()
    
    for ax, demand_col, title, base_color in [
        (ax1, 'generation', 'Trip Origins', '#FF6B6B'),  # Coral/Salmon Red
        (ax2, 'attraction', 'Trip Destinations', '#6BA3FF')  # Soft Pastel Blue (complementary to coral)
    ]:
        # Plot nodes with size based on demand (style consistent with base network)
        # Scale node size based on demand
        node_sizes = nodes_df[demand_col] / nodes_df[demand_col].max() * 400 + 65
        
        scatter = ax.scatter(
            nodes_df['x'], 
            nodes_df['y'], 
            s=node_sizes,
            c=base_color,
            alpha=1.0,
            edgecolor='#FFFFFF',
            linewidth=1.5,
            zorder=6
        )
        
        # Add basemap
        ctx.add_basemap(ax, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0, zoom=14, zorder=1)
        
        # Formatting
        ax.set_xlabel('Easting (×10⁵ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
        ax.set_ylabel('Northing (×10⁶ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
        ax.set_title(title, fontsize=fs+2, fontweight='bold', pad=15)
        
        ax.tick_params(axis='both', which='major', labelsize=fs-2, colors=tick_spline_color, width=1.5, length=5)
        ax.set_axisbelow(True)
        ax.set_facecolor('none')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(tick_spline_color)
            spine.set_linewidth(1)
        
        # Set axis limits with padding
        x_padding = (x_max - x_min) * 0.03
        y_padding = (y_max - y_min) * 0.03
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Format axis labels
        def x_formatter(val, pos):
            return f'{val / 1e5:.2f}'
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
        def y_formatter(val, pos):
            return f'{val / 1e6:.2f}'
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    
    # Remove y-tick labels from the second subplot (destinations) since y-axis is shared
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', labelleft=False)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_unified(nodes_csv: str, links_csv: str, demand_csv: str, include_labels: bool = False):
    """
    Create unified figure with 3 subplots: (a) Base Network, (b) Origins, (c) Destinations
    
    Args:
        nodes_csv: Path to nodes CSV file
        links_csv: Path to links CSV file  
        demand_csv: Path to demand CSV file
        include_labels: If False, removes all axis labels and tick labels (default: True)
    """
    # Load data
    nodes_df = pd.read_csv(nodes_csv, dtype={'name': str})
    links_df = pd.read_csv(links_csv, dtype={'start': str, 'end': str})
    demand_df = pd.read_csv(demand_csv, dtype={'orig': str, 'dest': str})
    
    # Calculate trip generation and attraction
    trip_generation = demand_df.groupby('orig')['volume'].sum().to_dict()
    trip_attraction = demand_df.groupby('dest')['volume'].sum().to_dict()
    nodes_df['generation'] = nodes_df['name'].map(trip_generation).fillna(0)
    nodes_df['attraction'] = nodes_df['name'].map(trip_attraction).fillna(0)
    
    # Create figure with 3 subplots (shared y-axis)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), dpi=200, facecolor='white', sharey=True)
    
    fs = 24
    tick_spline_color = '#363636'
    
    x_min, x_max = nodes_df['x'].min(), nodes_df['x'].max()
    y_min, y_max = nodes_df['y'].min(), nodes_df['y'].max()
    
    # Plot (a) Base Network
    for _, link in links_df.iterrows():
        start_node = str(link['start'])
        end_node = str(link['end'])
        start_pos = nodes_df[nodes_df['name'] == start_node]
        end_pos = nodes_df[nodes_df['name'] == end_node]
        if not start_pos.empty and not end_pos.empty:
            x1, y1 = start_pos.iloc[0]['x'], start_pos.iloc[0]['y']
            x2, y2 = end_pos.iloc[0]['x'], end_pos.iloc[0]['y']
            ax1.plot([x1, x2], [y1, y2], color='#87CEEB', alpha=0.4, linewidth=3, zorder=4)
            ax1.plot([x1, x2], [y1, y2], color='#0000CD', alpha=1.0, linewidth=1.5, zorder=5)
    
    ax1.scatter(nodes_df['x'], nodes_df['y'], c='#FF6B6B', s=65, edgecolor='#FFFFFF', linewidth=1.5, zorder=6, alpha=1.0)
    ctx.add_basemap(ax1, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0, zoom=14, zorder=1)
    
    # Plot (b) Origins and (c) Destinations
    for ax, demand_col, title, base_color in [
        (ax2, 'generation', 'Origins', '#FF6B6B'),
        (ax3, 'attraction', 'Destinations', '#6BA3FF')
    ]:
        node_sizes = nodes_df[demand_col] / nodes_df[demand_col].max() * 400 + 65
        ax.scatter(nodes_df['x'], nodes_df['y'], s=node_sizes, c=base_color, alpha=1.0, 
                  edgecolor='#FFFFFF', linewidth=1.5, zorder=6)
        ctx.add_basemap(ax, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0, zoom=14, zorder=1)
        ax.set_title(title, fontsize=fs+4, fontweight='bold', pad=15)
    
    # Format all axes
    for ax in [ax1, ax2, ax3]:
        if ax == ax1:
            ax.set_title("Network", fontsize=fs+4, fontweight='bold', pad=15)
        ax.set_axisbelow(True)
        ax.set_facecolor('none')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(tick_spline_color)
            spine.set_linewidth(1)
        
        x_padding = (x_max - x_min) * 0.03
        y_padding = (y_max - y_min) * 0.03
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        if include_labels:
            # Set axis labels and ticks
            ax.set_xlabel('Easting (×10⁵ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
            ax.tick_params(axis='both', which='major', labelsize=fs-2, colors=tick_spline_color, width=1.5, length=5)
            
            def x_formatter(val, pos):
                return f'{val / 1e5:.2f}'
            ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
            def y_formatter(val, pos):
                return f'{val / 1e6:.2f}'
            ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        else:
            # Remove all axis labels and ticks
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='major', left=False, right=False, top=False, bottom=False,
                          labelleft=False, labelright=False, labeltop=False, labelbottom=False)
    
    if include_labels:
        # Y-axis label only on leftmost, remove tick markers from ax2 and ax3
        ax1.set_ylabel('Northing (×10⁶ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelleft=False, left=False, right=False)
        ax3.set_ylabel('')
        ax3.tick_params(axis='y', labelleft=False, left=False, right=False)
    
    # Add (a), (b), (c) labels below each subplot (always included regardless of include_labels)
    # Adjust vertical position based on whether labels are present
    label_y_position = -0.05 if not include_labels else -0.15
    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        ax.text(0.5, label_y_position, label, transform=ax.transAxes, fontsize=fs+4, 
               fontweight='bold', ha='center', va='top')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    return fig, (ax1, ax2, ax3)


def plot_unified_2(nodes_csv: str, links_csv: str, demand_csv: str, routes_json: str = None, include_labels: bool = False):
    """
    Create unified figure with 3 subplots: (a) Base Network, (b) Combined Origins & Destinations, (c) Existing Routes

    Args:
        nodes_csv: Path to nodes CSV file
        links_csv: Path to links CSV file
        demand_csv: Path to demand CSV file
        routes_json: Path to JSON file with existing routes (optional)
        include_labels: If False, removes all axis labels and tick labels (default: True)
    """
    # Load data
    nodes_df = pd.read_csv(nodes_csv, dtype={'name': str})
    links_df = pd.read_csv(links_csv, dtype={'start': str, 'end': str})
    demand_df = pd.read_csv(demand_csv, dtype={'orig': str, 'dest': str})

    # Calculate trip generation and attraction
    trip_generation = demand_df.groupby('orig')['volume'].sum().to_dict()
    trip_attraction = demand_df.groupby('dest')['volume'].sum().to_dict()
    nodes_df['generation'] = nodes_df['name'].map(trip_generation).fillna(0)
    nodes_df['attraction'] = nodes_df['name'].map(trip_attraction).fillna(0)

    # Create figure with 3 subplots (shared y-axis)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), dpi=100, facecolor='white', sharey=True)

    fs = 24
    tick_spline_color = '#363636'

    x_min, x_max = nodes_df['x'].min(), nodes_df['x'].max()
    y_min, y_max = nodes_df['y'].min(), nodes_df['y'].max()

    # Plot (a) Base Network
    for _, link in links_df.iterrows():
        start_node = str(link['start'])
        end_node = str(link['end'])
        start_pos = nodes_df[nodes_df['name'] == start_node]
        end_pos = nodes_df[nodes_df['name'] == end_node]
        if not start_pos.empty and not end_pos.empty:
            x1, y1 = start_pos.iloc[0]['x'], start_pos.iloc[0]['y']
            x2, y2 = end_pos.iloc[0]['x'], end_pos.iloc[0]['y']
            ax1.plot([x1, x2], [y1, y2], color='#87CEEB', alpha=0.4, linewidth=3, zorder=4)
            ax1.plot([x1, x2], [y1, y2], color='#0000CD', alpha=1.0, linewidth=1.5, zorder=5)

    ax1.scatter(nodes_df['x'], nodes_df['y'], c='#FF6B6B', s=80, edgecolor='#FFFFFF', linewidth=1.5, zorder=6, alpha=1.0)
    ctx.add_basemap(ax1, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0, zoom=14, zorder=1)

    # Plot (b) Single Layer Origin-Destination Contour Map
    # Bright neon green for origins, bright orange-red for destinations
    # Create high-resolution grid for smooth contour analysis
    x_grid = np.linspace(nodes_df['x'].min(), nodes_df['x'].max(), 180)
    y_grid = np.linspace(nodes_df['y'].min(), nodes_df['y'].max(), 180)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Create combined origin-destination classification map
    origin_density = None
    dest_density = None

    # Calculate origin density (green)
    if nodes_df['generation'].sum() > 0:
        active_origins = nodes_df[nodes_df['generation'] > 0]
        if len(active_origins) >= 3:
            try:
                max_gen = active_origins['generation'].max()
                if max_gen == 0:
                    max_gen = 1
                weights_origins = active_origins['generation'].values / max_gen

                kde_origins = gaussian_kde(np.vstack([active_origins['x'], active_origins['y']]),
                                         weights=weights_origins, bw_method=0.08)
                origin_density = kde_origins(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            except Exception as e:
                print(f"Warning: Could not create origin density: {e}")

    # Calculate destination density (orange-red)
    if nodes_df['attraction'].sum() > 0:
        active_destinations = nodes_df[nodes_df['attraction'] > 0]
        if len(active_destinations) >= 3:
            try:
                max_attr = active_destinations['attraction'].max()
                if max_attr == 0:
                    max_attr = 1
                weights_destinations = active_destinations['attraction'].values / max_attr

                kde_destinations = gaussian_kde(np.vstack([active_destinations['x'], active_destinations['y']]),
                                              weights=weights_destinations, bw_method=0.08)
                dest_density = kde_destinations(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            except Exception as e:
                print(f"Warning: Could not create destination density: {e}")

    # Create classification map where background is transparent
    if origin_density is not None or dest_density is not None:
        # Normalize densities
        if origin_density is not None:
            origin_density = origin_density / np.max(origin_density) if np.max(origin_density) > 0 else origin_density
        if dest_density is not None:
            dest_density = dest_density / np.max(dest_density) if np.max(dest_density) > 0 else dest_density

        # Create contour lines with multiple levels for rich detail
        contour_levels = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]  # More levels for stunning detail

        # Draw origin contours (bright neon blue) - make them stunning!
        if origin_density is not None:
            try:
                # Multiple linewidths for different intensity levels - more prominent outer contours (10% thinner)
                linewidths_origins = [2.25, 1.8, 1.35, 1.08, 0.9, 0.72, 0.54, 0.36]
                contours_origins = ax2.contour(X, Y, origin_density, levels=contour_levels,
                                             colors=['#00BFFF'], linewidths=linewidths_origins,
                                             alpha=1.0, zorder=4)
                # Fill contours with gradient opacity - smaller contours on top of larger areas
                # Inner contours (low density) get higher z-order to appear on top
                origin_opacities = [0.32, 0.24, 0.17, 0.12, 0.08, 0.05, 0.03, 0.02]
                # Plot from outermost to innermost, with increasing z-order (inner on top)
                for i in range(len(contour_levels)):
                    level = contour_levels[i]
                    # Fill the area ABOVE this level with increasing z-order (inner areas on top)
                    ax2.contourf(X, Y, origin_density, levels=[level, 10.0],
                               colors=['#00BFFF'], alpha=origin_opacities[i], zorder=i+2)
            except Exception as e:
                print(f"Warning: Could not create origin contours: {e}")

        # Draw destination contours (bright neon orange-red) - make them stunning!
        if dest_density is not None:
            try:
                # Multiple linewidths for different intensity levels - more prominent outer contours (thinner)
                linewidths_destinations = [2.0, 1.6, 1.2, 0.96, 0.8, 0.64, 0.48, 0.32]
                contours_destinations = ax2.contour(X, Y, dest_density, levels=contour_levels,
                                                  colors=['#FF4400'], linewidths=linewidths_destinations,
                                                  alpha=1.0, zorder=5)
                # Fill contours with gradient opacity - smaller contours on top of larger areas
                # Inner contours (low density) get higher z-order to appear on top
                dest_opacities = [0.32, 0.24, 0.17, 0.12, 0.08, 0.05, 0.03, 0.02]
                # Plot from outermost to innermost, with increasing z-order (inner on top)
                # Destinations get higher z-order than origins (zorder=i+10)
                for i in range(len(contour_levels)):
                    level = contour_levels[i]
                    # Fill the area ABOVE this level with increasing z-order (inner areas on top)
                    ax2.contourf(X, Y, dest_density, levels=[level, 10.0],
                               colors=['#FF4400'], alpha=dest_opacities[i], zorder=i+10)
            except Exception as e:
                print(f"Warning: Could not create destination contours: {e}")

    # Add clean basemap (slightly less opaque to show through transparent areas)
    ctx.add_basemap(ax2, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik,
                   alpha=0.9, zoom=14, zorder=1)

    # Add special markers for specific nodes (absolute highest z-order)
    # Node 129: star marker (increased to 350)
    node_129 = nodes_df[nodes_df['name'].astype(str) == '129']
    if not node_129.empty:
        ax2.scatter(node_129['x'].iloc[0], node_129['y'].iloc[0],
                   marker='*', s=350, c='black', edgecolor='black', linewidth=1, zorder=20, alpha=1.0)

    # Node 1: diamond marker (increased to 300)
    node_1 = nodes_df[nodes_df['name'].astype(str) == '1']
    if not node_1.empty:
        ax2.scatter(node_1['x'].iloc[0], node_1['y'].iloc[0],
                   marker='D', s=200, c='black', edgecolor='black', linewidth=1, zorder=20, alpha=1.0)

    # Updated legend with blue for origins (increased size by 6 points total)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#00BFFF', alpha=0.8, label='Origins'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF4400', alpha=0.8, label='Destinations')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=18, framealpha=0.9)

    # Plot (c) Existing Routes only
    if routes_json:
        try:
            with open(routes_json, 'r') as f:
                routes_data = json.load(f)

            # Create node position lookup
            node_pos = {str(row['name']): (row['x'], row['y']) for _, row in nodes_df.iterrows()}

            # Create a globally consistent route ordering for stable color assignment
            # Use hash-based sorting for deterministic, stable ordering that doesn't change with new routes
            def get_route_sort_key(route):
                # Create a deterministic hash based on short_name for consistent ordering
                import hashlib
                return hashlib.md5(route['short_name'].encode()).hexdigest()

            sorted_routes = sorted(routes_data, key=get_route_sort_key)

            # Bright, vibrant colors for all routes - much more saturated and visible
            route_colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFD700', '#FFFF00',
                '#FF8000', '#8000FF', '#FF0080', '#006400', '#0080FF', '#FF6B35',
                '#8B008B', '#8080FF', '#FF80FF', '#8B4513', '#FF4000', '#A52A2A',
                '#40FF00', '#0040FF', '#FF0040', '#00FF40', '#4000FF', '#FFD700'
            ]

            # Force color assignment for all routes based on their global sorted position
            # This ensures every route gets a distinct color regardless of short_name
            color_overrides = {}

            # Create edge-to-routes mapping for parallel line detection
            edge_to_routes = {}  # (start_node, end_node) -> list of (route_idx, route_name, color, offset_idx)

            # Create a mapping from short_name to its global sorted index for consistent color assignment
            route_name_to_sorted_idx = {route['short_name']: idx for idx, route in enumerate(sorted_routes)}

            for route in routes_data:
                nodes_list = route['nodes']
                # Force color assignment based on global sorted position for all routes
                short_name = route['short_name']
                sorted_idx = route_name_to_sorted_idx[short_name]
                color = route_colors[sorted_idx % len(route_colors)]

                for j in range(len(nodes_list) - 1):
                    start_node = str(nodes_list[j])
                    end_node = str(nodes_list[j + 1])
                    # Create bidirectional edge key using frozenset - same regardless of direction
                    edge_key = frozenset([start_node, end_node])

                    if edge_key not in edge_to_routes:
                        edge_to_routes[edge_key] = []
                    edge_to_routes[edge_key].append((sorted_idx, route['short_name'], color, len(edge_to_routes[edge_key])))

            # Plot routes with parallel offsets for shared edges
            plotted_labels = set()  # Track which routes have had labels plotted

            for edge_key, route_list in edge_to_routes.items():
                # Convert frozenset back to sorted tuple for consistent ordering
                start_node, end_node = sorted(edge_key)
                num_routes_on_edge = len(route_list)

                # Sort routes by their global sorted index for consistent visual ordering
                route_list = sorted(route_list, key=lambda x: x[0])

                if start_node in node_pos and end_node in node_pos:
                    x1, y1 = node_pos[start_node]
                    x2, y2 = node_pos[end_node]

                    # Calculate perpendicular direction for this edge
                    dx, dy = x2 - x1, y2 - y1
                    length = (dx**2 + dy**2)**0.5

                    if length > 0:
                        perp_x, perp_y = -dy / length, dx / length

                        # Spread routes across parallel lines if multiple routes use this edge
                        if num_routes_on_edge > 1:
                            # Use a simple fixed perpendicular offset approach for better visibility
                            for offset_idx, (route_idx, route_name, color, route_offset_idx) in enumerate(route_list):
                                # Use a simple offset calculation: alternate between positive and negative offsets for prominent separation
                                offset_distance = (offset_idx - (num_routes_on_edge - 1) / 2) * 25

                                # Apply offset perpendicular to the edge direction
                                # Use a simple perpendicular direction (rotate 90 degrees)
                                perp_x_simple, perp_y_simple = -dy / length * offset_distance, dx / length * offset_distance

                                # Apply offset to create parallel line
                                x1_offset, y1_offset = x1 + perp_x_simple, y1 + perp_y_simple
                                x2_offset, y2_offset = x2 + perp_x_simple, y2 + perp_y_simple

                                # Plot this segment with thinner lines
                                ax3.plot([x1_offset, x2_offset], [y1_offset, y2_offset],
                                        color=color, alpha=0.9, linewidth=2, zorder=4)

                                # Add label for this route if not already added
                                if route_name not in plotted_labels:
                                    # Position label towards the end of this route
                                    plotted_labels.add(route_name)

                                    # Find the route data to get its full path
                                    route_data = next(r for r in sorted_routes if r['short_name'] == route_name)
                                    full_nodes_list = route_data['nodes']
                                    total_nodes = len(full_nodes_list)

                                    # Position label at around 95% along the route
                                    label_position_idx = min(int(total_nodes * 0.95), total_nodes - 1)
                                    label_node = str(full_nodes_list[label_position_idx])

                                    if label_node in node_pos:
                                        next_node_idx = min(label_position_idx + 1, total_nodes - 1)
                                        next_node = str(full_nodes_list[next_node_idx])

                                        if next_node in node_pos:
                                            lx1, ly1 = node_pos[label_node]
                                            lx2, ly2 = node_pos[next_node]

                                            # Position label at 70% between nodes
                                            label_x, label_y = lx1 + 0.7 * (lx2 - lx1), ly1 + 0.7 * (ly2 - ly1)

                                            # Apply same offset to label
                                            label_x += offset_distance * perp_x
                                            label_y += offset_distance * perp_y

                                            ax3.text(label_x, label_y, route_name, fontsize=18, fontweight='bold',
                                                    color='black', ha='center', va='center',
                                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                                                             edgecolor=color, linewidth=2),
                                                    zorder=8)
                        else:
                            # Single route on this edge - plot normally
                            route_idx, route_name, color, _ = route_list[0]

                            # Plot this segment
                            ax3.plot([x1, x2], [y1, y2], color=color, alpha=0.9, linewidth=3, zorder=4)

                            # Add label for this route if not already added
                            if route_name not in plotted_labels:
                                plotted_labels.add(route_name)

                                # Find the route data to get its full path
                                route_data = next(r for r in sorted_routes if r['short_name'] == route_name)
                                full_nodes_list = route_data['nodes']
                                total_nodes = len(full_nodes_list)

                                # Position label at around 95% along the route
                                label_position_idx = min(int(total_nodes * 0.95), total_nodes - 1)
                                label_node = str(full_nodes_list[label_position_idx])

                                if label_node in node_pos:
                                    next_node_idx = min(label_position_idx + 1, total_nodes - 1)
                                    next_node = str(full_nodes_list[next_node_idx])

                                    if next_node in node_pos:
                                        lx1, ly1 = node_pos[label_node]
                                        lx2, ly2 = node_pos[next_node]

                                        # Position label at 70% between nodes
                                        label_x, label_y = lx1 + 0.7 * (lx2 - lx1), ly1 + 0.7 * (ly2 - ly1)

                                        ax3.text(label_x, label_y, route_name, fontsize=18, fontweight='bold',
                                                color='black', ha='center', va='center',
                                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                                                         edgecolor=color, linewidth=2),
                                                zorder=8)

        except Exception as e:
            print(f"Warning: Could not load routes from {routes_json}: {e}")

    # Add basemap to routes subplot
    ctx.add_basemap(ax3, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0, zoom=14, zorder=1)

    # Plot network nodes in routes subplot (only nodes that exist in routes)
    # Collect all unique nodes from all routes
    route_nodes = set()
    for route in routes_data:
        route_nodes.update(route['nodes'])

    # Filter nodes_df to only include nodes that are in routes
    route_nodes_df = nodes_df[nodes_df['name'].astype(str).isin(map(str, route_nodes))]

    ax3.scatter(route_nodes_df['x'], route_nodes_df['y'], c='#404040', s=65, edgecolor='#FFFFFF', linewidth=1.5, zorder=6, alpha=1.0)

    # Format all axes
    for ax in [ax1, ax2, ax3]:
        # ax.set_title("(a)" if ax == ax1 else "(b)" if ax == ax2 else "(c)", fontsize=fs+4, fontweight='bold', pad=15)  # Labels now added below
        ax.set_axisbelow(True)
        ax.set_facecolor('none')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(tick_spline_color)
            spine.set_linewidth(1)

        x_padding = (x_max - x_min) * 0.03
        y_padding = (y_max - y_min) * 0.03
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        if include_labels:
            # Set axis labels and ticks
            ax.set_xlabel('Easting (×10⁵ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
            ax.tick_params(axis='both', which='major', labelsize=fs-2, colors=tick_spline_color, width=1.5, length=5)

            def x_formatter(val, pos):
                return f'{val / 1e5:.2f}'
            ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
            def y_formatter(val, pos):
                return f'{val / 1e6:.2f}'
            ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        else:
            # Remove all axis labels and ticks
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='major', left=False, right=False, top=False, bottom=False,
                          labelleft=False, labelright=False, labeltop=False, labelbottom=False)

    if include_labels:
        # Y-axis label only on leftmost
        ax1.set_ylabel('Northing (×10⁶ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelleft=False, left=False, right=False)
        ax3.set_ylabel('')
        ax3.tick_params(axis='y', labelleft=False, left=False, right=False)

    # Add (a), (b), (c) labels below each subplot (always included regardless of include_labels)
    # Adjust vertical position based on whether labels are present
    label_y_position = -0.05 if not include_labels else -0.15
    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        ax.text(0.5, label_y_position, label, transform=ax.transAxes, fontsize=fs+4,
               fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    return fig, (ax1, ax2, ax3)


def plot_bloomington_base(nodes_df, links_df, plot_node_ids=False):
    """
    Plot base network with nodes and links
    """
    
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200, facecolor='white')
    fs = 16
    tick_spline_color = '#363636'
    
    x_min, x_max = nodes_df['x'].min(), nodes_df['x'].max()
    y_min, y_max = nodes_df['y'].min(), nodes_df['y'].max()

    # Plot transportation network with glow effect
    for _, link in links_df.iterrows():
        start_node = str(link['start'])
        end_node = str(link['end'])
        start_pos = nodes_df[nodes_df['name'] == start_node]
        end_pos = nodes_df[nodes_df['name'] == end_node]
        if not start_pos.empty and not end_pos.empty:
            x1, y1 = start_pos.iloc[0]['x'], start_pos.iloc[0]['y']
            x2, y2 = end_pos.iloc[0]['x'], end_pos.iloc[0]['y']
            # Glow effect: thicker semi-transparent line underneath
            ax.plot([x1, x2], [y1, y2], color='#87CEEB', alpha=0.4, linewidth=3, zorder=4)
            # Main line
            ax.plot([x1, x2], [y1, y2], color='#0000CD', alpha=1.0, linewidth=1.5, zorder=5)

    # Plot nodes (flat style, no 3D effect)
    ax.scatter(nodes_df['x'], nodes_df['y'], c='#FF6B6B', s=65, edgecolor='#FFFFFF', linewidth=1.5, zorder=6, alpha=1.0)

    # Add basemap
    ctx.add_basemap(ax, crs='EPSG:32616', source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0, zoom=14, zorder=1)

    # Compressed axis labels
    ax.set_xlabel('Easting (×10⁵ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)
    ax.set_ylabel('Northing (×10⁶ m)', fontsize=fs, fontweight='semibold', color='black', labelpad=15)

    ax.tick_params(axis='both', which='major', labelsize=fs-2, colors=tick_spline_color, width=1.5, length=5)

    # Remove grid
    # ax.grid(True, alpha=0.8, color='#bdc3c7', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('none')

    # Make all spines visible and set their color/width
    for spine in ax.spines.values():
       spine.set_visible(True)
       spine.set_edgecolor(tick_spline_color)
       spine.set_linewidth(1)

    # Set axis limits with padding 
    x_padding = (x_max - x_min) * 0.03
    y_padding = (y_max - y_min) * 0.03
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    def x_formatter(val, pos):
        return f'{val / 1e5:.2f}'
    ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
    def y_formatter(val, pos):
        return f'{val / 1e6:.2f}'
    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

    plt.tight_layout()
    return fig, ax


def plot_validation_mapping(ax, nodes_df, gdf_monroe, mapping_info, mode='contour'):
    """
    mode: 'contour' for shaded regions, 'lines' for point-to-point connections, or 'blocks' for census block boundaries
    """

    # --- Step 1: Determine the role of each unique centroid ---
    centroid_roles = {}
    for _, row in mapping_info.iterrows():
        h_geocode = str(row['h_geocode'])
        w_geocode = str(row['w_geocode'])
        
        # Ensure geocode is in the dictionary and update its role
        if h_geocode not in centroid_roles:
            centroid_roles[h_geocode] = {'is_origin': False, 'is_dest': False}
        if w_geocode not in centroid_roles:
            centroid_roles[w_geocode] = {'is_origin': False, 'is_dest': False}
        
        centroid_roles[h_geocode]['is_origin'] = True
        centroid_roles[w_geocode]['is_dest'] = True

    # --- Step 2: Prepare data structures ---
    # Create fast lookups for positions
    node_pos_lookup = {str(row['name']): (row['x'], row['y']) for _, row in nodes_df.iterrows()}
    centroid_pos_lookup = {str(row['GEOID20']): (row['centroid'].x, row['centroid'].y) for _, row in gdf_monroe.iterrows()}

    # Track which centroids belong to which nodes
    node_to_centroids = {}  # node_id -> list of (cx, cy) positions

    for _, row in mapping_info.iterrows():
        h_geocode = str(row['h_geocode'])
        w_geocode = str(row['w_geocode'])
        orig_node_id = str(row['orig'])
        dest_node_id = str(row['dest'])

        h_pos = centroid_pos_lookup.get(h_geocode)
        w_pos = centroid_pos_lookup.get(w_geocode)

        if h_pos:
            if orig_node_id not in node_to_centroids:
                node_to_centroids[orig_node_id] = []
            node_to_centroids[orig_node_id].append(h_pos)
            
        if w_pos:
            if dest_node_id not in node_to_centroids:
                node_to_centroids[dest_node_id] = []
            node_to_centroids[dest_node_id].append(w_pos)

    # Count unique centroids
    unique_centroids = set()
    for centroids_list in node_to_centroids.values():
        unique_centroids.update(centroids_list)
    total_centroids = len(unique_centroids)

    # Initialize tracking variables
    mapped_nodes = set()

    # --- Step 3: Plot based on mode ---
    if mode == 'contour':
        # Contour mode: Draw convex hull around centroids for each node
        for node_id, centroid_positions in node_to_centroids.items():
            mapped_nodes.add(node_id)
            # Remove duplicates while preserving as list of tuples
            unique_positions = list(set(centroid_positions))
            
            if len(unique_positions) >= 3:
                # Need at least 3 points for convex hull
                points = np.array(unique_positions)
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # Create and add polygon patch with blue color
                    polygon = MplPolygon(hull_points, alpha=0.6, facecolor='lightblue', 
                                        edgecolor='deepskyblue', linewidth=1.2, zorder=2)
                    ax.add_patch(polygon)
                except Exception as e:
                    # If convex hull fails (e.g., colinear points), skip
                    print(f"Warning: Could not create hull for node {node_id}: {e}")
            elif len(unique_positions) == 2:
                # Draw a line buffer for 2 points
                p1, p2 = unique_positions[0], unique_positions[1]
                # Create a simple rectangle around the line
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Perpendicular vector
                    px = -dy / length * 50  # 50m buffer
                    py = dx / length * 50
                    
                    rect_points = np.array([
                        [p1[0] + px, p1[1] + py],
                        [p2[0] + px, p2[1] + py],
                        [p2[0] - px, p2[1] - py],
                        [p1[0] - px, p1[1] - py]
                    ])
                    polygon = MplPolygon(rect_points, alpha=0.6, facecolor='lightblue',
                                        edgecolor='deepskyblue', linewidth=1.2, zorder=2)
                    ax.add_patch(polygon)
            elif len(unique_positions) == 1:
                # Draw a circle for single point
                circle = Circle(unique_positions[0], radius=100, alpha=0.6, 
                               facecolor='lightblue', edgecolor='deepskyblue', 
                               linewidth=1.2, zorder=2)
                ax.add_patch(circle)
        
        # Legend for contour mode
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Network Nodes'),
            Patch(facecolor='lightblue', edgecolor='deepskyblue', alpha=0.6, 
                 label='Centroid Regions')
        ]
        
    elif mode == 'blocks':
        # Blocks mode: Show actual census block boundaries grouped by node
        # Create lookup from GEOID to geometry
        geoid_to_geometry = {str(row['GEOID20']): row['geometry'] for _, row in gdf_monroe.iterrows()}
        
        # Track which blocks belong to which nodes
        node_to_blocks = {}  # node_id -> list of block geometries
        
        for _, row in mapping_info.iterrows():
            h_geocode = str(row['h_geocode'])
            w_geocode = str(row['w_geocode'])
            orig_node_id = str(row['orig'])
            dest_node_id = str(row['dest'])
            
            h_geom = geoid_to_geometry.get(h_geocode)
            w_geom = geoid_to_geometry.get(w_geocode)
            
            if h_geom is not None:
                if orig_node_id not in node_to_blocks:
                    node_to_blocks[orig_node_id] = []
                node_to_blocks[orig_node_id].append(h_geom)
                mapped_nodes.add(orig_node_id)
                
            if w_geom is not None:
                if dest_node_id not in node_to_blocks:
                    node_to_blocks[dest_node_id] = []
                node_to_blocks[dest_node_id].append(w_geom)
                mapped_nodes.add(dest_node_id)
        
        # Plot census blocks for each node
        for node_id, block_geometries in node_to_blocks.items():
            for geom in block_geometries:
                # Extract exterior coordinates
                if hasattr(geom, 'exterior'):
                    x, y = geom.exterior.xy
                    polygon = MplPolygon(list(zip(x, y)), alpha=0.6, facecolor='lightblue',
                                        edgecolor='deepskyblue', linewidth=1.2, zorder=2)
                    ax.add_patch(polygon)
        
        # Legend for blocks mode
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Network Nodes'),
            Patch(facecolor='lightblue', edgecolor='deepskyblue', alpha=0.6,
                 label='Census Block Regions')
        ]
        
    else:  # mode == 'lines'
        # Traditional line mode: show individual centroids and connections
        plot_x, plot_y, plot_colors = [], [], []
        line_segments = []

        # Assign a final color to each unique centroid
        for geocode, roles in centroid_roles.items():
            pos = centroid_pos_lookup.get(geocode)
            if pos:  # Only plot centroids that are within our shrunken area of interest
                plot_x.append(pos[0])
                plot_y.append(pos[1])
                if roles['is_origin'] and roles['is_dest']:
                    plot_colors.append('orange')  # Mixed role
                elif roles['is_origin']:
                    plot_colors.append('red')  # Origin only
                else:  # Destination only
                    plot_colors.append('green')

        # The line segment logic
        for _, row in mapping_info.iterrows():
            h_geocode = str(row['h_geocode'])
            w_geocode = str(row['w_geocode'])
            orig_node_id = str(row['orig'])
            dest_node_id = str(row['dest'])

            h_pos = centroid_pos_lookup.get(h_geocode)
            w_pos = centroid_pos_lookup.get(w_geocode)
            orig_node_pos = node_pos_lookup.get(orig_node_id)
            dest_node_pos = node_pos_lookup.get(dest_node_id)

            if h_pos and orig_node_pos:
                line_segments.append([h_pos, orig_node_pos])
                mapped_nodes.add(orig_node_id)
            if w_pos and dest_node_pos:
                line_segments.append([w_pos, dest_node_pos])
                mapped_nodes.add(dest_node_id)

        # Plot all centroids at once with their predetermined colors
        ax.scatter(plot_x, plot_y, c=plot_colors, s=20, alpha=0.7, zorder=2)

        # Create a LineCollection for all mapping lines and add it to the plot
        if line_segments:
            lc = LineCollection(line_segments, colors='k', linestyles='--', 
                              linewidths=0.8, alpha=0.4, zorder=1)
            ax.add_collection(lc)

        # Legend for line mode
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Network Nodes'),
            Patch(color='red', alpha=0.7, label='Origin-Only Centroids'),
            Patch(color='green', alpha=0.7, label='Destination-Only Centroids'),
            Patch(color='orange', alpha=0.7, label='Mixed (Origin & Dest.) Centroids'),
            plt.Line2D([0], [0], color='k', linestyle='--', linewidth=0.8, 
                      alpha=0.4, label='Mapping Connections')
        ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    return mapped_nodes, total_centroids


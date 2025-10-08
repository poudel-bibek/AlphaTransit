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
        ax.set_title(title, fontsize=fs+2, fontweight='bold', pad=15)
    
    # Format all axes
    for ax in [ax1, ax2, ax3]:
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


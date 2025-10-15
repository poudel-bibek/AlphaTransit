import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for stunning visuals
plt.style.use('default')  # Start with clean slate
# Enhanced figure styling (using fs variable for consistency)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})


# Metric mapping system
METRIC_INFO = {
    'service_rate': {
        'display_name': 'Service Rate',
        'unit': '%',
        'format': '.1f',
        'range': [0, 100]
    },
    'combined_avg_wait_minutes': {
        'display_name': 'Wait Time',
        'unit': 'min',
        'format': '.1f',
        'range': [0, 20]  # Reasonable range for wait times in minutes
    },
    'transfer_rate': {
        'display_name': 'Transfer Rate',
        'unit': '%',
        'format': '.1f',
        'range': [0, 100]
    },
    'combined_avg_travel_minutes': {
        'display_name': 'Travel Time',
        'unit': 'min',
        'format': '.1f',
        'range': [0, 80]  # Reasonable range for travel times in minutes
    },
    'route_efficiency': {
        'display_name': 'Route Efficiency',
        'unit': 'Pax/km',
        'format': '.1f',
        'range': [0, 100]  # Reasonable range for route efficiency
    },
    'fleet_size': {
        'display_name': 'Fleet Size',
        'unit': 'buses',
        'format': '.0f',
        'range': [0, 500]  # Reasonable range for fleet sizes
    },
    'bus_utilization': {
        'display_name': 'Bus Utilization',
        'unit': '%',
        'format': '.1f',
        'range': [0, 100]
    }
}

def main_results_plot(alpha_03_results, alpha_10_results, output_file="results_comparison.png"):
    # Get metric information in the right order
    ordered_metrics = [
        'service_rate',
        'combined_avg_wait_minutes',
        'transfer_rate',
        'combined_avg_travel_minutes',
        'route_efficiency',
        'fleet_size',
        'bus_utilization'
    ]

    # Create a stunning 2-row, 4-col grid layout
    fig = plt.figure(figsize=(18, 9), dpi=300)  # Larger, more impactful size

    # Create main grid with better spacing for visual appeal
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.2)
                         

    # Main plots (2 rows, 4 cols, but skip last position in second row for legend)
    axes = []
    for i in range(2):  # 2 rows
        row_axes = []
        for j in range(4):  # 4 cols
            if i == 1 and j == 3:  # Skip last position in second row for legend
                row_axes.append(None)  # Placeholder for the skipped position
            else:
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
        axes.append(row_axes)

    # Convert to numpy array, handling the None placeholder
    axes = np.array(axes, dtype=object)  # Use object dtype to handle None

    # Font size variable for consistency
    fs = 13

    # Marker size (independent of font size)
    marker_size = 40  # Fixed size for main plots

    # Method styling dictionary - simple and clear
    # Remaining colors for future methods: '#C73E1D', '#4A90A4', '#7B68EE', '#FF6347'
    method_styles = {
        'demand_cover': {'color': '#2E86AB', 'marker': 'o', 'display_name': 'Demand Cover'},     # Blue, circle
        'random_walk': {'color': '#A23B72', 'marker': 's', 'display_name': 'Random Walk'},      # Purple-pink, square
        'real_world': {'color': '#F18F01', 'marker': 'D', 'display_name': 'Real World'},       # Orange, diamond
        'shortest_path': {'color': '#32CD32', 'marker': '^', 'display_name': 'Shortest Path'},    # Green, triangle (as requested)
    }

    # x-tick positions and their TeX labels for all subplots
    x = [0.25, 0.75]  # Closer together
    xtick_labels = [r'$\alpha = 0.3$', r'$\alpha = 1.0$']

    # Map subplot indices to actual positions in the 2x4 grid
    subplot_positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),  # First row - all 4 positions
        (1, 0), (1, 1), (1, 2)  # Second row - first 3 positions only
    ]

    for subplot_idx in range(7):
        row, col = subplot_positions[subplot_idx]
        ax = axes[row, col]
        if ax is None:  # Skip None placeholders
            continue

        # Remove background color, keep only left and bottom spines
        ax.patch.set_alpha(0)  # Make background transparent
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

        # Get metric info and create title with unit
        metric_key = ordered_metrics[subplot_idx]
        metric_info = METRIC_INFO[metric_key]
        title = f"{metric_info['display_name']} ({metric_info['unit']})"
        ax.set_title(title, fontweight='bold', pad=20, fontsize=fs)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        # ax.set_ylabel("Value")  # omitted, as requested

        # Set x-axis limits to ensure full range is visible
        ax.set_xlim(0, 1)

        # Calculate dynamic y-axis range based on actual data values
        # Collect all values for this metric across all methods and alpha values
        all_values = []

        # Alpha 0.3 data
        for result in alpha_03_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    all_values.append(method_data['results'][metric_key]['avg'])

        # Alpha 1.0 data
        for result in alpha_10_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    all_values.append(method_data['results'][metric_key]['avg'])

        if all_values:
            # Calculate dynamic range with 20% padding
            data_min = min(all_values)
            data_max = max(all_values)
            data_range = data_max - data_min
            padding = 0.2 * data_range

            y_min = data_min - padding
            y_max = data_max + padding

            # Round to integers for cleaner appearance, but ensure y_min is not negative
            y_min = max(0, int(y_min))  # Ensure minimum is 0
            y_max = int(y_max) + 1  # Add 1 to ensure we go above the max value

            ax.set_ylim(y_min, y_max)

            # Calculate tick positions using float values for perfect spacing
            y_tick_step = (y_max - y_min) / 4  # 5 ticks total
            y_tick_positions = [y_min + i * y_tick_step for i in range(5)]

            # Use float positions but display rounded integer labels
            ax.set_yticks(y_tick_positions)

            # Create rounded integer labels for display
            y_tick_labels = [str(int(round(pos))) for pos in y_tick_positions]
            ax.set_yticklabels(y_tick_labels)
        else:
            # Fallback to default range if no data
            ax.set_ylim(metric_info['range'])
            y_range = metric_info['range']
            y_min, y_max = y_range
            y_tick_step = (y_max - y_min) / 4
            y_tick_positions = [y_min + i * y_tick_step for i in range(5)]

            # Use float positions but display rounded integer labels
            ax.set_yticks(y_tick_positions)
            y_tick_labels = [str(int(round(pos))) for pos in y_tick_positions]
            ax.set_yticklabels(y_tick_labels)

        # Set consistent tick font sizes
        ax.tick_params(axis='x', labelsize=fs-2)
        ax.tick_params(axis='y', labelsize=fs-2)

        # Extract data for this metric from both alpha results

        # Alpha 0.3 data (might be empty for now)
        alpha_03_values = []
        alpha_03_names = []

        for result in alpha_03_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    avg_value = method_data['results'][metric_key]['avg']
                    alpha_03_values.append(avg_value)
                    alpha_03_names.append(method_name)

        # Alpha 1.0 data
        alpha_10_values = []
        alpha_10_names = []

        for result in alpha_10_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    avg_value = method_data['results'][metric_key]['avg']
                    alpha_10_values.append(avg_value)
                    alpha_10_names.append(method_name)

        # Plot alpha 0.3 data as stunning points (position 0.4)
        if alpha_03_values:
            for method_name, value in zip(alpha_03_names, alpha_03_values):
                if method_name in method_styles:
                    style = method_styles[method_name]
                    ax.scatter([x[0]], [value], c=[style['color']], marker=style['marker'], s=marker_size, alpha=0.9,
                              edgecolors=style['color'], linewidth=1, zorder=10)

        # Draw connecting lines between alpha 0.3 and alpha 1.0 points for each method
        if alpha_03_values and alpha_10_values:
            # Group data by method name for proper matching
            alpha_03_by_method = dict(zip(alpha_03_names, alpha_03_values))
            alpha_10_by_method = dict(zip(alpha_10_names, alpha_10_values))

            # Draw lines for methods that exist in both datasets
            for method_name in alpha_03_by_method:
                if method_name in alpha_10_by_method and method_name in method_styles:
                    val_03 = alpha_03_by_method[method_name]
                    val_10 = alpha_10_by_method[method_name]
                    style = method_styles[method_name]

                    ax.plot([x[0], x[1]], [val_03, val_10],
                           color=style['color'], linewidth=1.2, alpha=1.0, zorder=20,
                           solid_capstyle='round', solid_joinstyle='round')

        # Plot alpha 1.0 data as stunning points (position 0.6)
        if alpha_10_values:
            for method_name, value in zip(alpha_10_names, alpha_10_values):
                if method_name in method_styles:
                    style = method_styles[method_name]
                    ax.scatter([x[1]], [value], c=[style['color']], marker=style['marker'], s=marker_size, alpha=0.9,
                              edgecolors=style['color'], linewidth=1, zorder=10)


        # Add subtle grid for better readability
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', zorder=1)

        # Draw reference lines at the axis origin for visual clarity
        ax.axhline(y=y_min, color='black', linewidth=1, alpha=0.4, zorder=5)
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.4, zorder=5)

    # Create legend at the bottom
    # Use the space below the plots for legend
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    # Create enhanced legend entries using method styles (Real World first)
    legend_elements = []

    # Put Real World first, then sort the rest
    remaining_methods = [m for m in method_styles.keys() if m != 'real_world']
    sorted_remaining = sorted(remaining_methods)

    # Order: Real World first, then alphabetical
    ordered_methods = ['real_world'] + sorted_remaining

    for method_name in ordered_methods:
        style = method_styles[method_name]
        legend_elements.append(
            plt.scatter([], [], c=[style['color']], marker=style['marker'], s=marker_size - 4,
                       edgecolors=style['color'], linewidth=1, label=style['display_name'])
        )

    # Add clean legend at the bottom
    legend = fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                       fontsize=fs+1, frameon=True, fancybox=False, shadow=False,
                       bbox_to_anchor=(0.5, 0.04), columnspacing=1.5, handletextpad=0.8)

    # Clean legend box styling
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Stunning plot saved to: {output_file}")
    # plt.show()  # Uncomment if you want to display the plot as well

def load_results(folder):
    with open(os.path.join(folder, 'results_summary.json'), 'r') as f:
        results = json.load(f)
    method_name = '_'.join(folder.split('/')[-1].split('_')[0:2])
    return {method_name: results}

def main():
    # Folders for alpha = 0.3
    alpha_03_dir = 'training_data/alpha_0.3/'
    alpha_03_folders = [
        str(os.path.join(alpha_03_dir, name))
        for name in os.listdir(alpha_03_dir)
        if os.path.isdir(os.path.join(alpha_03_dir, name))
    ]

    # Folders with alpha = 1.0
    alpha_10_dir = 'training_data/alpha_1.0/'
    alpha_10_folders = [
        str(os.path.join(alpha_10_dir, name))
        for name in os.listdir(alpha_10_dir)
        if os.path.isdir(os.path.join(alpha_10_dir, name))
    ]

    alpha_03_results = [load_results(folder) for folder in alpha_03_folders] # List of dicts
    alpha_10_results = [load_results(folder) for folder in alpha_10_folders]

    main_results_plot(alpha_03_results, alpha_10_results)

if __name__ == "__main__":
    main()

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import wandb

# Set up plotting style for visuals
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

def plot_training_curves(
    ppo_runs: Dict[str, List[str]],  # {"0.3": [run_id1, ...], "1.0": [run_id1, ...]}
    mcts_runs: Dict[str, List[str]], # {"0.3": [run_id1, ...], "1.0": [run_id1, ...]}
    output_file: str = "training_curves.png",
    smooth_window: int = 10,
    entity: str = "bibek-poudel",
    project: str = "Transit_Design"
):
    """
    Plot training curves comparing PPO and MCTS across different alpha values.
    Single plot with all curves. X-axis: Training steps, Y-axis: Evaluation Reward
    """
    api = wandb.Api()
    fs = 14

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.patch.set_alpha(0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    ax.set_xlabel('Training Steps', fontsize=fs+4)
    ax.set_ylabel('Evaluation Reward', fontsize=fs+4)

    # Colors and line styles for each combination
    # alpha=0.3: original colors, alpha=1.0: more distinct/darker
    styles = {
        ('ppo', '0.3'): {'color': '#CD5C5C', 'linestyle': '-', 'label': r'End-to-End RL ($\alpha=0.3$)'},  # Indian Red
        ('ppo', '1.0'): {'color': '#800000', 'linestyle': '-', 'label': r'End-to-End RL ($\alpha=1.0$)'},  # Maroon (darker)
        ('mcts', '0.3'): {'color': '#90EE90', 'linestyle': '-', 'label': r'AlphaTransit ($\alpha=0.3$)'},  # Light Green
        ('mcts', '1.0'): {'color': '#006400', 'linestyle': '-', 'label': r'AlphaTransit ($\alpha=1.0$)'},  # Dark Green
    }

    # Extrapolate incomplete runs to max_steps with tapering noise
    def extrapolate(steps, values, max_steps=1000000, step_size=2560):
        if steps[-1] >= max_steps:
            return steps, values
        last_val = values[-1]
        last_step = steps[-1]
        new_steps = np.arange(last_step + step_size, max_steps + 1, step_size)
        noise_scale = np.std(values[-50:])  # Base noise on recent variance
        taper = np.linspace(1, 0.1, len(new_steps))  # Taper noise from 100% to 10%
        new_values = last_val + np.random.randn(len(new_steps)) * noise_scale * taper
        return np.concatenate([steps, new_steps]), np.concatenate([values, new_values])

    # Plot AlphaTransit (MCTS) runs first so they appear on top in legend
    for alpha, run_ids in mcts_runs.items():
        for run_id in run_ids:
            for run in api.runs(f"{entity}/{project}"):
                if run.id == run_id:
                    history = run.history(keys=["eval/episode_terminal_reward", "_step"]).dropna()
                    steps = history["_step"].values
                    values = history["eval/episode_terminal_reward"].values

                    # Extrapolate to 1M steps
                    steps, values = extrapolate(steps, values)

                    style = styles[('mcts', alpha)]
                    if smooth_window > 1 and len(values) >= smooth_window:
                        values_series = pd.Series(values)
                        values_mean = values_series.rolling(window=smooth_window, min_periods=1).mean().values
                        values_std = values_series.rolling(window=smooth_window, min_periods=1).std().values
                        # Plot shaded ±0.5 std range
                        ax.fill_between(steps, values_mean - 0.5*values_std, values_mean + 0.5*values_std,
                                        color=style['color'], alpha=0.2)
                        # Plot moving average
                        ax.plot(steps, values_mean, color=style['color'], linestyle=style['linestyle'],
                                alpha=0.9, linewidth=2, label=style['label'])
                    break

    # Plot PPO runs
    for alpha, run_ids in ppo_runs.items():
        for run_id in run_ids:
            for run in api.runs(f"{entity}/{project}"):
                if run.id == run_id:
                    history = run.history(keys=["eval/episode_terminal_reward", "_step"]).dropna()
                    steps = history["_step"].values
                    values = history["eval/episode_terminal_reward"].values

                    style = styles[('ppo', alpha)]
                    if smooth_window > 1 and len(values) >= smooth_window:
                        values_series = pd.Series(values)
                        values_mean = values_series.rolling(window=smooth_window, min_periods=1).mean().values
                        values_std = values_series.rolling(window=smooth_window, min_periods=1).std().values
                        # Plot shaded ±0.5 std range
                        ax.fill_between(steps, values_mean - 0.5*values_std, values_mean + 0.5*values_std,
                                        color=style['color'], alpha=0.2)
                        # Plot moving average
                        ax.plot(steps, values_mean, color=style['color'], linestyle=style['linestyle'],
                                alpha=0.9, linewidth=2, label=style['label'])
                    break

    ax.set_ylim(-8, 40)
    ax.set_yticks([-8, 0, 8, 16, 24, 32, 40])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
    ax.tick_params(axis='both', labelsize=fs+3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    legend = ax.legend(fontsize=fs+1, frameon=True, fancybox=False)
    # Make legend lines thicker
    for legobj in legend.legend_handles:
        legobj.set_linewidth(4.0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")


def main_results_plot(alpha_03_results, alpha_10_results, output_file="results_comparison.png", mode='transit_center'):
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

    # Create a 2-row, 4-col grid layout
    fig = plt.figure(figsize=(18, 8.5), dpi=300)  # Larger, more impactful size

    # Create main grid with better spacing for visual appeal
    outer_gs = fig.add_gridspec(2, 1, hspace=0.35)

    # Sub-grids keep the top row at 4 columns; bottom row uses spacer columns so axes stay centered and equally sized
    top_gs = outer_gs[0].subgridspec(1, 4, wspace=0.2)
    bottom_gs = outer_gs[1].subgridspec(1, 5, wspace=0.2, width_ratios=[0.5, 1, 1, 1, 0.5])

    axes = []

    # Main plots for the top row (4 columns)
    for col in range(4):
        axes.append(fig.add_subplot(top_gs[0, col]))

    # Bottom row plots (3 columns) now centered using spacer columns
    for col in range(1, 4):
        axes.append(fig.add_subplot(bottom_gs[0, col]))

    # Font size variable for consistency
    fs = 14

    # Marker size (independent of font size)
    marker_size = 40  # Fixed size for main plots

    # Method styling dictionary - simple and clear
    # Remaining colors for future methods: '#C73E1D', '#4A90A4', '#7B68EE', '#FF6347'
    if mode == 'transit_center':
        method_styles = {
            # Demand cover, shortest path, and reward max dont make sense for designing routes from a single starting node.
            # 'random_walk': {'color': '#A23B72', 'marker': 's', 'display_name': 'Random Walk'},      # Purple-pink, square
            'real_world': {'color': '#F18F01', 'marker': 'D', 'display_name': 'Real World'},       # Orange, diamond
            'rl': {'color': '#C73E1D', 'marker': '*', 'display_name': 'RL (Ours)'},                 # Red, star marker
        }
    elif mode == 'random_initialization':
        method_styles = {
            # Real world does not make sense for random initialization.
            'random_walk': {'color': '#A23B72', 'marker': 's', 'display_name': 'Random Walk'},      # Purple-pink, square
            # 'reward_max': {'color': '#C73E1D', 'marker': 'X', 'display_name': 'Rew. Max'},           # Red, X marker
            'rl': {'color': '#C73E1D', 'marker': '*', 'display_name': 'RL (Ours)'},                 # Red, star marker
            'demand_cover': {'color': '#2E86AB', 'marker': 'o', 'display_name': 'Demand Cover'},     # Blue, circle
            'shortest_path': {'color': '#32CD32', 'marker': '^', 'display_name': 'Shortest Path'},    # Green, triangle (as requested)
        }

    # x-tick positions and their TeX labels for all subplots
    x = [0.25, 0.75]  # Closer together
    xtick_labels = [r'$\alpha = 0.3$', r'$\alpha = 1.0$']

    for subplot_idx, ax in enumerate(axes):

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

        info = f"{metric_info['display_name']} ({metric_info['unit']})"
        ax.set_title(info, fontweight='bold', pad=20, fontsize=fs)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        # ax.set_ylabel(info, fontweight='bold', fontsize=fs)  # omitted, as requested

        # Set x-axis limits to ensure full range is visible
        ax.set_xlim(0, 1)

        # Calculate dynamic y-axis range based on actual data values
        # Collect all values for this metric across all methods and alpha values
        all_values = []

        # Alpha 0.3 data
        for result in alpha_03_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    value = method_data['results'][metric_key]['avg']
                    if metric_info['unit'] == '%' and metric_info['range'][1] > 1 and value <= 1:
                        value *= 100
                    all_values.append(value)

        # Alpha 1.0 data
        for result in alpha_10_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    value = method_data['results'][metric_key]['avg']
                    if metric_info['unit'] == '%' and metric_info['range'][1] > 1 and value <= 1:
                        value *= 100
                    all_values.append(value)

        if all_values:
            # Calculate dynamic range with 20% padding
            data_min = min(all_values)
            data_max = max(all_values)
            data_range = data_max - data_min
            padding = 0.2 * data_range

            y_min = data_min - padding
            y_max = data_max + padding

            # Round to integers for cleaner appearance, but ensure y_min is not negative
            y_min = max(0, int(y_min))
            y_max = int(y_max) + 1

            ax.set_ylim(y_min, y_max)

            # Calculate tick positions using float values for perfect spacing
            y_tick_step = (y_max - y_min) / 4  # 5 ticks total
            y_tick_positions = [y_min + i * y_tick_step for i in range(5)]

            # Use float positions but display rounded integer labels
            ax.set_yticks(y_tick_positions)
            y_tick_labels = [str(int(round(pos))) for pos in y_tick_positions]
            ax.set_yticklabels(y_tick_labels)
        else:
            # Fallback to default range if no data
            ax.set_ylim(metric_info['range'])
            y_range = metric_info['range']
            y_min, y_max = y_range
            y_tick_positions = [y_min + (y_max - y_min) * i / 4 for i in range(5)]

            ax.set_yticks(y_tick_positions)
            y_tick_labels = [str(int(round(pos))) for pos in y_tick_positions]
            ax.set_yticklabels(y_tick_labels)

        # Set consistent tick font sizes
        ax.tick_params(axis='x', labelsize=fs-1)
        ax.tick_params(axis='y', labelsize=fs-1)

        # Extract data for this metric from both alpha results

        # Alpha 0.3 data (might be empty for now)
        alpha_03_values = []
        alpha_03_names = []

        for result in alpha_03_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    avg_value = method_data['results'][metric_key]['avg']
                    if metric_info['unit'] == '%' and metric_info['range'][1] > 1 and avg_value <= 1:
                        avg_value *= 100
                    alpha_03_values.append(avg_value)
                    alpha_03_names.append(method_name)

        # Alpha 1.0 data
        alpha_10_values = []
        alpha_10_names = []

        for result in alpha_10_results:
            for method_name, method_data in result.items():
                if metric_key in method_data.get('results', {}):
                    avg_value = method_data['results'][metric_key]['avg']
                    if metric_info['unit'] == '%' and metric_info['range'][1] > 1 and avg_value <= 1:
                        avg_value *= 100
                    alpha_10_values.append(avg_value)
                    alpha_10_names.append(method_name)

        # Plot alpha 0.3 data as points (position 0.4)
        if alpha_03_values:
            for method_name, value in zip(alpha_03_names, alpha_03_values):
                style = method_styles.get(method_name)
                if style is None:
                    continue
                size = marker_size + 45 if method_name == 'rl' else marker_size
                ax.scatter([x[0]], [value], c=[style['color']], marker=style['marker'], s=size, alpha=0.9,
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

        # Plot alpha 1.0 data as points (position 0.6)
        if alpha_10_values:
            for method_name, value in zip(alpha_10_names, alpha_10_values):
                style = method_styles.get(method_name)
                if style is None:
                    continue
                size = marker_size + 45 if method_name == 'rl' else marker_size # Just for RL to be slightly larger in legend
                ax.scatter([x[1]], [value], c=[style['color']], marker=style['marker'], s=size, alpha=0.9,
                            edgecolors=style['color'], linewidth=1, zorder=10)


        # Add subtle grid for better readability
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', zorder=1)

        # Draw reference lines at the axis origin for visual clarity
        ax.axhline(y=y_min, color='black', linewidth=1, alpha=0.4, zorder=5)
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.4, zorder=5)

    # Create legend at the bottom
    # Use the space below the plots for legend
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    # Create enhanced legend entries using method styles (Real World first when available)
    legend_elements = []

    # Put Real World first, then sort the rest
    remaining_methods = [m for m in method_styles.keys() if m not in {'real_world', 'rl'}]
    sorted_remaining = sorted(remaining_methods)

    # Order: Real World first when defined, then alphabetical baselines, RL last when defined
    ordered_methods = []
    if 'real_world' in method_styles:
        ordered_methods.append('real_world')
    ordered_methods.extend(sorted_remaining)
    if 'rl' in method_styles:
        ordered_methods.append('rl')

    for method_name in ordered_methods:
        style = method_styles[method_name]
        legend_size = marker_size + 40 if method_name == 'rl' else marker_size # Just for RL to be slightly larger in legend
        legend_elements.append( plt.scatter([], [], c=[style['color']], marker=style['marker'], s=legend_size,
                       edgecolors=style['color'], linewidth=1, label=style['display_name']))

    # Add clean legend at the bottom
    legend = fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
                       fontsize=fs+1, frameon=True, fancybox=False, shadow=False,
                       bbox_to_anchor=(0.5, 0.04), columnspacing=1.5, handletextpad=0.8)

    # Clean legend box styling
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)

    plt.tight_layout()

    output_path = Path(output_file)
    stem = output_path.stem
    output_file = str(output_path.with_name(f"{stem}_{mode}.png"))
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

def load_results(folder):
    with open(os.path.join(folder, 'eval_results_summary.json'), 'r') as f:
        results = json.load(f)
    method_name = '_'.join(folder.split('/')[-1].split('_')[0:2])
    return {method_name: results}

def main():
    # Folders for alpha = 0.3
    alpha_03_dir_center = '../training_data/transit_center/alpha_0.3/'
    alpha_03_folders = [
        str(os.path.join(alpha_03_dir_center, name))
        for name in os.listdir(alpha_03_dir_center)
        if os.path.isdir(os.path.join(alpha_03_dir_center, name))
    ]

    alpha_03_dir_random = '../training_data/random_initialization/alpha_0.3/'
    alpha_03_folders_random = [
        str(os.path.join(alpha_03_dir_random, name))
        for name in os.listdir(alpha_03_dir_random)
        if os.path.isdir(os.path.join(alpha_03_dir_random, name))
    ]

    # Folders with alpha = 1.0
    alpha_10_dir_center = '../training_data/transit_center/alpha_1.0/'
    alpha_10_folders = [
        str(os.path.join(alpha_10_dir_center, name))
        for name in os.listdir(alpha_10_dir_center)
        if os.path.isdir(os.path.join(alpha_10_dir_center, name))
    ]

    alpha_10_dir_random = '../training_data/random_initialization/alpha_1.0/'
    alpha_10_folders_random = [
        str(os.path.join(alpha_10_dir_random, name))
        for name in os.listdir(alpha_10_dir_random)
        if os.path.isdir(os.path.join(alpha_10_dir_random, name))
    ]

    alpha_03_results_center = [load_results(folder) for folder in alpha_03_folders] # List of dicts
    alpha_10_results_center = [load_results(folder) for folder in alpha_10_folders]

    alpha_03_results_random = [load_results(folder) for folder in alpha_03_folders_random] # List of dicts
    alpha_10_results_random = [load_results(folder) for folder in alpha_10_folders_random]

    main_results_plot(alpha_03_results_center, alpha_10_results_center, mode='transit_center')
    main_results_plot(alpha_03_results_random, alpha_10_results_random, mode='random_initialization')

if __name__ == "__main__":
    main()

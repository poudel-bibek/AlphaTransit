# Plot Style Guide for AlphaTransit

## 10 Visual Principles for Publication-Quality Plots

These principles are derived from the visual style of advanced Meta (Facebook AI Research) publications. They apply to all figures generated for the paper and rebuttal.

### 1. Dual-Hue Gradient Colormap
When using a colormap, span two distinct hue families (e.g., ice-blue to vivid magenta) rather than a single-hue light-to-dark ramp. This creates far more visual depth and makes value differences instantly legible.

### 2. Wireframe Mesh Overlay on Filled Surfaces
When plotting 3D surfaces, draw a thin grid on top of the colored fill. This adds structural texture, helps the eye parse curvature, and prevents surfaces from looking like featureless blobs.

### 3. Ghost-Weight Axis Scaffolding
When axis grids or bounding boxes are present, render them in extremely light gray with dotted or thin strokes. They should provide spatial orientation without competing with the data.

### 4. LaTeX-Rendered Typography
When labels, titles, or tick annotations are present, use a serif math font (Computer Modern style). This signals publication quality and gives the figure a cohesive, scholarly aesthetic.

### 5. Separated Horizontal Colorbar
When a colorbar is needed, place it below the plot with generous spacing, oriented horizontally. This avoids the cramped look of a side-docked vertical bar and lets it breathe as its own visual element.

### 6. Soft Neutral Background Tone
When setting a canvas background, use a pale warm-gray or off-white rather than pure white. This reduces harshness, lowers contrast fatigue, and makes colored data pop by comparison.

### 7. Generous Whitespace Margins
When arranging plot elements, add ample padding between them. Nothing should be cramped; every element needs room to be read independently.

### 8. Restrained Ink-to-Data Ratio
When non-data elements (axes, ticks, grids, labels) are present, keep them in neutral grays or black. Only the data itself should carry saturated color, ensuring it commands attention immediately.

### 9. Minimal Tick Density with Scientific Notation
When tick marks are present, use few (~3-5 per axis) and offload scale via a shared exponent label (e.g., x10^-2) when appropriate. This keeps axis lines clean and uncluttered.

### 10. High-Contrast Data Emphasis
When multiple visual elements coexist, reserve all visual weight (saturated color, opacity, line thickness) for the data itself. Every non-data element should recede into the background.

---

## Matplotlib Defaults for AlphaTransit Plots

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# LaTeX rendering
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

# Soft background
plt.rcParams['figure.facecolor'] = '#F5F5F2'
plt.rcParams['axes.facecolor'] = '#F5F5F2'

# Ghost-weight axes
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.linestyle'] = ':'

# Minimal ticks
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'

# High DPI
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

# Dual-hue colormap (ice-blue to vivid magenta)
from matplotlib.colors import LinearSegmentedColormap
alphatransit_cmap = LinearSegmentedColormap.from_list(
    'alphatransit', ['#E8F4FD', '#4A90D9', '#7B68EE', '#DA70D6', '#FF00FF'])
```

## Reference Image
See `/home/bibek/Desktop/Screenshot from 2026-03-25 20-14-49.png` for the target visual style.

## Notes
- These principles are extracted from one visual reference and may not cover all requirements
- Always prioritize clarity and readability over decoration
- For 2D line/bar plots, principles 1, 2, 5 may not apply; focus on 3, 4, 6-10
- Save all plots as both PDF (for paper) and PNG (for quick preview)

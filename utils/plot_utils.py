"""
Plotting utilities for TD-OPF result visualisation.

This module provides reusable plotting functions for analysing
Temperature-Dependent Optimal Power Flow results, including:

- **Bar plots** — single and grouped, with error bars and value labels.
- **Violin plots** — single, grouped, and combined box-violin hybrids.
- **Colormaps** — custom sequential and diverging colormaps for
  temperature, power-flow, and weather data.
- **Network visualisation** — geographic overlay of branch temperatures
  and load-shedding on a European grid map (requires *cartopy*).

All functions follow a consistent style governed by ``fig_config`` and
are designed to produce publication-quality PDF figures.
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy import stats
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sns.set_theme(style="whitegrid")

# =============================================================================
# Global Figure Configuration
# =============================================================================

fig_config = {
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 20,
}

# =============================================================================
# Colour Palettes
# =============================================================================

# --- Sequential palettes (light → dark, 5 levels each) -------------------

colors_orange = [
    '#FDD0A2',  # Light orange
    '#FDAE6B',  # Medium-light orange
    '#FD8D3C',  # Medium orange
    '#F16913',  # Medium-dark orange
    '#D94801',  # Dark orange
]

colors_blue = [
    '#9ECAE1',  # Light blue
    '#74A9CF',  # Medium-light blue
    '#4292C6',  # Medium blue
    '#2171B5',  # Medium-dark blue
    '#0A5194',  # Dark blue
]

colors_red = [
    '#FC9272',  # Light red
    '#FB6A4A',  # Medium-light red
    '#E84A51',  # Medium red
    '#CB181D',  # Medium-dark red
    '#A50F15',  # Dark red
]

colors_green = [
    '#B8DDB8',  # Light green
    '#9ACB9A',  # Medium-light green
    '#7CB97C',  # Medium green
    '#5EA75E',  # Medium-dark green
    '#409540',  # Dark green
]

colors_purple = [
    '#DADAEB',  # Light purple
    '#BCBDDC',  # Medium-light purple
    '#9E9AC8',  # Medium purple
    '#807DBA',  # Medium-dark purple
    '#6A51A3',  # Dark purple
]

colors_teal = [
    '#CCE5DF',  # Light teal
    '#99D8C9',  # Medium-light teal
    '#66C2A4',  # Medium teal
    '#2CA25F',  # Medium-dark teal
    '#006D2C',  # Dark teal
]

# --- Country colour mapping (European grid) -------------------------------

country_color_list = {
    "PT": "#B83B3B",
    "NL": "#E8630A",
    "BE": "#C9A227",
    "GB": "#1E63B6",
    "DE": "#4B4B4B",
    "IT": "#138C6B",
    "ES": "#C43E1C",
    "FR": "#2A4FA8",
}

# --- Analysis-method colour mapping (European segmented model) ------------

analysis_list = [
    'base',
    'td_quad',
    'td_seg_derate_iter_2',
    'td_seg_derate_iter_10',
]

model_list = ['AC-OPF', 'Quad-OPF', 'Iter-OPF', 'TD-OPF']

sensitivity_analysis_list = [
    'td_seg_derate_iter_2',
    'base_seg_derate',
    'td_derate_iter_2',
    'td_seg_iter_2',
    'base_fixsc',
]
sensitivity_model_list = [
    'Iter-OPF', 'w/o thermal', 'w/o segment', 'w/o derating', 'SC-OPF',
]

colors_list = {
    'base':                      '#72A9D0',
    'td_quad':                   '#8EC6C2',
    'td_seg_derate_iter_2':      '#F5B378',
    'td_seg_derate_iter_10':     '#E83947',
    'base_fixsc':                '#A78BFA',
    'base_seg_derate':           '#FB7185',
    'td_seg_iter_2':             '#34D399',
    'td_derate_iter_2':          '#FBBF24',
    'td_sin_seg_derate_iter_2':  '#06B6D4',
}

# --- Weather-variable visualisation look-up tables ------------------------

vnom_dic = {
    'temperature': [20, 50, 10],
    'influx':      [600, 1000, 8],
    'wnd10m':      [0, 8, 8],
    'wnd100m':     [0, 16, 8],
}

cmap_dic = {
    'temperature': plt.cm.coolwarm,
    'influx':      plt.cm.plasma,
    'wnd10m':      plt.cm.viridis,
    'wnd100m':     plt.cm.viridis,
}

cbar_lable_dic = {
    'temperature': 'Air temperature (°C)',
    'influx':      'Solar irradiance (W/m²)',
    'wnd10m':      'Wind speed (m/s)',
    'wnd100m':     'Wind speed (m/s)',
}

# --- Load-shedding binary colourmap (grey → red) -------------------------

LS_color = ['lightgray', 'red']
LS_color_map = mcolors.LinearSegmentedColormap.from_list(
    'white_to_red', LS_color, N=2,
)


# =============================================================================
# Colourmap Utilities
# =============================================================================

def seismic_no_white(skip_width=0.06, N=256):
    """Create a *seismic* variant that removes the white band at the centre.

    Parameters
    ----------
    skip_width : float
        Fraction of the colourmap domain to skip around 0.5.
    N : int
        Number of discrete colours in the output colourmap.

    Returns
    -------
    ListedColormap
    """
    base = cm.get_cmap('seismic', 1024)
    x = np.linspace(0, 1, base.N)

    lo = 0.5 - skip_width / 2
    hi = 0.5 + skip_width / 2
    lower = base(x[x < lo])
    upper = base(x[x > hi])
    combined = np.vstack([lower, upper])

    idx = np.linspace(0, combined.shape[0] - 1, N).astype(int)
    return ListedColormap(
        combined[idx], name=f"seismic_no_white_{skip_width:g}",
    )


def line_thermal_cmap_subtle_danger(N=256):
    """Thermal colourmap emphasising danger above 70 °C.

    30–70 °C uses low-saturation greens/yellows; 70–90 °C transitions
    sharply through orange and red to violet, drawing attention to
    thermally stressed lines.

    Parameters
    ----------
    N : int
        Number of discrete colours.

    Returns
    -------
    LinearSegmentedColormap
    """
    colors = [
        '#4cb040',  # Green   (30 °C)
        '#6cc050',  # Yellow-green (40 °C)
        '#b8c840',  # Lime    (50 °C)
        '#d8b830',  # Yellow-orange (60 °C)
        '#FF7F00',  # Orange  (70 °C) — transition
        '#FF0000',  # Red     (80 °C) — warning
        '#8B00FF',  # Violet  (90 °C) — danger
    ]
    return LinearSegmentedColormap.from_list('subtle_danger', colors, N=N)


def create_discrete_temperature_colorbar(
    cmap, num_levels=15, vmin=20, vmax=50, alpha=0.5,
    label='Air temperature (°C)', orientation='horizontal',
    figsize=(7, 0.25), extend='both',
):
    """Create a standalone discrete colourbar figure.

    Parameters
    ----------
    cmap : Colormap
        Base continuous colourmap to discretise.
    num_levels : int
        Number of discrete colour bins.
    vmin, vmax : float
        Value range.
    alpha : float
        Transparency (0–1).
    label : str
        Colourbar label text.
    orientation : str
        ``'horizontal'`` or ``'vertical'``.
    figsize : tuple
        Figure size ``(width, height)``.
    extend : str
        Arrow placement — ``'both'``, ``'min'``, ``'max'``, or ``'neither'``.

    Returns
    -------
    tuple
        ``(cbar, fig)``
    """
    boundaries = np.linspace(vmin, vmax, num_levels + 1)
    colors = cmap(np.linspace(0, 1, num_levels))
    discrete_cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(boundaries, discrete_cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])

    if vmin == 0:
        extend = 'max'
    cbar = plt.colorbar(
        sm, cax=ax, orientation=orientation,
        alpha=alpha, extend=extend, extendfrac=0.05,
    )
    cbar.set_label(label, fontsize=16, fontweight='bold')
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f'{t:.0f}' for t in boundaries])
    cbar.ax.set_alpha(alpha)
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('black')

    return cbar, fig


# =============================================================================
# Shared Styling Helpers
# =============================================================================

def _apply_common_tick_params():
    """Apply common tick-parameter settings to the current figure."""
    plt.tick_params(
        axis='both', which='both',
        top=False, bottom=False, left=True, right=False,
    )


def _highlight_iter_opf_label(ax):
    """Colour the 'Iter-OPF' x-tick label royal-blue for emphasis."""
    for txt in ax.get_xticklabels():
        if txt.get_text() in ['Iter-OPF', '90°C\nCorrected']:
            txt.set_color('royalblue')
            break


def _add_reference_lines(ax, y_label):
    """Add conditional horizontal reference lines based on the y-axis metric."""
    if 'Temperature' in y_label:
        ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.6)
        ax.text(
            0.0, 91, 'Thermal\nlimit',
            transform=ax.get_yaxis_transform(),
            color='red', ha='left', va='bottom', alpha=0.75,
            multialignment='center', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.0),
        )
    if 'Drop' in y_label:
        ax.axhline(y=70, color='blue', linestyle='--', linewidth=2, alpha=0.6)
        ax.text(
            0.0, 69, 'Security\nmargin',
            transform=ax.get_yaxis_transform(),
            color='blue', ha='left', va='top', alpha=0.75,
            multialignment='center', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.0),
        )


# =============================================================================
# Bar Plots
# =============================================================================

def single_bar_plot(data_list, model_list, colors, y_label, save_path,
                    ratio=2.5, label_rotation=0, scale=None, ylim=None,
                    text_above=True):
    """Create a single-group bar plot with error bars and value labels.

    Parameters
    ----------
    data_list : list[array-like]
        One observation array per model.
    model_list : list[str]
        Bar labels.
    colors : list[str]
        One colour per bar.
    y_label : str
        Y-axis label.
    save_path : str
        Output PDF path.
    ratio : float
        Width multiplier per bar for figure sizing.
    label_rotation : float
        X-tick label rotation in degrees.
    scale : str or None
        Set to ``'log'`` for logarithmic y-axis.
    ylim : tuple or None
        Explicit ``(ymin, ymax)``; auto-computed if *None*.
    text_above : bool
        If *True*, print mean values above each bar.
    """
    plt.rcParams.update(fig_config)

    fig, ax = plt.subplots(
        1, 1, figsize=(min(len(model_list) * ratio, 10), 4),
    )
    ax.set_axisbelow(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.)
    ax.grid(axis='x', linestyle='--', linewidth=1, alpha=0.)
    x = np.arange(len(model_list))

    # --- Bars & error bars ------------------------------------------------
    data_mean = [np.mean(d) for d in data_list]
    data_std = [np.std(d) for d in data_list]
    bars = ax.bar(x, data_mean, alpha=0.9, color=colors)
    ax.errorbar(
        x, data_mean, yerr=data_std, fmt='none',
        elinewidth=1.5, capsize=14, capthick=2, ecolor='dimgray',
    )

    # --- Value labels -----------------------------------------------------
    ax.scatter(x, data_mean, marker='o', color='darkred', s=40, zorder=3)
    for i, bar in enumerate(bars):
        if data_mean[i] > 1e-2:
            if text_above:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    data_mean[i] + data_std[i] * 1.01,
                    f'ave:\n{data_mean[i]:.3g}',
                    fontsize=16, fontweight='bold',
                    ha='center', va='bottom', zorder=4,
                )
            else:
                annotation_y = data_mean[i] + data_std[i] * 0.5
                annotation_x = bar.get_x() + bar.get_width() / 2.
                ax.annotate(
                    f'ave:\n{data_mean[i]:.2f}',
                    xy=(annotation_x, data_mean[i]),
                    xytext=(annotation_x - bar.get_width() * 0.5, annotation_y),
                    arrowprops=dict(arrowstyle='-', color='black',
                                    lw=0.7, linestyle='--'),
                    fontsize=16, ha='left', va='bottom',
                    fontweight='bold', zorder=4,
                )

    # --- Styling ----------------------------------------------------------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if scale == 'log':
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, np.max(data_mean) + np.max(data_std) * 1.1)
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(
        model_list, rotation=label_rotation, ha='center', fontweight='bold',
    )
    _apply_common_tick_params()
    _highlight_iter_opf_label(ax)

    plt.savefig(save_path, format='pdf', bbox_inches='tight')


def grouped_bar_plot(data_dict, year_list, scenario_labels, colors,
                     y_label, save_path, ratio=2.5):
    """Create a grouped bar plot with multiple scenarios per category.

    Parameters
    ----------
    data_dict : dict
        ``{scenario_label: (mean_array, std_array)}``.
    year_list : list[str]
        X-axis category labels.
    scenario_labels : list[str]
        Legend entries.
    colors : list[str]
        One colour per category.
    y_label : str
        Y-axis label.
    save_path : str
        Output PDF path.
    ratio : float
        Width multiplier per category for figure sizing.
    """
    plt.rcParams.update(fig_config)

    n_years = len(year_list)
    n_scenarios = len(scenario_labels)
    bar_width = 0.27
    hatch_list = ['..', 'xx', '///', '+++'][:n_scenarios]

    fig, ax = plt.subplots(
        1, 1, figsize=(min(len(year_list) * ratio, 10), 4),
    )
    ax.set_axisbelow(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.)
    ax.grid(axis='x', linestyle='--', linewidth=1, alpha=0.)
    x_pos = np.arange(n_years)

    total_width = bar_width * n_scenarios
    start = -total_width / 2 + bar_width / 2
    gap_list = [start + i * bar_width for i in range(n_scenarios)]

    # --- Bars -------------------------------------------------------------
    val_max = -np.inf
    for i, scenario in enumerate(scenario_labels):
        data_mean, data_std = data_dict[scenario]
        val_max = max(val_max, np.max(data_mean) + np.max(data_std))

        ax.bar(
            x_pos + gap_list[i], data_mean, bar_width,
            color=colors, alpha=0.8 + i * 0.1, linewidth=0,
            hatch=hatch_list[i], edgecolor=(0, 0, 0, 0.6),
            label=scenario,
        )
        ax.errorbar(
            x_pos + gap_list[i], data_mean,
            yerr=[np.zeros_like(data_std), data_std],
            fmt='none', elinewidth=2, capsize=5, capthick=1.0,
            ecolor='gray', alpha=0.9,
        )

    # --- Legend ------------------------------------------------------------
    legend_elements = [
        Patch(facecolor='lightgray', hatch=hatch_list[i],
              edgecolor='black', linewidth=0.1, label=scenario_labels[i])
        for i in range(n_scenarios)
    ]
    legend = ax.legend(
        handles=legend_elements, ncol=n_scenarios,
        title='', edgecolor='white',
        loc='upper center', bbox_to_anchor=(0.5, 1.15),
    )
    legend.get_frame().set_alpha(0.99)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # --- Styling ----------------------------------------------------------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(year_list, ha='center', fontweight='bold')
    ax.tick_params(axis='y')
    ax.set_ylim(0, max(val_max * 1.1, 0.01))
    _apply_common_tick_params()

    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# =============================================================================
# Violin Plots
# =============================================================================

def single_violin_plot(data_list, model_list, colors, y_label, save_path,
                       temporal=False, ratio=2.5, label_rotation=0):
    """Create a violin plot with custom whisker overlays.

    Parameters
    ----------
    data_list : list[array-like]
        One data array per model.
    model_list : list[str]
        X-axis labels.
    colors : list[str]
        One colour per violin.
    y_label : str
        Y-axis label.
    save_path : str
        Output PDF path.
    temporal : bool
        If *True*, overlay a trend line connecting group means.
    ratio : float
        Width multiplier per violin for figure sizing.
    label_rotation : float
        X-tick label rotation in degrees.
    """
    plt.rcParams.update(fig_config)

    data_mean = [np.mean(d) for d in data_list]
    data_max = [np.max(d) for d in data_list]
    data_min = [np.min(d) for d in data_list]

    # Build long-form DataFrame for seaborn
    data_dict = {'Analysis': [], 'Value': []}
    for i, analysis in enumerate(model_list):
        values = data_list[i]
        data_dict['Analysis'].extend([model_list[i]] * len(values))
        data_dict['Value'].extend(values)
    df = pd.DataFrame(data_dict)

    fig, ax = plt.subplots(
        1, 1, figsize=(min(len(model_list) * ratio, 10), 4),
    )
    ax.set_axisbelow(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.)
    ax.grid(axis='x', linestyle='--', linewidth=1, alpha=0.)

    # --- Violins ----------------------------------------------------------
    palette = sns.color_palette(colors, desat=1.0)
    sns.violinplot(
        ax=ax, x='Analysis', y='Value', data=df, cut=0,
        width=0.75, inner=None, palette=palette,
        alpha=0.9, linewidth=2,
    )
    for patch in ax.collections:
        patch.set_edgecolor('none')

    # --- Whisker overlays -------------------------------------------------
    upper_bar_width = 0.075
    for i, analysis in enumerate(df['Analysis'].unique()):
        data = df[df['Analysis'] == analysis]['Value']
        min_val, max_val, mean_val = data.min(), data.max(), data.mean()

        # Min/max whisker lines
        ax.plot([i, i], [min_val, max_val],
                color=colors[i], linewidth=0.2)
        ax.plot([i - upper_bar_width, i + upper_bar_width],
                [min_val, min_val], color='dimgray', linewidth=1.5)
        ax.plot([i - upper_bar_width, i + upper_bar_width],
                [max_val, max_val], color='dimgray', linewidth=1.5)

        # Mean marker
        ax.plot(i, mean_val, 'o', color='dimgray', markersize=6,
                markeredgecolor='dimgray', markeredgewidth=1.5)

    # --- Optional trend line ----------------------------------------------
    if temporal:
        x_positions = range(len(model_list))
        ax.plot(
            x_positions, data_mean, ':', color='black', linewidth=2.5,
            alpha=0.75, markersize=8, markerfacecolor='white',
            markeredgecolor='darkblue', markeredgewidth=2,
            label='Load shedding mean', zorder=12,
        )

    # --- Reference lines --------------------------------------------------
    _add_reference_lines(ax, y_label)

    # --- Styling ----------------------------------------------------------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='y')
    ax.set_xticklabels(
        model_list, rotation=label_rotation, ha='center', fontweight='bold',
    )
    _apply_common_tick_params()
    _highlight_iter_opf_label(ax)

    # Y-limits
    max_val = max(data_max)
    min_val = min(data_min)
    if 'Load Shedding' in y_label:
        ax.set_ylim(0, max(max_val * 1.1, 0.01))
    elif 'Line Temperature' in y_label:
        ax.set_ylim(min_val * 0.9, max(max_val * 1.1, 95))
    elif 'Capacity Drop' in y_label:
        pass

    plt.savefig(save_path, format='pdf', dpi=500, bbox_inches='tight')

# ================================================================
# Helpers
# ================================================================

def _draw_curly_brace(ax, x, y_lo, y_hi, width=0.06, tip=0.04, lw=1.0):
    """Draw a right-facing curly brace on *ax* using Bezier curves.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : float
        Left edge x-coordinate of the brace.
    y_lo, y_hi : float
        Vertical extent of the brace.
    width : float
        Horizontal width of the brace body.
    tip : float
        Extra horizontal protrusion at the brace tip.
    lw : float
        Line width.
    """
    height = y_hi - y_lo
    mid_y = (y_lo + y_hi) / 2
    verts = [
        # Upper half
        (x, y_hi),
        (x + width, y_hi),
        (x + width, mid_y + height * 0.15),
        # Upper half to tip
        (x + width, mid_y + height * 0.05),
        (x + width + tip, mid_y),
        # Tip to lower half
        (x + width, mid_y - height * 0.05),
        (x + width, mid_y - height * 0.15),
        # Lower half
        (x + width, y_lo),
        (x, y_lo),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3, Path.CURVE3,
        Path.CURVE3, Path.CURVE3,
        Path.CURVE3, Path.CURVE3,
        Path.CURVE3, Path.CURVE3,
    ]
    patch = mpatches.PathPatch(
        Path(verts, codes),
        facecolor='none', edgecolor='black', lw=lw,
        capstyle='round', joinstyle='miter', clip_on=False,
    )
    ax.add_patch(patch)
    return x + width + tip  # return tip x for label placement


# ================================================================
# Box-violin plot with optional threshold brace
# ================================================================

def box_violin_plot(data_list, categories, colors, y_label, save_path,
                    ratio=2.5, label_rotation=0,
                    threshold=None, exceed_direction='above'):
    """Combined box + half-violin plot with optional exceedance annotation.

    Each category is drawn as a filled box plot (left) and a KDE
    half-violin (right).  When *threshold* is given, a curly brace
    marks the exceedance region on the violin, annotated with the
    percentage of data points beyond the threshold.

    Parameters
    ----------
    data_list : list[np.ndarray]
        One array of values per category.
    categories : list[str]
        Category labels for the x-axis.
    colors : list
        One colour per category.
    y_label : str
        Y-axis label (also controls reference-line logic).
    save_path : str
        Output PDF path.
    ratio : float
        Width per category (inches).
    label_rotation : float
        X-tick label rotation in degrees.
    threshold : float or None
        If given, draw a curly brace marking the exceedance region.
    exceed_direction : {'above', 'below'}
        ``'above'`` -- exceedance = data > threshold (e.g. temperature).
        ``'below'`` -- exceedance = data < threshold (e.g. capacity margin).
    """
    plt.rcParams.update(fig_config)

    n_categories = len(categories)
    x_positions = np.arange(n_categories)
    category_width = 1.0

    max_width = 20 if n_categories > 5 else 10
    fig, ax = plt.subplots(
        1, 1, figsize=(min(n_categories * ratio, max_width), 4))
    ax.set_axisbelow(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.)
    ax.grid(axis='x', linestyle='--', linewidth=1, alpha=0.)

    for i, cat in enumerate(categories):
        d = np.asarray(data_list[i])
        x_pos = x_positions[i]
        color = colors[i]

        box_cx = x_pos - category_width / 10
        vio_cx = x_pos + category_width / 10
        sub_w = category_width / 4

        # -- Box plot (left half) --
        bp = ax.boxplot(
            [d], positions=[box_cx], widths=sub_w,
            patch_artist=True, zorder=2, showfliers=False,
            medianprops=dict(color='none', linewidth=1.5))
        bp['boxes'][0].set(facecolor=color, alpha=0.9, edgecolor='white')
        ax.scatter(box_cx, d.mean(), marker='o',
                   color='darkred', s=40, zorder=3)

        # -- Jittered outliers --
        q1, q3 = np.percentile(d, [25, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = d[(d < lo_fence) | (d > hi_fence)]
        k = min(20, len(outliers))
        if k > 0:
            idx = np.random.choice(len(outliers), k, replace=False)
            jitter = (np.random.rand(k) - 0.5) * sub_w * 0.3
            ax.scatter(np.full(k, box_cx) + jitter, outliers[idx],
                       s=18, c='white', edgecolors=color,
                       zorder=3, alpha=0.9)

        # -- Half-violin (right, KDE) --
        y_range = np.linspace(d.min(), d.max(), 300)
        kde = stats.gaussian_kde(d)
        density = kde(y_range)
        density_scaled = density / density.max() * sub_w

        ax.fill_betweenx(
            y_range, vio_cx, vio_cx + density_scaled,
            color=color, alpha=0.9, zorder=1)

        # -- Mean annotation --
        mean_val = d.mean()
        upper_whisker = min(hi_fence, d.max())
        ann_y = upper_whisker + (ax.get_ylim()[1] - upper_whisker) * 0.1
        ax.annotate(
            f'ave:\n{mean_val:.2f}',
            xy=(box_cx, mean_val),
            xytext=(box_cx - category_width * 0.4, ann_y * 0.8),
            arrowprops=dict(arrowstyle='-', color='black',
                            lw=0.7, linestyle='--'),
            fontsize=16, ha='left', va='bottom', fontweight='bold')

        # -- Threshold exceedance brace --
        if threshold is not None:
            if exceed_direction == 'above':
                exceed_pct = np.sum(d > (threshold+1e-3)) / len(d) * 100
                y_lo, y_hi = threshold, d.max()
                mask = y_range >= threshold
            else:
                exceed_pct = np.sum(d < (threshold-1e-3)) / len(d) * 100
                y_lo, y_hi = d.min(), threshold
                mask = y_range <= threshold

            if exceed_pct > 0.1 and y_hi > y_lo and mask.any():
                brace_x = vio_cx + density_scaled[mask].max() + 0.02
                tip_x = _draw_curly_brace(ax, brace_x, y_lo, y_hi)
                ax.annotate(
                    f'{exceed_pct:.1f}%',
                    xy=(tip_x + 0.05, (y_lo + y_hi) / 2),
                    fontsize=14, fontweight='bold',
                    va='center', ha='left', color='black',
                    annotation_clip=False)

    # -- Reference lines & styling --
    _add_reference_lines(ax, y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=label_rotation,
                       ha='center', fontweight='bold')
    _apply_common_tick_params()
    _highlight_iter_opf_label(ax)

    plt.savefig(save_path, format='pdf', dpi=500, bbox_inches='tight')



# =============================================================================
# Network / Geographic Visualisation
# =============================================================================

def network_grid_vis(result, network, ncfile_slice, temperature,
                     ref_country=None, save_path=None):
    """Overlay OPF results on a geographic grid with a temperature heatmap.

    Plots branch conductor temperatures as coloured lines on top of a
    pcolormesh air-temperature background.  Buses experiencing significant
    load shedding (>0.1 % of total load) are highlighted with gold
    triangle markers.

    Parameters
    ----------
    result : dict
        OPF solution dictionary containing at least:

        - ``'con_temp'`` — conductor temperature per branch (2-D array).
        - ``'LS'`` — per-bus load shedding.
        - ``'PD'`` — per-bus active demand.
    network : pypsa.Network
        PyPSA network with bus coordinates and line topology.
    ncfile_slice : xarray.Dataset
        Weather data slice with ``'x'`` (lon) and ``'y'`` (lat) coords.
    temperature : array-like
        2-D air-temperature field for the background pcolormesh.
    ref_country : str or None
        If given, restrict load-shedding ratio calculation and bus
        highlighting to buses belonging to this country code
        (must match ``network.buses.country``).
    save_path : str or None
        If given, save the figure as a rasterised PDF at this path.
    """
    lon_grid = ncfile_slice['x'].data - 0.125
    lat_grid = ncfile_slice['y'].data - 0.125
    boundaries = [
        network.buses.x.min() - 0.125, network.buses.x.max() + 0.125,
        network.buses.y.min() - 0.125, network.buses.y.max() + 0.125,
    ]

    # --- Extract result vectors -------------------------------------------
    branch_temp = result['con_temp'].max(1)
    load_shedding = result['LS']
    bus_load = result['PD']

    # Load-shedding ratio (optionally scoped to a single country)
    if ref_country is not None:
        ref_index = network.buses.country.values == ref_country
        ls_ratio = load_shedding[ref_index] / bus_load[ref_index].sum() * 100
    else:
        ref_index = None
        ls_ratio = load_shedding / bus_load.sum() * 100

    # --- Figure & background heatmap (rasterised for compact PDF) ---------
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    pcolor = plt.pcolormesh(
        lon_grid, lat_grid, temperature, shading='auto',
        cmap='coolwarm', norm=plt.Normalize(vmin=20, vmax=50),
        alpha=0.5, zorder=0,
    )
    pcolor.set_rasterized(True)

    ocean = ax.add_feature(cfeature.OCEAN, zorder=1)
    ocean.set_rasterized(True)
    borders = ax.add_feature(cfeature.BORDERS, zorder=1)
    borders.set_rasterized(True)

    # --- Network lines (vector layer, coloured by conductor temp) ---------
    network.plot(
        bus_sizes=0,
        line_cmap=line_thermal_cmap_subtle_danger(N=6),
        line_norm=plt.Normalize(vmin=30, vmax=90),
        line_widths=2.75,
        line_colors=branch_temp,
        line_alpha=0.75,
        geomap=False,
        boundaries=boundaries,
    )

    # --- Bus markers ------------------------------------------------------
    bus_x = network.buses.x
    bus_y = network.buses.y
    ax.scatter(
        bus_x, bus_y, marker='.', c='gray',
        alpha=0.25, edgecolors=None, s=100, zorder=10,
    )

    # Highlight buses with significant load shedding (>0.1 %)
    buses_subset = (network.buses[ref_index] if ref_country is not None
                    else network.buses)
    for i, (_, bus) in enumerate(buses_subset.iterrows()):
        if ls_ratio[i] > 0.1:
            ax.scatter(
                bus_x[i], bus_y[i], marker='^',
                c='#FFD700', edgecolors='black',
                linewidths=2, s=300, zorder=1, alpha=1,
            )

    # --- Save -------------------------------------------------------------
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path, bbox_inches='tight', pad_inches=0.01,
            format='pdf', dpi=150,  # dpi only affects rasterised layers
        )
    plt.close()

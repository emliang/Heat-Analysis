#!/usr/bin/env python3
"""Draw the complete 180-mm main Fig. 3 from figure-level Source Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 96.0
MM_PER_INCH = 25.4
MEAN_COLOR = "#8B0000"
MEAN_MARKER_SIZE = 8.0
STAT_COLOR = "#5F5F5F"
PDF_METADATA = {
    "Creator": "plot_main_figure_03.py",
    "CreationDate": None,
    "ModDate": None,
}


def configure_fonts() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 5.8,
            "axes.labelsize": 7.0,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(top=False, right=False, direction="out", pad=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_linewidth(0.6)
        ax.spines[name].set_color("#555555")


def draw_boxes(
    ax: plt.Axes,
    observations: list[np.ndarray],
    positions: np.ndarray,
    colors: list[str],
    *,
    width: float,
) -> None:
    boxes = ax.boxplot(
        observations,
        positions=positions,
        widths=width,
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        medianprops={"color": STAT_COLOR, "linewidth": 0.7},
        whiskerprops={"color": STAT_COLOR, "linewidth": 0.6},
        capprops={"color": STAT_COLOR, "linewidth": 0.6},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.45)
        patch.set_alpha(0.9)


def population_kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.linspace(float(values.min()), float(values.max()), 300)
    if np.ptp(values) > 1e-12:
        density = stats.gaussian_kde(values)(grid)
    else:
        density = np.ones_like(grid)
    return grid, density


def add_mean_note(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    *,
    y_limits: tuple[float, float],
    placement: str,
    x_offset: float = -0.34,
) -> None:
    mean = float(np.mean(values))
    display = 0.0 if abs(mean) < 0.005 else mean
    span = y_limits[1] - y_limits[0]
    if placement == "lower-left":
        text_y = mean - 0.075 * span
        vertical_alignment = "top"
    elif placement == "upper-left":
        text_y = mean + 0.065 * span
        vertical_alignment = "bottom"
    else:
        raise ValueError(placement)
    ax.scatter(
        [x], [mean], s=MEAN_MARKER_SIZE, color=MEAN_COLOR, zorder=5, clip_on=True
    )
    ax.annotate(
        f"ave:\n{display:.2f}",
        xy=(x, mean),
        xytext=(x + x_offset, text_y),
        arrowprops={
            "arrowstyle": "-",
            "color": "#333333",
            "linewidth": 0.5,
            "linestyle": "--",
        },
        fontsize=6.0,
        fontweight="bold",
        ha="left",
        va=vertical_alignment,
        annotation_clip=False,
        zorder=6,
    )


def draw_curly_brace(ax: plt.Axes, x: float, y_lo: float, y_hi: float) -> float:
    width = 0.035
    tip = 0.025
    height = y_hi - y_lo
    middle = (y_hi + y_lo) / 2.0
    vertices = [
        (x, y_hi),
        (x + width, y_hi),
        (x + width, middle + height * 0.15),
        (x + width, middle + height * 0.05),
        (x + width + tip, middle),
        (x + width, middle - height * 0.05),
        (x + width, middle - height * 0.15),
        (x + width, y_lo),
        (x, y_lo),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(
        mpatches.PathPatch(
            MplPath(vertices, codes),
            facecolor="none",
            edgecolor="#333333",
            linewidth=0.55,
            capstyle="round",
            clip_on=False,
        )
    )
    return x + width + tip


def distribution_panel(
    ax: plt.Axes,
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    ylabel: str,
    y_limits: tuple[float, float],
    mean_placement: str,
    threshold: float | None = None,
    threshold_direction: str = "above",
    reference_label: str | None = None,
) -> None:
    positions = np.arange(len(labels), dtype=float)
    box_positions = positions - 0.10
    draw_boxes(ax, observations, box_positions, colors, width=0.24)
    sub_width = 0.25
    for x, box_x, values, color in zip(
        positions, box_positions, observations, colors, strict=True
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        centre = x + 0.08
        grid, density = population_kde(values)
        if float(density.max()) > 0:
            density = density / float(density.max()) * sub_width
        ax.fill_betweenx(
            grid,
            centre,
            centre + density,
            color=color,
            alpha=0.9,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            centre + density,
            grid,
            color=color,
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )
        add_mean_note(
            ax,
            float(box_x),
            values,
            y_limits=y_limits,
            placement=mean_placement,
        )

        if threshold is None:
            continue
        if threshold_direction == "above":
            percentage = float(np.mean(values > threshold + 1e-3) * 100.0)
            y_lo, y_hi = threshold, float(values.max())
            mask = grid >= threshold
        elif threshold_direction == "below":
            percentage = float(np.mean(values < threshold - 1e-3) * 100.0)
            y_lo, y_hi = float(values.min()), threshold
            mask = grid <= threshold
        else:
            raise ValueError(threshold_direction)
        if percentage > 0.1 and y_hi > y_lo and mask.any():
            brace_x = centre + float(density[mask].max()) + 0.018
            tip_x = draw_curly_brace(ax, brace_x, y_lo, y_hi)
            ax.text(
                tip_x + 0.032,
                (y_lo + y_hi) / 2.0,
                f"{percentage:.1f}%",
                fontsize=6.0,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=False,
            )

    if threshold is not None:
        if threshold_direction == "above":
            color = "#ee4b56"
            text_y = threshold + 1.5
            va = "bottom"
        else:
            color = "#3540e8"
            text_y = threshold - 1.0
            va = "top"
        ax.axhline(
            threshold,
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.78,
            zorder=0,
        )
        ax.text(
            0.01,
            text_y,
            reference_label,
            transform=ax.get_yaxis_transform(),
            fontsize=6.0,
            color=color,
            fontweight="bold",
            ha="left",
            va=va,
            multialignment="left",
        )

    ax.set_xlim(-0.56, len(labels) - 0.20)
    ax.set_ylim(*y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel(ylabel)
    style_axis(ax)


def runtime_panel(
    ax: plt.Axes,
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    y_limits: tuple[float, float],
) -> None:
    positions = np.arange(len(labels), dtype=float)
    draw_boxes(ax, observations, positions, colors, width=0.54)
    for x, values in zip(positions, observations, strict=True):
        add_mean_note(
            ax,
            float(x),
            values,
            y_limits=y_limits,
            placement="upper-left",
            x_offset=-0.35,
        )
    ax.set_xlim(-0.58, len(labels) - 0.42)
    ax.set_ylim(*y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Running Time (sec.)")
    style_axis(ax)


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.040,
        box.y1 + 0.012,
        label,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    source = args.source_dir.resolve()
    rules = json.loads((source / "Fig3_display_rules.json").read_text())
    scenarios = pd.read_csv(source / "Fig3cd_scenario_observations.csv")
    branches = pd.read_csv(source / "Fig3ab_scenario_line_observations.csv.gz")
    methods = list(rules["methods"])
    labels = [rules["method_labels"][method] for method in methods]
    colors = [rules["method_colors"][method] for method in methods]
    limits = {panel: tuple(values) for panel, values in rules["panel_limits"].items()}

    available_capacity = [
        branches[f"{method}__available_capacity_percent_of_nominal"].to_numpy(
            dtype=float
        )
        for method in methods
    ]
    line_temperature = [
        branches[f"{method}__line_temperature_c"].to_numpy(dtype=float)
        for method in methods
    ]
    load_shedding = [
        scenarios[f"{method}__load_shedding_percent"].to_numpy(dtype=float)
        for method in methods
    ]
    runtime = [
        scenarios[f"{method}__runtime_s"].to_numpy(dtype=float)
        for method in methods
    ]

    configure_fonts()
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    axes = {
        "a": fig.add_axes([0.065, 0.585, 0.405, 0.335]),
        "b": fig.add_axes([0.565, 0.585, 0.405, 0.335]),
        "c": fig.add_axes([0.065, 0.105, 0.405, 0.335]),
        "d": fig.add_axes([0.565, 0.105, 0.405, 0.335]),
    }

    distribution_panel(
        axes["a"],
        available_capacity,
        labels,
        colors,
        ylabel="Available Capacity (%)",
        y_limits=limits["a"],
        mean_placement="lower-left",
        threshold=float(rules["capacity_security_margin_percent"]),
        threshold_direction="below",
        reference_label="Security\nmargin",
    )
    axes["a"].set_yticks([50, 60, 70, 80, 90, 100])
    distribution_panel(
        axes["b"],
        line_temperature,
        labels,
        colors,
        ylabel="Line Temperature (°C)",
        y_limits=limits["b"],
        mean_placement="upper-left",
        threshold=float(rules["thermal_limit_c"]),
        threshold_direction="above",
        reference_label="Thermal\nlimit",
    )
    axes["b"].set_yticks([20, 40, 60, 80, 100, 120])
    distribution_panel(
        axes["c"],
        load_shedding,
        labels,
        colors,
        ylabel="Load Shedding (%)",
        y_limits=limits["c"],
        mean_placement="upper-left",
    )
    axes["c"].set_yticks([0.0, 0.2, 0.4])
    runtime_panel(
        axes["d"],
        runtime,
        labels,
        colors,
        y_limits=limits["d"],
    )
    axes["d"].set_yticks([0, 10, 20, 30, 40])
    for label, ax in axes.items():
        panel_label(fig, ax, label)

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.pdf,
        format="pdf",
        dpi=300,
        facecolor="white",
        metadata=PDF_METADATA,
    )
    fig.savefig(
        args.png,
        format="png",
        dpi=300,
        facecolor="white",
        metadata={"Software": "plot_main_figure_03.py"},
    )
    plt.close(fig)
    print(f"pdf={args.pdf}")
    print(f"png={args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

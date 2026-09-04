#!/usr/bin/env python3
"""Build the complete 180-mm Main Figure 1 candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


MM_PER_INCH = 25.4


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=5.8, fontweight="bold", labelpad=1.8)
    ax.set_ylabel(ylabel, fontsize=5.8, fontweight="bold", labelpad=2.0)
    ax.tick_params(axis="both", labelsize=5.0, width=0.55, length=2.2, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def add_panel_label(ax, label: str, x: float = -0.19, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        va="top",
        ha="left",
    )


def format_colorbar(colorbar, label: str, ticks=None) -> None:
    if ticks is not None:
        colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=5.0, width=0.5, length=2.0, pad=1.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label(label, fontsize=5.2, fontweight="bold", labelpad=1.2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--panel-d-image", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    pipeline_root = args.pipeline_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(pipeline_root / "code"))
    from ieee_heat_balance_figure_definition import evaluate_panels  # noqa: PLC0415
    from utils.heat_flow_utils import (  # noqa: PLC0415
        heat_banlance_equation,
        maximum_allowable_current,
    )

    panels = evaluate_panels(heat_banlance_equation, maximum_allowable_current)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.transparent": False,
        }
    )
    figure = plt.figure(figsize=(180.0 / MM_PER_INCH, 112.0 / MM_PER_INCH))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[0.52, 1.08],
        left=0.055,
        right=0.955,
        top=0.98,
        bottom=0.02,
        hspace=0.18,
    )
    top_grid = grid[0, 0].subgridspec(1, 3, wspace=0.42)

    def top_panel_axes(index: int):
        panel_grid = top_grid[0, index].subgridspec(
            1, 2, width_ratios=[1.0, 0.055], wspace=0.08
        )
        return figure.add_subplot(panel_grid[0, 0]), figure.add_subplot(
            panel_grid[0, 1]
        )

    ax_a, cax_a = top_panel_axes(0)
    air, current, temperature = panels["a"]
    x_grid, y_grid = np.meshgrid(air, current)
    filled = ax_a.contourf(x_grid, y_grid, temperature, 100, cmap="coolwarm")
    black = ax_a.contour(
        x_grid,
        y_grid,
        temperature,
        levels=[40, 60, 80, 100, 120, 140, 160],
        colors="#333333",
        linewidths=0.35,
    )
    # Keep all contour labels inside the compact panel instead of relying on
    # Matplotlib's edge-biased automatic placement.
    ax_a.clabel(
        black,
        inline=True,
        fontsize=5.0,
        fmt="%d",
        inline_spacing=2,
        manual=[
            (23.0, 328.0),
            (24.6, 720.0),
            (26.0, 948.5),
            (28.0, 1115.5),
            (31.0, 1247.0),
            (36.0, 1350.5),
            (43.0, 1434.5),
        ],
    )
    thermal = ax_a.contour(
        x_grid, y_grid, temperature, levels=[90], colors="red", linewidths=1.2
    )
    ax_a.clabel(
        thermal,
        inline=True,
        fontsize=5.0,
        fmt="%d",
        inline_spacing=2,
        manual=[(40.0, 905.5)],
    )
    style_axis(ax_a, "Air temperature (°C)", "Conductor current (A)")
    format_colorbar(
        figure.colorbar(filled, cax=cax_a),
        "Conductor temperature (°C)",
        ticks=[40, 80, 120, 160],
    )
    add_panel_label(ax_a, "a")

    ax_b, cax_b = top_panel_axes(1)
    wind, angle, temperature = panels["b"]
    x_grid, y_grid = np.meshgrid(wind, angle)
    filled = ax_b.contourf(x_grid, y_grid, temperature, 100, cmap="coolwarm")
    black = ax_b.contour(
        x_grid,
        y_grid,
        temperature,
        levels=[70, 80, 100, 110],
        colors="#333333",
        linewidths=0.35,
    )
    ax_b.clabel(black, inline=True, fontsize=5.0, fmt="%d", inline_spacing=2)
    thermal = ax_b.contour(
        x_grid, y_grid, temperature, levels=[90], colors="red", linewidths=1.2
    )
    ax_b.clabel(thermal, inline=True, fontsize=5.0, fmt="%d", inline_spacing=2)
    style_axis(ax_b, "Wind speed (m/s)", "Conductor–wind angle (°)")
    format_colorbar(
        figure.colorbar(filled, cax=cax_b),
        "Conductor temperature (°C)",
        ticks=[70, 80, 90, 100, 110],
    )
    add_panel_label(ax_b, "b")

    ax_c, cax_c = top_panel_axes(2)
    air, wind, maximum_current = panels["c"]
    x_grid, y_grid = np.meshgrid(air, wind)
    filled = ax_c.contourf(x_grid, y_grid, maximum_current, 100, cmap="viridis")
    black = ax_c.contour(
        x_grid,
        y_grid,
        maximum_current,
        levels=np.arange(700, 1500, 100),
        colors="#333333",
        linewidths=0.35,
    )
    ax_c.clabel(black, inline=True, fontsize=5.0, fmt="%d", inline_spacing=2)
    style_axis(ax_c, "Air temperature (°C)", "Wind speed (m/s)")
    format_colorbar(
        figure.colorbar(filled, cax=cax_c),
        "Maximum current limit (A)",
        ticks=[700, 900, 1100, 1300],
    )
    add_panel_label(ax_c, "c")

    ax_d = figure.add_subplot(grid[1, 0])
    panel_d = Image.open(args.panel_d_image.resolve()).convert("RGB")
    ax_d.imshow(panel_d)
    ax_d.set_axis_off()
    add_panel_label(ax_d, "d", x=-0.025, y=1.01)

    metadata = {
        "Title": "Figure 1 | Heat balance and segment-level Spain illustration",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_main_figure_01.py",
        "CreationDate": None,
        "ModDate": None,
    }
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, format="pdf", dpi=500, metadata=metadata)
    figure.savefig(
        args.output_png,
        format="png",
        dpi=300,
        metadata={"Software": "Matplotlib"},
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

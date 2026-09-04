#!/usr/bin/env python3
"""Reproduce Supplementary Fig. 1 from its compact plotted-data package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402
from shapely import wkt  # noqa: E402


def add_capacity_pie(ax, x, y, values, colors, radius):
    total = float(sum(values))
    if total <= 0:
        return
    start = 90.0
    for value, color in zip(values, colors):
        if value <= 0:
            continue
        theta = 360.0 * float(value) / total
        ax.add_patch(
            Wedge(
                (x, y),
                radius,
                start,
                start + theta,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=6,
            )
        )
        start += theta
    ax.add_patch(
        Wedge(
            (x, y),
            radius,
            0,
            360,
            facecolor="none",
            edgecolor="#3d3d3d",
            linewidth=0.35,
            zorder=7,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spain_grid_definition import line_width_from_gw, pie_radius_from_mw  # noqa: PLC0415

    source = args.source_dir
    buses = pd.read_csv(source / "spain_grid_buses.csv")
    lines = pd.read_csv(source / "spain_grid_ac_lines.csv")
    links = pd.read_csv(source / "spain_grid_dc_links.csv")
    capacity = pd.read_csv(source / "spain_grid_capacity_by_bus.csv")
    geometries = pd.read_csv(source / "spain_grid_context_geometries.csv")
    rules = json.loads((source / "spain_grid_plot_rules.json").read_text())

    category_order = rules["category_order"]
    category_palette = rules["category_palette"]
    bounds = rules["bounds"]
    context = gpd.GeoDataFrame(
        geometries[["geometry_id"]].copy(),
        geometry=geometries.geometry_wkt.map(wkt.loads),
        crs="EPSG:4326",
    )

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    fig = plt.figure(figsize=rules["canvas_inches"])
    gs = GridSpec(2, 1, height_ratios=[6.1, 0.68], hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    leg_ax = fig.add_subplot(gs[1, 0])
    leg_ax.axis("off")

    ax.set_facecolor("#b7dfea")
    context.plot(
        ax=ax,
        facecolor="#f7f7f5",
        edgecolor="#6f6f6f",
        linewidth=0.45,
        zorder=0,
    )
    ax.set_xlim(bounds["xmin"], bounds["xmax"])
    ax.set_ylim(bounds["ymin"], bounds["ymax"])

    for row in lines.itertuples(index=False):
        ax.plot(
            [row.x0, row.x1],
            [row.y0, row.y1],
            color=rules["ac_line_color"],
            linewidth=line_width_from_gw(float(row.s_nom_mva) / 1000.0),
            alpha=0.52,
            solid_capstyle="round",
            zorder=2,
        )
    for row in links.itertuples(index=False):
        ax.plot(
            [row.x0, row.x1],
            [row.y0, row.y1],
            color=rules["dc_link_color"],
            linewidth=line_width_from_gw(max(float(row.p_nom_mw) / 1000.0, 0.1)),
            alpha=0.85,
            solid_capstyle="round",
            zorder=3,
        )
    ax.scatter(
        buses.x,
        buses.y,
        s=5,
        color="#47525d",
        alpha=0.55,
        linewidths=0,
        zorder=4,
    )

    max_bus_capacity = float(capacity.total_capacity_mw.max())
    pie_colors = [category_palette[category] for category in category_order]
    for row in capacity.itertuples(index=False):
        values = [float(getattr(row, category)) for category in category_order]
        add_capacity_pie(
            ax,
            row.x,
            row.y,
            values,
            pie_colors,
            pie_radius_from_mw(row.total_capacity_mw, max_bus_capacity),
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("S")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    legend_heading = {
        "fontsize": 9.2,
        "fontfamily": "Arial",
        "fontweight": "bold",
        "va": "top",
    }
    legend_text = {"fontsize": 8.0, "fontfamily": "Arial", "va": "center"}
    x0 = 0.055
    leg_ax.text(x0, 0.94, "Capacity mix", transform=leg_ax.transAxes, **legend_heading)
    for category, y_pos in zip(category_order, [0.58, 0.36, 0.16]):
        leg_ax.scatter(
            [x0 + 0.020],
            [y_pos],
            s=32,
            color=category_palette[category],
            edgecolors="white",
            linewidths=0.35,
            transform=leg_ax.transAxes,
            clip_on=False,
        )
        leg_ax.text(
            x0 + 0.060,
            y_pos,
            category,
            ha="left",
            transform=leg_ax.transAxes,
            **legend_text,
        )

    x1 = 0.430
    leg_ax.text(x1, 0.94, "Bus capacity", transform=leg_ax.transAxes, **legend_heading)
    legend_capacity_scale = pie_radius_from_mw(10000, max_bus_capacity)
    for label, cap_mw, y_pos in (("10 GW", 10000, 0.58), ("5 GW", 5000, 0.36), ("1 GW", 1000, 0.16)):
        radius_ratio = pie_radius_from_mw(cap_mw, max_bus_capacity) / legend_capacity_scale
        leg_ax.scatter(
            [x1 + 0.026],
            [y_pos],
            s=58 * radius_ratio**2,
            facecolors="none",
            edgecolors="#4d4d4d",
            linewidths=0.65,
            transform=leg_ax.transAxes,
            clip_on=False,
        )
        leg_ax.text(x1 + 0.070, y_pos, label, ha="left", transform=leg_ax.transAxes, **legend_text)

    x2 = 0.700
    leg_ax.text(x2, 0.94, "Transmission", transform=leg_ax.transAxes, **legend_heading)
    for label, cap, color, y_pos in (
        ("AC 5 GW", 5.0, rules["ac_line_color"], 0.58),
        ("AC 2 GW", 2.0, rules["ac_line_color"], 0.36),
        ("DC link", 2.0, rules["dc_link_color"], 0.16),
    ):
        leg_ax.plot(
            [x2, x2 + 0.100],
            [y_pos, y_pos],
            color=color,
            linewidth=line_width_from_gw(cap),
            alpha=0.82,
            solid_capstyle="round",
            transform=leg_ax.transAxes,
            clip_on=False,
        )
        leg_ax.text(x2 + 0.125, y_pos, label, ha="left", transform=leg_ax.transAxes, **legend_text)

    metadata = {
        "Title": "Spanish grid representation",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_spain_grid_configuration.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(args.pdf, format="pdf", dpi=500, bbox_inches="tight", metadata=metadata)
    fig.savefig(
        args.png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

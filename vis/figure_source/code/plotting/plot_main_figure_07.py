#!/usr/bin/env python3
"""Draw the complete 180-mm main Fig. 7 from figure-level Source Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 118.0
MM_PER_INCH = 25.4
SCENARIO_ORDER = (
    "historical_reference",
    "historical_heatwave",
    "future_reference",
    "future_heatwave",
)


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
            "font.size": 6.0,
            "axes.labelsize": 7.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 6.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_axis(ax, *, grid: bool = True) -> None:
    ax.tick_params(top=False, right=False, direction="out", pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#555555")
    if grid:
        ax.grid(True, linewidth=0.35, alpha=0.22, color="#aaaaaa")
        ax.set_axisbelow(True)


def spatial_grid(table: pd.DataFrame, scenario: str):
    selected = table.loc[table["scenario"] == scenario]
    x_coordinates = (
        selected[["x_index", "x"]].drop_duplicates().sort_values("x_index")
    )
    y_coordinates = (
        selected[["y_index", "y"]].drop_duplicates().sort_values("y_index")
    )
    pivot = selected.pivot(
        index="y_index", columns="x_index", values="value"
    ).reindex(
        index=y_coordinates["y_index"], columns=x_coordinates["x_index"]
    )
    return (
        x_coordinates["x"].to_numpy(),
        y_coordinates["y"].to_numpy(),
        pivot.to_numpy(),
    )


def draw_panel_a(fig, source: Path, rules: dict) -> None:
    ax = fig.add_axes((0.070, 0.715, 0.880, 0.200))
    table = pd.read_csv(source / "Fig7a_national_hourly_profiles.csv")
    values = {
        scenario: group.sort_values("hour")["value"].to_numpy()
        for scenario, group in table.groupby("scenario")
    }
    palette = [plt.cm.coolwarm(index / 7) for index in range(8)]
    historical_color = palette[1]
    future_color = palette[-2]
    hours = np.arange(24)
    ax.fill_between(
        hours,
        values["historical_reference"],
        values["historical_heatwave"],
        alpha=0.12,
        color=historical_color,
    )
    ax.fill_between(
        hours,
        values["future_reference"],
        values["future_heatwave"],
        alpha=0.12,
        color=future_color,
    )
    line_specs = (
        ("historical_reference", historical_color, "-", 0.75),
        ("historical_heatwave", historical_color, "-.", 1.0),
        ("future_reference", future_color, "-", 0.75),
        ("future_heatwave", future_color, "-.", 1.0),
    )
    for scenario, colour, linestyle, alpha in line_specs:
        ax.plot(
            hours,
            values[scenario],
            color=colour,
            linestyle=linestyle,
            linewidth=1.25,
            alpha=alpha,
            label=rules["scenario_labels"][scenario],
        )
    margin = 0.06 * (table["value"].max() - table["value"].min())
    ax.set_xlim(0, 23)
    ax.set_ylim(table["value"].min() - margin, table["value"].max() + margin)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Air temperature (°C)")
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=7.0,
        columnspacing=1.2,
        handlelength=2.2,
    )
    style_axis(ax)


def draw_panel_b(fig, source: Path, rules: dict) -> None:
    spatial_path = source / "Fig7b_spatial_snapshot_fields.csv.gz"
    table = pd.read_csv(spatial_path)
    boundaries = np.linspace(
        float(rules["vmin"]),
        float(rules["vmax"]),
        int(rules["color_levels"]) + 1,
    )
    discrete_cmap = colors.ListedColormap(
        plt.cm.coolwarm(np.linspace(0, 1, int(rules["color_levels"])))
    )
    normalizer = colors.BoundaryNorm(boundaries, discrete_cmap.N)
    lefts = (0.055, 0.285, 0.515, 0.745)
    map_axes = []
    for left, scenario in zip(lefts, SCENARIO_ORDER, strict=True):
        ax = fig.add_axes(
            (left, 0.405, 0.200, 0.220),
            projection=ccrs.PlateCarree(),
        )
        xs, ys, values = spatial_grid(table, scenario)
        ax.pcolormesh(
            xs - 0.125,
            ys - 0.125,
            values,
            shading="auto",
            cmap=discrete_cmap,
            norm=normalizer,
            alpha=0.72,
            rasterized=False,
            zorder=0,
        )
        bounds = rules["bounds"]
        ax.set_extent(
            [bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.OCEAN, facecolor="#f1f3f5", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#444444", zorder=2)
        ax.coastlines(linewidth=0.45, color="#444444", zorder=2)
        ax.set_title(rules["scenario_labels"][scenario], pad=1.5, fontweight="bold")
        ax.set_axis_off()
        map_axes.append(ax)

    colorbar_ax = fig.add_axes((0.160, 0.375, 0.680, 0.018))
    scalar = cm.ScalarMappable(cmap=discrete_cmap, norm=normalizer)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_ax,
        orientation="horizontal",
        extend="both",
        extendfrac=0.025,
    )
    colorbar.set_ticks(boundaries)
    colorbar.set_ticklabels([f"{value:.0f}" for value in boundaries])
    colorbar.set_label("Air temperature (°C)", fontsize=6.0, fontweight="bold", labelpad=1.5)
    colorbar.ax.tick_params(labelsize=6.0, width=0.5, length=2.0, pad=1.0)
    colorbar.outline.set_linewidth(0.55)


def draw_panel_c(fig, source: Path, rules: dict) -> None:
    table = pd.read_csv(source / "Fig7c_sampled_regional_hourly_profiles.csv")
    future_color = plt.cm.coolwarm(6 / 7)
    lefts = (0.070, 0.370, 0.670)
    axes = []
    for left, region_number in zip(lefts, rules["displayed_region_numbers"], strict=True):
        ax = fig.add_axes((left, 0.055, 0.255, 0.190))
        region = table.loc[table["region_number"] == region_number]
        reference = region.loc[region["curve"] == "future_reference"].sort_values("hour")
        ax.plot(
            reference["hour"],
            reference["value"],
            color=future_color,
            linewidth=1.25,
            alpha=0.75,
        )
        for _, sample in region.loc[region["curve"] == "future_heatwave"].groupby("sample_rank"):
            sample = sample.sort_values("hour")
            ax.plot(
                sample["hour"],
                sample["value"],
                color=future_color,
                linestyle="--",
                linewidth=0.75,
                alpha=0.82,
            )
        ax.text(
            0.035,
            0.95,
            f"Region {region_number}",
            transform=ax.transAxes,
            fontsize=6.0,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.set_xlim(0, 23)
        ax.set_ylim(rules["regional_axis_min_c"], rules["regional_axis_max_c"])
        ax.set_xticks(range(0, 24, 4))
        ax.set_xlabel("Hour")
        if region_number == rules["displayed_region_numbers"][0]:
            ax.set_ylabel("Air temperature (°C)")
        else:
            ax.tick_params(labelleft=False)
        style_axis(ax)
        axes.append(ax)

    handles = [
        Line2D([], [], color=future_color, linewidth=1.25, alpha=0.75),
        Line2D([], [], color=future_color, linestyle="--", linewidth=1.0, alpha=0.82),
    ]
    fig.legend(
        handles,
        ["Future reference", "Projected heatwaves"],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.262),
        frameon=False,
        fontsize=7.0,
        columnspacing=1.4,
        handlelength=2.2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    configure_fonts()
    source = args.source_dir.resolve()
    rules = json.loads((source / "Fig7_display_rules.json").read_text())
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    draw_panel_a(fig, source, rules)
    draw_panel_b(fig, source, rules)
    draw_panel_c(fig, source, rules)
    for label, x, y in (("a", 0.020, 0.955), ("b", 0.020, 0.665), ("c", 0.020, 0.286)):
        fig.text(x, y, label, fontsize=7.0, fontweight="bold", ha="left", va="top")

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Figure 7 - Projected heatwave generation in Spain",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_main_figure_07.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(args.pdf, format="pdf", dpi=300, metadata=metadata)
    fig.savefig(args.png, format="png", dpi=300, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

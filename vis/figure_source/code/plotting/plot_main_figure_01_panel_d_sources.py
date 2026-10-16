#!/usr/bin/env python3
"""Reproduce the scientific source components used in Main Fig. 1d."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def save_figure(figure, pdf: Path, png: Path | None = None) -> None:
    metadata = {
        "Title": "Main Figure 1d source component",
        "Author": "Liang et al.",
        "Creator": (
            "publication_pipeline/code/plotting/"
            "plot_main_figure_01_panel_d_sources.py"
        ),
        "CreationDate": None,
        "ModDate": None,
    }
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf, format="pdf", dpi=500, bbox_inches="tight", metadata=metadata)
    if png is not None:
        figure.savefig(
            png,
            format="png",
            dpi=200,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
    plt.close(figure)


def weather_arrays(weather: pd.DataFrame) -> tuple[np.ndarray, ...]:
    ordered = weather.sort_values(["y_index", "x_index"])
    nx = int(ordered.x_index.max()) + 1
    ny = int(ordered.y_index.max()) + 1
    shape = (ny, nx)
    return (
        ordered.longitude.to_numpy().reshape(shape)[0],
        ordered.latitude.to_numpy().reshape(shape)[:, 0],
        ordered.map_longitude.to_numpy().reshape(shape)[0],
        ordered.map_latitude.to_numpy().reshape(shape)[:, 0],
        ordered.air_temperature_c.to_numpy().reshape(shape),
    )


def plot_network_map(source: Path, output: Path, rules: dict) -> None:
    buses = pd.read_csv(source / "Fig1d_network_buses.csv")
    lines = pd.read_csv(source / "Fig1d_network_lines.csv")
    links = pd.read_csv(source / "Fig1d_network_links.csv")
    weather = pd.read_csv(source / "Fig1d_temperature_grid.csv")
    _, _, map_x, map_y, temperature = weather_arrays(weather)

    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    axis.pcolormesh(
        map_x,
        map_y,
        temperature,
        shading="auto",
        cmap="coolwarm",
        norm=plt.Normalize(*rules["map_temperature_limits_c"]),
        alpha=0.7,
        zorder=0,
        transform=ccrs.PlateCarree(),
    )
    axis.add_feature(cfeature.OCEAN, zorder=1)
    axis.add_feature(cfeature.BORDERS, linewidth=0.7, zorder=3)
    for row in lines.itertuples(index=False):
        axis.plot(
            [row.x0, row.x1],
            [row.y0, row.y1],
            color="black",
            linewidth=2,
            alpha=0.25,
            zorder=4,
            transform=ccrs.PlateCarree(),
        )
    for row in links.itertuples(index=False):
        axis.plot(
            [row.x0, row.x1],
            [row.y0, row.y1],
            color="#77b255",
            linewidth=1.2,
            alpha=0.8,
            zorder=4,
            transform=ccrs.PlateCarree(),
        )
    axis.scatter(
        buses.x,
        buses.y,
        s=5,
        color="lightgray",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.75,
        zorder=5,
        transform=ccrs.PlateCarree(),
    )
    xmin, xmax, ymin, ymax = rules["map_boundaries"]
    axis.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
    axis.set_axis_off()
    save_figure(
        figure,
        output / "Fig1d_network_map.pdf",
        output / "Fig1d_network_map.png",
    )


def plot_selected_line(source: Path, output: Path, rules: dict) -> None:
    weather = pd.read_csv(source / "Fig1d_temperature_grid.csv")
    segments = pd.read_csv(source / "Fig1d_selected_line_segments.csv")
    x, y, _, _, temperature = weather_arrays(weather)
    x0 = float(segments.start_longitude.iloc[0])
    y0 = float(segments.start_latitude.iloc[0])
    x1 = float(segments.end_longitude.iloc[-1])
    y1 = float(segments.end_latitude.iloc[-1])
    ix0 = min(int(np.argmin(abs(x - x0))), int(np.argmin(abs(x - x1))))
    ix1 = max(int(np.argmin(abs(x - x0))), int(np.argmin(abs(x - x1))))
    iy0 = min(int(np.argmin(abs(y - y0))), int(np.argmin(abs(y - y1))))
    iy1 = max(int(np.argmin(abs(y - y0))), int(np.argmin(abs(y - y1))))

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.pcolormesh(
        x[ix0 - 1 : ix1 + 2],
        y[iy0 - 1 : iy1 + 2],
        temperature[iy0 - 1 : iy1 + 2, ix0 - 1 : ix1 + 2],
        vmin=rules["line_temperature_limits_c"][0],
        vmax=rules["line_temperature_limits_c"][1],
        shading="nearest",
        cmap="coolwarm",
        alpha=0.7,
    )
    axis.scatter([x0, x1], [y0, y1], s=1000, color="gray", zorder=5)
    colour_map = plt.cm.viridis
    normalizer = mcolors.Normalize(*rules["line_capacity_limits_percent"])
    for row in segments.itertuples(index=False):
        axis.plot(
            [row.start_longitude, row.end_longitude],
            [row.start_latitude, row.end_latitude],
            color=colour_map(normalizer(row.available_capacity_percent)),
            linewidth=8,
            alpha=0.9,
            solid_capstyle="round",
            zorder=6,
        )
        if row.segment_index > 1:
            axis.scatter(
                row.start_longitude,
                row.start_latitude,
                color="white",
                marker="o",
                s=20,
                alpha=0.9,
                zorder=10,
            )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    save_figure(
        figure,
        output / "Fig1d_selected_line.pdf",
        output / "Fig1d_selected_line.png",
    )


def plot_colorbar(
    output: Path,
    filename: str,
    cmap,
    boundaries: np.ndarray,
    label: str,
    orientation: str,
    figsize: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=figsize)
    normalizer = mcolors.BoundaryNorm(boundaries, cmap.N)
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        cax=axis,
        orientation=orientation,
        extend="both",
        ticks=boundaries,
    )
    colorbar.set_label(label, fontweight="bold")
    save_figure(figure, output / filename)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rules = json.loads((source / "Fig1d_plot_rules.json").read_text())

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    plot_network_map(source, output, rules)
    plot_selected_line(source, output, rules)
    plot_colorbar(
        output,
        "Fig1d_available_capacity_colorbar.pdf",
        plt.cm.viridis,
        np.linspace(50, 100, 6),
        "Available capacity (%)",
        "horizontal",
        (7, 0.25),
    )
    plot_colorbar(
        output,
        "Fig1d_air_temperature_colorbar.pdf",
        plt.cm.coolwarm,
        np.linspace(20, 50, 7),
        "Air temperature (°C)",
        "vertical",
        (0.25, 5),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

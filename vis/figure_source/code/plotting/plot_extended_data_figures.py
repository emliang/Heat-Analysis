#!/usr/bin/env python3
"""Draw complete 180-mm Extended Data Figs. 1 and 2 from Source Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib import patches as mpatches  # noqa: E402
from scipy import stats  # noqa: E402


MM_PER_INCH = 25.4
WIDTH_MM = 180.0
ED1_HEIGHT_MM = 124.0
ED2_HEIGHT_MM = 143.0
ED1_PACKAGE = "Extended_Data_Figure_01_spain_spatiotemporal"
ED2_PACKAGE = "Extended_Data_Figure_02_spain_cross_border_maps"
PDF_METADATA = {
    "Author": "Liang et al.",
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
            "font.size": 5.5,
            "axes.titlesize": 6.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.5,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.55,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def line_temperature_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "line_temperature",
        ["#4cb040", "#6cc050", "#b8c840", "#d8b830", "#FF7F00", "#FF0000", "#8B00FF"],
        N=256,
    )


def add_panel_label(ax, label: str, *, x: float = -0.14, y: float = 1.09) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def mean_text(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def draw_curly_brace(ax, x: float, y_lo: float, y_hi: float, width: float = 0.05) -> float:
    if y_hi <= y_lo:
        return x
    tip = width * 0.65
    height = y_hi - y_lo
    middle = (y_hi + y_lo) / 2
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
            linewidth=0.45,
            clip_on=False,
        )
    )
    return x + width + tip


def distribution_limits(observations: list[np.ndarray], metric: str, threshold: float | None) -> tuple[float, float]:
    values = np.concatenate(observations)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if metric == "available_capacity_percent_of_nominal":
        return min(45.0, minimum - 2.0), 104.0
    if metric == "line_temperature_c":
        return min(25.0, minimum - 2.0), max(96.0, maximum + 2.0)
    span = max(maximum - minimum, 0.05)
    lower = 0.0 if minimum >= 0 else minimum - 0.05 * span
    return lower, maximum + 0.22 * span


def draw_distribution(
    ax,
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    metric: str,
    *,
    threshold: float | None = None,
    exceed_direction: str = "above",
) -> None:
    positions = np.arange(len(labels), dtype=float)
    box_width = 0.22
    density_width = 0.22
    y_min, y_max = distribution_limits(observations, metric, threshold)
    y_span = y_max - y_min
    for index, (raw_values, color) in enumerate(zip(observations, colors, strict=True)):
        values = np.asarray(raw_values, dtype=float)
        box_x = positions[index] - 0.10
        violin_x = positions[index] + 0.06
        box = ax.boxplot(
            [values],
            positions=[box_x],
            widths=box_width,
            patch_artist=True,
            whis=1.5,
            showfliers=False,
            zorder=3,
            medianprops={"color": "dimgray", "linewidth": 0.65},
            whiskerprops={"color": "dimgray", "linewidth": 0.55},
            capprops={"color": "dimgray", "linewidth": 0.55},
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_edgecolor("white")
        box["boxes"][0].set_linewidth(0.4)
        box["boxes"][0].set_alpha(0.9)

        value_grid = np.linspace(values.min(), values.max(), 300)
        if np.ptp(values) > 1e-12:
            density = stats.gaussian_kde(values)(value_grid)
            density = density / density.max() * density_width
        else:
            density = np.full_like(value_grid, density_width * 0.05)
        ax.fill_betweenx(
            value_grid,
            violin_x,
            violin_x + density,
            color=color,
            alpha=0.9,
            linewidth=0,
            zorder=2,
        )
        ax.plot(violin_x + density, value_grid, color=color, linewidth=0.55, zorder=2)

        mean = float(values.mean())
        ax.scatter(box_x, mean, marker="o", color="#8B0000", s=4.0, zorder=4)
        q1, q3 = np.percentile(values, [25, 75])
        upper_fence = q3 + 1.5 * (q3 - q1)
        upper_whisker = float(values[values <= upper_fence].max())
        note_y = max(upper_whisker + 0.06 * y_span, mean + 0.09 * y_span)
        note_va = "bottom"
        if note_y > y_max - 0.05 * y_span:
            note_y = max(y_min + 0.06 * y_span, mean - 0.12 * y_span)
            note_va = "top"
        ax.annotate(
            f"ave:\n{mean_text(mean)}",
            xy=(box_x, mean),
            xytext=(positions[index] - 0.40, note_y),
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.4, "linestyle": "--"},
            fontsize=5.0,
            fontweight="bold",
            ha="left",
            va=note_va,
            annotation_clip=False,
            zorder=5,
        )

        if threshold is not None:
            if exceed_direction == "above":
                percentage = float(np.mean(values > threshold + 1e-3) * 100)
                y_lo, y_hi = threshold, float(values.max())
                mask = value_grid >= threshold
            else:
                percentage = float(np.mean(values < threshold - 1e-3) * 100)
                y_lo, y_hi = float(values.min()), threshold
                mask = value_grid <= threshold
            if percentage > 0.1 and y_hi > y_lo and mask.any():
                brace_x = violin_x + float(density[mask].max()) + 0.025
                tip_x = draw_curly_brace(ax, brace_x, y_lo, y_hi)
                ax.text(
                    tip_x + 0.025,
                    (y_lo + y_hi) / 2,
                    f"{percentage:.1f}%",
                    fontsize=5.2,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    clip_on=False,
                )

    if threshold is not None and metric == "line_temperature_c":
        ax.axhline(threshold, color="#D62728", linestyle="--", linewidth=0.8, alpha=0.75)
        ax.text(
            0.01,
            threshold + 0.018 * y_span,
            "Thermal\nlimit",
            transform=ax.get_yaxis_transform(),
            color="#D62728",
            fontsize=5.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    if threshold is not None and metric == "available_capacity_percent_of_nominal":
        ax.axhline(threshold, color="#3B4CFF", linestyle="--", linewidth=0.8, alpha=0.75)
        ax.text(
            0.01,
            threshold - 0.018 * y_span,
            "Security\nmargin",
            transform=ax.get_yaxis_transform(),
            color="#3B4CFF",
            fontsize=5.0,
            fontweight="bold",
            ha="left",
            va="top",
        )
    # Keep threshold annotations inside each panel, including the last group.
    ax.set_xlim(-0.55, len(labels) - 0.12)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=7.0)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.tick_params(top=False, right=False, direction="out", pad=1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.grid(False)


def draw_map(ax, buses, lines, branch_values, bus_values, weather_values, display) -> None:
    x_count = int(weather_values["x_index"].max()) + 1
    y_count = int(weather_values["y_index"].max()) + 1
    x_grid = weather_values.sort_values("x_index").drop_duplicates("x_index")["longitude_plot"].to_numpy()
    y_grid = weather_values.sort_values("y_index").drop_duplicates("y_index")["latitude_plot"].to_numpy()
    temperature = (
        weather_values.sort_values(["y_index", "x_index"])["air_temperature_c"]
        .to_numpy()
        .reshape(y_count, x_count)
    )
    field = ax.pcolormesh(
        x_grid,
        y_grid,
        temperature,
        shading="auto",
        cmap="coolwarm",
        norm=plt.Normalize(vmin=display["air_temperature_min_c"], vmax=display["air_temperature_max_c"]),
        alpha=display["air_temperature_alpha"],
        zorder=0,
        transform=ccrs.PlateCarree(),
    )
    field.set_rasterized(True)
    ax.set_facecolor("#DCEEF4")
    try:
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#666666", zorder=1)
        ax.coastlines(linewidth=0.25, color="#666666", zorder=1)
    except Exception:
        pass

    merge_keys = ["branch_index"]
    if "configuration_id" in lines and "configuration_id" in branch_values:
        merge_keys.insert(0, "configuration_id")
    merged_lines = lines.merge(branch_values, on=merge_keys, validate="one_to_one")
    segments = merged_lines[["longitude_0", "latitude_0", "longitude_1", "latitude_1"]].to_numpy(dtype=float).reshape(-1, 2, 2)
    collection = LineCollection(
        segments,
        cmap=line_temperature_cmap(),
        norm=plt.Normalize(vmin=display["line_temperature_min_c"], vmax=display["line_temperature_max_c"]),
        linewidths=0.42,
        alpha=0.82,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    collection.set_array(merged_lines["line_temperature_c"].to_numpy(dtype=float))
    ax.add_collection(collection)
    ax.scatter(
        buses["longitude"],
        buses["latitude"],
        marker=".",
        c="#555555",
        alpha=0.35,
        edgecolors="none",
        s=1.2,
        zorder=10,
        transform=ccrs.PlateCarree(),
    )
    bus_keys = ["bus_index"]
    if "configuration_id" in buses and "configuration_id" in bus_values:
        bus_keys.insert(0, "configuration_id")
    marked = buses.merge(bus_values, on=bus_keys, validate="one_to_one")
    marked = marked[marked["display_marker"].astype(bool)]
    if not marked.empty:
        ax.scatter(
            marked["longitude"],
            marked["latitude"],
            marker="^",
            c="#FFD700",
            edgecolors="black",
            linewidths=0.25,
            s=8.0,
            zorder=11,
            transform=ccrs.PlateCarree(),
        )
    offset = float(display["coordinate_offset_degrees"])
    ax.set_extent(
        [
            float(buses["longitude"].min()) - offset,
            float(buses["longitude"].max()) + offset,
            float(buses["latitude"].min()) - offset,
            float(buses["latitude"].max()) + offset,
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.set_axis_off()


def add_colorbar(fig, ax, spec: dict, kind: str) -> None:
    cmap = plt.get_cmap("coolwarm") if kind == "air_temperature" else line_temperature_cmap()
    boundaries = np.linspace(float(spec["vmin"]), float(spec["vmax"]), int(spec["levels"]) + 1)
    discrete = mcolors.ListedColormap(cmap(np.linspace(0, 1, int(spec["levels"]))))
    norm = mcolors.BoundaryNorm(boundaries, discrete.N)
    scalar = plt.cm.ScalarMappable(cmap=discrete, norm=norm)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        cax=ax,
        orientation="horizontal",
        alpha=float(spec["alpha"]),
        extend="both",
        extendfrac=0.04,
    )
    colorbar.set_label(spec["label"], fontsize=5.5, fontweight="bold", labelpad=1.0)
    colorbar.set_ticks(boundaries)
    colorbar.set_ticklabels([f"{value:.0f}" for value in boundaries])
    colorbar.ax.tick_params(labelsize=5.0, width=0.4, length=1.8, pad=1.0)
    colorbar.outline.set_linewidth(0.45)


def add_colorbar_pair(
    fig,
    specs: list[dict],
    figure_height_mm: float,
    *,
    bottom: float = 0.045,
) -> None:
    # Match the long, slender colour bars used in the submitted composite figures.
    bar_height = 2.2 / figure_height_mm
    axes = [
        fig.add_axes([0.085, bottom, 0.420, bar_height]),
        fig.add_axes([0.570, bottom, 0.420, bar_height]),
    ]
    for ax, spec in zip(axes, specs, strict=True):
        add_colorbar(fig, ax, spec, spec["kind"])


def save_figure(fig, pdf: Path, png: Path, title: str) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**PDF_METADATA, "Title": title, "Creator": "plot_extended_data_figures.py"}
    fig.savefig(pdf, format="pdf", dpi=300, metadata=metadata)
    fig.savefig(png, format="png", dpi=300, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def plot_ed1(source: Path, pdf: Path, png: Path) -> tuple[Path, Path]:
    metadata = json.loads((source / "ExtendedDataFig1_plot_metadata.json").read_text())
    scenarios = pd.read_csv(source / "ExtendedDataFig1_scenario_observations.csv")
    branches = pd.read_csv(source / "ExtendedDataFig1_scenario_branch_observations.csv.gz")
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, ED1_HEIGHT_MM / MM_PER_INCH), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.0],
        left=0.075,
        right=0.985,
        bottom=0.040,
        top=0.960,
        hspace=0.28,
        wspace=0.30,
    )
    distribution_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    panel_specs = [
        ("a", "future_hour", metadata["future_hours"], "branch", "available_capacity_percent_of_nominal", "Available Capacity (%)", metadata["colors"]["available_capacity"][-4:][::-1], 70.0, "below"),
        ("b", "future_year", metadata["future_years"], "branch", "available_capacity_percent_of_nominal", "Available Capacity (%)", metadata["colors"]["available_capacity"], 70.0, "below"),
        ("c", "future_hour", metadata["future_hours"], "branch", "line_temperature_c", "Line Temperature (°C)", metadata["colors"]["line_temperature"][-4:][::-1], 90.0, "above"),
        ("d", "future_year", metadata["future_years"], "branch", "line_temperature_c", "Line Temperature (°C)", metadata["colors"]["line_temperature"], 90.0, "above"),
        ("e", "future_hour", metadata["future_hours"], "scenario", "load_shedding_percent", "Load Shedding (%)", metadata["colors"]["load_shedding"][-4:][::-1], None, "above"),
        ("f", "future_year", metadata["future_years"], "scenario", "load_shedding_percent", "Load Shedding (%)", metadata["colors"]["load_shedding"], None, "above"),
    ]
    for ax, spec in zip(distribution_axes, panel_specs, strict=True):
        panel, group_field, group_values, level, value_field, ylabel, colors, threshold, direction = spec
        observations = []
        for group_value in group_values:
            ids = scenarios.loc[scenarios[group_field] == group_value, "scenario_id"]
            frame = branches if level == "branch" else scenarios
            observations.append(
                frame.loc[frame["scenario_id"].isin(ids), value_field].to_numpy(dtype=float)
            )
        labels = metadata["hour_labels"] if group_field == "future_hour" else [str(value) for value in group_values]
        draw_distribution(
            ax,
            observations,
            labels,
            colors,
            ylabel,
            value_field,
            threshold=threshold,
            exceed_direction=direction,
        )
        add_panel_label(ax, panel, x=-0.13, y=1.04)

    save_figure(fig, pdf, png, "Extended Data Figure 1 - Spanish spatiotemporal stress")
    return pdf, png


def plot_ed2(source: Path, pdf: Path, png: Path) -> tuple[Path, Path]:
    metadata = json.loads((source / "ExtendedDataFig2_plot_metadata.json").read_text())
    snapshots = pd.read_csv(source / "ExtendedDataFig2_map_snapshots.csv")
    map_buses = pd.read_csv(source / "ExtendedDataFig2_map_bus_observations.csv")
    map_branches = pd.read_csv(source / "ExtendedDataFig2_map_branch_observations.csv")
    map_weather = pd.read_csv(source / "ExtendedDataFig2_map_weather_fields.csv.gz")
    buses = pd.read_csv(source / "ExtendedDataFig2_represented_network_buses.csv")
    lines = pd.read_csv(source / "ExtendedDataFig2_represented_network_lines.csv")

    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, ED2_HEIGHT_MM / MM_PER_INCH), facecolor="white")
    row_ratios = []
    offset = float(metadata["map_display"]["coordinate_offset_degrees"])
    for configuration_id in metadata["configuration_ids"]:
        configuration_buses = buses[buses.configuration_id == configuration_id]
        longitude_span = float(configuration_buses.longitude.max() - configuration_buses.longitude.min()) + 2 * offset
        latitude_span = float(configuration_buses.latitude.max() - configuration_buses.latitude.min()) + 2 * offset
        row_ratios.append(latitude_span / longitude_span)

    grid = fig.add_gridspec(
        4,
        12,
        height_ratios=[*row_ratios, 0.085],
        left=0.085,
        right=0.99,
        bottom=0.045,
        top=0.97,
        hspace=0.15,
        wspace=0.15,
    )
    letters = iter("abcdefghi")
    years = [2026, 2028, 2030]
    configuration_display_labels = {
        "ES": "Spain",
        "ES-PT": "Spain ↔ Portugal",
        "ES-FR": "Spain ↔ France",
    }
    for row, configuration_id in enumerate(metadata["configuration_ids"]):
        for column, year in enumerate(years):
            ax = fig.add_subplot(grid[row, column * 4 : column * 4 + 4], projection=ccrs.PlateCarree())
            snapshot = snapshots[
                (snapshots.configuration_id == configuration_id)
                & (pd.to_datetime(snapshots.future_heatwave_datetime).dt.year == year)
            ].iloc[0]
            draw_map(
                ax,
                buses[buses.configuration_id == configuration_id],
                lines[lines.configuration_id == configuration_id],
                map_branches[map_branches.snapshot_id == snapshot.snapshot_id],
                map_buses[map_buses.snapshot_id == snapshot.snapshot_id],
                map_weather[map_weather.snapshot_id == snapshot.snapshot_id],
                metadata["map_display"],
            )
            add_panel_label(ax, next(letters), x=-0.035, y=1.015)
            if row == 0:
                ax.set_title(str(year), fontsize=7.0, fontweight="bold", pad=2.0)
            if column == 0:
                ax.text(
                    -0.10,
                    0.50,
                    configuration_display_labels[configuration_id],
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=7.0,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
    add_colorbar_pair(fig, metadata["colorbars"], ED2_HEIGHT_MM)
    save_figure(fig, pdf, png, "Extended Data Figure 2 - Spanish cross-border maps")
    return pdf, png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--figure", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    configure_fonts()
    outputs = []
    if args.source_dir is not None:
        if args.figure == "all":
            parser.error("--source-dir requires --figure 1 or --figure 2")
        if args.output_pdf is None or args.output_png is None:
            parser.error("--source-dir requires --output-pdf and --output-png")
        source = args.source_dir.resolve()
        pdf = args.output_pdf.resolve()
        png = args.output_png.resolve()
        if args.figure == "1":
            outputs.append(plot_ed1(source, pdf, png))
        else:
            outputs.append(plot_ed2(source, pdf, png))
    else:
        if args.pipeline_root is None:
            parser.error("provide either --pipeline-root or --source-dir")
        pipeline = args.pipeline_root.resolve()
        if args.figure in {"1", "all"}:
            package = pipeline / "extended_data_track" / ED1_PACKAGE
            outputs.append(
                plot_ed1(
                    package / "source_data" / "unpacked",
                    package / "artwork" / "Extended_Data_Figure_1_spain_spatiotemporal.pdf",
                    package / "preview" / "Extended_Data_Figure_1_spain_spatiotemporal.png",
                )
            )
        if args.figure in {"2", "all"}:
            package = pipeline / "extended_data_track" / ED2_PACKAGE
            outputs.append(
                plot_ed2(
                    package / "source_data" / "unpacked",
                    package / "artwork" / "Extended_Data_Figure_2_spain_cross_border_maps.pdf",
                    package / "preview" / "Extended_Data_Figure_2_spain_cross_border_maps.png",
                )
            )
    print(json.dumps({"artwork": [[str(path) for path in pair] for pair in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reproduce Supplementary Fig. 17 panels from its compact package only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib import patches as mpatches  # noqa: E402
from scipy import stats  # noqa: E402


PDF_METADATA = {
    "Creator": "publication_pipeline/code/plotting/plot_spain_spatiotemporal.py",
    "CreationDate": None,
    "ModDate": None,
}


def configure_matplotlib() -> None:
    sns.set_theme(style="whitegrid")
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pdf(fig, path: Path, *, dpi: int = 500, pad_inches: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def draw_curly_brace(ax, x: float, y_lo: float, y_hi: float) -> float:
    width = 0.06
    tip = 0.04
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
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
    ]
    ax.add_patch(
        mpatches.PathPatch(
            MplPath(vertices, codes),
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            capstyle="round",
            joinstyle="miter",
            clip_on=False,
        )
    )
    return x + width + tip


def box_violin_plot(
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    output: Path,
    *,
    threshold: float | None = None,
    exceed_direction: str = "above",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = np.arange(len(labels))
    category_width = 1.0
    sub_width = category_width / 4

    for index, raw_values in enumerate(observations):
        values = np.asarray(raw_values, dtype=float)
        box_centre = positions[index] - category_width / 10
        violin_centre = positions[index] + category_width / 10
        box = ax.boxplot(
            [values],
            positions=[box_centre],
            widths=sub_width,
            patch_artist=True,
            zorder=2,
            showfliers=False,
            medianprops={"color": "none", "linewidth": 1.5},
        )
        box["boxes"][0].set(facecolor=colors[index], alpha=0.9, edgecolor="white")
        ax.scatter(box_centre, values.mean(), marker="o", color="darkred", s=40, zorder=3)

        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = values[(values < lower_fence) | (values > upper_fence)]
        count = min(20, len(outliers))
        if count:
            generator = np.random.default_rng(1700 + index)
            selected = generator.choice(len(outliers), count, replace=False)
            jitter = (generator.random(count) - 0.5) * sub_width * 0.3
            ax.scatter(
                np.full(count, box_centre) + jitter,
                outliers[selected],
                s=18,
                c="white",
                edgecolors=colors[index],
                zorder=3,
                alpha=0.9,
            )

        value_grid = np.linspace(values.min(), values.max(), 300)
        if np.ptp(values) > 1e-12:
            density = stats.gaussian_kde(values)(value_grid)
            density = density / density.max() * sub_width
        else:
            density = np.full_like(value_grid, sub_width * 0.05)
        ax.fill_betweenx(
            value_grid,
            violin_centre,
            violin_centre + density,
            color=colors[index],
            alpha=0.9,
            zorder=1,
        )

        upper_whisker = min(upper_fence, values.max())
        annotation_y = upper_whisker + (ax.get_ylim()[1] - upper_whisker) * 0.1
        ax.annotate(
            f"ave:\n{values.mean():.2f}",
            xy=(box_centre, values.mean()),
            xytext=(box_centre - category_width * 0.4, annotation_y * 0.8),
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7, "linestyle": "--"},
            fontsize=16,
            ha="left",
            va="bottom",
            fontweight="bold",
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
                brace_x = violin_centre + density[mask].max() + 0.02
                tip_x = draw_curly_brace(ax, brace_x, y_lo, y_hi)
                ax.annotate(
                    f"{percentage:.1f}%",
                    xy=(tip_x + 0.05, (y_lo + y_hi) / 2),
                    fontsize=14,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    color="black",
                    annotation_clip=False,
                )

    if threshold is not None and "Temperature" in ylabel:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=2, alpha=0.6)
        ax.text(
            0,
            threshold + 1,
            "Thermal\nlimit",
            transform=ax.get_yaxis_transform(),
            color="red",
            ha="left",
            va="bottom",
            alpha=0.75,
            fontweight="bold",
        )
    if threshold is not None and "Capacity" in ylabel:
        ax.axhline(threshold, color="blue", linestyle="--", linewidth=2, alpha=0.6)
        ax.text(
            0,
            threshold - 1,
            "Security\nmargin",
            transform=ax.get_yaxis_transform(),
            color="blue",
            ha="left",
            va="top",
            alpha=0.75,
            fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, ha="center", fontweight="bold")
    ax.tick_params(top=False, bottom=False, left=True, right=False)
    ax.grid(False)
    save_pdf(fig, output)


def line_temperature_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "subtle_danger",
        ["#4cb040", "#6cc050", "#b8c840", "#d8b830", "#FF7F00", "#FF0000", "#8B00FF"],
        N=256,
    )


def plot_map(
    snapshot_id: str,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    branch_values: pd.DataFrame,
    bus_values: pd.DataFrame,
    weather_values: pd.DataFrame,
    display: dict,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    x_count = int(weather_values["x_index"].max()) + 1
    y_count = int(weather_values["y_index"].max()) + 1
    x_grid = weather_values.sort_values("x_index").drop_duplicates("x_index")["longitude_plot"].to_numpy()
    y_grid = weather_values.sort_values("y_index").drop_duplicates("y_index")["latitude_plot"].to_numpy()
    temperature = (
        weather_values.sort_values(["y_index", "x_index"])["air_temperature_c"]
        .to_numpy()
        .reshape(y_count, x_count)
    )
    pcolor = ax.pcolormesh(
        x_grid,
        y_grid,
        temperature,
        shading="auto",
        cmap="coolwarm",
        norm=plt.Normalize(
            vmin=float(display["air_temperature_min_c"]),
            vmax=float(display["air_temperature_max_c"]),
        ),
        alpha=float(display["air_temperature_alpha"]),
        zorder=0,
        transform=ccrs.PlateCarree(),
    )
    pcolor.set_rasterized(True)
    ocean = ax.add_feature(cfeature.OCEAN, zorder=1)
    ocean.set_rasterized(True)
    borders = ax.add_feature(cfeature.BORDERS, zorder=1)
    borders.set_rasterized(True)

    merged_lines = lines.merge(branch_values, on="branch_index", validate="one_to_one")
    segments = merged_lines[
        ["longitude_0", "latitude_0", "longitude_1", "latitude_1"]
    ].to_numpy(dtype=float).reshape(-1, 2, 2)
    collection = LineCollection(
        segments,
        cmap=line_temperature_cmap(),
        norm=plt.Normalize(
            vmin=float(display["line_temperature_min_c"]),
            vmax=float(display["line_temperature_max_c"]),
        ),
        linewidths=float(display["line_width"]),
        alpha=float(display["line_temperature_alpha"]),
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    collection.set_array(merged_lines["line_temperature_c"].to_numpy(dtype=float))
    ax.add_collection(collection)

    ax.scatter(
        buses["longitude"],
        buses["latitude"],
        marker=".",
        c="gray",
        alpha=0.25,
        edgecolors="none",
        s=float(display["bus_marker_size"]),
        zorder=10,
        transform=ccrs.PlateCarree(),
    )
    marked = buses.merge(bus_values, on="bus_index", validate="one_to_one")
    marked = marked[marked["display_marker"].astype(bool)]
    if not marked.empty:
        ax.scatter(
            marked["longitude"],
            marked["latitude"],
            marker="^",
            c="#FFD700",
            edgecolors="black",
            linewidths=2,
            s=float(display["load_shedding_marker_size"]),
            zorder=11,
            alpha=1,
            transform=ccrs.PlateCarree(),
        )
    ax.set_extent(
        [
            buses["longitude"].min() - float(display["coordinate_offset_degrees"]),
            buses["longitude"].max() + float(display["coordinate_offset_degrees"]),
            buses["latitude"].min() - float(display["coordinate_offset_degrees"]),
            buses["latitude"].max() + float(display["coordinate_offset_degrees"]),
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.set_axis_off()
    fig.tight_layout()
    save_pdf(fig, output, dpi=150, pad_inches=0.01)


def discrete_colorbar(
    output: Path,
    *,
    cmap,
    vmin: float,
    vmax: float,
    levels: int,
    alpha: float,
    label: str,
) -> None:
    boundaries = np.linspace(vmin, vmax, levels + 1)
    discrete = mcolors.ListedColormap(cmap(np.linspace(0, 1, levels)))
    norm = mcolors.BoundaryNorm(boundaries, discrete.N)
    fig, ax = plt.subplots(figsize=(8, 0.20))
    scalar = plt.cm.ScalarMappable(cmap=discrete, norm=norm)
    scalar.set_array([])
    colorbar = plt.colorbar(
        scalar,
        cax=ax,
        orientation="horizontal",
        alpha=alpha,
        extend="both",
        extendfrac=0.05,
    )
    colorbar.set_label(label, fontsize=16, fontweight="bold")
    colorbar.set_ticks(boundaries)
    colorbar.set_ticklabels([f"{tick:.0f}" for tick in boundaries])
    colorbar.outline.set_linewidth(1)
    colorbar.outline.set_edgecolor("black")
    save_pdf(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()

    package = args.package_dir.resolve()
    data = package / "data"
    artwork = package / "artwork"
    artwork.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "plot_metadata.json").read_text())
    scenarios = pd.read_csv(data / "scenario_observations.csv")
    branch_path = data / "scenario_branch_observations.csv"
    if not branch_path.exists():
        branch_path = data / "scenario_branch_observations.csv.gz"
    branches = pd.read_csv(branch_path)
    buses = pd.read_csv(data / "represented_network_buses.csv")
    lines = pd.read_csv(data / "represented_network_lines.csv")
    snapshots = pd.read_csv(data / "map_snapshots.csv")
    map_branches = pd.read_csv(data / "map_branch_observations.csv")
    map_buses = pd.read_csv(data / "map_bus_observations.csv")
    weather_path = data / "map_weather_fields.csv"
    if not weather_path.exists():
        weather_path = data / "map_weather_fields.csv.gz"
    map_weather = pd.read_csv(weather_path)
    configure_matplotlib()

    hours = metadata["future_hours"]
    hour_labels = metadata["hour_labels"]
    years = metadata["future_years"]
    hour_branch_groups = [
        branches[branches["scenario_id"].isin(scenarios.loc[scenarios.future_hour == hour, "scenario_id"])]
        for hour in hours
    ]
    year_branch_groups = [
        branches[branches["scenario_id"].isin(scenarios.loc[scenarios.future_year == year, "scenario_id"])]
        for year in years
    ]
    hour_scenario_groups = [scenarios[scenarios.future_hour == hour] for hour in hours]
    year_scenario_groups = [scenarios[scenarios.future_year == year] for year in years]

    box_violin_plot(
        [group["available_capacity_percent_of_nominal"].to_numpy() for group in hour_branch_groups],
        hour_labels,
        list(reversed(metadata["sequential_blue"]))[: len(hours)],
        "Available Capacity (%)",
        artwork / "hour_model_capacity_drop_box_violin.pdf",
        threshold=float(metadata["capacity_security_margin_percent"]),
        exceed_direction="below",
    )
    box_violin_plot(
        [group["available_capacity_percent_of_nominal"].to_numpy() for group in year_branch_groups],
        [str(year) for year in years],
        metadata["sequential_blue"],
        "Available Capacity (%)",
        artwork / "temperal_model_capacity_drop_box_violin.pdf",
        threshold=float(metadata["capacity_security_margin_percent"]),
        exceed_direction="below",
    )
    box_violin_plot(
        [group["line_temperature_c"].to_numpy() for group in hour_branch_groups],
        hour_labels,
        list(reversed(metadata["sequential_red"]))[: len(hours)],
        "Line Temperature (°C)",
        artwork / "hour_model_line_temp_box_violin.pdf",
        threshold=float(metadata["thermal_limit_c"]),
        exceed_direction="above",
    )
    box_violin_plot(
        [group["line_temperature_c"].to_numpy() for group in year_branch_groups],
        [str(year) for year in years],
        metadata["sequential_red"],
        "Line Temperature (°C)",
        artwork / "temperal_model_line_temp_box_violin.pdf",
        threshold=float(metadata["thermal_limit_c"]),
        exceed_direction="above",
    )
    box_violin_plot(
        [group["load_shedding_percent"].to_numpy() for group in hour_scenario_groups],
        hour_labels,
        list(reversed(metadata["sequential_orange"]))[: len(hours)],
        "Load Shedding (%)",
        artwork / "hour_model_load_shedding_box_violin.pdf",
    )
    box_violin_plot(
        [group["load_shedding_percent"].to_numpy() for group in year_scenario_groups],
        [str(year) for year in years],
        metadata["sequential_orange"],
        "Load Shedding (%)",
        artwork / "temporal_load_shedding_box_violin.pdf",
    )

    for snapshot, filename in zip(snapshots.itertuples(index=False), metadata["map_artwork"]):
        plot_map(
            snapshot.snapshot_id,
            buses,
            lines,
            map_branches[map_branches.snapshot_id == snapshot.snapshot_id],
            map_buses[map_buses.snapshot_id == snapshot.snapshot_id],
            map_weather[map_weather.snapshot_id == snapshot.snapshot_id],
            metadata["map_display"],
            artwork / filename,
        )
    for colorbar in metadata["colorbars"]:
        cmap = plt.cm.coolwarm if colorbar["kind"] == "air_temperature" else line_temperature_cmap()
        discrete_colorbar(
            artwork / colorbar["filename"],
            cmap=cmap,
            vmin=float(colorbar["vmin"]),
            vmax=float(colorbar["vmax"]),
            levels=int(colorbar["levels"]),
            alpha=float(colorbar["alpha"]),
            label=colorbar["label"],
        )
    print(json.dumps({"package": str(package), "artwork_files": len(list(artwork.glob('*.pdf'))) }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

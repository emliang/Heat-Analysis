#!/usr/bin/env python3
"""Reproduce one combined cross-border Supplementary figure."""

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
    "Creator": "publication_pipeline/code/plotting/plot_cross_border_supplement.py",
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
    plt.rcParams.update({
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
    })


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
    width, tip = 0.06, 0.04
    height = y_hi - y_lo
    middle = (y_hi + y_lo) / 2
    vertices = [
        (x, y_hi), (x + width, y_hi),
        (x + width, middle + height * 0.15),
        (x + width, middle + height * 0.05),
        (x + width + tip, middle),
        (x + width, middle - height * 0.05),
        (x + width, middle - height * 0.15),
        (x + width, y_lo), (x, y_lo),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(mpatches.PathPatch(
        MplPath(vertices, codes), facecolor="none", edgecolor="black",
        linewidth=1.0, capstyle="round", joinstyle="miter", clip_on=False,
    ))
    return x + width + tip


def box_violin_plot(observations, labels, colors, ylabel, output, *, threshold=None, exceed_direction="above") -> None:
    positions = np.arange(len(labels))
    category_width = 1.0
    sub_width = category_width / 4
    fig, ax = plt.subplots(figsize=(min(len(labels) * 2.5, 10), 4))
    for index, raw_values in enumerate(observations):
        values = np.asarray(raw_values, dtype=float)
        box_centre = positions[index] - category_width / 10
        violin_centre = positions[index] + category_width / 10
        box = ax.boxplot(
            [values], positions=[box_centre], widths=sub_width,
            patch_artist=True, zorder=2, showfliers=False,
            medianprops={"color": "none", "linewidth": 1.5},
        )
        box["boxes"][0].set(facecolor=colors[index], alpha=0.9, edgecolor="white")
        ax.scatter(box_centre, values.mean(), marker="o", color="darkred", s=40, zorder=3)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = values[(values < lower_fence) | (values > upper_fence)]
        count = min(20, len(outliers))
        if count:
            generator = np.random.default_rng(2500 + index)
            selected = generator.choice(len(outliers), count, replace=False)
            jitter = (generator.random(count) - 0.5) * sub_width * 0.3
            ax.scatter(
                np.full(count, box_centre) + jitter, outliers[selected],
                s=18, c="white", edgecolors=colors[index], zorder=3, alpha=0.9,
            )
        value_grid = np.linspace(values.min(), values.max(), 300)
        if np.ptp(values) > 1e-12:
            density = stats.gaussian_kde(values)(value_grid)
            density = density / density.max() * sub_width
        else:
            density = np.full_like(value_grid, sub_width * 0.05)
        ax.fill_betweenx(
            value_grid, violin_centre, violin_centre + density,
            color=colors[index], alpha=0.9, zorder=1,
        )
        upper_whisker = min(upper_fence, values.max())
        annotation_y = upper_whisker + (ax.get_ylim()[1] - upper_whisker) * 0.1
        ax.annotate(
            f"ave:\n{values.mean():.2f}", xy=(box_centre, values.mean()),
            xytext=(box_centre - category_width * 0.4, annotation_y * 0.8),
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7, "linestyle": "--"},
            fontsize=16, ha="left", va="bottom", fontweight="bold",
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
                tip_x = draw_curly_brace(ax, violin_centre + density[mask].max() + 0.02, y_lo, y_hi)
                ax.annotate(
                    f"{percentage:.1f}%", xy=(tip_x + 0.05, (y_lo + y_hi) / 2),
                    fontsize=14, fontweight="bold", va="center", ha="left",
                    color="black", annotation_clip=False,
                )
    if threshold is not None and "Temperature" in ylabel:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=2, alpha=0.6)
        ax.text(0, threshold + 1, "Thermal\nlimit", transform=ax.get_yaxis_transform(),
                color="red", ha="left", va="bottom", alpha=0.75, fontweight="bold")
    if threshold is not None and "Capacity" in ylabel:
        ax.axhline(threshold, color="blue", linestyle="--", linewidth=2, alpha=0.6)
        ax.text(0, threshold - 1, "Security\nmargin", transform=ax.get_yaxis_transform(),
                color="blue", ha="left", va="top", alpha=0.75, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="center", fontweight="bold")
    ax.tick_params(top=False, bottom=False, left=True, right=False)
    ax.grid(False)
    save_pdf(fig, output)


def line_temperature_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "subtle_danger",
        ["#4cb040", "#6cc050", "#b8c840", "#d8b830", "#FF7F00", "#FF0000", "#8B00FF"],
        N=256,
    )


def plot_map(buses, lines, branch_values, bus_values, weather_values, display, output) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    x_count = int(weather_values["x_index"].max()) + 1
    y_count = int(weather_values["y_index"].max()) + 1
    x_grid = weather_values.sort_values("x_index").drop_duplicates("x_index")["longitude_plot"].to_numpy()
    y_grid = weather_values.sort_values("y_index").drop_duplicates("y_index")["latitude_plot"].to_numpy()
    temperature = weather_values.sort_values(["y_index", "x_index"])["air_temperature_c"].to_numpy().reshape(y_count, x_count)
    pcolor = ax.pcolormesh(
        x_grid, y_grid, temperature, shading="auto", cmap="coolwarm",
        norm=plt.Normalize(vmin=display["air_temperature_min_c"], vmax=display["air_temperature_max_c"]),
        alpha=display["air_temperature_alpha"], zorder=0, transform=ccrs.PlateCarree(),
    )
    pcolor.set_rasterized(True)
    ax.add_feature(cfeature.OCEAN, zorder=1).set_rasterized(True)
    ax.add_feature(cfeature.BORDERS, zorder=1).set_rasterized(True)
    merged_lines = lines.merge(branch_values, on=["configuration_id", "branch_index"], validate="one_to_one")
    segments = merged_lines[["longitude_0", "latitude_0", "longitude_1", "latitude_1"]].to_numpy(dtype=float).reshape(-1, 2, 2)
    collection = LineCollection(
        segments, cmap=line_temperature_cmap(),
        norm=plt.Normalize(vmin=display["line_temperature_min_c"], vmax=display["line_temperature_max_c"]),
        linewidths=display["line_width"], alpha=display["line_temperature_alpha"],
        transform=ccrs.PlateCarree(), zorder=3,
    )
    collection.set_array(merged_lines["line_temperature_c"].to_numpy(dtype=float))
    ax.add_collection(collection)
    ax.scatter(
        buses["longitude"], buses["latitude"], marker=".", c="gray", alpha=0.25,
        edgecolors="none", s=display["bus_marker_size"], zorder=10,
        transform=ccrs.PlateCarree(),
    )
    marked = buses.merge(bus_values, on=["configuration_id", "bus_index"], validate="one_to_one")
    marked = marked[marked["display_marker"].astype(bool)]
    if not marked.empty:
        ax.scatter(
            marked["longitude"], marked["latitude"], marker="^", c="#FFD700",
            edgecolors="black", linewidths=2, s=display["load_shedding_marker_size"],
            zorder=11, alpha=1, transform=ccrs.PlateCarree(),
        )
    offset = display["coordinate_offset_degrees"]
    ax.set_extent([
        buses["longitude"].min() - offset, buses["longitude"].max() + offset,
        buses["latitude"].min() - offset, buses["latitude"].max() + offset,
    ], crs=ccrs.PlateCarree())
    ax.set_axis_off()
    fig.tight_layout()
    save_pdf(fig, output, dpi=150, pad_inches=0.01)


def discrete_colorbar(output, *, cmap, vmin, vmax, levels, alpha, label) -> None:
    boundaries = np.linspace(vmin, vmax, levels + 1)
    discrete = mcolors.ListedColormap(cmap(np.linspace(0, 1, levels)))
    norm = mcolors.BoundaryNorm(boundaries, discrete.N)
    fig, ax = plt.subplots(figsize=(8, 0.20))
    scalar = plt.cm.ScalarMappable(cmap=discrete, norm=norm)
    scalar.set_array([])
    colorbar = plt.colorbar(scalar, cax=ax, orientation="horizontal", alpha=alpha, extend="both", extendfrac=0.05)
    colorbar.set_label(label, fontsize=16, fontweight="bold")
    colorbar.set_ticks(boundaries)
    colorbar.set_ticklabels([f"{tick:.0f}" for tick in boundaries])
    colorbar.outline.set_linewidth(1)
    colorbar.outline.set_edgecolor("black")
    save_pdf(fig, output)


def plot_statistics(data: Path, artwork: Path) -> int:
    artwork.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "plot_metadata.json").read_text())
    scenarios = pd.read_csv(data / "scenario_observations.csv")
    legacy_branch_path = data / "scenario_branch_observations.csv"
    if not legacy_branch_path.exists():
        legacy_branch_path = data / "scenario_branch_observations.csv.gz"
    if legacy_branch_path.exists():
        temperatures = pd.read_csv(legacy_branch_path)
    else:
        temperatures = pd.read_csv(data / "line_temperature_observations.csv")
    for load_growth in metadata["load_growths"]:
        scenario_subset = scenarios[np.isclose(scenarios["load_growth"], load_growth)]
        temperature_subset = temperatures[
            np.isclose(temperatures["load_growth"], load_growth)
        ]
        ids = metadata["configuration_ids"]
        labels = metadata["configuration_labels"]
        colors = metadata["colors"]
        suffix = f"{load_growth}_{metadata['storage_state']}.pdf"
        box_violin_plot(
            [scenario_subset[scenario_subset.configuration_id == cid]["load_shedding_percent"].to_numpy() for cid in ids],
            labels, colors, "Load Shedding (%)",
            artwork / f"model_cross_border_load_shedding_box_violin.pdf_{suffix}",
        )
        box_violin_plot(
            [temperature_subset[temperature_subset.configuration_id == cid]["line_temperature_c"].to_numpy() for cid in ids],
            labels, colors, "Line Temperature (°C)",
            artwork / f"model_cross_border_line_temp_box_violin.pdf_{suffix}",
            threshold=metadata["thermal_limit_c"], exceed_direction="above",
        )
    return len(list(artwork.glob("*.pdf")))


def plot_maps(data: Path, artwork: Path) -> int:
    artwork.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "plot_metadata.json").read_text())
    snapshots = pd.read_csv(data / "map_snapshots.csv")
    buses = pd.read_csv(data / "represented_network_buses.csv")
    lines = pd.read_csv(data / "represented_network_lines.csv")
    branches = pd.read_csv(data / "map_branch_observations.csv")
    bus_values = pd.read_csv(data / "map_bus_observations.csv")
    weather_path = data / "map_weather_fields.csv"
    if not weather_path.exists():
        weather_path = data / "map_weather_fields.csv.gz"
    weather = pd.read_csv(weather_path)
    for snapshot in snapshots.itertuples(index=False):
        cid = snapshot.configuration_id
        plot_map(
            buses[buses.configuration_id == cid],
            lines[lines.configuration_id == cid],
            branches[branches.snapshot_id == snapshot.snapshot_id],
            bus_values[bus_values.snapshot_id == snapshot.snapshot_id],
            weather[weather.snapshot_id == snapshot.snapshot_id],
            metadata["map_display"], artwork / snapshot.artwork_filename,
        )
    for colorbar in metadata["colorbars"]:
        cmap = plt.cm.coolwarm if colorbar["kind"] == "air_temperature" else line_temperature_cmap()
        discrete_colorbar(artwork / colorbar["filename"], cmap=cmap, **{key: value for key, value in colorbar.items() if key not in {"filename", "kind"}})
    return len(list(artwork.glob("*.pdf")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--statistics-package", type=Path)
    parser.add_argument("--maps-package", type=Path)
    args = parser.parse_args()
    configure_matplotlib()
    if args.source_dir is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required with --source-dir")
        source = args.source_dir.resolve()
        output = args.output_dir.resolve()
        statistics_data = source / "statistics"
        maps_data = source / "maps"
        statistics_artwork = output / "statistics"
        maps_artwork = output / "maps"
    else:
        if args.statistics_package is None or args.maps_package is None:
            parser.error(
                "provide --source-dir/--output-dir or the two legacy package arguments"
            )
        statistics_package = args.statistics_package.resolve()
        maps_package = args.maps_package.resolve()
        statistics_data = statistics_package / "data"
        maps_data = maps_package / "data"
        statistics_artwork = statistics_package / "artwork"
        maps_artwork = maps_package / "artwork"
    counts = {
        "statistics_artwork": plot_statistics(statistics_data, statistics_artwork),
        "map_artwork": plot_maps(maps_data, maps_artwork),
    }
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

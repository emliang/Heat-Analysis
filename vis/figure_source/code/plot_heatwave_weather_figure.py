#!/usr/bin/env python3
"""Reproduce one Supplementary heatwave figure from its compact package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib import font_manager  # noqa: E402


CMAPS = {
    "temperature": plt.cm.coolwarm,
    "influx": plt.cm.plasma,
    "wnd10m": plt.cm.viridis,
}


def metadata() -> dict:
    return {
        "Creator": "publication_pipeline/code/plotting/plot_heatwave_weather_figure.py",
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "savefig.transparent": False,
        }
    )


def national_line_styles(rules: dict) -> list[tuple[str, object, str]]:
    cmap = CMAPS[rules["variable"]]
    palette = [cmap(index / 7) for index in range(8)]
    if "wnd" in rules["variable"]:
        palette = palette[::-1]
    historical_color = palette[1]
    future_color = palette[-2]
    return [
        ("historical_reference", historical_color, "-"),
        ("historical_heatwave", historical_color, "-."),
        ("future_reference", future_color, "-"),
        ("future_heatwave", future_color, "-."),
    ]


def plot_national_table(
    table: pd.DataFrame,
    output: Path,
    rules: dict,
    *,
    show_legend: bool = True,
    figsize: tuple[float, float] = (12.5, 3.5),
) -> None:
    values = {
        scenario: group.sort_values("hour").value.to_numpy()
        for scenario, group in table.groupby("scenario")
    }
    styles = national_line_styles(rules)
    historical_color = styles[0][1]
    future_color = styles[-1][1]
    fig = plt.figure(figsize=figsize)
    hours = np.arange(24)
    plt.fill_between(hours, values["historical_reference"], values["historical_heatwave"], alpha=0.1, color=historical_color)
    plt.plot(values["historical_reference"], color=historical_color, alpha=0.7, linewidth=2)
    plt.plot(values["historical_heatwave"], color=historical_color, linestyle="-.", linewidth=2)
    plt.fill_between(hours, values["future_reference"], values["future_heatwave"], alpha=0.1, color=future_color)
    plt.plot(values["future_reference"], color=future_color, alpha=0.7, linewidth=2)
    plt.plot(values["future_heatwave"], color=future_color, linestyle="-.", linewidth=2)
    if show_legend:
        for scenario, color, style in styles:
            plt.plot([], label=rules["scenario_labels"][scenario], color=color, linestyle=style, alpha=0.7 if "reference" in scenario else 1.0, linewidth=2.5)
    plt.xlim(0, 23)
    plt.xticks(range(24), fontsize=14)
    plt.xlabel("Hour", fontsize=16, fontweight="bold")
    plt.ylabel(rules["units"], fontsize=16, fontweight="bold")
    if show_legend:
        plt.legend(ncol=4, loc=2, frameon=False, columnspacing=1.0, bbox_to_anchor=(0.0, 1.2), prop={"size": 15, "weight": "bold"})
    plt.grid(linewidth=0.25, alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01, metadata=metadata())
    plt.close(fig)


def plot_national(source: Path, output: Path, rules: dict) -> None:
    plot_national_table(
        pd.read_csv(source / "national_hourly_profiles.csv"),
        output,
        rules,
    )


def spatial_grid(table: pd.DataFrame, scenario: str):
    selected = table[table.scenario == scenario]
    x_coordinates = selected[["x_index", "x"]].drop_duplicates().sort_values("x_index")
    y_coordinates = selected[["y_index", "y"]].drop_duplicates().sort_values("y_index")
    xs = x_coordinates.x.to_numpy()
    ys = y_coordinates.y.to_numpy()
    pivot = selected.pivot(index="y_index", columns="x_index", values="value").reindex(
        index=y_coordinates.y_index, columns=x_coordinates.x_index
    )
    return xs, ys, pivot.to_numpy()


def plot_spatial(source: Path, output_dir: Path, rules: dict) -> None:
    spatial_path = source / "spatial_snapshot_fields.csv"
    if not spatial_path.exists():
        spatial_path = source / "spatial_snapshot_fields.csv.gz"
    table = pd.read_csv(spatial_path)
    cmap = CMAPS[rules["variable"]]
    normalizer = plt.Normalize(vmin=rules["vmin"], vmax=rules["vmax"])
    for scenario in (
        "historical_reference",
        "historical_heatwave",
        "future_reference",
        "future_heatwave",
    ):
        xs, ys, values = spatial_grid(table, scenario)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plt.pcolormesh(xs - 0.125, ys - 0.125, values, shading="auto", cmap=cmap, norm=normalizer, alpha=0.7, zorder=0)
        ax.add_feature(cfeature.OCEAN, zorder=1)
        ax.add_feature(cfeature.BORDERS, zorder=1)
        label = rules["scenario_labels"][scenario]
        plt.title(label, fontsize=22, fontweight="bold")
        filename = f"{rules['future_year']}_{rules['historical_year']}_{rules['month']}_{rules['variable']}_{label}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.01, metadata={"Software": "Matplotlib"})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 0.2))
    ax.set_xticks([])
    ax.set_yticks([])
    boundaries = np.linspace(rules["vmin"], rules["vmax"], int(rules["color_levels"]) + 1)
    discrete_colors = cmap(np.linspace(0, 1, int(rules["color_levels"])))
    discrete_cmap = colors.ListedColormap(discrete_colors)
    normalizer = colors.BoundaryNorm(boundaries, discrete_cmap.N)
    scalar = cm.ScalarMappable(cmap=discrete_cmap, norm=normalizer)
    scalar.set_array([])
    extend = "max" if rules["vmin"] == 0 else "both"
    colorbar = fig.colorbar(scalar, cax=ax, orientation="horizontal", alpha=0.7, extend=extend, extendfrac=0.05)
    colorbar.set_label(rules["units"], fontsize=16, fontweight="bold")
    colorbar.set_ticks(boundaries)
    colorbar.set_ticklabels([f"{value:.0f}" for value in boundaries])
    colorbar.ax.set_alpha(0.7)
    colorbar.outline.set_linewidth(1)
    colorbar.outline.set_edgecolor("black")
    fig.savefig(output_dir / f"heatwave_{rules['variable']}_colorbar.pdf", format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01, metadata=metadata())
    plt.close(fig)


def plot_regions(source: Path, output_dir: Path, rules: dict) -> None:
    table = pd.read_csv(source / "sampled_regional_hourly_profiles.csv")
    cmap = CMAPS[rules["variable"]]
    palette = [cmap(index / 7) for index in range(8)]
    if "wnd" in rules["variable"]:
        palette = palette[::-1]
    future_color = palette[-2]
    global_min = float(table.value.min())
    global_max = float(table.value.max())
    for region_number, region in table.groupby("region_number"):
        fig = plt.figure(figsize=(5.5, 3.5))
        reference = region[region.curve == "future_reference"].sort_values("hour")
        plt.plot(reference.value.to_numpy(), color=future_color, alpha=0.7, linewidth=2)
        for _, sample in region[region.curve == "future_heatwave"].groupby("sample_rank"):
            plt.plot(sample.sort_values("hour").value.to_numpy(), color=future_color, linestyle="--", linewidth=1, alpha=0.8)
        plt.plot([], label="Reference", color=future_color, linewidth=2.5, alpha=0.7)
        plt.plot([], label="Heatwaves", color=future_color, linestyle="--", linewidth=2.5, alpha=0.8)
        plt.xlim(0, 23)
        plt.ylim(global_min * 0.9, global_max * 1.1)
        plt.xticks(range(0, 24, 4), fontsize=14)
        plt.xlabel("Hour", fontsize=18, fontweight="bold")
        plt.ylabel(rules["units"], fontsize=18, fontweight="bold")
        plt.grid(linewidth=0.25, alpha=0.25)
        plt.legend(ncol=2, loc=2, frameon=False, columnspacing=1, handlelength=1.5, bbox_to_anchor=(0, 1.25), prop={"size": 18, "weight": "bold"})
        plt.text(0.05, 0.95, f"Region {int(region_number)}", transform=plt.gca().transAxes, fontsize=18, verticalalignment="top", horizontalalignment="left")
        plt.tight_layout()
        filename = f"{int(region_number) - 1}_temperal_heatwave_{rules['variable']}_{rules['country']}_{rules['future_year']}_{rules['historical_year']}.pdf"
        fig.savefig(output_dir / filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01, metadata=metadata())
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_matplotlib()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rules = json.loads((source / "plot_metadata.json").read_text())
    national_name = f"heatwave_{rules['variable']}_{rules['country']}_{rules['future_year']}_{rules['historical_year']}.pdf"
    plot_national(source, output / national_name, rules)
    if rules["full_delivery"]:
        plot_spatial(source, output, rules)
        plot_regions(source, output, rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

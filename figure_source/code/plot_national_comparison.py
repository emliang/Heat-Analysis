#!/usr/bin/env python3
"""Reproduce Supplementary Fig. 15 panels from compact plotted data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib import patches as mpatches  # noqa: E402
from scipy import stats  # noqa: E402


PDF_METADATA = {
    "Creator": "publication_pipeline/code/plotting/plot_national_comparison.py",
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


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        dpi=500,
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def finish_axes(ax, labels: list[str], ylabel: str, rotation: float) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="center", fontweight="bold")
    ax.tick_params(axis="x", top=False, bottom=False)
    ax.tick_params(axis="y", left=True, right=False)


def single_bar_plot(
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_axisbelow(False)
    ax.grid(axis="y", linewidth=0, alpha=0)
    ax.grid(axis="x", linewidth=0, alpha=0)
    positions = np.arange(len(labels))
    means = np.array([np.mean(values) for values in observations], dtype=float)
    standard_deviations = np.array(
        [np.std(values, ddof=0) for values in observations], dtype=float
    )
    bars = ax.bar(positions, means, alpha=0.9, color=colors)
    ax.errorbar(
        positions,
        means,
        yerr=standard_deviations,
        fmt="none",
        elinewidth=1.5,
        capsize=14,
        capthick=2,
        ecolor="dimgray",
    )
    ax.scatter(positions, means, marker="o", color="darkred", s=40, zorder=3)
    for index, bar in enumerate(bars):
        if means[index] > 1e-2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                means[index] + standard_deviations[index] * 1.01,
                f"ave:\n{means[index]:.3g}",
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=4,
            )
    upper = means.max() + standard_deviations.max() * 1.1
    ax.set_ylim(0, upper if upper > 0 else 1)
    finish_axes(ax, labels, ylabel, 30)
    save_pdf(fig, output)


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
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    patch = mpatches.PathPatch(
        MplPath(vertices, codes),
        facecolor="none",
        edgecolor="black",
        linewidth=1,
        capstyle="round",
        joinstyle="miter",
        clip_on=False,
    )
    ax.add_patch(patch)
    return x + width + tip


def box_violin_plot(
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    output: Path,
    *,
    threshold: float,
    exceed_direction: str,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.grid(False)
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
            generator = np.random.default_rng(1500 + index)
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
        density = stats.gaussian_kde(values)(value_grid)
        density = density / density.max() * sub_width
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

    if exceed_direction == "above":
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
    else:
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
    finish_axes(ax, labels, ylabel, 0)
    save_pdf(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()

    package = args.package_dir.resolve()
    data_dir = package / "data"
    artwork_dir = package / "artwork"
    metadata = json.loads((data_dir / "plot_metadata.json").read_text())
    scenarios = pd.read_csv(data_dir / "scenario_observations.csv")
    branch_path = data_dir / "branch_observations.csv"
    if not branch_path.exists():
        branch_path = data_dir / "branch_observations.csv.gz"
    branches = pd.read_csv(branch_path)
    configure_matplotlib()

    order = metadata["country_order"]
    labels = [metadata["country_names"][country] for country in order]
    colors = [metadata["country_colors"][country] for country in order]

    scenario_metrics = (
        ("air_temperature_c", "Air Temperature (°C)", "multi_country_analysis_air_temp.pdf"),
        ("hourly_load_gw", "Hourly Load (GW)", "multi_country_analysis_load.pdf"),
        ("load_shedding_percent", "Load Shedding (%)", "multi_country_analysis_load_shedding.pdf"),
        ("runtime_s", "Running Time (sec.)", "multi_country_analysis_run_time.pdf"),
    )
    for column, ylabel, filename in scenario_metrics:
        values = [
            scenarios.loc[scenarios.country_code == country, column].to_numpy(dtype=float)
            for country in order
        ]
        single_bar_plot(values, labels, colors, ylabel, artwork_dir / filename)

    line_temperatures = [
        branches.loc[branches.country_code == country, "line_temperature_c"].to_numpy(dtype=float)
        for country in order
    ]
    box_violin_plot(
        line_temperatures,
        labels,
        colors,
        "Line Temperature (°C)",
        artwork_dir / "multi_country_analysis_line_temp.pdf",
        threshold=float(metadata["thermal_limit_c"]),
        exceed_direction="above",
    )
    capacity = [
        branches.loc[
            branches.country_code == country,
            "available_capacity_percent_of_nominal",
        ].to_numpy(dtype=float)
        for country in order
    ]
    box_violin_plot(
        capacity,
        labels,
        colors,
        "Available Capacity (%)",
        artwork_dir / "multi_country_analysis_capa_drop.pdf",
        threshold=float(metadata["capacity_security_margin_percent"]),
        exceed_direction="below",
    )
    print(json.dumps({"package": str(package), "artwork_files": len(list(artwork_dir.glob('*.pdf')))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

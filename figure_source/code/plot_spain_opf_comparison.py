#!/usr/bin/env python3
"""Reproduce single-country OPF-comparison panels from compact data."""

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
    "Creator": "publication_pipeline/code/plotting/plot_spain_opf_comparison.py",
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


def save_pdf(fig, path: Path, dpi: int = 500) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def finish_axes(ax, labels: list[str], ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(False)
    ax.grid(axis="y", linewidth=0, alpha=0)
    ax.grid(axis="x", linewidth=0, alpha=0)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, ha="center", fontweight="bold")
    ax.tick_params(top=False, bottom=False, left=True, right=False)
    for label in ax.get_xticklabels():
        if label.get_text() == "Iter-OPF":
            label.set_color("royalblue")
            break


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
    patch = mpatches.PathPatch(
        MplPath(vertices, codes),
        facecolor="none",
        edgecolor="black",
        linewidth=1.0,
        capstyle="round",
        joinstyle="miter",
        clip_on=False,
    )
    ax.add_patch(patch)
    return x + width + tip


def add_reference_line(ax, metric: str, threshold: float) -> None:
    if metric == "temperature":
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
    elif metric == "capacity":
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


def box_violin_plot(
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    output: Path,
    *,
    metric: str | None = None,
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
            generator = np.random.default_rng(1600 + index)
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

    if threshold is not None and metric is not None:
        add_reference_line(ax, metric, threshold)
    finish_axes(ax, labels, ylabel)
    save_pdf(fig, output)


def runtime_plot(
    scenarios: pd.DataFrame,
    methods: list[str],
    labels: list[str],
    colors: list[str],
    output: Path,
) -> None:
    observations = [
        scenarios[f"{method}__runtime_s"].to_numpy(dtype=float) for method in methods
    ]
    means = np.asarray([values.mean() for values in observations])
    deviations = np.asarray([values.std(ddof=0) for values in observations])
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = np.arange(len(methods))
    bars = ax.bar(positions, means, alpha=0.9, color=colors)
    ax.errorbar(
        positions,
        means,
        yerr=deviations,
        fmt="none",
        elinewidth=1.5,
        capsize=14,
        capthick=2,
        ecolor="dimgray",
    )
    ax.scatter(positions, means, marker="o", color="darkred", s=40, zorder=3)
    for index, bar in enumerate(bars):
        ax.annotate(
            f"ave:\n{means[index]:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, means[index]),
            xytext=(bar.get_x(), means[index] + deviations[index] * 0.5),
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7, "linestyle": "--"},
            fontsize=16,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylim(0, means.max() + deviations.max() * 1.1)
    finish_axes(ax, labels, "Running Time (sec.)")
    save_pdf(fig, output, dpi=300)


def scatter_plot(scenarios: pd.DataFrame, method: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 4))
    scatter = ax.scatter(
        scenarios.load_gw,
        scenarios[f"{method}__load_shedding_percent"],
        c=scenarios.air_temperature_c,
        cmap=plt.cm.coolwarm,
        alpha=0.9,
        s=100,
        edgecolors="white",
        linewidths=0.5,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Load demand (GW)", fontweight="bold")
    ax.set_ylabel("Load shedding (%)", fontweight="bold")
    colorbar.set_label("Air temperature (°C)", fontweight="bold")
    ax.tick_params(top=False, right=False, left=True, bottom=True)
    save_pdf(fig, output, dpi=300)


def write_panel_a_table(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{tabular}{c|cccc}",
        r"\toprule",
        r"\textbf{Year} & \textbf{Wind} (m/s) & \textbf{Solar} (W/m$^2$) & \textbf{Temp.} ($^\circ$C) & \textbf{Load} (GW) \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"\\textbf{{{int(row.future_year)}}} & "
            f"{row.wind_speed_mean_m_per_s:.2f} ($\\pm${row.wind_speed_sd_m_per_s_ddof_0:.2f}) & "
            f"{row.solar_irradiance_mean_w_per_m2:.2f} ($\\pm${row.solar_irradiance_sd_w_per_m2_ddof_0:.2f}) & "
            f"{row.air_temperature_mean_c:.2f} ($\\pm${row.air_temperature_sd_c_ddof_0:.2f}) & "
            f"{row.load_mean_gw:.2f} ($\\pm${row.load_sd_gw_ddof_0:.2f}) \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    data = package / "data"
    artwork = package / "artwork"
    artwork.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    metadata = json.loads((data / "plot_metadata.json").read_text())
    scenarios = pd.read_csv(data / "scenario_observations.csv")
    branch_path = data / "branch_observations.csv"
    if not branch_path.exists():
        branch_path = data / "branch_observations.csv.gz"
    branches = pd.read_csv(branch_path)
    annual = pd.read_csv(data / "annual_weather_load_summary.csv")
    methods = metadata["methods"]
    labels = [metadata["method_labels"][method] for method in methods]
    colors = [metadata["method_colors"][method] for method in methods]

    box_violin_plot(
        [
            branches[f"{method}__available_capacity_percent_of_nominal"].to_numpy()
            for method in methods
        ],
        labels,
        colors,
        "Available Capacity (%)",
        artwork / "model_capacity_drop_box_violin.pdf",
        metric="capacity",
        threshold=float(metadata["capacity_security_margin_percent"]),
        exceed_direction="below",
    )
    box_violin_plot(
        [branches[f"{method}__line_temperature_c"].to_numpy() for method in methods],
        labels,
        colors,
        "Line Temperature (°C)",
        artwork / "model_line_temp_box_violin.pdf",
        metric="temperature",
        threshold=float(metadata["thermal_limit_c"]),
        exceed_direction="above",
    )
    box_violin_plot(
        [scenarios[f"{method}__load_shedding_percent"].to_numpy() for method in methods],
        labels,
        colors,
        "Load Shedding (%)",
        artwork / "model_load_shedding_box_violin.pdf",
    )
    runtime_plot(scenarios, methods, labels, colors, artwork / "model_running_time.pdf")
    scatter_plot(
        scenarios,
        "td_seg_derate_iter_2",
        artwork / "load_load_shedding_scatter.pdf",
    )
    write_panel_a_table(annual, artwork / "panel_a_annual_weather_load_table.tex")
    print(json.dumps({"package": str(package), "artwork_files": 6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

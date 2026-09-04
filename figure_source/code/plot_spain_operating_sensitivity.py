#!/usr/bin/env python3
"""Reproduce single-country operating-sensitivity panels from compact data."""

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
from matplotlib import patches as mpatches  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy import stats  # noqa: E402


PDF_METADATA = {
    "Creator": "publication_pipeline/code/plotting/plot_spain_operating_sensitivity.py",
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


def finish_axes(ax, labels: list[str], ylabel: str, rotation: float = 0) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(False)
    ax.grid(axis="y", linewidth=0, alpha=0)
    ax.grid(axis="x", linewidth=0, alpha=0)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="center", fontweight="bold")
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
    rotation: float = 15,
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
            generator = np.random.default_rng(1800 + index)
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
    finish_axes(ax, labels, ylabel, rotation)
    save_pdf(fig, output)


def annual_statistics(
    frame: pd.DataFrame,
    cases: list[dict],
    ddof: int,
) -> tuple[list[int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    years = sorted(int(year) for year in frame.future_year.unique())
    output = {}
    for case in cases:
        column = f"{case['case_id']}__load_shedding_percent"
        means = []
        deviations = []
        for year in years:
            values = frame.loc[frame.future_year == year, column].to_numpy(dtype=float)
            means.append(values.mean())
            deviations.append(values.std(ddof=ddof))
        output[case["label"]] = (np.asarray(means), np.asarray(deviations))
    return years, output


def grouped_bar_plot(
    frame: pd.DataFrame,
    cases: list[dict],
    colors: list[str],
    output: Path,
    ddof: int,
) -> None:
    years, statistics_by_case = annual_statistics(frame, cases, ddof)
    labels = [case["label"] for case in cases]
    n_scenarios = len(labels)
    bar_width = 0.27
    hatches = ["..", "xx", "///", "+++"][:n_scenarios]
    positions = np.arange(len(years))
    total_width = bar_width * n_scenarios
    offsets = [
        -total_width / 2 + bar_width / 2 + index * bar_width
        for index in range(n_scenarios)
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    value_max = 0.0
    for index, label in enumerate(labels):
        means, deviations = statistics_by_case[label]
        value_max = max(value_max, float(np.max(means + deviations)))
        ax.bar(
            positions + offsets[index],
            means,
            bar_width,
            color=colors,
            alpha=0.8 + index * 0.1,
            linewidth=0,
            hatch=hatches[index],
            edgecolor=(0, 0, 0, 0.6),
            label=label,
        )
        ax.errorbar(
            positions + offsets[index],
            means,
            yerr=[np.zeros_like(deviations), deviations],
            fmt="none",
            elinewidth=2,
            capsize=5,
            capthick=1.0,
            ecolor="gray",
            alpha=0.9,
        )

    legend_handles = [
        mpatches.Patch(
            facecolor="lightgray",
            hatch=hatches[index],
            edgecolor="black",
            linewidth=0.1,
            label=labels[index],
        )
        for index in range(n_scenarios)
    ]
    legend = ax.legend(
        handles=legend_handles,
        ncol=n_scenarios,
        edgecolor="white",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )
    legend.get_frame().set_alpha(0.99)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(False)
    ax.grid(axis="y", linewidth=0, alpha=0)
    ax.grid(axis="x", linewidth=0, alpha=0)
    ax.set_ylabel("Load Shedding (%)", fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(years, ha="center", fontweight="bold")
    ax.set_ylim(0, max(value_max * 1.1, 0.01))
    ax.tick_params(top=False, bottom=False, left=True, right=False)
    save_pdf(fig, output, dpi=300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()

    package = args.package_dir.resolve()
    data = package / "data"
    artwork = package / "artwork"
    artwork.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "plot_metadata.json").read_text())
    ablation_scenarios = pd.read_csv(data / "ablation_scenario_observations.csv")
    branch_path = data / "ablation_branch_observations.csv"
    if not branch_path.exists():
        branch_path = data / "ablation_branch_observations.csv.gz"
    branches = pd.read_csv(branch_path)
    thermal = pd.read_csv(data / "thermal_sensitivity_observations.csv")
    load_growth = pd.read_csv(data / "load_growth_observations.csv")
    storage = pd.read_csv(data / "storage_soc_observations.csv")

    configure_matplotlib()
    methods = metadata["ablation_methods"]
    labels = [metadata["ablation_labels"][method] for method in methods]
    colors = [metadata["ablation_colors"][method] for method in methods]

    box_violin_plot(
        [
            branches[f"{method}__available_capacity_percent_of_nominal"].to_numpy()
            for method in methods
        ],
        labels,
        colors,
        "Available Capacity (%)",
        artwork / "sensitivity_model_capacity_drop_box_violin.pdf",
        metric="capacity",
        threshold=float(metadata["capacity_security_margin_percent"]),
        exceed_direction="below",
    )
    box_violin_plot(
        [branches[f"{method}__line_temperature_c"].to_numpy() for method in methods],
        labels,
        colors,
        "Line Temperature (\N{DEGREE SIGN}C)",
        artwork / "sensitivity_model_line_temp_box_violin.pdf",
        metric="temperature",
        threshold=float(metadata["base_thermal_limit_c"]),
        exceed_direction="above",
    )
    box_violin_plot(
        [
            ablation_scenarios[f"{method}__load_shedding_percent"].to_numpy()
            for method in methods
        ],
        labels,
        colors,
        "Load Shedding (%)",
        artwork / "sensitivity_model_load_shedding_box_violin.pdf",
    )

    thermal_cases = metadata["thermal_cases"]
    box_violin_plot(
        [
            thermal[f"{case['case_id']}__load_shedding_percent"].to_numpy()
            for case in thermal_cases
        ],
        [case["label"] for case in thermal_cases],
        metadata["sequential_red"],
        "Load Shedding (%)",
        artwork / "thermal_load_shedding_box_violin.pdf",
    )
    grouped_bar_plot(
        load_growth,
        metadata["load_growth_cases"],
        metadata["sequential_orange"],
        artwork / "load_sensitivity_grouped_bar.pdf",
        int(metadata["standard_deviation_ddof"]),
    )
    grouped_bar_plot(
        storage,
        metadata["storage_cases"],
        metadata["sequential_blue"],
        artwork / "storage_sensitivity_grouped_bar.pdf",
        int(metadata["standard_deviation_ddof"]),
    )
    print(json.dumps({"package": str(package), "artwork_files": 6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

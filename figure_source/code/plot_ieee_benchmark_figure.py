#!/usr/bin/env python3
"""Reproduce Supplementary Figs. 29--30 from compact plotted-data tables."""

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
from matplotlib.patches import Patch  # noqa: E402


HATCHES = ("..", "xx", "///", "+++")
PDF_METADATA = {
    "Creator": "publication_pipeline/code/plotting/plot_ieee_benchmark_figure.py",
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


def save_pdf(fig, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def scenario_legend(ax, labels: list[str], n_scenarios: int) -> None:
    handles = [
        Patch(
            facecolor="lightgray",
            hatch=HATCHES[index],
            edgecolor="black",
            linewidth=0.1,
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    legend = ax.legend(
        handles=handles,
        ncol=n_scenarios,
        edgecolor="white",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )
    legend.get_frame().set_alpha(0.99)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def finish_axes(ax, labels: list[str], ylabel: str, rotation: int) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(False)
    ax.grid(axis="y", linewidth=0, alpha=0)
    ax.grid(axis="x", linewidth=0, alpha=0)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, ha="center", fontweight="bold", rotation=rotation)
    ax.tick_params(top=False, bottom=False, left=True, right=False)
    for label in ax.get_xticklabels():
        if label.get_text() == "Iter-OPF":
            label.set_color("royalblue")


def grouped_bar_plot(
    table: pd.DataFrame,
    methods: list[str],
    method_labels: dict[str, str],
    method_colors: dict[str, str],
    weather: list[dict],
    output: Path,
    *,
    bar_width: float,
    rotation: int,
) -> None:
    scenario_labels = [scenario["label"] for scenario in weather]
    fig, ax = plt.subplots(
        figsize=(min(len(methods) * len(weather) * 2.5, 20), 5))
    x_positions = np.arange(len(methods))
    start = -(bar_width * len(weather)) / 2 + bar_width / 2
    offsets = [start + index * bar_width for index in range(len(weather))]
    maximum = 0.0
    for scenario_index, scenario in enumerate(weather):
        values = []
        for method in methods:
            selected = table[
                (table["method"] == method)
                & (table["weather_key"] == scenario["key"])
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one load-shedding value for {method}/{scenario['key']}; "
                    f"found {len(selected)}"
                )
            values.append(float(selected.iloc[0].load_shedding_percent))
        values_array = np.asarray(values)
        maximum = max(maximum, float(values_array.max()))
        bars = ax.bar(
            x_positions + offsets[scenario_index],
            values_array,
            bar_width,
            color=[method_colors[method] for method in methods],
            alpha=0.6 + scenario_index * 0.05,
            linewidth=0,
            hatch=HATCHES[scenario_index],
            edgecolor=(0, 0, 0, 0.6),
        )
        ax.plot(
            x_positions + offsets[scenario_index],
            values_array,
            "o",
            color="darkred",
            markersize=5,
            markeredgecolor="darkred",
            markeredgewidth=1.2,
        )
        for index, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                values_array[index] * 1.02,
                f"{values_array[index]:.2f}",
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=4,
            )
    finish_axes(
        ax,
        [method_labels[method] for method in methods],
        "Load Shedding (%)",
        rotation,
    )
    ax.set_ylim(0, max(maximum * 1.1, 1))
    scenario_legend(ax, scenario_labels, len(weather))
    save_pdf(fig, output)


def grouped_violin_plot(
    table: pd.DataFrame,
    methods: list[str],
    method_labels: dict[str, str],
    method_colors: dict[str, str],
    weather: list[dict],
    output: Path,
    *,
    violin_width: float,
    rotation: int,
    thermal_limit_c: float,
) -> None:
    scenario_labels = [scenario["label"] for scenario in weather]
    fig, ax = plt.subplots(
        figsize=(min(len(methods) * len(weather) * 2.5, 20), 5))
    x_positions = np.arange(len(methods))
    start = -(violin_width * len(weather)) / 2 + violin_width / 2
    offsets = [start + index * violin_width for index in range(len(weather))]
    value_min = np.inf
    value_max = -np.inf
    for scenario_index, scenario in enumerate(weather):
        for method_index, method in enumerate(methods):
            selected = table[
                (table["method"] == method)
                & (table["weather_key"] == scenario["key"])
            ].sort_values("line_index")
            values = selected.line_temperature_c.to_numpy(dtype=float)
            if not len(values):
                raise ValueError(f"No line temperatures for {method}/{scenario['key']}")
            centre = x_positions[method_index] + offsets[scenario_index]
            parts = ax.violinplot(
                [values],
                positions=[centre],
                widths=violin_width * 0.9,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(method_colors[method])
                body.set_edgecolor("black")
                body.set_alpha(0.6 + scenario_index * 0.05)
                body.set_hatch(HATCHES[scenario_index])
            lower = float(values.min())
            upper = float(values.max())
            mean = float(values.mean())
            cap_width = violin_width * 0.15
            ax.plot([centre, centre], [lower, upper], color=method_colors[method], linewidth=0.2)
            ax.plot([centre - cap_width, centre + cap_width], [lower, lower], color="dimgray", linewidth=1.5)
            ax.plot([centre - cap_width, centre + cap_width], [upper, upper], color="dimgray", linewidth=1.5)
            ax.plot(centre, mean, "o", color="darkred", markersize=5, markeredgewidth=1.2)
            value_min = min(value_min, lower)
            value_max = max(value_max, upper)
    ax.axhline(y=thermal_limit_c, color="red", linestyle="--", linewidth=2, alpha=0.6)
    ax.text(
        0,
        thermal_limit_c + 1,
        "Thermal limit",
        transform=ax.get_yaxis_transform(),
        color="red",
        ha="left",
        va="bottom",
        alpha=0.75,
        fontweight="bold",
    )
    finish_axes(
        ax,
        [method_labels[method] for method in methods],
        r"Line Temperature ($^\circ$C)",
        rotation,
    )
    ax.set_ylim(value_min * 0.9, max(value_max * 1.1, thermal_limit_c + 5))
    scenario_legend(ax, scenario_labels, len(weather))
    save_pdf(fig, output, dpi=500)


def runtime_plot(
    table: pd.DataFrame,
    methods: list[str],
    method_labels: dict[str, str],
    method_colors: dict[str, str],
    output: Path,
) -> None:
    observations = [
        table[table.method == method].runtime_s.to_numpy(dtype=float)
        for method in methods
    ]
    means = np.asarray([values.mean() for values in observations])
    standard_deviations = np.asarray([values.std(ddof=0) for values in observations])
    fig, ax = plt.subplots(figsize=((len(methods) // 5 + 1) * 10, 5))
    x_positions = np.arange(len(methods))
    bars = ax.bar(
        x_positions,
        means,
        alpha=0.9,
        color=[method_colors[method] for method in methods],
    )
    ax.errorbar(
        x_positions,
        means,
        yerr=standard_deviations,
        fmt="none",
        elinewidth=1.5,
        capsize=14,
        capthick=2,
        ecolor="dimgray",
    )
    ax.scatter(x_positions, means, marker="o", color="darkred", s=40, zorder=3)
    for index, bar in enumerate(bars):
        if means[index] > 1e-2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                means[index] + standard_deviations[index] * 1.01,
                f"ave:{means[index]:.2f}",
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=4,
            )
    finish_axes(
        ax,
        [method_labels[method] for method in methods],
        "Running Time (s)",
        30,
    )
    ax.set_ylim(0, float(means.max() + standard_deviations.max() * 1.1))
    save_pdf(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_matplotlib()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "plot_metadata.json").read_text())
    methods = metadata["methods"]
    labels = metadata["method_labels"]
    colours = metadata["method_colors"]
    weather = metadata["weather_scenarios"]
    shedding = pd.read_csv(source / "load_shedding.csv")
    temperatures = pd.read_csv(source / "line_temperatures.csv")
    if metadata["supplementary_figure"] == 29:
        grouped_bar_plot(
            shedding,
            methods,
            labels,
            colours,
            weather,
            output / "SA_load_shedding_0.9 Load Ratio.pdf",
            bar_width=0.37,
            rotation=30,
        )
        grouped_violin_plot(
            temperatures,
            methods,
            labels,
            colours,
            weather,
            output / "SA_temp_dist_0.9 Load Ratio.pdf",
            violin_width=0.37,
            rotation=30,
            thermal_limit_c=metadata["thermal_limit_c"],
        )
        runtime_plot(
            pd.read_csv(source / "runtime_observations.csv"),
            methods,
            labels,
            colours,
            output / "SA_run_time.pdf",
        )
    else:
        for load_ratio in metadata["load_ratios"]:
            load_table = shedding[np.isclose(shedding.load_ratio, load_ratio)]
            temperature_table = temperatures[np.isclose(temperatures.load_ratio, load_ratio)]
            title = f"{load_ratio:.1f} Load Ratio"
            grouped_bar_plot(
                load_table,
                methods,
                labels,
                colours,
                weather,
                output / f"WA_load_shedding_{title}.pdf",
                bar_width=0.2,
                rotation=0,
            )
            grouped_violin_plot(
                temperature_table,
                methods,
                labels,
                colours,
                weather,
                output / f"WA_temp_dist_{title}.pdf",
                violin_width=0.2,
                rotation=0,
                thermal_limit_c=metadata["thermal_limit_c"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

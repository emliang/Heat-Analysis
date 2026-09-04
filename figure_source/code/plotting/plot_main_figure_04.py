#!/usr/bin/env python3
"""Draw the complete 180-mm main Fig. 4 from figure-level Source Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 94.0
MM_PER_INCH = 25.4
MEAN_COLOR = "#8B0000"
MEAN_MARKER_SIZE = 8.0
GROUPED_MEAN_MARKER_SIZE = 2.2
STAT_COLOR = "#5F5F5F"
CASE_HATCHES = ["....", "xxxx", "////"]


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
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.0,
            "hatch.linewidth": 0.28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(top=False, right=False, direction="out", pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.grid(False)


def mean_text(value: float) -> str:
    display = 0.0 if abs(value) < 0.005 else value
    return f"{display:.2f}"


def tukey_whiskers(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = values[values >= q1 - 1.5 * iqr].min()
    upper = values[values <= q3 + 1.5 * iqr].max()
    return min(float(q1), float(lower)), max(float(q3), float(upper))


def half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    centre: float,
    width: float,
    color: str,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    value_grid = np.linspace(values.min(), values.max(), 280)
    if np.ptp(values) > 1e-12:
        density = stats.gaussian_kde(values)(value_grid)
        density = density / density.max() * width
    else:
        density = np.full_like(value_grid, width * 0.04)
    ax.fill_betweenx(
        value_grid,
        centre,
        centre + density,
        color=color,
        alpha=0.9,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        centre + density,
        value_grid,
        color=color,
        linewidth=0.7,
        alpha=0.95,
        zorder=2,
    )


def annotate_mean(
    ax: plt.Axes,
    *,
    x: float,
    values: np.ndarray,
    y_limits: tuple[float, float],
    placement: str,
) -> None:
    mean = float(np.mean(values))
    lower, upper = tukey_whiskers(values)
    del lower
    span = y_limits[1] - y_limits[0]
    if placement == "lower-left":
        text_y = mean - 0.07 * span
        vertical_alignment = "top"
    else:
        text_y = max(upper + 0.045 * span, mean + 0.07 * span)
        if text_y + 0.08 * span > y_limits[1]:
            text_y = mean - 0.07 * span
            vertical_alignment = "top"
        else:
            vertical_alignment = "bottom"
    ax.scatter(
        x, mean, marker="o", color=MEAN_COLOR, s=MEAN_MARKER_SIZE, zorder=5
    )
    ax.annotate(
        f"ave:\n{mean_text(mean)}",
        xy=(x, mean),
        xytext=(x - 0.34, text_y),
        arrowprops={
            "arrowstyle": "-",
            "color": "#333333",
            "lw": 0.45,
            "linestyle": "--",
        },
        fontsize=6.0,
        fontweight="bold",
        ha="left",
        va=vertical_alignment,
        annotation_clip=False,
        zorder=6,
    )


def distribution_panel(
    ax: plt.Axes,
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    ylabel: str,
    y_limits: tuple[float, float],
    mean_placement: str,
    thermal_limit: float | None = None,
) -> None:
    positions = np.arange(len(labels), dtype=float)
    for index, (values, color) in enumerate(zip(observations, colors, strict=True)):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        box_x = positions[index] - 0.10
        violin_x = positions[index] + 0.06
        box = ax.boxplot(
            [values],
            positions=[box_x],
            widths=0.24,
            patch_artist=True,
            whis=1.5,
            showfliers=False,
            zorder=3,
            medianprops={"color": STAT_COLOR, "linewidth": 0.72},
            whiskerprops={"color": STAT_COLOR, "linewidth": 0.62},
            capprops={"color": STAT_COLOR, "linewidth": 0.62},
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_edgecolor("white")
        box["boxes"][0].set_linewidth(0.4)
        box["boxes"][0].set_alpha(0.9)
        half_violin(ax, values, centre=violin_x, width=0.25, color=color)
        annotate_mean(
            ax,
            x=box_x,
            values=values,
            y_limits=y_limits,
            placement=mean_placement,
        )

    ax.set_xlim(-0.58, len(labels) - 0.40)
    ax.set_ylim(*y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="center")
    ax.set_ylabel(ylabel)
    if thermal_limit is not None:
        ax.axhline(
            thermal_limit,
            color="#FF4F58",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        ax.text(
            -0.53,
            thermal_limit + 2.0,
            "Thermal\nlimit",
            color="#FF4F58",
            fontsize=6.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            multialignment="center",
        )
    if y_limits[0] <= 0 <= y_limits[1]:
        ax.axhline(0.0, color="#777777", linewidth=0.4, zorder=1)
    style_axis(ax)


def grouped_year_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    panel: dict,
) -> None:
    years = sorted(int(value) for value in frame["future_year"].unique())
    case_ids = panel["case_ids"]
    labels = panel["display_labels"]
    year_colors = panel["year_colors"]
    positions = np.arange(len(years), dtype=float)
    width = 0.22
    offsets = (np.arange(len(case_ids)) - (len(case_ids) - 1) / 2) * width

    for case_index, case_id in enumerate(case_ids):
        observations = []
        case_positions = []
        for year_index, year in enumerate(years):
            values = frame.loc[
                (frame["case_id"] == case_id) & (frame["future_year"] == year),
                "load_shedding_percent",
            ].to_numpy(dtype=float)
            observations.append(values)
            case_positions.append(positions[year_index] + offsets[case_index])
        box = ax.boxplot(
            observations,
            positions=case_positions,
            widths=width * 0.82,
            patch_artist=True,
            whis=1.5,
            showfliers=False,
            medianprops={"color": STAT_COLOR, "linewidth": 0.72},
            whiskerprops={"color": STAT_COLOR, "linewidth": 0.62},
            capprops={"color": STAT_COLOR, "linewidth": 0.62},
        )
        for year_index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(year_colors[year_index])
            patch.set_edgecolor((0.30, 0.30, 0.30, 0.58))
            patch.set_linewidth(0.32)
            patch.set_hatch(CASE_HATCHES[case_index])
            patch.set_alpha(0.9)
        for position, values in zip(case_positions, observations, strict=True):
            ax.scatter(
                position,
                float(np.mean(values)),
                marker="o",
                color=MEAN_COLOR,
                s=GROUPED_MEAN_MARKER_SIZE,
                zorder=4,
            )

    handles = [
        mpatches.Patch(
            facecolor="lightgray",
            hatch=CASE_HATCHES[index],
            edgecolor="#666666",
            linewidth=0.3,
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    legend = ax.legend(
        handles=handles,
        ncol=len(labels),
        edgecolor="white",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderpad=0.2,
        handlelength=1.7,
        columnspacing=0.8,
        labelspacing=0.2,
    )
    legend.get_frame().set_alpha(0.99)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(year) for year in years])
    ax.set_ylabel("Load shedding (%)")
    upper = float(frame["load_shedding_percent"].max())
    rounded_upper = np.ceil(upper * 1.04 * 20.0) / 20.0
    ax.set_ylim(0.0, max(float(rounded_upper), 0.05))
    ax.axhline(0.0, color="#777777", linewidth=0.4, zorder=1)
    style_axis(ax)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.09,
        label,
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="top",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    configure_fonts()
    source_dir = args.source_dir.resolve()
    metadata = json.loads((source_dir / "Fig4_plot_metadata.json").read_text())
    panel_a_frame = pd.read_csv(source_dir / metadata["panels"]["a"]["source_file"])
    panel_b_frame = pd.read_csv(source_dir / metadata["panels"]["b"]["source_file"])
    panel_c_frame = pd.read_csv(source_dir / metadata["panels"]["c"]["source_file"])
    panel_d_frame = pd.read_csv(source_dir / metadata["panels"]["d"]["source_file"])

    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    axes = {
        "a": fig.add_axes((0.070, 0.585, 0.405, 0.335)),
        "b": fig.add_axes((0.570, 0.585, 0.405, 0.335)),
        "c": fig.add_axes((0.070, 0.115, 0.405, 0.335)),
        "d": fig.add_axes((0.570, 0.115, 0.405, 0.335)),
    }

    panel_a = metadata["panels"]["a"]
    observations_a = [
        panel_a_frame.loc[
            panel_a_frame["method_id"] == method, "load_shedding_percent"
        ].to_numpy(dtype=float)
        for method in panel_a["method_ids"]
    ]
    distribution_panel(
        axes["a"],
        observations_a,
        panel_a["display_labels"],
        panel_a["colors"],
        ylabel="Load shedding (%)",
        y_limits=tuple(panel_a["y_limits"]),
        mean_placement="auto",
    )

    panel_b = metadata["panels"]["b"]
    observations_b = [
        panel_b_frame[f"{method}__line_temperature_c"].to_numpy(dtype=float)
        for method in panel_b["method_ids"]
    ]
    distribution_panel(
        axes["b"],
        observations_b,
        panel_b["display_labels"],
        panel_b["colors"],
        ylabel="Line temperature (°C)",
        y_limits=tuple(panel_b["y_limits"]),
        mean_placement="lower-left",
        thermal_limit=float(panel_b["thermal_limit_c"]),
    )
    grouped_year_panel(axes["c"], panel_c_frame, metadata["panels"]["c"])
    grouped_year_panel(axes["d"], panel_d_frame, metadata["panels"]["d"])

    for label, ax in axes.items():
        panel_label(ax, label)

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {
        "Title": "Figure 4 - Operating sensitivity",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_main_figure_04.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(args.pdf, format="pdf", dpi=300, metadata=pdf_metadata)
    fig.savefig(args.png, format="png", dpi=300, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

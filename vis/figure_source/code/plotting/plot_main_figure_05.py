#!/usr/bin/env python3
"""Draw the complete 180-mm main Fig. 5 from figure-level Source Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 105.0
MM_PER_INCH = 25.4
MEAN_MARKER_SIZE = 8.0
PDF_METADATA = {
    "Creator": "plot_main_figure_05.py",
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
            "font.size": 5.8,
            "axes.labelsize": 7.0,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(top=False, right=False, direction="out", pad=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_linewidth(0.6)
        ax.spines[name].set_color("#555555")


def draw_boxes(
    ax: plt.Axes,
    observations: list[np.ndarray],
    positions: np.ndarray,
    colors: list[str],
    *,
    width: float,
) -> None:
    boxes = ax.boxplot(
        observations,
        positions=positions,
        widths=width,
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        medianprops={"color": "dimgray", "linewidth": 0.7},
        whiskerprops={"color": "dimgray", "linewidth": 0.6},
        capprops={"color": "dimgray", "linewidth": 0.6},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)
        patch.set_alpha(0.88)


def add_mean_note(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    *,
    y_span: float,
    placement: str,
    x_offset: float = -0.38,
) -> None:
    mean = float(np.mean(values))
    display = 0.0 if abs(mean) < 0.005 else mean
    ax.scatter(
        [x], [mean], s=MEAN_MARKER_SIZE, color="darkred", zorder=5, clip_on=True
    )
    if placement == "lower-left":
        text_y = mean - 0.085 * y_span
        va = "top"
    elif placement == "upper-left":
        text_y = mean + 0.075 * y_span
        va = "bottom"
    else:
        raise ValueError(placement)
    ax.annotate(
        f"ave:\n{display:.2f}",
        xy=(x, mean),
        xytext=(x + x_offset, text_y),
        arrowprops={
            "arrowstyle": "-",
            "color": "#333333",
            "linewidth": 0.55,
            "linestyle": "--",
        },
        fontsize=6.0,
        fontweight="bold",
        ha="left",
        va=va,
        annotation_clip=False,
        zorder=6,
    )


def draw_panel_a(
    ax: plt.Axes,
    scenarios: pd.DataFrame,
    order: list[str],
    short_labels: dict[str, str],
    colors: list[str],
) -> None:
    observations = [
        scenarios.loc[scenarios.country_code == country, "air_temperature_c"].to_numpy()
        for country in order
    ]
    positions = np.arange(len(order), dtype=float)
    draw_boxes(ax, observations, positions, colors, width=0.54)
    ax.set_ylim(15.0, 52.0)
    for x, values in zip(positions, observations, strict=True):
        add_mean_note(ax, float(x), values, y_span=37.0, placement="lower-left")
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_xticks(positions)
    ax.set_xticklabels([short_labels[country] for country in order], fontweight="bold")
    ax.set_yticks([20, 30, 40, 50])
    ax.set_ylabel("Air temperature (°C)")
    style_axis(ax)


def draw_panel_b(
    ax: plt.Axes,
    scenarios: pd.DataFrame,
    order: list[str],
    short_labels: dict[str, str],
    colors: list[str],
) -> None:
    observations = [
        scenarios.loc[
            scenarios.country_code == country, "load_shedding_percent"
        ].to_numpy()
        for country in order
    ]
    positions = np.arange(len(order), dtype=float)
    draw_boxes(ax, observations, positions, colors, width=0.54)
    ax.set_ylim(-0.015, 0.60)
    for x, values in zip(positions, observations, strict=True):
        add_mean_note(ax, float(x), values, y_span=0.615, placement="upper-left")
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_xticks(positions)
    ax.set_xticklabels([short_labels[country] for country in order], fontweight="bold")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylabel("Load shedding (%)")
    style_axis(ax)


def population_kde(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the approved SI density estimate using every observation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.linspace(float(values.min()), float(values.max()), 300)
    if np.ptp(values) > 1e-12:
        density = stats.gaussian_kde(values)(grid)
    else:
        density = np.ones_like(grid)
    return grid, density


def draw_curly_brace(
    ax: plt.Axes,
    x: float,
    y_lo: float,
    y_hi: float,
) -> float:
    width = 0.035
    tip = 0.025
    height = y_hi - y_lo
    middle = (y_hi + y_lo) / 2.0
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
            linewidth=0.55,
            capstyle="round",
            clip_on=False,
        )
    )
    return x + width + tip


def draw_panel_c(
    ax: plt.Axes,
    branches: pd.DataFrame,
    order: list[str],
    names: dict[str, str],
    colors: list[str],
    threshold: float,
) -> None:
    observations = [
        branches.loc[
            branches.country_code == country,
            "available_capacity_percent_of_nominal",
        ].to_numpy()
        for country in order
    ]
    positions = np.arange(len(order), dtype=float)
    box_positions = positions - 0.10
    draw_boxes(ax, observations, box_positions, colors, width=0.24)
    y_min, y_max = 45.0, 104.0
    sub_width = 0.25
    for x, box_x, values, color in zip(
        positions, box_positions, observations, colors, strict=True
    ):
        centre = x + 0.10
        grid, density = population_kde(values)
        if float(density.max()) > 0:
            density = density / float(density.max()) * sub_width
        ax.fill_betweenx(
            grid,
            centre,
            centre + density,
            color=color,
            alpha=0.88,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            centre + density,
            grid,
            color=color,
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )
        add_mean_note(
            ax,
            float(box_x),
            values,
            y_span=y_max - y_min,
            placement="lower-left",
            x_offset=-0.34,
        )
        percentage = float(np.mean(values < threshold - 1e-3) * 100.0)
        below = grid <= threshold
        if percentage > 0.1 and below.any() and float(values.min()) < threshold:
            brace_x = centre + float(density[below].max()) + 0.018
            tip_x = draw_curly_brace(ax, brace_x, float(values.min()), threshold)
            ax.text(
                tip_x + 0.035,
                (float(values.min()) + threshold) / 2.0,
                f"{percentage:.1f}%",
                fontsize=6.0,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=False,
            )

    ax.axhline(
        threshold,
        color="#4d55ff",
        linestyle="--",
        linewidth=0.9,
        alpha=0.78,
        zorder=0,
    )
    ax.text(
        0.01,
        threshold - 1.0,
        "Security\nmargin",
        transform=ax.get_yaxis_transform(),
        fontsize=6.0,
        color="#3540e8",
        fontweight="bold",
        ha="left",
        va="top",
        multialignment="left",
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.58, len(order) - 0.22)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [names[country] for country in order],
        rotation=18,
        ha="right",
        fontweight="bold",
    )
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_ylabel("Available Capacity (%)")
    style_axis(ax)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    source = args.source_dir.resolve()
    scenarios = pd.read_csv(source / "Fig5ab_scenario_observations.csv")
    branches = pd.read_csv(source / "Fig5c_available_capacity_observations.csv.gz")
    rules = json.loads((source / "Fig5_display_rules.json").read_text())
    order = list(rules["country_order"])
    short_labels = rules["country_short_labels"]
    names = rules["country_names"]
    colors = [rules["country_colors"][country] for country in order]

    configure_fonts()
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    ax_a = fig.add_axes([0.065, 0.585, 0.405, 0.335])
    ax_b = fig.add_axes([0.565, 0.585, 0.405, 0.335])
    ax_c = fig.add_axes([0.065, 0.105, 0.905, 0.335])

    draw_panel_a(ax_a, scenarios, order, short_labels, colors)
    draw_panel_b(ax_b, scenarios, order, short_labels, colors)
    draw_panel_c(
        ax_c,
        branches,
        order,
        names,
        colors,
        float(rules["capacity_security_margin_percent"]),
    )
    for label, ax in (("a", ax_a), ("b", ax_b), ("c", ax_c)):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.040,
            box.y1 + 0.012,
            label,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.pdf,
        format="pdf",
        dpi=300,
        facecolor="white",
        metadata=PDF_METADATA,
    )
    fig.savefig(
        args.png,
        format="png",
        dpi=300,
        facecolor="white",
        metadata={"Software": "plot_main_figure_05.py"},
    )
    plt.close(fig)
    print(f"pdf={args.pdf}")
    print(f"png={args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

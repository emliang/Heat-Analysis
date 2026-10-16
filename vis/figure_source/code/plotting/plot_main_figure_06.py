#!/usr/bin/env python3
"""Draw the complete 180-mm main Fig. 6 from figure-level Source Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 45.0
MM_PER_INCH = 25.4
MEAN_MARKER_SIZE = 8.0


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
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
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
        }
    )


def mean_text(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def upper_tukey_whisker(values: np.ndarray) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    return float(values[values <= upper_fence].max())


def draw_distribution_panel(ax, frame: pd.DataFrame, panel: dict) -> None:
    ids = panel["configuration_ids"]
    labels = panel["display_labels"]
    colors = panel["colors"]
    positions = np.arange(len(ids), dtype=float)
    y_min, y_max = panel["y_limits_percent"]
    y_span = y_max - y_min

    for index, (configuration_id, color) in enumerate(zip(ids, colors, strict=True)):
        values = frame.loc[
            frame.configuration_id == configuration_id, "load_shedding_percent"
        ].to_numpy(dtype=float)
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
            medianprops={"color": "dimgray", "linewidth": 0.75},
            whiskerprops={"color": "dimgray", "linewidth": 0.65},
            capprops={"color": "dimgray", "linewidth": 0.65},
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_edgecolor("white")
        box["boxes"][0].set_linewidth(0.45)
        box["boxes"][0].set_alpha(0.9)

        value_grid = np.linspace(values.min(), values.max(), 320)
        if np.ptp(values) > 1e-12:
            density = stats.gaussian_kde(values)(value_grid)
            density = density / density.max() * 0.25
        else:
            density = np.full_like(value_grid, 0.01)
        ax.fill_betweenx(
            value_grid,
            violin_x,
            violin_x + density,
            color=color,
            alpha=0.9,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            violin_x + density,
            value_grid,
            color=color,
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )

        mean = float(values.mean())
        ax.scatter(
            box_x,
            mean,
            marker="o",
            color="#8B0000",
            s=MEAN_MARKER_SIZE,
            zorder=4,
        )
        # Keep the mean note close to the visible distribution instead of
        # anchoring every note at the top of the panel.
        text_y = max(
            y_min + 0.20 * y_span,
            upper_tukey_whisker(values) + 0.08 * y_span,
            mean + 0.10 * y_span,
        )
        text_y = min(text_y, y_max - 0.08 * y_span)
        text_x = positions[index] - 0.34
        ax.annotate(
            f"ave:\n{mean_text(mean)}",
            xy=(box_x, mean),
            xytext=(text_x, text_y),
            arrowprops={
                "arrowstyle": "-",
                "color": "#333333",
                "lw": 0.45,
                "linestyle": "--",
            },
            fontsize=6.0,
            fontweight="bold",
            ha="left",
            va="top",
            annotation_clip=False,
            zorder=5,
        )

    ax.set_xlim(-0.58, len(ids) - 0.40)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(positions)
    tick_labels = [label.replace("↔", " ↔ ") for label in labels]
    ax.set_xticklabels(
        tick_labels,
        rotation=15,
        ha="center",
        fontsize=7.0,
        fontweight="bold",
    )
    ax.set_ylabel("Load shedding (%)")
    ax.tick_params(top=False, right=False, direction="out", pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.axhline(0.0, color="#777777", linewidth=0.45, zorder=1)
    ax.grid(False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    configure_fonts()
    source_dir = args.source_dir.resolve()
    metadata = json.loads((source_dir / "Fig6_plot_metadata.json").read_text())
    frames = {
        "a": pd.read_csv(source_dir / metadata["panels"]["a"]["source_file"]),
        "b": pd.read_csv(source_dir / metadata["panels"]["b"]["source_file"]),
    }

    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    axes = [
        fig.add_axes((0.070, 0.25, 0.405, 0.63)),
        fig.add_axes((0.570, 0.25, 0.405, 0.63)),
    ]
    for label, ax in zip(("a", "b"), axes, strict=True):
        draw_distribution_panel(ax, frames[label], metadata["panels"][label])
        ax.text(
            -0.16,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="top",
        )

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {
        "Title": "Figure 6 - Cross-border grid interconnection comparison",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_main_figure_06.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(args.pdf, format="pdf", dpi=300, metadata=pdf_metadata)
    fig.savefig(args.png, format="png", dpi=300, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

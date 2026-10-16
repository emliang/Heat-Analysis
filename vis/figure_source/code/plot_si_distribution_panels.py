#!/usr/bin/env python3
"""Generate versioned SI distribution-plot candidates from frozen source data.

This script is deliberately read-only with respect to the release candidate.  It
writes all candidate artwork, compact boxplot summaries and provenance records
under a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager, patches as mpatches  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
from scipy import stats  # noqa: E402


PDF_METADATA = {
    "Creator": "plot_si_distribution_candidates.py",
    "CreationDate": None,
    "ModDate": None,
}

HALF_WIDTH_FIGSIZE = (10, 4)
FULL_WIDTH_FIGSIZE = (20, 4)
FULL_WIDTH_TALL_FIGSIZE = (20, 5)


@dataclass(frozen=True)
class BoxGroup:
    figure: int
    panel: str
    category: str
    subgroup: str
    observation_unit: str
    values: np.ndarray


@dataclass
class MeanNote:
    annotation: Annotation
    mean_y: float
    x: float
    x_offset: float
    plot_span: float
    placement_mode: str
    placed_lower: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    sns.set_theme(style="white")
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


def save_figure(fig: plt.Figure, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        format="pdf",
        dpi=500,
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    preview = pdf_path.parent.parent.parent / "previews" / pdf_path.parent.name
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        preview / f"{pdf_path.stem}.png",
        format="png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def style_axes(ax: plt.Axes, labels: list[str], ylabel: str, rotation: float) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="center", fontweight="bold")
    ax.tick_params(top=False, bottom=False, left=True, right=False)


def tukey_whiskers(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower = values[values >= lower_fence].min()
    upper = values[values <= upper_fence].max()
    # Match Matplotlib's boxplot convention when an interpolated quartile lies
    # between the last inlier and the first outlier.
    lower = min(float(q1), float(lower))
    upper = max(float(q3), float(upper))
    return float(lower), float(upper)


def add_mean_marker(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    *,
    annotate: bool,
    x_offset: float = -0.38,
    plot_bounds: tuple[float, float] | None = None,
    annotation_placement: str = "auto",
) -> MeanNote | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    display_mean = 0.0 if abs(mean) < 0.005 else mean
    ax.scatter(x, mean, marker="o", color="darkred", s=40, zorder=4)
    if not annotate:
        return None
    annotation_y = mean_annotation_y(values)
    vertical_alignment = "bottom"
    plot_span = max(float(np.ptp(values)), abs(mean), 1e-6)
    placed_lower = False
    if plot_bounds is not None:
        lower_bound, upper_bound = plot_bounds
        plot_span = max(upper_bound - lower_bound, 1e-6)
        text_headroom = 0.08 * plot_span
        if annotation_placement == "lower-left":
            annotation_y = mean - 0.08 * plot_span
            vertical_alignment = "top"
            placed_lower = True
        elif annotation_placement == "auto":
            if annotation_y + text_headroom > upper_bound:
                annotation_y = mean - 0.08 * plot_span
                vertical_alignment = "top"
                placed_lower = True
        else:
            raise ValueError(f"Unknown mean-annotation placement: {annotation_placement}")
    annotation = ax.annotate(
        f"ave:\n{display_mean:.2f}",
        xy=(x, mean),
        xytext=(x + x_offset, annotation_y),
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7, "linestyle": "--"},
        fontsize=14,
        ha="left",
        va=vertical_alignment,
        fontweight="bold",
        annotation_clip=False,
        zorder=5,
    )
    return MeanNote(
        annotation=annotation,
        mean_y=mean,
        x=x,
        x_offset=x_offset,
        plot_span=plot_span,
        placement_mode=annotation_placement,
        placed_lower=placed_lower,
    )


def resolve_mean_note_collisions(
    fig: plt.Figure,
    ax: plt.Axes,
    notes: list[MeanNote],
    *,
    reference_threshold: float | None = None,
    reference_label: Text | None = None,
) -> None:
    """Move auto-positioned notes lower-left when final rendering conflicts.

    The original plotting rule uses an upper-left mean note unless that note
    would leave the axes. Reference-line panels need one additional check:
    upper notes must not overlap the thermal/security line or its label.
    """

    if not notes:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    threshold_y = None
    if reference_threshold is not None:
        threshold_y = float(ax.transData.transform((0.0, reference_threshold))[1])
    label_box = None
    if reference_label is not None:
        label_box = reference_label.get_window_extent(renderer=renderer).expanded(1.04, 1.08)

    moved = False
    for note in notes:
        if note.placement_mode != "auto" or note.placed_lower:
            continue
        note_box = note.annotation.get_window_extent(renderer=renderer)
        crosses_upper_bound = note_box.y1 > axes_box.y1
        crosses_reference = (
            threshold_y is not None
            and note_box.y0 - 2.0 <= threshold_y <= note_box.y1 + 2.0
        )
        overlaps_reference_label = label_box is not None and note_box.overlaps(label_box)
        if not (crosses_upper_bound or crosses_reference or overlaps_reference_label):
            continue
        note.annotation.xyann = (
            note.x + note.x_offset,
            note.mean_y - 0.08 * note.plot_span,
        )
        note.annotation.set_va("top")
        note.placed_lower = True
        moved = True
    if moved:
        fig.canvas.draw()


def mean_annotation_y(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    _, upper_whisker = tukey_whiskers(values)
    span = max(float(values.max() - values.min()), abs(upper_whisker), 1e-6)
    return upper_whisker + 0.06 * span


def rounded_visible_upper(value: float) -> float:
    target = max(value * 1.12, 1e-6)
    magnitude = 10.0 ** np.floor(np.log10(target))
    return float(np.ceil(target / magnitude) * magnitude)


def distribution_plot_bounds(
    observations: list[np.ndarray],
    *,
    threshold: float | None = None,
) -> tuple[float, float]:
    finite_values = np.concatenate(
        [np.asarray(values, dtype=float)[np.isfinite(values)] for values in observations]
    )
    lower = float(finite_values.min())
    upper = float(finite_values.max())
    if threshold is not None:
        lower = min(lower, threshold)
        upper = max(upper, threshold)
    span = max(upper - lower, abs(upper), 1e-6)
    margin = 0.05 * span
    return lower - margin, upper + margin


def boxplot_only(
    observations: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    output: Path,
    *,
    rotation: float = 0,
    figsize: tuple[float, float] = HALF_WIDTH_FIGSIZE,
    annotate_means: bool = True,
    y_axis_mode: str = "full_observation_range",
    mean_annotation_placement: str = "auto",
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(labels))
    box = ax.boxplot(
        observations,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        medianprops={"color": "dimgray", "linewidth": 1.2},
        whiskerprops={"color": "dimgray", "linewidth": 1.2},
        capprops={"color": "dimgray", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_alpha(0.9)
    style_axes(ax, labels, ylabel, rotation)
    finite_values = np.concatenate([np.asarray(values)[np.isfinite(values)] for values in observations])
    if y_axis_mode == "visible_statistics":
        visible_upper = max(
            max(
                float(np.mean(values)),
                tukey_whiskers(values)[1],
                mean_annotation_y(values) if annotate_means else 0.0,
            )
            for values in observations
        )
        upper = rounded_visible_upper(visible_upper)
    elif y_axis_mode == "full_observation_range":
        upper = float(finite_values.max()) * 1.06 if finite_values.size else 1.0
    else:
        raise ValueError(f"Unknown y-axis mode: {y_axis_mode}")
    upper = max(upper, 1e-6)
    ax.set_ylim(0, upper)
    mean_notes: list[MeanNote] = []
    for position, values in zip(positions, observations, strict=True):
        note = add_mean_marker(
            ax,
            float(position),
            values,
            annotate=annotate_means,
            plot_bounds=(0.0, upper),
            annotation_placement=mean_annotation_placement,
        )
        if note is not None:
            mean_notes.append(note)
    resolve_mean_note_collisions(fig, ax, mean_notes)
    save_figure(fig, output)


def draw_curly_brace(ax: plt.Axes, x: float, y_lo: float, y_hi: float) -> float:
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
    rotation: float = 0,
    figsize: tuple[float, float] = HALF_WIDTH_FIGSIZE,
    mean_annotation_placement: str = "auto",
) -> None:
    """Draw a half-violin with a standard box, median and annotated mean.

    Fliers are deliberately suppressed. The violin represents the complete
    observation population; the box and whiskers use Tukey's 1.5-IQR rule.
    """

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(labels))
    category_width = 1.0
    sub_width = category_width / 4
    plot_bounds = distribution_plot_bounds(observations, threshold=threshold)
    mean_notes: list[MeanNote] = []

    for index, raw_values in enumerate(observations):
        values = np.asarray(raw_values, dtype=float)
        values = values[np.isfinite(values)]
        box_centre = positions[index] - category_width / 10
        violin_centre = positions[index] + category_width / 10
        box = ax.boxplot(
            [values],
            positions=[box_centre],
            widths=sub_width,
            patch_artist=True,
            zorder=2,
            whis=1.5,
            showfliers=False,
            medianprops={"color": "dimgray", "linewidth": 1.2},
            whiskerprops={"color": "dimgray", "linewidth": 1.0},
            capprops={"color": "dimgray", "linewidth": 1.0},
        )
        box["boxes"][0].set(facecolor=colors[index], alpha=0.9, edgecolor="white")
        note = add_mean_marker(
            ax,
            float(box_centre),
            values,
            annotate=True,
            plot_bounds=plot_bounds,
            annotation_placement=mean_annotation_placement,
        )
        if note is not None:
            mean_notes.append(note)

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

    reference_label: Text | None = None
    if threshold is not None and "Temperature" in ylabel:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=2, alpha=0.6)
        reference_label = ax.text(
            0.0,
            threshold + 1,
            "Thermal\nlimit",
            transform=ax.get_yaxis_transform(),
            color="red",
            ha="left",
            va="bottom",
            alpha=0.75,
            multialignment="center",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.0},
        )
    if threshold is not None and "Capacity" in ylabel:
        ax.axhline(threshold, color="blue", linestyle="--", linewidth=2, alpha=0.6)
        reference_label = ax.text(
            0.0,
            threshold - 1,
            "Security\nmargin",
            transform=ax.get_yaxis_transform(),
            color="blue",
            ha="left",
            va="top",
            alpha=0.75,
            multialignment="center",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.0},
        )
    style_axes(ax, labels, ylabel, rotation)
    resolve_mean_note_collisions(
        fig,
        ax,
        mean_notes,
        reference_threshold=threshold,
        reference_label=reference_label,
    )
    save_figure(fig, output)


def grouped_year_boxplot(
    frame: pd.DataFrame,
    cases: list[dict],
    year_colors: list[str],
    output: Path,
) -> list[BoxGroup]:
    years = sorted(int(year) for year in frame.future_year.unique())
    positions = np.arange(len(years), dtype=float)
    n_cases = len(cases)
    width = 0.22 if n_cases <= 3 else 0.18
    offsets = (np.arange(n_cases) - (n_cases - 1) / 2) * width
    hatches = ["..", "xx", "///", "+++"][:n_cases]
    fig, ax = plt.subplots(figsize=(10, 4))
    groups: list[BoxGroup] = []
    upper = 0.0

    for case_index, case in enumerate(cases):
        column = f"{case['case_id']}__load_shedding_percent"
        observations = []
        case_positions = []
        for year_index, year in enumerate(years):
            values = frame.loc[frame.future_year == year, column].to_numpy(dtype=float)
            observations.append(values)
            case_positions.append(positions[year_index] + offsets[case_index])
            upper = max(upper, float(np.nanmax(values)))
            groups.append(
                BoxGroup(
                    figure=-1,
                    panel=output.stem,
                    category=str(year),
                    subgroup=case["label"],
                    observation_unit="heatwave scenario",
                    values=values,
                )
            )
        box = ax.boxplot(
            observations,
            positions=case_positions,
            widths=width * 0.82,
            patch_artist=True,
            whis=1.5,
            showfliers=False,
            medianprops={"color": "dimgray", "linewidth": 1.2},
            whiskerprops={"color": "dimgray", "linewidth": 1.0},
            capprops={"color": "dimgray", "linewidth": 1.0},
        )
        for year_index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(year_colors[year_index])
            patch.set_edgecolor((0, 0, 0, 0.55))
            patch.set_linewidth(0.5)
            patch.set_hatch(hatches[case_index])
            patch.set_alpha(0.9)
        for position, values in zip(case_positions, observations, strict=True):
            add_mean_marker(ax, float(position), values, annotate=False)

    handles = [
        mpatches.Patch(
            facecolor="lightgray",
            hatch=hatches[index],
            edgecolor="black",
            linewidth=0.4,
            label=case["label"],
        )
        for index, case in enumerate(cases)
    ]
    legend = ax.legend(
        handles=handles,
        ncol=n_cases,
        edgecolor="white",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )
    legend.get_frame().set_alpha(0.99)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    style_axes(ax, [str(year) for year in years], "Load Shedding (%)", 0)
    ax.set_ylim(0, max(upper * 1.06, 0.01))
    save_figure(fig, output)
    return groups


def summary_rows(groups: Iterable[BoxGroup]) -> list[dict]:
    rows = []
    for group in groups:
        values = np.asarray(group.values, dtype=float)
        values = values[np.isfinite(values)]
        lower, upper = tukey_whiskers(values)
        rows.append(
            {
                "supplementary_figure": group.figure,
                "candidate_panel": group.panel,
                "category": group.category,
                "subgroup": group.subgroup,
                "observation_unit": group.observation_unit,
                "n": len(values),
                "mean": float(values.mean()),
                "sd_ddof_0": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "q1": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q3": float(np.percentile(values, 75)),
                "maximum": float(values.max()),
                "lower_whisker_1_5_iqr": lower,
                "upper_whisker_1_5_iqr": upper,
                "n_outliers": int(np.sum((values < lower) | (values > upper))),
            }
        )
    return rows


def national_candidates(source: Path, output: Path) -> list[BoxGroup]:
    data = source / "Supplementary_Fig_09_national_comparison" / "data"
    if not data.is_dir():
        return []
    metadata = json.loads((data / "plot_metadata.json").read_text())
    scenarios = pd.read_csv(data / "scenario_observations.csv")
    order = metadata["country_order"]
    labels = [metadata["country_names"][country] for country in order]
    colors = [metadata["country_colors"][country] for country in order]
    metrics = (
        ("air_temperature_c", "Air Temperature (°C)", "air_temperature_boxplot_candidate.pdf"),
        ("hourly_load_gw", "Hourly Load (GW)", "hourly_load_boxplot_candidate.pdf"),
        ("load_shedding_percent", "Load Shedding (%)", "load_shedding_boxplot_candidate.pdf"),
        ("runtime_s", "Running Time (sec.)", "runtime_boxplot_candidate.pdf"),
    )
    groups = []
    for column, ylabel, filename in metrics:
        observations = [
            scenarios.loc[scenarios.country_code == country, column].to_numpy(dtype=float)
            for country in order
        ]
        boxplot_only(
            observations,
            labels,
            colors,
            ylabel,
            output / "Supplementary_Fig_09_national_comparison" / filename,
            rotation=30,
            figsize=HALF_WIDTH_FIGSIZE,
            y_axis_mode=(
                "visible_statistics"
                if column == "load_shedding_percent"
                else "full_observation_range"
            ),
            mean_annotation_placement=(
                "lower-left" if column == "air_temperature_c" else "auto"
            ),
        )
        groups.extend(
            BoxGroup(9, filename, label, "", "heatwave scenario", values)
            for label, values in zip(labels, observations, strict=True)
        )
    return groups


def national_box_violin_candidates(source: Path, output: Path) -> list[BoxGroup]:
    data = source / "Supplementary_Fig_09_national_comparison" / "data"
    if not data.is_dir():
        return []
    metadata = json.loads((data / "plot_metadata.json").read_text())
    branches = pd.read_csv(data / "branch_observations.csv")
    order = metadata["country_order"]
    labels = [metadata["country_names"][country] for country in order]
    colors = [metadata["country_colors"][country] for country in order]
    target = output / "Supplementary_Fig_09_national_comparison"
    specifications = (
        (
            "line_temperature_c",
            "Line Temperature (°C)",
            "line_temperature_box_violin_candidate.pdf",
            float(metadata["thermal_limit_c"]),
            "above",
        ),
        (
            "available_capacity_percent_of_nominal",
            "Available Capacity (%)",
            "available_capacity_box_violin_candidate.pdf",
            float(metadata["capacity_security_margin_percent"]),
            "below",
        ),
    )
    groups: list[BoxGroup] = []
    for column, ylabel, filename, threshold, direction in specifications:
        observations = [
            branches.loc[branches.country_code == country, column].to_numpy(dtype=float)
            for country in order
        ]
        box_violin_plot(
            observations,
            labels,
            colors,
            ylabel,
            target / filename,
            threshold=threshold,
            exceed_direction=direction,
            figsize=FULL_WIDTH_FIGSIZE,
            mean_annotation_placement=(
                "lower-left"
                if column == "available_capacity_percent_of_nominal"
                else "auto"
            ),
        )
        groups.extend(
            BoxGroup(9, filename, label, "", "scenario-line observation", values)
            for label, values in zip(labels, observations, strict=True)
        )
    return groups


def opf_box_violin_candidates(source: Path, output: Path) -> list[BoxGroup]:
    groups: list[BoxGroup] = []
    for figure, slug in (
        (10, "spain_opf_comparison"),
        (13, "italy_opf_comparison"),
        (16, "france_opf_comparison"),
    ):
        data = source / f"Supplementary_Fig_{figure:02d}_{slug}" / "data"
        if not data.is_dir():
            continue
        metadata = json.loads((data / "plot_metadata.json").read_text())
        scenarios = pd.read_csv(data / "scenario_observations.csv")
        branches = pd.read_csv(data / "branch_observations.csv")
        methods = metadata["methods"]
        labels = [metadata["method_labels"][method] for method in methods]
        colors = [metadata["method_colors"][method] for method in methods]
        target = output / f"Supplementary_Fig_{figure:02d}_{slug}"
        specifications = (
            (
                [branches[f"{method}__available_capacity_percent_of_nominal"].to_numpy(dtype=float) for method in methods],
                "Available Capacity (%)",
                "model_available_capacity_box_violin_candidate.pdf",
                float(metadata["capacity_security_margin_percent"]),
                "below",
                "scenario-line observation",
            ),
            (
                [branches[f"{method}__line_temperature_c"].to_numpy(dtype=float) for method in methods],
                "Line Temperature (°C)",
                "model_line_temperature_box_violin_candidate.pdf",
                float(metadata["thermal_limit_c"]),
                "above",
                "scenario-line observation",
            ),
            (
                [scenarios[f"{method}__load_shedding_percent"].to_numpy(dtype=float) for method in methods],
                "Load Shedding (%)",
                "model_load_shedding_box_violin_candidate.pdf",
                None,
                "above",
                "heatwave scenario",
            ),
        )
        for observations, ylabel, filename, threshold, direction, unit in specifications:
            box_violin_plot(
                observations,
                labels,
                colors,
                ylabel,
                target / filename,
                threshold=threshold,
                exceed_direction=direction,
            )
            groups.extend(
                BoxGroup(figure, filename, label, "", unit, values)
                for label, values in zip(labels, observations, strict=True)
            )
    return groups


def operating_box_violin_candidates(source: Path, output: Path) -> list[BoxGroup]:
    groups: list[BoxGroup] = []
    for figure, slug in (
        (12, "spain_operating_sensitivity"),
        (15, "italy_operating_sensitivity"),
        (18, "france_operating_sensitivity"),
    ):
        data = source / f"Supplementary_Fig_{figure:02d}_{slug}" / "data"
        if not data.is_dir():
            continue
        metadata = json.loads((data / "plot_metadata.json").read_text())
        branches = pd.read_csv(data / "ablation_branch_observations.csv")
        scenarios = pd.read_csv(data / "ablation_scenario_observations.csv")
        thermal = pd.read_csv(data / "thermal_sensitivity_observations.csv")
        methods = metadata["ablation_methods"]
        labels = [metadata["ablation_labels"][method] for method in methods]
        colors = [metadata["ablation_colors"][method] for method in methods]
        target = output / f"Supplementary_Fig_{figure:02d}_{slug}"
        specifications = (
            (
                [branches[f"{method}__available_capacity_percent_of_nominal"].to_numpy(dtype=float) for method in methods],
                labels,
                colors,
                "Available Capacity (%)",
                "model_available_capacity_box_violin_candidate.pdf",
                float(metadata["capacity_security_margin_percent"]),
                "below",
                "scenario-line observation",
            ),
            (
                [branches[f"{method}__line_temperature_c"].to_numpy(dtype=float) for method in methods],
                labels,
                colors,
                "Line Temperature (°C)",
                "model_line_temperature_box_violin_candidate.pdf",
                float(metadata["base_thermal_limit_c"]),
                "above",
                "scenario-line observation",
            ),
            (
                [scenarios[f"{method}__load_shedding_percent"].to_numpy(dtype=float) for method in methods],
                labels,
                colors,
                "Load Shedding (%)",
                "model_load_shedding_box_violin_candidate.pdf",
                None,
                "above",
                "heatwave scenario",
            ),
            (
                [thermal[f"{case['case_id']}__load_shedding_percent"].to_numpy(dtype=float) for case in metadata["thermal_cases"]],
                [case["label"] for case in metadata["thermal_cases"]],
                metadata["sequential_red"],
                "Load Shedding (%)",
                "thermal_load_shedding_box_violin_candidate.pdf",
                None,
                "above",
                "heatwave scenario",
            ),
        )
        for observations, plot_labels, plot_colors, ylabel, filename, threshold, direction, unit in specifications:
            box_violin_plot(
                observations,
                plot_labels,
                plot_colors,
                ylabel,
                target / filename,
                threshold=threshold,
                exceed_direction=direction,
                rotation=15,
            )
            groups.extend(
                BoxGroup(figure, filename, label, "", unit, values)
                for label, values in zip(plot_labels, observations, strict=True)
            )
    return groups


def opf_runtime_candidates(source: Path, output: Path) -> list[BoxGroup]:
    groups = []
    for figure, slug in (
        (10, "spain_opf_comparison"),
        (13, "italy_opf_comparison"),
        (16, "france_opf_comparison"),
    ):
        data = source / f"Supplementary_Fig_{figure:02d}_{slug}" / "data"
        if not data.is_dir():
            continue
        metadata = json.loads((data / "plot_metadata.json").read_text())
        scenarios = pd.read_csv(data / "scenario_observations.csv")
        methods = metadata["methods"]
        labels = [metadata["method_labels"][method] for method in methods]
        colors = [metadata["method_colors"][method] for method in methods]
        observations = [
            scenarios[f"{method}__runtime_s"].to_numpy(dtype=float) for method in methods
        ]
        filename = "model_runtime_boxplot_candidate.pdf"
        boxplot_only(
            observations,
            labels,
            colors,
            "Running Time (sec.)",
            output / f"Supplementary_Fig_{figure:02d}_{slug}" / filename,
            figsize=HALF_WIDTH_FIGSIZE,
        )
        groups.extend(
            BoxGroup(figure, filename, label, "", "heatwave scenario", values)
            for label, values in zip(labels, observations, strict=True)
        )
    return groups


def operating_sensitivity_candidates(source: Path, output: Path) -> list[BoxGroup]:
    groups = []
    for figure, slug in (
        (12, "spain_operating_sensitivity"),
        (15, "italy_operating_sensitivity"),
        (18, "france_operating_sensitivity"),
    ):
        data = source / f"Supplementary_Fig_{figure:02d}_{slug}" / "data"
        if not data.is_dir():
            continue
        metadata = json.loads((data / "plot_metadata.json").read_text())
        load = pd.read_csv(data / "load_growth_observations.csv")
        storage = pd.read_csv(data / "storage_soc_observations.csv")
        target = output / f"Supplementary_Fig_{figure:02d}_{slug}"
        load_groups = grouped_year_boxplot(
            load,
            metadata["load_growth_cases"],
            metadata["sequential_orange"],
            target / "load_growth_boxplot_candidate.pdf",
        )
        storage_groups = grouped_year_boxplot(
            storage,
            metadata["storage_cases"],
            metadata["sequential_blue"],
            target / "storage_soc_boxplot_candidate.pdf",
        )
        groups.extend(
            BoxGroup(figure, g.panel, g.category, g.subgroup, g.observation_unit, g.values)
            for g in load_groups + storage_groups
        )
    return groups


def spatiotemporal_box_violin_candidates(source: Path, output: Path) -> list[BoxGroup]:
    groups: list[BoxGroup] = []
    for figure, slug in (
        (11, "spain_spatiotemporal_stress"),
        (14, "italy_spatiotemporal_stress"),
        (17, "france_spatiotemporal_stress"),
    ):
        data = source / f"Supplementary_Fig_{figure:02d}_{slug}" / "data"
        if not data.is_dir():
            continue
        metadata = json.loads((data / "plot_metadata.json").read_text())
        scenarios = pd.read_csv(data / "scenario_observations.csv")
        branches = pd.read_csv(data / "scenario_branch_observations.csv")
        hours = metadata["future_hours"]
        years = metadata["future_years"]
        hour_branch_groups = [
            branches[branches.scenario_id.isin(scenarios.loc[scenarios.future_hour == hour, "scenario_id"])]
            for hour in hours
        ]
        year_branch_groups = [
            branches[branches.scenario_id.isin(scenarios.loc[scenarios.future_year == year, "scenario_id"])]
            for year in years
        ]
        hour_scenario_groups = [scenarios[scenarios.future_hour == hour] for hour in hours]
        year_scenario_groups = [scenarios[scenarios.future_year == year] for year in years]
        target = output / f"Supplementary_Fig_{figure:02d}_{slug}"
        specifications = (
            (
                [group.available_capacity_percent_of_nominal.to_numpy(dtype=float) for group in hour_branch_groups],
                metadata["hour_labels"],
                list(reversed(metadata["sequential_blue"]))[: len(hours)],
                "Available Capacity (%)",
                "hour_available_capacity_box_violin_candidate.pdf",
                float(metadata["capacity_security_margin_percent"]),
                "below",
                "scenario-line observation",
            ),
            (
                [group.available_capacity_percent_of_nominal.to_numpy(dtype=float) for group in year_branch_groups],
                [str(year) for year in years],
                metadata["sequential_blue"],
                "Available Capacity (%)",
                "year_available_capacity_box_violin_candidate.pdf",
                float(metadata["capacity_security_margin_percent"]),
                "below",
                "scenario-line observation",
            ),
            (
                [group.line_temperature_c.to_numpy(dtype=float) for group in hour_branch_groups],
                metadata["hour_labels"],
                list(reversed(metadata["sequential_red"]))[: len(hours)],
                "Line Temperature (°C)",
                "hour_line_temperature_box_violin_candidate.pdf",
                float(metadata["thermal_limit_c"]),
                "above",
                "scenario-line observation",
            ),
            (
                [group.line_temperature_c.to_numpy(dtype=float) for group in year_branch_groups],
                [str(year) for year in years],
                metadata["sequential_red"],
                "Line Temperature (°C)",
                "year_line_temperature_box_violin_candidate.pdf",
                float(metadata["thermal_limit_c"]),
                "above",
                "scenario-line observation",
            ),
            (
                [group.load_shedding_percent.to_numpy(dtype=float) for group in hour_scenario_groups],
                metadata["hour_labels"],
                list(reversed(metadata["sequential_orange"]))[: len(hours)],
                "Load Shedding (%)",
                "hour_load_shedding_box_violin_candidate.pdf",
                None,
                "above",
                "heatwave scenario",
            ),
            (
                [group.load_shedding_percent.to_numpy(dtype=float) for group in year_scenario_groups],
                [str(year) for year in years],
                metadata["sequential_orange"],
                "Load Shedding (%)",
                "year_load_shedding_box_violin_candidate.pdf",
                None,
                "above",
                "heatwave scenario",
            ),
        )
        for observations, plot_labels, plot_colors, ylabel, filename, threshold, direction, unit in specifications:
            box_violin_plot(
                observations,
                plot_labels,
                plot_colors,
                ylabel,
                target / filename,
                threshold=threshold,
                exceed_direction=direction,
            )
            groups.extend(
                BoxGroup(figure, filename, label, "", unit, values)
                for label, values in zip(plot_labels, observations, strict=True)
            )
    return groups


def cross_border_box_violin_candidates(source: Path, output: Path) -> list[BoxGroup]:
    groups: list[BoxGroup] = []
    for figure, slug in (
        (19, "spain_cross_border"),
        (20, "france_cross_border"),
    ):
        data = source / f"Supplementary_Fig_{figure:02d}_{slug}" / "data" / "statistics"
        if not data.is_dir():
            continue
        metadata = json.loads((data / "plot_metadata.json").read_text())
        scenarios = pd.read_csv(data / "scenario_observations.csv")
        temperatures = pd.read_csv(data / "line_temperature_observations.csv")
        target = output / f"Supplementary_Fig_{figure:02d}_{slug}"
        for load_growth in metadata["load_growths"]:
            scenario_subset = scenarios[np.isclose(scenarios.load_growth, load_growth)]
            temperature_subset = temperatures[np.isclose(temperatures.load_growth, load_growth)]
            ids = metadata["configuration_ids"]
            labels = metadata["configuration_labels"]
            colors = metadata["colors"]
            suffix = f"{load_growth}_{metadata['storage_state']}"
            specifications = (
                (
                    [scenario_subset.loc[scenario_subset.configuration_id == cid, "load_shedding_percent"].to_numpy(dtype=float) for cid in ids],
                    "Load Shedding (%)",
                    f"load_shedding_box_violin_candidate_{suffix}.pdf",
                    None,
                    "above",
                    "reference-country heatwave scenario",
                ),
                (
                    [temperature_subset.loc[temperature_subset.configuration_id == cid, "line_temperature_c"].to_numpy(dtype=float) for cid in ids],
                    "Line Temperature (°C)",
                    f"line_temperature_box_violin_candidate_{suffix}.pdf",
                    float(metadata["thermal_limit_c"]),
                    "above",
                    "scenario-line observation on a reference-country internal AC line",
                ),
            )
            for observations, ylabel, filename, threshold, direction, unit in specifications:
                box_violin_plot(
                    observations,
                    labels,
                    colors,
                    ylabel,
                    target / filename,
                    threshold=threshold,
                    exceed_direction=direction,
                    rotation=15,
                )
                groups.extend(
                    BoxGroup(figure, filename, label, suffix, unit, values)
                    for label, values in zip(labels, observations, strict=True)
                )
    return groups


def ieee_runtime_candidate(source: Path, output: Path) -> list[BoxGroup]:
    data = source / "Supplementary_Fig_21_ieee_thermal_sensitivity" / "data"
    if not data.is_dir():
        return []
    metadata = json.loads((data / "plot_metadata.json").read_text())
    runtime = pd.read_csv(data / "runtime_observations.csv")
    methods = metadata["methods"]
    labels = [metadata["method_labels"][method] for method in methods]
    colors = [metadata["method_colors"][method] for method in methods]
    observations = [
        runtime.loc[runtime.method == method, "runtime_s"].to_numpy(dtype=float)
        for method in methods
    ]
    filename = "runtime_boxplot_candidate.pdf"
    boxplot_only(
        observations,
        labels,
        colors,
        "Running Time (s)",
        output / "Supplementary_Fig_21_ieee_thermal_sensitivity" / filename,
        rotation=30,
        figsize=FULL_WIDTH_TALL_FIGSIZE,
    )
    return [
        BoxGroup(21, filename, label, "", "solver run across specified cases", values)
        for label, values in zip(labels, observations, strict=True)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_source_hashes(source: Path, path: Path) -> None:
    used_dirs = (
        "Supplementary_Fig_09_national_comparison",
        "Supplementary_Fig_10_spain_opf_comparison",
        "Supplementary_Fig_11_spain_spatiotemporal_stress",
        "Supplementary_Fig_12_spain_operating_sensitivity",
        "Supplementary_Fig_13_italy_opf_comparison",
        "Supplementary_Fig_14_italy_spatiotemporal_stress",
        "Supplementary_Fig_15_italy_operating_sensitivity",
        "Supplementary_Fig_16_france_opf_comparison",
        "Supplementary_Fig_17_france_spatiotemporal_stress",
        "Supplementary_Fig_18_france_operating_sensitivity",
        "Supplementary_Fig_19_spain_cross_border",
        "Supplementary_Fig_20_france_cross_border",
        "Supplementary_Fig_21_ieee_thermal_sensitivity",
        "Supplementary_Fig_22_ieee_weather_sensitivity",
    )
    rows = []
    for directory in used_dirs:
        for file_path in sorted((source / directory).rglob("*")):
            if file_path.is_file():
                rows.append(
                    {
                        "path_relative_to_source_root": str(file_path.relative_to(source)),
                        "bytes": file_path.stat().st_size,
                        "sha256": sha256(file_path),
                    }
                )
    write_csv(path, rows)


def write_register(path: Path) -> None:
    rows = [
        {
            "figure": "9",
            "source_panels": "four bars plus two box-violin distributions",
            "candidate": "four boxplots plus two revised box-violin panels",
            "decision": "VALID",
            "reason": "scenario and scenario-line observations are available",
        },
        {
            "figure": "10, 13, 16",
            "source_panels": "runtime bars plus three box-violin distributions per country",
            "candidate": "runtime boxplot plus revised box-violin panels",
            "decision": "VALID",
            "reason": "the same 480 scenarios and their scenario-line observations are available",
        },
        {
            "figure": "12, 15, 18",
            "source_panels": "two grouped bars plus four box-violin distributions per country",
            "candidate": "grouped boxplots plus revised box-violin panels",
            "decision": "VALID",
            "reason": "scenario and scenario-line observations are available for every setting",
        },
        {
            "figure": "11, 14, 17",
            "source_panels": "six spatiotemporal box-violin distributions per country",
            "candidate": "six revised box-violin panels per country",
            "decision": "VALID",
            "reason": "hourly and annual scenario/scenario-line populations are available",
        },
        {
            "figure": "19, 20",
            "source_panels": "four cross-border box-violin distributions per country",
            "candidate": "four revised box-violin panels per country",
            "decision": "VALID",
            "reason": "approved reference-country scenario and internal-line populations are available",
        },
        {
            "figure": "21",
            "source_panels": "IEEE runtime bar",
            "candidate": "runtime boxplot-only panel",
            "decision": "VALID",
            "reason": "12 specified solver runs per method (3 load ratios x 4 weather settings)",
        },
        {
            "figure": "21, 22",
            "source_panels": "IEEE load shedding by method and weather case",
            "candidate": "none; retain for separate point-plot decision",
            "decision": "NOT A BOXPLOT POPULATION",
            "reason": "each method-weather-load cell contains one deterministic value",
        },
    ]
    write_csv(path, rows)


def write_layout_register(path: Path) -> None:
    rows = [
        {
            "figure": "9",
            "candidate_panels": "air temperature; hourly load; load shedding; runtime",
            "layout_class": "half-width",
            "source_subfigure_width": "0.49 linewidth",
        },
        {
            "figure": "9",
            "candidate_panels": "line temperature; available capacity",
            "layout_class": "full-width",
            "source_subfigure_width": "0.99 linewidth",
        },
        {
            "figure": "10, 13, 16",
            "candidate_panels": "OPF distributions and runtime",
            "layout_class": "half-width",
            "source_subfigure_width": "0.48-0.49 linewidth",
        },
        {
            "figure": "11, 14, 17",
            "candidate_panels": "hourly and annual stress distributions",
            "layout_class": "half-width",
            "source_subfigure_width": "0.48-0.49 linewidth",
        },
        {
            "figure": "12, 15, 18",
            "candidate_panels": "ablation and operating-sensitivity distributions",
            "layout_class": "half-width",
            "source_subfigure_width": "0.48-0.49 linewidth",
        },
        {
            "figure": "19, 20",
            "candidate_panels": "cross-border distributions",
            "layout_class": "half-width",
            "source_subfigure_width": "0.49 textwidth",
        },
        {
            "figure": "21",
            "candidate_panels": "IEEE runtime",
            "layout_class": "full-width",
            "source_subfigure_width": "0.98 textwidth",
        },
    ]
    write_csv(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    configure_matplotlib()

    artwork = output / "artwork"
    reports = output / "reports"
    snapshot = output / "baseline_snapshot"
    groups: list[BoxGroup] = []
    groups.extend(national_candidates(source, artwork))
    groups.extend(national_box_violin_candidates(source, artwork))
    groups.extend(opf_runtime_candidates(source, artwork))
    groups.extend(opf_box_violin_candidates(source, artwork))
    groups.extend(operating_sensitivity_candidates(source, artwork))
    groups.extend(operating_box_violin_candidates(source, artwork))
    groups.extend(spatiotemporal_box_violin_candidates(source, artwork))
    groups.extend(cross_border_box_violin_candidates(source, artwork))
    groups.extend(ieee_runtime_candidate(source, artwork))
    write_csv(reports / "distribution_plot_statistics.csv", summary_rows(groups))
    write_register(reports / "distribution_plot_conversion_register.csv")
    write_layout_register(reports / "distribution_plot_layout_register.csv")
    write_source_hashes(source, snapshot / "relevant_source_files_sha256.csv")
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "baseline_summary.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source),
                "policy": "read-only baseline; no manuscript, SI artwork, or release data modified",
                "candidate_pdf_count": len(list(artwork.rglob("*.pdf"))),
                "candidate_png_count": len(list((output / "previews").rglob("*.png"))),
                "distribution_plot_definition": {
                    "centre": "median",
                    "median_line": "dimgray, 1.2 pt",
                    "box": "25th to 75th percentiles",
                    "whiskers": "most extreme observations within 1.5 times IQR",
                    "fliers": "not displayed",
                    "mean": "dark-red point",
                    "average_annotation": "shown for box-violin and single-column boxplots; omitted for grouped boxplots",
                    "average_annotation_placement": "upper-left of the mean unless the rendered two-line note exceeds the axes upper bound or intersects a reference line or label, then lower-left",
                    "thermal_limit_label": "two lines at the upper-left of the reference line",
                    "security_margin_label": "two lines at the lower-left of the reference line",
                    "national_panel_overrides": "air-temperature and available-capacity mean notes are always lower-left",
                    "density": "complete population shown only in box-violin panels",
                },
                "layout_classes": {
                    "half_width_figsize_inches": list(HALF_WIDTH_FIGSIZE),
                    "full_width_figsize_inches": list(FULL_WIDTH_FIGSIZE),
                    "full_width_tall_figsize_inches": list(FULL_WIDTH_TALL_FIGSIZE),
                    "assignment_record": "reports/distribution_plot_layout_register.csv",
                },
                "panel_specific_axis_rules": {
                    "Supplementary Fig. 9 load shedding": (
                        "upper limit covers displayed Tukey whiskers, mean points and "
                        "mean annotations; hidden fliers remain in the statistics report"
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_pdfs": len(list(artwork.rglob("*.pdf"))),
                "summary_groups": len(groups),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

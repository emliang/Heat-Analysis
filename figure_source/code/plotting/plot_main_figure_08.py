#!/usr/bin/env python3
"""Draw the complete 180-mm main Fig. 8 from its figure-level Source Data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 50.0
MM_PER_INCH = 25.4
PALETTE = {
    "weekday_observed": "#fabeaf",
    "weekend_observed": "#b7c6e7",
    "weekday_model": "#e83947",
    "weekend_model": "#72a9d0",
}
DERATING_COLOURS = {
    "ocgt": "#1f77b4",
    "ccgt": "#ff7f0e",
    "nuclear": "#2ca02c",
    "copper_windings": "#d62728",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_axis(ax, *, grid: bool) -> None:
    ax.tick_params(top=False, right=False, direction="out", pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#555555")
    if grid:
        ax.grid(True, linewidth=0.35, alpha=0.22, color="#aaaaaa")
        ax.set_axisbelow(True)


def draw_panel_a(ax, source_dir: Path) -> None:
    observations = pd.read_csv(source_dir / "Fig8a_demand_observations.csv")
    curves = pd.read_csv(source_dir / "Fig8a_fitted_demand_curves.csv")
    summary = pd.read_csv(source_dir / "Fig8a_demand_validation_summary.csv").iloc[0]
    weekday = observations["day_type"] == "weekday"
    weekend = ~weekday
    ax.scatter(
        observations.loc[weekday, "population_weighted_bait_c"],
        observations.loc[weekday, "observed_daily_demand_gw"],
        s=3.2,
        marker="o",
        color=PALETTE["weekday_observed"],
        linewidth=0,
        alpha=0.78,
        rasterized=False,
    )
    ax.scatter(
        observations.loc[weekend, "population_weighted_bait_c"],
        observations.loc[weekend, "observed_daily_demand_gw"],
        s=3.2,
        marker="s",
        color=PALETTE["weekend_observed"],
        linewidth=0,
        alpha=0.78,
        rasterized=False,
    )
    for day_type, colour in (
        ("weekday", PALETTE["weekday_model"]),
        ("weekend_or_holiday", PALETTE["weekend_model"]),
    ):
        curve = curves.loc[curves["day_type"] == day_type]
        ax.plot(
            curve["population_weighted_bait_c"],
            curve["fitted_daily_demand_gw"],
            color=colour,
            linewidth=1.35,
        )
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=PALETTE["weekday_observed"], markeredgewidth=0, markersize=3.7),
        Line2D([], [], marker="s", color="none", markerfacecolor=PALETTE["weekend_observed"], markeredgewidth=0, markersize=3.7),
        Line2D([], [], color=PALETTE["weekday_model"], linewidth=1.25),
        Line2D([], [], color=PALETTE["weekend_model"], linewidth=1.25),
    ]
    ax.legend(
        handles,
        ["Weekday demand", "Weekend demand", "Weekday model", "Weekend model"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.25,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=0.45,
        labelspacing=0.18,
        fontsize=5.5,
    )
    ax.text(
        0.035,
        0.045,
        f"RMSE = {float(summary['rmse_gw']):.2f} GW\n"
        f"MAPE = {float(summary['mape_fraction']) * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=6.0,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax.set_xlabel("BAIT (°C)")
    ax.set_ylabel("Daily demand (GW)")
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_ylim(17.1, 38.3)
    ax.set_yticks([20, 23, 26, 29, 32, 35, 38])
    style_axis(ax, grid=True)


def draw_panel_b(ax, source_dir: Path) -> None:
    curves = pd.read_csv(source_dir / "Fig8b_generator_derating_curves.csv")
    order = ["ocgt", "ccgt", "nuclear", "copper_windings"]
    for curve_id in order:
        curve = curves.loc[
            (curves["curve_id"] == curve_id)
            & (curves["displayed_in_figure"].astype(str).str.lower() == "true")
        ]
        label = str(curve["display_label"].iloc[0])
        ax.plot(
            curve["air_temperature_c"],
            curve["derating_factor"],
            color=DERATING_COLOURS[curve_id],
            linewidth=1.25,
            label=label,
        )
    ax.axhline(1.0, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Air temperature (°C)")
    ax.set_ylabel("Derating factor")
    ax.set_xlim(10, 50)
    ax.set_ylim(0.5, 1.2)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_yticks(np.arange(0.5, 1.21, 0.1))
    ax.legend(
        loc="lower left",
        frameon=True,
        borderpad=0.35,
        handlelength=1.5,
        labelspacing=0.26,
        fontsize=6.0,
    )
    style_axis(ax, grid=True)


def load_panel_c_inputs(source_dir: Path) -> tuple[dict, dict, dict, dict]:
    row = read_rows(source_dir / "Fig8c_conductor_model_parameters.csv")[0]
    conductor = {
        "diameter": float(row["diameter_m"]),
        "num_bundle": int(row["number_of_subconductors"]),
        "ref_temperature": float(row["reference_temperature_c"]),
        "max_temperature": float(row["thermal_limit_c"]),
        "resistance_ratio": float(row["resistance_temperature_coefficient_per_c"]),
        "unit_resistance": float(row["unit_resistance_ohm_per_m"]),
        "conductor_angle": float(row["conductor_angle_deg"]),
        "elevation": float(row["elevation_m"]),
        "inom": float(row["nominal_current_per_subconductor_a"]),
    }
    if float(row["convective_correction"]) != 1.0:
        conductor["convective_correction"] = float(row["convective_correction"])
    if float(row["radiative_correction"]) != 1.0:
        conductor["radiactive_correction"] = float(row["radiative_correction"])
    weather_values = {
        row["parameter"]: float(row["computation_value"])
        for row in read_rows(source_dir / "Fig8c_weather_parameters.csv")
    }
    weather = {
        "wind_speed": weather_values["wind_speed"],
        "wind_angle": np.array([weather_values["wind_angle"]]),
        "air_density": weather_values["air_density"],
        "air_viscosity": weather_values["air_viscosity"],
        "air_conductivity": weather_values["air_conductivity"],
        "air_temperature": 30.0,
        "radiation_emissivity": weather_values["radiation_emissivity"],
        "solar_absorptivity": weather_values["solar_absorptivity"],
        "solar_heat_intensity": weather_values["solar_heat_intensity"],
        "wind_height": weather_values["wind_height"],
    }
    rules = {
        row["axis"]: row
        for row in read_rows(source_dir / "Fig8c_grid_generation_rules.csv")
    }
    ambient_rule = rules["ambient_temperature"]
    current_rule = rules["current_per_subconductor"]
    ambient_grid = {
        "start_c": float(ambient_rule["start"]),
        "stop_exclusive_c": float(ambient_rule["stop_exclusive"]),
        "step_c": float(ambient_rule["step"]),
    }
    current_grid = {
        "start_nominal_multiplier": float(current_rule["start"]),
        "stop_nominal_multiplier_exclusive": float(current_rule["stop_exclusive"]),
        "step_a_per_subconductor": float(current_rule["step"]),
    }
    return conductor, weather, ambient_grid, current_grid


def draw_panel_c(ax, source_dir: Path, project_root: Path, fig, colorbar_ax) -> None:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conductor_heat_flow_definition import evaluate_grid  # noqa: PLC0415
    from utils.heat_flow_utils import heat_banlance_equation  # noqa: PLC0415

    conductor, weather, ambient_grid, current_grid = load_panel_c_inputs(source_dir)
    ambient, total_current, temperature = evaluate_grid(
        heat_banlance_equation,
        conductor,
        ambient_grid=ambient_grid,
        current_grid=current_grid,
        weather_parameters=weather,
    )
    x_grid, y_grid = np.meshgrid(ambient, total_current)
    filled = ax.contourf(
        x_grid,
        y_grid,
        temperature,
        14,
        cmap="coolwarm",
        alpha=0.95,
    )
    contour = pd.read_csv(source_dir / "Fig8c_thermal_limit_curve.csv")
    thermal_line, = ax.plot(
        contour["air_temperature_c"],
        contour["maximum_current_at_thermal_limit_a"],
        color="#e31a1c",
        linewidth=1.35,
        label="Thermal limit (90°C)",
    )
    nominal_current = conductor["inom"] * conductor["num_bundle"]
    nominal_line = ax.axhline(
        nominal_current,
        color="#2536d8",
        linewidth=1.35,
        label=f"Nominal current ({nominal_current:g} A)",
    )
    ax.set_xlabel("Air temperature (°C)")
    ax.set_ylabel("Conductor current (A)")
    ax.set_xlim(25, 50.5)
    ax.set_ylim(float(total_current.min()), float(total_current.max()))
    ax.set_xticks([25, 30, 35, 40, 45, 50])
    ax.legend(
        handles=[thermal_line, nominal_line],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.3,
        handlelength=1.35,
        labelspacing=0.22,
        fontsize=6.0,
    )
    style_axis(ax, grid=False)
    colorbar = fig.colorbar(filled, cax=colorbar_ax)
    colorbar.set_label("Conductor temperature (°C)", fontsize=6.0, fontweight="bold", labelpad=2.5)
    colorbar.ax.tick_params(labelsize=6.0, width=0.5, length=2.2, pad=1.2)
    colorbar.outline.set_linewidth(0.55)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    configure_fonts()
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    # Balance complete panel footprints. Panel c has a colourbar and therefore
    # uses a narrower plotting axis so that axis + colourbar reads at the same
    # visual width as panels a and b.
    panel_positions = (
        (0.055, 0.24, 0.245, 0.68),
        (0.360, 0.24, 0.245, 0.68),
        (0.665, 0.24, 0.215, 0.68),
    )
    axes = [fig.add_axes(position) for position in panel_positions]
    colorbar_ax = fig.add_axes((0.889, 0.24, 0.0105, 0.68))
    draw_panel_a(axes[0], args.source_dir)
    draw_panel_b(axes[1], args.source_dir)
    draw_panel_c(
        axes[2],
        args.source_dir,
        args.project_root.resolve(),
        fig,
        colorbar_ax,
    )
    panel_heights = [ax.get_position().height for ax in axes]
    if max(panel_heights) - min(panel_heights) > 1e-12:
        raise RuntimeError(f"Unequal panel heights: {panel_heights}")
    if axes[2].get_position().width >= axes[0].get_position().width:
        raise RuntimeError("Panel c must reserve internal width for its colourbar")
    for label, ax in zip(("a", "b", "c"), axes, strict=True):
        ax.text(
            -0.18,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="top",
        )

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Figure 8 - Heatwave effects on demand, generation and transmission",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_main_figure_08.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(args.pdf, format="pdf", dpi=300, metadata=metadata)
    fig.savefig(
        args.png,
        format="png",
        dpi=300,
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

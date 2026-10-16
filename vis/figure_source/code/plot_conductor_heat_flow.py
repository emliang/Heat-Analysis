#!/usr/bin/env python3
"""Reproduce Supplementary Fig. 4 from its deterministic rule definition."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


PANEL_FILENAMES = {
    "individual": "panel_a_individual.pdf",
    "corrected": "panel_b_corrected.pdf",
    "merged": "panel_c_merged.pdf",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_compact_inputs(source_dir: Path) -> tuple[list[dict], dict, dict, dict]:
    """Reconstruct deterministic model inputs from the compact CSV package."""
    model_rows = read_rows(source_dir / "conductor_heat_flow_model_parameters.csv")
    weather_rows = read_rows(
        source_dir / "conductor_heat_flow_weather_parameters.csv"
    )
    rule_rows = read_rows(source_dir / "conductor_heat_flow_generation_rules.csv")

    models = []
    for row in model_rows:
        conductor = {
            "diameter": float(row["diameter_m"]),
            "num_bundle": int(row["number_of_subconductors"]),
            "ref_temperature": float(row["reference_temperature_c"]),
            "max_temperature": float(row["thermal_limit_c"]),
            "resistance_ratio": float(
                row["resistance_temperature_coefficient_per_c"]
            ),
            "unit_resistance": float(row["unit_resistance_ohm_per_m"]),
            "conductor_angle": float(row["conductor_angle_deg"]),
            "elevation": float(row["elevation_m"]),
            "inom": float(row["nominal_current_per_subconductor_a"]),
        }
        if float(row["convective_correction"]) != 1.0:
            conductor["convective_correction"] = float(
                row["convective_correction"]
            )
        if float(row["radiative_correction"]) != 1.0:
            conductor["radiactive_correction"] = float(
                row["radiative_correction"]
            )
        models.append(
            {
                "panel": row["panel"],
                "model_id": row["model_id"],
                "display_label": row["model_label"],
                "conductor": conductor,
            }
        )

    weather_values = {
        row["parameter"]: float(row["computation_value"])
        for row in weather_rows
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
    rules = {row["axis"]: row for row in rule_rows}
    ambient_rule = rules["ambient_temperature"]
    current_rule = rules["current_per_subconductor"]
    ambient_grid = {
        "start_c": float(ambient_rule["start"]),
        "stop_exclusive_c": float(ambient_rule["stop_exclusive"]),
        "step_c": float(ambient_rule["step"]),
    }
    current_grid = {
        "start_nominal_multiplier": float(current_rule["start"]),
        "stop_nominal_multiplier_exclusive": float(
            current_rule["stop_exclusive"]
        ),
        "step_a_per_subconductor": float(current_rule["step"]),
    }
    return models, weather, ambient_grid, current_grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artwork-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conductor_heat_flow_definition import (  # noqa: PLC0415
        evaluate_grid,
    )
    from utils.heat_flow_utils import heat_banlance_equation  # noqa: PLC0415

    models, weather, ambient_grid, current_grid = load_compact_inputs(
        args.source_dir
    )
    models_by_id = {spec["model_id"]: spec for spec in models}
    if set(models_by_id) != set(PANEL_FILENAMES):
        raise ValueError("Compact model table does not match the three panels")
    args.artwork_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    sns.set_style("white")

    metadata = {
        "Title": "Conductor thermal-model comparison",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_conductor_heat_flow.py",
        "CreationDate": None,
        "ModDate": None,
    }

    for model_id, filename in PANEL_FILENAMES.items():
        conductor = dict(models_by_id[model_id]["conductor"])
        ambient, current, temperature = evaluate_grid(
            heat_banlance_equation,
            conductor,
            ambient_grid=ambient_grid,
            current_grid=current_grid,
            weather_parameters=weather,
        )
        x_grid, y_grid = np.meshgrid(ambient, current)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.6))
        # The current SI artwork retains two historical panel-specific choices:
        # panel b uses 14 filled-contour levels, and panel c uses smaller text.
        contour_levels = 14 if model_id == "corrected" else 15
        axis_label_fontsize = 16 if model_id == "merged" else 18
        tick_fontsize = 10 if model_id == "merged" else 16
        colorbar_label_fontsize = 14 if model_id == "merged" else 16
        filled = ax.contourf(
            x_grid,
            y_grid,
            temperature,
            contour_levels,
            cmap="coolwarm",
            alpha=0.95,
        )
        ax.axhline(
            y=float(conductor["inom"] * conductor["num_bundle"]),
            color="blue",
            linewidth=2.5,
        )
        ax.contour(
            x_grid,
            y_grid,
            temperature,
            levels=[90.0],
            colors="red",
            linewidths=2.5,
        )
        ax.set_xlabel(
            "Air Temperature (°C)",
            fontsize=axis_label_fontsize,
            fontweight="bold",
        )
        ax.set_ylabel(
            "Conductor Current (A)",
            fontsize=axis_label_fontsize,
            fontweight="bold",
        )
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        colorbar = fig.colorbar(filled, ax=ax)
        colorbar.set_label(
            "Conductor Temperature (°C)",
            fontsize=colorbar_label_fontsize,
            fontweight="bold",
        )
        fig.tight_layout()
        pdf_path = args.artwork_dir / filename
        png_path = args.preview_dir / filename.replace(".pdf", ".png")
        fig.savefig(
            pdf_path,
            format="pdf",
            dpi=500,
            bbox_inches="tight",
            metadata=metadata,
        )
        fig.savefig(
            png_path,
            format="png",
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 0.1))
    reference_conductor = models_by_id["corrected"]["conductor"]
    thermal_limit = reference_conductor["max_temperature"]
    nominal_total_current = (
        reference_conductor["inom"] * reference_conductor["num_bundle"]
    )
    legend = ax.legend(
        handles=[
            mlines.Line2D(
                [], [], color="red", linewidth=3,
                label=f"Thermal limit ({thermal_limit:g}°C)"
            ),
            mlines.Line2D(
                [], [], color="blue", linewidth=3,
                label=f"Nominal current ({nominal_total_current:g}A)"
            ),
        ],
        loc="center",
        ncol=2,
        columnspacing=5.0,
        fontsize=16,
        frameon=False,
        fancybox=False,
        handlelength=3.0,
        handletextpad=2,
    )
    for text in legend.texts:
        text.set_fontweight("bold")
    ax.axis("off")
    fig.savefig(
        args.artwork_dir / "legend.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        metadata=metadata,
    )
    fig.savefig(
        args.preview_dir / "legend.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

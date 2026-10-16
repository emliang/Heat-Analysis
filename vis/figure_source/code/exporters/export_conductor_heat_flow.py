#!/usr/bin/env python3
"""Export compact deterministic Supplementary Data for Supplementary Fig. 4.

The figure is generated from an explicit heat-balance rule and compact model,
weather and grid parameters. Dense rule-generated matrices are evaluated only
in memory and are not stored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-panel", type=Path, action="append", required=True)
    parser.add_argument("--original-legend", type=Path, required=True)
    args = parser.parse_args()

    if len(args.original_panel) != 3:
        raise ValueError("Exactly three --original-panel paths are required")

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from utils.heat_flow_utils import heat_banlance_equation  # noqa: PLC0415
    from conductor_heat_flow_definition import (  # noqa: PLC0415
        AMBIENT_GRID,
        CURRENT_GRID,
        MODEL_SPECS,
        WEATHER,
        evaluate_grid,
    )

    model_rows: list[dict] = []
    summary_rows: list[dict] = []

    for spec in MODEL_SPECS:
        conductor = dict(spec["conductor"])
        ambient_temperatures, total_currents, temperature_grid = evaluate_grid(
            heat_banlance_equation,
            conductor,
        )

        model_rows.append(
            {
                "panel": spec["panel"],
                "model_id": spec["model_id"],
                "model_label": spec["display_label"],
                "diameter_m": format_float(conductor["diameter"]),
                "number_of_subconductors": conductor["num_bundle"],
                "reference_temperature_c": format_float(
                    conductor["ref_temperature"]
                ),
                "thermal_limit_c": format_float(conductor["max_temperature"]),
                "resistance_temperature_coefficient_per_c": format_float(
                    conductor["resistance_ratio"]
                ),
                "unit_resistance_ohm_per_m": format_float(
                    conductor["unit_resistance"]
                ),
                "conductor_angle_deg": format_float(conductor["conductor_angle"]),
                "elevation_m": format_float(conductor["elevation"]),
                "nominal_current_per_subconductor_a": format_float(
                    conductor["inom"]
                ),
                "nominal_total_current_a": format_float(
                    conductor["inom"] * conductor["num_bundle"]
                ),
                "convective_correction": format_float(
                    conductor.get("convective_correction", 1.0)
                ),
                "radiative_correction": format_float(
                    conductor.get("radiactive_correction", 1.0)
                ),
            }
        )
        summary_rows.append(
            {
                "panel": spec["panel"],
                "model_id": spec["model_id"],
                "evaluated_grid_points_not_stored": int(temperature_grid.size),
                "ambient_temperature_points": len(ambient_temperatures),
                "total_current_points": len(total_currents),
                "ambient_temperature_min_c": format_float(
                    ambient_temperatures.min()
                ),
                "ambient_temperature_max_c": format_float(
                    ambient_temperatures.max()
                ),
                "total_current_min_a": format_float(total_currents.min()),
                "total_current_max_a": format_float(total_currents.max()),
                "conductor_temperature_min_c": format_float(
                    temperature_grid.min()
                ),
                "conductor_temperature_max_c": format_float(
                    temperature_grid.max()
                ),
            }
        )

    weather_rows = [
        {
            "parameter": "wind_speed",
            "computation_value": "0.61",
            "manuscript_display_value": "0.6",
            "unit": "m/s",
            "note": "The caption reports the computation value rounded to one decimal place.",
        },
        {
            "parameter": "wind_angle",
            "computation_value": "90",
            "manuscript_display_value": "",
            "unit": "degree",
            "note": "Wind direction relative to the conductor.",
        },
        {
            "parameter": "solar_heat_intensity",
            "computation_value": "900",
            "manuscript_display_value": "900",
            "unit": "W/m^2",
            "note": "ENTSO-E worst-case ambient setting used in the figure.",
        },
        {
            "parameter": "air_density",
            "computation_value": "1.029",
            "manuscript_display_value": "",
            "unit": "kg/m^3",
            "note": "Recorded notebook input; the active implementation calculates density dynamically.",
        },
        {
            "parameter": "air_viscosity",
            "computation_value": "2.043e-5",
            "manuscript_display_value": "",
            "unit": "kg/(m s)",
            "note": "Recorded notebook input; the active implementation calculates viscosity dynamically.",
        },
        {
            "parameter": "air_conductivity",
            "computation_value": "0.02945",
            "manuscript_display_value": "",
            "unit": "W/(m C)",
            "note": "Recorded notebook input; the active implementation calculates conductivity dynamically.",
        },
        {
            "parameter": "radiation_emissivity",
            "computation_value": "0.8",
            "manuscript_display_value": "",
            "unit": "dimensionless",
            "note": "",
        },
        {
            "parameter": "solar_absorptivity",
            "computation_value": "0.8",
            "manuscript_display_value": "",
            "unit": "dimensionless",
            "note": "",
        },
        {
            "parameter": "wind_height",
            "computation_value": "50",
            "manuscript_display_value": "",
            "unit": "m",
            "note": "The input wind speed is already specified at this height.",
        },
    ]

    model_path = output_dir / "conductor_heat_flow_model_parameters.csv"
    weather_path = output_dir / "conductor_heat_flow_weather_parameters.csv"
    grid_path = output_dir / "conductor_heat_flow_generation_rules.csv"
    summary_path = output_dir / "conductor_heat_flow_validation_summary.csv"
    legacy_dense_path = output_dir / "conductor_heat_flow_source_data.csv"
    legacy_dense_path.unlink(missing_ok=True)
    write_csv(model_path, list(model_rows[0]), model_rows)
    write_csv(weather_path, list(weather_rows[0]), weather_rows)
    grid_rows = [
        {
            "axis": "ambient_temperature",
            "start": format_float(AMBIENT_GRID["start_c"]),
            "stop_exclusive": format_float(AMBIENT_GRID["stop_exclusive_c"]),
            "step": format_float(AMBIENT_GRID["step_c"]),
            "reference": "absolute",
            "unit": "C",
        },
        {
            "axis": "current_per_subconductor",
            "start": format_float(CURRENT_GRID["start_nominal_multiplier"]),
            "stop_exclusive": format_float(
                CURRENT_GRID["stop_nominal_multiplier_exclusive"]
            ),
            "step": format_float(CURRENT_GRID["step_a_per_subconductor"]),
            "reference": "start/stop are multipliers of nominal current; step is absolute",
            "unit": "multiplier; A",
        },
    ]
    write_csv(grid_path, list(grid_rows[0]), grid_rows)
    write_csv(summary_path, list(summary_rows[0]), summary_rows)

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Supplementary Fig. 4 - Conductor heat-flow Supplementary Data\n"
        "==============================================================\n\n"
        "This compact package records the conductor models, weather settings, "
        "grid-generation rules and validation summaries used for panels a-c. "
        "The dense temperature matrices are deterministic outputs of the "
        "heat-balance equation and are generated in memory rather than stored.\n\n"
        "The individual and corrected models evaluate current per "
        "sub-conductor and report total four-bundle current; the merged model "
        "evaluates the equivalent single conductor directly. The responsible "
        "code uses 0.61 m/s wind speed; the figure caption reports this as "
        "0.6 m/s after rounding to one decimal place.\n\n"
        "The exporter and plotter call only the active deterministic "
        "heat-balance function. They do not rerun any OPF or heatwave "
        "simulation.\n\n"
        "Reproduction commands (run from the HeatAnalysis project root)\n"
        "-------------------------------------------------------------\n"
        "conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/exporters/"
        "export_conductor_heat_flow.py --project-root . --output-dir "
        "nature_final_materials/publication_pipeline/supplementary_track/"
        "Supplementary_Figure_04_conductor_heat_flow/data "
        "--original-panel <panel-a-current_vs_air_temp.pdf> "
        "--original-panel <panel-b-current_vs_air_temp.pdf> "
        "--original-panel <panel-c-current_vs_air_temp.pdf> "
        "--original-legend <legend.pdf>\n\n"
        "conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/plotting/"
        "plot_conductor_heat_flow.py --project-root . --artwork-dir "
        "nature_final_materials/publication_pipeline/supplementary_track/"
        "Supplementary_Figure_04_conductor_heat_flow/artwork --source-dir "
        "nature_final_materials/publication_pipeline/supplementary_track/"
        "Supplementary_Figure_04_conductor_heat_flow/data --preview-dir "
        "nature_final_materials/publication_pipeline/supplementary_track/"
        "Supplementary_Figure_04_conductor_heat_flow/preview\n",
        encoding="utf-8",
    )

    original_assets = [
        *(path.resolve() for path in args.original_panel),
        args.original_legend.resolve(),
    ]
    responsible_sources = [
        project_root / "utils/heat_flow_utils.py",
        project_root / "scripts/4.test_solve_heat_balance.ipynb",
        Path(__file__).resolve().parents[1] / "conductor_heat_flow_definition.py",
        *original_assets,
    ]
    outputs = [model_path, weather_path, grid_path, summary_path, readme_path]
    manifest = {
        "semantic_figure_id": "conductor_heat_flow",
        "supplementary_figure": 4,
        "status": "COMPACT_RULE_PACKAGE_IMPLEMENTED_ORIGINAL_STYLE_REPRODUCTION",
        "simulation_rerun": False,
        "deterministic_physical_grid_evaluation": True,
        "dense_rule_generated_matrix_stored": False,
        "ambient_temperature_grid": {
            "minimum_c": AMBIENT_GRID["start_c"],
            "maximum_c": AMBIENT_GRID["stop_exclusive_c"] - AMBIENT_GRID["step_c"],
            "upper_bound_exclusive_c": AMBIENT_GRID["stop_exclusive_c"],
            "step_c": AMBIENT_GRID["step_c"],
            "points": int(
                (AMBIENT_GRID["stop_exclusive_c"] - AMBIENT_GRID["start_c"])
                / AMBIENT_GRID["step_c"]
            ),
        },
        "responsible_sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in responsible_sources
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "display_preservation": {
            "manuscript_asset_replaced": False,
            "visual_definition": "responsible notebook cells 10, 23, 25, 27 and 29",
            "corrected_panel_filled_contour_levels": 14,
            "other_panel_filled_contour_levels": 15,
            "merged_panel_retains_smaller_original_typography": True,
            "wind_speed_computation_m_s": 0.61,
            "wind_speed_caption_rounded_m_s": 0.6,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

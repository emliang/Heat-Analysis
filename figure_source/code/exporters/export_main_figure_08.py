#!/usr/bin/env python3
"""Build the compact figure-level Source Data package for main Fig. 8.

The exporter consumes the three verified Supplementary component packages. It
does not rerun demand calibration, OPF, heatwave generation, or any simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


FIGURE_ID = "heatwave_impact_models"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_float(value: float) -> str:
    if math.isnan(float(value)):
        return ""
    return f"{float(value):.15g}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def deterministic_zip(source_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path == destination:
                continue
            info = zipfile.ZipInfo(path.name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def export_panel_a(component_dir: Path, output_dir: Path) -> dict:
    observations = pd.read_csv(
        component_dir / "demand_calibration_plotted_daily_observations.csv"
    )
    parameters = pd.read_csv(
        component_dir / "demand_calibration_model_parameters.csv"
    )
    summaries = pd.read_csv(
        component_dir / "demand_calibration_validation_summary.csv"
    )
    observations = observations.loc[observations["country_code"] == "ES"].copy()
    parameters = parameters.loc[parameters["country_code"] == "ES"].copy()
    summaries = summaries.loc[summaries["country_code"] == "ES"].copy()
    if len(parameters) != 1 or len(summaries) != 1 or len(observations) != 3225:
        raise ValueError("Unexpected Spanish demand-calibration component shape")
    observations.sort_values("date", inplace=True)
    observations.to_csv(
        output_dir / "Fig8a_demand_observations.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )
    parameters.to_csv(
        output_dir / "Fig8a_demand_model_parameters.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )
    summaries.to_csv(
        output_dir / "Fig8a_demand_validation_summary.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )

    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    from demand_calibration_definition import model_curve  # noqa: PLC0415

    row = parameters.iloc[0]
    parameter_names = (
        "Ph", "Pc", "Th", "Tc", "solar_gains", "wind_chill",
        "humidity_discomfort", "smoothing", "Pb", "alpha", "lower_blend",
        "upper_blend", "max_raw_var",
    )
    values = {name: float(row[name]) for name in parameter_names}
    bait = observations["population_weighted_bait_c"].to_numpy(dtype=float)
    bait_grid = np.linspace(float(bait.min()), float(bait.max()), 1000)
    curve_rows: list[dict] = []
    for day_type, weekday in (("weekday", True), ("weekend_or_holiday", False)):
        demand = model_curve(bait_grid, values, weekday=weekday)
        curve_rows.extend(
            {
                "day_type": day_type,
                "population_weighted_bait_c": format_float(x),
                "fitted_daily_demand_gw": format_float(y),
            }
            for x, y in zip(bait_grid, demand, strict=True)
        )
    write_rows(output_dir / "Fig8a_fitted_demand_curves.csv", curve_rows)
    return {
        "observations": len(observations),
        "weekday_observations": int((observations["day_type"] == "weekday").sum()),
        "weekend_or_holiday_observations": int(
            (observations["day_type"] == "weekend_or_holiday").sum()
        ),
        "curve_rows": len(curve_rows),
    }


def export_panel_b(component_dir: Path, output_dir: Path) -> dict:
    definitions = read_rows(component_dir / "generator_derating_curve_definitions.csv")
    rules = read_rows(component_dir / "generator_derating_generation_rules.csv")
    if len(rules) != 1 or rules[0]["axis"] != "air_temperature":
        raise ValueError("Unexpected generator-derating grid definition")

    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    from generator_derating_definition import (  # noqa: PLC0415
        displayed_mask,
        formula_values,
    )

    rule = rules[0]
    temperature = np.linspace(
        float(rule["minimum"]),
        float(rule["maximum"]),
        int(rule["points"]),
    )
    values = formula_values(temperature)
    curve_rows: list[dict] = []
    for definition in definitions:
        curve_id = definition["curve_id"]
        mask = displayed_mask(curve_id, temperature)
        curve_rows.extend(
            {
                "curve_id": curve_id,
                "display_label": definition["display_label"],
                "air_temperature_c": format_float(temp),
                "derating_factor": format_float(factor),
                "displayed_in_figure": "true" if shown else "false",
            }
            for temp, factor, shown in zip(
                temperature, values[curve_id], mask, strict=True
            )
        )
    write_rows(output_dir / "Fig8b_generator_derating_curves.csv", curve_rows)
    shutil.copyfile(
        component_dir / "generator_derating_curve_definitions.csv",
        output_dir / "Fig8b_generator_derating_definitions.csv",
    )
    shutil.copyfile(
        component_dir / "generator_derating_generation_rules.csv",
        output_dir / "Fig8b_generator_derating_generation_rules.csv",
    )
    return {
        "curve_count": len(definitions),
        "temperature_points_per_curve": len(temperature),
        "curve_rows": len(curve_rows),
        "displayed_rows": sum(row["displayed_in_figure"] == "true" for row in curve_rows),
    }


def export_panel_c(project_root: Path, component_dir: Path, output_dir: Path) -> dict:
    code_root = Path(__file__).resolve().parents[1]
    plotting_root = code_root / "plotting"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(plotting_root))
    from conductor_heat_flow_definition import evaluate_grid  # noqa: PLC0415
    from plot_conductor_heat_flow import load_compact_inputs  # noqa: PLC0415
    from utils.heat_flow_utils import heat_banlance_equation  # noqa: PLC0415

    models, weather, ambient_grid, current_grid = load_compact_inputs(component_dir)
    corrected = [model for model in models if model["model_id"] == "corrected"]
    if len(corrected) != 1:
        raise ValueError("Expected exactly one corrected conductor model")
    conductor = corrected[0]["conductor"]
    ambient, total_current, conductor_temperature = evaluate_grid(
        heat_banlance_equation,
        conductor,
        ambient_grid=ambient_grid,
        current_grid=current_grid,
        weather_parameters=weather,
    )
    thermal_limit = float(conductor["max_temperature"])
    contour_rows: list[dict] = []
    for column, ambient_temperature in enumerate(ambient):
        profile = conductor_temperature[:, column]
        if not np.all(np.diff(profile) >= 0):
            raise ValueError("Conductor temperature must increase with current")
        limit_current = float(np.interp(thermal_limit, profile, total_current))
        contour_rows.append(
            {
                "air_temperature_c": format_float(ambient_temperature),
                "thermal_limit_c": format_float(thermal_limit),
                "maximum_current_at_thermal_limit_a": format_float(limit_current),
            }
        )
    write_rows(output_dir / "Fig8c_thermal_limit_curve.csv", contour_rows)

    model_rows = read_rows(component_dir / "conductor_heat_flow_model_parameters.csv")
    corrected_rows = [row for row in model_rows if row["model_id"] == "corrected"]
    write_rows(output_dir / "Fig8c_conductor_model_parameters.csv", corrected_rows)
    shutil.copyfile(
        component_dir / "conductor_heat_flow_weather_parameters.csv",
        output_dir / "Fig8c_weather_parameters.csv",
    )
    shutil.copyfile(
        component_dir / "conductor_heat_flow_generation_rules.csv",
        output_dir / "Fig8c_grid_generation_rules.csv",
    )
    validation_rows = [{
        "model_id": "corrected",
        "ambient_temperature_points": len(ambient),
        "current_points": len(total_current),
        "evaluated_grid_points_not_stored": int(conductor_temperature.size),
        "conductor_temperature_min_c": format_float(conductor_temperature.min()),
        "conductor_temperature_max_c": format_float(conductor_temperature.max()),
        "nominal_total_current_a": format_float(
            conductor["inom"] * conductor["num_bundle"]
        ),
        "thermal_limit_c": format_float(thermal_limit),
    }]
    write_rows(output_dir / "Fig8c_validation_summary.csv", validation_rows)
    return {
        "ambient_temperature_points": len(ambient),
        "current_points": len(total_current),
        "evaluated_grid_points_not_stored": int(conductor_temperature.size),
        "thermal_limit_curve_rows": len(contour_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()

    component_root = (
        project_root
        / "nature_final_materials/publication_pipeline/supplementary_track"
    )
    demand_dir = component_root / "Supplementary_Figure_02_demand_calibration/data"
    derating_dir = component_root / "Supplementary_Figure_03_generator_derating/data"
    conductor_dir = component_root / "Supplementary_Figure_04_conductor_heat_flow/data"
    component_files = sorted(
        [path for path in demand_dir.iterdir() if path.is_file()]
        + [path for path in derating_dir.iterdir() if path.is_file()]
        + [path for path in conductor_dir.iterdir() if path.is_file()]
    )

    panel_summary = {
        "a": export_panel_a(demand_dir, output_dir),
        "b": export_panel_b(derating_dir, output_dir),
        "c": export_panel_c(project_root, conductor_dir, output_dir),
    }
    readme = (
        "Source Data for Fig. 8 - Heatwave effects on demand, generation and transmission\n"
        "================================================================================\n\n"
        "Panel a contains one row per valid day in the Spanish demand-calibration\n"
        "record, the corresponding fitted values, the two displayed fitted curves,\n"
        "model parameters and validation statistics. Demand is a daily average.\n\n"
        "Panel b contains the numerical values of the four displayed generator\n"
        "derating curves and their formula definitions.\n\n"
        "Panel c records the corrected conductor model, ambient-weather settings,\n"
        "grid-generation rules and displayed 90 C thermal-limit contour. The dense\n"
        "deterministic conductor-temperature matrix is generated in memory from the\n"
        "heat-balance equation and is intentionally not duplicated in this package.\n\n"
        "No power-flow, heatwave or calibration simulation is run by this exporter.\n"
        "All data derive from the verified Supplementary component packages.\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    manifest = {
        "figure_id": FIGURE_ID,
        "figure_number": 8,
        "simulation_rerun": False,
        "source_component_packages": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in component_files
        ],
        "panel_summary": panel_summary,
        "package_policy": {
            "one_source_data_file_per_figure": True,
            "dense_rule_generated_conductor_matrix_stored": False,
            "raw_simulation_archive_bundled": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    deterministic_zip(output_dir, args.zip_path.resolve())
    print(json.dumps(panel_summary, indent=2, sort_keys=True))
    print(f"source_data_zip={args.zip_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

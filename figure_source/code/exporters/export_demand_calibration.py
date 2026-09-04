#!/usr/bin/env python3
"""Export the compact plotted observations and rules for Supplementary Fig. 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.15g}"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-artwork-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    artwork_root = args.current_artwork_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from demand_calibration_definition import (  # noqa: PLC0415
        COUNTRIES,
        PARAMETER_NAMES,
        mape,
        rmse,
    )
    from scripts.main_demand_calibration import (  # noqa: PLC0415
        load_and_filter_country_data,
        load_weather_and_network_data,
        prepare_training_data,
    )
    from utils.demand_utils import _bait  # noqa: PLC0415

    observation_rows: list[dict] = []
    parameter_rows: list[dict] = []
    summary_rows: list[dict] = []
    model_paths: list[Path] = []
    artwork_paths: list[Path] = []

    for country in COUNTRIES:
        code = country["country_code"]
        year_list = [country["start_year"], country["end_year"]]
        model_path = (
            project_root
            / "models/demand_curve"
            / code
            / f"{code}_{year_list}_demand_curve.npy"
        )
        artwork_path = (
            artwork_root
            / code
            / f"{code}_demand_curve_{country['start_year']}_{country['end_year']}.pdf"
        )
        model_paths.append(model_path)
        artwork_paths.append(artwork_path)

        filtered_load, load_dates, _original_load, abnormal_indices = (
            load_and_filter_country_data(code, year_list)
        )
        temp, wind, solar, humidity, weather_dates, pop_ratio = (
            load_weather_and_network_data(code, year_list)
        )
        (
            daily_load,
            _delta_year,
            temp,
            wind,
            solar,
            humidity,
            weekday_index,
            weekend_index,
            _hourly_load,
            valid_dates,
            _training_data,
        ) = prepare_training_data(
            code,
            filtered_load,
            load_dates,
            weather_dates,
            temp,
            wind,
            solar,
            humidity,
            pop_ratio,
        )
        parameters = np.load(model_path, allow_pickle=True).item()
        bait_regions = _bait(
            temp,
            wind,
            solar,
            humidity,
            parameters,
            valid_date=valid_dates,
        )
        bait_country = (bait_regions * pop_ratio).sum(-1)
        predicted = (
            parameters["Pb"]
            + parameters["Ph"]
            * (np.maximum(parameters["Th"] - bait_regions, 0.0) * pop_ratio).sum(-1)
            + parameters["Pc"]
            * (np.maximum(bait_regions - parameters["Tc"], 0.0) * pop_ratio).sum(-1)
            + parameters["alpha"] * weekday_index
        )

        for date, day_load, day_bait, day_predicted, is_weekday in zip(
            valid_dates,
            daily_load,
            bait_country,
            predicted,
            weekday_index,
            strict=True,
        ):
            observation_rows.append(
                {
                    "country_code": code,
                    "date": np.datetime_as_string(np.datetime64(date), unit="D"),
                    "day_type": "weekday" if int(is_weekday) else "weekend_or_holiday",
                    "population_weighted_bait_c": format_float(day_bait),
                    "observed_daily_demand_gw": format_float(day_load),
                    "model_predicted_daily_demand_gw": format_float(day_predicted),
                }
            )

        parameter_row = {
            "country_code": code,
            "country_name": country["country_name"],
            "start_year": country["start_year"],
            "end_year": country["end_year"],
            "x_ticks_c": ";".join(format_float(value) for value in country["x_ticks_c"]),
        }
        parameter_row.update(
            {name: format_float(parameters[name]) for name in PARAMETER_NAMES}
        )
        parameter_row["model_record_sha256"] = sha256(model_path)
        parameter_rows.append(parameter_row)

        summary_rows.append(
            {
                "country_code": code,
                "hourly_load_records_in_window": len(filtered_load),
                "lower_tail_values_replaced": len(abnormal_indices),
                "weather_days_in_window": len(weather_dates),
                "valid_days_plotted": len(daily_load),
                "weather_days_excluded": len(weather_dates) - len(daily_load),
                "weekday_points": int(weekday_index.sum()),
                "weekend_or_holiday_points": int(weekend_index.sum()),
                "bait_min_c": format_float(bait_country.min()),
                "bait_max_c": format_float(bait_country.max()),
                "observed_demand_min_gw": format_float(daily_load.min()),
                "observed_demand_max_gw": format_float(daily_load.max()),
                "rmse_gw": format_float(rmse(daily_load, predicted)),
                "mape_fraction": format_float(mape(daily_load, predicted)),
            }
        )

    observations_path = output_dir / "demand_calibration_plotted_daily_observations.csv"
    parameters_path = output_dir / "demand_calibration_model_parameters.csv"
    rules_path = output_dir / "demand_calibration_generation_rules.csv"
    summary_path = output_dir / "demand_calibration_validation_summary.csv"
    provenance_path = output_dir / "demand_calibration_input_provenance.csv"

    write_csv(observations_path, list(observation_rows[0]), observation_rows)
    write_csv(parameters_path, list(parameter_rows[0]), parameter_rows)
    rule_rows = [
        {
            "component": "observed_points",
            "rule": "one point per valid day after lower-tail hourly-load replacement",
            "stored_or_generated": "stored_compact_table",
        },
        {
            "component": "valid_day",
            "rule": "exactly 24 positive hourly load records and matching daily weather",
            "stored_or_generated": "generated_by_exporter",
        },
        {
            "component": "weekday_classification",
            "rule": "Monday-Friday excluding national public holidays",
            "stored_or_generated": "stored_in_observation_table",
        },
        {
            "component": "weekend_classification",
            "rule": "Saturday-Sunday or national public holiday",
            "stored_or_generated": "stored_in_observation_table",
        },
        {
            "component": "model_curve",
            "rule": "Pb + Ph*max(Th-BAIT,0) + Pc*max(BAIT-Tc,0) + alpha*weekday",
            "stored_or_generated": "generated_in_memory_from_parameters",
        },
        {
            "component": "model_curve_grid",
            "rule": "1000 linearly spaced BAIT points over each country's plotted BAIT range",
            "stored_or_generated": "generated_in_memory_not_stored",
        },
        {
            "component": "outlier_filter",
            "rule": "replace values below trimmed_mean - 3.5*trimmed_sample_std with nearest normal value",
            "stored_or_generated": "generated_by_exporter",
        },
    ]
    write_csv(rules_path, list(rule_rows[0]), rule_rows)
    write_csv(summary_path, list(summary_rows[0]), summary_rows)

    fixed_inputs = [
        ("historical_hourly_load", project_root / "data/entsoe/MHLV_2015-2024.parquet"),
        ("daily_era5_weather", project_root / "data/weather/era5/era5_daily_avg_2015_2024.nc"),
        ("clustered_network", project_root / "data/EU/networks/base_s_75_elec.nc"),
        ("onshore_regions", project_root / "data/EU/regions_onshore_base_s_75.geojson"),
        ("population_weights", project_root / "data/EU/load_ratio_base_s_75.csv"),
    ]
    provenance_rows = [
        {
            "role": role,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "bundled": "false",
        }
        for role, path in fixed_inputs
    ]
    provenance_rows.extend(
        {
            "role": f"calibrated_model_{path.parent.name}",
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "bundled": "false",
        }
        for path in model_paths
    )
    write_csv(provenance_path, list(provenance_rows[0]), provenance_rows)

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Supplementary Fig. 2 - Demand calibration compact data package\n"
        "================================================================\n\n"
        "demand_calibration_plotted_daily_observations.csv contains only the "
        "daily points used by the eight displayed country panels, together "
        "with each point's fitted prediction for metric verification. It does "
        "not duplicate the hourly ENTSO-E records, ERA5 grids or regional "
        "weather matrices.\n\n"
        "demand_calibration_model_parameters.csv records the 13 calibrated "
        "BAIT-demand parameters for each country. The displayed model curves "
        "are deterministic and are regenerated in memory from these parameters "
        "and the rule file; the 1000-point curve vectors are not stored.\n\n"
        "demand_calibration_input_provenance.csv records immutable upstream "
        "paths, hashes and sizes without bundling those large inputs. The "
        "exporter reads pre-calibrated model records and does not rerun SCEM "
        "calibration or any power-system simulation.\n\n"
        "Reproduction\n"
        "------------\n"
        "From the HeatAnalysis project root, regenerate the compact table "
        "from the frozen records with:\n"
        "  conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/exporters/"
        "export_demand_calibration.py --project-root . --output-dir "
        "<data-dir> --current-artwork-root <current-demand-artwork-root>\n\n"
        "Regenerate all eight panels and the shared legend using only this "
        "compact data directory with:\n"
        "  conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/plotting/"
        "plot_demand_calibration.py --source-dir <data-dir> --artwork-dir "
        "<artwork-dir> --preview-dir <preview-dir>\n",
        encoding="utf-8",
    )

    responsible_sources = [
        project_root / "scripts/main_demand_calibration.py",
        project_root / "utils/demand_utils.py",
        project_root / "utils/heatwave_utils.py",
        project_root / "utils/network_process_utils.py",
        Path(__file__).resolve().parents[1] / "demand_calibration_definition.py",
    ]
    outputs = [
        observations_path,
        parameters_path,
        rules_path,
        summary_path,
        provenance_path,
        readme_path,
    ]
    manifest = {
        "semantic_figure_id": "demand_calibration",
        "supplementary_figure": 2,
        "status": "COMPACT_PLOTTED_OBSERVATIONS_AND_RULE_PACKAGE_IMPLEMENTED",
        "simulation_rerun": False,
        "calibration_rerun": False,
        "raw_inputs_bundled": False,
        "generated_curve_points_stored": False,
        "countries": [country["country_code"] for country in COUNTRIES],
        "plotted_observation_rows": len(observation_rows),
        "environment": {
            name: package_version(name)
            for name in ("numpy", "pandas", "xarray", "geopandas", "pypsa", "holidays")
        },
        "responsible_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in responsible_sources
        ],
        "current_manuscript_artwork": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in artwork_paths
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"countries": len(COUNTRIES), "observations": len(observation_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

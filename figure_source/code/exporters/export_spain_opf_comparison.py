#!/usr/bin/env python3
"""Export a compact single-country OPF-comparison plotted-data package.

The exporter reads the frozen country model-analysis CSV and the 1,920 existing
per-scenario OPF result records.  It does not run a power-flow simulation.  The
branch table stores only the two values used by the figure, aligned into one
row per scenario and branch across the four compared methods.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_deterministic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n", float_format="%.12g")


def result_path(project: Path, row, method: str, settings) -> Path:
    future = str(row.fut_heatwave_date)
    historical = datetime.strptime(
        str(row.his_heatwave_date), "%Y-%m-%d"
    ).strftime("%Y-%m-%d %H:%M:%S")
    name = (
        f"{settings.COUNTRY_CODE}_{settings.N_BUSES}_{method}_{future}_{historical}"
        f"_storage_{settings.STORAGE_STATE}_load_growth_{settings.LOAD_GROWTH}"
        f"_thermal_{settings.THERMAL_LIMIT_C}_results.npy"
    )
    return project / "models" / settings.COUNTRY_CODE / "ppc" / future / name


def canonical_scenarios(table: pd.DataFrame, settings) -> pd.DataFrame:
    method_tables = {}
    success_sets = {}
    for method in settings.METHODS:
        selected = table[table["TDPF_solver"] == method].copy()
        if len(selected) != settings.N_SCENARIOS:
            raise ValueError(
                f"Expected {settings.N_SCENARIOS} rows for {method}; found {len(selected)}"
            )
        if selected.duplicated(list(settings.SCENARIO_KEY_COLUMNS)).any():
            raise ValueError(f"Duplicate scenario key for {method}")
        success = selected[selected["solver_status"] == 1]
        success_sets[method] = set(
            map(tuple, success[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
        )
        method_tables[method] = selected

    common = set.intersection(*(success_sets[method] for method in settings.METHODS))
    union = set.union(*(success_sets[method] for method in settings.METHODS))
    if len(common) != settings.N_SCENARIOS or common != union:
        counts = {method: len(success_sets[method]) for method in settings.METHODS}
        raise ValueError(f"Methods do not share the full success set: {counts}")

    base = method_tables[settings.METHODS[0]].copy()
    base["_scenario_key"] = list(
        map(tuple, base[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
    )
    base = base[base["_scenario_key"].isin(common)].sort_values(
        list(settings.SCENARIO_KEY_COLUMNS)
    )
    base = base.reset_index(drop=True)
    base["scenario_id"] = [
        f"{settings.COUNTRY_CODE}-S{index:04d}" for index in range(1, len(base) + 1)
    ]

    for method in settings.METHODS[1:]:
        comparison = method_tables[method].copy()
        comparison["_scenario_key"] = list(
            map(tuple, comparison[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
        )
        comparison = comparison.set_index("_scenario_key")
        for _, row in base.iterrows():
            other = comparison.loc[[row["_scenario_key"]]].iloc[0]
            for column in settings.SCENARIO_CONTEXT_COLUMNS:
                left = float(row[column])
                right = float(other[column])
                if not np.isclose(left, right, rtol=0, atol=1e-10):
                    raise ValueError(
                        f"Scenario context mismatch for {method}/{row['scenario_id']}/{column}: "
                        f"{left} != {right}"
                    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--country", choices=("ES", "IT", "FR"), default="ES")
    args = parser.parse_args()

    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    sys.path.insert(0, str(pipeline / "code"))
    from single_country_performance_definition import opf_settings  # noqa: PLC0415

    settings = opf_settings(args.country)

    package = (
        pipeline
        / "supplementary_track"
        / settings.PACKAGE_NAME
    )
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = project / "models" / settings.COUNTRY_CODE / settings.MODEL_ANALYSIS_CSV
    table = pd.read_csv(
        summary_path,
        dtype={"fut_heatwave_date": str, "his_heatwave_date": str},
    )
    scenarios = canonical_scenarios(table, settings)
    annual_rows = []
    for year, group in scenarios.groupby("fut_heatwave_year", sort=True):
        annual_rows.append(
            {
                "future_year": int(year),
                "n_scenarios": len(group),
                "wind_speed_mean_m_per_s": fmt(group.wind_speed.mean()),
                "wind_speed_sd_m_per_s_ddof_0": fmt(group.wind_speed.std(ddof=0)),
                "solar_irradiance_mean_w_per_m2": fmt(group.solar_radia.mean()),
                "solar_irradiance_sd_w_per_m2_ddof_0": fmt(group.solar_radia.std(ddof=0)),
                "air_temperature_mean_c": fmt(group.air_temp.mean()),
                "air_temperature_sd_c_ddof_0": fmt(group.air_temp.std(ddof=0)),
                "load_mean_gw": fmt(group.load.mean()),
                "load_sd_gw_ddof_0": fmt(group.load.std(ddof=0)),
            }
        )
    write_csv(data_dir / "annual_weather_load_summary.csv", list(annual_rows[0]), annual_rows)

    scenario_frame = scenarios[
        ["scenario_id", *settings.SCENARIO_KEY_COLUMNS, *settings.SCENARIO_CONTEXT_COLUMNS]
    ].copy()
    scenario_frame = scenario_frame.rename(
        columns={
            "fut_heatwave_date": "future_heatwave_datetime",
            "his_heatwave_date": "historical_heatwave_date",
            "fut_heatwave_year": "future_year",
            "fut_heatwave_month": "future_month",
            "fut_heatwave_day": "future_day",
            "fut_heatwave_hour": "future_hour",
            "air_temp": "air_temperature_c",
            "wind_speed": "wind_speed_m_per_s",
            "solar_radia": "solar_irradiance_w_per_m2",
            "load": "load_gw",
        }
    )

    branch_columns = {
        "scenario_id": np.repeat(scenarios.scenario_id.to_numpy(), settings.N_BRANCHES),
        "branch_index": np.tile(np.arange(settings.N_BRANCHES), settings.N_SCENARIOS),
    }
    provenance_rows = []

    for method in settings.METHODS:
        selected = table[table["TDPF_solver"] == method].copy()
        selected["_scenario_key"] = list(
            map(tuple, selected[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
        )
        selected = selected.set_index("_scenario_key")
        load_shedding = []
        runtime = []
        temperatures = []
        capacity_drops = []

        for scenario in scenarios.itertuples(index=False):
            key = tuple(str(getattr(scenario, column)) for column in settings.SCENARIO_KEY_COLUMNS)
            row = selected.loc[[key]].iloc[0]
            path = result_path(project, row, method, settings)
            if not path.exists():
                raise FileNotFoundError(path)
            result = np.load(path, allow_pickle=True).item()
            if int(result.get("solver_status", row.solver_status)) != 1:
                raise ValueError(f"Unsuccessful result record: {path}")
            conductor = np.asarray(result["con_temp"], dtype=float)
            capacity = np.asarray(result["capacity_drop"], dtype=float).reshape(-1)
            if conductor.shape[0] != settings.N_BRANCHES:
                raise ValueError(f"Unexpected con_temp shape {conductor.shape}: {path}")
            if capacity.shape != (settings.N_BRANCHES,):
                raise ValueError(f"Unexpected capacity_drop shape {capacity.shape}: {path}")
            line_temperature = conductor.max(axis=1)
            temperatures.append(line_temperature)
            capacity_drops.append(capacity * 100.0)
            load_shedding.append(float(row.node_load_shedding))
            runtime.append(float(row.run_time))
            provenance_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "method": method,
                    "record_path_relative_to_project": str(path.relative_to(project)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "solver_status": 1,
                    "con_temp_shape": "x".join(map(str, conductor.shape)),
                    "capacity_drop_shape": str(capacity.shape[0]),
                }
            )

        scenario_frame[f"{method}__load_shedding_percent"] = load_shedding
        scenario_frame[f"{method}__runtime_s"] = runtime
        branch_columns[f"{method}__line_temperature_c"] = np.concatenate(temperatures)
        branch_columns[f"{method}__available_capacity_percent_of_nominal"] = np.concatenate(
            capacity_drops
        )

    scenario_frame.to_csv(
        data_dir / "scenario_observations.csv",
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    branch_frame = pd.DataFrame(branch_columns)
    write_deterministic_csv_gz(data_dir / "branch_observations.csv.gz", branch_frame)
    write_csv(data_dir / "record_provenance.csv", list(provenance_rows[0]), provenance_rows)

    metadata = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "country_code": settings.COUNTRY_CODE,
        "n_buses": settings.N_BUSES,
        "n_branches": settings.N_BRANCHES,
        "n_segments": settings.N_SEGMENTS,
        "n_scenarios": settings.N_SCENARIOS,
        "methods": list(settings.METHODS),
        "method_labels": settings.METHOD_LABELS,
        "method_colors": settings.METHOD_COLORS,
        "storage_state": settings.STORAGE_STATE,
        "load_growth": settings.LOAD_GROWTH,
        "thermal_limit_c": settings.THERMAL_LIMIT_C,
        "capacity_security_margin_percent": settings.CAPACITY_SECURITY_MARGIN_PERCENT,
        "common_success_scenarios": settings.N_SCENARIOS,
        "annual_standard_deviation_ddof": 0,
        "runtime_standard_deviation_ddof": 0,
        "panel_units": {
            "a": "96 scenarios per future year",
            "b": "scenario-branch observations",
            "c": "scenario-branch observations",
            "d": "scenario observations",
            "e": "scenario observations",
            "f": "scenario observations for td_seg_derate_iter_2",
        },
        "capacity_source_definition": (
            "The exported available-capacity columns are the upstream capacity_drop "
            "field multiplied by 100; source-code review identifies that upstream "
            "field as Imax/i_nom."
        ),
        "simulation_rerun": False,
        "source_csv_relative_to_project": str(summary_path.relative_to(project)),
    }
    (data_dir / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    code_paths = [
        project / "vis/2.eur_single_analysis.ipynb",
        project / "utils/plot_utils.py",
        pipeline / "code/single_country_performance_definition.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "status": "COMPACT_PLOTTED_DATA_EXPORTED",
        "simulation_rerun": False,
        "counts": {
            "annual_rows": len(annual_rows),
            "scenario_rows": len(scenario_frame),
            "branch_rows": len(branch_frame),
            "upstream_result_records": len(provenance_rows),
        },
        "upstream_summary_csv": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": sha256(summary_path),
        },
        "responsible_code": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in code_paths
        ],
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(data_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"package": str(package), "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

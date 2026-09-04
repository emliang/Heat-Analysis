#!/usr/bin/env python3
"""Export compact plotted data for a single-country operating-sensitivity figure.

The exporter reads existing country summary CSVs and the 2,400 per-scenario
ablation result records used by panels a-b. It does not run a power-flow
simulation. Only the scenario values and branch metrics required to reproduce
the six panels are retained in the delivery package.
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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def write_deterministic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    float_format="%.12g",
                )


def write_dict_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analysis_csv_path(project: Path, settings, case: dict) -> Path:
    name = (
        f"{settings.COUNTRY_CODE}_{settings.N_BUSES}_bus"
        f"_renewable_True_heatwave_True_storage_True_{case['storage_state']}"
        f"_load_growth_True_{case['load_growth']}"
        f"_max_temp_{case['thermal_limit_c']}_{case['suffix']}.csv"
    )
    return project / "models" / settings.COUNTRY_CODE / name


def base_case(settings) -> dict:
    return {
        "storage_state": settings.BASE_STORAGE_STATE,
        "load_growth": settings.BASE_LOAD_GROWTH,
        "thermal_limit_c": settings.BASE_THERMAL_LIMIT_C,
        "suffix": "model_analysis",
    }


def load_table(path: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if path not in cache:
        if not path.exists():
            raise FileNotFoundError(path)
        cache[path] = pd.read_csv(
            path,
            dtype={"fut_heatwave_date": str, "his_heatwave_date": str},
        )
    return cache[path]


def select_successful_case(
    path: Path,
    solver: str,
    settings,
    cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    table = load_table(path, cache)
    selected = table[table["TDPF_solver"] == solver].copy()
    if len(selected) != settings.N_SCENARIOS:
        raise ValueError(
            f"Expected {settings.N_SCENARIOS} rows for {solver} in {path.name}; "
            f"found {len(selected)}"
        )
    if selected.duplicated(list(settings.SCENARIO_KEY_COLUMNS)).any():
        raise ValueError(f"Duplicate scenario key for {solver} in {path.name}")
    unsuccessful = selected[selected["solver_status"] != 1]
    if not unsuccessful.empty:
        raise ValueError(
            f"Found {len(unsuccessful)} unsuccessful rows for {solver} in {path.name}"
        )
    return selected.sort_values(list(settings.SCENARIO_KEY_COLUMNS)).reset_index(drop=True)


def scenario_keys(frame: pd.DataFrame, settings) -> set[tuple[str, str]]:
    return set(
        map(tuple, frame[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
    )


def indexed(frame: pd.DataFrame, settings) -> pd.DataFrame:
    result = frame.copy()
    result["_scenario_key"] = list(
        map(tuple, result[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
    )
    return result.set_index("_scenario_key", drop=False)


def canonical_scenarios(selections: dict[str, pd.DataFrame], settings) -> pd.DataFrame:
    success_sets = {name: scenario_keys(frame, settings) for name, frame in selections.items()}
    common = set.intersection(*success_sets.values())
    union = set.union(*success_sets.values())
    if len(common) != settings.N_SCENARIOS or common != union:
        raise ValueError(
            "Figure configurations do not share the full success set: "
            + json.dumps({name: len(keys) for name, keys in success_sets.items()})
        )

    reference_name = f"ablation::{settings.ABLATION_METHODS[0]}"
    reference = selections[reference_name].copy()
    reference["_scenario_key"] = list(
        map(tuple, reference[list(settings.SCENARIO_KEY_COLUMNS)].astype(str).to_numpy())
    )
    reference = reference[reference["_scenario_key"].isin(common)].sort_values(
        list(settings.SCENARIO_KEY_COLUMNS)
    )
    reference = reference.reset_index(drop=True)
    reference["scenario_id"] = [
        f"{settings.COUNTRY_CODE}-S{index:04d}" for index in range(1, len(reference) + 1)
    ]

    for name, frame in selections.items():
        comparison = indexed(frame, settings)
        for row in reference.itertuples(index=False):
            key = tuple(str(getattr(row, column)) for column in settings.SCENARIO_KEY_COLUMNS)
            other = comparison.loc[[key]].iloc[0]
            for column in settings.SCENARIO_CONTEXT_COLUMNS:
                left = float(getattr(row, column))
                right = float(other[column])
                if not np.isclose(left, right, rtol=0, atol=1e-10):
                    raise ValueError(
                        f"Scenario context mismatch for {name}/{row.scenario_id}/{column}: "
                        f"{left} != {right}"
                    )
    return reference


def scenario_base_frame(scenarios: pd.DataFrame, settings) -> pd.DataFrame:
    result = scenarios[
        ["scenario_id", *settings.SCENARIO_KEY_COLUMNS, *settings.SCENARIO_CONTEXT_COLUMNS]
    ].copy()
    return result.rename(
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
        }
    )


def align_values(
    scenarios: pd.DataFrame,
    frame: pd.DataFrame,
    settings,
    columns: tuple[str, ...],
) -> dict[str, list[float]]:
    source = indexed(frame, settings)
    output = {column: [] for column in columns}
    for row in scenarios.itertuples(index=False):
        key = tuple(str(getattr(row, column)) for column in settings.SCENARIO_KEY_COLUMNS)
        selected = source.loc[[key]].iloc[0]
        for column in columns:
            output[column].append(float(selected[column]))
    return output


def result_path(project: Path, row, method: str, settings) -> Path:
    future = str(row.fut_heatwave_date)
    historical = datetime.strptime(
        str(row.his_heatwave_date), "%Y-%m-%d"
    ).strftime("%Y-%m-%d %H:%M:%S")
    name = (
        f"{settings.COUNTRY_CODE}_{settings.N_BUSES}_{method}_{future}_{historical}"
        f"_storage_{settings.BASE_STORAGE_STATE}"
        f"_load_growth_{settings.BASE_LOAD_GROWTH}"
        f"_thermal_{settings.BASE_THERMAL_LIMIT_C}_results.npy"
    )
    return project / "models" / settings.COUNTRY_CODE / "ppc" / future / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--country", choices=("ES", "IT", "FR"), default="ES")
    args = parser.parse_args()

    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    sys.path.insert(0, str(pipeline / "code"))
    from single_country_performance_definition import operating_settings  # noqa: PLC0415

    settings = operating_settings(args.country)

    package = (
        pipeline
        / "supplementary_track"
        / settings.PACKAGE_NAME
    )
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, pd.DataFrame] = {}
    selections: dict[str, pd.DataFrame] = {}
    base_path = analysis_csv_path(project, settings, base_case(settings))

    for method in settings.ABLATION_METHODS:
        selections[f"ablation::{method}"] = select_successful_case(
            base_path, method, settings, cache
        )
    for group_name, cases in (
        ("thermal", settings.THERMAL_CASES),
        ("load_growth", settings.LOAD_GROWTH_CASES),
        ("storage", settings.STORAGE_CASES),
    ):
        for case in cases:
            path = analysis_csv_path(project, settings, case)
            selections[f"{group_name}::{case['case_id']}"] = select_successful_case(
                path, case["solver"], settings, cache
            )

    scenarios = canonical_scenarios(selections, settings)
    common = scenario_base_frame(scenarios, settings)

    ablation_scenarios = common.copy()
    for method in settings.ABLATION_METHODS:
        values = align_values(
            scenarios,
            selections[f"ablation::{method}"],
            settings,
            ("node_load_shedding",),
        )
        ablation_scenarios[f"{method}__load_shedding_percent"] = values[
            "node_load_shedding"
        ]
    write_csv(data_dir / "ablation_scenario_observations.csv", ablation_scenarios)

    thermal = common.copy()
    for case in settings.THERMAL_CASES:
        values = align_values(
            scenarios,
            selections[f"thermal::{case['case_id']}"],
            settings,
            ("node_load_shedding",),
        )
        thermal[f"{case['case_id']}__load_shedding_percent"] = values[
            "node_load_shedding"
        ]
    write_csv(data_dir / "thermal_sensitivity_observations.csv", thermal)

    load_growth = common.copy()
    for case in settings.LOAD_GROWTH_CASES:
        values = align_values(
            scenarios,
            selections[f"load_growth::{case['case_id']}"],
            settings,
            ("load", "node_load_shedding"),
        )
        load_growth[f"{case['case_id']}__load_gw"] = values["load"]
        load_growth[f"{case['case_id']}__load_shedding_percent"] = values[
            "node_load_shedding"
        ]
    write_csv(data_dir / "load_growth_observations.csv", load_growth)

    storage = common.copy()
    for case in settings.STORAGE_CASES:
        values = align_values(
            scenarios,
            selections[f"storage::{case['case_id']}"],
            settings,
            ("load", "node_load_shedding"),
        )
        storage[f"{case['case_id']}__load_gw"] = values["load"]
        storage[f"{case['case_id']}__load_shedding_percent"] = values[
            "node_load_shedding"
        ]
    write_csv(data_dir / "storage_soc_observations.csv", storage)

    branch_columns = {
        "scenario_id": np.repeat(scenarios.scenario_id.to_numpy(), settings.N_BRANCHES),
        "branch_index": np.tile(np.arange(settings.N_BRANCHES), settings.N_SCENARIOS),
    }
    provenance_rows = []
    for method in settings.ABLATION_METHODS:
        frame = indexed(selections[f"ablation::{method}"], settings)
        temperatures = []
        capacity_factors = []
        for scenario in scenarios.itertuples(index=False):
            key = tuple(
                str(getattr(scenario, column))
                for column in settings.SCENARIO_KEY_COLUMNS
            )
            row = frame.loc[[key]].iloc[0]
            path = result_path(project, row, method, settings)
            if not path.exists():
                raise FileNotFoundError(path)
            result = np.load(path, allow_pickle=True).item()
            conductor = np.asarray(result["con_temp"], dtype=float)
            capacity = np.asarray(result["capacity_drop"], dtype=float).reshape(-1)
            if conductor.shape != (settings.N_BRANCHES, settings.N_SEGMENTS):
                raise ValueError(f"Unexpected con_temp shape {conductor.shape}: {path}")
            if capacity.shape != (settings.N_BRANCHES,):
                raise ValueError(f"Unexpected capacity_drop shape {capacity.shape}: {path}")
            if int(result.get("solver_status", row.solver_status)) != 1:
                raise ValueError(f"Unsuccessful result record: {path}")
            temperatures.append(conductor.max(axis=1))
            capacity_factors.append(capacity * 100.0)
            provenance_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "method": method,
                    "record_path_relative_to_project": str(path.relative_to(project)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "solver_status": 1,
                    "con_temp_shape": f"{settings.N_BRANCHES}x{settings.N_SEGMENTS}",
                    "capacity_factor_shape": str(settings.N_BRANCHES),
                }
            )
        branch_columns[f"{method}__line_temperature_c"] = np.concatenate(temperatures)
        branch_columns[
            f"{method}__available_capacity_percent_of_nominal"
        ] = np.concatenate(capacity_factors)

    branches = pd.DataFrame(branch_columns)
    write_deterministic_csv_gz(data_dir / "ablation_branch_observations.csv.gz", branches)
    write_dict_rows(data_dir / "record_provenance.csv", provenance_rows)

    source_paths = sorted(cache)
    metadata = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "country_code": settings.COUNTRY_CODE,
        "n_buses": settings.N_BUSES,
        "n_branches": settings.N_BRANCHES,
        "n_segments": settings.N_SEGMENTS,
        "n_scenarios": settings.N_SCENARIOS,
        "scenarios_per_year": settings.SCENARIOS_PER_YEAR,
        "common_success_scenarios_across_all_configurations": settings.N_SCENARIOS,
        "base_storage_state": settings.BASE_STORAGE_STATE,
        "base_load_growth": settings.BASE_LOAD_GROWTH,
        "base_thermal_limit_c": settings.BASE_THERMAL_LIMIT_C,
        "capacity_security_margin_percent": settings.CAPACITY_SECURITY_MARGIN_PERCENT,
        "standard_deviation_ddof": settings.STANDARD_DEVIATION_DDOF,
        "ablation_methods": list(settings.ABLATION_METHODS),
        "ablation_labels": settings.ABLATION_LABELS,
        "ablation_colors": settings.ABLATION_COLORS,
        "thermal_cases": list(settings.THERMAL_CASES),
        "load_growth_cases": list(settings.LOAD_GROWTH_CASES),
        "storage_cases": list(settings.STORAGE_CASES),
        "sequential_orange": list(settings.SEQUENTIAL_ORANGE),
        "sequential_blue": list(settings.SEQUENTIAL_BLUE),
        "sequential_red": list(settings.SEQUENTIAL_RED),
        "panel_units": {
            "a": "scenario-branch observations",
            "b": "scenario-branch observations",
            "c": "scenario observations",
            "d": "scenario observations",
            "e": "annual mean and population standard deviation over 96 scenario observations",
            "f": "annual mean and population standard deviation over 96 scenario observations",
        },
        "capacity_metric": {
            "source_field": "capacity_drop",
            "stored_name": "available_capacity_percent_of_nominal",
            "definition": "minimum weather-dependent current limit divided by nominal current, multiplied by 100",
            "current_artwork_axis_label": "Line Capacity Drop (%)",
        },
        "simulation_rerun": False,
        "source_csvs_relative_to_project": [
            str(path.relative_to(project)) for path in source_paths
        ],
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
            "ablation_scenario_rows": len(ablation_scenarios),
            "ablation_branch_rows": len(branches),
            "thermal_sensitivity_rows": len(thermal),
            "load_growth_rows": len(load_growth),
            "storage_soc_rows": len(storage),
            "upstream_result_records": len(provenance_rows),
            "upstream_summary_csvs": len(source_paths),
        },
        "upstream_summary_csvs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
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

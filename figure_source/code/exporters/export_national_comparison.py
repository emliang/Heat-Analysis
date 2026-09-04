#!/usr/bin/env python3
"""Export compact plotted observations for Supplementary Fig. 15.

The exporter reads the frozen eight-country model-analysis CSV files and the
existing Iter-OPF result records. It does not run a power-flow simulation.
Only scenario values and the two branch values plotted in the figure are
retained; full result dictionaries remain referenced by path and hash.
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


def result_path(project: Path, country_code: str, row, settings) -> Path:
    future = str(row.fut_heatwave_date)
    historical = datetime.strptime(
        str(row.his_heatwave_date), "%Y-%m-%d"
    ).strftime("%Y-%m-%d %H:%M:%S")
    filename = (
        f"{country_code}_{settings.N_BUSES[country_code]}_{settings.METHOD}_"
        f"{future}_{historical}_storage_{settings.STORAGE_STATE}_"
        f"load_growth_{settings.LOAD_GROWTH}_thermal_90_results.npy"
    )
    return project / "models" / country_code / "ppc" / future / filename


def select_scenarios(table: pd.DataFrame, country_code: str, settings) -> pd.DataFrame:
    selected = table[table["TDPF_solver"] == settings.METHOD].copy()
    if len(selected) != settings.N_SCENARIOS_PER_COUNTRY:
        raise ValueError(
            f"{country_code}: expected {settings.N_SCENARIOS_PER_COUNTRY} rows for "
            f"{settings.METHOD}; found {len(selected)}"
        )
    if selected.duplicated(list(settings.SCENARIO_KEY_COLUMNS)).any():
        raise ValueError(f"{country_code}: duplicate scenario key")
    failed = selected[selected["solver_status"] != 1]
    if not failed.empty:
        raise ValueError(
            f"{country_code}: expected the frozen full success set; found "
            f"{len(failed)} unsuccessful rows"
        )
    selected = selected.sort_values(list(settings.SCENARIO_KEY_COLUMNS)).reset_index(drop=True)
    selected.insert(
        0,
        "scenario_id",
        [f"{country_code}-S{index:04d}" for index in range(1, len(selected) + 1)],
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    sys.path.insert(0, str(pipeline / "code"))
    import national_comparison_definition as settings  # noqa: PLC0415

    package = pipeline / "supplementary_track" / settings.PACKAGE_NAME
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scenario_frames: list[pd.DataFrame] = []
    branch_frames: list[pd.DataFrame] = []
    provenance_rows: list[dict] = []
    source_csvs: list[dict] = []

    for country_code in settings.COUNTRY_ORDER:
        summary_path = (
            project
            / "models"
            / country_code
            / settings.model_analysis_filename(country_code)
        )
        table = pd.read_csv(
            summary_path,
            dtype={"fut_heatwave_date": str, "his_heatwave_date": str},
        )
        scenarios = select_scenarios(table, country_code, settings)
        scenario_frame = scenarios[
            [
                "scenario_id",
                *settings.SCENARIO_KEY_COLUMNS,
                *settings.SCENARIO_CONTEXT_COLUMNS,
                "node_load_shedding",
                "run_time",
            ]
        ].copy()
        scenario_frame.insert(0, "country_name", settings.COUNTRY_NAMES[country_code])
        scenario_frame.insert(0, "country_code", country_code)
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
                "load": "hourly_load_gw",
                "node_load_shedding": "load_shedding_percent",
                "run_time": "runtime_s",
            }
        )
        scenario_frames.append(scenario_frame)

        temperature_rows: list[np.ndarray] = []
        capacity_rows: list[np.ndarray] = []
        for row in scenarios.itertuples(index=False):
            path = result_path(project, country_code, row, settings)
            if not path.exists():
                raise FileNotFoundError(path)
            result = np.load(path, allow_pickle=True).item()
            if int(result.get("solver_status", row.solver_status)) != 1:
                raise ValueError(f"Unsuccessful result record: {path}")
            conductor = np.asarray(result["con_temp"], dtype=float)
            capacity = np.asarray(result["capacity_drop"], dtype=float).reshape(-1)
            expected_shape = (
                settings.N_BRANCHES[country_code],
                settings.N_SEGMENTS[country_code],
            )
            if conductor.shape != expected_shape:
                raise ValueError(
                    f"{country_code}: unexpected con_temp shape {conductor.shape}; "
                    f"expected {expected_shape}: {path}"
                )
            if capacity.shape != (settings.N_BRANCHES[country_code],):
                raise ValueError(
                    f"{country_code}: unexpected capacity_drop shape {capacity.shape}: {path}"
                )
            temperature_rows.append(conductor.max(axis=1))
            capacity_rows.append(capacity * 100.0)
            provenance_rows.append(
                {
                    "country_code": country_code,
                    "scenario_id": row.scenario_id,
                    "method": settings.METHOD,
                    "record_path_relative_to_project": str(path.relative_to(project)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "solver_status": 1,
                    "con_temp_shape": "x".join(map(str, conductor.shape)),
                    "capacity_drop_shape": str(capacity.shape[0]),
                }
            )

        n_scenarios = len(scenarios)
        n_branches = settings.N_BRANCHES[country_code]
        branch_frames.append(
            pd.DataFrame(
                {
                    "country_code": country_code,
                    "scenario_id": np.repeat(scenarios.scenario_id.to_numpy(), n_branches),
                    "branch_index": np.tile(np.arange(n_branches), n_scenarios),
                    "line_temperature_c": np.concatenate(temperature_rows),
                    "available_capacity_percent_of_nominal": np.concatenate(capacity_rows),
                }
            )
        )
        source_csvs.append(
            {
                "country_code": country_code,
                "path": str(summary_path),
                "bytes": summary_path.stat().st_size,
                "sha256": sha256(summary_path),
            }
        )

    scenario_data = pd.concat(scenario_frames, ignore_index=True)
    branch_data = pd.concat(branch_frames, ignore_index=True)
    provenance = pd.DataFrame(provenance_rows)
    write_csv(data_dir / "scenario_observations.csv", scenario_data)
    write_deterministic_csv_gz(data_dir / "branch_observations.csv.gz", branch_data)
    write_csv(data_dir / "record_provenance.csv", provenance)

    metadata = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "country_order": list(settings.COUNTRY_ORDER),
        "country_names": settings.COUNTRY_NAMES,
        "country_colors": settings.COUNTRY_COLORS,
        "n_buses": settings.N_BUSES,
        "n_branches": settings.N_BRANCHES,
        "n_segments": settings.N_SEGMENTS,
        "n_scenarios_per_country": settings.N_SCENARIOS_PER_COUNTRY,
        "method": settings.METHOD,
        "storage_state": settings.STORAGE_STATE,
        "load_growth": settings.LOAD_GROWTH,
        "thermal_limit_c": settings.THERMAL_LIMIT_C,
        "capacity_security_margin_percent": settings.CAPACITY_SECURITY_MARGIN_PERCENT,
        "standard_deviation_ddof": 0,
        "panel_units": {
            "a": "country-specific scenario observations",
            "b": "country-specific scenario observations",
            "c": "country-specific scenario observations",
            "d": "country-specific scenario observations",
            "e": "country-specific scenario-branch observations",
            "f": "country-specific scenario-branch observations",
        },
        "cross_country_alignment": (
            "No common scenario-key intersection is imposed across countries; each "
            "country contributes its own 480 projected hourly heatwave scenarios."
        ),
        "capacity_source_definition": (
            "The exported available_capacity_percent_of_nominal is the upstream "
            "capacity_drop field multiplied by 100; source-code review identifies "
            "that upstream field as Imax/i_nom."
        ),
        "simulation_rerun": False,
    }
    (data_dir / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    code_paths = [
        project / "vis/3.eur_multi_analysis.ipynb",
        project / "utils/plot_utils.py",
        pipeline / "code/national_comparison_definition.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "status": "COMPACT_PLOTTED_DATA_EXPORTED",
        "simulation_rerun": False,
        "counts": {
            "countries": len(settings.COUNTRY_ORDER),
            "scenario_rows": len(scenario_data),
            "branch_rows": len(branch_data),
            "upstream_result_records": len(provenance),
        },
        "upstream_summary_csvs": source_csvs,
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

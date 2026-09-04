#!/usr/bin/env python3
"""Export compact plotted data for Supplementary Fig. 17.

The exporter reads the frozen Spain summary table, the 480 successful Iter-OPF
records, the represented PyPSA-Eur network and four weather snapshots used by
the current figure. It does not run a power-flow or weather simulation.
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
import xarray as xr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
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


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analysis_csv(project: Path, settings) -> Path:
    return project / "models" / settings.COUNTRY_CODE / (
        f"{settings.COUNTRY_CODE}_{settings.N_BUSES}_bus"
        f"_renewable_True_heatwave_True_storage_True_{settings.STORAGE_STATE}"
        f"_load_growth_True_{settings.LOAD_GROWTH}"
        f"_max_temp_{settings.THERMAL_LIMIT_C}_model_analysis.csv"
    )


def result_path(project: Path, row, settings) -> Path:
    future = str(row.fut_heatwave_date)
    historical = datetime.strptime(str(row.his_heatwave_date), "%Y-%m-%d").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    name = (
        f"{settings.COUNTRY_CODE}_{settings.N_BUSES}_{settings.SOLVER}_{future}_{historical}"
        f"_storage_{settings.STORAGE_STATE}_load_growth_{settings.LOAD_GROWTH}"
        f"_thermal_{settings.THERMAL_LIMIT_C}_results.npy"
    )
    return project / "models" / settings.COUNTRY_CODE / "ppc" / future / name


def select_scenarios(path: Path, settings) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        dtype={"fut_heatwave_date": str, "his_heatwave_date": str},
    )
    selected = table[table["TDPF_solver"] == settings.SOLVER].copy()
    if len(selected) != settings.N_SCENARIOS:
        raise ValueError(f"Expected {settings.N_SCENARIOS} rows, found {len(selected)}")
    if selected.duplicated(list(settings.SCENARIO_KEY_COLUMNS)).any():
        raise ValueError("Duplicate scenario keys in baseline table")
    failed = selected[selected["solver_status"] != 1]
    if not failed.empty:
        raise ValueError(f"Found {len(failed)} unsuccessful Iter-OPF scenarios")
    selected = selected.sort_values(list(settings.SCENARIO_KEY_COLUMNS)).reset_index(drop=True)
    selected.insert(
        0,
        "scenario_id",
        [
            f"{settings.COUNTRY_CODE}-S{index:04d}"
            for index in range(1, len(selected) + 1)
        ],
    )
    return selected


def scenario_table(scenarios: pd.DataFrame, settings) -> pd.DataFrame:
    columns = [
        "scenario_id",
        *settings.SCENARIO_KEY_COLUMNS,
        *settings.SCENARIO_CONTEXT_COLUMNS,
    ]
    return scenarios[columns].rename(
        columns={
            "fut_heatwave_date": "future_heatwave_datetime",
            "his_heatwave_date": "historical_heatwave_date",
            "fut_heatwave_year": "future_year",
            "fut_heatwave_month": "future_month",
            "fut_heatwave_day": "future_day",
            "fut_heatwave_hour": "future_hour",
            "his_heatwave_year": "historical_year",
            "air_temp": "air_temperature_c",
            "wind_speed": "wind_speed_m_per_s",
            "solar_radia": "solar_irradiance_w_per_m2",
            "load": "load_gw",
            "node_load_shedding": "load_shedding_percent",
        }
    )


def export_network(project: Path, settings) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    from data_config import RATIO  # noqa: PLC0415
    from utils.network_process_utils import load_network_EU  # noqa: PLC0415

    network, _ = load_network_EU(settings.COUNTRY_CODE, RATIO)
    if len(network.buses) != settings.N_BUSES or len(network.lines) != settings.N_BRANCHES:
        raise ValueError(
            f"Unexpected represented network size: {len(network.buses)} buses, "
            f"{len(network.lines)} branches"
        )
    buses = network.buses[["x", "y", "country"]].copy().reset_index()
    buses = buses.rename(columns={buses.columns[0]: "bus_id", "x": "longitude", "y": "latitude"})
    buses.insert(0, "bus_index", np.arange(len(buses)))
    coordinates = buses.set_index("bus_id")[["longitude", "latitude"]]

    lines = network.lines[["bus0", "bus1"]].copy().reset_index()
    lines = lines.rename(columns={lines.columns[0]: "line_id"})
    lines.insert(0, "branch_index", np.arange(len(lines)))
    lines["longitude_0"] = lines["bus0"].map(coordinates["longitude"])
    lines["latitude_0"] = lines["bus0"].map(coordinates["latitude"])
    lines["longitude_1"] = lines["bus1"].map(coordinates["longitude"])
    lines["latitude_1"] = lines["bus1"].map(coordinates["latitude"])
    network_path = project / "data" / "EU" / "networks" / "base_s_75_elec.nc"
    return buses, lines, network_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--country", choices=("ES", "IT", "FR"), default="ES")
    args = parser.parse_args()

    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(pipeline / "code"))
    from single_country_performance_definition import spatiotemporal_settings  # noqa: PLC0415

    settings = spatiotemporal_settings(args.country)

    package = (
        pipeline
        / "supplementary_track"
        / settings.PACKAGE_NAME
    )
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_csv(project, settings)
    scenarios = select_scenarios(summary_path, settings)
    compact_scenarios = scenario_table(scenarios, settings)
    write_csv(data_dir / "scenario_observations.csv", compact_scenarios)

    branch_rows = {
        "scenario_id": np.repeat(scenarios["scenario_id"].to_numpy(), settings.N_BRANCHES),
        "branch_index": np.tile(np.arange(settings.N_BRANCHES), settings.N_SCENARIOS),
    }
    temperatures: list[np.ndarray] = []
    capacities: list[np.ndarray] = []
    provenance_rows: list[dict] = []
    loaded_results: dict[tuple[str, str], dict] = {}
    for row in scenarios.itertuples(index=False):
        path = result_path(project, row, settings)
        if not path.exists():
            raise FileNotFoundError(path)
        result = np.load(path, allow_pickle=True).item()
        conductor = np.asarray(result["con_temp"], dtype=float)
        capacity = np.asarray(result["capacity_drop"], dtype=float).reshape(-1)
        if conductor.shape != (settings.N_BRANCHES, settings.N_SEGMENTS):
            raise ValueError(f"Unexpected conductor-temperature shape {conductor.shape}: {path}")
        if capacity.shape != (settings.N_BRANCHES,):
            raise ValueError(f"Unexpected capacity shape {capacity.shape}: {path}")
        if int(result.get("solver_status", row.solver_status)) != 1:
            raise ValueError(f"Unsuccessful result record: {path}")
        temperatures.append(conductor.max(axis=1))
        capacities.append(capacity * 100.0)
        result_key = (str(row.fut_heatwave_date), str(row.his_heatwave_date))
        if result_key in loaded_results:
            raise ValueError(f"Duplicate result key: {result_key}")
        loaded_results[result_key] = result
        provenance_rows.append(
            {
                "scenario_id": row.scenario_id,
                "record_path_relative_to_project": str(path.relative_to(project)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "solver_status": 1,
                "con_temp_shape": f"{settings.N_BRANCHES}x{settings.N_SEGMENTS}",
                "capacity_factor_shape": str(settings.N_BRANCHES),
            }
        )
    branch_rows["line_temperature_c"] = np.concatenate(temperatures)
    branch_rows["available_capacity_percent_of_nominal"] = np.concatenate(capacities)
    branches = pd.DataFrame(branch_rows)
    write_csv_gz(data_dir / "scenario_branch_observations.csv.gz", branches)
    write_rows(data_dir / "result_record_provenance.csv", provenance_rows)

    buses, lines, network_path = export_network(project, settings)
    write_csv(data_dir / "represented_network_buses.csv", buses)
    write_csv(data_dir / "represented_network_lines.csv", lines)

    weather_path = (
        project
        / "models"
        / settings.COUNTRY_CODE
        / "simu_data"
        / settings.MAP_WEATHER_RECORD
    )
    weather = xr.open_dataset(weather_path)
    map_branch_frames = []
    map_bus_frames = []
    map_weather_frames = []
    snapshot_rows = []
    for snapshot_index, future in enumerate(settings.MAP_FUTURE_DATETIMES, start=1):
        matching = scenarios[
            (scenarios["fut_heatwave_date"] == future)
            & (scenarios["his_heatwave_date"] == settings.MAP_HISTORICAL_DATE)
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one map scenario for {future}, found {len(matching)}")
        scenario = matching.iloc[0]
        result = loaded_results[(future, settings.MAP_HISTORICAL_DATE)]
        snapshot_id = f"map-{snapshot_index:02d}"

        branch_frame = pd.DataFrame(
            {
                "snapshot_id": snapshot_id,
                "branch_index": np.arange(settings.N_BRANCHES),
                "line_temperature_c": np.asarray(result["con_temp"], dtype=float).max(axis=1),
            }
        )
        map_branch_frames.append(branch_frame)

        demand = np.asarray(result["PD"], dtype=float).reshape(-1)
        shedding = np.asarray(result["LS"], dtype=float).reshape(-1)
        if demand.shape != (settings.N_BUSES,) or shedding.shape != (settings.N_BUSES,):
            raise ValueError(f"Unexpected bus-vector shape in {future}")
        ratio = shedding / demand.sum() * 100.0
        map_bus_frames.append(
            pd.DataFrame(
                {
                    "snapshot_id": snapshot_id,
                    "bus_index": np.arange(settings.N_BUSES),
                    "active_demand_mw": demand,
                    "load_shedding_mw": shedding,
                    "load_shedding_percent_of_total_demand": ratio,
                    "display_marker": ratio > settings.LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT,
                }
            )
        )

        weather_slice = weather.sel(time=future)
        temperature = np.asarray(weather_slice["temperature"], dtype=float) - 273.15
        x_values = np.asarray(weather_slice["x"], dtype=float) - 0.125
        y_values = np.asarray(weather_slice["y"], dtype=float) - 0.125
        y_index, x_index = np.indices(temperature.shape)
        map_weather_frames.append(
            pd.DataFrame(
                {
                    "snapshot_id": snapshot_id,
                    "y_index": y_index.reshape(-1),
                    "x_index": x_index.reshape(-1),
                    "longitude_plot": np.tile(x_values, len(y_values)),
                    "latitude_plot": np.repeat(y_values, len(x_values)),
                    "air_temperature_c": temperature.reshape(-1),
                }
            )
        )
        snapshot_rows.append(
            {
                "snapshot_id": snapshot_id,
                "scenario_id": scenario["scenario_id"],
                "future_heatwave_datetime": future,
                "historical_heatwave_date": settings.MAP_HISTORICAL_DATE,
                "future_hour": int(scenario["fut_heatwave_hour"]),
                "weather_record_relative_to_project": str(weather_path.relative_to(project)),
            }
        )
    write_csv(data_dir / "map_snapshots.csv", pd.DataFrame(snapshot_rows))
    write_csv(data_dir / "map_branch_observations.csv", pd.concat(map_branch_frames, ignore_index=True))
    write_csv(data_dir / "map_bus_observations.csv", pd.concat(map_bus_frames, ignore_index=True))
    write_csv_gz(
        data_dir / "map_weather_fields.csv.gz",
        pd.concat(map_weather_frames, ignore_index=True),
    )

    metadata = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "country_code": settings.COUNTRY_CODE,
        "n_buses": settings.N_BUSES,
        "n_branches": settings.N_BRANCHES,
        "n_segments": settings.N_SEGMENTS,
        "n_scenarios": settings.N_SCENARIOS,
        "scenarios_per_year": settings.SCENARIOS_PER_YEAR,
        "scenarios_per_hour": settings.SCENARIOS_PER_HOUR,
        "solver": settings.SOLVER,
        "storage_state": settings.STORAGE_STATE,
        "load_growth": settings.LOAD_GROWTH,
        "thermal_limit_c": settings.THERMAL_LIMIT_C,
        "capacity_security_margin_percent": settings.CAPACITY_SECURITY_MARGIN_PERCENT,
        "load_shedding_marker_threshold_percent": settings.LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT,
        "future_years": list(settings.FUTURE_YEARS),
        "future_hours": list(settings.FUTURE_HOURS),
        "hour_labels": list(settings.HOUR_LABELS),
        "sequential_orange": list(settings.SEQUENTIAL_ORANGE),
        "sequential_blue": list(settings.SEQUENTIAL_BLUE),
        "sequential_red": list(settings.SEQUENTIAL_RED),
        "statistical_artwork": list(settings.STATISTICAL_ARTWORK),
        "map_artwork": list(settings.MAP_ARTWORK),
        "colorbar_artwork": list(settings.COLORBAR_ARTWORK),
        "panel_units": {
            "a-d": "scenario-branch observations",
            "e-f": "scenario observations",
            "g": "four selected hourly spatial snapshots",
        },
        "capacity_metric": {
            "source_field": "capacity_drop",
            "stored_name": "available_capacity_percent_of_nominal",
            "definition": "minimum weather-dependent current limit divided by nominal current, multiplied by 100",
            "current_artwork_axis_label": "Line Capacity Drop (%)",
        },
        "map_selection": {
            "future_datetimes": list(settings.MAP_FUTURE_DATETIMES),
            "historical_heatwave_date": settings.MAP_HISTORICAL_DATE,
            "source_notebook_rule": "first July scenario associated with historical heatwave year 2024 and heatwave index 0",
        },
        "map_display": {
            "air_temperature_min_c": 20,
            "air_temperature_max_c": 50,
            "air_temperature_alpha": 0.5,
            "line_temperature_min_c": 30,
            "line_temperature_max_c": 90,
            "line_temperature_alpha": 0.75,
            "line_width": 2.75,
            "bus_marker_size": 100,
            "load_shedding_marker_size": 300,
            "coordinate_offset_degrees": 0.125,
        },
        "colorbars": [
            {
                "filename": "color_bar_temperature.pdf",
                "kind": "air_temperature",
                "vmin": 20,
                "vmax": 50,
                "levels": 6,
                "alpha": 0.5,
                "label": "Air temperature (°C)",
            },
            {
                "filename": "color_bar_line_temp.pdf",
                "kind": "line_temperature",
                "vmin": 30,
                "vmax": 90,
                "levels": 6,
                "alpha": 0.9,
                "label": "Line temperature (°C)",
            },
        ],
        "simulation_rerun": False,
    }
    (data_dir / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    responsible_code = [
        project / "vis" / "2.eur_single_analysis.ipynb",
        project / "vis" / "4.grid_simu_vis.ipynb",
        project / "utils" / "plot_utils.py",
        pipeline / "code" / "single_country_performance_definition.py",
        Path(__file__).resolve(),
    ]
    upstream = [summary_path, network_path, weather_path]
    manifest = {
        "supplementary_figure": settings.SUPPLEMENTARY_FIGURE,
        "status": "COMPACT_PLOTTED_DATA_EXPORTED",
        "simulation_rerun": False,
        "counts": {
            "scenario_rows": len(compact_scenarios),
            "scenario_branch_rows": len(branches),
            "result_record_provenance_rows": len(provenance_rows),
            "network_bus_rows": len(buses),
            "network_line_rows": len(lines),
            "map_snapshot_rows": len(snapshot_rows),
            "map_branch_rows": sum(len(frame) for frame in map_branch_frames),
            "map_bus_rows": sum(len(frame) for frame in map_bus_frames),
            "map_weather_rows": sum(len(frame) for frame in map_weather_frames),
        },
        "upstream_inputs": [
            {
                "path": str(path),
                "path_relative_to_project": str(path.relative_to(project)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in upstream
        ],
        "responsible_code": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in responsible_code
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

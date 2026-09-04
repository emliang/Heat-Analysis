#!/usr/bin/env python3
"""Export compact plotted data for Supplementary Figs. 25--28.

This reads frozen result records, represented networks and selected weather
snapshots. It never reruns an OPF or weather simulation.
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
                frame.to_csv(text, index=False, lineterminator="\n", float_format="%.12g")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_entry(path: Path, project: Path | None = None) -> dict:
    entry = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if project is not None:
        entry["path_relative_to_project"] = str(path.relative_to(project))
    return entry


def config_token(configuration) -> str:
    return configuration if isinstance(configuration, str) else str(list(configuration))


def config_id(configuration) -> str:
    return configuration if isinstance(configuration, str) else "-".join(configuration)


def analysis_csv(project: Path, configuration, n_buses: int, load_growth: float, settings) -> Path:
    token = config_token(configuration)
    suffix = "model_analysis" if isinstance(configuration, str) and load_growth == 1.01 else "sensitivity_analysis"
    return project / "models" / token / (
        f"{token}_{n_buses}_bus_renewable_True_heatwave_True_storage_True_"
        f"{settings.STORAGE_STATE}_load_growth_True_{load_growth}_max_temp_"
        f"{settings.THERMAL_LIMIT_C}_{suffix}.csv"
    )


def result_path(project: Path, configuration, n_buses: int, future: str, historical: str, load_growth: float, settings) -> Path:
    token = config_token(configuration)
    historical_datetime = datetime.strptime(historical, "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")
    filename = (
        f"{token}_{n_buses}_{settings.SOLVER}_{future}_{historical_datetime}_"
        f"storage_{settings.STORAGE_STATE}_load_growth_{load_growth}_thermal_"
        f"{settings.THERMAL_LIMIT_C}_results.npy"
    )
    return project / "models" / token / "ppc" / future / filename


def select_summary(path: Path, settings) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"fut_heatwave_date": str, "his_heatwave_date": str})
    selected = table[table["TDPF_solver"] == settings.SOLVER].copy()
    selected = selected.sort_values(list(settings.SCENARIO_KEY_COLUMNS)).reset_index(drop=True)
    if len(selected) != settings.N_SCENARIOS:
        raise ValueError(f"Expected {settings.N_SCENARIOS} scenarios in {path}, found {len(selected)}")
    if selected.duplicated(list(settings.SCENARIO_KEY_COLUMNS)).any():
        raise ValueError(f"Duplicate scenario key in {path}")
    if not (selected["solver_status"] == 1).all():
        raise ValueError(f"Unsuccessful scenarios in {path}")
    return selected


def network_tables(configuration, network, reference_country: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    buses = network.buses[["x", "y", "country"]].copy().reset_index()
    buses = buses.rename(columns={buses.columns[0]: "bus_id", "x": "longitude", "y": "latitude"})
    buses.insert(0, "bus_index", np.arange(len(buses)))
    buses.insert(0, "configuration_id", config_id(configuration))
    buses["is_reference_country"] = buses["country"] == reference_country
    coordinates = buses.set_index("bus_id")[["longitude", "latitude", "country"]]

    lines = network.lines[["bus0", "bus1"]].copy().reset_index()
    lines = lines.rename(columns={lines.columns[0]: "line_id"})
    lines.insert(0, "branch_index", np.arange(len(lines)))
    lines.insert(0, "configuration_id", config_id(configuration))
    lines["country_0"] = lines["bus0"].map(coordinates["country"])
    lines["country_1"] = lines["bus1"].map(coordinates["country"])
    lines["is_reference_internal"] = (
        (lines["country_0"] == reference_country)
        & (lines["country_1"] == reference_country)
    )
    lines["longitude_0"] = lines["bus0"].map(coordinates["longitude"])
    lines["latitude_0"] = lines["bus0"].map(coordinates["latitude"])
    lines["longitude_1"] = lines["bus1"].map(coordinates["longitude"])
    lines["latitude_1"] = lines["bus1"].map(coordinates["latitude"])
    return buses, lines, lines["is_reference_internal"].to_numpy(dtype=bool)


def load_networks(project: Path, settings):
    from data_config import RATIO  # noqa: PLC0415
    from utils.network_process_utils import load_network_EU  # noqa: PLC0415

    networks = {}
    tables = {}
    for configuration in dict.fromkeys((*settings.STATISTICS_CONFIGS, *settings.MAP_CONFIGS)):
        loader_value = configuration if isinstance(configuration, str) else list(configuration)
        network, _ = load_network_EU(loader_value, RATIO)
        networks[config_id(configuration)] = network
        tables[config_id(configuration)] = network_tables(configuration, network, settings.COUNTRY_CODE)
    return networks, tables


def export_statistics(project: Path, pipeline: Path, settings, networks, tables) -> Path:
    from cross_border_supplement_definition import statistical_artwork  # noqa: PLC0415

    package = pipeline / "supplementary_track" / settings.STATISTICS_PACKAGE
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scenario_frames = []
    branch_frames = []
    alignment_frames = []
    provenance_rows = []
    summary_inputs = []
    reference_line_frames = []

    single_network = networks[settings.COUNTRY_CODE]
    single_line_ids = single_network.lines.index.astype(str).to_numpy()

    for load_growth in settings.LOAD_GROWTHS:
        summaries = {}
        for configuration in settings.STATISTICS_CONFIGS:
            cid = config_id(configuration)
            network = networks[cid]
            summary_path = analysis_csv(project, configuration, len(network.buses), load_growth, settings)
            summary = select_summary(summary_path, settings)
            summaries[cid] = summary
            summary_inputs.append(summary_path)

        base = summaries[settings.COUNTRY_CODE]
        base_keys = pd.MultiIndex.from_frame(base[list(settings.SCENARIO_KEY_COLUMNS)])
        alignment = base[list(settings.SCENARIO_KEY_COLUMNS)].copy()
        alignment.insert(0, "scenario_id", [
            f"{settings.COUNTRY_CODE}-LG{int(round(load_growth * 100)):03d}-S{i:04d}"
            for i in range(1, len(base) + 1)
        ])
        alignment.insert(1, "load_growth", load_growth)
        for configuration in settings.STATISTICS_CONFIGS:
            cid = config_id(configuration)
            keys = pd.MultiIndex.from_frame(summaries[cid][list(settings.SCENARIO_KEY_COLUMNS)])
            if not base_keys.equals(keys):
                raise ValueError(f"Scenario-key mismatch for {cid}, load growth {load_growth}")
            alignment[f"{cid}_successful"] = True
        alignment_frames.append(alignment)

        for scenario_number, base_row in enumerate(base.itertuples(index=False), start=1):
            scenario_id = f"{settings.COUNTRY_CODE}-LG{int(round(load_growth * 100)):03d}-S{scenario_number:04d}"
            future = str(base_row.fut_heatwave_date)
            historical = str(base_row.his_heatwave_date)
            single_path = result_path(
                project, settings.COUNTRY_CODE, len(single_network.buses),
                future, historical, load_growth, settings,
            )
            single_result = np.load(single_path, allow_pickle=True).item()
            if int(single_result["solver_status"]) != 1:
                raise ValueError(single_path)
            denominator = float(np.asarray(single_result["PD"], dtype=float).sum())

            for configuration_index, configuration in enumerate(settings.STATISTICS_CONFIGS):
                cid = config_id(configuration)
                network = networks[cid]
                path = result_path(
                    project, configuration, len(network.buses), future, historical,
                    load_growth, settings,
                )
                result = single_result if cid == settings.COUNTRY_CODE else np.load(path, allow_pickle=True).item()
                if int(result["solver_status"]) != 1:
                    raise ValueError(path)
                bus_ref = network.buses.country.to_numpy() == settings.COUNTRY_CODE
                _, line_table, line_ref = tables[cid]
                reference_lines = line_table[line_ref].copy().reset_index(drop=True)
                if len(reference_lines) != settings.N_REFERENCE_LINES:
                    raise ValueError(f"{cid} has {len(reference_lines)} reference-country lines")
                if not np.array_equal(reference_lines["line_id"].astype(str).to_numpy(), single_line_ids):
                    raise ValueError(f"Reference-country line identity/order mismatch for {cid}")
                if scenario_number == 1 and load_growth == settings.LOAD_GROWTHS[0]:
                    reference_lines.insert(1, "reference_branch_index", np.arange(len(reference_lines)))
                    reference_line_frames.append(reference_lines)

                load_shedding = float(np.asarray(result["LS"], dtype=float)[bus_ref].sum())
                scenario_frames.append(pd.DataFrame([{
                    "scenario_id": scenario_id,
                    "configuration_id": cid,
                    "configuration_order": configuration_index,
                    "configuration_label": settings.LABELS[configuration_index],
                    "load_growth": load_growth,
                    "future_heatwave_datetime": future,
                    "historical_heatwave_date": historical,
                    "future_year": int(base_row.fut_heatwave_year),
                    "future_hour": int(base_row.fut_heatwave_hour),
                    "reference_demand_mw": denominator,
                    "reference_load_shedding_mw": load_shedding,
                    "load_shedding_percent": load_shedding / denominator * 100.0,
                }]))

                conductor = np.asarray(result["con_temp"], dtype=float)
                capacity = np.asarray(result["capacity_drop"], dtype=float).reshape(-1)
                if conductor.shape[0] != len(network.lines) or conductor.shape[1] != settings.N_SEGMENTS:
                    raise ValueError(f"Unexpected conductor shape {conductor.shape}: {path}")
                if capacity.shape != (len(network.lines),):
                    raise ValueError(f"Unexpected capacity shape {capacity.shape}: {path}")
                branch_frames.append(pd.DataFrame({
                    "scenario_id": scenario_id,
                    "configuration_id": cid,
                    "configuration_order": configuration_index,
                    "load_growth": load_growth,
                    "reference_branch_index": np.arange(settings.N_REFERENCE_LINES),
                    "line_id": reference_lines["line_id"].astype(str).to_numpy(),
                    "line_temperature_c": conductor.max(axis=1)[line_ref],
                    "available_capacity_percent_of_nominal": capacity[line_ref] * 100.0,
                }))
                provenance_rows.append({
                    "scenario_id": scenario_id,
                    "configuration_id": cid,
                    "load_growth": load_growth,
                    "record_path_relative_to_project": str(path.relative_to(project)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "solver_status": 1,
                    "represented_line_count": len(network.lines),
                    "reference_internal_line_count": int(line_ref.sum()),
                })

    scenarios = pd.concat(scenario_frames, ignore_index=True)
    branches = pd.concat(branch_frames, ignore_index=True)
    alignment = pd.concat(alignment_frames, ignore_index=True)
    reference_lines = pd.concat(reference_line_frames, ignore_index=True)
    write_csv(data_dir / "scenario_observations.csv", scenarios)
    write_csv_gz(data_dir / "scenario_branch_observations.csv.gz", branches)
    write_csv(data_dir / "scenario_alignment.csv", alignment)
    write_csv(data_dir / "reference_network_lines.csv", reference_lines)
    write_rows(data_dir / "result_record_provenance.csv", provenance_rows)

    metadata = {
        "supplementary_figure": settings.STATISTICS_FIGURE,
        "reference_country": settings.COUNTRY_CODE,
        "solver": settings.SOLVER,
        "storage_state": settings.STORAGE_STATE,
        "load_growths": list(settings.LOAD_GROWTHS),
        "thermal_limit_c": settings.THERMAL_LIMIT_C,
        "capacity_security_margin_percent": settings.CAPACITY_SECURITY_MARGIN_PERCENT,
        "n_scenarios_per_load_growth": settings.N_SCENARIOS,
        "n_reference_internal_ac_lines": settings.N_REFERENCE_LINES,
        "configuration_ids": [config_id(value) for value in settings.STATISTICS_CONFIGS],
        "configuration_labels": list(settings.LABELS),
        "colors": list(settings.COLORS),
        "artwork": list(statistical_artwork(settings.COUNTRY_CODE)),
        "statistical_units": {
            "load_shedding": "one observation per scenario for the reference country",
            "line_temperature": "one observation per scenario and reference-country internal AC line",
            "available_capacity": "one observation per scenario and reference-country internal AC line",
        },
        "branch_population_rule": (
            "For every single or joint-grid configuration, branch metrics include only AC lines "
            "whose two endpoint buses are in the reference country. Cross-border effects remain "
            "represented through the joint OPF solution."
        ),
        "load_shedding_normalisation": (
            "Reference-country load shedding in every configuration is divided by total demand "
            "from the matched single-country result for that scenario."
        ),
        "capacity_metric": {
            "source_field": "capacity_drop",
            "stored_name": "available_capacity_percent_of_nominal",
            "definition": "weather-dependent current limit divided by nominal current, multiplied by 100",
            "current_artwork_axis_label": "Line Capacity Drop (%)",
        },
        "simulation_rerun": False,
    }
    (data_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    manifest = {
        "supplementary_figure": settings.STATISTICS_FIGURE,
        "status": "COMPACT_PLOTTED_DATA_EXPORTED",
        "simulation_rerun": False,
        "counts": {
            "scenario_rows": len(scenarios),
            "scenario_branch_rows": len(branches),
            "alignment_rows": len(alignment),
            "reference_network_line_rows": len(reference_lines),
            "result_record_provenance_rows": len(provenance_rows),
        },
        "upstream_inputs": [file_entry(path, project) for path in dict.fromkeys(summary_inputs)],
        "responsible_code": [
            file_entry(path)
            for path in (
                project / "vis" / "3.eur_multi_analysis.ipynb",
                project / "utils" / "plot_utils.py",
                pipeline / "code" / "cross_border_supplement_definition.py",
                Path(__file__).resolve(),
            )
        ],
        "files": [file_entry(path) for path in sorted(data_dir.iterdir()) if path.is_file() and path.name != "manifest.json"],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return package


def export_maps(project: Path, pipeline: Path, settings, networks, tables) -> Path:
    package = pipeline / "supplementary_track" / settings.MAPS_PACKAGE
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    bus_frames = []
    line_frames = []
    branch_frames = []
    bus_value_frames = []
    weather_frames = []
    snapshot_rows = []
    provenance_rows = []
    weather_inputs = []

    for configuration_order, configuration in enumerate(settings.MAP_CONFIGS):
        cid = config_id(configuration)
        token = config_token(configuration)
        network = networks[cid]
        buses, lines, _ = tables[cid]
        bus_frames.append(buses)
        line_frames.append(lines)
        datetimes = settings.SINGLE_MAP_DATETIMES if isinstance(configuration, str) else settings.JOINT_MAP_DATETIMES
        weather_path = project / "models" / token / "simu_data" / "future_weather_data_based_on_historical_hot_event_2024_0.nc"
        weather_inputs.append(weather_path)
        with xr.open_dataset(weather_path) as weather:
            for year_index, future in enumerate(datetimes):
                snapshot_id = f"{cid}-{future[:4]}"
                path = result_path(
                    project, configuration, len(network.buses), future,
                    settings.HISTORICAL_HEATWAVE_DATE, 1.01, settings,
                )
                result = np.load(path, allow_pickle=True).item()
                if int(result["solver_status"]) != 1:
                    raise ValueError(path)
                conductor = np.asarray(result["con_temp"], dtype=float)
                demand = np.asarray(result["PD"], dtype=float).reshape(-1)
                shedding = np.asarray(result["LS"], dtype=float).reshape(-1)
                if conductor.shape != (len(network.lines), settings.N_SEGMENTS):
                    raise ValueError(f"Unexpected conductor shape {conductor.shape}: {path}")
                if demand.shape != (len(network.buses),) or shedding.shape != (len(network.buses),):
                    raise ValueError(f"Unexpected bus vectors: {path}")
                reference = network.buses.country.to_numpy() == settings.COUNTRY_CODE
                reference_total_demand = float(demand[reference].sum())
                ratio = shedding / reference_total_demand * 100.0
                branch_frames.append(pd.DataFrame({
                    "snapshot_id": snapshot_id,
                    "configuration_id": cid,
                    "branch_index": np.arange(len(network.lines)),
                    "line_temperature_c": conductor.max(axis=1),
                }))
                bus_value_frames.append(pd.DataFrame({
                    "snapshot_id": snapshot_id,
                    "configuration_id": cid,
                    "bus_index": np.arange(len(network.buses)),
                    "active_demand_mw": demand,
                    "load_shedding_mw": shedding,
                    "load_shedding_percent_of_reference_demand": ratio,
                    "display_marker": reference & (ratio > settings.LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT),
                }))
                weather_slice = weather.sel(time=future)
                temperature = np.asarray(weather_slice["temperature"], dtype=float) - 273.15
                x_values = np.asarray(weather_slice["x"], dtype=float) - 0.125
                y_values = np.asarray(weather_slice["y"], dtype=float) - 0.125
                y_index, x_index = np.indices(temperature.shape)
                weather_frames.append(pd.DataFrame({
                    "snapshot_id": snapshot_id,
                    "y_index": y_index.reshape(-1),
                    "x_index": x_index.reshape(-1),
                    "longitude_plot": np.tile(x_values, len(y_values)),
                    "latitude_plot": np.repeat(y_values, len(x_values)),
                    "air_temperature_c": temperature.reshape(-1),
                }))
                filename = (
                    f"{token}_{settings.SOLVER}_{future}_"
                    f"{settings.HISTORICAL_HEATWAVE_DATE} 00:00:00.pdf"
                )
                original_relative = (
                    f"figs/{settings.COUNTRY_CODE}_simu/{settings.COUNTRY_CODE}_Grid/{filename}"
                    if isinstance(configuration, str)
                    else f"figs/MultiCountry/{settings.COUNTRY_CODE}/Grid/{filename}"
                )
                snapshot_rows.append({
                    "snapshot_id": snapshot_id,
                    "configuration_id": cid,
                    "configuration_order": configuration_order,
                    "display_year_order": year_index,
                    "future_heatwave_datetime": future,
                    "historical_heatwave_date": settings.HISTORICAL_HEATWAVE_DATE,
                    "artwork_filename": filename,
                    "original_artwork_relative_to_manuscript": original_relative,
                    "selection_basis": (
                        "author-selected sample snapshot; not a same-hour controlled comparison "
                        "and not an algorithmically selected peak"
                    ),
                })
                provenance_rows.append({
                    "snapshot_id": snapshot_id,
                    "configuration_id": cid,
                    "result_record_path_relative_to_project": str(path.relative_to(project)),
                    "result_record_bytes": path.stat().st_size,
                    "result_record_sha256": sha256(path),
                    "weather_record_path_relative_to_project": str(weather_path.relative_to(project)),
                    "weather_record_bytes": weather_path.stat().st_size,
                    "weather_record_sha256": sha256(weather_path),
                    "solver_status": 1,
                })

    buses = pd.concat(bus_frames, ignore_index=True)
    lines = pd.concat(line_frames, ignore_index=True)
    branches = pd.concat(branch_frames, ignore_index=True)
    bus_values = pd.concat(bus_value_frames, ignore_index=True)
    weather_values = pd.concat(weather_frames, ignore_index=True)
    snapshots = pd.DataFrame(snapshot_rows)
    write_csv(data_dir / "represented_network_buses.csv", buses)
    write_csv(data_dir / "represented_network_lines.csv", lines)
    write_csv(data_dir / "map_snapshots.csv", snapshots)
    write_csv(data_dir / "map_branch_observations.csv", branches)
    write_csv(data_dir / "map_bus_observations.csv", bus_values)
    write_csv_gz(data_dir / "map_weather_fields.csv.gz", weather_values)
    write_rows(data_dir / "record_provenance.csv", provenance_rows)

    metadata = {
        "supplementary_figure": settings.MAPS_FIGURE,
        "reference_country": settings.COUNTRY_CODE,
        "configuration_ids": [config_id(value) for value in settings.MAP_CONFIGS],
        "selected_snapshots": len(snapshots),
        "solver": settings.SOLVER,
        "storage_state": settings.STORAGE_STATE,
        "load_growth": 1.01,
        "thermal_limit_c": settings.THERMAL_LIMIT_C,
        "historical_heatwave_date": settings.HISTORICAL_HEATWAVE_DATE,
        "map_selection": (
            "Author-selected sample snapshots are preserved exactly. They are not aligned to a "
            "common hour across configurations and are not algorithmically selected peaks."
        ),
        "marker_scope": "Only reference-country buses can receive load-shedding markers.",
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
            {"filename": "color_bar_temperature.pdf", "kind": "air_temperature", "vmin": 20, "vmax": 50, "levels": 6, "alpha": 0.5, "label": "Air temperature (°C)"},
            {"filename": "color_bar_line_temp.pdf", "kind": "line_temperature", "vmin": 30, "vmax": 90, "levels": 6, "alpha": 0.9, "label": "Line temperature (°C)"},
        ],
        "simulation_rerun": False,
    }
    (data_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    network_path = project / "data" / "EU" / "networks" / "base_s_75_elec.nc"
    manifest = {
        "supplementary_figure": settings.MAPS_FIGURE,
        "status": "COMPACT_PLOTTED_DATA_EXPORTED",
        "simulation_rerun": False,
        "counts": {
            "network_bus_rows": len(buses),
            "network_line_rows": len(lines),
            "map_snapshot_rows": len(snapshots),
            "map_branch_rows": len(branches),
            "map_bus_rows": len(bus_values),
            "map_weather_rows": len(weather_values),
            "record_provenance_rows": len(provenance_rows),
        },
        "upstream_inputs": [file_entry(path, project) for path in (network_path, *dict.fromkeys(weather_inputs))],
        "responsible_code": [
            file_entry(path)
            for path in (
                project / "vis" / "4.grid_simu_vis.ipynb",
                project / "utils" / "plot_utils.py",
                pipeline / "code" / "cross_border_supplement_definition.py",
                Path(__file__).resolve(),
            )
        ],
        "files": [file_entry(path) for path in sorted(data_dir.iterdir()) if path.is_file() and path.name != "manifest.json"],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return package


def write_readmes(settings, statistics_package: Path, maps_package: Path) -> None:
    statistics_text = f"""# Supplementary Fig. {settings.STATISTICS_FIGURE} compact reproduction package

This package contains the plotted scenario and scenario-line values for the
{settings.COUNTRY_NAME} cross-border statistical comparison. Branch statistics
use only internal AC lines in the reference country. Frozen result records are
hashed but not copied, and no simulation is rerun.

The `artwork/` directory is generated only from `data/`. The submitted SI
artwork remains unchanged until author review of the corrected branch scope and
the capacity-metric terminology.
"""
    maps_text = f"""# Supplementary Fig. {settings.MAPS_FIGURE} compact reproduction package

This package contains represented-network geometry, selected result fields and
compact weather grids sufficient to reproduce the {settings.COUNTRY_NAME}
cross-border maps. The exact author-selected sample timestamps are preserved;
they are not same-hour controlled comparisons or algorithmically selected
peaks. Frozen NPY and NetCDF inputs are hashed but not copied.
"""
    (statistics_package / "README.md").write_text(statistics_text)
    (maps_package / "README.md").write_text(maps_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--country", choices=("ES", "FR"), required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(pipeline / "code"))
    from cross_border_supplement_definition import settings  # noqa: PLC0415

    cfg = settings(args.country)
    networks, tables = load_networks(project, cfg)
    statistics_package = export_statistics(project, pipeline, cfg, networks, tables)
    maps_package = export_maps(project, pipeline, cfg, networks, tables)
    write_readmes(cfg, statistics_package, maps_package)
    print(json.dumps({"statistics_package": str(statistics_package), "maps_package": str(maps_package)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare small reader-facing data from the checked ES-2030 example."""

import argparse
from datetime import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_european_example import check_bundle, example_settings, file_sha256


SNAPSHOT = "ES-2030"
PUBLISHED_DATA = ROOT / "vis/figure_source/data/Source_Data_Extended_Data_Fig_2.zip"


def published_table(archive, name):
    path = f"data/ExtendedDataFig2_{name}.csv"
    if name == "map_weather_fields":
        return pd.read_csv(io.BytesIO(gzip.decompress(archive.read(path + ".gz"))))
    return pd.read_csv(io.BytesIO(archive.read(path)))


def export(bundle, result_dir, source_root, output_dir, web_data):
    bundle = bundle.resolve()
    result_dir = result_dir.resolve()
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    web_data = web_data.resolve()
    case = check_bundle(bundle)["case"]
    if (case["timestamp"], case["historical_year"],
            case["historical_event_index"]) != ("2030-07-24T13:00:00", 2024, 0):
        raise ValueError("Expected the 2030 ES example with 2024 heatwave anchor")
    summary = json.loads((result_dir / "run_summary.json").read_text())
    if summary["case"] != case or summary["solver_status"] != 1:
        raise ValueError("Result does not match the checked input bundle")
    if summary["input_manifest_sha256"] != file_sha256(bundle / "manifest.json"):
        raise ValueError("Result input manifest differs from the bundle")
    result_path = Path(summary["result_npy"])
    if not result_path.is_file():
        result_path = next((result_dir / "models/ES/ppc").rglob("*_results.npy"))
    result = np.load(result_path, allow_pickle=True).item()
    if result["solver_status"] != 1:
        raise ValueError("Saved result has no successful solver status")

    with zipfile.ZipFile(PUBLISHED_DATA) as archive:
        snapshots = published_table(archive, "map_snapshots")
        snapshot = snapshots.set_index("snapshot_id").loc[SNAPSHOT]
        branches = published_table(archive, "map_branch_observations")
        branches = branches.loc[branches.snapshot_id == SNAPSHOT].sort_values("branch_index")
        buses = published_table(archive, "map_bus_observations")
        buses = buses.loc[buses.snapshot_id == SNAPSHOT].sort_values("bus_index")
        weather = published_table(archive, "map_weather_fields")
        weather = weather.loc[weather.snapshot_id == SNAPSHOT]
        network_buses = published_table(archive, "represented_network_buses")
        network_buses = network_buses.loc[network_buses.configuration_id == "ES"]
        network_lines = published_table(archive, "represented_network_lines")
        network_lines = network_lines.loc[network_lines.configuration_id == "ES"]
        provenance = published_table(archive, "record_provenance")
        reference = provenance.set_index("snapshot_id").loc[SNAPSHOT]
        if (snapshot.future_heatwave_datetime != "2030-07-24 13:00:00"
                or snapshot.historical_heatwave_date != "2024-07-30"):
            raise ValueError("Published snapshot differs from the example")
        if (len(branches), len(buses)) != (len(result["con_temp"]), len(result["LS"])):
            raise ValueError("Result and published map dimensions differ")
        line_error = np.max(np.abs(
            branches.line_temperature_c.to_numpy()
            - np.nanmax(result["con_temp"], axis=1)))
        bus_error = np.max(np.abs(
            buses.load_shedding_mw.to_numpy() - result["LS"]))
        if line_error > 0.001 or bus_error > 0.001:
            raise ValueError("Recomputed example disagrees with the published map")

    output_dir.mkdir(parents=True, exist_ok=True)
    web_data.parent.mkdir(parents=True, exist_ok=True)
    source_weather = (source_root / "models/ES/simu_data/"
                      "future_weather_data_based_on_historical_hot_event_2024_0.nc")
    with xr.open_dataset(source_weather) as full_weather:
        day = full_weather.sel(time=pd.date_range("2030-07-24", periods=24, freq="h"))
        if day.sizes["time"] != 24:
            raise ValueError("Expected all 24 hours of the 2030 example day")
        day[["temperature", "wnd10m", "influx", "lon", "lat"]].to_netcdf(
            output_dir / "weather_day.nc")

    os.environ["HEATANALYSIS_LOCAL_DATA"] = str(bundle / "data")
    os.environ["HEATANALYSIS_MODELS"] = str(bundle / "models")
    from TDOPF_eur import prepare_network
    from utils.network_process_utils import load_network_EU

    network, regions = load_network_EU("ES", 75)
    args, _, _ = example_settings(case)
    network = prepare_network(network, regions,
                              datetime.fromisoformat(case["timestamp"]),
                              "ES", 2024, 0, args)
    if not network.buses.index.equals(pd.Index(network_buses.sort_values("bus_index").bus_id)):
        raise ValueError("Prepared bus order differs from published map")
    if not network.lines.index.equals(pd.Index(network_lines.sort_values("branch_index").line_id.astype(str))):
        raise ValueError("Prepared line order differs from published map")

    demand_data = np.load(bundle / "models/ES/simu_data/"
                          "future_demand_data_based_on_historical_hot_event_2024_0.npy",
                          allow_pickle=True).item()
    curve = np.load(bundle / "models/demand_curve/ES/ES_[2015, 2024]_demand_curve.npy",
                    allow_pickle=True).item()
    day_index = demand_data["future_date"].index(pd.Timestamp("2030-07-24"))
    daily_demand = float(demand_data["daily_demand"][day_index][0])
    grown_base = float(curve["Pb"]) * (1.01 ** (2030 - 2025) - 1)
    ratios = np.reshape(curve["weekday_cooling_hour_ratio"], (-1))
    demand_hourly = pd.DataFrame({
        "hour": np.arange(24),
        "modelled_demand_mw": (daily_demand + grown_base) * ratios * 1000 * 24,
    })
    if abs(demand_hourly.loc[13, "modelled_demand_mw"]
           - network.buses.p_set.sum()) > 1e-5:
        raise ValueError("Hourly demand differs from prepared network")
    demand_hourly.to_csv(output_dir / "demand_hourly.csv", index=False)

    generator_inputs = network.generators[["bus", "carrier", "p_nom", "p_max_pu",
                                            "derating"]].copy()
    generator_inputs["bus_temperature_c"] = (
        network.buses.temperature.reindex(generator_inputs.bus).to_numpy())
    generator_inputs.index.name = "generator_id"
    generator_inputs.to_csv(output_dir / "generator_inputs.csv")
    seg_weight = network.segments[:, :, 2]
    line_inputs = pd.DataFrame({
        "line_id": network.lines.index,
        "air_temperature_c": np.sum(network.segment_weather[:, :, 0] * seg_weight, axis=1),
        "wind_speed_m_s": np.sum(network.segment_weather[:, :, 1] * seg_weight, axis=1),
        "solar_irradiance_w_m2": np.sum(network.segment_weather[:, :, 2] * seg_weight, axis=1),
        "available_current_a": np.asarray(result["Imax"]).reshape(-1),
        "line_temperature_c": np.nanmax(result["con_temp"], axis=1),
    })
    line_inputs.to_csv(output_dir / "line_inputs.csv", index=False)

    bus_map = network_buses.set_index("bus_index")
    line_map = network_lines.set_index("branch_index")
    map_data = {
        "case": {
            "snapshot_id": SNAPSHOT,
            "title": "Spain, 24 July 2030, 13:00",
            "historical_heatwave_date": "2024-07-30",
            "network": "281-bus represented Spanish grid",
            "load_growth_rate": 0.01,
            "initial_storage_soc": 0.8,
            "selection": "Illustrative published snapshot; not a peak or an outage forecast",
            "source_data": "Source_Data_Extended_Data_Fig_2.zip",
        },
        "weather": [
            {"lon": float(row.longitude_plot), "lat": float(row.latitude_plot),
             "temperature_c": float(row.air_temperature_c)}
            for row in weather.itertuples()],
        "buses": [
            {"id": str(bus_map.loc[row.bus_index, "bus_id"]),
             "lon": float(bus_map.loc[row.bus_index, "longitude"]),
             "lat": float(bus_map.loc[row.bus_index, "latitude"]),
             "demand_mw": float(row.active_demand_mw),
             "load_shedding_mw": float(row.load_shedding_mw),
             "display_marker": bool(row.display_marker)}
            for row in buses.itertuples()],
        "lines": [
            {"id": str(line_map.loc[row.branch_index, "line_id"]),
             "bus0": str(line_map.loc[row.branch_index, "bus0"]),
             "bus1": str(line_map.loc[row.branch_index, "bus1"]),
             "lon0": float(line_map.loc[row.branch_index, "longitude_0"]),
             "lat0": float(line_map.loc[row.branch_index, "latitude_0"]),
             "lon1": float(line_map.loc[row.branch_index, "longitude_1"]),
             "lat1": float(line_map.loc[row.branch_index, "latitude_1"]),
             "temperature_c": float(row.line_temperature_c)}
            for row in branches.itertuples()],
        "region_outline": mapping(regions.geometry.unary_union.simplify(0.02)),
    }
    payload = json.dumps(map_data, separators=(",", ":"), ensure_ascii=True) + "\n"
    (output_dir / "grid_result.json").write_text(payload)
    web_data.write_text(payload)
    checks = {
        "example_input_manifest_sha256": file_sha256(bundle / "manifest.json"),
        "example_result_sha256": file_sha256(result_path),
        "published_result_sha256": str(reference.result_record_sha256),
        "published_source_zip_sha256": file_sha256(PUBLISHED_DATA),
        "max_line_temperature_difference_c": float(line_error),
        "max_load_shedding_difference_mw": float(bus_error),
        "web_data_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (output_dir / "provenance.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "data/examples/es_2030-07-24_13_2024-0")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "vis/demos/data/spain_2030")
    parser.add_argument("--web-data", type=Path,
                        default=ROOT / "webpage/assets/data/spain_2030.json")
    options = parser.parse_args()
    print(json.dumps(export(options.bundle, options.result_dir,
                            options.source_root, options.output_dir,
                            options.web_data), indent=2))

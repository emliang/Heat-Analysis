#!/usr/bin/env python3
"""Extract the published ES-2030 hour into a small, checked input bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pypsa
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.country_network_filter import filter_network_by_countries


TIMESTAMP = "2030-07-24T13:00:00"
ANCHOR = "2024_0"
NETWORK = "data/EU/networks/base_s_75_elec.nc"
REGIONS = "data/EU/regions_onshore_base_s_75.geojson"
LOAD_RATIOS = "data/EU/load_ratio_base_s_75.csv"
WEATHER = f"models/ES/simu_data/future_weather_data_based_on_historical_hot_event_{ANCHOR}.nc"
DEMAND = f"models/ES/simu_data/future_demand_data_based_on_historical_hot_event_{ANCHOR}.npy"
CURVE = "models/demand_curve/ES/ES_[2015, 2024]_demand_curve.npy"
PROFILES = [f"models/ES/weather/{ANCHOR}_{tech}_profile_s_75.nc"
            for tech in ("solar", "onwind", "offwind-ac", "offwind-dc")]


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def prepare(source_root, destination):
    source_root = source_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source_paths = [NETWORK, REGIONS, LOAD_RATIOS, WEATHER, DEMAND, CURVE,
                    *PROFILES]
    for relative in source_paths:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{destination.name}.",
                                     dir=destination.parent) as directory:
        staging = Path(directory)
        files = []
        reference_snapshots = None
        components = None
        for relative in source_paths:
            source = source_root / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == NETWORK:
                network = filter_network_by_countries(
                    pypsa.Network(source), ["ES"],
                    keep_interconnections=True, verbose=False)
                orphan_stores = network.stores.index[
                    ~network.stores.bus.isin(network.buses.index)]
                if len(orphan_stores):
                    network.remove("Store", orphan_stores)
                reference_hours = network.snapshots[
                    (network.snapshots.month == 7) & (network.snapshots.hour == 13)]
                if len(reference_hours) != 31:
                    raise ValueError("Expected all July 13:00 reference hours")
                network.set_snapshots(reference_hours)
                reference_snapshots = len(network.snapshots)
                components = {
                    "buses": len(network.buses),
                    "generators": len(network.generators),
                    "lines": len(network.lines),
                    "links": len(network.links),
                    "storage_units": len(network.storage_units),
                    "stores": len(network.stores),
                }
                network.export_to_netcdf(target)
            elif relative == WEATHER or relative in PROFILES:
                with xr.open_dataset(source) as dataset:
                    hour = np.datetime64(TIMESTAMP)
                    if hour not in dataset.time.values:
                        raise ValueError(f"{TIMESTAMP} missing from {relative}")
                    dataset.sel(time=[hour]).to_netcdf(target)
            else:
                shutil.copy2(source, target)
            files.append({
                "bytes": target.stat().st_size,
                "path": relative,
                "sha256": digest(target),
                "source": relative,
                "source_sha256": digest(source),
            })
        manifest = {
            "schema_version": 1,
            "case": {
                "analysis_mode": "td_seg_derate_iter_2",
                "country": "ES",
                "historical_event_index": 0,
                "historical_year": 2024,
                "load_growth_rate": 1.01,
                "max_temperature": 90,
                "network_ratio": 75,
                "storage_ratio": 0.8,
                "timestamp": TIMESTAMP,
            },
            "reference_snapshot_id": "ES-2030",
            "files": files,
            "network_components": components,
            "network_reference_snapshots": reference_snapshots,
            "purpose": "local prepared-input example; not a paper-result archive",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("data/examples/es_2030-07-24_13_2024-0"))
    options = parser.parse_args()
    print(prepare(options.source_root, options.output))

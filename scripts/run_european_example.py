#!/usr/bin/env python3
"""Validate and run one prepared Spanish heatwave hour in isolation."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_bundle(bundle):
    """Check the manifest, input checksums and selected weather/profile hour."""
    bundle = Path(bundle).resolve()
    with open(bundle / "manifest.json") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported example manifest version")

    case = manifest["case"]
    if (case["country"], case["network_ratio"], case["analysis_mode"]) != (
            "ES", 75, "td_seg_derate_iter_2"):
        raise ValueError("This runner supports the prepared ES s_75 two-update case")
    anchor = f"{case['historical_year']}_{case['historical_event_index']}"
    expected = {
        "data/EU/networks/base_s_75_elec.nc",
        "data/EU/networks/historical_p_max_pu_s_75.nc",
        "data/EU/regions_onshore_base_s_75.geojson",
        "data/EU/load_ratio_base_s_75.csv",
        f"models/ES/simu_data/future_weather_data_based_on_historical_hot_event_{anchor}.nc",
        f"models/ES/simu_data/future_demand_data_based_on_historical_hot_event_{anchor}.npy",
        "models/demand_curve/ES/ES_[2015, 2024]_demand_curve.npy",
    }
    expected.update(
        f"models/ES/weather/{anchor}_{tech}_profile_s_75.nc"
        for tech in ("solar", "onwind", "offwind-ac", "offwind-dc")
    )
    listed = [item["path"] for item in manifest["files"]]
    if len(listed) != len(set(listed)) or set(listed) != expected:
        raise ValueError("Bundle manifest does not list exactly the required inputs")

    for item in manifest["files"]:
        path = (bundle / item["path"]).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise FileNotFoundError(f"Missing bundle input: {item['path']}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Bundle checksum differs: {item['path']}")

    timestamp = np.datetime64(case["timestamp"])
    netcdf_paths = [
        bundle / "models/ES/simu_data" /
        f"future_weather_data_based_on_historical_hot_event_{anchor}.nc"
    ]
    netcdf_paths.extend(
        bundle / "models/ES/weather" / f"{anchor}_{tech}_profile_s_75.nc"
        for tech in ("solar", "onwind", "offwind-ac", "offwind-dc")
    )
    for path in netcdf_paths:
        with xr.open_dataset(path) as dataset:
            if timestamp not in dataset.time.values:
                raise ValueError(f"Selected hour missing from {path}")
    from utils.renewable_history import historical_availability
    historical_availability(
        bundle / "data/EU/networks/historical_p_max_pu_s_75.nc", case["timestamp"])
    return manifest


def example_settings(case):
    """Use the European entry point's ACOPF, conductor and weather defaults."""
    args = {
        "date": None,
        "BaseMVA": 100.0,
        "phase_factor": 3.0,
        "renewable_mode": True,
        "heatwave_mode": True,
        "storage_mode": True,
        "DLR_mode": False,
        "load_growth": True,
        "reactive_demand_ratio": 0.15,
        "reactive_gen_upper": 0.8,
        "reactive_gen_lower": -0.8,
        "voltage_upper": 1.05,
        "voltage_lower": 0.95,
        "phase_angle_upper": 30,
        "phase_angle_lower": -30,
        "max_temperature": case["max_temperature"],
        "parallel_mode": False,
        "load_growth_rate": case["load_growth_rate"],
        "storage_ratio": case["storage_ratio"],
    }
    conductor = {
        "diameter": 18.881e-3,
        "num_bundle": 4,
        "ref_temperature": 20.0,
        "max_temperature": case["max_temperature"],
        "resistance_ratio": 0.00429,
        "unit_resistance": 0.03e-3 * 4,
        "conductor_angle": 0.0,
        "elevation": None,
        "convective_correction": 0.8,
        "radiactive_correction": 0.8,
    }
    weather = {
        "wind_speed": None,
        "wind_height": 10,
        "wind_angle": 90.0,
        "air_density": 1.029,
        "air_viscosity": 2.043e-5,
        "air_conductivity": 0.02945,
        "air_temperature": None,
        "radiation_emissivity": 0.8,
        "solar_absorptivity": 0.8,
        "solar_heat_intensity": None,
    }
    return args, conductor, weather


def run_example(bundle, output_dir):
    """Copy the checked inputs, run the selected hour and save its result."""
    manifest = check_bundle(bundle)
    bundle = Path(bundle).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{output_dir.name}.",
                                     dir=output_dir.parent) as temp_dir:
        staging = Path(temp_dir)
        for item in manifest["files"]:
            source = bundle / item["path"]
            target = staging / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(bundle / "manifest.json", staging / "input_manifest.json")
        os.replace(staging, output_dir)

    os.environ["HEATANALYSIS_LOCAL_DATA"] = str(output_dir / "data")
    os.environ["HEATANALYSIS_MODELS"] = str(output_dir / "models")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    from TDOPF_eur import run_single_simulation

    case = manifest["case"]
    args, conductor, weather = example_settings(case)
    result = run_single_simulation((
        case["analysis_mode"], datetime.fromisoformat(case["timestamp"]),
        case["country"], manifest["network_components"]["buses"], args,
        conductor, weather, case["historical_year"],
        case["historical_event_index"],
    ))
    result_csv = output_dir / "results.csv"
    result.to_csv(result_csv, index=False)
    result_files = list((output_dir / "models/ES/ppc").rglob("*_results.npy"))
    if len(result_files) != 1:
        raise RuntimeError(f"Expected one result file, found {len(result_files)}")
    solver_status = int(result.iloc[0]["solver_status"])
    summary = {
        "case": case,
        "input_manifest_sha256": file_sha256(output_dir / "input_manifest.json"),
        "result_csv": str(result_csv),
        "result_npy": str(result_files[0]),
        "solver_status": solver_status,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if solver_status != 1:
        raise RuntimeError(f"Example solver_status is {solver_status}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    if options.check_only:
        checked = check_bundle(options.bundle)
        print(f"Checked {len(checked['files'])} files for "
              f"{checked['case']['timestamp']}")
    else:
        if options.output_dir is None:
            parser.error("--output-dir is required unless --check-only is set")
        print(json.dumps(run_example(options.bundle, options.output_dir), indent=2))

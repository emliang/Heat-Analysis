#!/usr/bin/env python3
"""Export compact plotted-data packages for Supplementary Figs. 5--14."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def profile_rows(arrays: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for scenario, values in arrays.items():
        for hour, value in enumerate(values):
            rows.append({"scenario": scenario, "hour": hour, "value": fmt(value)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument(
        "--figures",
        type=int,
        nargs="*",
        help="Optional Supplementary figure numbers to export.",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    manuscript = args.manuscript_root.resolve()
    sys.path[:0] = [str(project), str(pipeline / "code")]

    from data_config import RATIO, TK, WEATHER  # noqa: PLC0415
    from heatwave_weather_definition import (  # noqa: PLC0415
        FIGURE_SPECS,
        FUTURE_YEAR,
        HEATWAVE_RANK_WEIGHTS,
        MONTH,
        REGIONAL_BUS_COUNT,
        REGIONAL_RANDOM_SEED,
        REGIONAL_SAMPLE_COUNT,
        SCENARIO_LABELS,
        SELECTED_HEATWAVE_RANK,
        SNAPSHOT_HOUR,
        package_name,
    )
    from utils.heatwave_utils import (  # noqa: PLC0415
        bias_correction,
        find_heatwave_days,
        interpolate_3h_to_1h,
        temperal_trend_data,
    )
    from utils.network_process_utils import load_network_EU  # noqa: PLC0415
    from utils.plot_utils import cbar_lable_dic, vnom_dic  # noqa: PLC0415

    weather_paths = {
        "era5": Path(WEATHER) / "era5/era5_hourly_summer_2019_2024.nc",
        "historical_rcp45": Path(WEATHER) / "rcp45/rcp45_3hourly_summer_2019_2024.nc",
        "future_rcp45": Path(WEATHER) / "rcp45/rcp45_3hourly_summer_2025_2030.nc",
    }
    datasets = {name: xr.open_dataset(path) for name, path in weather_paths.items()}
    input_fingerprints = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path),
        }
        for name, path in weather_paths.items()
    }
    code_sources = [
        project / "scripts/main_heatwaves_generation.py",
        project / "utils/heatwave_utils.py",
        project / "utils/network_process_utils.py",
        pipeline / "code/heatwave_weather_definition.py",
        Path(__file__).resolve(),
    ]

    requested_figures = set(args.figures or [spec["figure"] for spec in FIGURE_SPECS])
    for spec in FIGURE_SPECS:
        if spec["figure"] not in requested_figures:
            continue
        country = spec["country"]
        variable = spec["variable"]
        historical_year = spec["historical_year"]
        package = pipeline / "supplementary_track" / package_name(spec)
        data_dir = package / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        network, regions = load_network_EU([country], RATIO)
        bounds = {
            "xmin": float(network.buses.x.min() - 0.25),
            "xmax": float(network.buses.x.max() + 0.25),
            "ymin": float(network.buses.y.min() - 0.25),
            "ymax": float(network.buses.y.max() + 0.25),
        }
        bbox = {"x": slice(bounds["xmin"], bounds["xmax"]), "y": slice(bounds["ymin"], bounds["ymax"])}
        historical_start = datetime(historical_year, MONTH, 1, 0)
        historical_end = datetime(historical_year, MONTH, 30, 23)
        future_start = datetime(FUTURE_YEAR, MONTH, 1, 0)
        future_end = datetime(FUTURE_YEAR, MONTH, 30, 23)
        era5 = datasets["era5"].sel(time=slice(historical_start, historical_end), **bbox)
        historical_rcp = interpolate_3h_to_1h(
            datasets["historical_rcp45"].sel(time=slice(historical_start, historical_end), **bbox)
        )
        future_rcp = interpolate_3h_to_1h(
            datasets["future_rcp45"].sel(time=slice(future_start, future_end), **bbox)
        )
        historical_indices, historical_dates = find_heatwave_days(
            era5, regions, weights=list(HEATWAVE_RANK_WEIGHTS)
        )
        future_indices, future_dates = find_heatwave_days(
            future_rcp, regions, weights=list(HEATWAVE_RANK_WEIGHTS)
        )
        historical_date = pd.Timestamp(
            historical_dates[historical_indices[SELECTED_HEATWAVE_RANK - 1]]
        ).replace(hour=SNAPSHOT_HOUR)
        future_date = pd.Timestamp(
            future_dates[future_indices[SELECTED_HEATWAVE_RANK - 1]]
        ).replace(hour=SNAPSHOT_HOUR)

        bus_his_hw, bus_his_ref, bus_fut_ref, bus_fut_hw = temperal_trend_data(
            variable,
            regions,
            era5,
            historical_rcp,
            future_rcp,
            historical_date,
            future_date,
        )
        national = {
            "historical_reference": bus_his_ref.mean(1),
            "historical_heatwave": bus_his_hw.mean(1),
            "future_reference": bus_fut_ref.mean(1),
            "future_heatwave": bus_fut_hw.mean(1),
        }
        national_path = data_dir / "national_hourly_profiles.csv"
        write_csv(national_path, ["scenario", "hour", "value"], profile_rows(national))

        spatial_path = None
        regional_path = None
        sampled_bus_rows: list[dict] = []
        if spec["full_delivery"]:
            historical_hw = era5.sel(time=historical_date, method="nearest")
            historical_ref = historical_rcp.sel(time=historical_date, method="nearest")
            future_ref = future_rcp.sel(time=future_date, method="nearest")
            future_hw = bias_correction(variable, historical_hw, historical_ref, future_ref, smooth_grid=2)
            spatial_sources = {
                "historical_reference": historical_ref[variable],
                "historical_heatwave": historical_hw[variable],
                "future_reference": future_ref[variable],
                "future_heatwave": future_hw[variable],
            }
            spatial_rows = []
            for scenario, field in spatial_sources.items():
                values = field.values - TK if variable == "temperature" else field.values
                for yi, y in enumerate(field.y.values):
                    for xi, x in enumerate(field.x.values):
                        spatial_rows.append(
                            {
                                "scenario": scenario,
                                "y_index": yi,
                                "x_index": xi,
                                "x": fmt(x),
                                "y": fmt(y),
                                "value": fmt(values[yi, xi]),
                            }
                        )
            spatial_path = data_dir / "spatial_snapshot_fields.csv.gz"
            write_csv(
                spatial_path,
                ["scenario", "y_index", "x_index", "x", "y", "value"],
                spatial_rows,
            )

            rng = np.random.RandomState(REGIONAL_RANDOM_SEED)
            sampled_positions = rng.choice(len(network.buses), REGIONAL_BUS_COUNT, replace=False)
            sampled_bus_ids = [str(network.buses.index[position]) for position in sampled_positions]
            regional_rows = []
            sample_dates = []
            baseline_by_bus = bus_fut_ref[:, sampled_positions]
            for region_number, (position, bus_id) in enumerate(zip(sampled_positions, sampled_bus_ids), start=1):
                sampled_bus_rows.append(
                    {
                        "region_number": region_number,
                        "bus_position": int(position),
                        "bus_id": bus_id,
                    }
                )
                for hour, value in enumerate(baseline_by_bus[:, region_number - 1]):
                    regional_rows.append(
                        {
                            "region_number": region_number,
                            "bus_id": bus_id,
                            "curve": "future_reference",
                            "sample_rank": 0,
                            "historical_seed_date": "",
                            "hour": hour,
                            "value": fmt(value),
                        }
                    )
            for sample_rank, heatwave_index in enumerate(
                historical_indices[:REGIONAL_SAMPLE_COUNT], start=1
            ):
                sample_date = pd.Timestamp(historical_dates[heatwave_index])
                sample_dates.append(str(sample_date.date()))
                _, _, _, sampled_future_hw = temperal_trend_data(
                    variable,
                    regions,
                    era5,
                    historical_rcp,
                    future_rcp,
                    sample_date,
                    future_date,
                )
                for region_number, bus_id in enumerate(sampled_bus_ids, start=1):
                    for hour, value in enumerate(sampled_future_hw[:, sampled_positions[region_number - 1]]):
                        regional_rows.append(
                            {
                                "region_number": region_number,
                                "bus_id": bus_id,
                                "curve": "future_heatwave",
                                "sample_rank": sample_rank,
                                "historical_seed_date": str(sample_date.date()),
                                "hour": hour,
                                "value": fmt(value),
                            }
                        )
            regional_path = data_dir / "sampled_regional_hourly_profiles.csv"
            write_csv(
                regional_path,
                ["region_number", "bus_id", "curve", "sample_rank", "historical_seed_date", "hour", "value"],
                regional_rows,
            )
            write_csv(
                data_dir / "sampled_region_index.csv",
                ["region_number", "bus_position", "bus_id"],
                sampled_bus_rows,
            )

        historical_artwork_root = manuscript / "figs/heatwave_generation"
        metadata = {
            "supplementary_figure": spec["figure"],
            "semantic_figure_id": f"heatwave_{country}_{variable}",
            "country": country,
            "variable": variable,
            "historical_year": historical_year,
            "future_year": FUTURE_YEAR,
            "month": MONTH,
            "historical_heatwave_date": str(historical_date),
            "future_reference_date": str(future_date),
            "heatwave_rank_weights": list(HEATWAVE_RANK_WEIGHTS),
            "selected_heatwave_rank_one_based": SELECTED_HEATWAVE_RANK,
            "snapshot_hour": SNAPSHOT_HOUR,
            "network_resolution_percent": int(RATIO),
            "bounds": bounds,
            "scenario_labels": SCENARIO_LABELS,
            "units": cbar_lable_dic[variable],
            "vmin": vnom_dic[variable][0],
            "vmax": vnom_dic[variable][1],
            "color_levels": vnom_dic[variable][2],
            "full_delivery": bool(spec["full_delivery"]),
            "regional_random_seed": REGIONAL_RANDOM_SEED if spec["full_delivery"] else None,
            "regional_bus_count": REGIONAL_BUS_COUNT if spec["full_delivery"] else 0,
            "regional_sample_count": REGIONAL_SAMPLE_COUNT if spec["full_delivery"] else 0,
            "historical_artwork_root": str(historical_artwork_root),
        }
        (data_dir / "plot_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance_rows = []
        for name, fingerprint in input_fingerprints.items():
            provenance_rows.append({"role": name, **fingerprint})
        for path in code_sources:
            provenance_rows.append(
                {
                    "role": "responsible_code",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": sha256(path),
                }
            )
        write_csv(
            data_dir / "input_provenance.csv",
            ["role", "path", "bytes", "mtime_ns", "sha256"],
            provenance_rows,
        )
        output_paths = [national_path, data_dir / "plot_metadata.json", data_dir / "input_provenance.csv"]
        if spatial_path:
            output_paths.extend([spatial_path, regional_path, data_dir / "sampled_region_index.csv"])
        manifest = {
            "status": "COMPACT_PLOTTED_DATA_EXPORTED",
            "simulation_rerun": False,
            "figure": spec["figure"],
            "counts": {
                "national_profile_rows": 96,
                "spatial_rows": len(spatial_rows) if spec["full_delivery"] else 0,
                "regional_profile_rows": len(regional_rows) if spec["full_delivery"] else 0,
                "sampled_regions": len(sampled_bus_rows),
            },
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in output_paths
            ],
        }
        (data_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"figure": spec["figure"], "package": str(package), "dates": [str(historical_date), str(future_date)]}))

    for dataset in datasets.values():
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export compact plotted-data packages for Supplementary Figs. 29--30."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record_path(root: Path, load: float, weather: dict, method: str, max_temp: int) -> Path:
    return root / (
        f"load_{load}_temp_{weather['air_temp']}_wind_{weather['wind_speed']}"
        f"_{method}_maxtemp_{max_temp}.npy"
    )


def extract_records(record_root: Path, methods, weather_scenarios, loads, max_temp):
    load_shedding_rows = []
    temperature_rows = []
    provenance_rows = []
    for load in loads:
        for weather in weather_scenarios:
            for method in methods:
                path = record_path(record_root, load, weather, method, max_temp)
                if not path.exists():
                    raise FileNotFoundError(path)
                result = np.load(path, allow_pickle=True).item()
                pdemand = np.asarray(result["PD"], dtype=float)
                violations = np.asarray(result["p_eq_vio"], dtype=float)
                temperatures = np.asarray(result["con_temp"], dtype=float).reshape(-1)
                load_shedding = float(violations.sum() / pdemand.sum() * 100.0)
                row_base = {
                    "load_ratio": fmt(load),
                    "weather_key": weather["key"],
                    "weather_label": weather["label"],
                    "air_temperature_c": weather["air_temp"],
                    "wind_speed_m_per_s": fmt(weather["wind_speed"]),
                    "method": method,
                }
                load_shedding_rows.append(
                    {
                        **row_base,
                        "load_shedding_percent": fmt(load_shedding),
                        "total_active_demand_pu": fmt(pdemand.sum()),
                        "total_active_violation_pu": fmt(violations.sum()),
                    }
                )
                for line_index, value in enumerate(temperatures):
                    temperature_rows.append(
                        {**row_base, "line_index": line_index, "line_temperature_c": fmt(value)}
                    )
                provenance_rows.append(
                    {
                        **row_base,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "solver_status": result.get("solver_status", ""),
                        "runtime_s": fmt(result["runtime"]),
                    }
                )
    return load_shedding_rows, temperature_rows, provenance_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    pipeline = args.pipeline_root.resolve()
    sys.path.insert(0, str(pipeline / "code"))
    from ieee_benchmark_definition import (  # noqa: PLC0415
        FIGURE_29_METHODS,
        FIGURE_30_METHODS,
        LINE_COUNT,
        MAX_CONDUCTOR_TEMPERATURE_C,
        METHOD_COLORS,
        METHOD_LABELS,
        WEATHER_SCENARIOS,
    )

    record_root = project / "models/IEEE30/record/heatflow_analysis"
    summary_csv = project / "models/IEEE30/30_heatflow_analysis.csv"
    source_artwork = project / "models/IEEE30/sensitivity_analysis"
    code_paths = [
        project / "vis/5.ieee_simu_vis.ipynb",
        project / "utils/plot_utils.py",
        pipeline / "code/ieee_benchmark_definition.py",
        Path(__file__).resolve(),
    ]

    package29 = pipeline / "supplementary_track/Supplementary_Figure_29_ieee_thermal_sensitivity"
    data29 = package29 / "data"
    data29.mkdir(parents=True, exist_ok=True)
    weather29 = (WEATHER_SCENARIOS[0], WEATHER_SCENARIOS[-1])
    shedding29, temperatures29, provenance29 = extract_records(
        record_root, FIGURE_29_METHODS, weather29, (0.9,), MAX_CONDUCTOR_TEMPERATURE_C
    )
    summary = pd.read_csv(summary_csv)
    runtime_rows = []
    for method in FIGURE_29_METHODS:
        selected = summary[summary.TDPF_solver == method].copy()
        for _, row in selected.sort_values(["load_ratio", "air_temp", "wind_speed"]).iterrows():
            runtime_rows.append(
                {
                    "method": method,
                    "load_ratio": fmt(row.load_ratio),
                    "air_temperature_c": int(row.air_temp),
                    "wind_speed_m_per_s": fmt(row.wind_speed),
                    "runtime_s": fmt(row.run_time),
                }
            )
    write_csv(data29 / "load_shedding.csv", list(shedding29[0]), shedding29)
    write_csv(data29 / "line_temperatures.csv", list(temperatures29[0]), temperatures29)
    write_csv(data29 / "runtime_observations.csv", list(runtime_rows[0]), runtime_rows)
    write_csv(data29 / "record_provenance.csv", list(provenance29[0]), provenance29)
    metadata29 = {
        "supplementary_figure": 29,
        "methods": list(FIGURE_29_METHODS),
        "method_labels": METHOD_LABELS,
        "method_colors": METHOD_COLORS,
        "weather_scenarios": list(weather29),
        "load_ratios": [0.9],
        "line_count": LINE_COUNT,
        "thermal_limit_c": MAX_CONDUCTOR_TEMPERATURE_C,
        "runtime_population": "all 12 summary rows per method: 3 load ratios x 4 weather settings",
        "runtime_standard_deviation_ddof": 0,
        "source_artwork_root": str(source_artwork),
    }
    (data29 / "plot_metadata.json").write_text(json.dumps(metadata29, indent=2, sort_keys=True) + "\n")

    package30 = pipeline / "supplementary_track/Supplementary_Figure_30_ieee_weather_sensitivity"
    data30 = package30 / "data"
    data30.mkdir(parents=True, exist_ok=True)
    shedding30, temperatures30, provenance30 = extract_records(
        record_root, FIGURE_30_METHODS, WEATHER_SCENARIOS, (0.9, 1.0), MAX_CONDUCTOR_TEMPERATURE_C
    )
    write_csv(data30 / "load_shedding.csv", list(shedding30[0]), shedding30)
    write_csv(data30 / "line_temperatures.csv", list(temperatures30[0]), temperatures30)
    write_csv(data30 / "record_provenance.csv", list(provenance30[0]), provenance30)
    metadata30 = {
        "supplementary_figure": 30,
        "methods": list(FIGURE_30_METHODS),
        "method_labels": METHOD_LABELS,
        "method_colors": METHOD_COLORS,
        "weather_scenarios": list(WEATHER_SCENARIOS),
        "load_ratios": [0.9, 1.0],
        "line_count": LINE_COUNT,
        "thermal_limit_c": MAX_CONDUCTOR_TEMPERATURE_C,
        "source_artwork_root": str(source_artwork),
    }
    (data30 / "plot_metadata.json").write_text(json.dumps(metadata30, indent=2, sort_keys=True) + "\n")

    for figure, data_dir, counts in (
        (29, data29, {"load_shedding_rows": len(shedding29), "line_temperature_rows": len(temperatures29), "runtime_rows": len(runtime_rows), "record_count": len(provenance29)}),
        (30, data30, {"load_shedding_rows": len(shedding30), "line_temperature_rows": len(temperatures30), "runtime_rows": 0, "record_count": len(provenance30)}),
    ):
        provenance_code = [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in code_paths
        ]
        manifest = {
            "supplementary_figure": figure,
            "status": "COMPACT_PLOTTED_DATA_EXPORTED",
            "simulation_rerun": False,
            "counts": counts,
            "summary_csv": {"path": str(summary_csv), "bytes": summary_csv.stat().st_size, "sha256": sha256(summary_csv)} if figure == 29 else None,
            "responsible_code": provenance_code,
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(data_dir.iterdir()) if path.is_file() and path.name != "manifest.json"
            ],
        }
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"figure": figure, "package": str(data_dir.parent), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

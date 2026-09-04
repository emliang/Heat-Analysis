#!/usr/bin/env python3
"""Build deterministic figure-level Source Data for main Fig. 3.

The exporter consumes the verified Spanish OPF-comparison Supplementary
package. It does not reopen result NPY files or rerun any OPF simulation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd


FIGURE_ID = "spain_opf_comparison"
PACKAGE_NAME = "Main_Figure_03_opf_method_comparison"
COMPONENT_NAME = "Supplementary_Figure_16_spain_opf_comparison"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


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


def deterministic_zip(source_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    args = parser.parse_args()

    pipeline = args.pipeline_root.resolve()
    component = pipeline / "supplementary_track" / COMPONENT_NAME / "data"
    package = pipeline / "main_track" / PACKAGE_NAME
    output = package / "source_data" / "unpacked"
    output.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file():
            path.unlink()

    scenario_path = component / "scenario_observations.csv"
    branch_path = component / "branch_observations.csv.gz"
    metadata_path = component / "plot_metadata.json"
    manifest_path = component / "manifest.json"
    provenance_path = component / "record_provenance.csv"
    for path in (
        scenario_path,
        branch_path,
        metadata_path,
        manifest_path,
        provenance_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    scenarios = pd.read_csv(scenario_path)
    branches = pd.read_csv(branch_path)
    metadata = json.loads(metadata_path.read_text())
    methods = list(metadata["methods"])
    expected_methods = [
        "base",
        "td_quad",
        "td_seg_derate_iter_2",
        "td_seg_derate_iter_10",
    ]
    if methods != expected_methods:
        raise ValueError(f"Unexpected method order: {methods}")

    scenario_identity = [
        "scenario_id",
        "future_heatwave_datetime",
        "historical_heatwave_date",
        "future_year",
        "future_hour",
    ]
    scenario_metrics = [
        f"{method}__{metric}"
        for method in methods
        for metric in ("load_shedding_percent", "runtime_s")
    ]
    branch_identity = ["scenario_id", "branch_index"]
    branch_metrics = [
        f"{method}__{metric}"
        for method in methods
        for metric in (
            "available_capacity_percent_of_nominal",
            "line_temperature_c",
        )
    ]
    missing_scenario = set(scenario_identity + scenario_metrics) - set(scenarios.columns)
    missing_branch = set(branch_identity + branch_metrics) - set(branches.columns)
    if missing_scenario or missing_branch:
        raise ValueError(
            f"Component schema mismatch: scenarios={sorted(missing_scenario)}, "
            f"branches={sorted(missing_branch)}"
        )

    scenarios = scenarios[scenario_identity + scenario_metrics].copy()
    branches = branches[branch_identity + branch_metrics].copy()
    scenarios.sort_values("scenario_id", inplace=True)
    branches.sort_values(["scenario_id", "branch_index"], inplace=True)
    if len(scenarios) != 480 or scenarios["scenario_id"].nunique() != 480:
        raise ValueError("Expected 480 unique heatwave scenarios")
    if scenarios["scenario_id"].duplicated().any():
        raise ValueError("Duplicate scenario rows")
    if branches.duplicated(["scenario_id", "branch_index"]).any():
        raise ValueError("Duplicate scenario-line rows")
    if branches["scenario_id"].nunique() != 480:
        raise ValueError("Scenario-line data do not cover all 480 scenarios")
    if branches["branch_index"].nunique() != 442 or len(branches) != 480 * 442:
        raise ValueError("Expected 442 represented lines in each of 480 scenarios")
    if set(branches["scenario_id"]) != set(scenarios["scenario_id"]):
        raise ValueError("Scenario and scenario-line populations differ")

    scenario_output = output / "Fig3cd_scenario_observations.csv"
    branch_output = output / "Fig3ab_scenario_line_observations.csv.gz"
    write_csv(scenario_output, scenarios)
    write_csv_gz(branch_output, branches)

    display = {
        "semantic_figure_id": FIGURE_ID,
        "main_figure": 3,
        "country_code": "ES",
        "methods": methods,
        "method_labels": metadata["method_labels"],
        "method_colors": metadata["method_colors"],
        "n_scenarios": 480,
        "n_internal_ac_lines": 442,
        "capacity_security_margin_percent": float(
            metadata["capacity_security_margin_percent"]
        ),
        "thermal_limit_c": float(metadata["thermal_limit_c"]),
        "runtime_standard_deviation_ddof": int(
            metadata["runtime_standard_deviation_ddof"]
        ),
        "observation_units": {
            "Fig3a": "one internal AC-line observation within each of 480 heatwave scenarios",
            "Fig3b": "one internal AC-line observation within each of 480 heatwave scenarios",
            "Fig3c": "one of 480 heatwave scenarios",
            "Fig3d": "one of 480 heatwave scenarios",
        },
        "metric_definitions": {
            "available_capacity_percent_of_nominal": (
                "weather-dependent available line capacity relative to nominal rating"
            ),
            "line_temperature_c": "maximum conductor temperature for each represented line",
            "load_shedding_percent": (
                "scenario demand-generation mismatch divided by total demand, in percent"
            ),
            "runtime_s": "per-scenario solver running time, in seconds",
        },
        "distribution_encoding": {
            "box": "first to third quartile",
            "median": "thin dim-grey line",
            "whiskers": (
                "most extreme observations within 1.5 times the interquartile range"
            ),
            "fliers": "not displayed; all observations remain in Source Data and density estimates",
            "mean": "dark-red point with numerical annotation",
            "violin": (
                "complete observation population summarized by a Gaussian kernel-density estimate"
            ),
            "runtime": "box-and-whisker distribution with arithmetic mean; no bar chart",
        },
        "panel_limits": {
            "a": [45.0, 104.0],
            "b": [15.0, 138.0],
            "c": [-0.015, 0.56],
            "d": [0.0, 45.0],
        },
        "artwork": {
            "width_mm": 180.0,
            "height_mm": 96.0,
            "font_family": "Arial",
            "font_size_pt_range": [5.0, 7.0],
            "colour_mode": "RGB",
        },
        "source_component": COMPONENT_NAME,
        "simulation_rerun": False,
    }
    rules_path = output / "Fig3_display_rules.json"
    rules_path.write_text(json.dumps(display, indent=2, sort_keys=True) + "\n")

    input_paths = [
        scenario_path,
        branch_path,
        metadata_path,
        manifest_path,
        provenance_path,
    ]
    provenance_rows = [
        {
            "component_file": str(path.relative_to(pipeline)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths
    ]
    write_csv(output / "Fig3_input_provenance.csv", pd.DataFrame(provenance_rows))

    readme = "Source Data for main Figure 3\n\n"
    readme += "Fig3ab_scenario_line_observations.csv.gz contains the complete "
    readme += "480-scenario by 442-line populations for available capacity and "
    readme += "line temperature in panels a and b.\n"
    readme += "Fig3cd_scenario_observations.csv contains the complete 480-scenario "
    readme += "populations for load shedding and running time in panels c and d.\n"
    readme += "All four OPF methods are compared over the same 480 scenarios. The "
    readme += "exporter consumes the verified Supplementary Figure 16 component "
    readme += "and does not rerun OPF or duplicate upstream NPY records.\n"
    (output / "README.txt").write_text(readme, encoding="utf-8")

    source_manifest = {
        "semantic_figure_id": FIGURE_ID,
        "main_figure": 3,
        "status": "FIGURE_LEVEL_SOURCE_DATA_EXPORTED",
        "simulation_rerun": False,
        "source_component": COMPONENT_NAME,
        "counts": {
            "methods": len(methods),
            "successful_scenarios": len(scenarios),
            "represented_internal_ac_lines": branches["branch_index"].nunique(),
            "scenario_line_rows": len(branches),
            "scenario_method_observations_panels_c_d": len(scenarios) * len(methods),
            "scenario_line_method_observations_panels_a_b": len(branches) * len(methods),
        },
        "files": [],
    }
    source_manifest_path = output / "manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )
    source_files = [path for path in output.iterdir() if path.is_file()]
    source_manifest["files"] = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(source_files, key=lambda item: item.name)
        if path.name != "manifest.json"
    ]
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )

    zip_path = package / "source_data" / "Source_Data_Fig_3.zip"
    deterministic_zip(output, zip_path)
    print(f"scenario_rows={len(scenarios)}")
    print(f"scenario_line_rows={len(branches)}")
    print(f"source_data_zip={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

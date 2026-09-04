#!/usr/bin/env python3
"""Build deterministic figure-level Source Data for main Fig. 5.

The exporter consumes the verified national-comparison component package. It
does not reopen result NPY files or rerun any power-flow simulation.
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


FIGURE_ID = "national_grid_comparison"
PACKAGE_NAME = "Main_Figure_05_national_grid_comparison"
COMPONENT_NAME = "Supplementary_Figure_15_national_comparison"
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

    scenario_path = component / "scenario_observations.csv"
    branch_path = component / "branch_observations.csv.gz"
    metadata_path = component / "plot_metadata.json"
    component_manifest_path = component / "manifest.json"
    provenance_path = component / "record_provenance.csv"
    for path in (
        scenario_path,
        branch_path,
        metadata_path,
        component_manifest_path,
        provenance_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    scenarios = pd.read_csv(scenario_path)
    branches = pd.read_csv(branch_path)
    metadata = json.loads(metadata_path.read_text())
    order = list(metadata["country_order"])
    expected_scenarios = int(metadata["n_scenarios_per_country"])

    scenario_columns = [
        "country_code",
        "country_name",
        "scenario_id",
        "future_heatwave_datetime",
        "historical_heatwave_date",
        "air_temperature_c",
        "load_shedding_percent",
    ]
    branch_columns = [
        "country_code",
        "scenario_id",
        "branch_index",
        "available_capacity_percent_of_nominal",
    ]
    missing_scenario = set(scenario_columns) - set(scenarios.columns)
    missing_branch = set(branch_columns) - set(branches.columns)
    if missing_scenario or missing_branch:
        raise ValueError(
            f"Component schema mismatch: scenarios={sorted(missing_scenario)}, "
            f"branches={sorted(missing_branch)}"
        )
    scenarios = scenarios[scenario_columns].copy()
    branches = branches[branch_columns].copy()
    scenarios["country_code"] = pd.Categorical(
        scenarios["country_code"], categories=order, ordered=True
    )
    branches["country_code"] = pd.Categorical(
        branches["country_code"], categories=order, ordered=True
    )
    scenarios.sort_values(["country_code", "scenario_id"], inplace=True)
    branches.sort_values(
        ["country_code", "scenario_id", "branch_index"], inplace=True
    )
    scenarios["country_code"] = scenarios["country_code"].astype(str)
    branches["country_code"] = branches["country_code"].astype(str)

    scenario_counts = scenarios.groupby("country_code", observed=True).size().to_dict()
    if scenario_counts != {country: expected_scenarios for country in order}:
        raise ValueError(f"Unexpected scenario counts: {scenario_counts}")
    n_branches = {country: int(value) for country, value in metadata["n_branches"].items()}
    expected_branch_counts = {
        country: expected_scenarios * n_branches[country] for country in order
    }
    branch_counts = branches.groupby("country_code", observed=True).size().to_dict()
    if branch_counts != expected_branch_counts:
        raise ValueError(f"Unexpected branch-observation counts: {branch_counts}")

    scenario_output = output / "Fig5ab_scenario_observations.csv"
    branch_output = output / "Fig5c_available_capacity_observations.csv.gz"
    write_csv(scenario_output, scenarios)
    write_csv_gz(branch_output, branches)

    display = {
        "semantic_figure_id": FIGURE_ID,
        "main_figure": 5,
        "country_order": order,
        "country_short_labels": {
            country: ("UK" if country == "GB" else country) for country in order
        },
        "country_names": metadata["country_names"],
        "country_colors": metadata["country_colors"],
        "n_scenarios_per_country": expected_scenarios,
        "n_branches": n_branches,
        "capacity_security_margin_percent": float(
            metadata["capacity_security_margin_percent"]
        ),
        "observation_units": {
            "Fig5a": "one projected hourly heatwave scenario within each country",
            "Fig5b": "one projected hourly heatwave scenario within each country",
            "Fig5c": "one internal AC-line observation within each scenario and country",
        },
        "metric_definitions": {
            "air_temperature_c": "national spatial mean air temperature",
            "load_shedding_percent": "scenario-level load-shedding ratio",
            "available_capacity_percent_of_nominal": (
                "weather-dependent available line capacity relative to nominal rating"
            ),
        },
        "distribution_encoding": {
            "box": "first to third quartile",
            "median": "thin dim-grey line",
            "whiskers": "most extreme observations within 1.5 times the interquartile range",
            "fliers": "not displayed",
            "mean": "dark-red point with numerical annotation",
            "violin": "complete observation population summarized by a Gaussian kernel-density estimate",
        },
        "artwork": {
            "width_mm": 180.0,
            "height_mm": 105.0,
            "font_family": "Arial",
            "font_size_pt_range": [5.0, 7.0],
            "colour_mode": "RGB",
        },
        "cross_country_alignment": metadata["cross_country_alignment"],
        "simulation_rerun": False,
    }
    display_path = output / "Fig5_display_rules.json"
    display_path.write_text(json.dumps(display, indent=2, sort_keys=True) + "\n")

    input_paths = [
        scenario_path,
        branch_path,
        metadata_path,
        component_manifest_path,
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
    write_csv(output / "Fig5_input_provenance.csv", pd.DataFrame(provenance_rows))

    readme = "Source Data for main Figure 5\n\n"
    readme += "Fig5ab_scenario_observations.csv contains the complete plotted "
    readme += "scenario populations for panels a and b.\n"
    readme += "Fig5c_available_capacity_observations.csv.gz contains the complete "
    readme += "scenario-by-internal-AC-line population plotted in panel c.\n"
    readme += "The country scenario sets are intentionally country-specific. The "
    readme += "exporter consumes the verified Supplementary Figure 15 component and "
    readme += "does not rerun simulations or duplicate upstream NPY records.\n"
    (output / "README.txt").write_text(readme, encoding="utf-8")

    manifest = {
        "semantic_figure_id": FIGURE_ID,
        "main_figure": 5,
        "status": "FIGURE_LEVEL_SOURCE_DATA_EXPORTED",
        "simulation_rerun": False,
        "source_component": COMPONENT_NAME,
        "counts": {
            "countries": len(order),
            "scenario_rows": len(scenarios),
            "available_capacity_rows": len(branches),
            "scenario_rows_per_country": scenario_counts,
            "available_capacity_rows_per_country": branch_counts,
        },
        "files": [],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    source_files = [path for path in output.iterdir() if path.is_file()]
    manifest["files"] = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(source_files, key=lambda item: item.name)
        if path.name != "manifest.json"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    zip_path = package / "source_data" / "Source_Data_Fig_5.zip"
    deterministic_zip(output, zip_path)
    print(f"scenario_rows={len(scenarios)}")
    print(f"available_capacity_rows={len(branches)}")
    print(f"source_data_zip={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

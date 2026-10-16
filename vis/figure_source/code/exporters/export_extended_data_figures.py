#!/usr/bin/env python3
"""Export compact figure-level Source Data for Extended Data Figs. 1 and 2.

The exporter consumes the verified Supplementary Fig. 17 and 26 packages. It
does not reopen raw NPY/NetCDF records or rerun weather or OPF simulations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ED1_COMPONENT = "Supplementary_Figure_17_spain_spatiotemporal_stress"
ED2_COMPONENT = "Supplementary_Figure_26_spain_cross_border_maps"
ED1_PACKAGE = "Extended_Data_Figure_01_spain_spatiotemporal"
ED2_PACKAGE = "Extended_Data_Figure_02_spain_cross_border_maps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
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


def provenance_rows(project: Path, paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path_relative_to_project": str(path.relative_to(project)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def output_rows(project: Path, directory: Path) -> list[dict[str, object]]:
    return provenance_rows(
        project,
        [path for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"],
    )


def tukey_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower = float(values[values >= lower_fence].min())
    upper = float(values[values <= upper_fence].max())
    return {
        "n_observations": len(values),
        "mean": float(values.mean()),
        "sd_ddof_0": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "maximum": float(values.max()),
        "lower_whisker": min(float(q1), lower),
        "upper_whisker": max(float(q3), upper),
        "n_values_outside_whiskers_not_drawn": int(
            np.sum((values < lower_fence) | (values > upper_fence))
        ),
    }


def export_ed1(project: Path, pipeline: Path) -> Path:
    component = pipeline / "supplementary_track" / ED1_COMPONENT / "data"
    package = pipeline / "extended_data_track" / ED1_PACKAGE
    output = package / "source_data" / "unpacked"
    clean_files(output)

    paths = {
        name: component / name
        for name in (
            "scenario_observations.csv",
            "scenario_branch_observations.csv.gz",
            "result_record_provenance.csv",
            "plot_metadata.json",
            "manifest.json",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    scenarios = pd.read_csv(paths["scenario_observations.csv"])
    branches = pd.read_csv(paths["scenario_branch_observations.csv.gz"])
    records = pd.read_csv(paths["result_record_provenance.csv"])
    metadata = json.loads(paths["plot_metadata.json"].read_text())
    status = records[["scenario_id", "solver_status"]]
    scenarios = scenarios.merge(status, on="scenario_id", validate="one_to_one")
    branches = branches.merge(status, on="scenario_id", validate="many_to_one")
    scenarios.sort_values("scenario_id", inplace=True)
    branches.sort_values(["scenario_id", "branch_index"], inplace=True)

    if len(scenarios) != 480 or scenarios["scenario_id"].nunique() != 480:
        raise ValueError("Extended Data Fig. 1 requires 480 unique scenarios")
    if not (branches.groupby("scenario_id").size() == 442).all():
        raise ValueError("Extended Data Fig. 1 requires 442 branch observations per scenario")
    if set(branches["scenario_id"]) != set(scenarios["scenario_id"]):
        raise ValueError("Scenario and scenario-branch populations differ")

    scenario_columns = [
        "scenario_id",
        "future_heatwave_datetime",
        "historical_heatwave_date",
        "future_year",
        "future_hour",
        "air_temperature_c",
        "wind_speed_m_per_s",
        "solar_irradiance_w_per_m2",
        "load_gw",
        "load_shedding_percent",
        "solver_status",
    ]
    branch_columns = [
        "scenario_id",
        "branch_index",
        "available_capacity_percent_of_nominal",
        "line_temperature_c",
        "solver_status",
    ]
    write_csv(output / "ExtendedDataFig1_scenario_observations.csv", scenarios[scenario_columns])
    write_csv_gz(
        output / "ExtendedDataFig1_scenario_branch_observations.csv.gz",
        branches[branch_columns],
    )

    groups: list[dict[str, object]] = []
    panel_specs = [
        ("a", "future_hour", metadata["future_hours"], "branch", "available_capacity_percent_of_nominal"),
        ("b", "future_year", metadata["future_years"], "branch", "available_capacity_percent_of_nominal"),
        ("c", "future_hour", metadata["future_hours"], "branch", "line_temperature_c"),
        ("d", "future_year", metadata["future_years"], "branch", "line_temperature_c"),
        ("e", "future_hour", metadata["future_hours"], "scenario", "load_shedding_percent"),
        ("f", "future_year", metadata["future_years"], "scenario", "load_shedding_percent"),
    ]
    for panel, group_field, group_values, level, value_field in panel_specs:
        for group_order, group_value in enumerate(group_values):
            selected_scenarios = scenarios.loc[
                scenarios[group_field] == group_value, "scenario_id"
            ]
            if level == "branch":
                values = branches.loc[
                    branches["scenario_id"].isin(selected_scenarios), value_field
                ].to_numpy(dtype=float)
            else:
                values = scenarios.loc[scenarios[group_field] == group_value, value_field].to_numpy(dtype=float)
            groups.append(
                {
                    "panel": panel,
                    "group_field": group_field,
                    "group_value": group_value,
                    "group_order": group_order,
                    "value_field": value_field,
                    "observation_unit": "scenario-branch" if level == "branch" else "scenario",
                    "n_scenarios": len(selected_scenarios),
                    **tukey_summary(values),
                }
            )
    write_csv(output / "ExtendedDataFig1_distribution_summary.csv", pd.DataFrame(groups))

    export_metadata = {
        "extended_data_figure": 1,
        "semantic_figure_id": "spain_spatiotemporal_stress",
        "simulation_rerun": False,
        "n_scenarios": 480,
        "n_branches": 442,
        "panels": {
            "a-d": "one observation per projected hourly heatwave scenario and represented AC branch",
            "e-f": "one observation per projected hourly heatwave scenario",
        },
        "distribution_encoding": {
            "box": "first to third quartile",
            "median": "thin dark-grey line",
            "whiskers": "most extreme observations within 1.5 times the interquartile range",
            "violin": "Gaussian kernel-density estimate of all displayed observations",
            "mean": "dark-red point with numerical annotation",
            "fliers": "not displayed; all observations remain in the violin and Source Data",
        },
        "capacity_security_margin_percent": 70.0,
        "thermal_limit_c": 90.0,
        "capacity_axis_label": "Available Capacity (%)",
        "future_hours": metadata["future_hours"],
        "future_years": metadata["future_years"],
        "hour_labels": metadata["hour_labels"],
        "colors": {
            "available_capacity": metadata["sequential_blue"],
            "line_temperature": metadata["sequential_red"],
            "load_shedding": metadata["sequential_orange"],
        },
    }
    (output / "ExtendedDataFig1_plot_metadata.json").write_text(
        json.dumps(export_metadata, indent=2, sort_keys=True) + "\n"
    )
    prov = provenance_rows(project, list(paths.values()))
    with (output / "input_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prov[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prov)
    readme = """Source Data for Extended Data Fig. 1 - Spanish spatiotemporal stress
=====================================================================

Scope and statistical units
---------------------------
The independent simulation unit is one projected hourly heatwave scenario.
All 480 exported scenarios have solver_status=1; no scenario is excluded. The
represented Spanish grid contains 442 AC branches.

Panels a and c are grouped by hour: n_s=120 scenarios, n_l=442 branches and
n_obs=53,040 scenario-branch observations per hour. Panels b and d are grouped
by year: n_s=96, n_l=442 and n_obs=42,432 per year. Panels e and f contain one
load-shedding value per scenario (n=120 per hour and n=96 per year).

Boxes span Q1-Q3, centre lines are medians, whiskers are the most extreme
observations within 1.5 times the interquartile range, violin widths show a
Gaussian kernel-density estimate of all observations, and dark-red points show
means. Individual values outside the whiskers are not drawn separately but
remain in the violin and Source Data. sd_ddof_0 is the population standard
deviation. No statistical hypothesis test is applied.

Files, columns and units
------------------------
- ExtendedDataFig1_scenario_observations.csv: scenario_id; projected and
  historical dates; future_year and future_hour; air_temperature_c (degrees C),
  wind_speed_m_per_s, solar_irradiance_w_per_m2, load_gw,
  load_shedding_percent and solver_status. One row per scenario; supports e-f.
- ExtendedDataFig1_scenario_branch_observations.csv.gz: scenario_id,
  branch_index, available_capacity_percent_of_nominal, line_temperature_c and
  solver_status. One row per scenario-branch observation; supports a-d.
- ExtendedDataFig1_distribution_summary.csv: panel/group identifiers,
  observation_unit, n_scenarios, n_observations, mean, sd_ddof_0, minimum, Q1,
  median, Q3, maximum, whiskers and the number of values outside the whiskers.
- ExtendedDataFig1_plot_metadata.json: plotting settings and statistical
  encoding. input_provenance.csv and manifest.json record input/output hashes.

Grouping and filters
--------------------
Panels a, c and e group all 480 scenarios by
future_hour; panels b, d and f group it by future_year. No simulation is rerun
and no additional value filter is applied. Spatial grid maps are displayed and
delivered separately as Extended Data Fig. 2.

Source record pattern
---------------------
The exporter consumes the verified Supplementary Fig. 17 compact component,
which traces the active records under models/ES/ppc/<future-date>/*_results.npy
and the corresponding weather record. Exact source paths and SHA-256 hashes are
listed in input_provenance.csv and manifest.json.

Export:
  conda run --no-capture-output -n HEAT python nature_final_materials/publication_pipeline/code/exporters/export_extended_data_figures.py --pipeline-root nature_final_materials/publication_pipeline --figure 1
Plot:
  conda run --no-capture-output -n HEAT python nature_final_materials/publication_pipeline/code/plotting/plot_extended_data_figures.py --pipeline-root nature_final_materials/publication_pipeline --figure 1
"""
    (output / "README.txt").write_text(readme, encoding="utf-8")
    manifest = {
        "extended_data_figure": 1,
        "semantic_figure_id": "spain_spatiotemporal_stress",
        "simulation_rerun": False,
        "source_component": ED1_COMPONENT,
        "source_component_files": prov,
        "export_script": provenance_rows(project, [Path(__file__).resolve()])[0],
        "plot_script": provenance_rows(
            project, [pipeline / "code/plotting/plot_extended_data_figures.py"]
        )[0],
        "effective_settings": {
            "solver_success_rule": "solver_status=1 for all 480 scenarios",
            "comparison_set_rule": "all 480 scenarios; 442 represented AC branches per scenario",
            "groups": {
                "hour": "120 scenarios and 53,040 scenario-branch observations per group",
                "year": "96 scenarios and 42,432 scenario-branch observations per group",
            },
            "outlier_display": "not drawn separately; retained in violin and Source Data",
            "simulation_rerun": False,
        },
        "row_counts": {
            "scenario_observations": len(scenarios),
            "scenario_branch_observations": len(branches),
            "distribution_groups": len(groups),
        },
        "output_files": output_rows(project, output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    zip_path = package / "source_data" / "Source_Data_Extended_Data_Fig_1.zip"
    deterministic_zip(output, zip_path)
    return package


def export_ed2(project: Path, pipeline: Path) -> Path:
    component = pipeline / "supplementary_track" / ED2_COMPONENT / "data"
    package = pipeline / "extended_data_track" / ED2_PACKAGE
    output = package / "source_data" / "unpacked"
    clean_files(output)
    names = [
        "map_snapshots.csv",
        "map_bus_observations.csv",
        "map_branch_observations.csv",
        "map_weather_fields.csv.gz",
        "represented_network_buses.csv",
        "represented_network_lines.csv",
        "record_provenance.csv",
        "plot_metadata.json",
        "manifest.json",
    ]
    paths = {name: component / name for name in names}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    snapshots = pd.read_csv(paths["map_snapshots.csv"])
    buses = pd.read_csv(paths["map_bus_observations.csv"])
    branches = pd.read_csv(paths["map_branch_observations.csv"])
    weather = pd.read_csv(paths["map_weather_fields.csv.gz"])
    networks_buses = pd.read_csv(paths["represented_network_buses.csv"])
    networks_lines = pd.read_csv(paths["represented_network_lines.csv"])
    records = pd.read_csv(paths["record_provenance.csv"])
    metadata = json.loads(paths["plot_metadata.json"].read_text())

    selection_basis = (
        "author-selected illustrative 13:00 snapshot; not an algorithmically selected peak"
    )
    snapshots["selection_basis"] = selection_basis

    if len(snapshots) != 9 or snapshots["snapshot_id"].nunique() != 9:
        raise ValueError("Extended Data Fig. 2 requires nine unique snapshots")
    if set(snapshots["snapshot_id"]) != set(records["snapshot_id"]):
        raise ValueError("Snapshot and provenance populations differ")
    if not buses.loc[buses["display_marker"].astype(bool), "configuration_id"].isin(
        metadata["configuration_ids"]
    ).all():
        raise ValueError("Unexpected load-shedding marker configuration")

    for name, frame in (
        ("map_snapshots.csv", snapshots),
        ("map_bus_observations.csv", buses),
        ("map_branch_observations.csv", branches),
        ("represented_network_buses.csv", networks_buses),
        ("represented_network_lines.csv", networks_lines),
        ("record_provenance.csv", records),
    ):
        write_csv(output / f"ExtendedDataFig2_{name}", frame)
    write_csv_gz(output / "ExtendedDataFig2_map_weather_fields.csv.gz", weather)

    config_labels = {"ES": "Spain", "ES-PT": "Spain+Portugal", "ES-FR": "Spain+France"}
    export_metadata = {
        "extended_data_figure": 2,
        "semantic_figure_id": "spain_cross_border_maps",
        "simulation_rerun": False,
        "configuration_ids": metadata["configuration_ids"],
        "configuration_labels": config_labels,
        "reference_country": metadata["reference_country"],
        "marker_scope": metadata["marker_scope"],
        "load_growth": metadata["load_growth"],
        "storage_state": metadata["storage_state"],
        "thermal_limit_c": metadata["thermal_limit_c"],
        "map_selection": (
            "Author-selected illustrative 13:00 snapshots are preserved for "
            "2026, 2028 and 2030; they are not algorithmically selected peaks."
        ),
        "map_display": metadata["map_display"],
        "colorbars": metadata["colorbars"],
        "observation_units": {
            "bus": "one represented bus in one selected spatial snapshot",
            "branch": "one represented AC branch in one selected spatial snapshot",
            "weather": "one gridded air-temperature cell in one selected spatial snapshot",
        },
    }
    (output / "ExtendedDataFig2_plot_metadata.json").write_text(
        json.dumps(export_metadata, indent=2, sort_keys=True) + "\n"
    )
    prov = provenance_rows(project, list(paths.values()))
    with (output / "input_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prov[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prov)
    readme = """Source Data for Extended Data Fig. 2 - Spanish cross-border maps
=================================================================

Scope and displayed units
-------------------------
The package contains nine author-selected illustrative 13:00 snapshots: Spain,
Spain+Portugal and Spain+France in 2026, 2028 and 2030. All use 1% annual load
growth and 80% initial storage state of charge. The snapshots are not
algorithmically selected peaks. All nine records have solver_status=1. No
statistical aggregation or hypothesis test is applied.

The plotted units are one represented bus, one represented AC branch and one
gridded weather cell within one selected spatial snapshot. Only buses in Spain,
the reference country, can receive a displayed load-shedding marker; the full
represented network for each configuration is retained.

Files, columns and units
------------------------
- ExtendedDataFig2_map_snapshots.csv: snapshot/configuration IDs, display order,
  future date/hour, historical event date, original artwork path and selection
  basis.
- ExtendedDataFig2_map_bus_observations.csv: snapshot/configuration/bus IDs,
  active_demand_mw, load_shedding_mw,
  load_shedding_percent_of_reference_demand and display_marker.
- ExtendedDataFig2_map_branch_observations.csv: snapshot/configuration/branch IDs
  and line_temperature_c.
- ExtendedDataFig2_map_weather_fields.csv.gz: snapshot/grid indices,
  longitude_plot, latitude_plot and air_temperature_c.
- ExtendedDataFig2_represented_network_buses.csv: bus ID, coordinates, country
  and is_reference_country for every represented configuration.
- ExtendedDataFig2_represented_network_lines.csv: branch ID, endpoints,
  endpoint countries, is_reference_internal and line geometry.
- ExtendedDataFig2_record_provenance.csv: exact OPF and weather record paths,
  byte counts, SHA-256 hashes and solver_status for each snapshot.
- ExtendedDataFig2_plot_metadata.json: load-growth, storage, temperature scales,
  marker scope, observation units and plotting settings.
- input_provenance.csv and manifest.json: package provenance and input/output
  hashes.

Filters and source record pattern
---------------------------------
No simulation is rerun and no value filter is applied. The exporter consumes
the verified Supplementary Fig. 26 compact component. Exact result/weather
records follow the paths in ExtendedDataFig2_record_provenance.csv; input files
and SHA-256 hashes are also recorded in input_provenance.csv and manifest.json.

Export:
  conda run --no-capture-output -n HEAT python nature_final_materials/publication_pipeline/code/exporters/export_extended_data_figures.py --pipeline-root nature_final_materials/publication_pipeline --figure 2
Plot:
  conda run --no-capture-output -n HEAT python nature_final_materials/publication_pipeline/code/plotting/plot_extended_data_figures.py --pipeline-root nature_final_materials/publication_pipeline --figure 2
"""
    (output / "README.txt").write_text(readme, encoding="utf-8")
    manifest = {
        "extended_data_figure": 2,
        "semantic_figure_id": "spain_cross_border_maps",
        "simulation_rerun": False,
        "source_component": ED2_COMPONENT,
        "source_component_files": prov,
        "export_script": provenance_rows(project, [Path(__file__).resolve()])[0],
        "plot_script": provenance_rows(
            project, [pipeline / "code/plotting/plot_extended_data_figures.py"]
        )[0],
        "effective_settings": {
            "solver_success_rule": "solver_status=1 for all nine displayed snapshots",
            "comparison_set_rule": "three author-selected 13:00 snapshots for each represented configuration",
            "load_growth": float(metadata["load_growth"]),
            "storage_state": float(metadata["storage_state"]),
            "reference_country": metadata["reference_country"],
            "marker_scope": metadata["marker_scope"],
            "simulation_rerun": False,
        },
        "row_counts": {
            "map_snapshots": len(snapshots),
            "map_bus_observations": len(buses),
            "map_branch_observations": len(branches),
            "map_weather_field_observations": len(weather),
            "represented_network_buses": len(networks_buses),
            "represented_network_lines": len(networks_lines),
        },
        "output_files": output_rows(project, output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    zip_path = package / "source_data" / "Source_Data_Extended_Data_Fig_2.zip"
    deterministic_zip(output, zip_path)
    return package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--figure", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()
    pipeline = args.pipeline_root.resolve()
    project = pipeline.parents[1]
    outputs = []
    if args.figure in {"1", "all"}:
        outputs.append(export_ed1(project, pipeline))
    if args.figure in {"2", "all"}:
        outputs.append(export_ed2(project, pipeline))
    print(json.dumps({"packages": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

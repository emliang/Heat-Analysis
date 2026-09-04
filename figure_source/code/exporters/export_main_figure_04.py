#!/usr/bin/env python3
"""Build compact figure-level Source Data for main Fig. 4.

The exporter reads the verified Spain operating-sensitivity Supplementary
package and selects only the observations displayed in main Fig. 4. It never
reruns an OPF, weather-generation, or demand simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


FIGURE_ID = "spain_operating_sensitivity"
UPSTREAM_PACKAGE = "Supplementary_Figure_18_spain_operating_sensitivity"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tukey_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def long_scenario_panel(
    frame: pd.DataFrame,
    cases: list[dict],
    *,
    value_suffix: str,
    id_key: str,
    label_key: str,
    order_key: str,
) -> pd.DataFrame:
    identifiers = [case[id_key] for case in cases]
    value_columns = [f"{identifier}__{value_suffix}" for identifier in identifiers]
    identity_columns = [
        "scenario_id",
        "future_heatwave_datetime",
        "historical_heatwave_date",
        "future_year",
        "future_hour",
    ]
    long = frame[identity_columns + value_columns].melt(
        id_vars=identity_columns,
        value_vars=value_columns,
        var_name="value_column",
        value_name=value_suffix,
    )
    long[id_key] = long["value_column"].str.removesuffix(f"__{value_suffix}")
    lookup = {case[id_key]: case for case in cases}
    long[order_key] = long[id_key].map(
        {identifier: index for index, identifier in enumerate(identifiers)}
    )
    long[label_key] = long[id_key].map(
        {identifier: lookup[identifier]["label"] for identifier in identifiers}
    )
    return long.drop(columns="value_column")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    pipeline = project_root / "nature_final_materials/publication_pipeline"
    upstream = pipeline / "supplementary_track" / UPSTREAM_PACKAGE / "data"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()

    upstream_metadata = json.loads((upstream / "plot_metadata.json").read_text())
    scenario = pd.read_csv(upstream / "ablation_scenario_observations.csv")
    branch = pd.read_csv(upstream / "ablation_branch_observations.csv.gz")
    load = pd.read_csv(upstream / "load_growth_observations.csv")
    storage = pd.read_csv(upstream / "storage_soc_observations.csv")
    methods = list(upstream_metadata["ablation_methods"])
    labels = upstream_metadata["ablation_labels"]
    colors = upstream_metadata["ablation_colors"]

    scenario_ids = set(scenario["scenario_id"])
    for name, frame in (("branch", branch), ("load", load), ("storage", storage)):
        if set(frame["scenario_id"]) != scenario_ids:
            raise ValueError(f"{name}: scenario set differs from ablation scenario set")
    if len(scenario_ids) != 480:
        raise ValueError(f"Expected 480 common scenarios, found {len(scenario_ids)}")
    if branch.duplicated(["scenario_id", "branch_index"]).any():
        raise ValueError("Duplicate scenario/branch observations")
    if branch["branch_index"].nunique() != 442:
        raise ValueError("Expected 442 represented branches")

    method_cases = [
        {"method_id": method, "label": labels[method], "color": colors[method]}
        for method in methods
    ]
    panel_a = long_scenario_panel(
        scenario,
        method_cases,
        value_suffix="load_shedding_percent",
        id_key="method_id",
        label_key="display_label",
        order_key="method_order",
    )
    panel_a.sort_values(["method_order", "scenario_id"], inplace=True)
    panel_a.to_csv(
        output_dir / "Fig4a_model_ablation_load_shedding.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )

    branch_identity = branch[["scenario_id", "branch_index"]].copy()
    panel_b = branch_identity
    for method in methods:
        panel_b[f"{method}__line_temperature_c"] = branch[
            f"{method}__line_temperature_c"
        ]
    panel_b.to_csv(
        output_dir / "Fig4b_model_ablation_line_temperature.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        lineterminator="\n",
        float_format="%.15g",
    )

    load_cases = list(upstream_metadata["load_growth_cases"])
    storage_cases = list(upstream_metadata["storage_cases"])
    panel_c = long_scenario_panel(
        load,
        load_cases,
        value_suffix="load_shedding_percent",
        id_key="case_id",
        label_key="display_label",
        order_key="case_order",
    )
    panel_d = long_scenario_panel(
        storage,
        storage_cases,
        value_suffix="load_shedding_percent",
        id_key="case_id",
        label_key="display_label",
        order_key="case_order",
    )
    for frame, cases, setting_key in (
        (panel_c, load_cases, "load_growth"),
        (panel_d, storage_cases, "storage_state"),
    ):
        frame["setting_value"] = frame["case_id"].map(
            {
                case["case_id"]: case[setting_key]
                for case in cases
            }
        )
        frame.sort_values(["future_year", "case_order", "scenario_id"], inplace=True)
    panel_c.to_csv(
        output_dir / "Fig4c_load_growth_load_shedding.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )
    panel_d.to_csv(
        output_dir / "Fig4d_storage_soc_load_shedding.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )

    alignment = scenario[
        [
            "scenario_id",
            "future_heatwave_datetime",
            "historical_heatwave_date",
            "future_year",
            "future_hour",
        ]
    ].copy()
    alignment.to_csv(
        output_dir / "Fig4_common_scenario_alignment.csv",
        index=False,
        lineterminator="\n",
    )

    summary_rows: list[dict] = []
    for panel, frame, group_columns, value_column, unit in (
        ("a", panel_a, ["method_id", "method_order", "display_label"], "load_shedding_percent", "heatwave scenario"),
        ("c", panel_c, ["future_year", "case_id", "case_order", "display_label"], "load_shedding_percent", "heatwave scenario"),
        ("d", panel_d, ["future_year", "case_id", "case_order", "display_label"], "load_shedding_percent", "heatwave scenario"),
    ):
        for keys, group in frame.groupby(group_columns, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"panel": panel, "observation_unit": unit}
            row.update(dict(zip(group_columns, keys, strict=True)))
            row.update(tukey_summary(group[value_column].to_numpy(dtype=float)))
            summary_rows.append(row)
    for method_order, method in enumerate(methods):
        values = branch[f"{method}__line_temperature_c"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "panel": "b",
                "observation_unit": "scenario-branch observation",
                "method_id": method,
                "method_order": method_order,
                "display_label": labels[method],
                **tukey_summary(values),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "Fig4_distribution_summary.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )

    y_max_a = max(0.9, float(panel_a["load_shedding_percent"].max()) * 1.04)
    metadata = {
        "figure_number": 4,
        "semantic_figure_id": FIGURE_ID,
        "country_code": "ES",
        "simulation_rerun": False,
        "common_success_scenarios": 480,
        "represented_branches": 442,
        "distribution_definition": {
            "mean": "arithmetic mean, shown as a dark-red point; numerical values are annotated in panels a and b",
            "box": "first to third quartile",
            "median": "dark-grey line",
            "whiskers": "most extreme observations within 1.5 times the interquartile range",
            "violin": "Gaussian kernel-density estimate of every observation in panels a and b",
            "outlier_points": "not drawn; all observations remain in the Source Data and panel a-b density estimates",
        },
        "panels": {
            "a": {
                "source_file": "Fig4a_model_ablation_load_shedding.csv",
                "observation_unit": "heatwave scenario",
                "method_ids": methods,
                "display_labels": [labels[method] for method in methods],
                "colors": [colors[method] for method in methods],
                "y_limits": [0.0, y_max_a],
            },
            "b": {
                "source_file": "Fig4b_model_ablation_line_temperature.csv.gz",
                "observation_unit": "scenario-branch observation",
                "method_ids": methods,
                "display_labels": [labels[method] for method in methods],
                "colors": [colors[method] for method in methods],
                "thermal_limit_c": float(upstream_metadata["base_thermal_limit_c"]),
                "y_limits": [15.0, 135.0],
            },
            "c": {
                "source_file": "Fig4c_load_growth_load_shedding.csv",
                "observation_unit": "heatwave scenario",
                "case_ids": [case["case_id"] for case in load_cases],
                "display_labels": [case["label"] for case in load_cases],
                "year_colors": upstream_metadata["sequential_orange"],
            },
            "d": {
                "source_file": "Fig4d_storage_soc_load_shedding.csv",
                "observation_unit": "heatwave scenario",
                "case_ids": [case["case_id"] for case in storage_cases],
                "display_labels": [case["label"] for case in storage_cases],
                "year_colors": upstream_metadata["sequential_blue"],
            },
        },
    }
    (output_dir / "Fig4_plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance_rows = []
    for name in (
        "ablation_scenario_observations.csv",
        "ablation_branch_observations.csv.gz",
        "load_growth_observations.csv",
        "storage_soc_observations.csv",
        "plot_metadata.json",
        "manifest.json",
    ):
        path = upstream / name
        provenance_rows.append(
            {
                "path_relative_to_project": str(path.relative_to(project_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(output_dir / "input_provenance.csv", provenance_rows)
    (output_dir / "README.txt").write_text(
        "Source Data for Fig. 4 - Operating sensitivity\n"
        "================================================\n\n"
        "Panel a contains one load-shedding observation per projected hourly\n"
        "heatwave scenario and model setting. Panel b contains line-temperature\n"
        "observations for every scenario and represented branch; the compressed\n"
        "wide table retains one row per scenario-branch pair. Panels c and d\n"
        "contain one load-shedding observation per scenario for each year and\n"
        "load-growth or initial-storage-state-of-charge setting. All direct\n"
        "comparisons use the same 480 scenarios.\n\n"
        "Boxes span the first to third quartile; dark-grey lines show medians;\n"
        "whiskers use Tukey's 1.5-IQR rule; dark-red points show arithmetic means.\n"
        "Outlier points are not drawn, but no observations are removed from the\n"
        "Source Data or panel a-b density estimates.\n\n"
        "The package is derived from the verified Spain operating-sensitivity\n"
        "Supplementary package. No simulation is rerun.\n",
        encoding="utf-8",
    )
    manifest = {
        "figure_id": FIGURE_ID,
        "figure_number": 4,
        "simulation_rerun": False,
        "panel_counts": {
            "a": {"methods": 5, "scenarios_per_method": 480, "rows": len(panel_a)},
            "b": {"methods": 5, "scenarios": 480, "branches": 442, "wide_rows": len(panel_b)},
            "c": {"settings": 3, "scenarios_per_setting": 480, "scenarios_per_year_and_setting": 96, "rows": len(panel_c)},
            "d": {"settings": 3, "scenarios_per_setting": 480, "scenarios_per_year_and_setting": 96, "rows": len(panel_d)},
        },
        "source_component_files": provenance_rows,
        "package_policy": {
            "one_source_data_file_per_figure": True,
            "raw_simulation_archive_bundled": False,
            "branch_table_compressed_wide": True,
            "reason_branch_data_retained": "panel b reports the full scenario-branch line-temperature distribution",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    deterministic_zip(output_dir, args.zip_path.resolve())
    print(json.dumps(manifest["panel_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

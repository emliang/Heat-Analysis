#!/usr/bin/env python3
"""Build compact figure-level Source Data for main Fig. 6.

The exporter reads the verified Spain and France cross-border Supplementary
packages. It selects the 1% load-growth comparison used by main Fig. 6 and
never reruns an OPF or weather simulation.
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


FIGURE_ID = "cross_border_load_shedding"
LOAD_GROWTH = 1.01
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
COMPONENTS = {
    "ES": "Supplementary_Figure_25_spain_cross_border_statistics",
    "FR": "Supplementary_Figure_27_france_cross_border_statistics",
}
OUTPUT_FILES = {
    "ES": "Fig6a_spain_load_shedding.csv",
    "FR": "Fig6b_france_load_shedding.csv",
}
DISPLAY_LABELS = {
    "ES": {
        "ES": "Spain",
        "ES-FR": "Spain↔France",
        "ES-PT": "Spain↔Portugal",
    },
    "FR": {
        "FR": "France",
        "FR-IT": "France↔Italy",
        "FR-ES": "France↔Spain",
        "FR-GB": "France↔UK",
    },
}
DISPLAY_LIMITS = {"ES": [0.0, 0.60], "FR": [0.0, 8.5]}


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
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    lower = float(values[values >= lower_fence].min())
    upper = float(values[values <= upper_fence].max())
    return {
        "n_scenarios": len(values),
        "mean_percent": float(values.mean()),
        "sd_percent_ddof_0": float(values.std(ddof=0)),
        "minimum_percent": float(values.min()),
        "q1_percent": float(q1),
        "median_percent": float(median),
        "q3_percent": float(q3),
        "maximum_percent": float(values.max()),
        "lower_whisker_percent": min(float(q1), lower),
        "upper_whisker_percent": max(float(q3), upper),
        "n_values_outside_whiskers_not_drawn": int(
            np.sum((values < lower_fence) | (values > upper_fence))
        ),
    }


def write_provenance(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    pipeline = project_root / "nature_final_materials/publication_pipeline"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()

    summary_rows: list[dict] = []
    alignment_rows: list[pd.DataFrame] = []
    provenance_rows: list[dict] = []
    panel_counts: dict[str, dict] = {}
    metadata: dict = {
        "figure_number": 6,
        "semantic_figure_id": FIGURE_ID,
        "load_growth": LOAD_GROWTH,
        "observation_unit": "one reference-country load-shedding ratio per aligned projected hourly heatwave scenario and interconnection setting",
        "distribution_definition": {
            "mean": "arithmetic mean, shown as a dark-red point and numerical annotation",
            "box": "first to third quartile",
            "median": "dark-grey line",
            "whiskers": "most extreme observations within 1.5 times the interquartile range",
            "violin": "Gaussian kernel-density estimate of all 480 scenario values",
            "outlier_points": "not drawn; all values remain included in the violin and Source Data",
        },
        "panels": {},
        "simulation_rerun": False,
    }

    component_root = pipeline / "supplementary_track"
    for panel, country in zip(("a", "b"), ("ES", "FR"), strict=True):
        data_dir = component_root / COMPONENTS[country] / "data"
        scenario_path = data_dir / "scenario_observations.csv"
        alignment_path = data_dir / "scenario_alignment.csv"
        metadata_path = data_dir / "plot_metadata.json"
        manifest_path = data_dir / "manifest.json"
        upstream_metadata = json.loads(metadata_path.read_text())

        scenarios = pd.read_csv(scenario_path)
        selected = scenarios[np.isclose(scenarios["load_growth"], LOAD_GROWTH)].copy()
        configuration_ids = list(upstream_metadata["configuration_ids"])
        expected_ids = set(selected.loc[selected.configuration_id == configuration_ids[0], "scenario_id"])
        if len(expected_ids) != 480:
            raise ValueError(f"{country}: expected 480 reference scenarios, found {len(expected_ids)}")
        for configuration_id in configuration_ids:
            subset_ids = set(selected.loc[selected.configuration_id == configuration_id, "scenario_id"])
            if subset_ids != expected_ids:
                raise ValueError(f"{country}: scenario set differs for {configuration_id}")
        if selected.duplicated(["configuration_id", "scenario_id"]).any():
            raise ValueError(f"{country}: duplicate configuration/scenario observations")

        selected["reference_country"] = country
        selected["display_label"] = selected["configuration_id"].map(DISPLAY_LABELS[country])
        selected.sort_values(["configuration_order", "scenario_id"], inplace=True)
        columns = [
            "reference_country",
            "configuration_id",
            "configuration_order",
            "display_label",
            "scenario_id",
            "future_heatwave_datetime",
            "historical_heatwave_date",
            "future_year",
            "future_hour",
            "reference_demand_mw",
            "reference_load_shedding_mw",
            "load_shedding_percent",
        ]
        selected[columns].to_csv(
            output_dir / OUTPUT_FILES[country],
            index=False,
            lineterminator="\n",
            float_format="%.15g",
        )

        alignment = pd.read_csv(alignment_path)
        alignment = alignment[np.isclose(alignment["load_growth"], LOAD_GROWTH)].copy()
        success_columns = [column for column in alignment if column.endswith("_successful")]
        if len(alignment) != 480 or not alignment[success_columns].all(axis=None):
            raise ValueError(f"{country}: 480-scenario alignment is invalid")
        alignment.insert(0, "reference_country", country)
        alignment_rows.append(
            alignment[
                ["reference_country", "scenario_id", "fut_heatwave_date", "his_heatwave_date"]
            ].rename(
                columns={
                    "fut_heatwave_date": "future_heatwave_datetime",
                    "his_heatwave_date": "historical_heatwave_date",
                }
            )
        )

        for order, configuration_id in enumerate(configuration_ids):
            values = selected.loc[
                selected.configuration_id == configuration_id, "load_shedding_percent"
            ].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "panel": panel,
                    "reference_country": country,
                    "configuration_id": configuration_id,
                    "configuration_order": order,
                    "display_label": DISPLAY_LABELS[country][configuration_id],
                    **tukey_summary(values),
                }
            )

        panel_counts[panel] = {
            "reference_country": country,
            "configurations": len(configuration_ids),
            "scenarios_per_configuration": 480,
            "observation_rows": len(selected),
        }
        metadata["panels"][panel] = {
            "reference_country": country,
            "configuration_ids": configuration_ids,
            "display_labels": [DISPLAY_LABELS[country][value] for value in configuration_ids],
            "colors": upstream_metadata["colors"],
            "y_limits_percent": DISPLAY_LIMITS[country],
            "source_file": OUTPUT_FILES[country],
        }
        for path in (scenario_path, alignment_path, metadata_path, manifest_path):
            provenance_rows.append(
                {
                    "reference_country": country,
                    "path_relative_to_project": str(path.relative_to(project_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    pd.concat(alignment_rows, ignore_index=True).to_csv(
        output_dir / "Fig6_common_scenario_alignment.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "Fig6_distribution_summary.csv",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )
    write_provenance(output_dir / "input_provenance.csv", provenance_rows)
    (output_dir / "Fig6_plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "Source Data for Fig. 6 - Cross-border load-shedding comparison\n"
        "=================================================================\n\n"
        "Panels a and b contain one load-shedding observation for each of 480\n"
        "aligned projected hourly heatwave scenarios in every displayed Spain\n"
        "or France interconnection setting at 1% annual load growth. Load shedding\n"
        "is measured for the reference country and normalised by its matched\n"
        "single-country demand. The alignment table records the same 480 scenarios\n"
        "used in each comparison. The summary table records the displayed box, whisker, median\n"
        "and mean statistics. Outlier points are not drawn, but no observations are\n"
        "removed from the Source Data or density estimates.\n\n"
        "The package is derived from the verified Supplementary cross-border data.\n"
        "No OPF, weather generation or other simulation is rerun.\n",
        encoding="utf-8",
    )
    manifest = {
        "figure_id": FIGURE_ID,
        "figure_number": 6,
        "simulation_rerun": False,
        "load_growth": LOAD_GROWTH,
        "panel_counts": panel_counts,
        "source_component_files": provenance_rows,
        "package_policy": {
            "one_source_data_file_per_figure": True,
            "raw_simulation_archive_bundled": False,
            "branch_observations_bundled": False,
            "reason_branch_data_omitted": "active main Fig. 6 contains only scenario-level load-shedding panels",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    deterministic_zip(output_dir, args.zip_path.resolve())
    print(json.dumps(panel_counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

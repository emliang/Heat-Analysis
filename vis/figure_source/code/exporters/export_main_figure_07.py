#!/usr/bin/env python3
"""Build compact figure-level Source Data for main Fig. 7.

The exporter consumes the verified Spanish heatwave/weather component package.
It does not reopen weather archives or rerun heatwave generation.
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


FIGURE_ID = "spain_heatwave_generation"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DISPLAYED_REGIONS = (1, 2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        binary_handle = path.open("wb")
        compressed_handle = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary_handle,
            mtime=0,
        )
        handle = io.TextIOWrapper(compressed_handle, newline="", encoding="utf-8")
    else:
        handle = path.open("w", newline="", encoding="utf-8")
    with handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file():
            path.unlink()

    component = (
        project
        / "nature_final_materials/publication_pipeline/supplementary_track/"
        "Supplementary_Figure_05_heatwave_ES_temperature/data"
    )
    national = read_rows(component / "national_hourly_profiles.csv")
    spatial = read_rows(component / "spatial_snapshot_fields.csv.gz")
    all_regional = read_rows(component / "sampled_regional_hourly_profiles.csv")
    all_region_index = read_rows(component / "sampled_region_index.csv")
    regional = [
        row for row in all_regional
        if int(row["region_number"]) in DISPLAYED_REGIONS
    ]
    region_index = [
        row for row in all_region_index
        if int(row["region_number"]) in DISPLAYED_REGIONS
    ]
    if len(national) != 96 or len(spatial) != 6528:
        raise ValueError("Unexpected national or spatial component shape")
    if len(regional) != 432 or len(region_index) != 3:
        raise ValueError("Unexpected displayed regional component shape")

    write_rows(output / "Fig7a_national_hourly_profiles.csv", national)
    write_rows(output / "Fig7b_spatial_snapshot_fields.csv.gz", spatial)
    write_rows(output / "Fig7c_sampled_regional_hourly_profiles.csv", regional)
    write_rows(output / "Fig7c_sampled_region_index.csv", region_index)

    component_metadata = json.loads((component / "plot_metadata.json").read_text())
    all_values = [float(row["value"]) for row in all_regional]
    display_rules = {
        "displayed_region_numbers": list(DISPLAYED_REGIONS),
        "regional_axis_min_c": min(all_values) * 0.9,
        "regional_axis_max_c": max(all_values) * 1.1,
        "regional_heatwave_samples": 5,
        "regional_random_seed": component_metadata["regional_random_seed"],
        "selected_heatwave_rank_one_based": component_metadata[
            "selected_heatwave_rank_one_based"
        ],
        "historical_heatwave_date": component_metadata["historical_heatwave_date"],
        "future_reference_date": component_metadata["future_reference_date"],
        "snapshot_hour": component_metadata["snapshot_hour"],
        "bounds": component_metadata["bounds"],
        "scenario_labels": component_metadata["scenario_labels"],
        "units": component_metadata["units"],
        "vmin": component_metadata["vmin"],
        "vmax": component_metadata["vmax"],
        "color_levels": component_metadata["color_levels"],
    }
    (output / "Fig7_display_rules.json").write_text(
        json.dumps(display_rules, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance_rows = read_rows(component / "input_provenance.csv")
    portable_provenance = []
    for row in provenance_rows:
        source_path = Path(row["path"]).resolve()
        try:
            relative_path = source_path.relative_to(project)
        except ValueError as exc:
            raise ValueError(
                f"Fig. 7 provenance path is outside the project root: {source_path}"
            ) from exc
        portable_provenance.append(
            {
                "role": row["role"],
                "path": (Path(project.name) / relative_path).as_posix(),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
    write_rows(output / "Fig7_input_provenance.csv", portable_provenance)

    readme = (
        "Source Data for Fig. 7 - Projected heatwave generation in Spain\n"
        "=================================================================\n\n"
        "Panel a contains the four displayed national spatial-mean hourly\n"
        "temperature profiles. Panel b contains the four displayed spatial\n"
        "temperature fields. Panel c contains the three displayed sampled bus\n"
        "regions, each with one future-reference profile and five projected\n"
        "heatwave profiles. Fig7_display_rules.json records the selected regions,\n"
        "dates, snapshot hour, colour range and original common regional axis.\n\n"
        "The exporter consumes the verified compact Spanish weather package. It\n"
        "does not copy NetCDF archives or rerun weather/heatwave generation.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    source_files = sorted(path for path in component.iterdir() if path.is_file())
    manifest = {
        "figure_id": FIGURE_ID,
        "figure_number": 7,
        "simulation_rerun": False,
        "source_component_package": {
            "path": str(component.relative_to(project)),
            "files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in source_files
            ],
        },
        "counts": {
            "national_profile_rows": len(national),
            "spatial_field_rows": len(spatial),
            "displayed_regional_rows": len(regional),
            "displayed_regions": len(region_index),
            "heatwave_samples_per_region": 5,
        },
        "package_policy": {
            "one_source_data_file_per_figure": True,
            "raw_weather_archives_bundled": False,
            "unplotted_regional_profiles_bundled": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    deterministic_zip(output, args.zip_path.resolve())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"source_data_zip={args.zip_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

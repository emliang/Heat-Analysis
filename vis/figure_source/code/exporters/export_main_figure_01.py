#!/usr/bin/env python3
"""Export compact parameters and provenance for Main Figure 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_zip(unpacked: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(unpacked.iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--panel-d-image", type=Path, required=True)
    parser.add_argument("--panel-d-pptx", type=Path, required=True)
    parser.add_argument("--unpacked", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    pipeline_root = args.pipeline_root.resolve()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(pipeline_root / "code"))
    from ieee_heat_balance_figure_definition import (  # noqa: PLC0415
        IEEE_CONDUCTOR,
        IEEE_WEATHER,
        PANEL_RULES,
    )
    from main_figure_01_panel_d_definition import (  # noqa: PLC0415
        extract_panel_d_records,
    )

    unpacked = args.unpacked.resolve()
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True)

    conductor_rows = [
        {"parameter": key, "value": value, "unit": {
            "diameter": "m",
            "ref_temperature": "degC",
            "max_temperature": "degC",
            "resistance_ratio": "per_degC",
            "unit_resistance": "ohm_per_m",
            "conductor_angle": "deg",
            "elevation": "m",
            "inom": "A",
            "num_bundle": "count",
            "conductor_name": "text",
        }[key]}
        for key, value in IEEE_CONDUCTOR.items()
    ]
    write_rows(unpacked / "Fig1abc_ieee_conductor_parameters.csv", conductor_rows)

    weather_units = {
        "wind_speed": "m_per_s",
        "wind_angle": "deg",
        "air_density": "kg_per_m3",
        "air_viscosity": "kg_per_m_s",
        "air_conductivity": "W_per_m_degC",
        "air_temperature": "degC",
        "radiation_emissivity": "dimensionless",
        "solar_absorptivity": "dimensionless",
        "solar_heat_intensity": "W_per_m2",
        "wind_height": "m",
    }
    weather_rows = []
    for key, value in IEEE_WEATHER.items():
        scalar = float(value[0]) if hasattr(value, "__len__") else value
        weather_rows.append(
            {"parameter": key, "value": scalar, "unit": weather_units[key]}
        )
    write_rows(unpacked / "Fig1abc_ieee_weather_parameters.csv", weather_rows)

    rule_rows = []
    for panel, rule in PANEL_RULES.items():
        row = {"panel": panel}
        row.update(rule)
        rule_rows.append(row)
    fieldnames = sorted({key for row in rule_rows for key in row})
    with (unpacked / "Fig1abc_generation_rules.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rule_rows)

    panel_d_records, panel_d_metadata = extract_panel_d_records(project_root)
    for filename, frame in panel_d_records.items():
        frame.to_csv(
            unpacked / filename,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
        )
    (unpacked / "Fig1d_plot_rules.json").write_text(
        json.dumps(panel_d_metadata["rules"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    panel_d_provenance = [
        {
            "role": role,
            "path": str(path.relative_to(project_root)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for role, path in panel_d_metadata["provenance_paths"].items()
    ]
    write_rows(unpacked / "Fig1d_input_provenance.csv", panel_d_provenance)

    provenance = [
        {
            "figure_panels": "Fig. 1a-c",
            "source": "scripts/4.test_solve_heat_balance.ipynb and utils/heat_flow_utils.py",
            "status": "deterministic calculations reproduced from compact parameters",
            "numeric_source_data": "generated in memory; dense matrices not duplicated",
        },
        {
            "figure_panels": "Fig. 1d",
            "source": "vis/1.Spain_grid_example.ipynb cells 9 and 15--19",
            "status": "compact plotted records and source-component generation verified; editable PowerPoint composition retained",
            "numeric_source_data": "Spanish network, temperature grid and eight selected-line segment records",
        },
    ]
    write_rows(unpacked / "Fig1_panel_provenance.csv", provenance)

    readme = """Main Figure 1 Source Data and provenance

Panels a-c are deterministic IEEE Std 738 heat-balance calculations. The
conductor parameters, weather parameters and axis-generation rules are stored
as compact CSV files. Dense mechanically generated matrices are reconstructed
in memory by the plotting code and are not duplicated in this ZIP.

Panel d uses the represented 75% clustered Spanish network and the selected
2026-07-29 14:00 heatwave weather field. The package stores the compact plotted
network, temperature grid and eight segment records for the 130.6-km selected
line. The source-component plotting script reconstructs the map, line inset and
both colour bars. The editable PowerPoint is retained for the final composition,
and the current manuscript PNG remains the display asset used in the assembled
figure.

No grid simulation is rerun by this package.
"""
    (unpacked / "README.txt").write_text(readme, encoding="utf-8")
    manifest = {
        "main_figure": 1,
        "semantic_figure_id": "heat_balance_and_spain_example",
        "panels_a_c": {
            "standard": "IEEE Std 738-2012",
            "calculation_source": "utils/heat_flow_utils.py",
            "simulation_rerun": False,
            "dense_matrices_stored": False,
        },
        "panel_d": {
            "source_pptx_sha256": sha256(args.panel_d_pptx.resolve()),
            "display_png_sha256": sha256(args.panel_d_image.resolve()),
            "exact_segment_values_verified": True,
            "segment_count": int(panel_d_metadata["rules"]["segment_count"]),
            "selected_line_id": panel_d_metadata["rules"]["selected_line_id"],
            "source_component_generation": (
                "code/plotting/plot_main_figure_01_panel_d_sources.py"
            ),
        },
        "project_root_label": project_root.name,
    }
    (unpacked / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    deterministic_zip(unpacked, args.zip.resolve())
    print(f"created {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

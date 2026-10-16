#!/usr/bin/env python3
"""Export compact deterministic rules for Supplementary Fig. 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.12g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-artwork", type=Path, required=True)
    parser.add_argument("--temperature-points", type=int, default=200)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from generator_derating_definition import (  # noqa: PLC0415
        CURVES,
        TEMPERATURE_GRID,
        displayed_mask,
        formula_values,
        temperature_values,
    )
    from utils.heat_flow_utils import generator_derating  # noqa: PLC0415

    temperature = temperature_values(args.temperature_points)
    empirical = formula_values(temperature)
    simulation = {
        curve["curve_id"]: np.asarray(
            [
                generator_derating(
                    curve["carrier_name"],
                    {"air_temperature": float(value)},
                )
                for value in temperature
            ],
            dtype=float,
        )
        for curve in CURVES
    }

    definition_rows = [dict(curve) for curve in CURVES]
    summary_rows: list[dict] = []
    for curve in CURVES:
        curve_id = curve["curve_id"]
        plotted_mask = displayed_mask(curve_id, temperature)
        difference = np.abs(empirical[curve_id] - simulation[curve_id])
        mismatch = plotted_mask & (difference > 1e-12)
        mismatch_temperatures = temperature[mismatch]
        summary_rows.append(
            {
                "curve_id": curve_id,
                "display_label": curve["display_label"],
                "evaluated_points_not_stored": len(temperature),
                "plotted_points_not_stored": int(plotted_mask.sum()),
                "mismatched_plotted_points": int(mismatch.sum()),
                "max_abs_difference_on_plotted_points": format_float(
                    difference[plotted_mask].max()
                ),
                "mismatch_temperature_min_c": format_float(
                    mismatch_temperatures.min() if mismatch_temperatures.size else None
                ),
                "mismatch_temperature_max_c": format_float(
                    mismatch_temperatures.max() if mismatch_temperatures.size else None
                ),
                "mismatches_within_stated_scope_t_ge_20c": int(
                    (mismatch & (temperature >= 20.0)).sum()
                ),
            }
        )

    definition_path = output_dir / "generator_derating_curve_definitions.csv"
    rules_path = output_dir / "generator_derating_generation_rules.csv"
    summary_path = output_dir / "generator_derating_validation_summary.csv"
    for legacy_name in (
        "generator_derating_source_data.csv",
        "generator_derating_active_model_audit.csv",
    ):
        (output_dir / legacy_name).unlink(missing_ok=True)

    write_csv(definition_path, list(definition_rows[0]), definition_rows)
    rule_rows = [
        {
            "axis": "air_temperature",
            "minimum": format_float(TEMPERATURE_GRID["minimum_c"]),
            "maximum": format_float(TEMPERATURE_GRID["maximum_c"]),
            "points": args.temperature_points,
            "spacing": "linear inclusive endpoints",
            "unit": "C",
        }
    ]
    write_csv(rules_path, list(rule_rows[0]), rule_rows)
    write_csv(summary_path, list(summary_rows[0]), summary_rows)

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Supplementary Fig. 3 - Generator derating compact rule package\n"
        "================================================================\n\n"
        "The plotted curve points are exact deterministic outputs of the four "
        "recorded formulas and the linear temperature-grid rule. They are "
        "generated in memory and are not stored as a data table.\n\n"
        "generator_derating_curve_definitions.csv records the formula and "
        "display condition for each curve. generator_derating_generation_rules.csv "
        "records the temperature range and point count. The four-row validation "
        "summary records the comparison with the active simulation function.\n\n"
        "The manuscript states that conventional-generator derating applies at "
        "ambient temperatures of at least 20 degrees Celsius. Within that range, "
        "the displayed curves agree with the active simulation function. The "
        "plotter reads this compact package and evaluates the shared "
        "deterministic formulas in memory; it does not rerun any simulation.\n\n"
        "Reproduction\n"
        "------------\n"
        "From the HeatAnalysis project root, regenerate the compact rule "
        "package from the responsible model definitions with:\n"
        "  conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/exporters/"
        "export_generator_derating.py --project-root . --output-dir "
        "<data-dir> --current-artwork <current-derating-curve.pdf>\n\n"
        "Regenerate the editable panel using only this compact package and "
        "the shared deterministic formula implementation with:\n"
        "  conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/plotting/"
        "plot_generator_derating.py --source-dir <data-dir> --pdf "
        "<output.pdf> --png <preview.png>\n",
        encoding="utf-8",
    )

    responsible_sources = [
        project_root / "utils/heat_flow_utils.py",
        project_root / "scripts/4.test_solve_heat_balance.ipynb",
        Path(__file__).resolve().parents[1] / "generator_derating_definition.py",
        args.current_artwork.resolve(),
    ]
    outputs = [definition_path, rules_path, summary_path, readme_path]
    manifest = {
        "semantic_figure_id": "generator_derating",
        "supplementary_figure": 3,
        "status": "COMPACT_RULE_PACKAGE_IMPLEMENTED_ORIGINAL_STYLE_REPRODUCTION",
        "simulation_rerun": False,
        "generated_curve_points_stored": False,
        "temperature_grid": {
            "minimum_c": TEMPERATURE_GRID["minimum_c"],
            "maximum_c": TEMPERATURE_GRID["maximum_c"],
            "points": args.temperature_points,
        },
        "responsible_sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in responsible_sources
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "known_discrepancy": {
            "description": (
                "The current notebook artwork uses uncapped empirical OCGT and "
                "CCGT formulas below 20 degrees Celsius, whereas the active "
                "simulation function caps all derating factors at 1."
            ),
            "within_stated_model_scope_t_ge_20c": False,
            "within_stated_model_scope_result_effect": False,
            "disposition": (
                "Retain the original 10-50 degree Celsius display range and "
                "uncapped notebook curves; the discrepancy is outside the stated "
                "model scope and has no simulation-result effect."
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "evaluated_points_not_stored": len(temperature),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

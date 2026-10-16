#!/usr/bin/env python3
"""Export compact provenance records for conceptual Main Figure 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slide_media(pptx: Path, slide_number: int) -> list[str]:
    rel_path = f"ppt/slides/_rels/slide{slide_number}.xml.rels"
    with zipfile.ZipFile(pptx) as archive:
        root = ET.fromstring(archive.read(rel_path))
    targets = []
    for relation in root.findall("r:Relationship", REL_NS):
        target = relation.attrib.get("Target", "")
        if "/ppt/media/" in target or "../media/" in target:
            targets.append(Path(target).name)
    return sorted(set(targets))


def write_package(
    pptx: Path, unpacked: Path, generated_dir: Path, scenario_path: Path
) -> None:
    unpacked.mkdir(parents=True, exist_ok=True)
    panels = [
        {
            "figure_panel": "Fig. 2a",
            "original_ppt_slide": 2,
            "editable_ppt_slide": 1,
            "role": "Conceptual heatwave-impact and grid-model framework",
            "data_status": "Conceptual diagram; no numerical Source Data",
            "asset_origin": "Author-provided editable PowerPoint with embedded illustrative assets",
        },
        {
            "figure_panel": "Fig. 2b",
            "original_ppt_slide": 3,
            "editable_ppt_slide": 2,
            "role": "Conceptual temperature-dependent AC-OPF workflow and illustrative output",
            "data_status": (
                "Method schematic with unchanged author-provided hourly snapshots and "
                "a regenerated illustrative OPF-result map"
            ),
            "asset_origin": (
                "Author-provided editable PowerPoint plus recorded OPF/weather inputs "
                "identified in Fig2_panel_b_scenario.json"
            ),
        },
    ]
    with (unpacked / "Fig2_panel_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(panels[0]))
        writer.writeheader()
        writer.writerows(panels)

    with zipfile.ZipFile(pptx) as archive:
        records = []
        for panel, original_slide, editable_slide in (
            ("Fig. 2a", 2, 1),
            ("Fig. 2b", 3, 2),
        ):
            for media_name in slide_media(pptx, editable_slide):
                payload = archive.read(f"ppt/media/{media_name}")
                records.append(
                    {
                        "figure_panel": panel,
                        "original_ppt_slide": original_slide,
                        "editable_ppt_slide": editable_slide,
                        "embedded_asset": media_name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    with (unpacked / "Fig2_embedded_asset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    generated_records = []
    for path in sorted(generated_dir.glob("panel_b_*")):
        if not path.is_file() or path.name == scenario_path.name:
            continue
        generated_records.append(
            {
                "figure_panel": "Fig. 2b",
                "generated_asset": path.name,
                "role": (
                    "illustrative OPF-result grid map"
                    if "grid_map" in path.name
                    else "line-temperature colourbar"
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with (unpacked / "Fig2_panel_b_generated_assets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(generated_records[0]))
        writer.writeheader()
        writer.writerows(generated_records)
    shutil.copy2(scenario_path, unpacked / "Fig2_panel_b_scenario.json")

    readme = """Main Figure 2 provenance package

Figure 2 is a conceptual and methodological display item. It is not generated
from a numerical plotting table, so no synthetic numerical Source Data are
provided. The editable PowerPoint is the authoritative assembled figure source.

Fig2_panel_provenance.csv records the role and evidence boundary of each panel.
Fig2_embedded_asset_manifest.csv records the embedded image assets used by the
selected PowerPoint slides. Fig2_panel_b_generated_assets.csv and
Fig2_panel_b_scenario.json record the exact map/colorbar assets and the existing
OPF/weather records from which the right-hand map was regenerated. The three
left-hand hourly snapshots are retained byte-for-byte from the author source.

No simulation is rerun by this package.
"""
    (unpacked / "README.txt").write_text(readme, encoding="utf-8")
    manifest = {
        "main_figure": 2,
        "semantic_figure_id": "heatwave_aware_opf_framework",
        "source_type": "author_provided_editable_powerpoint",
        "selected_original_source_slides": [2, 3],
        "editable_delivery_slides": [1, 2],
        "original_slide_1_included": False,
        "numerical_source_data_applicable": False,
        "simulation_rerun": False,
        "pptx_sha256": sha256(pptx),
        "embedded_asset_records": len(records),
        "panel_b_generated_asset_records": len(generated_records),
        "panel_b_scenario_sha256": sha256(scenario_path),
    }
    (unpacked / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def deterministic_zip(unpacked: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(unpacked.iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--unpacked", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    if args.unpacked.exists():
        shutil.rmtree(args.unpacked)
    write_package(
        args.pptx.resolve(),
        args.unpacked.resolve(),
        args.generated_dir.resolve(),
        args.scenario.resolve(),
    )
    deterministic_zip(args.unpacked.resolve(), args.zip.resolve())
    print(f"created {args.zip}")


if __name__ == "__main__":
    main()

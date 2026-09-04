#!/usr/bin/env python3
"""Export the compact plotted-data package for Supplementary Fig. 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


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


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-artwork", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data_config import LOCAL_DATA, RATIO  # noqa: PLC0415
    from spain_grid_definition import (  # noqa: PLC0415
        CATEGORY_ORDER,
        CATEGORY_PALETTE,
        CONVENTIONAL_CARRIERS,
        RENEWABLE_CARRIERS,
        STORAGE_CARRIERS,
        capacity_category,
    )
    from utils.network_process_utils import load_network_EU  # noqa: PLC0415

    network, _ = load_network_EU(["ES"], RATIO)
    network_file = Path(LOCAL_DATA) / f"EU/networks/base_s_{RATIO}_elec.nc"
    country_shapes_file = Path(LOCAL_DATA) / "EU/country_shapes.geojson"

    bus_rows = [
        {
            "bus_id": str(bus_id),
            "x": format_float(row.x),
            "y": format_float(row.y),
            "country": str(row.get("country", "")),
        }
        for bus_id, row in network.buses.sort_index().iterrows()
    ]

    line_rows = []
    for line_id, row in network.lines.sort_index().iterrows():
        bus0 = network.buses.loc[row.bus0]
        bus1 = network.buses.loc[row.bus1]
        line_rows.append(
            {
                "line_id": str(line_id),
                "bus0": str(row.bus0),
                "bus1": str(row.bus1),
                "x0": format_float(bus0.x),
                "y0": format_float(bus0.y),
                "x1": format_float(bus1.x),
                "y1": format_float(bus1.y),
                "s_nom_mva": format_float(row.s_nom),
            }
        )

    link_rows = []
    for link_id, row in network.links.sort_index().iterrows():
        bus0 = network.buses.loc[row.bus0]
        bus1 = network.buses.loc[row.bus1]
        link_rows.append(
            {
                "link_id": str(link_id),
                "bus0": str(row.bus0),
                "bus1": str(row.bus1),
                "x0": format_float(bus0.x),
                "y0": format_float(bus0.y),
                "x1": format_float(bus1.x),
                "y1": format_float(bus1.y),
                "p_nom_mw": format_float(max(float(row.get("p_nom", 0.0)), 0.0)),
            }
        )

    asset_rows: list[dict] = []
    for asset_id, row in network.generators.sort_index().iterrows():
        if float(row.p_nom) <= 0:
            continue
        bus = network.buses.loc[row.bus]
        asset_rows.append(
            {
                "component": "generator",
                "asset_id": str(asset_id),
                "bus_id": str(row.bus),
                "x": format_float(bus.x),
                "y": format_float(bus.y),
                "carrier": str(row.carrier),
                "category": capacity_category(str(row.carrier)),
                "capacity_value": float(row.p_nom),
                "capacity_basis": "p_nom_mw",
            }
        )
    for asset_id, row in network.storage_units.sort_index().iterrows():
        if float(row.p_nom) <= 0:
            continue
        bus = network.buses.loc[row.bus]
        asset_rows.append(
            {
                "component": "storage_unit",
                "asset_id": str(asset_id),
                "bus_id": str(row.bus),
                "x": format_float(bus.x),
                "y": format_float(bus.y),
                "carrier": str(row.carrier),
                "category": "Storage",
                "capacity_value": float(row.p_nom),
                "capacity_basis": "p_nom_mw",
            }
        )
    positive_stores = network.stores[network.stores.e_nom.fillna(0) > 0]
    for asset_id, row in positive_stores.sort_index().iterrows():
        bus = network.buses.loc[row.bus]
        asset_rows.append(
            {
                "component": "store",
                "asset_id": str(asset_id),
                "bus_id": str(row.bus),
                "x": format_float(bus.x),
                "y": format_float(bus.y),
                "carrier": str(row.carrier),
                "category": "Storage",
                "capacity_value": float(row.e_nom),
                "capacity_basis": "e_nom_mwh",
            }
        )
    asset_rows.sort(key=lambda row: (row["component"], row["asset_id"]))

    assets = pd.DataFrame(asset_rows)
    grouped = (
        assets.groupby(["bus_id", "x", "y", "category"], as_index=False)[
            "capacity_value"
        ]
        .sum()
        .pivot_table(
            index=["bus_id", "x", "y"],
            columns="category",
            values="capacity_value",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=list(CATEGORY_ORDER), fill_value=0.0)
        .reset_index()
    )
    grouped["total_capacity_mw"] = grouped[list(CATEGORY_ORDER)].sum(axis=1)
    grouped = grouped[grouped.total_capacity_mw > 0].sort_values("bus_id")
    capacity_rows = [
        {
            "bus_id": str(row.bus_id),
            "x": format_float(row.x),
            "y": format_float(row.y),
            **{category: format_float(getattr(row, category)) for category in CATEGORY_ORDER},
            "total_capacity_mw": format_float(row.total_capacity_mw),
        }
        for row in grouped.itertuples(index=False)
    ]

    # The submitted artwork uses the fallback boundary in the historical plot
    # cell. Recording it here removes the notebook-state dependency.
    boundary_padding_deg = 0.7
    boundaries = {
        "xmin": float(network.buses.x.min() - boundary_padding_deg),
        "xmax": float(network.buses.x.max() + boundary_padding_deg),
        "ymin": float(network.buses.y.min() - boundary_padding_deg),
        "ymax": float(network.buses.y.max() + boundary_padding_deg),
    }
    context = gpd.read_file(country_shapes_file)
    display_box = box(
        boundaries["xmin"], boundaries["ymin"], boundaries["xmax"], boundaries["ymax"]
    )
    geometry_rows = [
        {
            "geometry_id": str(index),
            "geometry_wkt": geometry.wkt,
        }
        for index, geometry in sorted(
            ((str(index), geometry) for index, geometry in context.geometry.items()),
            key=lambda item: item[0],
        )
        if geometry is not None and not geometry.is_empty and geometry.intersects(display_box)
    ]

    paths = {
        "buses": output_dir / "spain_grid_buses.csv",
        "lines": output_dir / "spain_grid_ac_lines.csv",
        "links": output_dir / "spain_grid_dc_links.csv",
        "assets": output_dir / "spain_grid_capacity_assets.csv",
        "capacity": output_dir / "spain_grid_capacity_by_bus.csv",
        "geometry": output_dir / "spain_grid_context_geometries.csv",
        "rules": output_dir / "spain_grid_plot_rules.json",
        "provenance": output_dir / "spain_grid_input_provenance.csv",
    }
    write_csv(paths["buses"], list(bus_rows[0]), bus_rows)
    write_csv(paths["lines"], list(line_rows[0]), line_rows)
    write_csv(paths["links"], list(link_rows[0]), link_rows)
    write_csv(paths["assets"], list(asset_rows[0]), asset_rows)
    write_csv(paths["capacity"], list(capacity_rows[0]), capacity_rows)
    write_csv(paths["geometry"], list(geometry_rows[0]), geometry_rows)

    rules = {
        "country_code": "ES",
        "network_resolution_percent": int(RATIO),
        "category_order": list(CATEGORY_ORDER),
        "category_palette": CATEGORY_PALETTE,
        "renewable_carriers": sorted(RENEWABLE_CARRIERS),
        "conventional_carriers": sorted(CONVENTIONAL_CARRIERS),
        "storage_carriers": sorted(STORAGE_CARRIERS),
        "unknown_generator_carrier_fallback": "Conventional",
        "bounds": boundaries,
        "boundary_padding_deg": boundary_padding_deg,
        "canvas_inches": [8.8, 7.6],
        "line_width_rule": "0.35 + 2.4 * sqrt(capacity_gw / 5.0)",
        "pie_radius_rule": "0.035 + 0.22 * sqrt(capacity_mw / max_capacity_mw)",
        "ac_line_color": "#059aa6",
        "dc_link_color": "#a0008b",
    }
    paths["rules"].write_text(
        json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    responsible_sources = [
        network_file,
        country_shapes_file,
        project_root / "utils/network_process_utils.py",
        project_root / "vis/1.Spain_grid_example.ipynb",
        Path(__file__).resolve().parents[1] / "spain_grid_definition.py",
        args.current_artwork.resolve(),
    ]
    provenance_rows = [
        {
            "role": role,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for role, path in zip(
            [
                "network_snapshot",
                "country_geometry",
                "network_loader",
                "historical_plot_owner",
                "shared_definition",
                "current_manuscript_artwork",
            ],
            responsible_sources,
        )
    ]
    write_csv(paths["provenance"], list(provenance_rows[0]), provenance_rows)

    readme = output_dir / "README.txt"
    readme.write_text(
        "Supplementary Fig. 1 - Spanish grid representation\n"
        "===================================================\n\n"
        "This compact package contains only the plotted Spanish bus locations, "
        "AC lines, DC links, generation/storage assets, bus-level capacity "
        "aggregates and simplified country-boundary geometries. It does not "
        "duplicate the complete PyPSA network or any OPF result.\n\n"
        "The current network has no stores with positive e_nom, so the displayed "
        "storage category contains storage-unit power capacity only. The exporter "
        "retains an explicit capacity_basis field to prevent accidental mixing of "
        "power and energy capacity in future snapshots.\n\n"
        "Rebuild compact package from the frozen public-grid snapshot:\n"
        "  conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/exporters/"
        "export_spain_grid_configuration.py --project-root . --output-dir "
        "<data-dir> --current-artwork <current-figure.pdf>\n\n"
        "Rebuild editable QA artwork from this compact package only:\n"
        "  conda run --no-capture-output -n HEAT python "
        "nature_final_materials/publication_pipeline/code/plotting/"
        "plot_spain_grid_configuration.py --source-dir <data-dir> --pdf "
        "<output.pdf> --png <preview.png>\n",
        encoding="utf-8",
    )

    output_paths = [*paths.values(), readme]
    manifest = {
        "semantic_figure_id": "spain_grid_configuration",
        "supplementary_figure": 1,
        "status": "COMPACT_PLOTTED_DATA_PACKAGE_IMPLEMENTED",
        "simulation_rerun": False,
        "network_resolution_percent": int(RATIO),
        "counts": {
            "buses": len(bus_rows),
            "ac_lines": len(line_rows),
            "dc_links": len(link_rows),
            "capacity_assets": len(asset_rows),
            "buses_with_capacity": len(capacity_rows),
            "positive_e_nom_stores": len(positive_stores),
            "context_geometries": len(geometry_rows),
        },
        "capacity_totals": {
            category: float(grouped[category].sum()) for category in CATEGORY_ORDER
        },
        "responsible_sources": provenance_rows,
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        ],
        "known_inconsistency": {
            "description": (
                "The historical notebook can combine storage_units.p_nom and "
                "stores.e_nom in one displayed storage category."
            ),
            "current_snapshot_effect": False,
            "evidence": "No store in the Spanish 75% network has positive e_nom.",
            "disposition": "Preserve current artwork; retain capacity_basis in exported data.",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

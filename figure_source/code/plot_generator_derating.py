#!/usr/bin/env python3
"""Reproduce Supplementary Fig. 3 from its deterministic rule definition."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generator_derating_definition import (  # noqa: PLC0415
        formula_values,
    )

    with (
        args.source_dir / "generator_derating_generation_rules.csv"
    ).open(newline="", encoding="utf-8") as handle:
        rules = list(csv.DictReader(handle))
    with (
        args.source_dir / "generator_derating_curve_definitions.csv"
    ).open(newline="", encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    if len(rules) != 1 or rules[0]["axis"] != "air_temperature":
        raise ValueError("Expected one air-temperature generation rule")
    rule = rules[0]
    temperature = np.linspace(
        float(rule["minimum"]),
        float(rule["maximum"]),
        int(rule["points"]),
    )
    values = formula_values(temperature)
    expected_curve_ids = {"ocgt", "ccgt", "nuclear", "copper_windings"}
    if {row["curve_id"] for row in curves} != expected_curve_ids:
        raise ValueError("Curve-definition table does not match the four figure curves")

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    for curve in curves:
        curve_id = curve["curve_id"]
        displayed = values[curve_id].copy()
        if curve["display_condition"] == "T > 40 C":
            displayed[temperature <= 40.0] = float("nan")
        elif curve["display_condition"] != "all generated temperatures":
            raise ValueError(
                f"Unsupported display condition: {curve['display_condition']}"
            )
        ax.plot(
            temperature,
            displayed,
            linewidth=2,
            label=curve["display_label"],
        )
    ax.plot(temperature, [1.0] * len(temperature), linewidth=1.5, linestyle=":", color="gray")
    ax.set_xlabel("Air Temperature (°C)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Derating Factor", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, loc="lower center", frameon=True, columnspacing=1.5)
    ax.set_xlim(float(rule["minimum"]), float(rule["maximum"]))
    ax.set_ylim(0.5, 1.2)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelsize=16,
    )
    ax.grid(linewidth=0.5, alpha=0.25)

    metadata = {
        "Title": "Generator derating under ambient temperature",
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_generator_derating.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(args.pdf, format="pdf", dpi=300, bbox_inches="tight", metadata=metadata)
    fig.savefig(
        args.png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reproduce the country-profile panels of Supplementary Fig. 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from plot_heatwave_weather_figure import (
    configure_matplotlib,
    metadata as pdf_metadata,
    national_line_styles,
    plot_national_table,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    table = pd.read_csv(source / "national_hourly_profiles.csv")
    metadata = json.loads((source / "plot_metadata.json").read_text())
    components = metadata["components"]
    for rules in components:
        country = rules["country"]
        country_table = table[table["country"] == country].drop(columns="country")
        filename = (
            f"heatwave_{rules['variable']}_{country}_"
            f"{rules['future_year']}_{rules['historical_year']}.pdf"
        )
        plot_national_table(
            country_table,
            output / filename,
            rules,
            show_legend=False,
            figsize=(9.0, 3.5),
        )

    legend_rules = components[0]
    styles = national_line_styles(legend_rules)
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=style,
            alpha=0.7 if "reference" in scenario else 1.0,
            linewidth=2.5,
        )
        for scenario, color, style in styles
    ]
    labels = [
        legend_rules["scenario_labels"][scenario]
        for scenario, _, _ in styles
    ]
    legend_fig = plt.figure(figsize=(10.788, 0.43))
    legend_fig.legend(
        handles=handles,
        labels=labels,
        ncol=4,
        loc="center",
        mode="expand",
        bbox_to_anchor=(0.01, 0.02, 0.98, 0.96),
        frameon=False,
        columnspacing=1.0,
        prop={"size": 15, "weight": "bold"},
    )
    legend_fig.savefig(
        output / "heatwave_temperature_profile_legend.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0433,
        metadata=pdf_metadata(),
    )
    plt.close(legend_fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

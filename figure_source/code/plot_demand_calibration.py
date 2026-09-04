#!/usr/bin/env python3
"""Reproduce the Supplementary Fig. 2 country panels from compact tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def read_parameter_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(
    fig,
    pdf: Path,
    png: Path,
    title: str,
    *,
    pad_inches: float = 0.1,
) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": title,
        "Author": "Liang et al.",
        "Creator": "publication_pipeline/code/plotting/plot_demand_calibration.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        pdf,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
        metadata=metadata,
    )
    fig.savefig(
        png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artwork-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demand_calibration_definition import (  # noqa: PLC0415
        PALETTE,
        mape,
        model_curve,
        rmse,
    )

    observations = pd.read_csv(
        args.source_dir / "demand_calibration_plotted_daily_observations.csv"
    )
    parameter_rows = read_parameter_rows(
        args.source_dir / "demand_calibration_model_parameters.csv"
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    sns.set_style("whitegrid")

    for row in parameter_rows:
        code = row["country_code"]
        data = observations[observations["country_code"] == code].copy()
        data.sort_values("date", inplace=True)
        parameters = {
            name: float(row[name])
            for name in (
                "Ph", "Pc", "Th", "Tc", "solar_gains", "wind_chill",
                "humidity_discomfort", "smoothing", "Pb", "alpha",
                "lower_blend", "upper_blend", "max_raw_var",
            )
        }
        weekday = data["day_type"] == "weekday"
        weekend = ~weekday
        bait = data["population_weighted_bait_c"].to_numpy(dtype=float)
        observed = data["observed_daily_demand_gw"].to_numpy(dtype=float)
        predicted = data["model_predicted_daily_demand_gw"].to_numpy(dtype=float)
        bait_grid = np.linspace(float(bait.min()), float(bait.max()), 1000)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            data.loc[weekday, "population_weighted_bait_c"],
            data.loc[weekday, "observed_daily_demand_gw"],
            marker="o", color=PALETTE["weekday_observed"], linewidth=0,
            alpha=1.0, s=10,
        )
        ax.scatter(
            data.loc[weekend, "population_weighted_bait_c"],
            data.loc[weekend, "observed_daily_demand_gw"],
            marker="s", color=PALETTE["weekend_observed"], linewidth=0,
            alpha=1.0, s=10,
        )
        ax.plot(
            bait_grid,
            model_curve(bait_grid, parameters, weekday=True),
            alpha=1.0, color=PALETTE["weekday_model"], linewidth=2,
        )
        ax.plot(
            bait_grid,
            model_curve(bait_grid, parameters, weekday=False),
            alpha=1.0, color=PALETTE["weekend_model"], linewidth=2,
        )
        ax.text(
            float(bait.min()),
            float(observed.min()),
            f"RMSE = {rmse(observed, predicted):.2f}\n"
            f"MAPE = {mape(observed, predicted) * 100:.2f}%",
            bbox={"facecolor": "lightgray", "alpha": 0.1, "pad": 10},
            fontsize=14,
        )
        ax.set_title(row["country_name"], fontsize=22, fontweight="bold")
        ax.set_ylabel("Daily Demand (GW)", fontsize=18, fontweight="bold")
        ax.set_xlabel("BAIT (°C)", fontsize=18, fontweight="bold")
        ax.tick_params(
            axis="both", which="both", top=False, bottom=True, left=True,
            right=False, labelsize=16,
        )
        ax.set_xticks([float(value) for value in row["x_ticks_c"].split(";")])
        minimum_tick = int(float(observed.min()) * 0.9 // 1)
        maximum_tick = int(float(observed.max()) * 1.1 // 1 + 1)
        ax.set_yticks(np.linspace(minimum_tick, maximum_tick, 10).astype(int))
        ax.set_ylim(float(observed.min()) * 0.9, float(observed.max()) * 1.1)
        ax.grid(linewidth=0.5, alpha=0.25)

        years = f"{row['start_year']}_{row['end_year']}"
        filename = f"{code}_demand_curve_{years}"
        save_figure(
            fig,
            args.artwork_dir / f"{filename}.pdf",
            args.preview_dir / f"{filename}.png",
            f"Demand calibration for {row['country_name']}",
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["weekday_observed"], markersize=8, linewidth=0),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["weekend_observed"], markersize=8, linewidth=0),
        Line2D([0], [0], color=PALETTE["weekday_model"], linewidth=2, alpha=0.8),
        Line2D([0], [0], color=PALETTE["weekend_model"], linewidth=2, alpha=0.8),
    ]
    labels = [
        "Weekday demand",
        "Weekend demand",
        "Weekday model",
        "Weekend model",
    ]
    legend_fig = plt.figure(figsize=(12, 0.36))
    legend_fig.legend(
        handles=handles,
        labels=labels,
        ncol=4,
        loc="center",
        mode="expand",
        bbox_to_anchor=(0.01, 0.02, 0.98, 0.96),
        prop={"size": 15, "weight": "bold"},
        frameon=False,
        columnspacing=1.5,
    )
    save_figure(
        legend_fig,
        args.artwork_dir / "legend.pdf",
        args.preview_dir / "legend.png",
        "Demand calibration legend",
        pad_inches=0.01,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

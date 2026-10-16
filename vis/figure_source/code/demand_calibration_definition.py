"""Shared deterministic definition for Supplementary Fig. 2 demand calibration."""

from __future__ import annotations

import numpy as np


COUNTRIES = (
    {"country_code": "ES", "country_name": "Spain", "start_year": 2015, "end_year": 2024, "x_ticks_c": (5, 10, 15, 20, 25)},
    {"country_code": "PT", "country_name": "Portugal", "start_year": 2015, "end_year": 2024, "x_ticks_c": (5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5)},
    {"country_code": "FR", "country_name": "France", "start_year": 2015, "end_year": 2024, "x_ticks_c": (0, 5, 10, 15, 20, 25)},
    {"country_code": "IT", "country_name": "Italy", "start_year": 2015, "end_year": 2024, "x_ticks_c": (0, 5, 10, 15, 20, 25)},
    {"country_code": "DE", "country_name": "Germany", "start_year": 2015, "end_year": 2024, "x_ticks_c": (-5, 0, 5, 10, 15, 20, 25)},
    {"country_code": "GB", "country_name": "UK", "start_year": 2015, "end_year": 2020, "x_ticks_c": (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20)},
    {"country_code": "BE", "country_name": "Belgium", "start_year": 2015, "end_year": 2024, "x_ticks_c": (0, 5, 10, 15, 20)},
    {"country_code": "NL", "country_name": "Netherlands", "start_year": 2015, "end_year": 2024, "x_ticks_c": (0, 5, 10, 15, 20, 25)},
)

PARAMETER_NAMES = (
    "Ph",
    "Pc",
    "Th",
    "Tc",
    "solar_gains",
    "wind_chill",
    "humidity_discomfort",
    "smoothing",
    "Pb",
    "alpha",
    "lower_blend",
    "upper_blend",
    "max_raw_var",
)

PALETTE = {
    "weekday_observed": "#fabeaf",
    "weekend_observed": "#b7c6e7",
    "weekday_model": "#e83947",
    "weekend_model": "#72a9d0",
}


def model_curve(
    bait_c: np.ndarray,
    parameters: dict[str, float],
    *,
    weekday: bool,
) -> np.ndarray:
    """Return the displayed national demand curve for a BAIT grid."""
    bait_c = np.asarray(bait_c, dtype=float)
    demand = (
        parameters["Pb"]
        + parameters["Ph"] * np.maximum(parameters["Th"] - bait_c, 0.0)
        + parameters["Pc"] * np.maximum(bait_c - parameters["Tc"], 0.0)
    )
    if weekday:
        demand = demand + parameters["alpha"]
    return demand


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def mape(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((observed - predicted) / observed)))

"""Deterministic definition used to generate Supplementary Fig. 4 in memory."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


AMBIENT_GRID = {
    "start_c": 25.0,
    "stop_exclusive_c": 51.0,
    "step_c": 0.5,
}

CURRENT_GRID = {
    "start_nominal_multiplier": 0.5,
    "stop_nominal_multiplier_exclusive": 1.1,
    "step_a_per_subconductor": 1.0,
}

WEATHER = {
    "wind_speed": 0.61,
    "wind_angle": np.array([90.0]),
    "air_density": 1.029,
    "air_viscosity": 2.043e-5,
    "air_conductivity": 0.02945,
    "air_temperature": 30.0,
    "radiation_emissivity": 0.8,
    "solar_absorptivity": 0.8,
    "solar_heat_intensity": 900.0,
    "wind_height": 50.0,
}

BASE_CONDUCTOR = {
    "diameter": 18.881e-3,
    "num_bundle": 4,
    "ref_temperature": 20.0,
    "max_temperature": 90.0,
    "resistance_ratio": 0.00429,
    "unit_resistance": 0.03e-3 * 4,
    "conductor_angle": 0.0,
    "elevation": 0.0,
    "inom": 2580.0 / 4.0,
}

MODEL_SPECS = (
    {
        "model_id": "individual",
        "panel": "a",
        "display_label": "Individual conductor modelling",
        "conductor": BASE_CONDUCTOR,
    },
    {
        "model_id": "corrected",
        "panel": "b",
        "display_label": "Corrected conductor modelling",
        "conductor": {
            **BASE_CONDUCTOR,
            "convective_correction": 0.8,
            "radiactive_correction": 0.8,
        },
    },
    {
        "model_id": "merged",
        "panel": "c",
        "display_label": "Merged conductor modelling",
        "conductor": {
            "diameter": 18.881e-3 * 2.5,
            "num_bundle": 1,
            "ref_temperature": 20.0,
            "max_temperature": 90.0,
            "resistance_ratio": 0.00429,
            "unit_resistance": 0.03e-3,
            "conductor_angle": 0.0,
            "elevation": 50.0,
            "inom": 2580.0,
        },
    },
)


def evaluate_grid(
    heat_balance_equation: Callable,
    conductor: dict,
    *,
    ambient_grid: dict = AMBIENT_GRID,
    current_grid: dict = CURRENT_GRID,
    weather_parameters: dict = WEATHER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate axes and temperatures without persisting the dense matrix."""
    ambient_temperatures = np.arange(
        ambient_grid["start_c"],
        ambient_grid["stop_exclusive_c"],
        ambient_grid["step_c"],
    )
    per_subconductor_currents = np.arange(
        conductor["inom"] * current_grid["start_nominal_multiplier"],
        conductor["inom"] * current_grid["stop_nominal_multiplier_exclusive"],
        current_grid["step_a_per_subconductor"],
    )
    total_currents = per_subconductor_currents * conductor["num_bundle"]
    temperature_columns = []
    for ambient_temperature in ambient_temperatures:
        weather = dict(weather_parameters)
        weather["wind_angle"] = np.asarray(
            weather_parameters["wind_angle"], dtype=float
        ).copy()
        weather["air_temperature"] = np.array([ambient_temperature])
        temperature_columns.append(
            np.asarray(
                heat_balance_equation(
                    per_subconductor_currents,
                    conductor.copy(),
                    weather,
                ),
                dtype=float,
            )
        )
    return (
        ambient_temperatures,
        total_currents,
        np.stack(temperature_columns, axis=1),
    )

"""Deterministic IEEE Std 738 inputs for Main Figure 1a-c."""

from __future__ import annotations

import numpy as np


IEEE_CONDUCTOR = {
    "conductor_name": "795 kcmil 26/7 Drake ACSR",
    "diameter": 28.1e-3,
    "num_bundle": 1,
    "ref_temperature": 25.0,
    "max_temperature": 90.0,
    "resistance_ratio": 0.00429,
    "unit_resistance": 7.283e-5,
    "conductor_angle": 0.0,
    "elevation": 100.0,
    "inom": 1000.0,
}

IEEE_WEATHER = {
    "wind_speed": 0.61,
    "wind_angle": np.array([90.0]),
    "air_density": 1.029,
    "air_viscosity": 2.043e-5,
    "air_conductivity": 0.02945,
    "air_temperature": 30.0,
    "radiation_emissivity": 0.8,
    "solar_absorptivity": 0.8,
    "solar_heat_intensity": 1000.0,
    "wind_height": 50.0,
}

PANEL_RULES = {
    "a": {
        "x_axis": "air_temperature_c",
        "x_start": 20.0,
        "x_stop_exclusive": 50.0,
        "x_step": 0.2,
        "y_axis": "conductor_current_a",
        "y_start": 0.0,
        "y_stop_exclusive": 1500.0,
        "y_step": 0.5,
        "quantity": "conductor_temperature_c",
    },
    "b": {
        "x_axis": "wind_speed_m_per_s",
        "x_start": 0.5,
        "x_stop_exclusive": 2.0,
        "x_step": 0.01,
        "y_axis": "conductor_wind_angle_deg",
        "y_start": 10.0,
        "y_stop_exclusive": 90.0,
        "y_step": 0.1,
        "quantity": "conductor_temperature_c",
        "fixed_current_a": 1000.0,
        "fixed_air_temperature_c": 30.0,
    },
    "c": {
        "x_axis": "air_temperature_c",
        "x_start": 20.0,
        "x_stop_exclusive": 50.0,
        "x_step": 1.0,
        "y_axis": "wind_speed_m_per_s",
        "y_start": 0.2,
        "y_stop_exclusive": 2.0,
        "y_step": 0.1,
        "quantity": "maximum_current_limit_a",
        "thermal_limit_c": 90.0,
    },
}


def axis(rule: dict, prefix: str) -> np.ndarray:
    return np.arange(
        rule[f"{prefix}_start"],
        rule[f"{prefix}_stop_exclusive"],
        rule[f"{prefix}_step"],
    )


def evaluate_panels(heat_balance_equation, maximum_allowable_current):
    conductor = dict(IEEE_CONDUCTOR)

    rule = PANEL_RULES["a"]
    air_a = axis(rule, "x")
    current_a = axis(rule, "y")
    columns = []
    for air_temperature in air_a:
        weather = dict(IEEE_WEATHER)
        weather["wind_angle"] = IEEE_WEATHER["wind_angle"].copy()
        weather["air_temperature"] = air_temperature
        columns.append(
            np.asarray(
                heat_balance_equation(current_a, conductor.copy(), weather),
                dtype=float,
            )
        )
    panel_a = (air_a, current_a, np.stack(columns, axis=1))

    rule = PANEL_RULES["b"]
    wind_b = axis(rule, "x")
    angle_b = axis(rule, "y")
    rows = []
    for wind_speed in wind_b:
        weather = dict(IEEE_WEATHER)
        weather["wind_angle"] = angle_b.copy()
        weather["wind_speed"] = wind_speed
        weather["air_temperature"] = rule["fixed_air_temperature_c"]
        rows.append(
            np.asarray(
                heat_balance_equation(
                    rule["fixed_current_a"], conductor.copy(), weather
                ),
                dtype=float,
            )
        )
    panel_b = (wind_b, angle_b, np.stack(rows, axis=1))

    rule = PANEL_RULES["c"]
    air_c = axis(rule, "x")
    wind_c = axis(rule, "y")
    columns = []
    for air_temperature in air_c:
        weather = dict(IEEE_WEATHER)
        weather["wind_angle"] = IEEE_WEATHER["wind_angle"].copy()
        weather["wind_speed"] = wind_c.copy()
        weather["air_temperature"] = air_temperature
        columns.append(
            np.asarray(
                maximum_allowable_current(conductor.copy(), weather),
                dtype=float,
            )
        )
    panel_c = (air_c, wind_c, np.stack(columns, axis=1))
    return {"a": panel_a, "b": panel_b, "c": panel_c}

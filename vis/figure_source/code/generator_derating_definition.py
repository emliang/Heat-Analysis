"""Deterministic curve definition used to generate Supplementary Fig. 3."""

from __future__ import annotations

import numpy as np


TEMPERATURE_GRID = {
    "minimum_c": 10.0,
    "maximum_c": 50.0,
    "points": 200,
}

CURVES = (
    {
        "curve_id": "ocgt",
        "display_label": "OCGT (Open Cycle Gas Turbine)",
        "carrier_name": "OCGT",
        "formula": "(-0.6854*T + 110)/100",
        "display_condition": "all generated temperatures",
    },
    {
        "curve_id": "ccgt",
        "display_label": "CCGT (Combined-Cycle Gas Turbine)",
        "carrier_name": "CCGT",
        "formula": "(-0.6854*T/2 + 105)/100",
        "display_condition": "all generated temperatures",
    },
    {
        "curve_id": "nuclear",
        "display_label": "Nuclear Generator",
        "carrier_name": "nuclear",
        "formula": "(101.3042 - 0.1387*T - 0.0010*T^2)/100",
        "display_condition": "all generated temperatures",
    },
    {
        "curve_id": "copper_windings",
        "display_label": "Copper Windings",
        "carrier_name": "default",
        "formula": "sqrt(((180-T)*(1+0.0039*(40-20)))/((180-40)*(1+0.0039*(T-20))))",
        "display_condition": "T > 40 C",
    },
)


def temperature_values(points: int | None = None) -> np.ndarray:
    """Return the original equally spaced display grid."""
    return np.linspace(
        TEMPERATURE_GRID["minimum_c"],
        TEMPERATURE_GRID["maximum_c"],
        points or TEMPERATURE_GRID["points"],
    )


def formula_values(temperature_c: np.ndarray) -> dict[str, np.ndarray]:
    """Return the empirical formulas used by the responsible notebook cell."""
    return {
        "ocgt": (-0.6854 * temperature_c + 110.0) / 100.0,
        "ccgt": (-0.6854 * temperature_c / 2.0 + 105.0) / 100.0,
        "nuclear": (
            101.3042
            - 0.1387 * temperature_c
            - 0.0010 * temperature_c**2
        ) / 100.0,
        "copper_windings": np.sqrt(
            ((180.0 - temperature_c) * (1.0 + 0.0039 * (40.0 - 20.0)))
            / (
                (180.0 - 40.0)
                * (1.0 + 0.0039 * (temperature_c - 20.0))
            )
        ),
    }


def displayed_mask(curve_id: str, temperature_c: np.ndarray) -> np.ndarray:
    """Return the mask used by the current artwork for one curve."""
    if curve_id == "copper_windings":
        return temperature_c > 40.0
    return np.ones_like(temperature_c, dtype=bool)

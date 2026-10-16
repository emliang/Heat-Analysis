"""Frozen plotting definitions for Supplementary Figs. 29--30."""

from __future__ import annotations


METHOD_LABELS = {
    "base": "AC-OPF",
    "td_quad": "Quad-OPF",
    "td_derate_iter_2": "Iter-OPF",
    "td_derate_iter_10": "TD-OPF",
    "base_derate": "w/o thermal",
    "td_iter_2": "w/o derating",
    "base_acsc": "AC-SC-OPF",
    "base_dcsc": "DC-SC-OPF",
    "base_fixsc": "Fixed-SC-OPF",
}

METHOD_COLORS = {
    "base": "#72A9D0",
    "td_quad": "#8EC6C2",
    "td_derate_iter_2": "#F5B378",
    "td_derate_iter_10": "#E83947",
    "base_derate": "#E86AAC",
    "td_iter_2": "#7952B3",
    "base_acsc": "#2E8B57",
    "base_dcsc": "#A3D5DD",
    "base_fixsc": "#78CC6B",
}

FIGURE_29_METHODS = (
    "base", "td_quad", "td_derate_iter_2", "td_derate_iter_10",
    "base_derate", "td_iter_2", "base_acsc", "base_dcsc", "base_fixsc",
)
FIGURE_30_METHODS = ("base", "td_derate_iter_2", "base_acsc", "base_fixsc")

WEATHER_SCENARIOS = (
    {"key": "mild", "label": "Mild Weather", "wind_speed": 0.61, "air_temp": 25},
    {"key": "low_wind", "label": "Low-wind Weather", "wind_speed": 0.10, "air_temp": 25},
    {"key": "high_temperature", "label": "High-temp. Weather", "wind_speed": 0.61, "air_temp": 45},
    {"key": "extreme", "label": "Extreme Weather", "wind_speed": 0.10, "air_temp": 45},
)

MAX_CONDUCTOR_TEMPERATURE_C = 90
LINE_COUNT = 41


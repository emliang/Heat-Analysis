"""Frozen definitions for Supplementary Figs. 5--14."""

from __future__ import annotations


FIGURE_SPECS = (
    {"figure": 5, "country": "ES", "variable": "temperature", "historical_year": 2022, "full_delivery": True},
    {"figure": 6, "country": "ES", "variable": "influx", "historical_year": 2022, "full_delivery": True},
    {"figure": 7, "country": "ES", "variable": "wnd10m", "historical_year": 2022, "full_delivery": True},
    {"figure": 8, "country": "IT", "variable": "temperature", "historical_year": 2019, "full_delivery": False},
    {"figure": 9, "country": "FR", "variable": "temperature", "historical_year": 2022, "full_delivery": False},
    {"figure": 10, "country": "PT", "variable": "temperature", "historical_year": 2022, "full_delivery": False},
    {"figure": 11, "country": "DE", "variable": "temperature", "historical_year": 2022, "full_delivery": False},
    {"figure": 12, "country": "GB", "variable": "temperature", "historical_year": 2019, "full_delivery": False},
    {"figure": 13, "country": "BE", "variable": "temperature", "historical_year": 2022, "full_delivery": False},
    {"figure": 14, "country": "NL", "variable": "temperature", "historical_year": 2022, "full_delivery": False},
)

FUTURE_YEAR = 2030
MONTH = 7
HEATWAVE_RANK_WEIGHTS = (0.9, 0.1)
SELECTED_HEATWAVE_RANK = 1
SNAPSHOT_HOUR = 14
REGIONAL_SAMPLE_COUNT = 5
REGIONAL_BUS_COUNT = 9
REGIONAL_RANDOM_SEED = 2026
SCENARIO_ORDER = (
    "historical_reference",
    "historical_heatwave",
    "future_reference",
    "future_heatwave",
)
SCENARIO_LABELS = {
    "historical_reference": "Historical reference",
    "historical_heatwave": "Historical heatwave",
    "future_reference": "Future reference",
    "future_heatwave": "Future heatwave",
}


def package_name(spec: dict) -> str:
    return (
        f"Supplementary_Figure_{spec['figure']:02d}_"
        f"heatwave_{spec['country']}_{spec['variable']}"
    )


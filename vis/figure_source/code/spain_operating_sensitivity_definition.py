"""Frozen data and plotting definitions for Supplementary Fig. 18."""

from __future__ import annotations


SUPPLEMENTARY_FIGURE = 18
COUNTRY_CODE = "ES"
N_BUSES = 281
N_BRANCHES = 442
N_SCENARIOS = 480
SCENARIOS_PER_YEAR = 96
BASE_STORAGE_STATE = 0.8
BASE_LOAD_GROWTH = 1.01
BASE_THERMAL_LIMIT_C = 90
CAPACITY_SECURITY_MARGIN_PERCENT = 70.0
STANDARD_DEVIATION_DDOF = 0

SCENARIO_KEY_COLUMNS = ("fut_heatwave_date", "his_heatwave_date")
SCENARIO_CONTEXT_COLUMNS = (
    "fut_heatwave_year",
    "fut_heatwave_month",
    "fut_heatwave_day",
    "fut_heatwave_hour",
    "air_temp",
    "wind_speed",
    "solar_radia",
)

ABLATION_METHODS = (
    "td_seg_derate_iter_2",
    "base_seg_derate",
    "td_derate_iter_2",
    "td_seg_iter_2",
    "base_fixsc",
)
ABLATION_LABELS = {
    "td_seg_derate_iter_2": "Iter-OPF",
    "base_seg_derate": "w/o thermal",
    "td_derate_iter_2": "w/o segment",
    "td_seg_iter_2": "w/o derating",
    "base_fixsc": "SC-OPF",
}
ABLATION_COLORS = {
    "td_seg_derate_iter_2": "#F5B378",
    "base_seg_derate": "#FB7185",
    "td_derate_iter_2": "#FBBF24",
    "td_seg_iter_2": "#34D399",
    "base_fixsc": "#A78BFA",
}

THERMAL_CASES = (
    {
        "case_id": "individual_90c",
        "label": "90\N{DEGREE SIGN}C\nIndividual",
        "storage_state": 0.8,
        "load_growth": 1.01,
        "thermal_limit_c": 90,
        "suffix": "model_analysis",
        "solver": "td_sin_seg_derate_iter_2",
    },
    {
        "case_id": "corrected_90c",
        "label": "90\N{DEGREE SIGN}C\nCorrected",
        "storage_state": 0.8,
        "load_growth": 1.01,
        "thermal_limit_c": 90,
        "suffix": "model_analysis",
        "solver": "td_seg_derate_iter_2",
    },
    *(
        {
            "case_id": f"corrected_{temperature}c",
            "label": f"{temperature}\N{DEGREE SIGN}C\nCorrected",
            "storage_state": 0.8,
            "load_growth": 1.01,
            "thermal_limit_c": temperature,
            "suffix": "thermal_analysis",
            "solver": "td_seg_derate_iter_2",
        }
        for temperature in (120, 150, 180)
    ),
)

LOAD_GROWTH_CASES = tuple(
    {
        "case_id": f"growth_{percent}pct",
        "label": f"GR {percent}%",
        "storage_state": 0.8,
        "load_growth": rate,
        "thermal_limit_c": 90,
        "suffix": "model_analysis" if rate == 1.01 else "sensitivity_analysis",
        "solver": "td_seg_derate_iter_2",
    }
    for percent, rate in ((1, 1.01), (2, 1.02), (3, 1.03))
)

STORAGE_CASES = tuple(
    {
        "case_id": f"soc_{percent}pct",
        "label": f"SoC {percent}%",
        "storage_state": state,
        "load_growth": 1.01,
        "thermal_limit_c": 90,
        "suffix": "model_analysis" if state == 1.0 else "sensitivity_analysis",
        "solver": "td_seg_derate_iter_2",
    }
    for percent, state in ((0, 0.0), (50, 0.5), (100, 1.0))
)

SEQUENTIAL_ORANGE = ("#FDD0A2", "#FDAE6B", "#FD8D3C", "#F16913", "#D94801")
SEQUENTIAL_BLUE = ("#9ECAE1", "#74A9CF", "#4292C6", "#2171B5", "#0A5194")
SEQUENTIAL_RED = ("#FC9272", "#FB6A4A", "#E84A51", "#CB181D", "#A50F15")

ARTWORK_FILES = (
    "sensitivity_model_capacity_drop_box_violin.pdf",
    "sensitivity_model_line_temp_box_violin.pdf",
    "sensitivity_model_load_shedding_box_violin.pdf",
    "thermal_load_shedding_box_violin.pdf",
    "load_sensitivity_grouped_bar.pdf",
    "storage_sensitivity_grouped_bar.pdf",
)

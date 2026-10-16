"""Frozen data and plotting definitions for Supplementary Fig. 16."""

from __future__ import annotations


SUPPLEMENTARY_FIGURE = 16
COUNTRY_CODE = "ES"
N_BUSES = 281
N_BRANCHES = 442
N_SCENARIOS = 480
STORAGE_STATE = 0.8
LOAD_GROWTH = 1.01
THERMAL_LIMIT_C = 90

METHODS = (
    "base",
    "td_quad",
    "td_seg_derate_iter_2",
    "td_seg_derate_iter_10",
)

METHOD_LABELS = {
    "base": "AC-OPF",
    "td_quad": "Quad-OPF",
    "td_seg_derate_iter_2": "Iter-OPF",
    "td_seg_derate_iter_10": "TD-OPF",
}

METHOD_COLORS = {
    "base": "#72A9D0",
    "td_quad": "#8EC6C2",
    "td_seg_derate_iter_2": "#F5B378",
    "td_seg_derate_iter_10": "#E83947",
}

SCENARIO_KEY_COLUMNS = ("fut_heatwave_date", "his_heatwave_date")
SCENARIO_CONTEXT_COLUMNS = (
    "fut_heatwave_year",
    "fut_heatwave_month",
    "fut_heatwave_day",
    "fut_heatwave_hour",
    "air_temp",
    "wind_speed",
    "solar_radia",
    "load",
)

MODEL_ANALYSIS_CSV = (
    "ES_281_bus_renewable_True_heatwave_True_storage_True_0.8_"
    "load_growth_True_1.01_max_temp_90_model_analysis.csv"
)

ARTWORK_FILES = (
    "model_capacity_drop_box_violin.pdf",
    "model_line_temp_box_violin.pdf",
    "model_load_shedding_box_violin.pdf",
    "model_running_time.pdf",
    "load_load_shedding_scatter.pdf",
)

CAPACITY_SECURITY_MARGIN_PERCENT = 70.0


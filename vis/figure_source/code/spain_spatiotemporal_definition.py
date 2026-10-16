"""Frozen data and plotting definitions for Supplementary Fig. 17."""

from __future__ import annotations


SUPPLEMENTARY_FIGURE = 17
COUNTRY_CODE = "ES"
N_BUSES = 281
N_BRANCHES = 442
N_SEGMENTS = 18
N_SCENARIOS = 480
SCENARIOS_PER_YEAR = 96
SCENARIOS_PER_HOUR = 120
SOLVER = "td_seg_derate_iter_2"
STORAGE_STATE = 0.8
LOAD_GROWTH = 1.01
THERMAL_LIMIT_C = 90
CAPACITY_SECURITY_MARGIN_PERCENT = 70.0
LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT = 0.1

FUTURE_YEARS = (2026, 2027, 2028, 2029, 2030)
FUTURE_HOURS = (12, 13, 14, 15)
HOUR_LABELS = ("12:00", "13:00", "14:00", "15:00")
SEQUENTIAL_ORANGE = ("#FDD0A2", "#FDAE6B", "#FD8D3C", "#F16913", "#D94801")
SEQUENTIAL_BLUE = ("#9ECAE1", "#74A9CF", "#4292C6", "#2171B5", "#0A5194")
SEQUENTIAL_RED = ("#FC9272", "#FB6A4A", "#E84A51", "#CB181D", "#A50F15")

SCENARIO_KEY_COLUMNS = ("fut_heatwave_date", "his_heatwave_date")
SCENARIO_CONTEXT_COLUMNS = (
    "fut_heatwave_year",
    "fut_heatwave_month",
    "fut_heatwave_day",
    "fut_heatwave_hour",
    "his_heatwave_year",
    "air_temp",
    "wind_speed",
    "solar_radia",
    "load",
    "node_load_shedding",
)

MAP_HISTORICAL_DATE = "2024-07-30"
MAP_WEATHER_RECORD = "future_weather_data_based_on_historical_hot_event_2024_0.nc"
MAP_FUTURE_DATETIMES = tuple(
    f"2030-07-24 {hour:02d}:00:00" for hour in FUTURE_HOURS
)

STATISTICAL_ARTWORK = (
    "hour_model_capacity_drop_box_violin.pdf",
    "temperal_model_capacity_drop_box_violin.pdf",
    "hour_model_line_temp_box_violin.pdf",
    "temperal_model_line_temp_box_violin.pdf",
    "hour_model_load_shedding_box_violin.pdf",
    "temporal_load_shedding_box_violin.pdf",
)
MAP_ARTWORK = tuple(
    f"ES_td_seg_derate_iter_2_{future}_{MAP_HISTORICAL_DATE} 00:00:00.pdf"
    for future in MAP_FUTURE_DATETIMES
)
COLORBAR_ARTWORK = ("color_bar_temperature.pdf", "color_bar_line_temp.pdf")


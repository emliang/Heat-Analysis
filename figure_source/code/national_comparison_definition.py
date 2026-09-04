"""Frozen settings for the eight-country Supplementary Fig. 15 summary."""

from __future__ import annotations


SUPPLEMENTARY_FIGURE = 15
PACKAGE_NAME = "Supplementary_Figure_15_national_comparison"

COUNTRY_ORDER = ("PT", "NL", "BE", "GB", "DE", "ES", "IT", "FR")
COUNTRY_NAMES = {
    "PT": "Portugal",
    "NL": "Netherlands",
    "BE": "Belgium",
    "GB": "UK",
    "DE": "Germany",
    "ES": "Spain",
    "IT": "Italy",
    "FR": "France",
}
COUNTRY_COLORS = {
    "PT": "#B83B3B",
    "NL": "#E8630A",
    "BE": "#C9A227",
    "GB": "#1E63B6",
    "DE": "#4B4B4B",
    "ES": "#C43E1C",
    "IT": "#138C6B",
    "FR": "#2A4FA8",
}
N_BUSES = {"PT": 81, "NL": 34, "BE": 42, "GB": 319, "DE": 484, "ES": 281, "IT": 373, "FR": 439}
N_BRANCHES = {"PT": 129, "NL": 41, "BE": 50, "GB": 426, "DE": 682, "ES": 442, "IT": 522, "FR": 711}
N_SEGMENTS = {"PT": 10, "NL": 8, "BE": 8, "GB": 10, "DE": 18, "ES": 18, "IT": 14, "FR": 18}

METHOD = "td_seg_derate_iter_2"
N_SCENARIOS_PER_COUNTRY = 480
STORAGE_STATE = 0.8
LOAD_GROWTH = 1.01
THERMAL_LIMIT_C = 90.0
CAPACITY_SECURITY_MARGIN_PERCENT = 70.0

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

ARTWORK_FILES = (
    "multi_country_analysis_air_temp.pdf",
    "multi_country_analysis_load.pdf",
    "multi_country_analysis_load_shedding.pdf",
    "multi_country_analysis_run_time.pdf",
    "multi_country_analysis_line_temp.pdf",
    "multi_country_analysis_capa_drop.pdf",
)


def model_analysis_filename(country_code: str) -> str:
    return (
        f"{country_code}_{N_BUSES[country_code]}_bus_renewable_True_"
        "heatwave_True_storage_True_0.8_load_growth_True_1.01_"
        "max_temp_90_model_analysis.csv"
    )

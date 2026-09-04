"""Frozen settings for Supplementary Figs. 25--28 cross-border packages."""

from __future__ import annotations

from types import SimpleNamespace


SOLVER = "td_seg_derate_iter_2"
STORAGE_STATE = 0.8
LOAD_GROWTHS = (1.01, 1.03)
THERMAL_LIMIT_C = 90
N_SCENARIOS = 480
CAPACITY_SECURITY_MARGIN_PERCENT = 70.0
LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT = 0.1
HISTORICAL_HEATWAVE_DATE = "2024-07-30"
N_SEGMENTS = 18
SCENARIO_KEY_COLUMNS = ("fut_heatwave_date", "his_heatwave_date")


COUNTRIES = {
    "ES": {
        "name": "Spain",
        "slug": "spain",
        "n_buses": 281,
        "n_reference_lines": 442,
        "statistics_figure": 25,
        "maps_figure": 26,
        "pairs": (("ES", "PT"), ("ES", "FR")),
        "statistics_pairs": (("ES", "FR"), ("ES", "PT")),
        "labels": ("Spain", r"Spain$\leftrightarrow$France", r"Spain$\leftrightarrow$Portugal"),
        "colors": ("#C43E1C", "#2A4FA8", "#B83B3B"),
        "single_map_datetimes": (
            "2026-07-28 13:00:00",
            "2028-07-21 13:00:00",
            "2030-07-24 13:00:00",
        ),
        "joint_map_datetimes": (
            "2026-07-28 13:00:00",
            "2028-07-21 13:00:00",
            "2030-07-24 13:00:00",
        ),
    },
    "FR": {
        "name": "France",
        "slug": "france",
        "n_buses": 439,
        "n_reference_lines": 711,
        "statistics_figure": 27,
        "maps_figure": 28,
        "pairs": (("FR", "IT"), ("FR", "ES"), ("FR", "GB")),
        "statistics_pairs": (("FR", "IT"), ("FR", "ES"), ("FR", "GB")),
        "labels": (
            "France",
            r"France$\leftrightarrow$Italy",
            r"France$\leftrightarrow$Spain",
            r"France$\leftrightarrow$UK",
        ),
        "colors": ("#2A4FA8", "#138C6B", "#C43E1C", "#1E63B6"),
        "single_map_datetimes": (
            "2026-07-29 12:00:00",
            "2028-07-22 12:00:00",
            "2030-07-24 13:00:00",
        ),
        "joint_map_datetimes": (
            "2026-07-29 13:00:00",
            "2028-07-22 13:00:00",
            "2030-07-24 13:00:00",
        ),
    },
}


def configuration_id(value: str | tuple[str, str]) -> str:
    if isinstance(value, str):
        return value
    return "-".join(value)


def model_token(value: str | tuple[str, str]) -> str:
    if isinstance(value, str):
        return value
    return str(list(value))


def settings(country_code: str) -> SimpleNamespace:
    code = country_code.upper()
    if code not in COUNTRIES:
        raise ValueError(f"Unsupported cross-border reference country: {country_code}")
    country = COUNTRIES[code]
    statistics_configs = (code, *country["statistics_pairs"])
    map_configs = (code, *country["pairs"])
    return SimpleNamespace(
        COUNTRY_CODE=code,
        COUNTRY_NAME=country["name"],
        COUNTRY_SLUG=country["slug"],
        N_BUSES=country["n_buses"],
        N_REFERENCE_LINES=country["n_reference_lines"],
        N_SEGMENTS=N_SEGMENTS,
        STATISTICS_FIGURE=country["statistics_figure"],
        MAPS_FIGURE=country["maps_figure"],
        STATISTICS_PACKAGE=(
            f"Supplementary_Figure_{country['statistics_figure']:02d}_"
            f"{country['slug']}_cross_border_statistics"
        ),
        MAPS_PACKAGE=(
            f"Supplementary_Figure_{country['maps_figure']:02d}_"
            f"{country['slug']}_cross_border_maps"
        ),
        STATISTICS_CONFIGS=statistics_configs,
        MAP_CONFIGS=map_configs,
        LABELS=country["labels"],
        COLORS=country["colors"],
        SINGLE_MAP_DATETIMES=country["single_map_datetimes"],
        JOINT_MAP_DATETIMES=country["joint_map_datetimes"],
        SOLVER=SOLVER,
        STORAGE_STATE=STORAGE_STATE,
        LOAD_GROWTHS=LOAD_GROWTHS,
        THERMAL_LIMIT_C=THERMAL_LIMIT_C,
        N_SCENARIOS=N_SCENARIOS,
        CAPACITY_SECURITY_MARGIN_PERCENT=CAPACITY_SECURITY_MARGIN_PERCENT,
        LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT=LOAD_SHEDDING_MARKER_THRESHOLD_PERCENT,
        HISTORICAL_HEATWAVE_DATE=HISTORICAL_HEATWAVE_DATE,
        SCENARIO_KEY_COLUMNS=SCENARIO_KEY_COLUMNS,
    )


def statistical_artwork(country_code: str) -> tuple[str, ...]:
    result = []
    for load_growth in LOAD_GROWTHS:
        for metric in ("load_shedding", "line_temp", "line_capacity_drop"):
            result.append(
                f"model_cross_border_{metric}_box_violin.pdf_"
                f"{load_growth}_{STORAGE_STATE}.pdf"
            )
    return tuple(result)


def map_artwork(country_code: str) -> tuple[str, ...]:
    cfg = settings(country_code)
    result = []
    for configuration in cfg.MAP_CONFIGS:
        datetimes = (
            cfg.SINGLE_MAP_DATETIMES
            if isinstance(configuration, str)
            else cfg.JOINT_MAP_DATETIMES
        )
        token = model_token(configuration)
        for future in datetimes:
            result.append(
                f"{token}_{SOLVER}_{future}_{HISTORICAL_HEATWAVE_DATE} 00:00:00.pdf"
            )
    return tuple(result)

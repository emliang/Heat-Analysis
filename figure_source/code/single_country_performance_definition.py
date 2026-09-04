"""Country-specific settings for the repeated single-country SI analyses."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import spain_operating_sensitivity_definition as operating_base
import spain_opf_definition as opf_base
import spain_spatiotemporal_definition as spatiotemporal_base


COUNTRIES = {
    "ES": {
        "name": "Spain",
        "slug": "spain",
        "n_buses": 281,
        "n_branches": 442,
        "n_segments": 18,
        "figures": {"opf": 16, "spatiotemporal": 17, "operating": 18},
        "map_date": "2030-07-24",
    },
    "IT": {
        "name": "Italy",
        "slug": "italy",
        "n_buses": 373,
        "n_branches": 522,
        "n_segments": 14,
        "figures": {"opf": 19, "spatiotemporal": 20, "operating": 21},
        "map_date": "2030-07-25",
    },
    "FR": {
        "name": "France",
        "slug": "france",
        "n_buses": 439,
        "n_branches": 711,
        "n_segments": 18,
        "figures": {"opf": 22, "spatiotemporal": 23, "operating": 24},
        "map_date": "2030-07-23",
    },
}


def _uppercase_values(module: ModuleType) -> dict:
    return {name: getattr(module, name) for name in dir(module) if name.isupper()}


def _country(country_code: str) -> dict:
    country_code = country_code.upper()
    if country_code not in COUNTRIES:
        raise ValueError(f"Unsupported single-country SI analysis: {country_code}")
    return {"country_code": country_code, **COUNTRIES[country_code]}


def opf_settings(country_code: str) -> SimpleNamespace:
    country = _country(country_code)
    values = _uppercase_values(opf_base)
    values.update(
        {
            "SUPPLEMENTARY_FIGURE": country["figures"]["opf"],
            "COUNTRY_CODE": country["country_code"],
            "COUNTRY_NAME": country["name"],
            "N_BUSES": country["n_buses"],
            "N_BRANCHES": country["n_branches"],
            "N_SEGMENTS": country["n_segments"],
            "PACKAGE_NAME": (
                f"Supplementary_Figure_{country['figures']['opf']:02d}_"
                f"{country['slug']}_opf_comparison"
            ),
            "MODEL_ANALYSIS_CSV": (
                f"{country['country_code']}_{country['n_buses']}_bus_renewable_True_"
                "heatwave_True_storage_True_0.8_load_growth_True_1.01_"
                "max_temp_90_model_analysis.csv"
            ),
        }
    )
    return SimpleNamespace(**values)


def operating_settings(country_code: str) -> SimpleNamespace:
    country = _country(country_code)
    values = _uppercase_values(operating_base)
    values.update(
        {
            "SUPPLEMENTARY_FIGURE": country["figures"]["operating"],
            "COUNTRY_CODE": country["country_code"],
            "COUNTRY_NAME": country["name"],
            "N_BUSES": country["n_buses"],
            "N_BRANCHES": country["n_branches"],
            "N_SEGMENTS": country["n_segments"],
            "PACKAGE_NAME": (
                f"Supplementary_Figure_{country['figures']['operating']:02d}_"
                f"{country['slug']}_operating_sensitivity"
            ),
        }
    )
    return SimpleNamespace(**values)


def spatiotemporal_settings(country_code: str) -> SimpleNamespace:
    country = _country(country_code)
    values = _uppercase_values(spatiotemporal_base)
    map_datetimes = tuple(
        f"{country['map_date']} {hour:02d}:00:00" for hour in values["FUTURE_HOURS"]
    )
    values.update(
        {
            "SUPPLEMENTARY_FIGURE": country["figures"]["spatiotemporal"],
            "COUNTRY_CODE": country["country_code"],
            "COUNTRY_NAME": country["name"],
            "N_BUSES": country["n_buses"],
            "N_BRANCHES": country["n_branches"],
            "N_SEGMENTS": country["n_segments"],
            "PACKAGE_NAME": (
                f"Supplementary_Figure_{country['figures']['spatiotemporal']:02d}_"
                f"{country['slug']}_spatiotemporal_stress"
            ),
            "MAP_FUTURE_DATETIMES": map_datetimes,
            "MAP_ARTWORK": tuple(
                f"{country['country_code']}_td_seg_derate_iter_2_{future}_"
                f"{values['MAP_HISTORICAL_DATE']} 00:00:00.pdf"
                for future in map_datetimes
            ),
        }
    )
    return SimpleNamespace(**values)

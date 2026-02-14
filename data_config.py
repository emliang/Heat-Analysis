"""
Project-wide configuration and constants.

This module centralises paths, look-up tables, physical constants, and
calibration parameters used across the TD-OPF analysis pipeline.

Sections
--------
1. Path resolution helpers
2. Directory constants
3. Country metadata
4. Physical constants & colour palette
5. Network clustering defaults
6. Demand-model calibration parameters
7. IEEE conductor defaults
"""

import os
import sys
import warnings

import holidays

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Path Resolution Helpers
# =============================================================================

def _find_directory(name, max_levels=3):
    """Walk up from cwd to locate a sibling/ancestor directory.

    Parameters
    ----------
    name : str
        Directory name to search for (e.g. ``'data'``, ``'models'``).
    max_levels : int
        Maximum number of parent directories to traverse.

    Returns
    -------
    str
        Absolute path to the found directory.

    Raises
    ------
    FileNotFoundError
        If the directory is not found within *max_levels* levels.
    """
    base = os.getcwd()
    for _ in range(max_levels):
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
        base = os.path.dirname(base)
    raise FileNotFoundError(
        f"Cannot find '{name}' directory within {max_levels} levels of cwd"
    )


def get_local_data_path():
    """Return the path to the ``data/`` directory."""
    return _find_directory("data", max_levels=2)


def get_local_model_path():
    """Return the path to the ``models/`` directory."""
    return _find_directory("models", max_levels=3)


def get_pypsa_path():
    """Return the path to the ``pypsa-eur/`` directory."""
    return _find_directory("pypsa-eur", max_levels=3)


def get_holiday_list(country_code, year):
    """Return the public-holiday calendar for a country and year.

    Parameters
    ----------
    country_code : str
        Two-letter ISO 3166-1 country code.
    year : int
        Calendar year.

    Returns
    -------
    holidays.HolidayBase
        Holiday calendar instance.

    Raises
    ------
    ValueError
        If *country_code* is not supported.
    """
    _HOLIDAY_MAP = {
        'BE': holidays.Belgium,
        'FR': holidays.France,
        'IT': holidays.Italy,
        'ES': holidays.Spain,
        'GB': holidays.UnitedKingdom,
        'DE': holidays.Germany,
        'PT': holidays.Portugal,
        'NL': holidays.Netherlands,
    }
    cls = _HOLIDAY_MAP.get(country_code)
    if cls is None:
        raise ValueError(f"Unsupported country code: {country_code}")
    return cls(years=year)


# =============================================================================
# 2. Directory Constants
# =============================================================================

# These files (ERA5 hourly reanalysis, RCP 4.5 projections, elevation) are too large to include in the repository.  Download them FIRST and point. EXTERNAL to the directory where they are stored.
# Download link (Google Drive):
#   https://drive.google.com/drive/folders/1SJmglPiEMTw--xggqzeiSjmKBCI17cnK
#
# Expected contents of EXTERNAL:
#   era5/era5_hourly_summer_2019_2024.nc   — ERA5 hourly reanalysis
#   era5/era5_daily_avg_2015_2024.nc        — ERA5 daily-averaged weather
#   rcp45/rcp45_3hourly_summer_2019_2024.nc — RCP 4.5 historical baseline
#   rcp45/rcp45_3hourly_summer_2025_2030.nc — RCP 4.5 future projections
#   elevation.nc                            — Surface elevation grid
# Adjust the EXTERNAL directory name as you like
EXTERNAL = "/Volumes/T9"
WEATHER_DAILY = EXTERNAL + "/era5/"


# PyPSA-Eur — open-source European transmission network model.
# Build it locally first by following https://pypsa-eur.readthedocs.io/
#
# Only the configuration file and exclusion / siting rasters are read
# directly from the pypsa-eur/ tree.  Network snapshots, bus regions,
# and load ratios have been reorganised into data/EU/ (see LOCAL_DATA).
#
# Files consumed from pypsa-eur/ by this project:
#
#   config/config.yaml
#       Renewable technology siting parameters (capacity_per_sqkm,
#       exclusion zones, correction factors, resource methods, etc.)
#       Used by: main_build_simulation_profile.py
#
#   data/natura.tiff
#       Natura 2000 protected-area raster (renewable exclusion zone).
#
#   data/bundle/corine/g250_clc06_V18_5.tif
#       CORINE land-cover raster (renewable siting exclusion codes).
#
#   data/shipdensity_raster.tif
#       Shipping-lane density raster (offshore wind exclusion).
#
#   data/bundle/GEBCO_2014_2D.nc
#       GEBCO bathymetry (max depth constraint for offshore wind).
#
PYPSA = get_pypsa_path()
sys.path.append(PYPSA)
RESOURCES = os.path.join(PYPSA, "resources")
DATA = os.path.join(PYPSA, "data")
CUTOUT = os.path.join(PYPSA, "cutouts")

# Local project data — included in the repository (or generated locally).
#
# Network data originally produced by PyPSA-Eur are stored here after
# reorganisation so they can be versioned with the project.
#
# data/EU/ contents consumed by the active code:
#
#   networks/base_s_{ratio}_elec.nc
#       Pan-European solved network (buses, generators, lines, links,
#       storage, loads, time-series).  Loaded by load_network_EU() in
#       network_process_utils.py; used by TDOPF_eur.py and all scripts.
#
#   regions_onshore_base_s_{ratio}.geojson
#       Voronoi bus regions for spatial weather / demand aggregation.
#
#   regions_offshore_base_s_{ratio}.geojson
#       Offshore bus regions (used in renewable profile generation).
#
#   load_ratio_base_s_{ratio}.csv
#       Population-weighted load fractions per bus region.
#
#   country_shapes.geojson / offshore_shapes.geojson
#       National boundary and EEZ polygons for exclusion-zone filtering.
#
#   {CC}/filtered_shapes/
#       Country-level clipped shape caches (generated at runtime).
#
LOCAL_DATA = get_local_data_path()
DEMAND = os.path.join(LOCAL_DATA, "entsoe")
MODELS = get_local_model_path()


# =============================================================================
# 3. Country Metadata
# =============================================================================

country_name = {
    'AL': 'Albania',
    'AT': 'Austria',
    'BA': 'Bosnia and Herzegovina',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CH': 'Switzerland',
    'CZ': 'Czechia',
    'DE': 'Germany',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'ES': 'Spain',
    'EU': 'EU',
    'FI': 'Finland',
    'FR': 'France',
    'GB': 'UK',
    'GR': 'Greece',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LT': 'Lithuania',
    'LV': 'Latvia',
    'ME': 'Montenegro',
    'MK': 'North Macedonia',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'RS': 'Serbia',
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
    'TR': 'Turkey',
    'UA': 'Ukraine',
    'VA': 'Vatican City',
    'XK': 'Kosovo',
}


# =============================================================================
# 4. Physical Constants
# =============================================================================

BASELINE_YEAR = 2025
"""Baseline year for demand-growth projections."""

TK = 273.15
"""Offset from Kelvin to Celsius."""

# Colour palette for demand-curve / visualisation plots
red = '#d62728'    # model curve – weekday
blue = '#1f77b4'   # model curve – weekend
lred = '#ff9896'   # scatter – weekday observed
lblue = '#aec7e8'  # scatter – weekend observed


# =============================================================================
# 5. Network Clustering Defaults
# =============================================================================

RATIO = 75
"""Default PyPSA-Eur cluster ratio (approximate number of buses ÷ 75)."""

country_bus = {
    # Single-country networks
    'BE': 42,
    'DE': 484,
    'ES': 281,
    'FR': 439,
    'GB': 319,
    'IT': 373,
    'NL': 34,
    'PT': 81,
    # Cross-border interconnected networks
    ('ES', 'FR'): 720,
    ('ES', 'PT'): 362,
    ('FR', 'ES'): 720,
    ('FR', 'GB'): 813,
    ('FR', 'IT'): 758,
}


# =============================================================================
# 6. Demand-Model Calibration Parameters
# =============================================================================
#
# Reference (for validation only — not used directly in the paper):
#   Staffell, I., Pfenninger, S., & Johnson, N. (2023).
#   "A global model of hourly space heating and cooling demand at
#   multiple spatial scales." Nature Energy, 8(12), 1328–1344.
#
# Each entry contains:
#   heating_threshold (°C)  — temperature below which heating demand rises
#   cooling_threshold (°C)  — temperature above which cooling demand rises
#   p_heating (GW/°C)       — heating sensitivity (daily average)
#   p_cooling (GW/°C)       — cooling sensitivity (daily average)

demand_model_para = {
    'AT': {'heating_threshold': 10.67, 'cooling_threshold': 24.83,
           'p_heating': 0.12, 'p_cooling': 0.30},
    'BE': {'heating_threshold': 11.28, 'cooling_threshold': 24.63,
           'p_heating': 0.14, 'p_cooling': 0.43},
    'BG': {'heating_threshold': 13.52, 'cooling_threshold': 21.62,
           'p_heating': 0.12, 'p_cooling': 0.10},
    'CH': {'heating_threshold': 12.54, 'cooling_threshold': 25.84,
           'p_heating': 0.10, 'p_cooling': 0.17},
    'CZ': {'heating_threshold': 14.18, 'cooling_threshold': 100,
           'p_heating': 0.10, 'p_cooling': 0},
    'DE': {'heating_threshold': 11.39, 'cooling_threshold': 23.06,
           'p_heating': 0.51, 'p_cooling': 0.55},
    'DK': {'heating_threshold': 11.01, 'cooling_threshold': 22.70,
           'p_heating': 0.06, 'p_cooling': 0.21},
    'ES': {'heating_threshold': 14.67, 'cooling_threshold': 19.82,
           'p_heating': 0.65, 'p_cooling': 0.88},
    'FR': {'heating_threshold': 12.98, 'cooling_threshold': 22.69,
           'p_heating': 2.37, 'p_cooling': 1.59},
    'GB': {'heating_threshold': 12.19, 'cooling_threshold': 20.91,
           'p_heating': 0.83, 'p_cooling': 1.36},
    'GR': {'heating_threshold': 13.74, 'cooling_threshold': 22.59,
           'p_heating': 0.18, 'p_cooling': 0.35},
    'HU': {'heating_threshold': 13.54, 'cooling_threshold': 24.58,
           'p_heating': 0.05, 'p_cooling': 0.35},
    'IT': {'heating_threshold': 12.01, 'cooling_threshold': 20.56,
           'p_heating': 0.62, 'p_cooling': 1.46},
    'PL': {'heating_threshold': 13.01, 'cooling_threshold': 23.86,
           'p_heating': 0.17, 'p_cooling': 0.82},
    'PT': {'heating_threshold': 13.39, 'cooling_threshold': 21.74,
           'p_heating': 0.18, 'p_cooling': 0.34},
    'RO': {'heating_threshold': 15.68, 'cooling_threshold': 22.16,
           'p_heating': 0.09, 'p_cooling': 0.20},
    'SK': {'heating_threshold': 11.98, 'cooling_threshold': 24.95,
           'p_heating': 0.04, 'p_cooling': 0.08},
}


# =============================================================================
# 7. IEEE Conductor Defaults
# =============================================================================
"""795 kcmil 26/7 Drake ACSR conductor (IEEE Std 738-2012 defaults)."""
ieee_conductor_config = {
    'diameter': 28.1e-3,           # m
    'ref_temperature': 25,         # °C
    'max_temperature': 90,         # °C
    'resistance_ratio': 0.00429,   # Ω/°C
    'unit_resistance': 7.283e-5,   # Ω/m
}

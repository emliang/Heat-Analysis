#!/usr/bin/env python3
"""Build simulation-ready weather, demand and renewable profiles.

This script processes historical ERA5 reanalysis and RCP 4.5 climate
projections to generate the input data required by the Temperature-
Dependent Optimal Power Flow (TD-OPF) simulation:

1. **Demand profiles** — future daily peak-load estimates derived from
   bias-corrected heatwave weather via the BAIT demand model.
2. **Weather snapshots** — hourly gridded fields (temperature, solar,
   wind) for each heatwave event, saved as NetCDF.
3. **Renewable profiles** — bus-level capacity-factor time-series for
   solar, onshore wind, and offshore wind technologies, generated with
   `atlite` from the weather snapshots.

Both **single-country** and **multi-country** (cross-border) modes are
supported.

Output layout
-------------
    models/{COUNTRY}/simu_data/   — demand arrays & weather NetCDF
    models/{COUNTRY}/weather/     — renewable profile NetCDF

Section index
-------------
1. Data Loading
2. Date Utilities
3. Demand Estimation
4. Heatwave Event Processing
5. Network Utilities
6. Renewable Profile Generation
7. Entry Points
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import calendar
import functools
import os
import sys
from datetime import datetime

import atlite
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from yaml import Loader

sys.path.append(os.path.join(os.getcwd(), "."))
from data_config import *                    # noqa: F403
from utils.network_process_utils import *    # noqa: F403
from utils.heat_flow_utils import *          # noqa: F403
from utils.demand_utils import _bait
from utils.heatwave_utils import *           # noqa: F403


# ===========================================================================
# 1. Data Loading
# ===========================================================================

def load_weather_data():
    """Load all required weather datasets from disk.

    Returns
    -------
    datasets : dict[str, xr.Dataset]
        Keys: ``'his_era5'``, ``'his_rcp45'``, ``'fut_rcp45'``, ``'elevation'``.
    """
    print("Loading weather datasets...")
    datasets = {
        'his_era5':  xr.open_dataset(
            EXTERNAL + "/era5/era5_hourly_summer_2019_2024.nc"),
        'his_rcp45': xr.open_dataset(
            EXTERNAL + "/rcp45/rcp45_3hourly_summer_2019_2024.nc"),
        'fut_rcp45': xr.open_dataset(
            EXTERNAL + "/rcp45/rcp45_3hourly_summer_2025_2030.nc"),
        'elevation': xr.open_dataset(
            EXTERNAL + "/elevation.nc"),
    }
    print("✅ Weather datasets loaded successfully")
    return datasets


def setup_country_network(country_list, ratio):
    """Load the PyPSA-Eur network for *country_list* and derive bounds.

    Parameters
    ----------
    country_list : list[str]
        Two-letter country codes (e.g. ``['FR', 'DE']``).
    ratio : int
        Clustering ratio passed to :func:`load_network_EU`.

    Returns
    -------
    network : pypsa.Network
    regions : gpd.GeoDataFrame
    pop_ratio : np.ndarray
        Per-bus population weights (normalised within regions).
    bounds : dict
        Spatial bounding box with keys ``x_min``, ``x_max``, ``y_min``,
        ``y_max`` (0.25° buffer included).
    """
    print(f"Setting up network for {country_list}...")
    network, regions = load_network_EU(country_list, ratio)
    pop_ratio = network.buses.loc[regions.index, 'pop_ratio'].values

    bounds = {
        'x_min': network.buses['x'].min() - 0.25,
        'x_max': network.buses['x'].max() + 0.25,
        'y_min': network.buses['y'].min() - 0.25,
        'y_max': network.buses['y'].max() + 0.25,
    }
    print(f"✅ Network loaded: {len(network.buses)} buses, bounds: {bounds}")
    return network, regions, pop_ratio, bounds


def crop_weather_data(datasets, bounds):
    """Crop all datasets to *bounds* and align the elevation grid.

    Parameters
    ----------
    datasets : dict[str, xr.Dataset]
        Output of :func:`load_weather_data`.
    bounds : dict
        Bounding box from :func:`setup_country_network`.

    Returns
    -------
    cropped : dict[str, xr.Dataset]
        Same keys as *datasets* plus ``'elevation_aligned'``.
    """
    print("Cropping weather data to country bounds...")
    cropped = {}
    for name, dataset in datasets.items():
        if name == 'elevation':
            # Elevation uses (lon, lat) coordinates
            cropped[name] = dataset.sel(
                lon=slice(bounds['x_min'], bounds['x_max']),
                lat=slice(bounds['y_min'], bounds['y_max']),
            )
        else:
            cropped[name] = dataset.sel(
                x=slice(bounds['x_min'], bounds['x_max']),
                y=slice(bounds['y_min'], bounds['y_max']),
            )

    # Align static elevation to the weather grid
    elevation_renamed = cropped['elevation'].rename({'lon': 'x', 'lat': 'y'})
    cropped['elevation_aligned'] = elevation_renamed.interp_like(
        cropped['his_era5'], method='linear',
    )
    print("✅ Weather data cropped and aligned")
    return cropped


# ===========================================================================
# 2. Date Utilities
# ===========================================================================

def get_date_range(year, month):
    """Return ``(start, end)`` datetimes spanning day 3–30 of *month*.

    Parameters
    ----------
    year, month : int
        Target year and month.

    Returns
    -------
    tuple[datetime, datetime]
    """
    return datetime(year, month, 3, 0), datetime(year, month, 30, 23)


def calculate_3day_date_range(target_date, buffer_days=2):
    """Return a date window ending on *target_date*, going back *buffer_days*.

    Parameters
    ----------
    target_date : datetime
        The heatwave peak day.
    buffer_days : int
        Number of preceding days to include.

    Returns
    -------
    start_date, end_date : datetime
    """
    if target_date.day < 3:
        # Handle month boundary
        last_month_days = calendar.monthrange(
            target_date.year, target_date.month - 1)[1]
        start_date = datetime(
            target_date.year, target_date.month - 1,
            last_month_days + target_date.day - buffer_days, 0, 0, 0,
        )
    else:
        start_date = datetime(
            target_date.year, target_date.month,
            target_date.day - buffer_days, 0, 0, 0,
        )
    end_date = datetime(
        target_date.year, target_date.month, target_date.day, 23, 0, 0,
    )
    return start_date, end_date


# ===========================================================================
# 3. Demand Estimation
# ===========================================================================

def calculate_future_demand(his_era5_slice, fut_heatwave, regions,
                            demand_model, pop_ratio):
    """Estimate future peak demand using the BAIT thermal comfort model.

    The function aggregates gridded weather to bus-level averages, computes
    the Cooling Degree Days (CDD) above the comfort threshold, and applies
    the calibrated demand model.

    Parameters
    ----------
    his_era5_slice : xr.Dataset
        Historical ERA5 slice (used for humidity, which is unavailable in
        the bias-corrected future fields).
    fut_heatwave : xr.Dataset
        Bias-corrected future heatwave weather.
    regions : gpd.GeoDataFrame
        Bus regions for spatial aggregation.
    demand_model : dict
        Calibrated BAIT demand parameters.
    pop_ratio : np.ndarray
        Per-bus population weights (sum == 1).

    Returns
    -------
    bait_weighted : float
        Population-weighted BAIT index.
    demand : float
        Estimated daily peak demand [GW].
    """
    # Aggregate to bus-level averages
    his_humidity = aggregate_regional_data(
        his_era5_slice, regions, variables=['humidity'])['humidity'].data
    fut_temperature = aggregate_regional_data(
        fut_heatwave, regions, variables=['temperature'])['temperature'].data
    fut_wind = aggregate_regional_data(
        fut_heatwave, regions, variables=['wnd10m'])['wnd10m'].data
    fut_solar = aggregate_regional_data(
        fut_heatwave, regions, variables=['influx'])['influx'].data

    # Daily averages (reshape: 24 hours × 3 days × buses)
    temp_avg     = fut_temperature.reshape(24, 3, -1).mean(0) - TK
    solar_avg    = fut_solar.reshape(24, 3, -1).mean(0)
    wind_avg     = fut_wind.reshape(24, 3, -1).mean(0)
    humidity_avg = his_humidity.reshape(24, 3, -1).mean(0)

    # BAIT model → CDD → demand
    bait = _bait(temp_avg, wind_avg, solar_avg, humidity_avg, demand_model)[-1]
    CDD = np.maximum(bait - demand_model['Tc'], 0)
    demand = (demand_model['Pb']
              + demand_model['Pc'] * (CDD * pop_ratio).sum(-1)
              + demand_model['alpha'])

    bait_weighted = (bait * pop_ratio).sum()
    return bait_weighted, demand


# ===========================================================================
# 4. Heatwave Event Processing
# ===========================================================================

def process_heatwave_events(network, country_list, weather_datasets,
                            regions, demand_model_dict, config, output_dir):
    """Generate demand & weather profiles for every (hist, fut) event pair.

    For each historical seed year and its top heatwave events, the function
    iterates over future years and months, applies bias-corrected delta
    mapping, estimates demand, and saves the results.

    Parameters
    ----------
    network : pypsa.Network
    country_list : list[str]
    weather_datasets : dict[str, xr.Dataset]
        Cropped weather datasets from :func:`crop_weather_data`.
    regions : gpd.GeoDataFrame
    demand_model_dict : dict[str, dict]
        ``{country_code: calibrated_demand_model}``.
    config : dict
        Scenario configuration (years, months, event counts).
    output_dir : str
        Directory for output files.
    """
    print(f"\n🌡️  Processing heatwave events...")
    print(f"📊 Configuration: {config}")

    ref_country = country_list[0]
    country_index = [bus[:2] == ref_country for bus in regions.index]
    ref_country_region = regions[country_index]

    for hist_year in config['historical_years']:
        for hist_idx in range(config['num_historical_events']):
            print(f"\n{'=' * 60}")
            print(f"Processing historical year {hist_year}, event {hist_idx}")
            print(f"{'=' * 60}")

            # Initialize data containers
            demand_data = {
                'daily_demand': [], 'future_date': [],
                'historical_date': [], 'daily_bait': [],
            }
            future_heatwave_profile = []

            for fut_year in config['future_years']:
                for month in config['hot_months']:
                    # --- Identify hottest days ---------------------------
                    start_date, end_date = get_date_range(hist_year, month)
                    his_hw_idx, his_hw_dates = find_heatwave_days(
                        weather_datasets['his_era5'].sel(
                            time=slice(start_date, end_date)),
                        ref_country_region, weights=[0.9, 0.1],
                    )
                    start_date, end_date = get_date_range(fut_year, month)
                    fut_hw_idx, fut_hw_dates = find_heatwave_days(
                        weather_datasets['fut_rcp45'].sel(
                            time=slice(start_date, end_date)),
                        ref_country_region, weights=[0.9, 0.1],
                    )

                    # Fix on one historical seed event
                    hist_hot_idx = his_hw_idx[hist_idx]
                    hist_heatwave_date = pd.to_datetime(
                        his_hw_dates[hist_hot_idx])

                    # Iterate over top-N future events
                    for fut_idx in range(config['num_future_events']):
                        fut_heatwave_idx = fut_hw_idx[fut_idx]
                        fut_heatwave_date = pd.to_datetime(
                            fut_hw_dates[fut_heatwave_idx])

                        # 3-day windows for demand estimation
                        fut_start, fut_end = calculate_3day_date_range(
                            fut_heatwave_date)
                        hist_start, hist_end = calculate_3day_date_range(
                            hist_heatwave_date)

                        # Extract 3-day weather slices
                        his_era5_3d = weather_datasets['his_era5'].sel(
                            time=slice(hist_start, hist_end))
                        his_rcp45_3d = weather_datasets['his_rcp45'].sel(
                            time=slice(hist_start, hist_end))
                        fut_rcp45_3d = weather_datasets['fut_rcp45'].sel(
                            time=slice(fut_start, fut_end))

                        # Interpolate 3-hourly → hourly & bias-correct
                        his_rcp45_3d_hourly = interpolate_3h_to_1h(
                            his_rcp45_3d)
                        fut_rcp45_3d_hourly = interpolate_3h_to_1h(
                            fut_rcp45_3d)
                        fut_heatwave_3d = bias_correction(
                            ['temperature', 'influx', 'wnd10m', 'wnd100m'],
                            his_era5_3d, his_rcp45_3d_hourly,
                            fut_rcp45_3d_hourly, smooth_grid=2,
                        )

                        # Estimate demand for each country
                        country_demand_list = []
                        bait_list = []
                        for country, demand_model in demand_model_dict.items():
                            c_idx = [bus[:2] == country
                                     for bus in regions.index]
                            c_region = regions[c_idx]
                            c_pop = network.buses.loc[
                                c_region.index, 'pop_ratio'].values
                            c_pop = c_pop / c_pop.sum()
                            bait_w, demand = calculate_future_demand(
                                his_era5_3d, fut_heatwave_3d,
                                c_region, demand_model, c_pop,
                            )
                            country_demand_list.append(demand)
                            bait_list.append(bait_w)

                        # Store results
                        demand_data['daily_demand'].append(
                            country_demand_list)
                        demand_data['daily_bait'].append(bait_list)
                        demand_data['future_date'].append(fut_heatwave_date)
                        demand_data['historical_date'].append(
                            hist_heatwave_date)

                        print(
                            f"DEMAND | Country: {country_list} "
                            f"| Future: {demand_data['future_date'][-1]} "
                            f"| Historical: "
                            f"{demand_data['historical_date'][-1]} "
                            f"| Load: "
                            f"{demand_data['daily_demand'][-1]:.2f} GW "
                            f"| Bait: "
                            f"{demand_data['daily_bait'][-1]:.2f}"
                        )

                        # Keep the peak day (last day) for weather output
                        day_start = datetime(
                            fut_year, month, fut_heatwave_date.day, 0)
                        day_end = datetime(
                            fut_year, month, fut_heatwave_date.day, 23)
                        fut_heatwave_1d = fut_heatwave_3d.sel(
                            time=slice(day_start, day_end))
                        future_heatwave_profile.append(fut_heatwave_1d)

            # Concatenate all events and attach elevation
            future_heatwave_profile = xr.concat(
                future_heatwave_profile, dim='time')
            future_heatwave_profile['height'] = (
                weather_datasets['elevation_aligned']['z'])
            future_heatwave_profile['lon'] = future_heatwave_profile['x']
            future_heatwave_profile['lat'] = future_heatwave_profile['y']

            # Save to disk
            demand_file = (
                f'{output_dir}/future_demand_data_based_on_'
                f'historical_hot_event_{hist_year}_{hist_idx}.npy')
            weather_file = (
                f'{output_dir}/future_weather_data_based_on_'
                f'historical_hot_event_{hist_year}_{hist_idx}.nc')

            np.save(demand_file, demand_data)
            future_heatwave_profile.to_netcdf(weather_file)
            print(f"✅ Saved data for historical year "
                  f"{hist_year}, event {hist_idx}")

    # Save date catalogues
    np.save(f'{output_dir}/future_hot_dates.npy',
            demand_data['future_date'])
    np.save(f'{output_dir}/historical_hot_dates.npy',
            demand_data['historical_date'])
    print("✅ Heatwave event processing completed!")


# ===========================================================================
# 5. Network Utilities
# ===========================================================================

def identify_cross_country_lines(network):
    """Identify transmission lines and links that cross country borders.

    Parameters
    ----------
    network : pypsa.Network
        Network with a ``'country'`` attribute on buses.

    Returns
    -------
    cross_country_lines : list
        Line indices where ``bus0`` and ``bus1`` belong to different
        countries.
    cross_country_links : list
        Link indices where ``bus0`` and ``bus1`` belong to different
        countries.
    """
    cross_country_lines = []
    for line_idx in network.lines.index:
        bus0 = network.lines.loc[line_idx, 'bus0']
        bus1 = network.lines.loc[line_idx, 'bus1']
        if network.buses.loc[bus0, 'country'] != \
                network.buses.loc[bus1, 'country']:
            cross_country_lines.append(line_idx)

    cross_country_links = []
    for link_idx in network.links.index:
        bus0 = network.links.loc[link_idx, 'bus0']
        bus1 = network.links.loc[link_idx, 'bus1']
        if network.buses.loc[bus0, 'country'] != \
                network.buses.loc[bus1, 'country']:
            cross_country_links.append(link_idx)

    return cross_country_lines, cross_country_links


# ===========================================================================
# 6. Renewable Profile Generation
# ===========================================================================

def generate_single_renewable_profile(technology, weather_cutout_path,
                                      country_code, ratio, output_path,
                                      network, verbose=True):
    """Generate a bus-level renewable capacity-factor profile.

    Uses an ``atlite.Cutout`` built from processed weather data together
    with the PyPSA-Eur exclusion / siting configuration to produce
    per-bus time-series.

    Parameters
    ----------
    technology : str
        One of ``'solar'``, ``'onwind'``, ``'offwind-ac'``,
        ``'offwind-dc'``.
    weather_cutout_path : str
        Path to the atlite-compatible NetCDF cutout.
    country_code : str
        Two-letter country code.
    ratio : int
        Clustering ratio for region files.
    output_path : str
        Destination NetCDF file.
    network : pypsa.Network
    verbose : bool

    Returns
    -------
    ds : xr.Dataset
        Merged dataset with ``'profile'``, ``'weight'``,
        ``'p_nom_max'``, ``'potential'``.
    """
    # Load PyPSA-Eur renewable configuration
    with open(PYPSA + "/config/config.yaml", "r") as f:
        pypsa_config = yaml.load(f, Loader)
    params = pypsa_config['renewable'][technology]

    correction_factor = params.get("correction_factor", 1.0)
    capacity_per_sqkm = params["capacity_per_sqkm"]
    p_nom_max_meth = params.get("potential", "conservative")

    # Load weather cutout
    cutout = atlite.Cutout(weather_cutout_path)

    # ----- Region geometries -----------------------------------------------
    if technology in ('onwind', 'solar'):
        regions_path = (
            LOCAL_DATA + f"/EU/regions_onshore_base_s_{ratio}.geojson")
    else:
        regions_path = (
            LOCAL_DATA + f"/EU/regions_offshore_base_s_{ratio}.geojson")

    regions = load_and_filter_regional_data(network, regions_path)
    buses = regions.index

    # ----- Filtered shape files --------------------------------------------
    country_shapes_path = LOCAL_DATA + "/EU/country_shapes.geojson"
    offshore_shapes_path = LOCAL_DATA + "/EU/offshore_shapes.geojson"

    shape_dir = LOCAL_DATA + f'/EU/{country_code}/filtered_shapes'
    os.makedirs(shape_dir, exist_ok=True)
    filtered_country, filtered_offshore = filter_and_save_shapes(
        country_code, regions, country_shapes_path,
        offshore_shapes_path, shape_dir, verbose=True,
    )

    # ----- Exclusion zones -------------------------------------------------
    res = params.get("excluder_resolution", 100)
    excluder = atlite.ExclusionContainer(crs=3035, res=res)

    # Natura 2000 protected areas
    if params.get("natura", False):
        natura_path = PYPSA + "/data/natura.tiff"
        if os.path.exists(natura_path):
            excluder.add_raster(natura_path, nodata=0,
                                allow_no_overlap=True)

    # CORINE land-cover codes
    corine = params.get("corine", {})
    corine_path = PYPSA + "/data/bundle/corine/g250_clc06_V18_5.tif"
    if "grid_codes" in corine and os.path.exists(corine_path):
        excluder.add_raster(
            corine_path, codes=corine["grid_codes"],
            invert=True, crs=3035,
        )
    if ("distance" in corine and corine["distance"] > 0.0
            and os.path.exists(corine_path)):
        excluder.add_raster(
            corine_path, codes=corine["distance_grid_codes"],
            buffer=corine["distance"], crs=3035,
        )

    # Additional exclusions for offshore technologies
    if technology.startswith('offwind'):
        ship_path = PYPSA + "/data/shipdensity_raster.tif"
        gebco_path = PYPSA + "/data/bundle/GEBCO_2014_2D.nc"

        if "ship_threshold" in params and os.path.exists(ship_path):
            threshold = params["ship_threshold"] * 8760 * 6
            excluder.add_raster(
                ship_path,
                codes=functools.partial(np.less, threshold),
                crs=4326, allow_no_overlap=True,
            )
        if params.get("max_depth") and os.path.exists(gebco_path):
            excluder.add_raster(
                gebco_path,
                codes=functools.partial(np.greater, -params["max_depth"]),
                crs=4326, nodata=-1000,
            )
        if "min_shore_distance" in params and os.path.exists(
                filtered_country):
            excluder.add_geometry(
                filtered_country, buffer=params["min_shore_distance"])
        if "max_shore_distance" in params and os.path.exists(
                filtered_offshore):
            excluder.add_geometry(
                filtered_offshore,
                buffer=params["max_shore_distance"], invert=True,
            )

    # ----- Availability & resource calculation -----------------------------
    availability = cutout.availabilitymatrix(
        regions, excluder, nprocesses=8)

    resource = params["resource"].copy()
    func = getattr(cutout, resource.pop("method"))

    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(
        area.values.reshape(cutout.shape),
        [cutout.coords["y"], cutout.coords["x"]],
    )

    potential = capacity_per_sqkm * availability.sum("bus") * area
    capacity_factor = correction_factor * func(
        capacity_factor=True, **resource)
    layout = capacity_factor * area * capacity_per_sqkm

    profile, capacities = func(
        matrix=availability.stack(spatial=["y", "x"]),
        layout=layout,
        index=buses,
        per_unit=True,
        return_capacity=True,
        **resource,
    )

    # ----- Potential capacity ----------------------------------------------
    if p_nom_max_meth == "simple":
        p_nom_max = capacity_per_sqkm * availability @ area
    elif p_nom_max_meth == "conservative":
        max_cap_factor = capacity_factor.where(
            availability != 0).max(["x", "y"])
        p_nom_max = capacities / max_cap_factor

    # ----- Assemble output dataset -----------------------------------------
    ds = xr.merge([
        (correction_factor * profile).rename("profile"),
        capacities.rename("weight"),
        p_nom_max.rename("p_nom_max"),
        potential.rename("potential"),
    ])

    # Filter by minimum capacity factor and p_nom_max
    ds = ds.sel(bus=(
        (ds["profile"].mean("time") > params.get("min_p_max_pu", 0.0))
        & (ds["p_nom_max"] > params.get("min_p_nom_max", 0.0))
    ))

    # Optional clipping of low capacity factors
    if "clip_p_max_pu" in params:
        min_pu = params["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(ds["profile"] >= min_pu, 0)

    ds.to_netcdf(output_path)

    if verbose:
        print(f"   - Buses with capacity: {len(ds.bus)}")
        print(f"   - Mean capacity factor: "
              f"{ds['profile'].mean().values:.3f}")
        print(f"   - Max potential: "
              f"{ds['potential'].max().values:.1f} MW/km²")
    return ds


def generate_renewable_profiles_from_weather(
        weather_nc_path, country_code, heatwave_year, heatwave_index,
        ratio, network, output_dir=None, verbose=True):
    """Generate renewable profiles for all carriers from one weather file.

    Parameters
    ----------
    weather_nc_path : str
        Path to the atlite-compatible weather NetCDF.
    country_code : str
    heatwave_year : int
    heatwave_index : int
    ratio : int
    network : pypsa.Network
    output_dir : str or None
    verbose : bool

    Returns
    -------
    generated_profiles : dict[str, str]
        ``{technology: output_path}`` for each successfully generated
        profile.
    """
    if output_dir is None:
        output_dir = MODELS + f'/{country_code}/weather'
    os.makedirs(output_dir, exist_ok=True)

    renewable_carriers = ['solar', 'onwind', 'offwind-ac', 'offwind-dc']

    generated_profiles = {}
    for technology in renewable_carriers:
        profile_path = os.path.join(
            output_dir,
            f'{heatwave_year}_{heatwave_index}_{technology}'
            f'_profile_s_{ratio}.nc',
        )
        generate_single_renewable_profile(
            technology=technology,
            weather_cutout_path=weather_nc_path,
            country_code=country_code,
            ratio=ratio,
            output_path=profile_path,
            network=network,
            verbose=verbose,
        )
        generated_profiles[technology] = profile_path
        if verbose:
            print(f"✅ {technology} profile generated: {profile_path}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RENEWABLE PROFILE GENERATION SUMMARY")
        print(f"✅ Successful: {len(generated_profiles)}")
        print(f"{'=' * 60}")
    return generated_profiles


def load_renewable_profiles(country_code, heatwave_year, heatwave_index,
                            ratio):
    """Load previously saved renewable profiles for one scenario.

    Parameters
    ----------
    country_code : str
    heatwave_year : int
    heatwave_index : int
    ratio : int

    Returns
    -------
    profiles : dict[str, xr.Dataset | None]
        One entry per carrier; ``None`` if the file is missing.
    """
    weather_path = MODELS + f'/{country_code}/weather'
    technologies = ['solar', 'onwind', 'offwind-ac', 'offwind-dc']

    profiles = {}
    for tech in technologies:
        fpath = (f'{weather_path}/{heatwave_year}_{heatwave_index}'
                 f'_{tech}_profile_s_{ratio}.nc')
        try:
            profiles[tech] = xr.open_dataset(fpath)
            print(f"Loaded {tech} profile: {fpath}")
        except FileNotFoundError:
            print(f"Warning: {tech} profile not found: {fpath}")
            profiles[tech] = None
    return profiles


def generate_renewable_profiles_for_all_scenarios(
        country_code, ratio, config, network, verbose=False):
    """Loop over all (year, event) pairs and generate renewable profiles.

    Parameters
    ----------
    country_code : str
    ratio : int
    config : dict
    network : pypsa.Network
    verbose : bool
    """
    weather_path = MODELS + f'/{country_code}/weather'
    os.makedirs(weather_path, exist_ok=True)

    hist_years = config['historical_years']
    n_events = config['num_historical_events']
    total = len(hist_years) * n_events
    current = 0

    print(f"🚀 Starting renewable profile generation for {country_code}")
    print(f"📊 Processing {len(hist_years)} years × {n_events} events")

    for hw_year in hist_years:
        for hw_idx in range(n_events):
            current += 1
            print(f"\n{'=' * 80}")
            print(f"SCENARIO {current}/{total}: "
                  f"Year {hw_year}, Event {hw_idx}")
            print(f"{'=' * 80}")

            weather_nc = (
                MODELS + f'/{country_code}/simu_data/'
                f'future_weather_data_based_on_'
                f'historical_hot_event_{hw_year}_{hw_idx}.nc')

            generated = generate_renewable_profiles_from_weather(
                weather_nc_path=weather_nc,
                country_code=country_code,
                heatwave_year=hw_year,
                heatwave_index=hw_idx,
                ratio=ratio,
                network=network,
                output_dir=weather_path,
                verbose=verbose,
            )
            print(f"✅ Scenario {current} completed — "
                  f"{len(generated)} profiles")

    print("\n🎉 Renewable profile generation completed for all scenarios!")


# ===========================================================================
# 7. Entry Points
# ===========================================================================

def main_single(config):
    """Run the full pipeline for individual countries (single-country mode).

    Parameters
    ----------
    config : dict
        Scenario configuration dictionary.
    """
    print("🌍 Starting heat analysis simulation profile builder")
    print(f"⚙️  Configuration: {config}")

    for country in ['IT']:
        # Step 0 — Load data
        weather_datasets = load_weather_data()
        network, regions, pop_ratio, bounds = setup_country_network(
            [country], RATIO)
        weather_datasets = crop_weather_data(weather_datasets, bounds)

        # Step 0b — Load demand calibration model
        print("Loading demand calibration model...")
        demand_model = load_demand_profile(country)
        demand_model_dict = {country: demand_model}
        print("✅ Demand model loaded")

        # Output directory
        output_dir = MODELS + f'/{country}/simu_data'
        os.makedirs(output_dir, exist_ok=True)

        # Step 1 — Demand profiles
        print("\n" + "=" * 80)
        print("STEP 1: GENERATING DEMAND PROFILES")
        print("=" * 80)
        process_heatwave_events(
            network, [country], weather_datasets, regions,
            demand_model_dict, config, output_dir,
        )

        # Step 2 — Renewable profiles
        print("\n" + "=" * 80)
        print("STEP 2: GENERATING RENEWABLE PROFILES")
        print("=" * 80)
        generate_renewable_profiles_for_all_scenarios(
            country, RATIO, config, network)

        print(f"\n🎉 All processing completed successfully for {country}!")


def main_multi(config):
    """Run the full pipeline for country pairs (multi-country mode).

    The returned network contains all countries with cross-border lines
    and links.  Load ratios are individually normalised for demand
    calculation and individually stored.

    Parameters
    ----------
    config : dict
        Scenario configuration dictionary.
    """
    print("🌍 Starting heat analysis simulation profile builder")
    print(f"⚙️  Configuration: {config}")

    country_pairs = [
        ['ES', 'FR'], ['FR', 'ES'], ['FR', 'IT'],
        ['FR', 'GB'], ['FR', 'DE'],
    ]
    for country_list in country_pairs:
        # Step 0 — Load data
        weather_datasets = load_weather_data()
        network, regions, _, bounds = setup_country_network(
            country_list, RATIO)
        weather_datasets = crop_weather_data(weather_datasets, bounds)

        # Step 0b — Load demand calibration models (one per country)
        print("Loading demand calibration models...")
        demand_model_dict = {
            c: load_demand_profile(c) for c in country_list
        }
        print("✅ Demand models loaded")

        # Output directory
        output_dir = MODELS + f'/{country_list}/simu_data'
        os.makedirs(output_dir, exist_ok=True)

        # Step 1 — Demand profiles
        print("\n" + "=" * 80)
        print("STEP 1: GENERATING DEMAND PROFILES")
        print("=" * 80)
        process_heatwave_events(
            network, country_list, weather_datasets, regions,
            demand_model_dict, config, output_dir,
        )

        # Step 2 — Renewable profiles
        print("\n" + "=" * 80)
        print("STEP 2: GENERATING RENEWABLE PROFILES")
        print("=" * 80)
        generate_renewable_profiles_for_all_scenarios(
            country_list, RATIO, config, network)

        print(f"\n🎉 All processing completed for {country_list}!")


# ===========================================================================
# __main__
# ===========================================================================

if __name__ == "__main__":
    config = {
        'num_historical_events': 2,
        'num_future_events': 2,
        'historical_years': [2019, 2022, 2024],
        'future_years': [2026, 2027, 2028, 2029, 2030],
        'hot_months': [6, 7],
        'demand_smoothing_days': 3,
    }
    main_single(config)
    main_multi(config)

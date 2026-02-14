"""Temperature-Dependent AC Optimal Power Flow for European networks.

This is the **main entry point** for running TD-ACOPF simulations on
PyPSA-Eur networks under projected heatwave scenarios.  The pipeline is:

1. Load a PyPSA-Eur network filtered by country (or country pair).
2. For each future heatwave timestep:
   a. Attach spatially-resolved weather (temperature, wind, solar,
      elevation) to buses and line segments.
   b. Compute temperature-dependent demand (from calibrated models).
   c. Update renewable capacity factors from weather-driven profiles.
   d. Apply generator derating based on ambient temperature.
3. Convert the PyPSA network to a PyPower ``ppc`` dict.
4. Solve the TD-ACOPF iteratively (resistance ↔ temperature loop).
5. Post-process and record results to CSV.

Simulation sweeps (sensitivity / model comparison) are orchestrated by
:func:`main`, which can run in sequential or parallel mode.

Section index
-------------
1. Network Data Loading (weather, demand, renewables, derating)
2. Simulation Setup
3. Iterative TD-ACOPF Solver
4. Single-Timestep Driver & Post-Processing
5. Parallel / Batch Orchestration
6. Entry Point & Parameter Sweeps
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import copy
import multiprocessing as mp
import os
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from data_config import *  # noqa: F403
from utils.heat_flow_utils import *  # noqa: F403
from utils.heatwave_utils import aggregate_regional_data
from utils.network_process_utils import *  # noqa: F403
from utils.opf_pyomo_utils import *  # noqa: F403

# For macOS/Windows compatibility with multiprocessing
mp.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ===========================================================================
# 1. Network Data Loading
# ===========================================================================

def network_load_weather(network, regions, ncfile_slice):
    """Attach weather fields from a NetCDF time-slice to the network.

    Weather variables (temperature, wind speed, solar irradiance, wind
    azimuth, elevation) are assigned to:

    * **Buses** — nearest grid-cell value, then overwritten with
      area-weighted regional averages where regions are available.
    * **Line segments** — per-segment nearest grid-cell lookup, stored
      in ``network.segment_weather`` as a 3-D array
      ``(n_lines, max_segs, 5)``.

    Parameters
    ----------
    network : pypsa.Network
        Network with ``buses``, ``lines``, and ``segments`` attached.
    regions : gpd.GeoDataFrame
        Onshore regions for area-weighted aggregation.
    ncfile_slice : xr.Dataset
        Single-timestep weather dataset with variables
        ``temperature``, ``wnd10m``, ``influx``, ``wnd_azimuth``,
        ``height``.
    """
    variables = ['temperature', 'wnd10m', 'influx', 'wnd_azimuth', 'height']

    # --- Bus-level weather (nearest grid cell) ---
    bus_loc = network.buses[['x', 'y']].values
    x_da = xr.DataArray(bus_loc[:, 0], dims=["bus"])
    y_da = xr.DataArray(bus_loc[:, 1], dims=["bus"])
    bus_slice = ncfile_slice.sel(x=x_da, y=y_da, method="nearest")

    network.buses['temperature'] = (
        bus_slice['temperature'].values.reshape(-1) - TK)
    network.buses['wnd10m'] = bus_slice['wnd10m'].values.reshape(-1)
    network.buses['influx'] = bus_slice['influx'].values.reshape(-1)
    network.buses['wnd_azimuth'] = (
        bus_slice['wnd_azimuth'].values.reshape(-1))
    network.buses['height'] = np.maximum(
        bus_slice['height'].values.reshape(-1), 0)

    # --- Overwrite with regional averages where available ---
    reg_slice = aggregate_regional_data(
        ncfile_slice, regions, variables=variables)
    idx = network.buses.index.isin(regions.index)
    network.buses.loc[idx, 'temperature'] = (
        reg_slice['temperature'].values.reshape(-1) - TK)
    network.buses.loc[idx, 'wnd10m'] = (
        reg_slice['wnd10m'].values.reshape(-1))
    network.buses.loc[idx, 'influx'] = (
        reg_slice['influx'].values.reshape(-1))
    network.buses.loc[idx, 'wnd_azimuth'] = (
        reg_slice['wnd_azimuth'].values.reshape(-1))
    network.buses.loc[idx, 'height'] = np.maximum(
        reg_slice['height'].values.reshape(-1), 0)

    # --- Segment-level weather ---
    seg_info = network.segments
    nline, nseg = seg_info.shape[0], seg_info.shape[1]
    seg_locs = seg_info[:, :, 0:2].reshape(-1, 2)
    x_da = xr.DataArray(seg_locs[:, 0], dims=["seg"])
    y_da = xr.DataArray(seg_locs[:, 1], dims=["seg"])
    seg_slice = ncfile_slice.sel(x=x_da, y=y_da, method="nearest")

    segment_weather = np.zeros(shape=[nline, nseg, 5])
    segment_weather[:, :, 0] = (
        seg_slice['temperature'].data.reshape(nline, nseg) - TK)
    segment_weather[:, :, 1] = (
        seg_slice['wnd10m'].data.reshape(nline, nseg))
    segment_weather[:, :, 2] = (
        seg_slice['influx'].data.reshape(nline, nseg))
    segment_weather[:, :, 3] = (
        seg_slice['wnd_azimuth'].data.reshape(nline, nseg))
    segment_weather[:, :, 4] = np.maximum(
        seg_slice['height'].data.reshape(nline, nseg), 0)
    network.segment_weather = segment_weather


def network_load_demand(network, date, country_code, load_demand, args):
    """Distribute hourly demand to buses using load ratios.

    Parameters
    ----------
    network : pypsa.Network
        Network with ``buses['load_ratio']`` already assigned.
    date : datetime
        Simulation timestamp (hour-level).
    country_code : str or list[str]
        Country code(s) in the network.
    load_demand : dict
        Pre-computed demand dict with keys ``daily_demand``,
        ``future_date``, ``historical_date``.
    args : dict
        Must contain ``load_growth_rate``, ``reactive_demand_ratio``.

    Returns
    -------
    pypsa.Network
    """
    country_list = ([country_code] if not isinstance(country_code, list)
                    else country_code)
    daily_demand_list = load_demand['daily_demand']
    future_dates = load_demand['future_date']
    historical_dates = load_demand['historical_date']

    index = future_dates.index(
        pd.to_datetime(datetime(date.year, date.month, date.day, 0, 0, 0)))
    daily_load = daily_demand_list[index]
    network.historical_heatwave_date = historical_dates[index]

    for i, country in enumerate(country_list):
        demand_curve = load_demand_profile(country)
        hour_ratio = np.reshape(
            demand_curve['weekday_cooling_hour_ratio'], (-1))
        # Apply base-load growth
        growth = (args['load_growth_rate']
                  ** max(date.year - BASELINE_YEAR, 0) - 1)
        daily_grown = daily_load[i] + demand_curve['Pb'] * growth
        hourly = daily_grown * hour_ratio[date.hour] * 1e3 * 24

        mask = (network.buses.country.values == country)
        lr = network.buses.loc[mask, 'load_ratio']
        lr = lr / lr.sum()
        network.buses.loc[mask, 'p_set'] = hourly * lr
        network.buses.loc[mask, 'q_set'] = (
            network.buses.loc[mask, 'p_set']
            * args['reactive_demand_ratio'])

    return network


def network_load_renewable(network, solar_profile, onwind_profile,
                           offwind_ac_profile, offwind_dc_profile):
    """Update generator ``p_max_pu`` from weather-driven profiles.

    Parameters
    ----------
    network : pypsa.Network
    solar_profile, onwind_profile : xr.Dataset
    offwind_ac_profile, offwind_dc_profile : xr.Dataset

    Returns
    -------
    pypsa.Network
    """
    net_month = pd.to_datetime(
        network.generators_t['p_max_pu'].index).month
    prof_month = pd.to_datetime(solar_profile.time.values).month
    net_hour = pd.to_datetime(
        network.generators_t['p_max_pu'].index).hour
    prof_hour = pd.to_datetime(solar_profile.time.values).hour

    for gen_id in network.generators.index:
        gen = network.generators.loc[gen_id]
        bus_id = gen['bus']
        tech = gen['carrier']

        if tech in ('solar', 'solar-hsat') and bus_id in solar_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = (
                solar_profile.sel(bus=bus_id).profile.data)
        elif tech == 'onwind' and bus_id in onwind_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = (
                onwind_profile.sel(bus=bus_id).profile.data)
        elif tech == 'offwind-ac' and bus_id in offwind_ac_profile.bus:
            network.generators.loc[gen_id, 'p_max_pu'] = (
                offwind_ac_profile.sel(bus=bus_id).profile.data)
        elif (tech in ('offwind-dc', 'offwind-float')
              and bus_id in offwind_dc_profile.bus):
            network.generators.loc[gen_id, 'p_max_pu'] = (
                offwind_dc_profile.sel(bus=bus_id).profile.data)
        elif gen_id in network.generators_t['p_max_pu'].columns:
            # Fall back to historical monthly-hourly average
            network.generators.loc[gen_id, 'p_max_pu'] = (
                network.generators_t['p_max_pu'].loc[
                    (net_month == prof_month) & (net_hour == prof_hour),
                    gen_id,
                ].mean())

    return network


def network_load_derating(network):
    """Apply temperature-dependent generator derating.

    Parameters
    ----------
    network : pypsa.Network
        Must have ``buses['temperature']`` populated.

    Returns
    -------
    pypsa.Network
    """
    for gen_idx, gen in network.generators.iterrows():
        bus_name = gen['bus']
        gen_weather = {
            'air_temperature': network.buses.loc[bus_name, 'temperature'],
        }
        network.generators.loc[gen_idx, 'derating'] = generator_derating(
            gen['carrier'], gen_weather)
    return network


# ===========================================================================
# 2. Simulation Setup
# ===========================================================================

def prepare_network(network, regions, date, country_code,
                    heatwave_year, heatwave_index, args):
    """Load weather, demand, renewables, and derating for one timestep.

    Parameters
    ----------
    network : pypsa.Network
    regions : gpd.GeoDataFrame
    date : datetime
        UTC timestamp to simulate.
    country_code : str or list[str]
    heatwave_year : int
        Historical heatwave year used as bias-correction anchor.
    heatwave_index : int
        Index of the heatwave event within that year.
    args : dict

    Returns
    -------
    pypsa.Network
        Fully prepared network ready for OPF.
    """
    # Weather
    weather_path = (MODELS + f'/{country_code}/simu_data/'
                    f'future_weather_data_based_on_historical_hot_event_'
                    f'{heatwave_year}_{heatwave_index}.nc')
    weather_cutout = xr.open_dataset(weather_path).sel(time=date)
    network_load_weather(network, regions, weather_cutout)

    # Demand
    demand_path = (f'models/{country_code}/simu_data/'
                   f'future_demand_data_based_on_historical_hot_event_'
                   f'{heatwave_year}_{heatwave_index}.npy')
    load_demand = np.load(demand_path, allow_pickle=True).item()
    network = network_load_demand(
        network, date, country_code, load_demand, args)

    # Renewable profiles
    ratio = network.resolution
    wp = MODELS + f'/{country_code}/weather'
    hw = f'{heatwave_year}_{heatwave_index}'
    solar = xr.open_dataset(
        f'{wp}/{hw}_solar_profile_s_{ratio}.nc').sel(time=date)
    onwind = xr.open_dataset(
        f'{wp}/{hw}_onwind_profile_s_{ratio}.nc').sel(time=date)
    offwind_ac = xr.open_dataset(
        f'{wp}/{hw}_offwind-ac_profile_s_{ratio}.nc').sel(time=date)
    offwind_dc = xr.open_dataset(
        f'{wp}/{hw}_offwind-dc_profile_s_{ratio}.nc').sel(time=date)
    network = network_load_renewable(
        network, solar, onwind, offwind_ac, offwind_dc)

    # Generator derating
    network = network_load_derating(network)

    return network


def setup_heatwave_analysis(country_code, args, nbus,
                            label='TDOPF_heatflow_analysis'):
    """Build the output CSV path for a heatwave analysis run.

    Parameters
    ----------
    country_code : str or list[str]
    args : dict
    nbus : int
    label : str

    Returns
    -------
    str
        Path to the results CSV file.
    """
    record_path = os.path.join('models', f"{country_code}")
    os.makedirs(record_path, exist_ok=True)
    os.makedirs(os.path.join(record_path, 'ppc'), exist_ok=True)

    df_path = os.path.join(
        record_path,
        f'{country_code}_{nbus}_bus_'
        f'renewable_{args["renewable_mode"]}_'
        f'heatwave_{args["heatwave_mode"]}_'
        f'storage_{args["storage_mode"]}_{args["storage_ratio"]}_'
        f'load_growth_{args["load_growth"]}_{args["load_growth_rate"]}_'
        f'max_temp_{args["max_temperature"]}_'
        f'{label}.csv',
    )
    return df_path


def init_analysis(network, weather, conductor, tdpf_analysis):
    """Initialise weather / conductor dicts from network segment data.

    Depending on whether ``'seg'`` appears in *tdpf_analysis*, weather
    arrays are either per-segment (2-D) or line-averaged (1-D
    column-vectors).  Conductor angle is computed from bus coordinates,
    and the current constraint is solved.

    Parameters
    ----------
    network : pypsa.Network
    weather : dict
        Mutable — populated in-place.
    conductor : dict
        Mutable — populated in-place.
    tdpf_analysis : str
        Analysis mode tag (e.g. ``'td_seg_derate_iter_2'``).
    """
    segment_wea = copy.deepcopy(network.segment_weather)
    bus_wea = copy.deepcopy(
        network.buses[['temperature', 'wnd10m', 'influx',
                        'wnd_azimuth', 'height']].values)
    seg_prop = network.segments[:, :, 2]

    if 'seg' in tdpf_analysis:
        weather['air_temperature'] = segment_wea[:, :, 0]
        weather['wind_speed'] = segment_wea[:, :, 1]
        weather['solar_heat_intensity'] = segment_wea[:, :, 2]
        weather['wind_angle'] = segment_wea[:, :, 3]
        conductor['elevation'] = segment_wea[:, :, 4]
    else:
        weather['air_temperature'] = np.sum(
            segment_wea[:, :, 0] * seg_prop, axis=1, keepdims=True)
        weather['wind_speed'] = np.sum(
            segment_wea[:, :, 1] * seg_prop, axis=1, keepdims=True)
        weather['solar_heat_intensity'] = np.sum(
            segment_wea[:, :, 2] * seg_prop, axis=1, keepdims=True)
        weather['wind_angle'] = np.sum(
            segment_wea[:, :, 3] * seg_prop, axis=1, keepdims=True)
        conductor['elevation'] = np.sum(
            segment_wea[:, :, 4] * seg_prop, axis=1, keepdims=True)

    network.num_bundle = conductor['num_bundle']

    # Conductor angle from bus coordinates
    xe = (network.buses.loc[network.lines['bus1'], ['x', 'y']].values
          - network.buses.loc[network.lines['bus0'], ['x', 'y']].values)
    conductor_angle = (
        90 - np.rad2deg(np.arctan2(xe[:, 1], xe[:, 0]))) % 360
    conductor['conductor_angle'] = np.reshape(conductor_angle, (-1, 1))

    if 'derate' not in tdpf_analysis:
        network.generators['derating'] = 1

    conductor['num_parallel'] = network.lines['num_parallel'].values
    cal_current_constraint(network, conductor, weather, tdpf_analysis)


# ===========================================================================
# 3. Iterative TD-ACOPF Solver
# ===========================================================================

def run_td_acopf_eur(ppc, base_result, contingency, tdpf_analysis,
                     conductor, weather, num_iter=10, tol=1e-4):
    """Run the iterative temperature-dependent ACOPF loop.

    At each iteration the conductor temperature is recomputed from the
    branch current, and the resistance is updated before re-solving.
    Convergence is checked on generation, temperature, and current.

    Parameters
    ----------
    ppc : dict
        PyPower case dictionary.
    base_result : dict or None
        Warm-start from a previous solution.
    contingency : list
        N-1 contingency branch pairs.
    tdpf_analysis : str
        Analysis mode tag.
    conductor, weather : dict
    num_iter : int
    tol : float
        Convergence tolerance.

    Returns
    -------
    ppc : dict
        Updated PyPower case (resistance modified).
    result : dict
        Final OPF solution.
    """
    num_parallel = ppc['branch'][:, -3].copy()
    num_parallel[num_parallel == 0] = 1
    num_bundle = conductor['num_bundle']
    baseMva = ppc['baseMVA']
    baseKV = ppc['bus'][0, idx_bus.BASE_KV]
    BaseI = baseMva / baseKV
    air_temp = weather['air_temperature']

    # Base resistance (immutable copy)
    base_resistance = np.expand_dims(
        ppc['branch'][:, idx_brch.BR_R].copy(), 1)
    base_seg_resistance = ppc['segment'][:, :, 2] * base_resistance

    # Initial resistance from ambient temperature
    ppc['branch'][:, idx_brch.BR_R] = (
        calculate_temperature_dependent_resistance(
            base_resistance, base_seg_resistance,
            air_temp, conductor, tdpf_analysis))

    # Warm-start: update resistance from previous current
    if base_result is not None:
        I_pu = base_result['I_pu']
        I = I_pu * BaseI * 1000
        per_I = I / num_parallel / num_bundle
        con_temp = heat_banlance_equation(
            np.expand_dims(per_I, 1), conductor, weather)
        ppc['branch'][:, idx_brch.BR_R] = (
            calculate_temperature_dependent_resistance(
                base_resistance, base_seg_resistance,
                con_temp, conductor, tdpf_analysis))

    # Parse analysis flags
    fixsc = 'fixsc' in tdpf_analysis
    safe_margin = 0.7 if fixsc else 1
    dcsc = 'dcsc' in tdpf_analysis
    acsc = 'acsc' in tdpf_analysis
    tdacsc = 'tdacsc' in tdpf_analysis
    quad_con = 'quad' in tdpf_analysis
    tem_cons = 'td' in tdpf_analysis

    prev_pg = prev_temp = prev_I = 0

    for it in range(num_iter):
        instance = ACOPF(
            ppc,
            initial_value=base_result,
            contingency=contingency,
            dc_sc=dcsc, ac_sc=acsc, fix_sc=fixsc, tdac_sc=tdacsc,
            safe_margin=safe_margin,
            quad_con=quad_con, td_cons=tem_cons,
            conductor=conductor, weather=weather,
            angle_cons=False, qlim=True,
        )
        result = instance.solve()

        I_pu = result['I_pu']
        I = I_pu * BaseI * 1000
        per_I = I / num_parallel / num_bundle
        con_temp = heat_banlance_equation(
            np.expand_dims(per_I, 1), conductor, weather)
        ppc['branch'][:, idx_brch.BR_R] = (
            calculate_temperature_dependent_resistance(
                base_resistance, base_seg_resistance,
                con_temp, conductor, tdpf_analysis))

        # Convergence check
        if (it > 0
                and np.max(np.abs(result['PG'] - prev_pg)) < tol
                and np.max(np.abs(con_temp - prev_temp)) < tol
                and np.max(np.abs(I - prev_I) / (BaseI * 1000)) < tol):
            print(f'Converged in {it} iterations')
            break
        if result['solver_status'] == 0:
            print('IPOPT failed')
            break

        print(f'{it}-th iter with obj: {result["obj"]}', end='\r')
        base_result = result
        prev_pg = base_result['PG']
        prev_temp = con_temp
        prev_I = I

    return ppc, result


# ===========================================================================
# 4. Single-Timestep Driver & Post-Processing
# ===========================================================================

def TDACOPF_eur(df, network, country_code, utc_timestamp, args,
                conductor, weather, expr_id, tdpf_analysis='base'):
    """Run one TD-ACOPF simulation and append results to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Accumulator for result rows.
    network : pypsa.Network
    country_code : str or list[str]
    utc_timestamp : datetime
    args : dict
    conductor, weather : dict
    expr_id : str
        Unique experiment identifier.
    tdpf_analysis : str
        Analysis mode tag (``'base'``, ``'td_seg_derate_iter_2'``, etc.).

    Returns
    -------
    pd.DataFrame
        *df* with one new row appended.
    """
    init_analysis(network, weather, conductor, tdpf_analysis)
    ppc_expr = pypsa_pypower(network, args)

    # Output directory
    ppc_path = f'models/{country_code}/ppc/{utc_timestamp}'
    result_file = ppc_path + f'/{expr_id}_results.npy'
    os.makedirs(ppc_path, exist_ok=True)

    # Parse analysis flags
    fixsc = 'fixsc' in tdpf_analysis
    safe_margin = 0.7 if fixsc else 1

    # --- Solve ---
    st = time.time()
    if 'base' in tdpf_analysis:
        instance = ACOPF(
            ppc_expr, safe_margin=safe_margin,
            fix_sc=fixsc, qlim=True,
        )
        result = instance.solve()
    elif 'td' in tdpf_analysis:
        num_iter = (int(re.findall(r'\d+', tdpf_analysis)[0])
                    if 'iter' in tdpf_analysis else 1)
        ppc_expr, result = run_td_acopf_eur(
            ppc_expr, None, [], tdpf_analysis,
            conductor, weather, num_iter)
    else:
        raise NotImplementedError(
            f"Analysis type '{tdpf_analysis}' not supported")
    result['runtime'] = time.time() - st

    # Post-process and save
    result = post_process_results(
        result, network, weather, conductor, ppc_expr['baseI'])
    np.save(result_file, result, allow_pickle=True)

    # Build result row
    result_row = {
        'TDPF_solver': tdpf_analysis,
        'country_code': country_code,
        'nbus': len(ppc_expr['bus']),
        'renewable_mode': args['renewable_mode'],
        'heatwave_mode': args['heatwave_mode'],
        'load_growth': args['load_growth'],
        'max_temperature': args['max_temperature'],
        'his_heatwave_date': network.historical_heatwave_date,
        'his_heatwave_year': network.historical_heatwave_date.year,
        'exper_id': expr_id,
        'fut_heatwave_date': utc_timestamp,
        'fut_heatwave_year': utc_timestamp.year,
        'fut_heatwave_month': utc_timestamp.month,
        'fut_heatwave_day': utc_timestamp.day,
        'fut_heatwave_hour': utc_timestamp.hour,
        'air_temp': np.mean(network.buses['temperature'].values),
        'wind_speed': np.mean(network.buses['wnd10m'].values),
        'solar_radia': np.mean(network.buses['influx'].values),
    }
    result_row = ACOPF_analysis_eur(
        country_code, network, args, ppc_expr,
        conductor, result, result_row, utc_timestamp)
    df = df._append(result_row, ignore_index=True)
    return df


def ACOPF_analysis_eur(country_code, network, args, ppc_expr,
                       conductor, result, result_row, utc_timestamp):
    """Compute summary metrics from a solved ACOPF and update *result_row*.

    Metrics include load shedding counts at multiple thresholds,
    branch flow / temperature ratios, conductor temperature statistics,
    generation cost, and capacity-drop statistics.

    Parameters
    ----------
    country_code : str or list[str]
    network : pypsa.Network
    args : dict
    ppc_expr : dict
        PyPower case dictionary.
    conductor : dict
    result : dict
        OPF solution dictionary.
    result_row : dict
        Mutable — updated in-place **and** returned.
    utc_timestamp : datetime

    Returns
    -------
    dict
        Updated *result_row*.
    """
    baseMva = ppc_expr['baseMVA']
    BaseI = ppc_expr['baseI']

    Pg = result['PG']
    Pd = result['PD']
    Pg_capacity = result['PG_capacity']
    ST_capacity = result['ST_capacity']
    cost = result['obj']

    Branch_status = np.sign(
        ppc_expr['branch'][:, idx_brch.BR_STATUS]
        * ppc_expr['branch'][:, idx_brch.RATE_A])
    Gen_status = np.sign(
        ppc_expr['gen'][:, idx_gen.GEN_STATUS]
        * ppc_expr['gen'][:, idx_gen.PMAX])
    Smax = ppc_expr['branch'][:, idx_brch.RATE_A] / baseMva
    Pmax = ppc_expr['gen'][:, idx_gen.PMAX] / baseMva
    S_pu = result['S_pu']
    I_pu = result['I_pu']
    slack_p_eq = result['LS']

    Sratio = S_pu / (Smax + 1e-8) * Branch_status
    Inom = result['inom']
    capacity_drop = result['capacity_drop']
    seg_prop = ppc_expr['segment'][:, :, 2]
    seg_mask = np.sign(seg_prop)
    seg_temp = result['con_temp'] * seg_mask
    branch_temp = result['con_temp'].max(1)
    Tratio = branch_temp / conductor['max_temperature']

    # Attach to network for downstream visualisation
    network.lines['Sratio'] = Sratio
    network.lines['Tratio'] = Tratio
    network.lines['branch_temp'] = branch_temp
    network.buses['load_shedding'] = slack_p_eq

    # --- Populate result_row ---
    result_row['solver_status'] = result['solver_status']
    # Total three-phase load in GW
    result_row['load'] = np.sum(Pd) * baseMva * 3 / 1000
    result_row['node_load_shedding'] = (
        np.sum(slack_p_eq) / np.sum(Pd) * 100)

    # Load-shedding node counts at various thresholds
    result_row['node_mis_match_0001_num'] = int(
        np.sum(slack_p_eq >= 0.001))
    result_row['node_mis_match_001_num'] = int(
        np.sum(slack_p_eq >= 0.01))
    result_row['node_mis_match_01_num'] = int(
        np.sum(slack_p_eq >= 0.1))

    # Branch flow / temperature counts at safety thresholds
    for thresh in (0.7, 0.8, 0.9):
        result_row[f'{thresh}_flow_num'] = int(
            np.sum(Sratio >= thresh))
        result_row[f'{thresh}_temp_num'] = int(
            np.sum(Tratio >= thresh))

    result_row['tem_con_mean'] = (
        np.sum(branch_temp) / np.sum(Branch_status))
    result_row['tem_con_max'] = np.max(seg_temp)
    result_row['run_time'] = result['runtime']
    result_row['gen_cost'] = cost
    result_row['capacity_drop_min'] = capacity_drop.min()
    result_row['capacity_drop_max'] = capacity_drop.max()
    result_row['capacity_drop_mean'] = capacity_drop.mean()
    result_row['capacity_drop_std'] = capacity_drop.std()

    print(
        f"RESULTS : success {result['solver_status']} "
        f"| Demand: {Pd.sum():.2f} "
        f"| Gen: {Pg_capacity.sum():.2f} "
        f"| Storage: {ST_capacity.sum():.2f} "
        f"| LS ratio: {result_row['node_load_shedding']:.2f} "
        f"| LS num: {result_row['node_mis_match_01_num']:.0f} "
        f"| Max temp: {result_row['tem_con_max']:.2f} "
        f"| Runtime: {result_row['run_time']:.2f}s")

    return result_row


# ===========================================================================
# 5. Parallel / Batch Orchestration
# ===========================================================================

def run_single_simulation(params):
    """Run one TD-ACOPF for a single parameter tuple.

    Designed to be called by :func:`multiprocessing.Pool.map`.

    Parameters
    ----------
    params : tuple
        ``(tdpf_analysis, fut_date, country_code, nbus, args,
        conductor, weather, his_heatwave_year, his_heatwave_index)``

    Returns
    -------
    pd.DataFrame
        Single-row result.
    """
    (tdpf_analysis, fut_date, country_code, nbus, args,
     conductor, weather, his_heatwave_year, his_heatwave_index) = params

    network, regions = load_network_EU(country_code, RATIO)
    network = prepare_network(
        network, regions, fut_date, country_code,
        his_heatwave_year, his_heatwave_index, args)

    # Build experiment identifier
    base_id = (f'{country_code}_{nbus}_{tdpf_analysis}_'
               f'{fut_date}_{network.historical_heatwave_date}')
    para_id = (f'storage_{args["storage_ratio"]}_'
               f'load_growth_{args["load_growth_rate"]}_'
               f'thermal_{args["max_temperature"]}')
    expr_id = f'{base_id}_{para_id}'

    temp_df = pd.DataFrame(columns=['exper_id'])
    temp_df = TDACOPF_eur(
        temp_df, network, country_code, fut_date,
        args, conductor, weather, expr_id, tdpf_analysis)
    print(f'expr_id: {expr_id}')
    return temp_df


def run_parallel_simulations(analysis_list, country_code, nbus, args,
                             conductor, weather, df_path=None):
    """Sweep over analysis types, historical events, and future dates.

    For each analysis type the function builds all parameter
    combinations (historical year × event × future hot date × hour),
    then runs them either sequentially or via a multiprocessing pool.

    Parameters
    ----------
    analysis_list : list[str]
        Analysis mode tags to sweep.
    country_code : str or list[str]
    nbus : int
    args : dict
        Must contain ``parallel_mode`` (bool).
    conductor, weather : dict
    df_path : str or None
        Path to write the combined CSV.

    Returns
    -------
    pd.DataFrame
        Combined results across all sweeps.
    """
    future_heatwave_year_list = [2026, 2027, 2028, 2029, 2030]
    future_heatwave_month_list = [6, 7]
    historical_heatwave_year_list = [2019, 2022, 2024]
    hot_hour_list = [12, 13, 14, 15]
    num_his_heat_event = 2
    num_future_heat_event = 2

    futurel_hot_day_list = np.load(
        f'models/{country_code}/simu_data/future_hot_dates.npy',
        allow_pickle=True)
    year_set = set(future_heatwave_year_list)
    month_set = set(future_heatwave_month_list)
    cnt = defaultdict(int)
    hot_days = []
    for d in futurel_hot_day_list:
        if d.year in year_set and d.month in month_set:
            ym = (d.year, d.month)
            if cnt[ym] < num_future_heat_event:
                hot_days.append(d)
                cnt[ym] += 1
    future_hot_date_list = [
        datetime(d.year, d.month, d.day, h)
        for d in hot_days
        for h in hot_hour_list
    ]

    all_results = []

    for tdpf_analysis in analysis_list:
        if 'td_sin' in tdpf_analysis:
            conductor['convective_correction'] = 1
            conductor['radiactive_correction'] = 1

        param_list = []
        for his_year in historical_heatwave_year_list:
            for his_idx in range(num_his_heat_event):
                for fut_date in future_hot_date_list:
                    params = (
                        tdpf_analysis, fut_date, country_code, nbus,
                        args, conductor, weather, his_year, his_idx,
                    )
                    param_list.append(params)
                    if not args['parallel_mode']:
                        df = run_single_simulation(params)
                        all_results.append(df)

        if args['parallel_mode']:
            num_sims = len(param_list)
            num_procs = min(mp.cpu_count() // 2, num_sims)
            print(f"Running {num_sims} simulations in parallel "
                  f"using {num_procs} processes...")
            with mp.Pool(processes=num_procs) as pool:
                chunksize = max(1, num_sims // (num_procs * 4))
                results = pool.map(
                    run_single_simulation, param_list,
                    chunksize=chunksize)
            all_results.extend(results)

        # Save after each analysis type
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(df_path, index=False)
        print(f"{tdpf_analysis} simulations completed. "
              f"Results saved to {df_path}")

    return df_all


# ===========================================================================
# 6. Entry Point & Parameter Sweeps
# ===========================================================================

def main(country_code='ES'):
    """Run the full TD-ACOPF analysis pipeline for *country_code*.

    The function defines default ACOPF parameters, conductor
    properties (Al/St 240/40 four-bundle 380 kV), and IEEE-standard
    weather initialisations, then runs multiple sensitivity sweeps:

    * **Load-growth analysis** — 2 % and 3 % annual growth.
    * **Thermal-limit analysis** — 120, 150, 180 °C limits.
    * **Model comparison** — base, quadratic, segmented, derated,
      iterative, and security-constrained variants.

    For multi-country lists the function runs a single sensitivity
    sweep with varying load growth.

    Parameters
    ----------
    country_code : str or list[str]
        Two-letter country code(s).
    """
    # --- ACOPF solver parameters ---
    args = {
        'date': None,
        'BaseMVA': 100.0,
        'phase_factor': 3.0,            # balanced three-phase ACOPF
        'renewable_mode': True,
        'heatwave_mode': True,
        'storage_mode': True,
        'DLR_mode': False,
        'load_growth': True,             # base-load growth to 2050
        'reactive_demand_ratio': 0.15,
        'reactive_gen_upper': 0.8,
        'reactive_gen_lower': -0.8,
        'voltage_upper': 1.05,
        'voltage_lower': 0.95,
        'phase_angle_upper': 30,
        'phase_angle_lower': -30,
    }

    # --- Al/St 240/40 four-bundle 380 kV conductor ---
    conductor = {
        'diameter': 18.881e-3,           # m  (√((240+40)/π) × 2)
        'num_bundle': 4,
        'ref_temperature': 20.0,         # °C
        'max_temperature': 90.0,         # °C
        'resistance_ratio': 0.00429,     # Ω/°C
        'unit_resistance': 0.03e-3 * 4,  # Ω/m (four-bundle)
        'conductor_angle': 0.0,
        'elevation': None,
        'convective_correction': 0.8,
        'radiactive_correction': 0.8,
    }
    args['max_temperature'] = conductor['max_temperature']

    # --- IEEE-standard weather initialisation ---
    weather = {
        'wind_speed': None,              # m/s (populated per timestep)
        'wind_height': 10,
        'wind_angle': 90.0,             # °
        'air_density': 1.029,           # kg/m³
        'air_viscosity': 2.043e-5,      # kg/(m·s)
        'air_conductivity': 0.02945,    # W/(m·°C)
        'air_temperature': None,        # °C
        'radiation_emissivity': 0.8,
        'solar_absorptivity': 0.8,
        'solar_heat_intensity': None,   # W/m²
    }

    args['parallel_mode'] = False

    # --- Load network ---
    network, _ = load_network_EU(country_code, RATIO)
    nbus = network.buses.shape[0]

    if isinstance(country_code, list):
        # ---- Multi-country: load-growth sensitivity ----
        analysis_list = ['td_seg_derate_iter_2']
        args['max_temperature'] = conductor['max_temperature'] = 90
        for storage_ratio, load_growth_rate in [
            [0.8, 1.01], [0.8, 1.02], [0.8, 1.03],
        ]:
            args['load_growth_rate'] = load_growth_rate
            args['storage_ratio'] = storage_ratio
            df_path = setup_heatwave_analysis(
                country_code, args, nbus,
                label='sensitivity_analysis')
            run_parallel_simulations(
                analysis_list, country_code, nbus,
                args, conductor, weather, df_path)
    else:
        # ---- Single country ----
        analysis_list = ['td_seg_derate_iter_2']

        # 1) Load-growth analysis: 2 %, 3 %
        args['max_temperature'] = conductor['max_temperature'] = 90
        args['storage_ratio'] = 0.8
        for load_growth_rate in [1.02, 1.03]:
            args['load_growth_rate'] = load_growth_rate
            df_path = setup_heatwave_analysis(
                country_code, args, nbus,
                label='sensitivity_analysis')
            run_parallel_simulations(
                analysis_list, country_code, nbus,
                args, conductor, weather, df_path)

        # 2) Thermal-limit analysis: 120, 150, 180 °C
        args['max_temperature'] = conductor['max_temperature'] = 90
        args['load_growth_rate'] = 1.01
        args['storage_ratio'] = 0.8
        for thermal_limit in [120, 150, 180]:
            args['max_temperature'] = thermal_limit
            conductor['max_temperature'] = thermal_limit
            df_path = setup_heatwave_analysis(
                country_code, args, nbus,
                label='thermal_analysis')
            run_parallel_simulations(
                analysis_list, country_code, nbus,
                args, conductor, weather, df_path)

        # 3) Model comparison
        args['max_temperature'] = conductor['max_temperature'] = 90
        args['load_growth_rate'] = 1.01
        args['storage_ratio'] = 0.8
        analysis_list = [
            'base',
            'td_quad',
            'td_seg_derate_iter_2',
            'td_seg_derate_iter_10',
            'base_fixsc',
            'base_seg_derate',
            'td_derate_iter_2',
            'td_seg_iter_2',
            'td_sin_seg_derate_iter_2',
        ]
        df_path = setup_heatwave_analysis(
            country_code, args, nbus,
            label='model_analysis')
        run_parallel_simulations(
            analysis_list, country_code, nbus,
            args, conductor, weather, df_path)


if __name__ == "__main__":
    # Single-country runs
    # small: 'PT', 'NL', 'BE'  |  mid: 'IT', 'ES', 'GB'  |  large: 'FR', 'DE'
    for country_code in ['ES']:
        main(country_code)

    # Multi-country (cross-border) runs
    # for country_list in [['ES', 'PT'], ['ES', 'FR'], ['FR', 'ES'],
    #                      ['FR', 'IT'], ['FR', 'GB'], ['FR', 'DE']]:
    #     main(country_list)

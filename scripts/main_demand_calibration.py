#!/usr/bin/env python3
"""Demand calibration using the BAIT (Bio-meteorological Apparent Index
of Temperature) model.

This script calibrates country-level electricity demand models against
ENTSO-E historical load data and ERA5 weather fields.  The calibration
optimises 13 parameters (heating/cooling coefficients, temperature
thresholds, solar/wind/humidity adjustments, etc.) using the Shuffled
Complex Evolution Metropolis (SCEM) algorithm.

Pipeline
--------
1. Load ENTSO-E hourly demand and filter outliers (3.5-sigma rule).
2. Load daily-averaged ERA5 weather and PyPSA network bus regions.
3. Build training features (BAIT, HDD/CDD, weekday flags).
4. Run SCEM optimisation (or load existing parameters).
5. Compute hourly demand-shape ratios and save the calibrated model.
6. Visualise demand curves and temperature–BAIT relationships.

Usage
-----
    python test_demand_calibration.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import holidays  # noqa: F401 — used implicitly via get_holiday_list
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# Project imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_config import *                # noqa: F403
from utils.network_process_utils import *  # noqa: F403
from utils.demand_utils import _bait, compute_rmse, compute_mape, compute_std, SCEM
from utils.heatwave_utils import *       # noqa: F403

sns.set_style("white")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGMA_MULTIPLIER = 3.5  # Outlier detection threshold (lower tail only)
TRIM_PERCENT = 1        # Percentile trim for robust mean / std

# Colour palette for demand curve plots
red = '#d62728'    # model curve – weekday
blue = '#1f77b4'   # model curve – weekend
lred = '#ff9896'   # scatter – weekday observed
lblue = '#aec7e8'  # scatter – weekend observed

PARAM_NAMES = [
    'Ph', 'Pc', 'Th', 'Tc', 'solar_gains', 'wind_chill',
    'humidity_discomfort', 'smoothing', 'Pb', 'alpha',
    'lower_blend', 'upper_blend', 'max_raw_var',
]


# ===========================================================================
# Data Loading & Filtering
# ===========================================================================


def robust_mean_std(data, trim_percent=TRIM_PERCENT):
    """Compute mean and std after trimming extreme percentiles.

    Parameters
    ----------
    data : array-like
        Input data.
    trim_percent : float
        Percentage to trim from each tail.

    Returns
    -------
    mean : float
    std : float
    """
    lower = np.percentile(data, trim_percent)
    upper = np.percentile(data, 100 - trim_percent)
    trimmed = data[(data >= lower) & (data <= upper)]
    return np.mean(trimmed), np.std(trimmed, ddof=1)


def filter_abnormal_values(data, replace_method='nearest'):
    """Detect lower-tail outliers via 3.5-sigma rule and replace them.

    Only the lower tail is flagged (demand dips from outages or data
    errors); upper-tail values are kept.

    Parameters
    ----------
    data : np.ndarray
        Load values.
    replace_method : {'nearest'}
        Replace each outlier with its nearest normal value.

    Returns
    -------
    filtered_data : np.ndarray
    abnormal_indices : np.ndarray
    lower_bound : float
    """
    mean, std = robust_mean_std(data)
    lower_bound = mean - SIGMA_MULTIPLIER * std
    abnormal_mask = data < lower_bound
    abnormal_indices = np.where(abnormal_mask)[0]

    filtered_data = data.copy()
    if replace_method == 'nearest' and len(abnormal_indices) > 0:
        normal_indices = np.where(~abnormal_mask)[0]
        for idx in abnormal_indices:
            distances = np.abs(normal_indices - idx)
            filtered_data[idx] = data[normal_indices[np.argmin(distances)]]

    return filtered_data, abnormal_indices, lower_bound


def load_and_filter_country_data(selected_country, year_list):
    """Load ENTSO-E demand and apply outlier filtering.

    Parameters
    ----------
    selected_country : str
        Two-letter country code (e.g. ``'DE'``).
    year_list : list of int
        ``[start_year, end_year]``.

    Returns
    -------
    country_load_value : np.ndarray
        Filtered hourly load (MW).
    date_list_load : pd.DatetimeIndex
    country_load_value_original : np.ndarray
    abnormal_indices : np.ndarray
    """
    load_all = pd.read_parquet(f"{DEMAND}/MHLV_2015-2024.parquet")
    country_load = load_all[load_all['CountryCode'] == selected_country]
    country_load = country_load[
        (country_load['DateShort'] >= pd.to_datetime(f'{year_list[0]}-01-01')) &
        (country_load['DateShort'] <= pd.to_datetime(f'{year_list[1]}-12-31'))
    ]

    date_list_load = pd.to_datetime(
        country_load['DateShort'], format='%Y-%m-%d %H:%M')
    country_load_value = country_load['Value'].values.reshape(-1)
    country_load_value_original = country_load_value.copy()

    country_load_value, abnormal_indices, _ = filter_abnormal_values(
        country_load_value)

    if len(abnormal_indices) > 0:
        print(f"  Abnormal values (first 10): "
              f"{country_load_value_original[abnormal_indices[:10]]}")
        print(f"  Replaced with:              "
              f"{country_load_value[abnormal_indices[:10]]}")

    return (country_load_value, date_list_load,
            country_load_value_original, abnormal_indices)


# ===========================================================================
# Weather & Network Data
# ===========================================================================


def load_weather_and_network_data(selected_country, year_list):
    """Load daily ERA5 weather and PyPSA bus-region population ratios.

    Parameters
    ----------
    selected_country : str
    year_list : list of int
        ``[start_year, end_year]``.

    Returns
    -------
    temp_data, wind_data, solar_data, humidity_data : np.ndarray
        Shape ``(n_days, n_regions)``.
    date_list : np.ndarray of datetime.date
    pop_ratio : np.ndarray
        Normalised population weights per bus region.
    """
    network, regions = load_network_EU([selected_country])
    pop_ratio = network.buses.loc[regions.index, 'pop_ratio'].values
    pop_ratio = pop_ratio / pop_ratio.sum()

    ncfile = xr.open_dataset(
        WEATHER_DAILY + '/era5_daily_avg_2015_2024.nc')
    date_s = datetime(year_list[0], 1, 1)
    date_e = datetime(year_list[-1], 12, 31, 23, 59, 59)
    x_min = network.buses['x'].min() - 1
    x_max = network.buses['x'].max() + 1
    y_min = network.buses['y'].min() - 1
    y_max = network.buses['y'].max() + 1
    ncfile_slice = ncfile.sel(
        time=slice(date_s, date_e),
        x=slice(x_min, x_max),
        y=slice(y_min, y_max),
    )

    region_ave = aggregate_regional_data(ncfile_slice, regions)

    temp_data = region_ave['temperature'].data - TK
    wind_data = region_ave['wnd10m'].data
    solar_data = region_ave['influx'].data
    humidity_data = region_ave['humidity'].data
    date_list = region_ave.time.data.astype('datetime64[D]').astype('O')

    return temp_data, wind_data, solar_data, humidity_data, date_list, pop_ratio


# ===========================================================================
# Training Data Preparation
# ===========================================================================


def prepare_training_data(selected_country, country_load_value,
                          date_list_load, date_list, temp_data, wind_data,
                          solar_data, humidity_data, pop_ratio):
    """Align daily load with weather features and compute calendar flags.

    Only days with exactly 24 positive hourly readings are kept.

    Returns
    -------
    load : np.ndarray
        Daily average demand (GW).
    delta_year : np.ndarray
    temp, wind, solar, humidity : np.ndarray
    weekday_index, weekend_index : np.ndarray (int)
    hourly_load : np.ndarray
    valid_date : np.ndarray
    training_data_df : pd.DataFrame
    """
    training_data_df = pd.DataFrame(columns=['date', 'load'])
    hourly_load = []
    valid_date_index = []

    for t in range(date_list.shape[0]):
        year = date_list[t].year
        month = date_list[t].month
        day = date_list[t].day
        day_of_week = date_list[t].weekday()

        holiday_list = get_holiday_list(selected_country, year)
        holiday_flag = float(
            date_list[t].strftime('%Y-%m-%d') in holiday_list)

        daily_load = country_load_value[
            (date_list_load.dt.year == year) &
            (date_list_load.dt.month == month) &
            (date_list_load.dt.day == day)
        ]

        if daily_load.shape[0] == 24 and daily_load.min() > 0:
            valid_date_index.append(t)
            hourly_load.append(np.reshape(daily_load, (1, -1)))
            training_data_df = training_data_df._append({
                'date': f'{year}-{month}-{day}',
                'year': year,
                'day_of_week': day_of_week,
                'holiday': holiday_flag,
                'load': daily_load.sum() / 1e3 / 24,
            }, ignore_index=True)

    valid_date_index = np.array(valid_date_index)
    valid_date = date_list[valid_date_index]
    hourly_load = np.array(hourly_load)

    weekday_index = (
        (training_data_df['day_of_week'] < 5) &
        (training_data_df['holiday'] == 0)
    ).values.astype(int)
    weekend_index = (
        (training_data_df['day_of_week'] >= 5) |
        (training_data_df['holiday'] == 1)
    ).values.astype(int)

    load = training_data_df['load'].values
    delta_year = training_data_df['year'].values - BASELINE_YEAR

    return (load, delta_year,
            temp_data[valid_date_index], wind_data[valid_date_index],
            solar_data[valid_date_index], humidity_data[valid_date_index],
            weekday_index, weekend_index, hourly_load, valid_date,
            training_data_df)


# ===========================================================================
# Optimisation Setup & Execution
# ===========================================================================


def setup_optimization_problem(load, temp, wind, solar, humidity,
                               delta_year, valid_date, weekday_index,
                               pop_ratio, x_init):
    """Define the SCEM fitness function and parameter bounds.

    Parameters
    ----------
    load, temp, wind, solar, humidity : np.ndarray
        Training features.
    delta_year : np.ndarray
    valid_date : np.ndarray
    weekday_index : np.ndarray
    pop_ratio : np.ndarray
    x_init : np.ndarray or None
        Warm-start vector in *original* scale; ``None`` → midpoint.

    Returns
    -------
    problem, options : dict
    xl, xu, x_range : np.ndarray
    """
    Pl, Pu = load.min(), load.max()

    # Bounds (centre ± offset notation retained for readability):
    # [Ph, Pc, Th, Tc, solar, wind_chill, humidity, smooth,
    #  Pb, alpha, lo_blend, hi_blend, max_var]
    xl = np.array([
        0,   0,  5,  17,
        0,              -0.13 - 0.25,  0,              0,
        Pl,  0,  15 - 3,  23 - 3,  0.5 - 0.3,
    ])
    xu = np.array([
        2.5, 2., 17, 23,
        0.019 + 0.1,    0,             0.05 + 0.05,    1,
        Pu,  Pl, 15 + 3,  23 + 3,  0.5 + 0.3,
    ])
    x_range = xu - xl

    if x_init is not None:
        x_init = (x_init - xl) / x_range
    else:
        x_init = 0.5 * np.ones(len(x_range))

    def fitness_function(x):
        params = x * x_range + xl
        Ph, Pc, Th, Tc, solar_gains, wind_chill, \
            humidity_discomfort, smoothing, Pb, alpha, \
            lower_blend, upper_blend, max_raw_var = params

        para = {
            'solar_gains': solar_gains, 'wind_chill': wind_chill,
            'humidity_discomfort': humidity_discomfort,
            'smoothing': smoothing,
            'lower_blend': lower_blend, 'upper_blend': upper_blend,
            'max_raw_var': max_raw_var,
        }
        bait = _bait(temp, wind, solar, humidity, para,
                     valid_date=valid_date)
        HDD = np.maximum(Th - bait, 0)
        CDD = np.maximum(bait - Tc, 0)
        demand = (Pb
                  + Ph * (HDD * pop_ratio).sum(-1)
                  + Pc * (CDD * pop_ratio).sum(-1)
                  + alpha * weekday_index)
        return compute_rmse(load, demand) + compute_mape(load, demand)

    problem = {
        'fitness_function': fitness_function,
        'ndim_problem': len(x_range),
        'lower_boundary': np.zeros(len(x_range)),
        'upper_boundary': np.ones(len(x_range)),
    }
    options = {
        'max_function_evaluations': 50000,
        'seed_rng': 2030,
        'mean': x_init,
        'alpha': 0.25,
        'sigma': 0.5,
        'n_individuals': max(1000, 4 * len(x_range)),
        'n_parents': max(20, len(x_range)),
    }
    return problem, options, xl, xu, x_range


def run_optimization(problem, options, selected_country):
    """Run SCEM and return the results dict."""
    opt = SCEM(problem, options)
    results = opt.optimize()
    print(f"  {selected_country}: evals={results['n_function_evaluations']}, "
          f"best_y={results['best_so_far_y']:.6f}")
    return results


def decode_results(results, x_range, xl):
    """Decode normalised optimisation output into a parameter dict."""
    values = results['best_so_far_x'] * x_range + xl
    para = dict(zip(PARAM_NAMES, values))
    para['growth_rate'] = 0  # Disabled; kept for interface compatibility
    return para


# ===========================================================================
# Post-Processing
# ===========================================================================


def process_optimization_results(para, temp, wind, solar, humidity,
                                 pop_ratio, weekday_index, hourly_load,
                                 selected_country, year_list, valid_date):
    """Compute hourly demand-shape ratios and add them to *para*.

    The ratios are derived from the top-10-percentile heating / cooling
    days, split by weekday vs weekend.
    """
    bait = _bait(temp, wind, solar, humidity, para, valid_date=valid_date)
    HDD = (np.maximum(para['Th'] - bait, 0) * pop_ratio).sum(-1)
    CDD = (np.maximum(bait - para['Tc'], 0) * pop_ratio).sum(-1)

    wk_mask = weekday_index == 1
    we_mask = weekday_index == 0

    def _top_shape(mask, degree_days):
        """Mean hourly profile of top-10 % degree-day subset."""
        subset = hourly_load[mask]
        dd = degree_days[mask]
        profile = np.mean(subset[dd > np.percentile(dd, 90)], axis=0)
        return profile / profile.sum()

    para['weekday_heating_hour_ratio'] = _top_shape(wk_mask, HDD)
    para['weekend_heating_hour_ratio'] = _top_shape(we_mask, HDD)
    para['weekday_cooling_hour_ratio'] = _top_shape(wk_mask, CDD)
    para['weekend_cooling_hour_ratio'] = _top_shape(we_mask, CDD)

    return para


# ===========================================================================
# Visualisation
# ===========================================================================


def visualize_filtering_results(country_load_original, country_load_filtered,
                                date_list_load, abnormal_indices,
                                selected_country):
    """Side-by-side time-series and histogram of original vs filtered load."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    mean, std = robust_mean_std(country_load_original)
    lower_bound = mean - SIGMA_MULTIPLIER * std
    upper_bound = mean + SIGMA_MULTIPLIER * std

    # Time series
    axes[0].plot(date_list_load, country_load_original,
                 alpha=0.7, color='red', linewidth=1, label='Original')
    axes[0].plot(date_list_load, country_load_filtered,
                 alpha=0.8, color='blue', linewidth=1, label='Filtered')
    if len(abnormal_indices) > 0:
        axes[0].scatter(
            date_list_load.iloc[abnormal_indices],
            country_load_original[abnormal_indices],
            color='red', s=20, alpha=0.8,
            label=f'Abnormal ({len(abnormal_indices)})')
    axes[0].set_title(f'{selected_country} Load: Original vs Filtered')
    axes[0].set_ylabel('Load (MW)')
    axes[0].legend()
    axes[0].grid(linewidth=0.4, alpha=0.5)

    # Histogram
    axes[1].hist(country_load_original, bins=50, alpha=0.6,
                 color='red', label='Original', density=True)
    axes[1].hist(country_load_filtered, bins=50, alpha=0.6,
                 color='blue', label='Filtered', density=True)
    axes[1].axvline(lower_bound, color='black', linestyle='--',
                    alpha=0.8, label=f'{SIGMA_MULTIPLIER}σ bounds')
    axes[1].axvline(upper_bound, color='black', linestyle='--', alpha=0.8)
    axes[1].set_title('Load Distribution: Original vs Filtered')
    axes[1].set_xlabel('Load (MW)')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.show()


def visualize_demand_curve(para, temp, wind, solar, humidity, delta_year,
                           pop_ratio, load, weekday_index, selected_country,
                           year_list, valid_date,
                           show_legend=False, show_title=True):
    """Scatter plot of BAIT vs daily demand with fitted model curves."""
    Ph, Pc = para['Ph'], para['Pc']
    Th, Tc = para['Th'], para['Tc']
    Pb, alpha = para['Pb'], para['alpha']

    plt.figure(figsize=(6, 5))

    bait = _bait(temp, wind, solar, humidity, para, valid_date=valid_date)
    HDD = np.maximum(Th - bait, 0)
    CDD = np.maximum(bait - Tc, 0)
    demand = (Pb
              + Ph * (HDD * pop_ratio).sum(-1)
              + Pc * (CDD * pop_ratio).sum(-1)
              + alpha * weekday_index)

    # Observed data scatter
    wk = weekday_index == 1
    we = weekday_index == 0
    weekday_bait = (bait[wk] * pop_ratio).sum(-1)
    weekend_bait = (bait[we] * pop_ratio).sum(-1)

    plt.scatter(weekday_bait, load[wk],
                marker='o', color=lred, linewidth=0, alpha=0.8, s=10)
    plt.scatter(weekend_bait, load[we],
                marker='s', color=lblue, linewidth=0, alpha=0.8, s=10)

    # Model curves
    bait_avg = (bait * pop_ratio).sum(-1)
    bait_sim = np.linspace(bait_avg.min(), bait_avg.max(), 1000)
    work_demand = (Pb + Ph * np.maximum(Th - bait_sim, 0)
                   + Pc * np.maximum(bait_sim - Tc, 0) + alpha)
    week_demand = (Pb + Ph * np.maximum(Th - bait_sim, 0)
                   + Pc * np.maximum(bait_sim - Tc, 0))
    plt.scatter(bait_sim, work_demand, marker='.', alpha=0.5, color=red, s=10)
    plt.scatter(bait_sim, week_demand, marker='.', alpha=0.5, color=blue, s=10)

    # Error metrics
    pred_load = np.zeros(load.shape)
    pred_load[wk] = demand[wk]
    pred_load[we] = demand[we]

    print(f"  Weekday — MAPE: {compute_mape(load[wk], demand[wk]):.4f}, "
          f"RMSE: {compute_rmse(load[wk], demand[wk]):.4f}")
    print(f"  Weekend — MAPE: {compute_mape(load[we], demand[we]):.4f}, "
          f"RMSE: {compute_rmse(load[we], demand[we]):.4f}")
    print(f"  Overall — MAPE: {compute_mape(load, pred_load):.4f}, "
          f"RMSE: {compute_rmse(load, pred_load):.4f}")

    plt.text(
        bait_avg.min(), load.min(),
        f'RMSE = {compute_rmse(load, pred_load):.2f}\n'
        f'MAPE = {compute_mape(load, pred_load) * 100:.2f}%',
        bbox={'facecolor': 'lightgray', 'alpha': 0.1, 'pad': 10},
        fontsize=14,
    )

    # Legend
    if show_legend:
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=lred,
                   markersize=8, linewidth=0, alpha=0.99),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=lblue,
                   markersize=8, linewidth=0, alpha=0.99),
            Line2D([0], [0], color=red, linewidth=2, alpha=0.8),
            Line2D([0], [0], color=blue, linewidth=2, alpha=0.8),
        ]
        labels = ['Weekday demand', 'Weekend demand',
                  'Model prediction', 'Model prediction']
        plt.legend(handles=handles, labels=labels, ncol=2,
                   columnspacing=0.75, loc=9, fontsize=14)

    if show_title:
        plt.title(f'{country_name[selected_country]}',
                  fontsize=22, fontweight='bold')

    plt.ylabel('Daily Demand (GW)', fontsize=18, fontweight='bold')
    plt.xlabel('BAIT (°C)', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16)

    n_ticks = 10
    min_val = int(load.min() * 0.9 // 1)
    max_val = int(load.max() * 1.1 // 1 + 1)
    plt.yticks(np.linspace(min_val, max_val, n_ticks).astype(int),
               fontsize=16)
    plt.ylim(load.min() * 0.9, load.max() * 1.1)
    plt.tick_params(axis='both', which='both',
                    top=False, bottom=True, left=True, right=False)
    plt.grid(linewidth=0.5, alpha=0.25)


def visualize_temp_bait_relationship(para, temp, wind, solar, humidity,
                                     pop_ratio, selected_country,
                                     year_list, valid_date):
    """Scatter temperature vs BAIT, colour-coded by wind / solar / humidity."""
    bait = _bait(temp, wind, solar, humidity, para, valid_date=valid_date)
    bait_avg = (bait * pop_ratio).sum(-1)
    ave_temp = (temp * pop_ratio).sum(-1)

    weather_vars = {
        'ave_wind': (
            (wind * pop_ratio).sum(-1),
            plt.cm.viridis,
            'Wind (m/s)',
        ),
        'ave_solar': (
            (solar * pop_ratio).sum(-1),
            plt.cm.plasma,
            r'Solar (W/m$^2$)',
        ),
        'ave_humidity': (
            (humidity * pop_ratio).sum(-1),
            plt.cm.winter,
            'Specific Humidity (g/kg)',
        ),
    }

    models_dir = os.path.join(MODELS, "demand_curve", selected_country)

    for key, (values, cmap, label) in weather_vars.items():
        plt.figure(figsize=(6, 7))
        plt.scatter(ave_temp, bait_avg, marker='o', c=values,
                    linewidth=0, alpha=0.75, s=20, cmap=cmap)
        plt.plot([ave_temp.min(), ave_temp.max()],
                 [ave_temp.min(), ave_temp.max()],
                 color='black', linewidth=1)

        cb = plt.colorbar(orientation='horizontal', pad=0.08,
                          aspect=30, shrink=0.9, location='top')
        cb.set_label(label, size=18, labelpad=10, fontweight='bold')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(labelsize=14)

        plt.xlim(ave_temp.min(), ave_temp.max())
        plt.ylim(ave_temp.min(), ave_temp.max())
        plt.xlabel('Temperature (°C)', fontsize=18, fontweight='bold')
        plt.ylabel('BAIT (°C)', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(linewidth=0.5, alpha=0.25)
        plt.savefig(
            os.path.join(models_dir,
                         f'{selected_country}_{year_list}_temp_bait_{key}.pdf'),
            bbox_inches='tight')
        plt.close()


# ===========================================================================
# Orchestration
# ===========================================================================


def main(selected_country, year_list, calibration=True):
    """Run the full demand calibration pipeline for one country.

    Parameters
    ----------
    selected_country : str
        Two-letter country code.
    year_list : list of int
        ``[start_year, end_year]``.
    calibration : bool
        If ``True``, run SCEM optimisation.  If ``False``, load existing
        parameters from disk.
    """
    print(f"\n{'=' * 60}")
    print(f"Demand calibration: {selected_country} "
          f"({year_list[0]}–{year_list[1]})")
    print('=' * 60)

    models_dir = os.path.join(MODELS, "demand_curve", selected_country)
    os.makedirs(models_dir, exist_ok=True)
    demand_curve_path = os.path.join(
        models_dir, f'{selected_country}_{year_list}_demand_curve.npy')

    # 1. Load and filter demand data
    print("\n1. Loading and filtering country data...")
    (country_load_value, date_list_load,
     country_load_original, abnormal_indices) = \
        load_and_filter_country_data(selected_country, year_list)

    # 2. Load weather and network data
    print("\n2. Loading weather and network data...")
    (temp_data, wind_data, solar_data, humidity_data,
     date_list, pop_ratio) = \
        load_weather_and_network_data(selected_country, year_list)

    # 3. Prepare training features
    print("\n3. Preparing training data...")
    (load, delta_year, temp, wind, solar, humidity,
     weekday_index, weekend_index, hourly_load, valid_date,
     training_data_df) = prepare_training_data(
        selected_country, country_load_value, date_list_load, date_list,
        temp_data, wind_data, solar_data, humidity_data, pop_ratio)

    if calibration:
        # 4. Warm-start from existing model if available
        x_init = None
        if os.path.exists(demand_curve_path):
            existing = np.load(
                demand_curve_path, allow_pickle=True).item()
            x_init = np.array([existing[k] for k in PARAM_NAMES])

        print("\n4. Setting up optimisation problem...")
        problem, options, xl, xu, x_range = setup_optimization_problem(
            load, temp, wind, solar, humidity, delta_year,
            valid_date, weekday_index, pop_ratio, x_init)

        # 5. Run optimisation
        print("\n5. Running optimisation...")
        results = run_optimization(problem, options, selected_country)
        para = decode_results(results, x_range, xl)
    else:
        # Load pre-calibrated parameters
        print("\n4. Loading existing parameters...")
        para = np.load(demand_curve_path, allow_pickle=True).item()

    # 6. Compute hourly shape ratios and save
    print("\n6. Processing results and saving...")
    para = process_optimization_results(
        para, temp, wind, solar, humidity, pop_ratio,
        weekday_index, hourly_load, selected_country, year_list, valid_date)
    np.save(demand_curve_path, para, allow_pickle=True)

    print(f'\n  Calibrated parameters:')
    print(f'  Pb={para["Pb"]:.2f}, Ph={para["Ph"]:.2f}, '
          f'Pc={para["Pc"]:.2f}, Th={para["Th"]:.1f}, Tc={para["Tc"]:.1f}')
    print(f'  solar={para["solar_gains"]:.4f}, '
          f'wind={para["wind_chill"]:.4f}, '
          f'humidity={para["humidity_discomfort"]:.4f}, '
          f'smooth={para["smoothing"]:.2f}')
    print(f'  alpha={para["alpha"]:.2f}, '
          f'blend=[{para["lower_blend"]:.1f}, {para["upper_blend"]:.1f}], '
          f'max_var={para["max_raw_var"]:.2f}')

    # 7. Visualise
    print("\n7. Creating visualisations...")
    visualize_demand_curve(
        para, temp, wind, solar, humidity, delta_year, pop_ratio, load,
        weekday_index, selected_country, year_list, valid_date)
    plt.savefig(
        os.path.join(models_dir,
                     f'{selected_country}_{year_list}_demand_curve.pdf'),
        dpi=300, bbox_inches='tight')

    visualize_temp_bait_relationship(
        para, temp, wind, solar, humidity, pop_ratio,
        selected_country, year_list, valid_date)

    print(f"\nDone: {selected_country}")


# ===========================================================================
# Entry Points
# ===========================================================================

def run_sequential():
    """Calibrate all countries sequentially."""
    for country in ['PT', 'ES', 'FR', 'IT', 'DE', 'BE', 'NL', 'GB']:
        yl = [2015, 2020] if country == 'GB' else [2015, 2024]
        main(country, yl, calibration=False)


def _process_country_year(args):
    """Worker for multiprocessing — unpack args and call ``main()``."""
    country, year_list = args
    main(country, year_list)


def run_multiprocess(n_workers=4):
    """Calibrate all countries in parallel using a process pool.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes.
    """
    tasks = [
        ('PT', [2015, 2024]),
        ('ES', [2015, 2024]),
        ('FR', [2015, 2024]),
        ('IT', [2015, 2024]),
        ('DE', [2015, 2024]),
        ('BE', [2015, 2024]),
        ('NL', [2015, 2024]),
        ('GB', [2015, 2020]),
    ]
    with Pool(processes=n_workers) as pool:
        pool.map(_process_country_year, tasks)


if __name__ == "__main__":
    # Switch to run_multiprocess(n_workers=4) for parallel execution
    run_sequential()

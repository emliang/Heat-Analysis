"""Heatwave scenario construction utilities.

This module provides the data-processing pipeline for building
bias-corrected future heatwave weather profiles from ERA5 reanalysis
and RCP 4.5 climate projections:

1. **Spatial aggregation** — map gridded weather fields to bus-level
   regional averages via ``atlite`` indicator matrices.
2. **Temporal interpolation** — up-sample 3-hourly RCP 4.5 data to
   hourly resolution.
3. **Heatwave ranking** — identify the most severe days using
   reciprocal-rank fusion of temperature and wind indices.
4. **Spatial smoothing** — 2-D / 3-D convolution for bias fields.
5. **Bias correction** — delta-mapping (additive for temperature /
   solar, multiplicative for wind) with physical guard-rails.

Section index
-------------
1. Spatial Aggregation
2. Temporal Trend Extraction
3. Heatwave Day Ranking
4. Temporal Interpolation
5. Spatial Smoothing
6. Bias Correction
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr

import atlite
from scipy import sparse
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

from data_config import *  # noqa: F403  (provides TK, etc.)


# ===========================================================================
# 1. Spatial Aggregation
# ===========================================================================

def aggregate_regional_data(ncfile_slice, regions, variables=None):
    """Aggregate gridded weather data to bus-level regional averages.

    Uses an ``atlite.Cutout`` indicator matrix to compute
    area-weighted averages of each weather variable over the
    geometries defined in *regions*.

    Parameters
    ----------
    ncfile_slice : xr.Dataset
        Gridded weather data (must contain the requested variables).
    regions : gpd.GeoDataFrame
        Bus regions with geometry used for spatial weighting.
    variables : list[str] or None
        Variables to aggregate.  Defaults to
        ``['temperature', 'wnd10m', 'influx', 'humidity']``.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(time, regions)`` containing the
        weighted regional averages for each variable.
    """
    buses = regions.index

    cutout = atlite.Cutout(path='/default.nc', data=ncfile_slice)

    if variables is None:
        variables = ['temperature', 'wnd10m', 'influx', 'humidity']

    # Indicator matrix mapping grid cells → regions
    matrix = cutout.indicatormatrix(regions.geometry)

    # When there is only one time step xarray stores time as a scalar
    time_coord = cutout.data.time
    if time_coord.ndim == 0:
        time_coord = [time_coord.values]

    ds_result = xr.Dataset(
        coords={'regions': buses.values, 'time': time_coord})

    y_size = len(cutout.data.coords['y'])
    x_size = len(cutout.data.coords['x'])

    for variable in variables:
        var_data = cutout.data[variable].values

        # Reshape to (time, space)
        if var_data.ndim == 3:
            n_timesteps = var_data.shape[0]
            var_data_2d = var_data.reshape(n_timesteps, y_size * x_size)
        else:
            n_timesteps = 1
            var_data_2d = var_data.reshape(n_timesteps, y_size * x_size)

        # Area-weighted sum → normalise to get average
        if sparse.issparse(matrix):
            weights_sum = matrix.sum(axis=1).A.flatten()
            weighted_sums = matrix.dot(var_data_2d.T).T
        else:
            weights_sum = matrix.sum(axis=1)
            weighted_sums = var_data_2d @ matrix.T

        mask = weights_sum > 0
        region_averages = np.zeros((n_timesteps, len(weights_sum)))
        region_averages[:, mask] = weighted_sums[:, mask] / weights_sum[mask]

        ds_result[variable] = xr.DataArray(
            region_averages,
            dims=['time', 'regions'],
            coords={'time': time_coord, 'regions': buses.values},
        )

    return ds_result


# ===========================================================================
# 2. Temporal Trend Extraction
# ===========================================================================

def temperal_trend_data(variable, regions,
                        his_era5_ncfile_slice,
                        his_rcp45_ncfile_slice_hourly,
                        fut_rcp45_ncfile_slice_hourly,
                        historical_date, future_date):
    """Compute 24-hour bus-level averages for four weather scenarios.

    Extracts a single day (00:00–23:00) from each of the four scenario
    datasets, applies bias correction to produce the future heatwave
    field, aggregates to bus regions, and converts temperature to °C.

    Parameters
    ----------
    variable : str
        Weather variable name (e.g. ``'temperature'``).
    regions : gpd.GeoDataFrame
        Bus regions for spatial aggregation.
    his_era5_ncfile_slice : xr.Dataset
        Historical ERA5 reanalysis (hourly).
    his_rcp45_ncfile_slice_hourly : xr.Dataset
        Historical RCP 4.5 (hourly, interpolated).
    fut_rcp45_ncfile_slice_hourly : xr.Dataset
        Future RCP 4.5 (hourly, interpolated).
    historical_date, future_date : datetime-like
        Target dates for the historical heatwave and future event.

    Returns
    -------
    bus_his_hw, bus_his_bl, bus_fut_bl, bus_fut_hw : np.ndarray
        Shape ``(24, n_buses)``.  Historical heatwave, historical
        baseline, future baseline, future heatwave.
    """
    # Build day-long time slices
    his_start = pd.Timestamp(historical_date).strftime('%Y-%m-%d-00:00:00')
    his_end   = pd.Timestamp(historical_date).strftime('%Y-%m-%d-23:00:00')
    fut_start = pd.Timestamp(future_date).strftime('%Y-%m-%d-00:00:00')
    fut_end   = pd.Timestamp(future_date).strftime('%Y-%m-%d-23:00:00')

    weather_his_hw = his_era5_ncfile_slice.sel(
        time=slice(his_start, his_end))
    weather_his_bl = his_rcp45_ncfile_slice_hourly.sel(
        time=slice(his_start, his_end))
    weather_fut_bl = fut_rcp45_ncfile_slice_hourly.sel(
        time=slice(fut_start, fut_end))
    weather_fut_hw = bias_correction(
        variable, weather_his_hw, weather_his_bl, weather_fut_bl)

    # Aggregate to bus-level averages
    bus_his_hw = aggregate_regional_data(
        weather_his_hw, regions, variables=[variable])[variable].data
    bus_his_bl = aggregate_regional_data(
        weather_his_bl, regions, variables=[variable])[variable].data
    bus_fut_bl = aggregate_regional_data(
        weather_fut_bl, regions, variables=[variable])[variable].data
    bus_fut_hw = aggregate_regional_data(
        weather_fut_hw, regions, variables=[variable])[variable].data

    # Convert Kelvin → Celsius for temperature
    if variable == 'temperature':
        bus_his_hw -= TK
        bus_his_bl -= TK
        bus_fut_bl -= TK
        bus_fut_hw -= TK

    return bus_his_hw, bus_his_bl, bus_fut_bl, bus_fut_hw


# ===========================================================================
# 3. Heatwave Day Ranking
# ===========================================================================

def reciprocal_rank_fusion(indices_list, weights=[0.5, 0.5], k=100):
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    indices_list : list[np.ndarray]
        Each element is an argsort index array (descending importance).
    weights : list[float]
        Per-list importance weights.
    k : int
        RRF smoothing constant.

    Returns
    -------
    np.ndarray
        Fused ranking (indices sorted by descending combined score).
    """
    n = len(indices_list[0])
    scores = np.zeros(n)

    for indices, weight in zip(indices_list, weights):
        ranks = np.empty_like(indices)
        ranks[indices] = np.arange(1, len(indices) + 1)
        scores += weight / (k + ranks)

    return np.argsort(scores)[::-1]


def find_heatwave_days(weather_ncfile, regions, weights=[0.9, 0.1]):
    """Rank days by heatwave severity using temperature and wind.

    Days with **high temperature** and **low wind** are ranked highest
    via reciprocal-rank fusion.

    Parameters
    ----------
    weather_ncfile : xr.Dataset
        Gridded weather data covering the period of interest.
    regions : gpd.GeoDataFrame
        Bus regions for spatial averaging.
    weights : list[float]
        ``[temperature_weight, wind_weight]`` for RRF.

    Returns
    -------
    heatwave_index : np.ndarray
        Day indices sorted from most to least severe.
    dates : np.ndarray
        Corresponding date stamps.
    """
    bus_temp = aggregate_regional_data(
        weather_ncfile, regions, variables=['temperature'],
    )['temperature'] - TK
    bus_wind = aggregate_regional_data(
        weather_ncfile, regions, variables=['wnd10m'],
    )['wnd10m']

    # Daily, nationally averaged values
    daily_temp = bus_temp.mean(dim='regions').resample(time='1D').mean()
    daily_wind = bus_wind.mean(dim='regions').resample(time='1D').mean()

    temp_index = np.argsort(daily_temp.values)[::-1]   # high → low
    wind_index = np.argsort(daily_wind.values)          # low → high

    heatwave_index = reciprocal_rank_fusion(
        [temp_index, wind_index], weights=weights)

    return heatwave_index, daily_temp.time.values


# ===========================================================================
# 4. Temporal Interpolation
# ===========================================================================

def interpolate_3h_to_1h(rcp45_ncfile, method='linear'):
    """Interpolate 3-hourly RCP 4.5 data to hourly resolution.

    The 3-hourly timestamps are treated as bin midpoints and shifted by
    +1 hour before linear interpolation.  Boundary NaNs are filled with
    nearest values (forward-fill then back-fill).

    Parameters
    ----------
    rcp45_ncfile : xr.Dataset
        3-hourly RCP 4.5 dataset.
    method : str
        Interpolation method passed to ``xr.Dataset.interp``.

    Returns
    -------
    xr.Dataset
        Hourly-resolution dataset.
    """
    rcp45_shifted = rcp45_ncfile.copy()
    rcp45_shifted['time'] = rcp45_shifted['time'] + pd.Timedelta(hours=1)

    # Hourly target range (extend by 2 h to cover the last 3-h bin)
    start = rcp45_ncfile.time.values[0]
    end = rcp45_ncfile.time.values[-1] + pd.Timedelta(hours=2)
    hourly_time = pd.date_range(start=start, end=end, freq='1h')

    hourly = rcp45_shifted.interp(time=hourly_time, method=method)

    # Fill boundary NaNs
    hourly = hourly.ffill(dim='time').bfill(dim='time')

    return hourly


# ===========================================================================
# 5. Spatial Smoothing
# ===========================================================================

def apply_smoothing(data, kernel_size=3, kernel_type='uniform',
                    smooth_time=False, time_window=3):
    """Apply spatial (and optionally temporal) convolution smoothing.

    Parameters
    ----------
    data : np.ndarray
        2-D ``(lat, lon)`` or 3-D ``(time, lat, lon)`` array.
    kernel_size : int
        Side length of the square spatial kernel (odd recommended).
    kernel_type : {'uniform', 'gaussian'}
        Kernel shape.
    smooth_time : bool
        If *True* and *data* is 3-D, also smooth along the time axis.
    time_window : int
        Window length for temporal smoothing.

    Returns
    -------
    np.ndarray
        Smoothed array with the same shape as *data*.

    Raises
    ------
    ValueError
        If *data* is neither 2-D nor 3-D, or *kernel_type* is unknown.
    """
    # --- Build spatial kernel ----------------------------------------------
    if kernel_type == 'uniform':
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    elif kernel_type == 'gaussian':
        sigma = kernel_size / 6.0
        ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
    else:
        raise ValueError("kernel_type must be 'uniform' or 'gaussian'")

    # --- 2-D ---------------------------------------------------------------
    if data.ndim == 2:
        return convolve2d(data, kernel, mode='same', boundary='symm')

    # --- 3-D ---------------------------------------------------------------
    if data.ndim == 3:
        smoothed = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed[i] = convolve2d(
                data[i], kernel, mode='same', boundary='symm')

        if smooth_time:
            if kernel_type == 'uniform':
                time_kernel = np.ones(time_window) / time_window
            else:  # gaussian
                sigma_t = time_window / 6.0
                ax_t = np.arange(
                    -(time_window // 2), time_window // 2 + 1)
                time_kernel = np.exp(-(ax_t ** 2) / (2 * sigma_t ** 2))
                time_kernel /= time_kernel.sum()
            smoothed = convolve1d(
                smoothed, time_kernel, axis=0, mode='nearest')

        return smoothed

    raise ValueError("Data must be 2D or 3D array")


# ===========================================================================
# 6. Bias Correction
# ===========================================================================

def bias_correction(variable_list, weather_his_heatwave,
                    weather_his_baseline, weather_fut_rcp45,
                    smooth_grid=2):
    """Apply delta-mapping bias correction to produce future heatwave fields.

    For temperature and solar irradiance an **additive** correction is
    used; for wind variables a **multiplicative** (log-normal) correction
    is applied.  All bias fields are spatially smoothed before
    application, and the final values are clipped to physically
    plausible ranges.

    Parameters
    ----------
    variable_list : str or list[str]
        Variable name(s) to bias-correct.
    weather_his_heatwave : xr.Dataset
        Historical ERA5 heatwave snapshot.
    weather_his_baseline : xr.Dataset
        Historical RCP 4.5 baseline (same date as heatwave).
    weather_fut_rcp45 : xr.Dataset
        Future RCP 4.5 baseline (target date).
    smooth_grid : int
        Half-width of the spatial smoothing kernel
        (full width = ``2 * smooth_grid + 1``).

    Returns
    -------
    xr.Dataset
        Copy of *weather_his_heatwave* re-stamped to the future date
        with bias-corrected variable fields.
    """
    weather_future = weather_his_heatwave.copy(deep=True)
    weather_future['time'] = weather_fut_rcp45.time.values
    spatial_kernel = 2 * smooth_grid + 1

    if not isinstance(variable_list, list):
        variable_list = [variable_list]

    # Empirical guard-rails to prevent extreme artefacts
    CONSTRAINTS = {
        'temperature': {'bias': (-15, 15),   'final': (-10 + TK, 50 + TK)},
        'influx':      {'bias': (-250, 250), 'final': (0, 950)},
        'wnd10m':      {'bias': (-1, 1),     'final': (0, 40)},
        'wnd100m':     {'bias': (-1, 1),     'final': (0, 50)},
    }

    for variable in variable_list:
        his_hw = weather_his_heatwave[variable].values
        his_bl = weather_his_baseline[variable].values
        fut_bl = weather_fut_rcp45[variable].values

        if variable in ('temperature', 'influx'):
            # Additive bias correction
            bias = np.clip(his_hw - his_bl, *CONSTRAINTS[variable]['bias'])
            smoothed_bias = apply_smoothing(bias, kernel_size=spatial_kernel)
            fut_heatwave = fut_bl + smoothed_bias
        elif variable in ('wnd10m', 'wnd100m'):
            # Multiplicative bias correction (log-normal assumption)
            bias = np.clip(
                np.log(5 + his_hw) - np.log(5 + his_bl),
                *CONSTRAINTS[variable]['bias'],
            )
            smoothed_bias = apply_smoothing(bias, kernel_size=spatial_kernel)
            fut_heatwave = fut_bl * np.exp(smoothed_bias)

        # Physical value clipping
        fut_heatwave = np.clip(fut_heatwave, *CONSTRAINTS[variable]['final'])
        weather_future[variable].values = fut_heatwave

    return weather_future

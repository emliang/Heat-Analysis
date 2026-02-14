"""Demand-modelling utilities: BAIT thermal-comfort model, SCEM optimiser, and error metrics.

This module provides the core building blocks for temperature-dependent
electricity demand calibration and estimation:

1. **SCEM optimiser** — Smoothed Cross-Entropy Method, a derivative-free
   stochastic optimiser used to fit the BAIT demand model parameters.
2. **BAIT model** — Bio-meteorological Apparent Index of Temperature,
   translating weather conditions into a thermal-comfort index that
   drives the demand curve.
3. **Temperature smoothing** — temporal lag-weighting helpers that
   account for building thermal inertia.
4. **Error metrics** — R², RMSE, standard deviation of error, and MAPE.

Section index
-------------
1. SCEM Optimiser
2. BAIT Model
3. Temperature Smoothing
4. Error Metrics
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import copy

import numpy as np
import pandas as pd

# CEM base class from the PyPop7 library
from pypop7.optimizers.cem.cem import CEM

from utils.heat_flow_utils import wind_speed_hight


# ===========================================================================
# 1. SCEM Optimiser
# ===========================================================================

class SCEM(CEM):
    """Smoothed Cross-Entropy Method (SCEM) for derivative-free optimisation.

    Extends the base CEM with Laplace-smoothed parameter updates and
    pre-allocated buffers for reduced memory allocation overhead.

    Parameters
    ----------
    problem : dict
        Standard PyPop7 problem definition (``ndim_problem``, bounds, …).
    options : dict
        Optimiser options.  ``'alpha'`` (float, default 0.8) controls the
        exponential smoothing factor for mean and sigma updates.
    """

    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)
        assert 0.0 <= self.alpha <= 1.0

        # Pre-allocate arrays to avoid repeated memory allocation
        self._x_buffer = np.empty((self.n_individuals, self.ndim_problem))
        self._y_buffer = np.empty((self.n_individuals,))
        self._parent_indices = np.empty(self.n_parents, dtype=int)

        # Cache boundary-checking flag
        self._has_bounds = (hasattr(self, 'lower_boundary')
                           and hasattr(self, 'upper_boundary'))

    def initialize(self, is_restart=False):
        """Initialise mean and reuse pre-allocated sample / fitness buffers."""
        mean = self._initialize_mean(is_restart)
        self._x_buffer.fill(0)
        self._y_buffer.fill(np.inf)
        return mean, self._x_buffer, self._y_buffer

    def iterate(self, mean=None, x=None, y=None, args=None):
        """Sample population, clip to bounds, and evaluate fitness."""
        # Vectorised sampling (faster than per-individual loop)
        noise = self.rng_optimization.standard_normal(
            (self.n_individuals, self.ndim_problem))
        x[:] = mean + self._sigmas * noise

        if self._has_bounds:
            np.clip(x, self.lower_boundary, self.upper_boundary, out=x)

        # Batch evaluation path (if supported), else sequential
        if (hasattr(self, '_can_vectorize_fitness')
                and self._can_vectorize_fitness):
            y[:] = self._evaluate_fitness_batch(x, args)
        else:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return x, y
                y[i] = self._evaluate_fitness(x[i], args)

        return x, y

    def _update_parameters(self, mean=None, x=None, y=None):
        """Update mean and sigma using the best *n_parents* samples."""
        # argpartition is O(n) vs O(n log n) for full sort
        self._parent_indices[:] = np.argpartition(
            y, self.n_parents)[:self.n_parents]
        xx = x[self._parent_indices]

        # Smoothed mean update
        new_mean = np.mean(xx, axis=0)
        mean[:] = self.alpha * new_mean + (1.0 - self.alpha) * mean
        if self._has_bounds:
            np.clip(mean, self.lower_boundary, self.upper_boundary, out=mean)

        # Smoothed sigma update
        new_std = np.std(xx, axis=0)
        self._sigmas[:] = self.alpha * new_std + (1.0 - self.alpha) * self._sigmas

        return mean

    def optimize(self, fitness_function=None, args=None):
        """Run the SCEM optimisation loop until termination."""
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()

        while not self._check_terminations():
            x, y = self.iterate(mean, x, y, args)
            self._print_verbose_info(fitness, y)
            if self._check_terminations():
                break
            self._n_generations += 1
            mean = self._update_parameters(mean, x, y)

        return self._collect(fitness, y, mean)

    def _evaluate_fitness_batch(self, population, args=None):
        """Vectorised fitness evaluation fallback (per-individual loop)."""
        if args is None:
            return np.array([self._evaluate_fitness(ind)
                             for ind in population])
        return np.array([self._evaluate_fitness(ind, args)
                         for ind in population])


# ===========================================================================
# 2. BAIT Model
# ===========================================================================

def _bait(tem, wind, solar, humidity, para, valid_date=None):
    """Compute the Building-Adjusted Internal Temperature (BAIT) index.

    The BAIT model transforms raw weather variables into an effective
    *perceived* temperature that accounts for solar gains, wind chill,
    humidity discomfort, and building thermal inertia.

    Reference
    ---------
    Staffell, I., Pfenninger, S., & Johnson, N. (2023).
    *A global model of hourly space heating and cooling demand at
    multiple spatial scales.* Nature Energy, 8(12), 1328–1344.

    Parameters
    ----------
    tem : array_like
        Air temperature [°C].
    wind : array_like
        Wind speed at 10 m [m/s].
    solar : array_like
        Surface solar irradiance [W/m²].
    humidity : array_like
        Specific humidity [kg/kg].
    para : dict
        Calibrated model parameters (``smoothing``, ``solar_gains``,
        ``wind_chill``, ``humidity_discomfort``, ``Tc``, ``Pb``, ``Pc``,
        ``alpha``, ``lower_blend``, ``upper_blend``, ``max_raw_var``).
    valid_date : array_like or None
        If provided, masks lagged values whose dates are not consecutive.

    Returns
    -------
    bait : np.ndarray
        BAIT index with the same shape as *tem*.
    """
    smoothing           = para['smoothing']
    solar_gains         = para['solar_gains']
    wind_chill          = para['wind_chill']
    humidity_discomfort = para['humidity_discomfort']

    # Convert 10 m wind to 2 m height
    wind2m = wind_speed_hight(wind, hight0=10, hight1=2)

    bait = copy.deepcopy(tem)

    # --- Weather adjustments -----------------------------------------------
    # Comfort set-points (functions of air temperature)
    setpoint_S = 100 + 7 * tem                     # W/m²
    setpoint_W = 4.5 - 0.025 * tem                 # m/s
    setpoint_H = np.e ** (1.1 + 0.06 * tem)        # g water / kg air
    setpoint_T = 16                                 # °C (discomfort pivot)

    # Solar gain: sunny → warmer
    bait = bait + (solar - setpoint_S) * solar_gains
    # Wind chill: windy → colder
    bait = bait + (wind2m - setpoint_W) * wind_chill
    # Humidity discomfort: amplifies deviation from comfort
    discomfort = bait - setpoint_T
    bait = (
        setpoint_T
        + discomfort
        + discomfort * (humidity - setpoint_H) * humidity_discomfort
    )

    # --- Temporal smoothing ------------------------------------------------
    # 2nd-day weight is the square of the 1st-day weight (compounded decay)
    bait = smooth_temperature_df(
        pd.DataFrame(bait),
        weights=[smoothing, smoothing ** 2],
        valid_date=valid_date,
    ).values

    # --- Sigmoid blend with raw temperature --------------------------------
    # At low temperatures, raw T is blended back in to prevent over-cooling
    lower_blend = para['lower_blend']
    upper_blend = para['upper_blend']
    max_raw_var = para['max_raw_var']

    avg_blend = (lower_blend + upper_blend) / 2
    dif_blend = upper_blend - lower_blend
    blend = (tem - avg_blend) * 10 / dif_blend
    blend = max_raw_var / (1 + np.exp(-blend))

    bait = (tem * blend) + (bait * (1 - blend))

    return bait


# ===========================================================================
# 3. Temperature Smoothing
# ===========================================================================

def smooth_temperature_df(temperature_df, weights, valid_date=None):
    """Smooth a temperature DataFrame over time using lagged-day weights.

    Each column is treated as an independent location; rows are time steps
    (days).  Lagged values that fall on non-consecutive dates (when
    *valid_date* is supplied) are masked to ``NaN`` and then back-filled.

    Parameters
    ----------
    temperature_df : pd.DataFrame
        Shape ``(n_days, n_locations)``.
    weights : list[float]
        Lag-day weights (index 0 → 1-day lag, index 1 → 2-day lag, …).
    valid_date : array_like or None
        Date stamps matching rows; used to detect non-consecutive gaps.

    Returns
    -------
    pd.DataFrame
        Smoothed temperatures, same shape as input.
    """
    assert isinstance(temperature_df, pd.DataFrame), \
        "Input must be a pandas DataFrame"

    smoothed_df = temperature_df.copy()

    for i, w in enumerate(weights):
        if w == 0:
            continue
        lag_days = i + 1
        lagged = temperature_df.shift(lag_days)

        # Mask lagged values whose source date is not the expected lag
        if valid_date is not None:
            valid_date_array = np.array(valid_date)
            expected_lag_dates = (
                valid_date_array - pd.Timedelta(days=lag_days))
            actual_lag_dates = np.concatenate(
                [np.array([None] * lag_days),
                 valid_date_array[:-lag_days]])
            valid_lag_mask = expected_lag_dates == actual_lag_dates
            valid_lag_mask_2d = np.tile(
                valid_lag_mask[:, None], (1, temperature_df.shape[1]))
            lagged = lagged.where(valid_lag_mask_2d, np.nan)

        # Back-fill leading NaNs to avoid propagation
        lagged.fillna(method='bfill', inplace=True)
        smoothed_df += lagged * w

    # Normalise by total weight
    total_weight = 1 + sum(weights)
    smoothed_df /= total_weight

    return smoothed_df


def smooth_temperature(temperature, weights):
    """Smooth a temperature Series over time with lagged-day weights.

    This is a simplified, single-location variant of
    :func:`smooth_temperature_df`.

    Parameters
    ----------
    temperature : pd.Series
        Daily temperature time-series for one location.
    weights : list[float]
        Lag-day weights (index 0 → 1-day lag, index 1 → 2-day lag, …).

    Returns
    -------
    pd.Series
        Smoothed and re-normalised temperature series.
    """
    assert isinstance(temperature, pd.Series)
    lag = temperature.copy()
    smooth = temperature.copy()

    for w in weights:
        lag = lag.shift(1, fill_value=lag[0])
        if w != 0:
            smooth = (smooth + (lag * w)).reindex()

    smooth = smooth.reindex().dropna()
    return smooth / (1 + sum(weights))


# ===========================================================================
# 4. Error Metrics
# ===========================================================================

def compute_r2(y_data, y_pred):
    """Coefficient of determination (R²).

    Parameters
    ----------
    y_data, y_pred : array_like
        Observed and predicted values.

    Returns
    -------
    float
    """
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_rmse(y_data, y_pred):
    """Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_data, y_pred : array_like

    Returns
    -------
    float
    """
    return np.sqrt(np.mean((y_data - y_pred) ** 2))


def compute_std(y_data, y_pred):
    """Standard deviation of absolute errors.

    Parameters
    ----------
    y_data, y_pred : array_like

    Returns
    -------
    float
    """
    return np.std(np.abs(y_data - y_pred))


def compute_mape(y_data, y_pred):
    """Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_data, y_pred : array_like

    Returns
    -------
    float
    """
    return np.mean(np.abs(y_data - y_pred) / y_data)

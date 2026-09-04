"""
Thermal heat-flow utilities for overhead conductors.

This module implements the steady-state heat balance equation for
bare overhead conductors following IEEE Std 738-2012, including:

- **Convective cooling** (forced and natural)
- **Radiative cooling** (Stefan–Boltzmann)
- **Solar heating**
- **Joule (resistive) heating** with temperature-dependent resistance
- **Heat-balance solvers** (bisection and Newton–Raphson)
- **Quadratic heat-balance approximation** for OPF constraints
- **Maximum allowable current** and **generator derating** models
- **Humidity calculations** (relative and specific) from ERA5 fields

All functions operate element-wise on NumPy arrays, supporting
vectorised computation over multiple branches and conductor segments.
"""

import copy
import math

import numpy as np
from scipy.interpolate import interp1d  # noqa: F401 — used by callers

# =============================================================================
# Physical Constants
# =============================================================================

STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
TK = 273.15                 # Kelvin-to-Celsius offset


# =============================================================================
# Humidity Calculations
# =============================================================================

def calculate_relative_humidity(t2m, d2m):
    """Compute relative humidity from 2 m temperature and dewpoint.

    Uses the Magnus formula for saturation vapour pressure.

    Parameters
    ----------
    t2m : array-like
        2 m air temperature (K).
    d2m : array-like
        2 m dewpoint temperature (K).

    Returns
    -------
    ndarray
        Relative humidity (0–100 %).
    """
    t_c = t2m - TK
    td_c = d2m - TK
    e_s = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e_a = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip((e_a / e_s) * 100.0, 0, 100)


def calculate_specific_humidity(t2m, d2m, pressure=101325.0):
    """Compute specific humidity from dewpoint and surface pressure.

    Parameters
    ----------
    t2m : array-like
        2 m air temperature (K) — unused, kept for API consistency.
    d2m : array-like
        2 m dewpoint temperature (K).
    pressure : float or array-like
        Atmospheric pressure (Pa, default sea level).

    Returns
    -------
    ndarray
        Specific humidity (kg/kg moist air).
    """
    td_c = d2m - TK
    e_a_pa = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5)) * 100.0
    q = (0.622 * e_a_pa) / (pressure - 0.378 * e_a_pa)
    return np.maximum(q, 0)


# =============================================================================
# Dynamic Air Properties (IEEE Std 738, Eqs. 13a–15a)
# =============================================================================

def dynamic_air_viscosity(T_film):
    """Dynamic viscosity of air [kg/(m·s)] at film temperature *T_film* (°C)."""
    return (1.458e-6 * (T_film + TK) ** 1.5) / (T_film + 383.4)


def dynamic_air_density(T_film, elevation, conductor_height=50):
    """Dynamic density of air [kg/m³] at film temperature and elevation."""
    H = elevation + conductor_height
    return ((1.293 - 1.525e-4 * H + 6.379e-9 * H ** 2)
            / (1 + 0.00367 * T_film))


def dynamic_air_conductivity(T_film):
    """Thermal conductivity of air [W/(m·°C)] at film temperature."""
    return 2.424e-2 + 7.477e-5 * T_film - 4.407e-9 * T_film ** 2


# =============================================================================
# Wind Helpers
# =============================================================================

def wind_speed_hight(wind0, hight0=10, hight1=50, Hellmann_Exponent=0.16):
    """Adjust wind speed from *hight0* to *hight1* using the power law."""
    return wind0 * (hight1 / hight0) ** Hellmann_Exponent


def wind_direction_factor(wind_angle=math.pi / 2):
    """Wind direction correction factor *K_phi* (IEEE Std 738, Eq. 4a)."""
    return (1.194
            - np.cos(wind_angle)
            + 0.194 * np.cos(2 * wind_angle)
            + 0.368 * np.sin(2 * wind_angle))


def Reynolds_number(diameter, wind_speed, air_density, air_viscosity):
    """Reynolds number for flow around a cylinder."""
    return diameter * wind_speed * air_density / air_viscosity


# =============================================================================
# Heat-Balance Components
# =============================================================================

def convective_cooling(tem_con, conductor, weather):
    """Convective heat loss per unit length [W/m].

    Computes both forced-convection correlations (low-Re and high-Re)
    and the natural-convection estimate, returning the maximum.

    Parameters
    ----------
    tem_con : array-like
        Conductor surface temperature (°C).
    conductor : dict
        Conductor specification.
    weather : dict
        Weather conditions.

    Returns
    -------
    ndarray
        Convective cooling rate (W/m).
    """
    tem_air = weather['air_temperature']
    wind_angle = weather['wind_angle']
    wind_speed = weather['wind_speed']
    wind_height = weather['wind_height']
    diameter = conductor['diameter']
    conductor_angle = conductor['conductor_angle']
    elevation = conductor['elevation']

    T_film = (tem_con + tem_air) / 2
    k_air = dynamic_air_conductivity(T_film)
    rho_air = dynamic_air_density(T_film, elevation)
    mu_air = dynamic_air_viscosity(T_film)

    # Wind-to-conductor angle correction
    if isinstance(wind_angle, float):
        wind_angle = np.array([wind_angle])
    angle_diff = np.abs(wind_angle % 180 - conductor_angle % 180)
    angle_diff[angle_diff > 90] = 180 - angle_diff[angle_diff > 90]
    K_phi = wind_direction_factor(np.deg2rad(angle_diff))

    # Adjust wind speed to conductor height
    wind_speed = wind_speed_hight(wind_speed, hight0=wind_height)

    N_re = Reynolds_number(diameter, wind_speed, rho_air, mu_air)

    # Forced convection (low-Re and high-Re correlations)
    hc_low = K_phi * k_air * (1.01 + 1.35 * N_re ** 0.52) * (tem_con - tem_air)
    hc_high = 0.754 * K_phi * k_air * N_re ** 0.6 * (tem_con - tem_air)

    # Natural convection
    hc_nat = 3.645 * rho_air ** 0.5 * diameter ** 0.75 * (tem_con - tem_air) ** 1.25

    return np.maximum(np.maximum(hc_low, hc_high), hc_nat)


def radiative_cooling(tem_con, conductor, weather):
    """Radiative heat loss per unit length [W/m] (Stefan–Boltzmann)."""
    tem_air = weather['air_temperature']
    emissivity = weather['radiation_emissivity']
    diameter = conductor['diameter']
    return (np.pi * STEFAN_BOLTZMANN * diameter * emissivity
            * ((tem_con + TK) ** 4 - (tem_air + TK) ** 4))


def solar_heating(conductor, weather):
    """Solar heat gain per unit length [W/m]."""
    absorptivity = weather['solar_absorptivity']
    solar_flux = weather['solar_heat_intensity']
    diameter = conductor['diameter']
    return absorptivity * diameter * solar_flux


def joule_heating(tem_con, current, conductor):
    """Resistive (I²R) heat gain per unit length [W/m]."""
    R_T = td_resistance(
        conductor['unit_resistance'], tem_con,
        conductor['ref_temperature'], conductor['resistance_ratio'],
    )
    return current ** 2 * R_T


def td_resistance(resistance, tem_con, tem_ref, resistance_ratio):
    """Temperature-dependent resistance: R(T) = R_ref × (1 + α·(T − T_ref))."""
    return resistance * (1 + (tem_con - tem_ref) * resistance_ratio)


# =============================================================================
# Heat-Balance Equation & Solvers
# =============================================================================

def heat_surplus(tem_con, current, conductor, weather):
    """Net heat surplus [W/m]: positive means conductor is still heating."""
    conv_corr = conductor.get('convective_correction', 1)
    rad_corr = conductor.get('radiactive_correction', 1)

    Hj = joule_heating(tem_con, current, conductor)
    Hs = solar_heating(conductor, weather)
    Hc = convective_cooling(tem_con, conductor, weather) * conv_corr
    Hr = radiative_cooling(tem_con, conductor, weather) * rad_corr
    return Hj + Hs - (Hc + Hr)


def df_heat_surplus(tem_con, current, conductor, weather):
    """Numerical derivative of heat surplus w.r.t. conductor temperature."""
    delta = 1e-6
    return ((heat_surplus(tem_con + delta, current, conductor, weather)
             - heat_surplus(tem_con, current, conductor, weather)) / delta)


def heat_banlance_equation(current, conductor, weather,
                           alg='bisection', tol=1e-5):
    """Solve the steady-state heat-balance equation for conductor temperature.

    Parameters
    ----------
    current : array-like
        Conductor current (A).
    conductor : dict
        Conductor specification.
    weather : dict
        Weather conditions.
    alg : str
        ``'bisection'`` (robust) or ``'newton'`` (fast but may diverge).
    tol : float
        Convergence tolerance on conductor temperature (°C).

    Returns
    -------
    ndarray
        Equilibrium conductor temperature (°C).
    """
    tem_air = weather['air_temperature']

    if alg == 'bisection':
        tem_upper = 200
        delta_H = heat_surplus(tem_air, current, conductor, weather)
        tem_lo = tem_air * np.ones(delta_H.shape)
        tem_hi = tem_upper * np.ones(delta_H.shape)
        tem_prev = 0
        for _ in range(1000):
            tem_mid = (tem_lo + tem_hi) / 2
            delta_H = heat_surplus(tem_mid, current, conductor, weather)
            tem_lo[delta_H >= 0] = tem_mid[delta_H >= 0]
            tem_hi[delta_H < 0] = tem_mid[delta_H < 0]
            if np.abs(tem_prev - tem_mid).max() <= tol:
                break
            tem_prev = tem_mid
        return tem_mid

    elif alg == 'newton':
        max_tem = conductor['max_temperature']
        tem_con = (copy.deepcopy(tem_air) + max_tem) / 2
        for _ in range(1000):
            delta = (heat_surplus(tem_con, current, conductor, weather)
                     / df_heat_surplus(tem_con, current, conductor, weather))
            tem_con = tem_con - delta
            if np.abs(delta).max() <= tol:
                break
        return tem_con

    else:
        raise ValueError(f"Unknown algorithm: '{alg}'")


# =============================================================================
# Quadratic Approximation of Heat Balance
# =============================================================================

def coefficient_quadratic_approximation(conductor, weather):
    """Compute quadratic-approximation coefficients (β₀, β₁, β₂).

    The approximation is: T_c ≈ β₀ + β₁·I² + β₂·I⁴

    Parameters
    ----------
    conductor : dict
        Conductor specification.
    weather : dict
        Weather conditions.

    Returns
    -------
    tuple
        ``(beta0, beta1, beta2)`` coefficient arrays.
    """
    tem_air = weather['air_temperature']
    tem_max = conductor['max_temperature']
    tem_ref = conductor['ref_temperature']
    diameter = conductor['diameter']
    unit_resistance = conductor['unit_resistance']
    resistance_ratio = conductor['resistance_ratio']
    emissivity = weather['radiation_emissivity']
    absorptivity = weather['solar_absorptivity']
    solar_flux = weather['solar_heat_intensity']
    wind_angle = weather['wind_angle']
    wind_speed = weather['wind_speed']
    wind_height = weather['wind_height']
    air_density = weather['air_density']
    air_conductivity = weather['air_conductivity']
    conductor_angle = conductor['conductor_angle']

    R_ta = td_resistance(unit_resistance, tem_air, tem_ref, resistance_ratio)
    R_tmax = td_resistance(unit_resistance, tem_max, tem_ref, resistance_ratio)

    # Wind direction correction
    if isinstance(wind_angle, float):
        wind_angle = np.array([wind_angle])
    angle_diff = np.abs(wind_angle % 180 - conductor_angle % 180)
    angle_diff[angle_diff > 90] = 180 - angle_diff[angle_diff > 90]
    K_phi = wind_direction_factor(np.deg2rad(angle_diff))
    wind_speed = wind_speed_hight(wind_speed, hight0=wind_height)

    # Linearised convective / radiative coefficients
    Hc = np.maximum(
        3.07 / diameter * K_phi * (air_density * diameter * wind_speed) ** 0.471,
        8.35 / diameter * K_phi * (air_density * diameter * wind_speed) ** 0.8,
    )
    Hr0 = 4 * STEFAN_BOLTZMANN * emissivity * (tem_air + TK) ** 3
    k1 = 6 * STEFAN_BOLTZMANN * emissivity * (tem_air + TK) ** 2
    k2 = R_tmax / (np.pi * diameter * (Hc + Hr0 + k1 * (tem_max - tem_air)))

    b0 = np.pi * diameter * (Hc + Hr0)
    b1 = unit_resistance * resistance_ratio - np.pi * diameter * k1 * k2

    beta0 = tem_air + absorptivity * diameter * solar_flux / b0
    beta1 = R_ta / b0
    beta2 = k2 * b1 / b0

    return beta0, beta1, beta2


def quadratic_heat_balance_approximation(current, conductor, weather):
    """Approximate conductor temperature: T_c ≈ β₀ + β₁·I² + β₂·I⁴."""
    beta0, beta1, beta2 = coefficient_quadratic_approximation(conductor, weather)
    return beta0 + beta1 * current ** 2 + beta2 * current ** 4


def quadratic_maximum_current(conductor, weather):
    """Maximum current from the quadratic approximation (closed-form inverse)."""
    beta0, beta1, beta2 = coefficient_quadratic_approximation(conductor, weather)
    T_max = conductor['max_temperature']
    numerator = -beta1 + np.sqrt(beta1 ** 2 - 4 * beta2 * (beta0 - T_max))
    denominator = 2 * beta2
    return np.sqrt(numerator / denominator)


# =============================================================================
# Maximum Allowable Current & Generator Derating
# =============================================================================

def maximum_allowable_current(conductor, weather):
    """Compute the steady-state thermal rating (ampacity) per conductor [A].

    Solves I_max = √((H_c + H_r − H_s) / R(T_max)) at the maximum
    permissible conductor temperature.
    """
    tem_max = conductor['max_temperature']
    tem_ref = conductor['ref_temperature']
    unit_resistance = conductor['unit_resistance']
    resistance_ratio = conductor['resistance_ratio']
    R_Tmax = td_resistance(unit_resistance, tem_max, tem_ref, resistance_ratio)

    conv_corr = conductor.get('convective_correction', 1)
    rad_corr = conductor.get('radiactive_correction', 1)

    Hs = solar_heating(conductor, weather)
    Hc = convective_cooling(tem_max, conductor, weather) * conv_corr
    Hr = radiative_cooling(tem_max, conductor, weather) * rad_corr

    return ((Hr + Hc - Hs) / R_Tmax) ** 0.5


def generator_derating(gen_type='default', weather=None):
    """Compute generator output derating factor (0–1) due to high ambient temperature.

    Parameters
    ----------
    gen_type : str
        Generator type: ``'OCGT'``, ``'CCGT'``, ``'nuclear'``, or
        ``'default'`` (copper-winding thermal limit).
    weather : dict
        Must contain ``'air_temperature'`` (°C).

    Returns
    -------
    float or ndarray
        Derating factor (capped at 1.0).
    """
    tem_air = weather['air_temperature']

    if gen_type == 'OCGT':
        derating = (-0.6854 * tem_air + 110) / 100
    elif gen_type == 'CCGT':
        derating = (-0.6854 * tem_air / 2 + 105) / 100
    elif gen_type == 'nuclear':
        derating = (101.3042 - 0.1387 * tem_air - 0.0010 * tem_air ** 2) / 100
    elif gen_type == 'default':
        # Copper-winding thermal limit model
        derating = (((180 - tem_air) * (1 + 0.0039 * (40 - 20)))
                    / ((180 - 40) * (1 + 0.0039 * (tem_air - 20)))) ** 0.5
    else:
        derating = 1

    return min(derating, 1)


# =============================================================================
# Temperature-Dependent Resistance Update
# =============================================================================

def calculate_temperature_dependent_resistance(base_resistance,
                                               base_seg_resistance,
                                               air_temp, conductor,
                                               tdpf_analysis):
    """Update branch resistance for the current air temperature.

    Parameters
    ----------
    base_resistance : ndarray
        Per-branch base resistance at reference temperature.
    base_seg_resistance : ndarray
        Per-segment base resistance (used when ``'seg'`` in *tdpf_analysis*).
    air_temp : float or ndarray
        Ambient air temperature (°C).
    conductor : dict
        Conductor specification (needs ``ref_temperature``, ``resistance_ratio``).
    tdpf_analysis : str
        Analysis type — if it contains ``'seg'``, segment-level
        resistance is summed; otherwise branch-level is returned.

    Returns
    -------
    ndarray
        Updated branch resistance values.
    """
    ref_temp = conductor['ref_temperature']
    alpha = conductor['resistance_ratio']

    if 'seg' in tdpf_analysis:
        td_seg_R = td_resistance(base_seg_resistance, air_temp, ref_temp, alpha)
        return np.sum(td_seg_R, axis=-1)
    else:
        td_branch_R = td_resistance(base_resistance, air_temp, ref_temp, alpha)
        return np.squeeze(td_branch_R, axis=1)


# =============================================================================
# Post-Processing & Network-Level Utilities
# =============================================================================

def post_process_results(result, network, weather, conductor, BaseI):
    """Enrich OPF result dict with conductor temperatures and capacity info.

    Parameters
    ----------
    result : dict
        OPF solution (must contain ``'I_pu'``).
    network : pypsa.Network
        Network with segment weather and line parameters.
    weather : dict
        Weather template (deep-copied internally).
    conductor : dict
        Conductor specification (deep-copied internally).
    BaseI : float
        Base current (kA) for per-unit conversion.

    Returns
    -------
    dict
        Updated *result* with ``'con_temp'``, ``'inom'``, ``'Imax'``,
        ``'capacity_drop'``, and ``'BaseI'`` fields added.
    """
    num_parallel = network.lines['num_parallel'].values.copy()
    num_parallel[num_parallel == 0] = 1
    current = (result['I_pu'] * BaseI * 1000
               / (num_parallel * conductor['num_bundle']))

    # Build per-segment weather and conductor dicts
    weather_s = copy.deepcopy(weather)
    conductor_s = copy.deepcopy(conductor)
    seg_wea = network.segment_weather
    weather_s['air_temperature'] = seg_wea[:, :, 0]
    weather_s['wind_speed'] = seg_wea[:, :, 1]
    weather_s['solar_heat_intensity'] = seg_wea[:, :, 2]
    weather_s['wind_angle'] = seg_wea[:, :, 3]
    conductor_s['elevation'] = seg_wea[:, :, 4]

    # Conductor bearing angle
    xe = (network.buses.loc[network.lines['bus1'], ['x', 'y']].values
          - network.buses.loc[network.lines['bus0'], ['x', 'y']].values)
    conductor_s['conductor_angle'] = np.reshape(
        (90 - np.rad2deg(np.arctan2(xe[:, 1], xe[:, 0]))) % 360, (-1, 1)
    )

    # Compute per-segment conductor temperature
    seg_mask = np.sign(network.segments[:, :, 2])
    con_temp = heat_banlance_equation(
        np.expand_dims(current, 1), conductor_s, weather_s,
    ) * seg_mask
    result['con_temp'] = con_temp

    result['inom'] = network.lines.i_nom.values * 1000
    result['Imax'] = network.lines['Imax']
    result['capacity_drop'] = network.lines['capacity_drop']
    result['BaseI'] = BaseI
    return result


def cal_current_constraint(network, conductor, weather, TDPF_analysis):
    """Compute and store per-line current limits on the network.

    Sets ``network.lines['Imax']`` and ``network.lines['capacity_drop']``
    in-place based on the analysis type.

    Parameters
    ----------
    network : pypsa.Network
        Network whose ``lines`` DataFrame is updated in-place.
    conductor : dict
        Conductor specification.
    weather : dict
        Weather conditions.
    TDPF_analysis : str
        Analysis type selector.
    """
    i_nom = network.lines.i_nom.values * 1000

    if 'base' in TDPF_analysis:
        Imax = i_nom
        est_capacity_drop = 1

    if 'fixsc' in TDPF_analysis:
        Imax = i_nom
        est_capacity_drop = 0.7

    if 'td' in TDPF_analysis:
        if 'qua' in TDPF_analysis:
            Imax = quadratic_maximum_current(conductor, weather)
        else:
            Imax = maximum_allowable_current(conductor, weather)
        Imax = np.min(Imax, axis=1) * conductor['num_bundle']
        Imax = np.minimum(Imax, i_nom)
        est_capacity_drop = Imax / i_nom

    network.lines['Imax'] = Imax
    network.lines['capacity_drop'] = est_capacity_drop

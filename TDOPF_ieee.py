"""
Temperature-Dependent Optimal Power Flow (TD-OPF) for IEEE Test Systems.

This module implements thermal-dependent AC Optimal Power Flow analysis
for IEEE bus systems (e.g., IEEE 30-bus). It supports multiple analysis
modes including:
    - Base AC-OPF (no thermal effects)
    - Quadratic thermal constraints (Quad-OPF)
    - Iterative temperature-dependent OPF (Iter-OPF / TD-OPF)
    - Generator derating under high ambient temperatures
    - Security-constrained variants (AC-SC, DC-SC, Fix-SC)

The thermal model follows IEEE Std 738-2012 for overhead conductor
heat balance and dynamic line rating calculations.

Usage:
    python TDOPF_ieee.py
"""

import os
import re
import copy
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt

from pypower import idx_bus, idx_gen, idx_brch
from pypower.api import ext2int

from utils.opf_pyomo_utils import *
from utils.heat_flow_utils import *
from utils.network_process_utils import *


# =============================================================================
# Helper Functions — System Initialization
# =============================================================================

def initialize_base_values(ppc):
    """Extract and calculate per-unit base values from power system data.

    Args:
        ppc (dict): PYPOWER case dict with 'baseMVA' and 'bus' fields.

    Returns:
        tuple: (baseMva, baseKV, baseI) — base MVA, base kV, and base current.
    """
    baseMva = ppc['baseMVA']
    baseKV = ppc['bus'][0, idx_bus.BASE_KV]
    baseI = baseMva / baseKV

    ppc['baseKV'] = baseKV
    ppc['baseI'] = baseI

    return baseMva, baseKV, baseI


def setup_weather_conditions(conductor, weather, tdpf_analysis):
    """Configure weather conditions based on the analysis type.

    For 'base' analyses the ambient temperature is reset to the conductor
    reference temperature so that thermal effects are neutralised.

    Args:
        conductor (dict): Conductor specification containing 'ref_temperature'.
        weather (dict): Weather conditions dict (modified in-place for 'base').
        tdpf_analysis (str): Analysis type identifier string.

    Returns:
        tuple: (weather, air_temp) — updated weather dict and the effective
               ambient air temperature used for this run.
    """
    ref_temp = copy.deepcopy(conductor['ref_temperature'])
    air_temp = copy.deepcopy(weather['air_temperature'])

    if 'base' in tdpf_analysis:
        air_temp = ref_temp

    return weather, air_temp


def apply_thermal_effects(ppc, conductor, weather, tdpf_analysis):
    """Apply thermal derating effects to branch ratings and generators.

    Calculates the maximum allowable current for each branch based on
    conductor specs and weather, then optionally applies generator derating.

    Args:
        ppc (dict): PYPOWER case dict (modified in-place).
        conductor (dict): Conductor specification.
        weather (dict): Weather conditions.
        tdpf_analysis (str): Analysis type — if it contains 'derate',
                             generator PMAX values are scaled down.

    Returns:
        dict: The updated ppc dict.
    """
    gen_type = 'default'

    # Thermal line rating (Imax stored in RATE_B in kA)
    Imax = maximum_allowable_current(conductor, weather)
    ppc['branch'][:, idx_brch.RATE_B] = Imax / 1000

    # Generator derating under elevated ambient temperature
    if 'derate' in tdpf_analysis:
        Derating = np.minimum(generator_derating(gen_type, weather), 1)
        ppc['gen'][:, idx_gen.PMAX] *= Derating

    return ppc


# =============================================================================
# Core Solvers
# =============================================================================

def run_base_acopf(ppc_base, fixsc=False, safe_margin=1.0):
    """Run a standard (non-thermal) AC Optimal Power Flow.

    Args:
        ppc_base (dict): PYPOWER case dict for the base case.
        fixsc (bool): Whether to use fixed security constraints.
        safe_margin (float): Thermal safe margin factor.

    Returns:
        tuple: (result, base_runtime, base_cost)
    """
    st = time.time()

    instance = ACOPF(
        ppc_base,
        voltage_form='polar',
        qlim=True,
        angle_cons=True,
        reactive_demand_ratio='auto',
        safe_margin=safe_margin,
        fix_sc=fixsc,
        tol=1e-3,
    )
    result = instance.solve()

    et = time.time()
    base_runtime = et - st
    result['run_time'] = base_runtime
    base_cost = result['obj']
    result['speedup'] = 1.0

    return result, base_runtime, base_cost


def run_td_acopf_eur(ppc, base_result, contingency, tdpf_analysis,
                     base_resistance, base_seg_resistance,
                     conductor, weather,
                     num_iter=10, tol=1e-3):
    """Run temperature-dependent ACOPF with iterative resistance updates.

    At each iteration the conductor temperature is recalculated from
    the branch currents, the branch resistance is updated accordingly,
    and the OPF is re-solved until the solution converges.

    Args:
        ppc (dict): PYPOWER case dict (modified in-place).
        base_result (dict | None): Warm-start result from a prior solve.
        contingency (list): Contingency branch indices.
        tdpf_analysis (str): Analysis type identifier.
        base_resistance (ndarray): Reference resistance values at ref temp.
        base_seg_resistance: Unused (reserved for segmented conductors).
        conductor (dict): Conductor specification.
        weather (dict): Weather conditions.
        num_iter (int): Maximum number of OPF–thermal iterations.
        tol (float): Convergence tolerance (checked on PG, temp, and I).

    Returns:
        tuple: (ppc, result) — updated case dict and the final OPF result.
    """
    num_bundle = 1
    ref_temp = conductor['ref_temperature']
    resistance_ratio = conductor['resistance_ratio']
    baseMva = ppc['baseMVA']
    baseKV = ppc['bus'][0, idx_bus.BASE_KV]
    baseI = baseMva / baseKV

    prev_pg = 0
    prev_temp = 0
    prev_I = 0

    # --- Warm-start: update resistance from prior solution ----------------
    if base_result is not None:
        I_pu = base_result['I_pu']
        I = I_pu * baseI * 1000
        per_I = I / num_bundle
        con_temp = heat_banlance_equation(
            np.expand_dims(per_I, 1), conductor, weather
        )
        base_resistance = np.reshape(base_resistance, (-1, 1))
        td_branch_resistance = td_resistance(
            base_resistance, con_temp, ref_temp, resistance_ratio
        )
        ppc['branch'][:, idx_brch.BR_R] = np.squeeze(td_branch_resistance, 1)

    # --- Parse analysis flags ---------------------------------------------
    fixsc = 'fixsc' in tdpf_analysis
    safe_margin = 0.7 if fixsc else 1
    dcsc = 'dcsc' in tdpf_analysis
    acsc = 'acsc' in tdpf_analysis
    tdacsc = 'tdacsc' in tdpf_analysis
    quad_con = 'quad' in tdpf_analysis
    tem_cons = 'td' in tdpf_analysis

    # --- Iterative TD-ACOPF loop -----------------------------------------
    for iter_num in range(num_iter):
        instance = ACOPF(
            ppc,
            initial_value=base_result,
            voltage_form='polar',
            contingency=contingency,
            dc_sc=dcsc,
            ac_sc=acsc,
            fix_sc=fixsc,
            tdac_sc=tdacsc,
            safe_margin=safe_margin,
            quad_con=quad_con,
            td_cons=tem_cons,
            conductor=conductor,
            weather=weather,
            angle_cons=True,
            qlim=True,
            reactive_demand_ratio='auto',
        )
        result = instance.solve()

        # Recompute conductor temperature from new currents
        I_pu = result['I_pu']
        I = I_pu * baseI * 1000
        per_I = I / num_bundle
        con_temp = heat_banlance_equation(
            np.expand_dims(per_I, 1), conductor, weather
        )
        td_branch_resistance = td_resistance(
            base_resistance, con_temp, ref_temp, resistance_ratio
        )
        ppc['branch'][:, [idx_brch.BR_R]] = np.reshape(
            td_branch_resistance, (-1, 1)
        )

        # Check convergence
        if (iter_num > 0
                and np.max(np.abs(result['PG'] - prev_pg)) < tol
                and np.max(np.abs(con_temp - prev_temp)) < tol
                and np.max(np.abs(I - prev_I)) < tol):
            break

        # Prepare next iteration
        base_result = result
        prev_pg = base_result['PG']
        prev_temp = con_temp
        prev_I = I

    return ppc, result


# =============================================================================
# Result Analysis
# =============================================================================

def ACOPF_analysis_ieee(ppc_expr, conductor, result, result_row,
                        base_cost, tdpf_analysis):
    """Compute and store summary metrics from an ACOPF solution.

    Populates *result_row* (dict) with key performance indicators such as
    total load, load-shedding ratio, flow / temperature congestion counts,
    and prints a one-line summary to stdout.

    Args:
        ppc_expr (dict): PYPOWER case dict used in the experiment.
        conductor (dict): Conductor specification (for max temp reference).
        result (dict): OPF solution dictionary.
        result_row (dict): Mutable dict to receive metric values.
        base_cost (float): Objective from the base-case OPF (unused
                           in current metrics but kept for extensibility).
        tdpf_analysis (str): Analysis type label (printed in summary).

    Returns:
        dict: The updated *result_row*.
    """
    baseMva = ppc_expr['baseMVA']

    # --- Extract solution vectors -----------------------------------------
    Pg = result['PG']
    Pd = result['PD']
    Pg_capacity = result['PG_capacity']
    ST_capacity = result['ST_capacity']
    cost = result['obj']
    S_pu = result['S_pu']
    I_pu = result['I_pu']
    slack_p_eq = result['LS']
    branch_temp = result['con_temp']

    # --- Branch and generator status masks --------------------------------
    Branch_status = np.sign(
        ppc_expr['branch'][:, idx_brch.BR_STATUS]
        * ppc_expr['branch'][:, idx_brch.RATE_A]
    )
    Smax = ppc_expr['branch'][:, idx_brch.RATE_A] / baseMva

    # --- Utilisation ratios -----------------------------------------------
    Sratio = S_pu / (Smax + 1e-8) * Branch_status
    Tratio = branch_temp / conductor['max_temperature']

    # --- Populate result row ----------------------------------------------
    result_row['solver_status'] = result['solver_status']
    result_row['load'] = np.sum(Pd) * baseMva * 3 / 1000  # GW (three-phase)
    result_row['node_load_shedding'] = np.sum(slack_p_eq) / np.sum(Pd) * 100
    result_row['gen_cost'] = cost
    result_row['tem_con_mean'] = np.sum(branch_temp) / np.sum(Branch_status)
    result_row['tem_con_max'] = np.max(branch_temp)
    result_row['run_time'] = result['runtime']

    # Load-shedding node counts at various thresholds
    for threshold in [0.001, 0.01, 0.1]:
        label = str(threshold).replace('0.', '').rstrip('0') or '0'
        key = f'node_mis_match_{label}_num'
        result_row[key] = np.sum((slack_p_eq >= threshold).astype(int))

    # Flow and temperature congestion counts at safety thresholds
    for safe_thresh in [0.7, 0.8, 0.9]:
        result_row[f'{safe_thresh}_flow_num'] = int(
            np.sum((Sratio >= safe_thresh).astype(int))
        )
        result_row[f'{safe_thresh}_temp_num'] = int(
            np.sum((Tratio >= safe_thresh).astype(int))
        )

    # --- Console summary --------------------------------------------------
    print(
        f"RESULTS : success {result['solver_status']} "
        f"| {tdpf_analysis} "
        f"| Demand: {Pd.sum():.2f} "
        f"| Gen: {Pg_capacity.sum():.2f} "
        f"| Storage: {ST_capacity.sum():.2f} "
        f"| LS ratio: {result_row['node_load_shedding']:.2f} "
        f"| LS num: {result_row['node_mis_match_01_num']:.0f} "
        f"| Max temp: {result_row['tem_con_max']:.2f} "
        f"| Runtime: {result_row['run_time']:.2f}s"
    )
    return result_row


# =============================================================================
# Main Experiment Entry Point
# =============================================================================

def TDACOPF(exper_name, df, ppc, conductor, weather, load_ratio,
            tdpf_analysis):
    """Run a single thermal-dependent ACOPF experiment and record results.

    This is the top-level driver that:
      1. Sets up base values, weather, and thermal effects.
      2. Solves the base-case OPF.
      3. Dispatches to the appropriate TD-OPF variant.
      4. Computes conductor temperatures and records metrics.
      5. Saves results to disk (`.npy`) and appends a row to *df*.

    Args:
        exper_name (str): Experiment identifier (used for output paths).
        df (DataFrame): Results table to append the new row to.
        ppc (dict): PYPOWER case dict (will be modified).
        conductor (dict): Conductor specification.
        weather (dict): Weather conditions.
        load_ratio (float): Scaling factor applied to bus loads.
        tdpf_analysis (str): Analysis type selector. Supported values:
            - ``'base'`` — plain AC-OPF
            - ``'base_*'`` — base AC-OPF with security constraints
            - ``'td_*'`` — iterative temperature-dependent OPF

    Returns:
        DataFrame: Updated *df* with the new experiment row appended.
    """
    # --- Output directory -------------------------------------------------
    record_path = f'models/IEEE30/record/{exper_name}/'
    os.makedirs(record_path, exist_ok=True)

    # --- System base values -----------------------------------------------
    baseMva, baseKV, baseI = initialize_base_values(ppc)
    base_resistance = np.reshape(ppc['branch'][:, idx_brch.BR_R], (-1, 1))

    # --- Weather configuration --------------------------------------------
    weather, air_temp = setup_weather_conditions(conductor, weather,
                                                 tdpf_analysis)

    # --- Security-constraint flags ----------------------------------------
    contingency = identify_contingency(ppc) if 'sc' in tdpf_analysis else []
    fixsc = 'fixsc' in tdpf_analysis
    safe_margin = 0.7 if fixsc else 1
    dcsc = 'dcsc' in tdpf_analysis
    acsc = 'acsc' in tdpf_analysis
    tdacsc = 'tdacsc' in tdpf_analysis

    # --- Thermal effects & load scaling -----------------------------------
    ppc = apply_thermal_effects(ppc, conductor, weather, tdpf_analysis)
    ppc['bus'][:, idx_bus.PD] *= load_ratio
    ppc['bus'][:, idx_bus.QD] *= load_ratio

    ppc_base = copy.deepcopy(ppc)
    ppc_expr = copy.deepcopy(ppc)

    # --- Stage 1: Base-case ACOPF -----------------------------------------
    base_result, base_runtime, base_cost = run_base_acopf(ppc_base)

    # --- Stage 2: Thermal-dependent ACOPF dispatch ------------------------
    st = time.time()

    if tdpf_analysis == 'base':
        # Plain AC-OPF — reuse the base result directly
        result = base_result

    elif 'base' in tdpf_analysis:
        # Base analysis with optional security constraints
        instance = ACOPF(
            ppc_base,
            initial_value=base_result,
            voltage_form='polar',
            contingency=contingency,
            safe_margin=safe_margin,
            fix_sc=fixsc,
            dc_sc=dcsc,
            ac_sc=acsc,
            tdac_sc=tdacsc,
            qlim=True,
            angle_cons=True,
            reactive_demand_ratio='auto',
        )
        result = instance.solve()

    elif 'td' in tdpf_analysis:
        # Iterative temperature-dependent OPF
        num_iter = 1
        if 'iter' in tdpf_analysis:
            num_iter = int(re.findall(r'\d+', tdpf_analysis)[0])
        ppc_expr, result = run_td_acopf_eur(
            ppc_expr, base_result, contingency, tdpf_analysis,
            base_resistance, None, conductor, weather,
            num_iter, tol=1e-5,
        )

    else:
        raise NotImplementedError(
            f"Analysis type '{tdpf_analysis}' is not supported."
        )

    et = time.time()

    # --- Post-processing: conductor temperature & timing ------------------
    current = result['I_pu'] * baseI * 1000
    result['con_temp'] = heat_banlance_equation(current, conductor, weather)
    result['base_runtime'] = base_runtime
    result['runtime'] = base_runtime if tdpf_analysis == 'base' else (et - st)
    result['speedup'] = (et - st) / base_runtime
    result['Imax'] = ppc['branch'][:, idx_brch.RATE_B]

    # --- Save result to disk ----------------------------------------------
    wind_speed = weather['wind_speed']
    air_temp = weather['air_temperature']
    max_conductor_temp = conductor['max_temperature']

    filename = (
        f'load_{load_ratio}_temp_{air_temp}_wind_{wind_speed}'
        f'_{tdpf_analysis}_maxtemp_{max_conductor_temp}.npy'
    )
    np.save(os.path.join(record_path, filename), result)
    np.save(os.path.join(record_path, 'ppc_base.npy'), ppc_base)

    # --- Build result row and append to DataFrame -------------------------
    result_row = {
        'TDPF_solver': tdpf_analysis,
        'load_ratio': load_ratio,
        'air_temp': air_temp,
        'wind_speed': wind_speed,
        'max_conductor_temp': max_conductor_temp,
        'exper_id': exper_name,
    }
    result_row = ACOPF_analysis_ieee(
        ppc_expr, conductor, result, result_row, base_cost, tdpf_analysis
    )
    df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)

    return df


# =============================================================================
# Script Entry Point
# =============================================================================

def main():
    """Run the full TD-OPF experiment sweep on the IEEE 30-bus system.

    Default conductor and weather parameters follow IEEE Std 738-2012
    (795 kcmil 26/7 Drake ACSR conductor).
    """

    # --- Conductor parameters (Drake ACSR 795 kcmil 26/7) ----------------
    conductor = {
        'diameter': 28.1e-3,           # m
        'ref_temperature': 25,         # °C
        'max_temperature': 90,         # °C
        'resistance_ratio': 0.00429,   # Ω/°C
        'unit_resistance': 7.283e-5,   # Ω/m
        'conductor_angle': 0,
        'elevation': 100,
        'num_bundle': 1,
    }

    # --- Weather parameters (IEEE standard defaults) ----------------------
    weather = {
        'wind_speed': 0.61,            # m/s
        'wind_angle': np.array([90]),   # degree
        'wind_height': 10,
        'air_density': 1.029,          # kg/m³
        'air_viscosity': 2.043e-5,     # kg/(m·s)
        'air_conductivity': 0.02945,   # W/(m·°C)
        'air_temperature': 25,         # °C
        'radiation_emissivity': 0.8,
        'solar_absorptivity': 0.8,
        'solar_heat_intensity': 800,   # W/m²
    }

    # --- Load IEEE 30-bus system ------------------------------------------
    nbus = 30
    data = loadmat(f'data/ieee_data/casefiles_mat/case_{nbus}.mat')
    ppc_mat = data.get('mpc')

    ppc = {
        'version': int(ppc_mat['version'][0, 0]),
        'baseMVA': float(ppc_mat['baseMVA'][0, 0]),
        'bus': ppc_mat['bus'][0, 0],
        'gen': ppc_mat['gen'][0, 0],
        'branch': ppc_mat['branch'][0, 0],
        'gencost': ppc_mat['gencost'][0, 0],
    }
    ppc = ext2int(ppc)

    # --- Experiment scenarios ---------------------------------------------
    load_scenarios = [0.8, 0.9, 1.0]  # light / default / heavy load

    # (wind_speed m/s, air_temperature °C)
    weather_scenarios = [
        (0.61, 25),  # mild weather
        (0.10, 25),  # mild + low wind
        (0.61, 45),  # mild + high temperature
        (0.10, 45),  # extreme weather
    ]

    # Primary analysis methods
    analysis_list = [
        'base',                # AC-OPF
        'td_quad',             # Quad-OPF
        'td_derate_iter_2',    # Iter-OPF  (2 iterations)
        'td_derate_iter_10',   # TD-OPF    (10 iterations)
    ]

    # Sensitivity / ablation studies
    sa_list = [
        'base_derate',   # w/o thermal constraints
        'td_iter_2',     # w/o generator derating
        'base_acsc',     # AC security-constrained
        'base_dcsc',     # DC security-constrained
        'base_fixsc',    # Fixed security-constrained
    ]

    # --- Run experiment sweep ---------------------------------------------
    exper_name = 'heatflow_analysis'
    df_path = f'models/IEEE{nbus}/{nbus}_{exper_name}.csv'
    df = pd.DataFrame()

    for load in load_scenarios:
        for wind, air_temp in weather_scenarios:
            weather['wind_speed'] = wind
            weather['air_temperature'] = air_temp

            for tdpf_analysis in analysis_list + sa_list:
                print(
                    f"load: {load}, wind: {wind}, "
                    f"air_temp: {air_temp}, tdpf_analysis: {tdpf_analysis}"
                )
                df = TDACOPF(
                    exper_name, df, copy.deepcopy(ppc),
                    conductor, weather, load, tdpf_analysis,
                )

    df.to_csv(df_path, index=False)


if __name__ == '__main__':
    main()

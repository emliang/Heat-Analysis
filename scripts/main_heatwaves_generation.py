#!/usr/bin/env python3
"""Future heatwave scenario generation via bias-corrected delta mapping.

This script generates spatially resolved, hourly heatwave weather
profiles for European countries by combining:

- **Historical ERA5 reanalysis** (observed heatwave events)
- **RCP 4.5 climate projections** (EURO-CORDEX, CCLM4-8-17 / MPI-ESM-LR)

Three types of output figures are produced for each (country, variable,
year) combination:

1. **Spatial maps** — gridded snapshots of the four weather scenarios
   (historical reference / heatwave, future reference / heatwave).
2. **National-average diurnal profiles** — 24-hour curves comparing
   historical and future reference / heatwave conditions.
3. **Sampled regional profiles** — per-bus diurnal curves showing
   ensemble spread from multiple historical seed events.

Usage
-----
    python test_heatwaves_generation.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(os.getcwd(), "."))
from data_config import *                    # noqa: F403
from utils.network_process_utils import *    # noqa: F403
from utils.heat_flow_utils import *          # noqa: F403
from utils.heatwave_utils import *           # noqa: F403
from utils.plot_utils import *               # noqa: F403

# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})


# ===========================================================================
# Helper: build colour list from a variable's colourmap
# ===========================================================================

def _build_cmap_colors(variable, n_colors=8):
    """Return a list of *n_colors* sampled from the variable's colourmap.

    For wind variables the palette is reversed (high wind → cool colour).
    """
    cmap = cmap_dic[variable]
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    if 'wnd' in variable:
        colors = colors[::-1]
    return colors


# ===========================================================================
# Plot A — Spatial maps
# ===========================================================================

def plot_spatial_maps(variable, weather_profiles, name_list, cmap, vnom,
                      cbar_label, selected_country, fut_year, his_year,
                      heatwave_month):
    """Save a spatial snapshot for each of the four weather scenarios.

    Parameters
    ----------
    weather_profiles : list of xarray.DataArray
        Four gridded fields: historical baseline, historical heatwave,
        future baseline, future heatwave.
    name_list : list of str
        Display names matching *weather_profiles*.
    """
    for profile, name in zip(weather_profiles, name_list):
        data = profile.data - TK if variable == 'temperature' else profile.data
        lon = profile['x'].data - 0.125
        lat = profile['y'].data - 0.125

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plt.pcolormesh(
            lon, lat, data, shading='auto', cmap=cmap,
            norm=plt.Normalize(vmin=vnom[0], vmax=vnom[1]), alpha=0.7,
            zorder=0,
        )
        ax.add_feature(cfeature.OCEAN, zorder=1)
        ax.add_feature(cfeature.BORDERS, zorder=1)
        plt.title(name, fontsize=22, fontweight='bold')
        plt.savefig(
            MODELS + f'/heatwave/{selected_country}/'
            f'{fut_year}_{his_year}_{heatwave_month}_{variable}_{name}.png',
            dpi=300, bbox_inches='tight', pad_inches=0.01,
        )
        plt.close()

    # Shared colourbar (saved once per variable)
    create_discrete_temperature_colorbar(
        cmap=cmap, num_levels=vnom[2], vmin=vnom[0], vmax=vnom[1],
        alpha=0.7, label=cbar_label, extend='both', figsize=(14, 0.2),
    )
    plt.savefig(
        MODELS + f'/heatwave/heatwave_{variable}_colorbar.pdf',
        format='pdf', dpi=300, pad_inches=0.01, bbox_inches='tight',
    )
    plt.close()


# ===========================================================================
# Plot B — National-average diurnal profiles
# ===========================================================================

def plot_national_average(variable, cbar_label, cmap_colors, regions,
                          his_era5_slice, his_rcp45_hourly,
                          fut_rcp45_hourly, historical_date, future_date,
                          selected_country, fut_year, his_year):
    """24-hour national-average comparison of reference vs heatwave."""
    (bus_his_hw, bus_his_bl,
     bus_fut_bl, bus_fut_hw) = temperal_trend_data(
        variable, regions, his_era5_slice, his_rcp45_hourly,
        fut_rcp45_hourly, historical_date, future_date,
    )

    # National averages (mean across buses)
    nat_his_bl = bus_his_bl.mean(1)
    nat_his_hw = bus_his_hw.mean(1)
    nat_fut_bl = bus_fut_bl.mean(1)
    nat_fut_hw = bus_fut_hw.mean(1)

    c_his = cmap_colors[1]
    c_fut = cmap_colors[-2]

    plt.figure(figsize=(12.5, 3.5))

    # Historical band + curves
    plt.fill_between(range(24), nat_his_bl, nat_his_hw,
                     alpha=0.1, color=c_his)
    plt.plot(nat_his_bl, c=c_his, alpha=0.7, linewidth=2)
    plt.plot(nat_his_hw, c=c_his, linestyle='-.', linewidth=2)

    # Future band + curves
    plt.fill_between(range(24), nat_fut_bl, nat_fut_hw,
                     alpha=0.1, color=c_fut)
    plt.plot(nat_fut_bl, c=c_fut, alpha=0.7, linewidth=2)
    plt.plot(nat_fut_hw, c=c_fut, linestyle='-.', linewidth=2)

    # Legend entries (empty plots for label handles)
    plt.plot([], label='Historical reference', c=c_his,
             alpha=0.7, linewidth=2.5)
    plt.plot([], label='Historical heatwave', c=c_his,
             linestyle='-.', linewidth=2.5)
    plt.plot([], label='Future reference', c=c_fut,
             alpha=0.7, linewidth=2.5)
    plt.plot([], label='Future heatwave', c=c_fut,
             linestyle='-.', linewidth=2.5)

    all_vals = np.concatenate([nat_his_bl, nat_his_hw,
                               nat_fut_bl, nat_fut_hw])
    plt.xlim(0, 23)
    plt.ylim(all_vals.min() * 0.9, all_vals.max() * 1.1)
    plt.xticks(range(24), fontsize=14)
    plt.xlabel('Hour', fontsize=16, fontweight='bold')
    plt.ylabel(cbar_label, fontsize=16, fontweight='bold')
    plt.legend(
        ncol=4, loc=2, frameon=False, columnspacing=1.0,
        bbox_to_anchor=(0.0, 1.2),
        prop={'size': 15, 'weight': 'bold'},
    )
    plt.grid(linewidth=0.25, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        MODELS + f'/heatwave/{selected_country}/'
        f'heatwave_{variable}_{selected_country}_{fut_year}_{his_year}.pdf',
        format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.01,
    )
    plt.close()


# ===========================================================================
# Plot C — Sampled regional profiles
# ===========================================================================

def plot_sampled_regions(variable, cbar_label, cmap_colors, regions, network,
                         his_era5_slice, his_rcp45_hourly, fut_rcp45_hourly,
                         historical_heatwave_index_list,
                         historical_heatwave_date_list, future_date,
                         selected_country, fut_year, his_year,
                         n_regions=9, n_samples=5):
    """Per-region diurnal curves showing ensemble spread."""
    np.random.seed(2026)
    bus_indices = np.random.choice(
        len(network.buses), n_regions, replace=False)

    # Collect heatwave ensembles and track global min/max
    hw_list = []
    fut_max, fut_min = 0.0, 1e6

    for i in historical_heatwave_index_list[:n_samples]:
        hist_date = pd.Timestamp(
            historical_heatwave_date_list[i])
        _, _, bus_fut_bl, bus_fut_hw = temperal_trend_data(
            variable, regions, his_era5_slice, his_rcp45_hourly,
            fut_rcp45_hourly, hist_date, future_date,
        )
        fut_max = np.maximum(fut_max,
                             bus_fut_hw[:, bus_indices].max())
        fut_min = np.minimum(fut_min,
                             bus_fut_hw[:, bus_indices].min())
        hw_list.append(bus_fut_hw)

    fut_max = max(fut_max, bus_fut_bl[:, bus_indices].max())
    fut_min = min(fut_min, bus_fut_bl[:, bus_indices].min())
    c_fut = cmap_colors[-2]

    for k, bus_idx in enumerate(bus_indices):
        plt.figure(figsize=(5.5, 3.5))
        plt.plot(bus_fut_bl[:, bus_idx], c=c_fut,
                 alpha=0.7, linewidth=2)
        for hw in hw_list:
            plt.plot(hw[:, bus_idx], c=c_fut,
                     linestyle='--', linewidth=1, alpha=0.8)

        # Legend handles
        plt.plot([], label='Reference', c=c_fut,
                 linewidth=2.5, alpha=0.7)
        plt.plot([], label='Heatwaves', c=c_fut,
                 linestyle='--', linewidth=2.5, alpha=0.8)

        plt.xlim(0, 23)
        plt.ylim(fut_min * 0.9, fut_max * 1.1)
        plt.xticks(range(0, 24, 4), fontsize=14)
        plt.xlabel('Hour', fontsize=18, fontweight='bold')
        plt.ylabel(cbar_label, fontsize=18, fontweight='bold')
        plt.grid(linewidth=0.25, alpha=0.25)
        plt.legend(
            ncol=2, loc=2, frameon=False, columnspacing=1,
            handlelength=1.5, bbox_to_anchor=(0, 1.25),
            prop={'size': 18, 'weight': 'bold'},
        )
        plt.text(
            0.05, 0.95, f'Region {k + 1}',
            transform=plt.gca().transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='left',
        )
        plt.tight_layout()
        plt.savefig(
            MODELS + f'/heatwave/{selected_country}/'
            f'{k}_temperal_heatwave_{variable}'
            f'_{selected_country}_{fut_year}_{his_year}.pdf',
            format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.01,
        )
        plt.close()


# ===========================================================================
# Main pipeline
# ===========================================================================

def main(selected_country, variable,
         fut_year=2030, his_year=2022, heatwave_month=7,
         plot_ave=True, plot_spatial=True, plot_sample=False):
    """Generate bias-corrected heatwave scenarios for one country/variable.

    Parameters
    ----------
    selected_country : str
        Two-letter country code.
    variable : str
        ERA5 variable name (``'temperature'``, ``'influx'``, ``'wnd10m'``).
    fut_year : int
        Target future year from RCP 4.5 projections.
    his_year : int
        Historical reference year in ERA5.
    heatwave_month : int
        Month of interest (6 = Jun, 7 = Jul, 8 = Aug).
    plot_ave : bool
        Generate national-average diurnal profile.
    plot_spatial : bool
        Generate spatial snapshot maps.
    plot_sample : bool
        Generate sampled regional ensemble plots.
    """
    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    his_era5 = xr.open_dataset(
        EXTERNAL + '/era5/era5_hourly_summer_2019_2024.nc')
    his_rcp45 = xr.open_dataset(
        EXTERNAL + '/rcp45/rcp45_3hourly_summer_2019_2024.nc')
    fut_rcp45 = xr.open_dataset(
        EXTERNAL + '/rcp45/rcp45_3hourly_summer_2025_2030.nc')

    # ------------------------------------------------------------------
    # 2. Load network & determine spatial extent
    # ------------------------------------------------------------------
    os.makedirs(MODELS + f'/heatwave/{selected_country}', exist_ok=True)
    network, regions = load_network_EU([selected_country], RATIO)
    country_range = [
        network.buses.x.min() - 0.25, network.buses.x.max() + 0.25,
        network.buses.y.min() - 0.25, network.buses.y.max() + 0.25,
    ]

    # ------------------------------------------------------------------
    # 3. Slice & interpolate to hourly resolution
    # ------------------------------------------------------------------
    date_s = datetime(his_year, heatwave_month, 1, 0)
    date_e = datetime(his_year, heatwave_month, 30, 23)
    fut_date_s = datetime(fut_year, heatwave_month, 1, 0)
    fut_date_e = datetime(fut_year, heatwave_month, 30, 23)

    bbox = dict(
        x=slice(country_range[0], country_range[1]),
        y=slice(country_range[2], country_range[3]),
    )
    his_era5_slice = his_era5.sel(time=slice(date_s, date_e), **bbox)
    his_rcp45_slice = his_rcp45.sel(time=slice(date_s, date_e), **bbox)
    fut_rcp45_slice = fut_rcp45.sel(time=slice(fut_date_s, fut_date_e), **bbox)

    his_rcp45_hourly = interpolate_3h_to_1h(his_rcp45_slice)
    fut_rcp45_hourly = interpolate_3h_to_1h(fut_rcp45_slice)

    # ------------------------------------------------------------------
    # 4. Identify heatwave days
    # ------------------------------------------------------------------
    his_hw_idx, his_hw_dates = find_heatwave_days(
        his_era5_slice, regions, weights=[0.9, 0.1])
    fut_hw_idx, fut_hw_dates = find_heatwave_days(
        fut_rcp45_hourly, regions, weights=[0.9, 0.1])

    historical_date = pd.Timestamp(
        his_hw_dates[his_hw_idx[0]]).replace(hour=14)
    future_date = pd.Timestamp(
        fut_hw_dates[fut_hw_idx[0]]).replace(hour=14)
    print(f'{selected_country}: hist={historical_date.date()}, '
          f'fut={future_date.date()}')

    # ------------------------------------------------------------------
    # 5. Variable-specific colour setup
    # ------------------------------------------------------------------
    vnom = vnom_dic[variable]
    cmap = cmap_dic[variable]
    cbar_label = cbar_lable_dic[variable]
    cmap_colors = _build_cmap_colors(variable)

    # ------------------------------------------------------------------
    # 6. Bias-correct future heatwave snapshot
    # ------------------------------------------------------------------
    weather_his_hw = his_era5_slice.sel(
        time=historical_date, method='nearest')
    weather_his_bl = his_rcp45_hourly.sel(
        time=historical_date, method='nearest')
    weather_fut_bl = fut_rcp45_hourly.sel(
        time=future_date, method='nearest')
    weather_fut_hw = bias_correction(
        variable, weather_his_hw, weather_his_bl, weather_fut_bl)

    # ------------------------------------------------------------------
    # 7. Generate requested plots
    # ------------------------------------------------------------------
    if plot_spatial:
        plot_spatial_maps(
            variable,
            [weather_his_bl[variable], weather_his_hw[variable],
             weather_fut_bl[variable], weather_fut_hw[variable]],
            ['Historical reference', 'Historical heatwave',
             'Future reference', 'Future heatwave'],
            cmap, vnom, cbar_label,
            selected_country, fut_year, his_year, heatwave_month,
        )

    if plot_ave:
        plot_national_average(
            variable, cbar_label, cmap_colors, regions,
            his_era5_slice, his_rcp45_hourly, fut_rcp45_hourly,
            historical_date, future_date,
            selected_country, fut_year, his_year,
        )

    if plot_sample:
        plot_sampled_regions(
            variable, cbar_label, cmap_colors, regions, network,
            his_era5_slice, his_rcp45_hourly, fut_rcp45_hourly,
            his_hw_idx, his_hw_dates, future_date,
            selected_country, fut_year, his_year,
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    PLOT_AVE = True
    PLOT_SPATIAL = False
    PLOT_SAMPLE = False

    for country in ['FR', 'ES', 'IT', 'BE', 'GB', 'DE', 'PT', 'NL']:
        for var in ['temperature', 'influx', 'wnd10m']:
            for hist_yr in [2019, 2022, 2024]:
                for fut_yr in [2026, 2027, 2028, 2029, 2030]:
                    main(country, var,
                         fut_year=fut_yr, his_year=hist_yr,
                         plot_ave=PLOT_AVE, plot_spatial=PLOT_SPATIAL,
                         plot_sample=PLOT_SAMPLE)

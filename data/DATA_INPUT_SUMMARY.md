# Data Input Summary

Input guide for the full HeatAnalysis pipeline. Shared paths and model defaults
are defined in [`data_config.py`](../data_config.py); experiment and scenario
settings also appear in the corresponding entry-point scripts.

For the self-contained one-hour example, use
[`examples/es_2030-07-24_13_2024-0/`](examples/es_2030-07-24_13_2024-0/).
Its manifest lists the bundled inputs and hashes. The
[introductory notebooks](../vis/demos/README.md) use that bundle and small
companion datasets. The [publication figure packages](../vis/figure_source/README.md)
contain their own figure data and do not require the full pipeline inputs below.

> All paths below are relative to the project root (`HeatAnalysis/`)
> unless marked as external.

---

## 1. External Large Datasets

Downloaded separately (Google Drive or Copernicus CDS). Set `EXTERNAL` in `data_config.py` to the folder containing these files.

| File | Format | Description | Used by |
|------|--------|-------------|---------|
| `era5/era5_hourly_summer_2019_2024.nc` | NetCDF | ERA5 hourly reanalysis (summer months) — temperature, wind, solar, humidity | Heatwave generation, profile builder |
| `era5/era5_daily_avg_2015_2024.nc` | NetCDF | ERA5 daily-averaged weather over Europe, 2015–2024 | Demand calibration |
| `rcp45/rcp45_3hourly_summer_2019_2024.nc` | NetCDF | EURO-CORDEX RCP 4.5 (CCLM4-8-17 / MPI-ESM-LR), historical baseline | Heatwave generation, profile builder |
| `rcp45/rcp45_3hourly_summer_2025_2030.nc` | NetCDF | EURO-CORDEX RCP 4.5 future projections, 2025–2030 | Heatwave generation, profile builder |
| `elevation.nc` | NetCDF | Surface elevation grid | Profile builder (conductor height) |

---

## 2. PyPSA-Eur Files

Read directly from the `pypsa-eur/` tree.  Network snapshots and bus regions have been reorganised into `data/EU/` (see section 3).

| File | Format | Description | Used by |
|------|--------|-------------|---------|
| `config/config.yaml` | YAML | Renewable siting parameters (capacity per sqkm, exclusion zones, correction factors) | `main_build_simulation_profile.py` |
| `data/natura.tiff` | GeoTIFF | Natura 2000 protected-area raster | Profile builder (exclusion) |
| `data/bundle/corine/g250_clc06_V18_5.tif` | GeoTIFF | CORINE land-cover raster | Profile builder (exclusion) |
| `data/shipdensity_raster.tif` | GeoTIFF | Shipping-lane density raster | Profile builder (offshore wind exclusion) |
| `data/bundle/GEBCO_2014_2D.nc` | NetCDF | GEBCO bathymetry (ocean depth) | Profile builder (offshore wind max depth) |

---

## 3. Local Project Data (`data/`)

The regional GeoJSON files, load-ratio CSVs, demand Parquet file and IEEE case
listed below are included. Full network snapshots and the two shape files
must be obtained separately or prepared with PyPSA-Eur. The one-hour example
contains its own network snapshot.

### 3a. Network & Regions (`data/EU/`)

`{ratio}` = network clustering level (50 / 75 / 100).

| File | Format | Description | Used by |
|------|--------|-------------|---------|
| `networks/base_s_{ratio}_elec.nc` | NetCDF | Pan-European solved network (buses, generators, lines, links, storage, loads, time-series) | All scripts, `TDOPF_eur.py` (download from google drive or prepared via pypsa-eur) |
| `regions_onshore_base_s_{ratio}.geojson` | GeoJSON | Voronoi bus regions (onshore) | Demand calibration, heatwave generation, profile builder, TDOPF |
| `regions_offshore_base_s_{ratio}.geojson` | GeoJSON | Voronoi bus regions (offshore) | Profile builder |
| `load_ratio_base_s_{ratio}.csv` | CSV | Population-weighted load fractions per bus | `network_process_utils.py` |
| `country_shapes.geojson` | GeoJSON | Country boundaries | Renewable profile builder; obtain from PyPSA-Eur |
| `offshore_shapes.geojson` | GeoJSON | Offshore boundaries | Renewable profile builder; obtain from PyPSA-Eur |

### 3b. Demand (`data/entsoe/`)

| File | Format | Description | Used by |
|------|--------|-------------|---------|
| `MHLV_2015-2024.parquet` | Parquet | ENTSO-E hourly electricity demand (MW) per country, 2015–2024 | `main_demand_calibration.py` |

### 3c. IEEE Test Cases (`data/ieee_data/`)

| File | Format | Description | Used by |
|------|--------|-------------|---------|
| `casefiles_mat/case_30.mat` | MATLAB `.mat` | IEEE 30-bus system (buses, generators, branches, costs) | `TDOPF_ieee.py` |

---

## 4. Generated Intermediates (`models/`)

The input-generation stages produce calibrated demand curves, hourly scenario
inputs and renewable profiles. The repository includes demand curves and a
subset of Spanish profiles, not the complete multi-country experiment inputs.
Additional pre-built inputs are available on
[Google Drive](https://drive.google.com/drive/folders/1SJmglPiEMTw--xggqzeiSjmKBCI17cnK).
Weather illustrations are optional outputs, not solver inputs.

### 4a. Demand Curves

| File | Format | Produced by | Consumed by |
|------|--------|-------------|-------------|
| `demand_curve/{CC}/{CC}_{years}_demand_curve.npy` | NumPy | `main_demand_calibration.py` | Profile builder, `TDOPF_eur.py` |

### 4b. Heatwave Scenarios

| File | Format | Produced by | Consumed by |
|------|--------|-------------|-------------|
| `heatwave/{CC}/*.pdf` | PDF | `main_heatwaves_generation.py` | Visual inspection only |

### 4c. Simulation Profiles

| File | Format | Produced by | Consumed by |
|------|--------|-------------|-------------|
| `{CC}/simu_data/future_demand_data_based_on_historical_hot_event_{year}_{idx}.npy` | NumPy | Profile builder | `TDOPF_eur.py` |
| `{CC}/simu_data/future_weather_data_based_on_historical_hot_event_{year}_{idx}.nc` | NetCDF | Profile builder | `TDOPF_eur.py` |
| `{CC}/simu_data/future_hot_dates.npy` | NumPy | Profile builder | `TDOPF_eur.py` |

### 4d. Renewable Capacity Factors

| File | Format | Produced by | Consumed by |
|------|--------|-------------|-------------|
| `{CC}/weather/{year}_{idx}_solar_profile_s_{ratio}.nc` | NetCDF | Profile builder (Atlite) | `TDOPF_eur.py` |
| `{CC}/weather/{year}_{idx}_onwind_profile_s_{ratio}.nc` | NetCDF | Profile builder (Atlite) | `TDOPF_eur.py` |
| `{CC}/weather/{year}_{idx}_offwind-ac_profile_s_{ratio}.nc` | NetCDF | Profile builder (Atlite) | `TDOPF_eur.py` |
| `{CC}/weather/{year}_{idx}_offwind-dc_profile_s_{ratio}.nc` | NetCDF | Profile builder (Atlite) | `TDOPF_eur.py` |

---

## 5. Outputs

The repository retains selected saved results and figures under `models/`.
New runs can write to the same paths; use a separate output directory to
preserve those files. Smoke-check outputs and `models/runs/` are ignored by Git.
The submitted figure data are distributed separately under `vis/figure_source/`.

| File | Format | Produced by |
|------|--------|-------------|
| `models/{CC}/{CC}_{nbus}_bus_..._analysis.csv` | CSV | `TDOPF_eur.py` |
| `models/{CC}/ppc/{timestamp}/{expr_id}_results.npy` | NumPy | `TDOPF_eur.py` |
| `models/IEEE30/30_heatflow_analysis.csv` | CSV | `TDOPF_ieee.py` |

---

*Paths are relative to `HeatAnalysis/` unless prefixed with `EXTERNAL`.
Shared defaults are defined in [`data_config.py`](../data_config.py); sweep
parameters and analysis modes are selected in the experiment entry points.*

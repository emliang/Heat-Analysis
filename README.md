<p align="center">
  <h1 align="center">HeatAnalysis</h1>
  <p align="center">
    <b>Heatwave Impact on European Electricity Grids:<br>Temperature-Dependent Optimal Power Flow Framework</b>
  </p>
  <p align="center">
    <a href="#-installation"><img src="https://img.shields.io/badge/python-%3E%3D3.9-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--4.0-green" alt="License"></a>
    <a href="https://github.com/PyPSA/pypsa-eur"><img src="https://img.shields.io/badge/built%20on-PyPSA--Eur-orange" alt="PyPSA-Eur"></a>
    <a href="https://coin-or.github.io/Ipopt/"><img src="https://img.shields.io/badge/solver-IPOPT-red" alt="IPOPT"></a>
    <a href="https://standards.ieee.org/ieee/738/6228/"><img src="https://img.shields.io/badge/standard-IEEE%20738--2012-lightblue" alt="IEEE 738"></a>
  </p>
  <p align="center">
    <a href="#-background">Background</a> &nbsp;|&nbsp;
    <a href="#-methodology">Methodology</a> &nbsp;|&nbsp;
    <a href="#-key-features">Features</a> &nbsp;|&nbsp;
    <a href="#-workflow">Workflow</a> &nbsp;|&nbsp;
    <a href="#-installation">Installation</a> &nbsp;|&nbsp;
    <a href="#-citation">Citation</a>
  </p>
</p>

---

## 🌡 Background

Climate change is increasing the frequency, intensity, and duration of [**heatwaves**](https://climate.copernicus.eu/heatwaves-brief-introduction) across Europe. These extreme heat events pose a **triple threat** to electricity grids:

- **Surging demand** — cooling loads spike during heatwaves, pushing peak consumption to record levels.
- **Reduced supply** — thermal and nuclear generators derate under high ambient temperatures; renewable output fluctuates with weather.
- **Degraded transmission capacity** — overhead conductors heat up, increasing resistance and sagging, which forces operators to reduce power flow to maintain safety clearances.

<p align="center">
  <img src="images/temperature.png" width="700" /><br>
  <em>European summer (JJA) temperature anomalies relative to 1991-2020 baseline.
  Credit: <a href="https://climate.copernicus.eu/european-heatwave-july-2023-longer-term-context">C3S/ECMWF/KNMI</a></em>
</p>

Understanding these compounding effects is essential for enhancing grid resilience. This framework provides a **quantitative, physics-based assessment** of heatwave impacts on European power systems, from projected weather scenarios all the way to optimal power flow analysis.

<p align="center">
  <img src="images/heatwave.png" width="550" /><br>
  <em>European surface temperature during a major heatwave event.
  Credit: <a href="https://www.cpc.ncep.noaa.gov/">NOAA Climate Prediction Center</a></em>
</p>

---

## 🔬 Methodology

The framework integrates **climate projections**, **thermal modeling**, and **power system optimisation** into a unified simulation pipeline:

<p align="center">
  <img src="images/framework_update_2025.png" width="800" /><br>
  <em>Overview of the HeatAnalysis framework: heatwaves simultaneously reduce transmission capacity (left),
  increase cooling demand, derate generators, and alter renewable output (right).</em>
</p>

The key methodological components are:

| Component | Description |
|-----------|-------------|
| **Future Heatwave Projection** | Generates projected heatwave events (2026-2030) via bias-corrected delta mapping from historical extremes (2019, 2022, 2024) onto climate projections |
| **Weather-Driven Demand Model** | BAIT thermal-comfort index calibrated against ENTSO-E hourly load data to capture temperature-demand coupling |
| **Renewable Generation** | Weather-driven capacity factors computed via [Atlite](https://github.com/PyPSA/atlite) for solar, onshore wind, and offshore wind |
| **Conductor Thermal Model** | IEEE Std 738-2012 heat-balance equation solved per line segment to obtain spatially-resolved conductor temperatures |
| **Multi-Segment Modelling** | Transmission lines subdivided along the ERA5 grid to capture localised thermal hotspots |
| **Generator Derating** | Temperature-dependent capacity reduction for conventional generators |
| **Iterative TD-ACOPF** | Temperature-dependent AC optimal power flow with electricity-temperature feedback loop until convergence |

The iterative TD-ACOPF solver alternates between the AC-OPF solution (which determines branch currents) and the heat-balance equation (which determines conductor temperatures), updating network parameters at each step:

<p align="center">
  <img src="images/opf_analysis.png" width="800" /><br>
  <em>Iterative TD-ACOPF workflow: projected heatwave snapshots (left) are fed into the OPF ↔ heat-flow
  iteration loop (centre), producing spatially-resolved congestion and load-shedding maps (right).</em>
</p>

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **IEEE 738 Heat-Balance Model** | Steady-state conductor temperature from convective/radiative cooling and solar/Joule heating, with Bisection/Newton solvers |
| **Iterative TD-ACOPF** | electricity-temperature feedback loop updating network parameters at each OPF iteration until convergence |
| **Pyomo / IPOPT Solver** | Full AC power flow with rectangular or polar voltage formulations, N-1 security constraints (AC & linearised LODF), DC lines, storage, and load-shedding |
| **Generator Derating** | Weather-dependent capacity modeling for thermal, nuclear, and renewable generators |
| **Heatwave Scenarios** | Bias-corrected future heatwave profiles from ERA5 reanalysis and RCP45 climate projections via delta mapping |
| **Multi-Country Networks** | Supports PyPSA-Eur networks (ES, FR, IT, GB, DE, PT, NL, BE) and IEEE 30-bus benchmarks |
| **Parallel Simulation** | Multiprocessing support for large-scale sensitivity sweeps across heatwave years, load-growth rates, storage ratios, and thermal limits |

---

## 📁 Project Structure

```
HeatAnalysis/
│
├── TDOPF_eur.py                  # Main entry point: European TD-ACOPF pipeline
├── TDOPF_ieee.py                 # IEEE test-system TD-OPF analysis
├── data_config.py                # Project-wide paths, constants, and parameters
│
├── utils/                        # Core library modules
│   ├── heat_flow_utils.py        #   IEEE 738 heat-balance & ampacity calculations
│   ├── opf_pyomo_utils.py        #   ACOPF solver (Pyomo/IPOPT)
│   ├── network_process_utils.py  #   Network loading, segmentation, PyPSA→PyPower
│   ├── demand_utils.py           #   BAIT demand model & SCEM calibration
│   ├── heatwave_utils.py         #   Heatwave scenario generation & bias correction
│   ├── country_network_filter.py #   Country-level PyPSA network filtering
│   └── plot_utils.py             #   Shared plotting and visualisation helpers
│
├── scripts/                      # Data-processing & calibration scripts
│   ├── 0.download_weather_data.ipynb       # Download ERA5 / CMIP6 data via CDS API
│   ├── 1.process_weather_profile.ipynb     # Process weather & demand time-series
│   ├── 2.test_demand_calibration.ipynb     # Visualise demand-model calibration
│   ├── 3.test_creat_heatwaves.ipynb        # Visualise generated heatwave scenarios
│   ├── 4.test_solve_heat_balance.ipynb     # Heat-balance equation validation
│   ├── concurrent_download.py              # Concurrent ERA5 download utility
│   ├── main_build_simulation_profile.py    # Build simulation-ready profiles
│   ├── main_demand_calibration.py          # Run demand-model calibration
│   └── main_heatwaves_generation.py        # Generate future heatwave scenarios
│
├── vis/                          # Result visualisation notebooks
│   ├── 1.Spain_grid_example.ipynb          # Illustrative Spanish grid example
│   ├── 2.eur_single_analysis.ipynb         # Single-country result analysis
│   ├── 3.eur_multi_analysis.ipynb          # Multi-country & cross-border analysis
│   ├── 4.grid_simu_vis.ipynb               # Network-level simulation visualisation
│   └── 5.ieee_simu_vis.ipynb               # IEEE test-case visualisation
│
├── data/                         # Input data (not tracked in git)
│   ├── EU/                       #   PyPSA-Eur network files & region shapes
│   ├── era5/                     #   ERA5 reanalysis weather data
│   ├── entsoe/                   #   ENTSO-E hourly demand data
│   └── ieee_data/                #   IEEE 30-bus MATPOWER case files
│
└── models/                       # Intermediate & output data (not tracked in git)
    ├── {country_code}/           #   Per-country simulation results & profiles
    ├── demand_curve/             #   Calibrated demand-model parameters
    ├── heatwave/                 #   Generated heatwave scenarios
    └── IEEE30/                   #   IEEE 30-bus results
```

---

## 📊 Data Sources

| Data | Description |
|------|-------------|
| [PyPSA-Eur](https://pypsa-eur.readthedocs.io/) | Open-source European transmission network model |
| [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels) | Historical hourly global climate reanalysis from ECMWF |
| [C3S Energy](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-projections) | Future climate projections for energy applications |
| [ENTSO-E](https://www.entsoe.eu/data/power-stats/) | Historical hourly country-level power demand data |
| [Atlite](https://github.com/PyPSA/atlite) | Weather-driven renewable generation model |

---

## 🔄 Workflow

The analysis pipeline follows **seven stages** from raw data to publication-quality figures:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Stage 0  │───▶│ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │───▶│ Stage 5  │───▶│ Stage 6  │
│   Data   │    │ Weather  │    │  Demand  │    │ Heatwave │    │  Build   │    │ TD-ACOPF │    │  Result  │
│ Download │    │ Process  │    │  Calib.  │    │ Scenario │    │ Profiles │    │  Solver  │    │   Vis.   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Stage 0 — Data Acquisition

Download ERA5 reanalysis and climate projection data from the Copernicus Climate Data Store.

```bash
# Interactive (notebook)
jupyter notebook scripts/0.download_weather_data.ipynb

# Or use the concurrent downloader
python scripts/concurrent_download.py
```

### Stage 1 — Weather & Demand Processing

Extract country-level weather slices, compute derived variables (humidity, wind components), and prepare ENTSO-E hourly demand time-series.

```bash
jupyter notebook scripts/1.process_weather_profile.ipynb
```

### Stage 2 — Demand-Model Calibration

Calibrate the BAIT thermal-comfort demand model per country using SCEM optimisation, fitting temperature-demand relationships to historical ENTSO-E data.

```bash
python scripts/main_demand_calibration.py

# Visualise results:
jupyter notebook scripts/2.test_demand_calibration.ipynb
```

### Stage 3 — Heatwave Scenario Generation

Construct future heatwave scenarios (2026-2030) by applying bias-corrected delta mapping from historical extreme events (2019, 2022, 2024) onto climate projections.

```bash
python scripts/main_heatwaves_generation.py

# Visualise results:
jupyter notebook scripts/3.test_creat_heatwaves.ipynb
```

### Stage 4 — Build Simulation Profiles

Assemble simulation-ready inputs: weather profiles (temperature, wind, solar), demand profiles, and renewable capacity factors per bus and timestep.

```bash
python scripts/main_build_simulation_profile.py
```

### Stage 5 — Run TD-ACOPF

Execute the temperature-dependent ACOPF analysis across countries, heatwave scenarios, and sensitivity parameters.

```bash
# European networks (single or multi-country)
python TDOPF_eur.py

# IEEE 30-bus benchmark
python TDOPF_ieee.py
```

### Stage 6 — Result Visualisation

Analyse and visualise simulation results using the dedicated notebooks in [`vis/`](vis/). These produce publication-quality figures for load-shedding maps, branch congestion, conductor temperature distributions, and cross-border flow analysis.

```bash
# Open any visualisation notebook, e.g.:
jupyter notebook vis/2.eur_single_analysis.ipynb
```

| Notebook | Description |
|----------|-------------|
| [`1.Spain_grid_example.ipynb`](vis/1.Spain_grid_example.ipynb) | Illustrative grid overlay on geographic map |
| [`2.eur_single_analysis.ipynb`](vis/2.eur_single_analysis.ipynb) | Single-country load-shedding, congestion, and thermal analysis |
| [`3.eur_multi_analysis.ipynb`](vis/3.eur_multi_analysis.ipynb) | Cross-border flow and multi-country comparison |
| [`4.grid_simu_vis.ipynb`](vis/4.grid_simu_vis.ipynb) | Spatial network-level simulation maps |
| [`5.ieee_simu_vis.ipynb`](vis/5.ieee_simu_vis.ipynb) | IEEE 30-bus case study figures |

---

## 🛠 Installation

### Prerequisites

| Requirement | Note |
|-------------|------|
| **Python** | >= 3.9 |
| **[IPOPT](https://coin-or.github.io/Ipopt/)** | Non-linear solver required by Pyomo |
| **[CDS API key](https://cds.climate.copernicus.eu/how-to-api)** | Only needed for data download (Stage 0) |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/HeatAnalysis.git
cd HeatAnalysis

# 2. Create a virtual environment (recommended)
conda create -n heatanalysis python=3.11
conda activate heatanalysis

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install IPOPT solver (pick one)
brew install ipopt                       # macOS
sudo apt install coinor-libipopt-dev     # Ubuntu/Debian
conda install -c conda-forge ipopt       # Conda (any OS)
```

### Data Setup

1. Obtain a PyPSA-Eur base network (see [PyPSA-Eur docs](https://pypsa-eur.readthedocs.io/)) and place files under `data/EU/`.
2. Download ENTSO-E hourly demand data from the [Transparency Platform](https://transparency.entsoe.eu/) into `data/entsoe/`.
3. Configure local data paths in `data_config.py` to match your directory layout.

---

<!-- ## ⚡ Analysis Modes

The TD-ACOPF supports multiple analysis configurations controlled by a string tag:

| Tag | Thermal Model | Derating | Segmented | Security | Iterations |
|-----|:---:|:---:|:---:|:---:|:---:|
| `base` | - | - | - | - | 1 |
| `td_quad` | Quadratic | - | - | - | 1 |
| `td_seg_derate_iter_2` | Iterative | Yes | Yes | - | 2 |
| `td_seg_derate_iter_10` | Iterative | Yes | Yes | - | 10 |
| `td_derate_iter_2` | Iterative | Yes | - | - | 2 |
| `td_seg_iter_2` | Iterative | - | Yes | - | 2 |
| `td_sin_seg_derate_iter_2` | Iterative (single) | Yes | Yes | - | 2 |
| `base_fixsc` | - | - | - | Fixed (0.7) | 1 |
| `base_seg_derate` | - | Yes | Yes | - | 1 |

--- -->

## 📚 References

This project builds on the following standards, tools, and data sources:

- **IEEE Std 738-2012** — [IEEE Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors](https://standards.ieee.org/ieee/738/6228/)
- **PyPSA-Eur** — [PyPSA-Eur: An Open Optimisation Model of the European Transmission System](https://doi.org/10.1016/j.enconman.2018.08.084), *Energy Conversion and Management*, 2019
- **ERA5 Reanalysis** — [The ERA5 Global Reanalysis](https://doi.org/10.1002/qj.3803), *Quarterly Journal of the Royal Meteorological Society*, 2020
- **ENTSO-E** — [Transparency Platform](https://transparency.entsoe.eu/)
- **Atlite** — [https://github.com/PyPSA/atlite](https://github.com/PyPSA/atlite)

---

## 🪪 License

This project is licensed under the **[Creative Commons Attribution 4.0 International (CC-BY-4.0)](LICENSE)**.

You are free to **share** and **adapt** the material for any purpose, provided you give appropriate attribution.

---

## ✨ Citation

If you find HeatAnalysis helpful in your research, please consider citing:

```bibtex
@software{heatanalysis2025,
  title   = {HeatAnalysis: Temperature-Dependent Optimal Power Flow
             under Heatwave Scenarios},
  year    = {2025},
  url     = {https://github.com/<your-username>/HeatAnalysis}
}
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better.

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. Open a **Pull Request**

---

## 📬 Contact

If you have questions or encounter any issues, please [open an issue](https://github.com/<your-username>/HeatAnalysis/issues) on GitHub.

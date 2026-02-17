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
    <a href="https://emliang.github.io/Heat-Analysis/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=github" alt="Project Page"></a>
  </p>
  <p align="center">
    <a href="https://emliang.github.io/Heat-Analysis/">Project Page</a> &nbsp;|&nbsp;
    <a href="#-background">Background</a> &nbsp;|&nbsp;
    <a href="#-methodology">Methodology</a> &nbsp;|&nbsp;
    <a href="#-data-sources--flow">Data</a> &nbsp;|&nbsp;
    <a href="#-workflow">Workflow</a> &nbsp;|&nbsp;
    <a href="#-quick-start">Quick Start</a> &nbsp;|&nbsp;
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
  <img src="webpage/images/temperature.png" width="700" /><br>
  <em>European summer (JJA) temperature anomalies relative to 1991-2020 baseline.
  Credit: <a href="https://climate.copernicus.eu/european-heatwave-july-2023-longer-term-context">C3S/ECMWF/KNMI</a></em>
</p>

Understanding these compounding effects is essential for enhancing grid resilience. This framework provides a **quantitative, physics-based assessment** of heatwave impacts on European power systems, from projected weather scenarios all the way to optimal power flow analysis.

<p align="center">
  <img src="webpage/images/heatwave.png" width="550" /><br>
  <em>European surface temperature during a major heatwave event.
  Credit: <a href="https://www.cpc.ncep.noaa.gov/">NOAA Climate Prediction Center</a></em>
</p>

---

## 🔬 Methodology

The framework integrates **climate projections**, **thermal modeling**, and **power system optimisation** into a unified simulation pipeline:

<p align="center">
  <img src="webpage/images/framework_update_2025.png" width="800" /><br>
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
  <img src="webpage/images/opf_analysis.png" width="800" /><br>
  <em>Iterative TD-ACOPF workflow: projected heatwave snapshots (left) are fed into the OPF ↔ heat-flow
  iteration loop (centre), producing spatially-resolved congestion and load-shedding maps (right).</em>
</p>

---
<!-- 
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

--- -->

## 📁 Project Structure

```
HeatAnalysis/
├── TDOPF_eur.py / TDOPF_ieee.py   # TD-ACOPF entry points (European / IEEE)
├── data_config.py                  # All paths, constants, and parameters
├── utils/                          # Library modules (heat flow, OPF, demand, …)
├── scripts/                        # Data download, calibration, profile building
├── vis/                            # Result visualisation notebooks
├── data/                           # Input data  
└── models/                         # Outputs & intermediates 
```

---

## 📊 Data Sources & Flow

| Source | Used for |
|--------|----------|
| [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels) | Hourly weather reanalysis (temperature, wind, radiation) |
| [EURO-CORDEX RCP 4.5](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-projections) | Future climate projections for heatwave generation |
| [ENTSO-E](https://www.entsoe.eu/data/power-stats/) | Historical hourly country-level electricity demand |
| [PyPSA-Eur](https://pypsa-eur.readthedocs.io/) | European transmission network, bus regions, exclusion rasters |
| [Atlite](https://github.com/PyPSA/atlite) | Weather-driven renewable capacity factor calculation |

```
ERA5 + RCP 4.5 + ENTSO-E + PyPSA-Eur
          │
          ├─► Demand calibration     ──► demand curves
          ├─► Heatwave generation    ──► future weather scenarios
          ├─► Renewable profiles     ──► solar / wind capacity factors
          │
          └─► TD-ACOPF  ──► congestion, load-shedding, thermal maps
                              (TDOPF_eur.py / TDOPF_ieee.py)
```

> Full file-level inventory: [`DATA_INPUT_SUMMARY.md`](data/DATA_INPUT_SUMMARY.md)

---


## 🔄 Workflow

**Quick run** — 

Simple benchmark without external data or preparation
```bash
python TDOPF_ieee.py         # IEEE 30-bus benchmark
```

## 🛠 Quick Start

**All data required to run the project** is available on Google Drive :

> **[Download from Google Drive](https://drive.google.com/drive/folders/1SJmglPiEMTw--xggqzeiSjmKBCI17cnK)** (so you can skip Stages 0–4)
>
> The shared folder contains:
> - **Weather data** — ERA5 hourly & daily reanalysis, RCP 4.5 climate projections, elevation grid
> - **Simulation data** — pre-built heatwave scenarios and renewable capacity factors 
> - **PyPSA-Eur data** — network snapshots, bus regions, exclusion rasters, and config
> - **EU Gird data** - power grid data derived from pypsa-eur

After downloading:
1. Set `EXTERNAL` in [`data_config.py`](data_config.py) to the folder (e.g., WeatherData) containing `era5/`, `rcp45/`, and `elevation.nc`.
2. Copy the pre-built country folders (e.g. `ES/`, `FR/`, `GB/`, …) into `models/` — each contains `simu_data/` (demand & weather profiles) and `weather/` (renewable capacity factors) 
3. Copy the pre-built grid data (`networks/`) into `data/EU` as local data.

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/HeatAnalysis.git
cd HeatAnalysis
conda create -n heatanalysis python=3.11 && conda activate heatanalysis
pip install -r requirements.txt

# 2. Install IPOPT solver (pick one)
brew install ipopt                       # macOS
sudo apt install coinor-libipopt-dev     # Ubuntu/Debian
conda install -c conda-forge ipopt       # Conda (any OS)

# 3. Run
python TDOPF_eur.py          # European networks
```

> Detailed file inventory: [`DATA_INPUT_SUMMARY.md`](data/DATA_INPUT_SUMMARY.md) &
> [`data_config.py`](data_config.py).

---


**Full pipeline** — from raw data to figures:

| Stage | What | Command |
|:-----:|------|---------|
| 0 | Download weather data | `jupyter notebook scripts/0.download_weather_data.ipynb` |
| 1 | Process weather profiles | `jupyter notebook scripts/1.process_weather_profile.ipynb` |
| 2 | Calibrate demand model | `python scripts/main_demand_calibration.py` |
| 3 | Generate heatwave scenarios | `python scripts/main_heatwaves_generation.py` |
| 4 | Build simulation profiles | `python scripts/main_build_simulation_profile.py` |
| 5 | **Run TD-ACOPF** | `python TDOPF_eur.py` / `python TDOPF_ieee.py` |
| 6 | Visualise results | Notebooks in [`vis/`](vis/) (single-country, multi-country, IEEE) |

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
@article{heatanalysis2025,
  title   = {European Electricity Grids May Exhibit Heatwave-induced Capacity Bottlenecks},
  author  = {Liang, Enming and Chen, Minghua and Keshav, Srinivasan},
  year    = {2025}
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

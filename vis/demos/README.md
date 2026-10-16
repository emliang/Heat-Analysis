# Spain 2030 visual examples

These three notebooks follow the same represented Spanish grid and projected
heatwave hour, 24 July 2030 at 13:00. Start Jupyter from the repository root:

```bash
jupyter notebook vis/demos/
```

1. `01_heatwave_weather.ipynb` shows the projected weather field and the 24-hour
   profiles for the selected day.
2. `02_weather_to_grid.ipynb` shows the resulting demand profile, generator
   availability factors and weather-dependent transmission limits. It does not
   solve the OPF.
3. `03_solve_and_explore.ipynb` displays the selected published grid snapshot,
   runs the same one-hour OPF using the bundled inputs and compares the outputs.
   An IPOPT executable is required for the solve.

The small inputs in `data/spain_2030/` are ready to use. The OPF input bundle is
in `../../data/examples/es_2030-07-24_13_2024-0/`; no separate European data
download is needed to run the notebooks. The website reads a generated copy of
`data/spain_2030/grid_result.json` and does not run an optimisation model.

This hour is an illustrative scenario under the represented public-data grid
configuration. It is not a forecast that a heatwave or outage will occur on
24 July 2030. See `data/spain_2030/README.md` for the data sources and the
commands used to prepare the example.

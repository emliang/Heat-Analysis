# Data for the Spain 2030 examples

The selected hour is the `ES-2030` snapshot of Extended Data Fig. 2:
24 July 2030, 13:00, with the 2024-07-30 historical heatwave anchor, 1% annual
base-demand growth and 80% initial storage state of charge.

- `weather_day.nc` contains the 24 projected hourly weather fields for that day.
- `demand_hourly.csv` follows the study's calibrated demand model and hourly
  demand shape. Growth is applied to the base-demand component only.
- `generator_inputs.csv` and `line_inputs.csv` contain the selected hour's
  weather-dependent model inputs. The latter also includes solved line
  temperatures for comparison.
- `grid_result.json` contains the published `ES-2030` map values: weather
  cells, represented buses and lines, and displayed load-shedding markers.
  The website receives an identical generated copy.
- `provenance.json` records input and result hashes and the maximum numerical
  differences between a fresh example run and the published map values.

The published map values come from
`vis/figure_source/data/Source_Data_Extended_Data_Fig_2.zip`. The weather-day
and model-input files were extracted from the prepared simulation data and the
checked one-hour run. To regenerate the input bundle and visualisation data
from the full prepared-data checkout, use:

```bash
python scripts/prepare_spain_2030_example.py --source-root /path/to/HeatAnalysis --output /tmp/heat-analysis-es-2030-inputs
python scripts/run_european_example.py --bundle /tmp/heat-analysis-es-2030-inputs --output-dir /tmp/heat-analysis-es-2030-run
python scripts/export_spain_2030_vis.py --bundle /tmp/heat-analysis-es-2030-inputs --result-dir /tmp/heat-analysis-es-2030-run --source-root /path/to/HeatAnalysis --output-dir /tmp/heat-analysis-es-2030-vis --web-data /tmp/heat-analysis-es-2030-web.json
```

The first two commands require fresh output directories and will not overwrite
existing runs. The solve needs IPOPT; the export checks against the published
figure-source ZIP. To run only the notebooks, use the bundled data; the full
prepared-data checkout is not required.

# Visualisation

## Result analysis notebooks

These notebooks inspect saved simulation records; they do not run the OPF.
Open them from the repository root with `jupyter notebook vis/`. Each notebook
also locates the repository correctly when its kernel starts in `vis/`.

| Notebook | Input scope |
| --- | --- |
| `1.Spain_grid_example.ipynb` | Full Spanish network and the selected 2022-based scenario records |
| `2.eur_single_analysis.ipynb` | Full single-country batch CSV and per-scenario results |
| `3.eur_multi_analysis.ipynb` | Full single-country and cross-border batch results |
| `4.grid_simu_vis.ipynb` | Full European network, weather and grid-result records |
| `5.ieee_simu_vis.ipynb` | Full IEEE `heatflow_analysis` results |

The bundled European one-hour test and the IEEE `--smoke` run are smaller than
the analyses above. Use the commands in the root README to run those tests;
their outputs are not substitutes for full batch records. Publication figure
reproduction is documented separately in [`figure_source/README.md`](figure_source/README.md).

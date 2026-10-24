# Publication figure source

This directory contains the data-to-figure code used for the main, Extended
Data and Supplementary figures accompanying:

> Heatwave-induced capacity bottlenecks in European electricity grids

Published in *Nature Energy* (2026): https://doi.org/10.1038/s41560-026-02151-1.

Manuscript reference: `NENERGY-24102996D`.

The scripts read the frozen Source Data or Supplementary Data packages in
`data/`. They do not rerun the power-flow simulations. The simulation,
calibration and scenario generation code remains in the repository root.

## Environment

The figure environment uses Python 3.10. Install the repository dependencies
from the repository root with:

```bash
python -m pip install -r requirements.txt
```

Arial was used for the submitted artwork when available. A metrically
compatible sans-serif fallback can alter typography without changing the
numerical content. Cartopy may retrieve its standard Natural Earth map layers
on first use.

## Source Data for main and Extended Data figures

The nine `Source_Data_*.zip` files supplied with the article are included in
`data/`. From the repository root, check the complete set with:

```bash
python vis/figure_source/verify_source_data_reproduction.py
```

Use `--source-data-dir` only to check a different copy of these packages.

The validation extracts each package into a temporary directory and runs the
corresponding plotting command using only the packaged data and this public
code. Generated files are retained only when `--output-dir` is supplied.

The individual plotting entry points are:

| Display item | Entry point |
| --- | --- |
| Main Fig. 1, panel-d source components | `code/plotting/plot_main_figure_01_panel_d_sources.py` |
| Main Fig. 1, panels a-c and complete assembly | `code/plotting/plot_main_figure_01.py` |
| Main Fig. 3 | `code/plotting/plot_main_figure_03.py` |
| Main Fig. 4 | `code/plotting/plot_main_figure_04.py` |
| Main Fig. 5 | `code/plotting/plot_main_figure_05.py` |
| Main Fig. 6 | `code/plotting/plot_main_figure_06.py` |
| Main Fig. 7 | `code/plotting/plot_main_figure_07.py` |
| Main Fig. 8 | `code/plotting/plot_main_figure_08.py` |
| Extended Data Figs. 1-2 | `code/plotting/plot_extended_data_figures.py` |

Main Fig. 1a-c is evaluated from the stated IEEE heat-balance parameters. The
panel-d plotting entry point reconstructs the numerical map, line inset and
colour-bar components from Source Data. Their final layout, and the schematic
in Main Fig. 2, use the editable artwork files supplied separately with the
article.

## Supplementary figures

The ten `Supplementary_Data_*.zip` files are also included in `data/`. Their
`README.txt` files provide package-specific commands. The public copies of
the Supplementary plotting entry points and their compact model definitions
are in `code/`; shared heat-flow functions are in `utils/`.

The source-data exporters used to derive the compact figure tables from the
frozen simulation records are retained in `code/exporters/`. These exporters
require the corresponding simulation records described in the article's Data
Availability statement; the plotting commands above require only the submitted
Source Data packages.

## Licence

These files are distributed under the repository-level licence.

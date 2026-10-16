# Project website

`index.html` contains the page, `assets/css/custom.css` its styles, and
`assets/js/grid-explorer.js` the interactive Spain 2030 map. The map reads
`assets/data/spain_2030.json`; it does not run the OPF solver in the browser.

From the repository root, preview the site with:

```bash
python -m http.server 8000 --directory webpage
```

Open http://localhost:8000. Use an HTTP server rather than opening the HTML
directly so the browser can load the map data.

The three [Spain 2030 notebooks](../vis/demos/README.md) explain the same
example. `scripts/export_spain_2030_vis.py` prepares the companion notebook
and website data from checked example results and publication Source Data;
see its `--help` options before regenerating these files.

GitHub Pages deployment is configured in
[deploy.yml](../.github/workflows/deploy.yml). The `images/` directory also
contains illustrations used by the repository README, so those assets should
be retained even when they are not used on the website itself.

Template attribution is retained in `README.txt` and `LICENSE.txt`; the page
footer records the figure and illustration credits.

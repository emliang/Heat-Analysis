"""Checks for the small Spain 2030 reader examples and web export."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vis/demos/data/spain_2030"


class Spain2030VisTests(unittest.TestCase):
    def test_reader_notebooks_have_valid_code_cells(self):
        for name in ("01_heatwave_weather.ipynb",
                     "02_weather_to_grid.ipynb",
                     "03_solve_and_explore.ipynb"):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / "vis/demos" / name).read_text())
                self.assertEqual(notebook["nbformat"], 4)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), name, "exec")

    def test_web_export_is_same_saved_example(self):
        notebook_data = (DATA / "grid_result.json").read_bytes()
        web_data = (ROOT / "webpage/assets/data/spain_2030.json").read_bytes()
        self.assertEqual(notebook_data, web_data)
        self.assertEqual(hashlib.sha256(web_data).hexdigest(),
                         json.loads((DATA / "provenance.json").read_text())
                         ["web_data_sha256"])
        map_data = json.loads(web_data)
        self.assertEqual(map_data["case"]["snapshot_id"], "ES-2030")
        self.assertEqual(len(map_data["buses"]), 281)
        self.assertEqual(len(map_data["lines"]), 442)
        self.assertEqual(sum(bus["display_marker"] for bus in map_data["buses"]), 1)

    def test_day_and_hourly_demand_match_selected_case(self):
        with xr.open_dataset(DATA / "weather_day.nc") as weather:
            self.assertEqual(weather.sizes["time"], 24)
            self.assertIn(np.datetime64("2030-07-24T13:00:00"),
                          weather.time.values)
        demand = pd.read_csv(DATA / "demand_hourly.csv")
        self.assertEqual(demand.hour.tolist(), list(range(24)))
        self.assertGreater(demand.loc[13, "modelled_demand_mw"], 0)
        lines = pd.read_csv(DATA / "line_inputs.csv")
        self.assertEqual(len(lines), 442)
        self.assertTrue(lines.available_current_a.gt(0).all())

    def test_rerun_agrees_with_published_snapshot(self):
        checks = json.loads((DATA / "provenance.json").read_text())
        self.assertLess(checks["max_line_temperature_difference_c"], 0.001)
        self.assertLess(checks["max_load_shedding_difference_mw"], 0.001)


if __name__ == "__main__":
    unittest.main()

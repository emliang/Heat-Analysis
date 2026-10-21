"""Prepared European example checks that do not run the OPF solver."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from data_config import get_local_data_path, get_local_model_path, get_pypsa_path
from scripts.run_european_example import check_bundle, example_settings


class EuropeanExampleTests(unittest.TestCase):
    def test_repository_bundle_is_complete(self):
        bundle = (Path(__file__).resolve().parents[1] / "data/examples" /
                  "es_2030-07-24_13_2024-0")
        manifest = check_bundle(bundle)
        self.assertEqual(manifest["case"]["country"], "ES")
        self.assertEqual(manifest["case"]["timestamp"], "2030-07-24T13:00:00")
        self.assertEqual(manifest["network_components"]["stores"], 0)
        self.assertEqual(len(manifest["files"]), 11)

    def make_bundle(self, directory):
        bundle = Path(directory)
        case = {
            "analysis_mode": "td_seg_derate_iter_2",
            "country": "ES",
            "timestamp": "2030-07-24T13:00:00",
            "historical_year": 2024,
            "historical_event_index": 0,
            "network_ratio": 75,
            "load_growth_rate": 1.01,
            "storage_ratio": 0.8,
            "max_temperature": 90,
        }
        paths = [
            Path("data/EU/networks/base_s_75_elec.nc"),
            Path("data/EU/regions_onshore_base_s_75.geojson"),
            Path("data/EU/load_ratio_base_s_75.csv"),
            Path("models/ES/simu_data/") /
            "future_weather_data_based_on_historical_hot_event_2024_0.nc",
            Path("models/ES/simu_data/") /
            "future_demand_data_based_on_historical_hot_event_2024_0.npy",
            Path("models/demand_curve/ES/ES_[2015, 2024]_demand_curve.npy"),
        ]
        paths.extend(
            Path("models/ES/weather") / f"2024_0_{tech}_profile_s_75.nc"
            for tech in ("solar", "onwind", "offwind-ac", "offwind-dc")
        )
        paths.append(Path("data/EU/networks/historical_p_max_pu_s_75.nc"))
        files = []
        for path in paths:
            target = bundle / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"test input")
            files.append({
                "path": str(path),
                "sha256": hashlib.sha256(b"test input").hexdigest(),
            })
        (bundle / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "case": case,
            "files": files,
        }))
        return bundle, paths[3]

    def test_bundle_checks_selected_hour(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, _ = self.make_bundle(directory)
            hour = np.datetime64("2030-07-24T13:00:00", "ns")
            dataset = xr.Dataset(
                {"p_max_pu": (("month", "hour", "generator"), [[[0.5]]])},
                coords={"time": [hour], "month": [7], "hour": [13],
                        "generator": ["test-generator"]})
            with patch("scripts.run_european_example.xr.open_dataset",
                       return_value=dataset):
                self.assertEqual(check_bundle(bundle)["case"]["country"],
                                 "ES")

    def test_bundle_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, path = self.make_bundle(directory)
            (bundle / path).write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                check_bundle(bundle)

    def test_bundle_rejects_missing_required_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, _ = self.make_bundle(directory)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"].pop()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "required inputs"):
                check_bundle(bundle)

    def test_bundle_rejects_other_analysis_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, _ = self.make_bundle(directory)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["case"]["analysis_mode"] = "base"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "supports the prepared"):
                check_bundle(bundle)

    def test_settings_match_selected_case(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, _ = self.make_bundle(directory)
            case = json.loads((bundle / "manifest.json").read_text())["case"]
            args, conductor, weather = example_settings(case)
            self.assertEqual(args["load_growth_rate"], 1.01)
            self.assertEqual(args["storage_ratio"], 0.8)
            self.assertEqual(args["phase_factor"], 3.0)
            self.assertEqual(conductor["max_temperature"], 90)
            self.assertEqual(weather["wind_height"], 10)

    def test_prepared_run_does_not_need_pypsa_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent-pypsa"
            with patch.dict(os.environ, {"HEATANALYSIS_PYPSA": str(missing)}):
                self.assertEqual(get_pypsa_path(required=False), str(missing))
                with self.assertRaises(FileNotFoundError):
                    get_pypsa_path()

    def test_external_data_and_model_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            models = Path(directory) / "models"
            data.mkdir()
            models.mkdir()
            with patch.dict(os.environ, {
                    "HEATANALYSIS_LOCAL_DATA": str(data),
                    "HEATANALYSIS_MODELS": str(models)}):
                self.assertEqual(get_local_data_path(), str(data))
                self.assertEqual(get_local_model_path(), str(models))


if __name__ == "__main__":
    unittest.main()

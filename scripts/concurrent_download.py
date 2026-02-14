#!/usr/bin/env python3
"""Concurrent ERA5 weather-data downloader via the CDS API.

This script downloads ERA5 reanalysis data from the Copernicus Climate
Data Store (CDS) using a thread-pool to run multiple requests in
parallel, dramatically reducing total wall-clock time compared to
sequential downloading.

Usage
-----
    python test_concurrent_download.py

Adjust ``DownloadConfig`` attributes (years, variables, bounding box,
worker count) before running.
"""

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cdsapi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Full catalogue of ERA5 variable groups used in this project.
# Un-comment the groups you need; the downloader iterates over each group
# independently so they can be downloaded in separate files.
DEFAULT_VARIABLE_GROUPS: List[List[str]] = [
    ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    ["100m_u_component_of_wind", "100m_v_component_of_wind"],
    ["2m_dewpoint_temperature", "2m_temperature"],
    ["toa_incident_solar_radiation",
     "surface_solar_radiation_downward_clear_sky"],
    ["surface_solar_radiation_downwards",
     "clear_sky_direct_solar_radiation_at_surface"],
    ["soil_temperature_level_4", "runoff"],
    ["forecast_albedo", "forecast_surface_roughness"],
    ["surface_runoff"],
]

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


@dataclass
class DownloadConfig:
    """Configuration for a batch of CDS API downloads.

    Parameters
    ----------
    dataset : str
        CDS dataset identifier.
    year_list : list of int
        Calendar years to retrieve.
    variable_groups : list of list of str
        Each inner list is a group of ERA5 variable names that will be
        downloaded together into one file.
    area : list of float
        Bounding box ``[north, west, south, east]`` in degrees.
    max_workers : int
        Maximum number of concurrent download threads.
    """

    dataset: str = "reanalysis-era5-single-levels"
    year_list: List[int] = field(
        default_factory=lambda: list(range(2015, 2025)),
    )
    variable_groups: List[List[str]] = field(
        default_factory=lambda: DEFAULT_VARIABLE_GROUPS,
    )
    area: List[float] = field(
        default_factory=lambda: [65, -12, 33, 45],  # Europe
    )
    max_workers: int = 4


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class ConcurrentCDSDownloader:
    """Thread-pool wrapper around the CDS API client.

    Parameters
    ----------
    config : DownloadConfig
        Download configuration (years, variables, region, workers).
    """

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # -- request helpers ----------------------------------------------------

    def _build_request(self, year: int,
                       variables: List[str]) -> Dict[str, Any]:
        """Build a CDS API request dictionary.

        Parameters
        ----------
        year : int
            Calendar year.
        variables : list of str
            ERA5 variable names to include.

        Returns
        -------
        dict
            Request payload for ``cdsapi.Client.retrieve()``.
        """
        return {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [str(year)],
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "data_format": "netcdf",
            "download_format": "zip",
            "area": self.config.area,
        }

    # -- single download ----------------------------------------------------

    def _download_one(self, year: int,
                      variables: List[str]) -> Dict[str, Any]:
        """Execute a single CDS retrieve-and-download.

        Parameters
        ----------
        year : int
            Calendar year.
        variables : list of str
            ERA5 variable names.

        Returns
        -------
        dict
            Status record with keys ``success``, ``year``, ``variable``,
            ``duration``, and ``message``.
        """
        t0 = time.time()
        try:
            client = cdsapi.Client()
            request = self._build_request(year, variables)
            result = client.retrieve(self.config.dataset, request)
            result.download()

            duration = time.time() - t0
            status = {
                "success": True,
                "year": year,
                "variable": variables,
                "duration": duration,
                "message": (f"Downloaded year {year}, "
                            f"variables {variables}"),
            }
            with self._lock:
                self.results.append(status)
                logger.info("OK  %s (%.1f s)", status["message"], duration)
            return status

        except Exception as exc:
            duration = time.time() - t0
            status = {
                "success": False,
                "year": year,
                "variable": variables,
                "duration": duration,
                "message": (f"FAILED year {year}, "
                            f"variables {variables}: {exc}"),
            }
            with self._lock:
                self.results.append(status)
                logger.error("ERR %s (%.1f s)", status["message"], duration)
            return status

    # -- batch download -----------------------------------------------------

    def download_all(self) -> List[Dict[str, Any]]:
        """Download every (year, variable-group) combination concurrently.

        Returns
        -------
        list of dict
            One status record per request.
        """
        tasks = [
            (year, var_group)
            for year in self.config.year_list
            for var_group in self.config.variable_groups
        ]

        logger.info(
            "Starting %d requests with %d workers ...",
            len(tasks), self.config.max_workers,
        )
        t0 = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers,
        ) as pool:
            future_map = {
                pool.submit(self._download_one, year, var_group): (year, var_group)
                for year, var_group in tasks
            }

            done = 0
            for future in concurrent.futures.as_completed(future_map):
                info = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Unhandled exception for %s: %s", info, exc)
                done += 1
                logger.info("Progress: %d / %d (%.0f%%)",
                            done, len(tasks), 100 * done / len(tasks))

        elapsed = time.time() - t0
        ok = sum(1 for r in self.results if r["success"])
        fail = len(self.results) - ok
        logger.info(
            "Finished in %.1f s — %d succeeded, %d failed", elapsed, ok, fail,
        )
        return self.results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a concurrent ERA5 download with default configuration."""
    config = DownloadConfig(max_workers=10)
    downloader = ConcurrentCDSDownloader(config)
    downloader.download_all()


if __name__ == "__main__":
    main()

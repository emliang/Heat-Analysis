"""Historical mean availability derived from the PyPSA network.

Used where matching future weather-driven renewable profiles are unavailable.
Values are available output limits, not dispatched generation.
"""

from functools import lru_cache

import pandas as pd
import xarray as xr


@lru_cache(maxsize=8)
def load_historical_availability(path):
    """Read the prepared monthly-hourly availability once per process."""
    with xr.open_dataset(path) as dataset:
        return dataset['p_max_pu'].load()


def historical_availability(path, date):
    date = pd.Timestamp(date)
    try:
        values = load_historical_availability(str(path)).sel(
            month=date.month, hour=date.hour).to_series()
    except KeyError as exc:
        raise ValueError(
            f'Historical availability has no month={date.month}, hour={date.hour}: {path}') from exc
    return values

"""Historical mean availability derived from the PyPSA network.

Used where matching future weather-driven renewable profiles are unavailable.
Values are available output limits, not dispatched generation.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import pandas as pd
import xarray as xr


def export_historical_availability(network, output, source):
    """Export PyPSA historical monthly-hourly means, preserving the source series."""
    profiles = network.generators_t['p_max_pu']
    times = pd.DatetimeIndex(profiles.index)
    means = profiles.groupby([times.month, times.hour]).mean()
    means.index.names = ['month', 'hour']
    means.columns.name = 'generator'
    dataset = means.stack(dropna=False).rename('p_max_pu').to_xarray().to_dataset()
    dataset.p_max_pu.attrs.update(
        units='p.u.', description='Historical mean available generation per unit of nominal capacity')
    source = Path(source)
    dataset.attrs.update(
        source_network=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        averaging='Arithmetic mean over matching calendar month and hour; missing values excluded',
        purpose='Fallback when future weather-driven renewable profiles are unavailable',
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    return output


@lru_cache(maxsize=8)
def load_historical_availability(path):
    """Read once per process; regenerate the file when its network changes."""
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

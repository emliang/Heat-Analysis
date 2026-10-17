"""Export the saved Spain 2030 weather hour for the website; no simulation."""

import json
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'vis/demos/data/spain_2030/weather_day.nc'
    snapshot = json.loads((root / 'webpage/assets/data/spain_2030.json').read_text())
    timestamp = '2030-07-24T13:00:00'
    cells = []
    with xr.open_dataset(source) as ds:
        hour = ds.sel(time=timestamp)
        # The publication map stores plotting coordinates offset by half a cell.
        dx = float(ds.x.diff('x')[0]) / 2
        dy = float(ds.y.diff('y')[0]) / 2
        for cell in snapshot['weather']:
            point = hour.sel(x=cell['lon'] + dx, y=cell['lat'] + dy)
            temperature = float(point.temperature) - 273.15
            wind = float(point.wnd10m)
            solar = float(point.influx)
            if not np.isfinite([temperature, wind, solar]).all():
                raise ValueError('Non-finite weather value')
            if abs(temperature - cell['temperature_c']) > 1e-5:
                raise ValueError('Weather temperature differs from the existing map')
            cells.append({'lon': cell['lon'], 'lat': cell['lat'],
                          'source_lon': float(point.x), 'source_lat': float(point.y),
                          'temperature_c': cell['temperature_c'],
                          'wind_speed_m_s': wind, 'solar_irradiance_w_m2': solar})
    target = root / 'webpage/assets/data/spain_2030_weather.json'
    target.write_text(json.dumps({'timestamp': timestamp,
                                  'source': 'vis/demos/data/spain_2030/weather_day.nc',
                                  'cells': cells}, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'Exported {len(cells)} cells at {timestamp}; all temperatures match the existing map')


if __name__ == '__main__':
    main()

"""Export the SI Spain configuration for the website, without running OPF."""

import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'vis/figure_source/data/Supplementary_Data_01_Figs01-08.zip'
    prefix = 'data/Supplementary_Fig_01_spain_grid_configuration/data/'
    with ZipFile(source) as archive:
        def rows(name):
            return list(csv.DictReader(io.StringIO(
                archive.read(prefix + name).decode())))

        buses = rows('spain_grid_buses.csv')
        capacity = rows('spain_grid_capacity_by_bus.csv')
        lines = rows('spain_grid_ac_lines.csv')
        links = rows('spain_grid_dc_links.csv')
        rules = json.loads(archive.read(prefix + 'spain_grid_plot_rules.json'))

    snapshot = json.loads((root / 'webpage/assets/data/spain_2030.json').read_text())
    assert {b['bus_id'] for b in buses} == {b['id'] for b in snapshot['buses']}
    assert {l['line_id'] for l in lines} == {l['id'] for l in snapshot['lines']}
    by_id = {b['id']: b for b in snapshot['buses']}
    for bus in buses:
        assert abs(float(bus['x']) - by_id[bus['bus_id']]['lon']) < 1e-7
        assert abs(float(bus['y']) - by_id[bus['bus_id']]['lat']) < 1e-7
    categories = rules['category_order']
    payload = {
        'source': source.name,
        'categories': categories,
        'palette': rules['category_palette'],
        'capacity_by_bus': {r['bus_id']: {c: float(r[c]) for c in categories}
                            for r in capacity},
        'ac_capacity_mva': {r['line_id']: float(r['s_nom_mva']) for r in lines},
        'dc_links': [{'id': r['link_id'], 'bus0': r['bus0'], 'bus1': r['bus1'],
                      'capacity_mw': float(r['p_nom_mw'])} for r in links],
    }
    target = root / 'webpage/assets/data/spain_grid_configuration.json'
    target.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    print(f'Exported {len(buses)} buses, {len(lines)} AC lines, {len(links)} DC links')


if __name__ == '__main__':
    main()

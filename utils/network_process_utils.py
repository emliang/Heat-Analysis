"""Network loading, filtering, segmentation, and PyPSA → PyPower conversion.

This module provides the complete pipeline for preparing a power-system
model for ACOPF analysis:

1. **Network loading** — read a PyPSA-Eur network, filter by country,
   attach population / load ratios, and apply line segmentation.
2. **Regional data helpers** — load GeoJSON region geometries, filter
   country / offshore shapes for exclusion zones.
3. **Generator merging** — combine generators at the same bus with
   identical marginal cost into a single equivalent unit.
4. **Line segmentation** — divide transmission lines into sub-segments
   that align with the weather-data grid (0.25° ERA5 resolution).
5. **Graph utilities** — connectivity checks and N-1 contingency
   identification via minimum spanning tree.
6. **PyPSA → PyPower conversion** — ``PypsaPypower`` class that builds
   a PyPower ``ppc`` dict (buses, generators, branches, DC lines,
   storage) from a PyPSA ``Network``.

Section index
-------------
1. Regional Data Helpers
2. Load-Ratio Assignment
3. Network Loading
4. Generator Merging
5. Line Segmentation
6. Timezone & Graph Utilities
7. PyPower Cleanup
8. PyPSA → PyPower Converter
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import pytz
from pypower import idx_brch
from pytz import country_timezones

from data_config import *  # noqa: F403
from utils.country_network_filter import filter_network_by_countries


# ===========================================================================
# 1. Regional Data Helpers
# ===========================================================================

def load_and_filter_regional_data(network, regions_path, verbose=False):
    """Load a GeoJSON regions file and keep only buses in *network*.

    Parameters
    ----------
    network : pypsa.Network
        Filtered network whose bus index defines the retained regions.
    regions_path : str
        Path to a ``regions_onshore_*.geojson`` file.
    verbose : bool

    Returns
    -------
    gpd.GeoDataFrame
        Regions whose index matches ``network.buses.index``.
    """
    regions = gpd.GeoDataFrame()
    if os.path.exists(regions_path):
        if verbose:
            print(f"Loading EU regions from: {regions_path}")
        regions = gpd.read_file(regions_path)
        regions = regions.set_index("name").rename_axis("bus")
        if not regions.empty:
            regions = regions[regions.index.isin(network.buses.index)]
            if verbose:
                print(f"Filtered regions: {len(regions)} regions "
                      f"for selected countries")
    return regions


def filter_and_save_shapes(country_list, regions, country_shapes_path,
                           offshore_shapes_path, output_dir=None,
                           verbose=False):
    """Filter country / offshore shape files and save as GeoJSON.

    Parameters
    ----------
    country_list : str or list[str]
        Country codes to retain.
    regions : gpd.GeoDataFrame
        Filtered regions (used for context only; filtering is by country).
    country_shapes_path, offshore_shapes_path : str
        Paths to the full-EU shape files.
    output_dir : str or None
        Directory for filtered outputs.
    verbose : bool

    Returns
    -------
    tuple[str, str]
        ``(filtered_country_path, filtered_offshore_path)``.
    """
    if isinstance(country_list, str):
        country_list = [country_list]

    # Country shapes
    filtered_country_path = os.path.join(
        output_dir, "filtered_country_shapes.geojson")
    country_shapes = gpd.read_file(country_shapes_path)
    filtered_country = country_shapes[
        country_shapes['name'].isin(country_list)]
    filtered_country.to_file(filtered_country_path, driver='GeoJSON')

    # Offshore shapes
    filtered_offshore_path = os.path.join(
        output_dir, "filtered_offshore_shapes.geojson")
    offshore_shapes = gpd.read_file(offshore_shapes_path)
    filtered_offshore = offshore_shapes[
        offshore_shapes['name'].isin(country_list)]
    filtered_offshore.to_file(filtered_offshore_path, driver='GeoJSON')

    return filtered_country_path, filtered_offshore_path


# ===========================================================================
# 2. Load-Ratio Assignment
# ===========================================================================

def load_and_apply_load_ratios(country_list, network, regions,
                               load_ratio_csv, verbose=False):
    """Read GDP/population load ratios and assign them to network buses.

    Parameters
    ----------
    country_list : list[str]
        Country codes (for logging only).
    network : pypsa.Network
        Network whose ``buses`` DataFrame will be updated in-place with
        ``load_ratio``, ``pop_ratio``, and ``regions`` columns.
    regions : gpd.GeoDataFrame
        Filtered regions matching *network* buses.
    load_ratio_csv : str
        Path to CSV with columns ``name``, ``load_factor_gdp_pop``,
        ``load_factor_pop_only``.
    verbose : bool
    """
    pop_ratio = np.array([])

    # Initialise bus attributes
    network.buses['load_ratio'] = 0.0
    network.buses['pop_ratio'] = 0.0
    network.buses['regions'] = 0.0

    if not os.path.exists(load_ratio_csv):
        raise ValueError(
            f"No file found for load ratio: {load_ratio_csv}")

    if verbose:
        print(f"Loading EU population ratios from: {load_ratio_csv}")
    load_ratio_df = pd.read_csv(load_ratio_csv)

    if regions.empty or 'name' not in load_ratio_df.columns:
        raise ValueError(
            f"No load ratio data found for regions: {regions.index}")

    filtered = load_ratio_df[load_ratio_df['name'].isin(regions.index)]
    load_ratio = filtered['load_factor_gdp_pop'].values
    pop_ratio = filtered['load_factor_pop_only'].values

    # Map ratios to buses
    if not regions.empty and len(pop_ratio) > 0:
        common_buses = network.buses.index.intersection(regions.index)
        if verbose:
            print(f"Common buses between network and regions: "
                  f"{len(common_buses)}")
        if len(common_buses) > 0:
            for i, bus in enumerate(regions.index):
                if bus in common_buses and i < len(pop_ratio):
                    network.buses.loc[bus, 'pop_ratio'] = pop_ratio[i]
                    network.buses.loc[bus, 'load_ratio'] = load_ratio[i]
                    network.buses.loc[bus, 'regions'] = 1.0
        elif verbose:
            print("Warning: No common buses found between network "
                  "and regions")


# ===========================================================================
# 3. Network Loading
# ===========================================================================

def load_network_EU(country_list, ratio=75, verbose=False):
    """Load the PyPSA-Eur network and filter for selected countries.

    The function loads the full EU network, filters buses / generators /
    lines / links to the requested countries, attaches population and
    load ratios, and applies line segmentation.

    Parameters
    ----------
    country_list : list[str] or str
        Country codes (e.g. ``['ES', 'FR']``).
    ratio : int
        Clustering ratio for the network file (25, 50, 75, 100).
    verbose : bool

    Returns
    -------
    network : pypsa.Network
        Filtered and segmented network.
    regions : gpd.GeoDataFrame
        Matching onshore regions.

    Examples
    --------
    >>> network, regions = load_network_EU(['ES', 'FR'], ratio=75)
    """
    if not isinstance(country_list, list):
        country_list = [country_list]

    network_filename = LOCAL_DATA + f"/EU/networks/base_s_{ratio}_elec.nc"
    regions_path = LOCAL_DATA + f"/EU/regions_onshore_base_s_{ratio}.geojson"
    load_ratio_csv = LOCAL_DATA + f"/EU/load_ratio_base_s_{ratio}.csv"

    print(f"Loading EU network from: {network_filename}")
    network = pypsa.Network(network_filename)

    # Filter for selected countries
    if verbose:
        print(f"Filtering network for countries: {country_list}")
    sub_network = filter_network_by_countries(
        network, country_list, keep_interconnections=True, verbose=False)

    # Regional data
    regions = load_and_filter_regional_data(sub_network, regions_path)

    # Population / load ratios
    load_and_apply_load_ratios(
        country_list, sub_network, regions, load_ratio_csv)

    # Line segmentation
    sub_network.resolution = ratio
    sub_network = divide_segments(sub_network)

    if verbose:
        print(f"Multi-country network loaded successfully:")
        print(f"  Countries: {country_list}")
        print(f"  Buses: {len(sub_network.buses)}")
        print(f"  Generators: {len(sub_network.generators)}")
        print(f"  Lines: {len(sub_network.lines)}")
        print(f"  Regions: {len(regions)}")
        print(f"  Pop ratio sum: "
              f"{sub_network.buses['pop_ratio'].sum():.4f}")
        print(f"  Load ratio sum: "
              f"{sub_network.buses['load_ratio'].sum():.4f}")

    return sub_network, regions


def load_demand_profile(selected_country, year_list=None):
    """Load a pre-calibrated demand model for *selected_country*.

    Parameters
    ----------
    selected_country : str
        Two-letter country code.
    year_list : list[int] or None
        Calibration years.  Defaults to ``[2015, 2020]`` for GB,
        ``[2015, 2024]`` otherwise.

    Returns
    -------
    dict
        Demand-model parameter dictionary.
    """
    if year_list is None:
        year_list = ([2015, 2020] if selected_country in ('GB',)
                     else [2015, 2024])
    path = (MODELS + f'/demand_curve/{selected_country}/'
            f'{selected_country}_{year_list}_demand_curve.npy')
    return np.load(path, allow_pickle=True).item()


def load_network(country_code):
    """Load a single-country network using legacy per-country file layout.

    Parameters
    ----------
    country_code : str
        Two-letter country code.

    Returns
    -------
    network : pypsa.Network
    regions : gpd.GeoDataFrame
    """
    ratio_map = {'GB': 75, 'FR': 75, 'DE': 100, 'IT': 100,
                 'ES': 50, 'BE': 100}
    ratio = ratio_map.get(country_code, 75)

    if ratio < 100:
        network_filename = RESOURCES + f"{country_code}/networks/elec_s_{ratio}.nc"
        regions_path = RESOURCES + f"{country_code}/regions_onshore_elec_s_{ratio}.geojson"
        pop_ratio_csv = RESOURCES + f"{country_code}/pop_layout_elec_s_{ratio}.csv"
    else:
        network_filename = RESOURCES + f"{country_code}/networks/elec_s.nc"
        regions_path = RESOURCES + f"{country_code}/regions_onshore_elec_s.geojson"
        pop_ratio_csv = RESOURCES + f"{country_code}/pop_layout_elec_s.csv"

    network = pypsa.Network(network_filename)
    regions = gpd.read_file(regions_path)
    regions = regions.set_index("name").rename_axis("bus")
    pop_ratio_df = pd.read_csv(pop_ratio_csv)
    pop_ratio = pop_ratio_df['fraction'].values

    # Regional population ratio
    network.buses['pop_ratio'] = 0.0
    network.buses['regions'] = 0.0
    network.buses.loc[regions.index, 'pop_ratio'] = (
        pop_ratio / pop_ratio.sum())
    network.buses.loc[regions.index, 'regions'] = 1.0

    # Regional load ratio from time-series
    network.buses['load_ratio'] = 0.0
    load_ratio = network.loads_t['p_set'].iloc[1]
    network.buses.loc[network.loads_t['p_set'].columns, 'load_ratio'] = (
        load_ratio / load_ratio.sum())

    network.resolution = ratio
    network = divide_segments(network)

    return network, regions


# ===========================================================================
# 4. Generator Merging
# ===========================================================================

def merge_generators(network, verbose=False):
    """Merge generators at the same bus with identical marginal cost.

    Capacities (``p_nom``, ``p_nom_min``, ``p_nom_max``, ``p_nom_opt``)
    are summed; all other static attributes are taken from the first
    generator in each merge group.

    Parameters
    ----------
    network : pypsa.Network
    verbose : bool

    Returns
    -------
    pypsa.Network
        The *same* network with merged generators.

    Examples
    --------
    >>> network, regions = load_network_EU(['FR', 'IT'], 50)
    >>> network = merge_generators(network, verbose=True)
    """
    if network.generators.empty:
        if verbose:
            print("No generators to merge")
        return network

    generators = network.generators.copy()

    # Group key: bus + marginal_cost (rounded to avoid float issues)
    generators['merge_key'] = (
        generators['bus'].astype(str) + '_'
        + generators['marginal_cost'].round(6).astype(str)
    )

    merge_groups = generators.groupby('merge_key')
    groups_to_merge = {
        key: group for key, group in merge_groups if len(group) > 1}

    if not groups_to_merge:
        if verbose:
            print("No generators to merge (no duplicates found)")
        return network

    if verbose:
        n_before = sum(len(g) for g in groups_to_merge.values())
        print(f"Found {len(groups_to_merge)} merge groups "
              f"({n_before} generators → {len(groups_to_merge)})")

    generators_to_remove = []

    for _, group in groups_to_merge.items():
        indices = group.index.tolist()
        merged_name = indices[0]

        # Sum capacity columns
        generators.loc[merged_name, 'p_nom'] = group['p_nom'].sum()
        for attr in ('p_nom_min', 'p_nom_max', 'p_nom_opt'):
            if attr in generators.columns:
                generators.loc[merged_name, attr] = group[attr].sum()
        if 'p_nom_extendable' in group.columns:
            generators.loc[merged_name, 'p_nom_extendable'] = (
                group['p_nom_extendable'].any())

        # Mark duplicates for removal (keep first)
        generators_to_remove.extend(indices[1:])

    remaining = generators[
        ~generators.index.isin(generators_to_remove)].copy()
    remaining.drop(columns=['merge_key'], inplace=True, errors='ignore')
    network.generators = remaining

    if verbose:
        print(f"Generator merging complete. "
              f"Network now has {len(network.generators)} generators")

    return network


# ===========================================================================
# 5. Line Segmentation
# ===========================================================================

def calculate_intersections(x1, y1, x2, y2, resolution=0.25):
    """Find where a line segment crosses the weather-data grid.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Endpoint coordinates (lon, lat).
    resolution : float
        Grid spacing (default 0.25° for ERA5).

    Returns
    -------
    list[tuple[float, float]]
        Sorted intersection points from ``(x1, y1)`` to ``(x2, y2)``.
    """
    offset = resolution / 2
    points = [(x1, y1), (x2, y2)]

    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    # Vertical grid lines
    x_start = np.floor((min_x - offset) / resolution) * resolution + offset
    x_end = np.ceil((max_x - offset) / resolution) * resolution + offset
    for x in np.arange(x_start, x_end + resolution, resolution):
        if x1 != x2:
            y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            if min_y <= y <= max_y and min_x <= x <= max_x:
                points.append((x, y))

    # Horizontal grid lines
    y_start = np.floor((min_y - offset) / resolution) * resolution + offset
    y_end = np.ceil((max_y - offset) / resolution) * resolution + offset
    for y in np.arange(y_start, y_end + resolution, resolution):
        if y1 != y2:
            x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if min_y <= y <= max_y and min_x <= x <= max_x:
                points.append((x, y))

    # Deduplicate and sort by distance from start
    points = list(set(points))
    points.sort(key=lambda p: np.hypot(p[0] - x1, p[1] - y1))
    return points


def visualize_intersections(points, segment_info, resolution=0.25):
    """Debug plot showing a line's intersection with the weather grid."""
    x_vals, y_vals = zip(*points)
    plt.figure(figsize=(10, 10))
    plt.plot(x_vals, y_vals, 'b-', label='Line Segment')
    plt.scatter(x_vals, y_vals, color='red', s=100, zorder=5,
                label='Intersection Points')

    grid_x_s = np.floor(min(x_vals) / resolution) * resolution + 0.125
    grid_x_e = np.ceil(max(x_vals) / resolution) * resolution + 0.125
    grid_y_s = np.floor(min(y_vals) / resolution) * resolution + 0.125
    grid_y_e = np.ceil(max(y_vals) / resolution) * resolution + 0.125

    for x in np.arange(grid_x_s, grid_x_e, resolution):
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in np.arange(grid_y_s, grid_y_e, resolution):
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    segment_info = np.array(segment_info)
    plt.scatter(segment_info[:, 0], segment_info[:, 1], s=15, zorder=10)
    plt.grid(False)
    plt.title('Intersection of Line with Grid')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()


def calculate_segment_info(points, spacing, max_num_seg):
    """Compute per-segment grid-cell centre and length proportion.

    Parameters
    ----------
    points : list[tuple]
        Sorted intersection points.
    spacing : float
        Grid cell spacing.
    max_num_seg : int
        Maximum segment count (shorter lines are zero-padded).

    Returns
    -------
    list[list[float]]
        ``[[centre_x, centre_y, proportion], ...]`` padded to
        *max_num_seg* entries.
    """
    (x1, y1), (x2, y2) = points[0], points[-1]
    total_length = np.hypot(x2 - x1, y2 - y1)
    segment_info = []

    for i in range(len(points) - 1):
        px1, py1 = points[i]
        px2, py2 = points[i + 1]
        seg_len = np.hypot(px2 - px1, py2 - py1)
        proportion = seg_len / total_length
        cx = np.round((px1 + px2) / (2 * spacing)) * spacing
        cy = np.round((py1 + py2) / (2 * spacing)) * spacing
        segment_info.append([cx, cy, proportion])

    # Pad with zero-proportion entries at the last centre
    for _ in range(len(points) - 1, max_num_seg):
        segment_info.append([cx, cy, 0])

    return segment_info


def divide_segments(network, seg_len=25, grid_space=0.25):
    """Assign weather-grid segments to every transmission line.

    Each line is intersected with the ERA5 grid; the resulting segments
    record which grid cell they fall in and what fraction of the total
    line length they represent.  Results are stored in
    ``network.segments`` as a 3-D array ``(n_lines, max_segs, 3)``.

    Parameters
    ----------
    network : pypsa.Network
        Must have ``lines`` with ``bus0``, ``bus1``, ``length``.
    seg_len : int
        Nominal segment length [km] used to size the array.
    grid_space : float
        Grid spacing [°].

    Returns
    -------
    pypsa.Network
        The same *network* with ``segments`` attached.
    """
    max_segs = int(max(network.lines['length']) / seg_len + 1) * 2
    network.segments = np.zeros(
        [network.lines.shape[0], max_segs, 3])

    for i, line_id in enumerate(network.lines.index):
        line = network.lines.loc[line_id]
        fx, fy = network.buses.loc[line['bus0'], ['x', 'y']]
        tx, ty = network.buses.loc[line['bus1'], ['x', 'y']]
        intersections = calculate_intersections(
            fx, fy, tx, ty, grid_space)
        seg_info = calculate_segment_info(
            intersections, grid_space, max_segs)
        network.lines.loc[line_id, 'num_seg'] = len(seg_info)
        network.segments[i, :len(seg_info)] = seg_info

    return network


# ===========================================================================
# 6. Timezone & Graph Utilities
# ===========================================================================

def get_timezone_difference(country_code, utc_datetime):
    """Return the UTC offset (hours) for *country_code* at *utc_datetime*.

    Parameters
    ----------
    country_code : str
        Two-letter ISO 3166-1 code.
    utc_datetime : datetime
        Reference time (DST-aware offset is returned).

    Returns
    -------
    int
        Offset in hours (positive = east of UTC).
    """
    if country_code not in country_timezones:
        raise ValueError("Invalid country code")

    if (utc_datetime.tzinfo is None
            or utc_datetime.tzinfo.utcoffset(utc_datetime) is None):
        utc_datetime = pytz.utc.localize(utc_datetime)

    tz = pytz.timezone(country_timezones[country_code][0])
    local_dt = utc_datetime.astimezone(tz)
    offset = local_dt.utcoffset().total_seconds() / 3600
    return int(offset)


def check_connectivity(ppc):
    """Check whether the PyPower network graph is connected.

    First tests AC branches only; if disconnected, adds DC lines and
    re-tests.

    Parameters
    ----------
    ppc : dict
        PyPower case dictionary.

    Returns
    -------
    bool
    """
    G = nx.Graph()
    for i, bus in enumerate(ppc['bus']):
        G.add_node(i, pos=(bus[-2], bus[-1]))
    G.add_weighted_edges_from([
        (int(b[0]), int(b[1]), 1)
        for b in ppc['branch'] if b[idx_brch.BR_STATUS] == 1])
    if nx.is_connected(G):
        print('Network Connected without DCLine')
        return True
    G.add_weighted_edges_from([
        (int(d[0]), int(d[1]), 1)
        for d in ppc['dcline'] if d[2] == 1])
    if nx.is_connected(G):
        print('Network Connected with DCLine')
        return True
    print('Network Dis-Connected')
    return False


# Keep old name as alias for backward compatibility
check_connectiveness = check_connectivity


def identify_contingency(ppc):
    """Identify removable branches via minimum spanning tree.

    Branches **not** in the MST can be tripped without disconnecting
    the network, making them candidates for N-1 contingency analysis.

    Parameters
    ----------
    ppc : dict
        PyPower case dictionary.

    Returns
    -------
    list[tuple[int, int]]
        Branch ``(from_bus, to_bus)`` pairs that can be removed.
    """
    G = nx.Graph()
    G.add_weighted_edges_from([
        (int(b[0]), int(b[1]), 1)
        for b in ppc['branch'] if b[idx_brch.BR_STATUS] == 1])
    mst = nx.minimum_spanning_tree(G)
    return list(set(G.edges()) - set(mst.edges()))


# ===========================================================================
# 7. PyPower Cleanup
# ===========================================================================

def remove_isolated_elements(ppc):
    """Remove isolated buses and re-index a PyPower case dict.

    Buses that are not an endpoint of any active branch or DC line are
    removed together with their generators.  All bus indices are then
    re-mapped to a contiguous 0-based range.

    Parameters
    ----------
    ppc : dict
        PyPower case dictionary (modified in-place **and** returned).

    Returns
    -------
    dict
    """
    # Step 1: Identify connected buses
    connected = set()
    for branch in ppc['branch']:
        if branch[idx_brch.BR_STATUS] == 1:
            connected.add(branch[0])
            connected.add(branch[1])
    for dcline in ppc['dcline']:
        if dcline[2] == 1:
            connected.add(dcline[0])
            connected.add(dcline[1])

    # Step 2: Filter buses
    new_bus_idx = [i for i, bus in enumerate(ppc['bus'])
                   if bus[0] in connected]
    ppc['bus'] = ppc['bus'][new_bus_idx]
    ppc['pop_ratio'] = ppc['pop_ratio'][new_bus_idx]

    # Step 3: Filter generators
    new_gen_idx = [i for i, gen in enumerate(ppc['gen'])
                   if gen[0] in connected]
    ppc['gen'] = ppc['gen'][new_gen_idx]
    ppc['gencost'] = ppc['gencost'][new_gen_idx]

    # Step 4: Filter branches
    new_br_idx = [i for i, br in enumerate(ppc['branch'])
                  if br[0] in connected and br[1] in connected]
    ppc['branch'] = ppc['branch'][new_br_idx]

    # Step 5: Filter DC lines
    new_dc_idx = [i for i, dc in enumerate(ppc['dcline'])
                  if dc[0] in connected and dc[1] in connected]
    ppc['dcline'] = ppc['dcline'][new_dc_idx]
    ppc['dclinecost'] = ppc['dclinecost'][new_dc_idx]

    # Step 6: Re-index buses to contiguous 0-based
    old_ids = ppc['bus'][:, 0]
    bus_map = {old: new for new, old in enumerate(old_ids)}
    ppc['bus'][:, 0] = np.arange(len(old_ids))
    for gen in ppc['gen']:
        gen[0] = bus_map[gen[0]]
    for branch in ppc['branch']:
        branch[0] = bus_map[branch[0]]
        branch[1] = bus_map[branch[1]]
    for dcline in ppc['dcline']:
        dcline[0] = bus_map[dcline[0]]
        dcline[1] = bus_map[dcline[1]]

    return ppc


# ===========================================================================
# 8. PyPSA → PyPower Converter
# ===========================================================================

class PypsaPypower:
    """Convert a PyPSA ``Network`` into a PyPower ``ppc`` case dict.

    Parameters
    ----------
    network : pypsa.Network
        Source network (must have ``generators``, ``buses``, ``lines``,
        ``links``, and optionally ``storage_units``).
    args : dict or None
        Conversion options — ``BaseMVA``, ``phase_factor``,
        ``voltage_upper/lower``, ``phase_angle_upper/lower``,
        ``reactive_gen_upper/lower``, ``storage_mode``,
        ``storage_ratio``.
    """

    def __init__(self, network, args=None):
        self.network = network
        self.args = args or {}
        self.ppc = None
        self._cache = {}

    def convert(self):
        """Run the full conversion pipeline.

        Returns
        -------
        dict
            PyPower case dictionary.
        """
        self.ppc = self._initialize_ppc()
        self._convert_generators()
        self._convert_buses()
        self._convert_ac_lines()
        self._convert_dc_lines()
        if self.args.get('storage_mode'):
            self._convert_storage()
        return self.ppc

    # --- Internal helpers --------------------------------------------------

    def _initialize_ppc(self):
        """Create an empty PyPower case dict."""
        return {
            'version': '2',
            'baseMVA': self.args['BaseMVA'],
            'bus': [], 'gen': [], 'gencost': [],
            'dcline': [], 'dclinecost': [], 'branch': [],
        }

    def _convert_generators(self):
        """Populate ``ppc['gen']`` and ``ppc['gencost']``.

        Generator data columns:
        ``bus  Pg  Qg  Qmax  Qmin  Vg  mBase  status  Pmax  Pmin  Extendable``

        Cost data columns:
        ``2  startup  shutdown  n  c(n-1)  …  c0``
        """
        phase_factor = self.args['phase_factor']
        gen_data = []
        gen_cost_data = []
        generators = self.network.generators
        gen_ids = generators.index.to_list()
        PV_bus = []

        for gen_name in gen_ids:
            if generators.loc[gen_name, 'p_nom'] <= 0:
                continue
            bus_name = generators.loc[gen_name, 'bus']
            PV_bus.append(bus_name)
            bus_idx = self.network.buses.index.tolist().index(bus_name)
            status = 1
            p_nom = generators.loc[gen_name, 'p_nom'] / phase_factor
            derating = self.network.generators.loc[gen_name, 'derating']

            Pmax = (generators.loc[gen_name, 'p_max_pu']
                    * p_nom * derating)
            Pmin = generators.loc[gen_name, 'p_min_pu'] * p_nom
            Qmax = Pmax * self.args['reactive_gen_upper']
            Qmin = Pmax * self.args['reactive_gen_lower']
            Extendable = generators.loc[gen_name, 'p_nom_extendable']

            gen_data.append([
                bus_idx, 0, 0, Qmax, Qmin, 1, p_nom, status,
                Pmax, Pmin, Extendable,
            ])

            marginal_cost = generators.loc[gen_name, 'marginal_cost']
            extend_cost = generators.loc[gen_name, 'capital_cost']
            gen_cost_data.append([
                2, 0, 0, 3, 0, marginal_cost, 0, extend_cost,
            ])

        self.ppc['gen'] = np.array(gen_data)
        self.ppc['gencost'] = np.array(gen_cost_data)
        self.PV_bus = PV_bus

    def _convert_buses(self):
        """Populate ``ppc['bus']`` and derived fields.

        Bus data columns:
        ``bus_i  type  Pd  Qd  Gs  Bs  area  Vm  Va  baseKV  zone
        Vmax  Vmin  busx  busy``
        """
        bus = self.network.buses
        pop_ratio = self.network.buses['pop_ratio'].values
        bus_data = []

        for i, bus_name in enumerate(bus.index):
            control = bus.loc[bus_name, 'control']
            bus_type = {'PV': 2, 'PQ': 1, 'Slack': 3}.get(control)
            if bus_type is None:
                raise KeyError(
                    f"Unknown control type '{control}' at bus {bus_name}")

            Pd = Qd = 0
            if 'p_set' in self.network.buses:
                Pd = bus.loc[bus_name, 'p_set'] / self.args['phase_factor']
                Qd = bus.loc[bus_name, 'q_set'] / self.args['phase_factor']

            baseKV = (bus.loc[bus_name, 'v_nom']
                      / np.sqrt(self.args['phase_factor']))

            bus_data.append([
                i, bus_type, Pd, Qd,
                0, 0, 0,  # Gs, Bs, area
                1, 0, baseKV, 0,  # Vm, Va, baseKV, zone
                self.args['voltage_upper'], self.args['voltage_lower'],
                bus.loc[bus_name, 'x'], bus.loc[bus_name, 'y'],
            ])

        self.ppc['bus'] = np.array(bus_data)
        self.ppc['pop_ratio'] = np.array(pop_ratio)
        self.ppc['baseKV'] = baseKV
        self.ppc['baseI'] = self.ppc['baseMVA'] / self.ppc['baseKV']

    def _convert_ac_lines(self):
        """Populate ``ppc['branch']`` and ``ppc['segment']``.

        Branch data columns:
        ``fbus  tbus  r  x  b  rateA  rateB  rateC  ratio  angle  status
        angmin  angmax  num_line  num_seg  line_length``
        """
        branch_data = []
        segment_data = self.network.segments
        branch = self.network.lines
        line_type = self.network.line_types
        baseMVA = self.ppc['baseMVA']
        phase_factor = self.args['phase_factor']
        segment_len = 25

        for i, branch_name in enumerate(branch.index):
            fbus_name = branch.loc[branch_name, 'bus0']
            tbus_name = branch.loc[branch_name, 'bus1']
            fbus = self.network.buses.index.tolist().index(fbus_name)
            tbus = self.network.buses.index.tolist().index(tbus_name)
            conductor_type = branch.loc[branch_name, 'type']
            line_length = branch.loc[branch_name, 'length']
            num_line = branch.loc[branch_name, 'num_parallel']
            num_seg = int(line_length // segment_len + 1)

            conductor = line_type.loc[conductor_type]
            unit_x = conductor['x_per_length']     # Ω/km
            unit_r = conductor['r_per_length']      # Ω/km
            unit_c = conductor['c_per_length']      # nF/km
            f_nom = conductor['f_nom']

            v_nom = self.ppc['baseKV']
            BaseZ = v_nom ** 2 / baseMVA
            BaseY = 1 / BaseZ

            s_nom = branch.loc[branch_name, 's_nom'] / phase_factor
            rateA = s_nom
            rateB = (branch.loc[branch_name, 'Imax']
                     * num_line / 1000)
            rateC = s_nom

            if rateA == 0:
                branch_data.append([
                    fbus, tbus, 9999, 9999, 0,
                    0, 0, 0, 0, 0, 0,
                    -9999, 9999, num_line, num_seg, line_length,
                ])
            else:
                angmin = self.args['phase_angle_lower']
                angmax = self.args['phase_angle_upper']
                Br_R = unit_r * line_length / num_line / BaseZ
                Br_X = unit_x * line_length / num_line / BaseZ
                Br_B = (2 * np.pi * f_nom * unit_c
                        * line_length * num_line * 1e-9 / BaseY)
                branch_data.append([
                    fbus, tbus, Br_R, Br_X, Br_B,
                    rateA, rateB, rateC,
                    0, 0, 1, angmin, angmax,
                    num_line, num_seg, line_length,
                ])

        self.ppc['branch'] = np.array(branch_data)
        self.ppc['segment'] = np.array(segment_data)

    def _convert_dc_lines(self):
        """Populate ``ppc['dcline']`` and ``ppc['dclinecost']``.

        DC-line data columns:
        ``fbus  tbus  status  Pf  Pt  Qf  Qt  Vf  Vt  Pmin  Pmax
        QminF  QmaxF  QminT  QmaxT  loss0  loss1``
        """
        dc_data = []
        dc_cost_data = []
        links = self.network.links
        phase_factor = self.args['phase_factor']

        for dcline_name in links.index:
            fbus_name = links.loc[dcline_name, 'bus0']
            tbus_name = links.loc[dcline_name, 'bus1']
            fbus = self.network.buses.index.tolist().index(fbus_name)
            tbus = self.network.buses.index.tolist().index(tbus_name)
            efficiency = links.loc[dcline_name, 'efficiency']
            p_nom = (links.loc[dcline_name, 'p_nom']
                     / phase_factor * efficiency)
            Pmin = links.loc[dcline_name, 'p_min_pu'] * p_nom
            Pmax = links.loc[dcline_name, 'p_max_pu'] * p_nom

            marginal_cost = links.loc[dcline_name, 'marginal_cost']
            marginal_cost_q = links.loc[dcline_name,
                                        'marginal_cost_quadratic']
            capital_cost = links.loc[dcline_name, 'capital_cost']

            dc_data.append([
                fbus, tbus, 1, 0, 0, 0, 0, 1, 1,
                Pmin, Pmax, 0, 0, 0, 0, 0, 0,
            ])
            dc_cost_data.append([
                2, 0, 0, 3, marginal_cost_q, marginal_cost, capital_cost,
            ])

        self.ppc['dcline'] = np.array(dc_data)
        self.ppc['dclinecost'] = np.array(dc_cost_data)

    def _convert_storage(self):
        """Populate ``ppc['storage']`` and ``ppc['storagecost']``."""
        storage_data = []
        storage_cost_data = []
        storage = self.network.storage_units
        phase_factor = self.args['phase_factor']

        for st_name in storage.index:
            bus_name = storage.loc[st_name, 'bus']
            bus_idx = self.network.buses.index.tolist().index(bus_name)
            eff_disp = storage.loc[st_name, 'efficiency_dispatch']
            p_nom = (storage.loc[st_name, 'p_nom']
                     / phase_factor * self.args['storage_ratio'])
            Pmin = storage.loc[st_name, 'p_min_pu'] * p_nom * eff_disp
            Pmax = storage.loc[st_name, 'p_max_pu'] * p_nom * eff_disp
            eff_store = storage.loc[st_name, 'efficiency_store']

            marginal_cost = storage.loc[st_name, 'marginal_cost']
            marginal_cost_q = storage.loc[st_name,
                                          'marginal_cost_quadratic']
            capital_cost = storage.loc[st_name, 'capital_cost']

            storage_data.append([
                bus_idx, 1, Pmin, Pmax, eff_store, eff_disp,
            ])
            storage_cost_data.append([
                2, 0, 0, 3, marginal_cost_q, marginal_cost, capital_cost,
            ])

        self.ppc['storage'] = np.array(storage_data)
        self.ppc['storagecost'] = np.array(storage_cost_data)


def pypsa_pypower(network, args=None):
    """Convenience wrapper: convert a PyPSA network to a PyPower case dict.

    Parameters
    ----------
    network : pypsa.Network
    args : dict or None
        Conversion options (see :class:`PypsaPypower`).

    Returns
    -------
    dict
        PyPower ``ppc`` case dictionary.
    """
    converter = PypsaPypower(network, args)
    return converter.convert()

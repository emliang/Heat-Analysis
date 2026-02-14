#!/usr/bin/env python3
"""Country-level network filtering utilities for PyPSA-Eur.

This module extracts a sub-network from a full PyPSA-Eur model by
retaining only the buses, generators, lines, links, loads, and storage
elements that belong to a user-specified set of countries.  Optionally,
cross-border interconnections between the selected countries can be
kept or discarded.

After filtering, isolated buses (those with no remaining lines or
links) and their dependent components are iteratively removed to
produce a self-consistent network.

Section index
-------------
1. Network Filtering
2. Isolated-Bus Removal
3. Network Diagnostics
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from typing import List

import pandas as pd
import pypsa


# ===========================================================================
# 1. Network Filtering
# ===========================================================================

def filter_network_by_countries(
        network: pypsa.Network,
        selected_countries: List[str],
        keep_interconnections: bool = True,
        verbose: bool = True) -> pypsa.Network:
    """Filter a PyPSA network to include only selected countries.

    Parameters
    ----------
    network : pypsa.Network
        The original (unmodified) PyPSA network.
    selected_countries : list[str]
        Two-letter country codes to keep (e.g. ``['ES', 'FR', 'DE']``).
    keep_interconnections : bool
        If *True*, transmission lines/links whose **both** endpoints
        are in *selected_countries* are retained (including cross-border
        lines).  If *False*, only lines within a single country are kept.
    verbose : bool
        Print filtering statistics.

    Returns
    -------
    pypsa.Network
        A copy of *network* containing only the selected components.

    Examples
    --------
    >>> network = pypsa.Network('elec_s_50.nc')
    >>> filtered = filter_network_by_countries(network, ['ES', 'FR', 'IT'])
    """
    if verbose:
        print(f"Original network: {len(network.buses)} buses, "
              f"{len(network.lines)} lines, "
              f"{len(network.generators)} generators")
        print(f"Filtering for countries: {selected_countries}")

    filtered = network.copy()

    # 1. Buses ---------------------------------------------------------------
    country_buses = filtered.buses[
        filtered.buses['country'].isin(selected_countries)
    ].index

    # 2. Generators (with positive capacity) ---------------------------------
    gen_mask = (filtered.generators['bus'].isin(country_buses)
                & (filtered.generators['p_nom'] > 0))
    country_generators = filtered.generators[gen_mask].index

    # 3. Loads ---------------------------------------------------------------
    country_loads = filtered.loads[
        filtered.loads['bus'].isin(country_buses)
    ].index

    # 4. Storage units & stores ----------------------------------------------
    if len(filtered.storage_units) > 0:
        su_mask = (filtered.storage_units['bus'].isin(country_buses)
                   & (filtered.storage_units['p_nom'] > 0))
        country_storage = filtered.storage_units[su_mask].index
    else:
        country_storage = []

    if len(filtered.stores) > 0:
        country_stores = filtered.stores[
            filtered.stores['bus'].isin(country_buses)
        ].index
    else:
        country_stores = []

    # 5. Lines (AC) ----------------------------------------------------------
    if keep_interconnections:
        line_mask = (filtered.lines['bus0'].isin(country_buses)
                     & filtered.lines['bus1'].isin(country_buses))
    else:
        lc0 = filtered.buses.loc[filtered.lines['bus0'], 'country'].values
        lc1 = filtered.buses.loc[filtered.lines['bus1'], 'country'].values
        line_mask = (
            pd.Series(lc0, index=filtered.lines.index).isin(
                selected_countries)
            & pd.Series(lc1, index=filtered.lines.index).isin(
                selected_countries)
            & (lc0 == lc1)
        )
    country_lines = filtered.lines[line_mask].index

    # 6. Links (DC / HVDC) ---------------------------------------------------
    if len(filtered.links) > 0:
        if keep_interconnections:
            link_mask = (filtered.links['bus0'].isin(country_buses)
                         & filtered.links['bus1'].isin(country_buses))
        else:
            lkc0 = filtered.buses.loc[
                filtered.links['bus0'], 'country'].values
            lkc1 = filtered.buses.loc[
                filtered.links['bus1'], 'country'].values
            link_mask = (
                pd.Series(lkc0, index=filtered.links.index).isin(
                    selected_countries)
                & pd.Series(lkc1, index=filtered.links.index).isin(
                    selected_countries)
                & (lkc0 == lkc1)
            )
        country_links = filtered.links[link_mask].index
    else:
        country_links = []

    # 7. Remove components outside selected countries ------------------------
    _remove_diff(filtered, "Bus", country_buses)
    _remove_diff(filtered, "Generator", country_generators)
    _remove_diff(filtered, "Load", country_loads)
    _remove_diff(filtered, "Line", country_lines)
    _remove_diff(filtered, "StorageUnit", country_storage)
    _remove_diff(filtered, "Store", country_stores)

    # Remove buses that only host stores (not transmission buses)
    filtered.remove("Bus", filtered.stores.bus)

    _remove_diff(filtered, "Link", country_links)

    # Remove links connected to store buses
    store_buses = set(filtered.stores.bus)
    links_on_stores = filtered.links.index[
        filtered.links.bus0.isin(store_buses)
        | filtered.links.bus1.isin(store_buses)
    ]
    filtered.remove("Link", links_on_stores)

    if verbose:
        print(f"Filtered network: {len(filtered.buses)} buses, "
              f"{len(filtered.lines)} lines, "
              f"{len(filtered.generators)} generators")

    # 8. Clean up isolated buses ---------------------------------------------
    filtered = remove_isolated_buses(filtered, verbose=verbose)

    return filtered


def _remove_diff(network, component, keep_index):
    """Remove *component* entries not present in *keep_index*.

    Parameters
    ----------
    network : pypsa.Network
    component : str
        PyPSA component name (e.g. ``'Bus'``, ``'Line'``).
    keep_index : pd.Index or list
        Indices to retain.
    """
    df = getattr(network, network.components[component]["list_name"])
    to_remove = df.index.difference(keep_index)
    if len(to_remove) > 0:
        network.remove(component, to_remove)


# ===========================================================================
# 2. Isolated-Bus Removal
# ===========================================================================

def remove_isolated_buses(network: pypsa.Network,
                          verbose: bool = True) -> pypsa.Network:
    """Iteratively remove buses with no lines or links, and their dependents.

    Each iteration identifies buses that are not an endpoint of any AC
    line or DC link, removes those buses together with their generators,
    loads, storage units, and stores, then repeats until no isolated
    buses remain (or a safety limit of 10 iterations is reached).

    Parameters
    ----------
    network : pypsa.Network
        Network to clean (modified in place **and** returned).
    verbose : bool
        Print per-iteration removal statistics.

    Returns
    -------
    pypsa.Network
        The same *network* object, with isolated elements removed.
    """
    totals = {
        'buses': 0, 'generators': 0, 'loads': 0,
        'storage': 0, 'stores': 0,
    }

    for iteration in range(1, 11):  # safety cap at 10
        connected = _connected_buses(network)
        isolated = list(set(network.buses.index) - connected)

        if len(isolated) == 0:
            break

        if verbose:
            print(f"Iteration {iteration}: "
                  f"Found {len(isolated)} isolated buses")

        # Remove dependents attached to isolated buses
        for attr, component, label in [
            ('generators',    'Generator',   'generators'),
            ('loads',         'Load',        'loads'),
            ('storage_units', 'StorageUnit', 'storage'),
            ('stores',        'Store',       'stores'),
        ]:
            df = getattr(network, attr)
            if len(df) == 0:
                continue
            victims = df[df['bus'].isin(isolated)].index
            if len(victims) > 0:
                network.remove(component, victims)
                totals[label] += len(victims)
                if verbose:
                    print(f"  Removed {len(victims)} {label}")

        # Remove the buses themselves
        network.remove("Bus", isolated)
        totals['buses'] += len(isolated)
        if verbose:
            print(f"  Removed {len(isolated)} isolated buses")
    else:
        print("Warning: Maximum iterations reached in isolated bus removal")

    if verbose:
        if totals['buses'] > 0:
            print(f"\nTotal isolated elements removed:")
            for label, count in totals.items():
                if count > 0:
                    print(f"  {label.capitalize()}: {count}")
            print(f"Final network: {len(network.buses)} buses, "
                  f"{len(network.lines)} lines, "
                  f"{len(network.generators)} generators")
        else:
            print("No isolated buses found.")

    return network


def _connected_buses(network):
    """Return the set of bus names that appear as endpoints of lines/links.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    set[str]
    """
    connected = set()
    if len(network.lines) > 0:
        connected.update(network.lines['bus0'].values)
        connected.update(network.lines['bus1'].values)
    if len(network.links) > 0:
        connected.update(network.links['bus0'].values)
        connected.update(network.links['bus1'].values)
    return connected


# ===========================================================================
# 3. Network Diagnostics
# ===========================================================================

def find_isolated_buses(network: pypsa.Network) -> List[str]:
    """Return bus names that have no lines or links.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    list[str]
        Isolated bus names.
    """
    return list(set(network.buses.index) - _connected_buses(network))


def get_country_statistics(network: pypsa.Network) -> pd.DataFrame:
    """Compute per-country summary statistics for the network.

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    pd.DataFrame
        Columns: ``country``, ``n_buses``, ``n_generators``,
        ``n_conv_generators``, ``n_ren_generators``,
        ``total_capacity_GW``, ``conv_capacity_GW``, ``ren_capacity_GW``.
    """
    CONVENTIONAL = {
        'nuclear', 'oil', 'OCGT', 'CCGT', 'coal',
        'lignite', 'geothermal', 'biomass',
    }
    RENEWABLE = {
        'solar', 'onwind', 'offwind-ac', 'offwind-dc', 'hydro',
    }

    stats = []
    for country in network.buses['country'].unique():
        c_buses = network.buses[network.buses['country'] == country]
        c_gens = network.generators[
            network.generators['bus'].isin(c_buses.index)]
        conv = c_gens[c_gens['carrier'].isin(CONVENTIONAL)]
        ren = c_gens[c_gens['carrier'].isin(RENEWABLE)]

        stats.append({
            'country': country,
            'n_buses': len(c_buses),
            'n_generators': len(c_gens),
            'n_conv_generators': len(conv),
            'n_ren_generators': len(ren),
            'total_capacity_GW': c_gens['p_nom'].sum() / 1e3,
            'conv_capacity_GW': conv['p_nom'].sum() / 1e3,
            'ren_capacity_GW': ren['p_nom'].sum() / 1e3,
        })

    return pd.DataFrame(stats).sort_values('n_buses', ascending=False)

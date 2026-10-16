"""Shared scientific and display definitions for Supplementary Fig. 1."""

from __future__ import annotations

import numpy as np


CATEGORY_ORDER = ("Renewable", "Conventional", "Storage")
CATEGORY_PALETTE = {
    "Renewable": "#ffd21f",
    "Conventional": "#c23b3b",
    "Storage": "#1f78b4",
}
RENEWABLE_CARRIERS = {
    "solar",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "ror",
    "biomass",
}
CONVENTIONAL_CARRIERS = {"CCGT", "OCGT", "coal", "oil", "nuclear"}
STORAGE_CARRIERS = {"hydro", "PHS", "battery", "H2"}


def capacity_category(carrier: str) -> str:
    if carrier in RENEWABLE_CARRIERS:
        return "Renewable"
    if carrier in CONVENTIONAL_CARRIERS:
        return "Conventional"
    if carrier in STORAGE_CARRIERS:
        return "Storage"
    return "Conventional"


def line_width_from_gw(capacity_gw):
    return 0.35 + 2.4 * np.sqrt(np.asarray(capacity_gw, dtype=float) / 5.0)


def pie_radius_from_mw(capacity_mw: float, max_capacity_mw: float) -> float:
    return 0.035 + 0.22 * np.sqrt(float(capacity_mw) / float(max_capacity_mw))

#!/usr/bin/env python3
"""Export historical monthly-hourly generator availability from a PyPSA network."""

import argparse
from pathlib import Path
import sys

import pypsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.renewable_history import export_historical_availability


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(export_historical_availability(
        pypsa.Network(args.network), args.output, args.network))

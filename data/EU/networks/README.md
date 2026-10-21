# Historical generator availability

`historical_p_max_pu_s_75.nc` contains historical mean availability derived from
the PyPSA network. These values are used where matching future weather-driven
renewable profiles are unavailable; matching Atlite profiles take precedence.

- Coordinates: `month`, `hour`, and `generator` (the network's generator ID).
- Variable: `p_max_pu`, the mean available output per unit of nominal capacity,
  not actual dispatched generation.
- Coverage: June to August, with 24 hourly means per month. The bundled Spain
  example contains only the July 13:00 values needed for that example.
- Source: the NetCDF attributes retain the source network filename and SHA-256
  checksum. The file must correspond to the network used in the simulation.

The simulation loads the file once per process and selects the matching month
and hour. Missing historical values are not extrapolated or replaced with zero.
Generators without a time-varying availability series retain their static setting.

To regenerate from a corresponding PyPSA network, run from the repository root:

```bash
python scripts/prepare_renewable_history.py \
  --network data/EU/networks/base_s_75_elec.nc \
  --output data/EU/networks/historical_p_max_pu_s_75.nc
```

This preprocessing computes historical means only; it does not run OPF.

#!/usr/bin/env python3
"""P2 - how far each network's coordinates fall from the range the legacy embedding was built for.

The legacy SpatialMLP input is ((lat - 28.6) / 0.2, (lon - 77.1) / 0.3), designed so Delhi maps to about
[-1, 1]. This prints, per dataset, the raw coordinate range and the resulting MLP input range, plus the
input range under training-network normalization (mean / std of all stations, as an illustration).
No training. Out: results/exp_P2_coordinate_range.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import DataConfig  # noqa: E402
from src.data_beijing import prepare_real_beijing  # noqa: E402
from src.data_pemsbay import prepare_real_pemsbay  # noqa: E402
from src.data_real import prepare_real_delhi  # noqa: E402
from src.data_traffic_real import prepare_real_traffic  # noqa: E402
from src.data_weather import prepare_real_weather  # noqa: E402

SETS = {"delhi_air_quality": (prepare_real_delhi, 60, 30), "beijing_air_quality": (prepare_real_beijing, 30, 15),
        "ghcn_weather": (prepare_real_weather, 30, 15), "pemsd7_traffic": (prepare_real_traffic, 48, 12),
        "pemsbay_traffic": (prepare_real_pemsbay, 48, 12)}
out = {}
for name, (prep, w, h) in SETS.items():
    c = prep(DataConfig(window_size=w, horizon=h, n_features=1), hold_out_stations=[])[3]["coords_all"].astype(float)
    legacy = np.stack([(c[:, 0] - 28.6) / 0.2, (c[:, 1] - 77.1) / 0.3], 1)
    norm = (c - c.mean(0)) / (c.std(0) + 1e-6)
    out[name] = {"n": len(c), "lat": [round(c[:, 0].min(), 3), round(c[:, 0].max(), 3)],
                 "lon": [round(c[:, 1].min(), 3), round(c[:, 1].max(), 3)],
                 "legacy_input_lat": [round(legacy[:, 0].min(), 1), round(legacy[:, 0].max(), 1)],
                 "legacy_input_lon": [round(legacy[:, 1].min(), 1), round(legacy[:, 1].max(), 1)],
                 "legacy_max_abs": round(float(np.abs(legacy).max()), 1),
                 "legacy_within_station_spread_lat": round(float(legacy[:, 0].std()), 3),
                 "legacy_within_station_spread_lon": round(float(legacy[:, 1].std()), 3),
                 "normalized_max_abs": round(float(np.abs(norm).max()), 2)}
    r = out[name]
    print(f"{name:20s} N={r['n']:3d} lat {r['lat']} lon {r['lon']} -> legacy input lat {r['legacy_input_lat']} "
          f"lon {r['legacy_input_lon']} (|max| {r['legacy_max_abs']}, spread {r['legacy_within_station_spread_lat']}/"
          f"{r['legacy_within_station_spread_lon']}); normalized |max| {r['normalized_max_abs']}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                 "exp_P2_coordinate_range.json"), "w"), indent=1)

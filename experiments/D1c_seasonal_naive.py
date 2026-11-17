#!/usr/bin/env python3
"""D1c - seasonal-naive baseline on the same held-out sensors and windows as D1/D1b. No training.

A seasonal-naive forecast repeats the sensor's own value one period earlier:
  daily networks (Delhi, Beijing, weather): period 7 days, i.e. the last observed week repeated;
  traffic: the same time of the previous day, 288 steps on PeMSD7-M (5-minute cadence) and 48 steps on PeMS-BAY
  (30-minute cadence after the loader's subsampling).
For PeMSD7-M the previous day lies outside the 48-step input window, so it is read from the sensor's own series
before the forecast origin (never from the forecast period). Windows that start before one full day of data
exists have no seasonal-naive forecast; they are stored as NaN and every comparison in the paper restricts all
methods to the windows that have one (`first_valid_window`).

Out: results/D1c/<domain>_trial<k>.npz with S_seasonal (R, W, H), first_valid_window.
Run: python3 experiments/D1c_seasonal_naive.py <domain|all>
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from D1_sensor_level_scores import DOMAINS, make_loader  # noqa: E402
from src.data import DataConfig  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
# One day for the traffic networks: PeMSD7-M is at 5-minute cadence (288 steps); PeMS-BAY is loaded at 30-minute
# cadence (src/data_pemsbay.py subsamples every 6th 5-minute step), so one day is 48 steps.
PERIOD = {"delhi_air_quality": 7, "beijing_air_quality": 7, "ghcn_weather": 7,
          "pemsd7_traffic": 288, "pemsbay_traffic": 48}


def seasonal_errors(ds, held, period, sd):
    """(len(held), windows, horizon) absolute errors in original units; NaN where no forecast exists."""
    H, Wn = ds.horizon, ds.window_size
    feats = ds.features[:, held, 0].numpy()                    # (T, R)
    out = np.full((len(held), len(ds.indices), H), np.nan, dtype=np.float32)
    for w, start in enumerate(ds.indices):
        end = start + Wn
        y = feats[end:end + H]                                  # (H, R)
        if period <= Wn:                                        # inside the input window (daily networks)
            x = feats[start:end]
            pred = x[-period + (np.arange(H) % period)]
        else:                                                   # before the window, still before the origin
            idx = end + np.arange(H) - period
            if idx.min() < 0 or idx.max() >= end:
                continue
            pred = feats[idx]
        out[:, w, :] = (np.abs(y - pred) * sd).T
    return out


def run(domain):
    spec = DOMAINS[domain]
    os.makedirs(os.path.join(RES, "D1c"), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(RES, "D1b", f"{domain}_trial*.npz"))):
        dest = os.path.join(RES, "D1c", os.path.basename(f))
        if os.path.exists(dest):
            continue
        z = np.load(f, allow_pickle=True)
        held = [int(i) for i in z["held"]]
        sd = float(z["sd"])
        cfg = DataConfig(window_size=spec["window"], horizon=spec["horizon"], n_features=1)
        _, _, _, meta = spec["prepare"](cfg, hold_out_stations=held)
        ds = make_loader(meta, spec["window"], spec["horizon"]).dataset
        S = seasonal_errors(ds, held, PERIOD[domain], sd)
        valid = np.nonzero(~np.isnan(S[0, :, 0]))[0]
        first = int(valid[0]) if valid.size else -1
        np.savez_compressed(dest, held=np.array(held), S_seasonal=S, first_valid_window=first)
        print(f"{domain} {os.path.basename(f)}: windows {S.shape[1]}, first valid {first}, MAE {np.nanmean(S):.4g}",
              flush=True)


if __name__ == "__main__":
    which = list(DOMAINS) if len(sys.argv) < 2 or sys.argv[1] == "all" else [sys.argv[1]]
    for d in which:
        run(d)

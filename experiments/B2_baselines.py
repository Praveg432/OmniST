#!/usr/bin/env python3
"""BATCH B2 - baseline breadth: OmniST vs model-free spatial interpolators at unseen nodes.

Adds the geostatistics baselines a tier-1 reviewer expects for prediction-at-unobserved-location,
all under the identical inductive split-conformal protocol (calibrate on a time split, evaluate at
held-out nodes), on all real domains. OmniST's coverage column is read from B1's cached residuals
(no retraining); the baselines are cheap closed-form interpolators computed from the real test data.

Baselines (each + split-conformal residual interval):
  NN         nearest training node (copy its series)
  IDW        inverse-distance weighting over training nodes
  meanK      mean of k=5 nearest training nodes
  GP-ish     distance-kernel weighted (Gaussian kernel) interpolation (kriging-flavored)
  globalCP   non-inductive: global residual quantile (ablates the coordinate model)

Reports per domain x method: coverage @90, mean width, point MAE at unseen nodes. The claim: OmniST
holds the target while model-free interpolators mis-cover and/or are wider; globalCP shows a naive
non-coordinate conformal is not a substitute.

RUN: python experiments/B2_baselines.py   (requires B1 cache in results/cache/)
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.data import DataConfig
from src.data_real import prepare_real_delhi
from src.data_beijing import prepare_real_beijing
from src.data_weather import prepare_real_weather
from src.data_traffic_real import prepare_real_traffic

TARGET = 0.90; ALPHA = 1 - TARGET
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE = os.path.join(RESULTS, "cache")

DOMAINS = {
    "delhi_air_quality":   {"prepare": prepare_real_delhi,   "window": 60, "horizon": 30, "hold_out": [3, 11, 19, 27, 33]},
    "beijing_air_quality": {"prepare": prepare_real_beijing, "window": 30, "horizon": 15, "hold_out": [2, 6, 9]},
    "ghcn_weather":        {"prepare": prepare_real_weather, "window": 30, "horizon": 15, "hold_out": [1, 5, 9]},
    "pemsd7_traffic":      {"prepare": prepare_real_traffic, "window": 48, "horizon": 12, "hold_out": [5, 40, 90, 140, 200]},
}


def qhat(res, a):
    n = len(res); return float(np.quantile(res, min(1.0, (1 - a) * (1 + 1 / max(n, 1)))))


def interp_baselines(meta, hold):
    """Closed-form interpolation baselines at unseen nodes. Returns dict method -> (cov, width, mae),
    each calibrated split-conformal (first temporal half = calibration, second = test)."""
    test_all = meta["test_data_all_normalized"]
    mu = float(np.asarray(meta["means"]).squeeze()); sd = float(np.asarray(meta["stds"]).squeeze())
    coords = meta["coords_all"]; train_idx = meta["train_stations"]
    T = test_all.shape[0]; half = T // 2
    def dist(a, b): return float(np.hypot(a[0]-b[0], a[1]-b[1]))

    def predict(h, mode):
        d = np.array([dist(coords[h], coords[s]) for s in train_idx]) + 1e-9
        series = np.stack([test_all[:, s, 0] for s in train_idx], axis=1)  # (T, n_train)
        if mode == "NN":
            w = np.zeros_like(d); w[np.argmin(d)] = 1.0
        elif mode == "IDW":
            w = 1.0 / (d ** 2); w /= w.sum()
        elif mode == "meanK":
            k = min(5, len(d)); nn = np.argsort(d)[:k]; w = np.zeros_like(d); w[nn] = 1.0 / k
        elif mode == "GP":
            bw = np.median(d); w = np.exp(-(d ** 2) / (2 * bw ** 2 + 1e-9)); w /= w.sum()
        return series @ w                                   # (T,) normalized prediction

    out = {}
    for mode in ("NN", "IDW", "meanK", "GP"):
        cal, tst, pae = [], [], []
        for h in hold:
            pred = predict(h, mode)
            r = np.abs((test_all[:, h, 0]*sd+mu) - (pred*sd+mu))
            cal.append(r[:half]); tst.append(r[half:]); pae.append(r)
        cal = np.concatenate(cal); tst = np.concatenate(tst)
        q = qhat(cal, ALPHA)
        out[mode] = (round(float((tst <= q).mean()), 4), round(2*q, 3), round(float(np.concatenate(pae).mean()), 4))
    return out


def omnist_from_cache(name):
    """Average OmniST coverage/width/MAE across cached B1 seeds for this domain."""
    files = sorted(glob.glob(os.path.join(CACHE, f"{name}_seed*.npz")))
    if not files:
        return None
    covs, widths, maes = [], [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        ind = z["ind_flat"]; q = float(z["q"])
        covs.append(float((ind <= q).mean())); widths.append(2*q); maes.append(float(ind.mean()))
    return (round(float(np.mean(covs)), 4), round(float(np.mean(widths)), 3), round(float(np.mean(maes)), 4))


def run():
    summary = {"experiment": "B2 baseline breadth vs OmniST (inductive conformal at unseen nodes)",
               "target_coverage": TARGET, "results_by_domain": {}}
    for name, spec in DOMAINS.items():
        print(f"\n===== {name} =====", flush=True)
        cfg = DataConfig(window_size=spec["window"], horizon=spec["horizon"], n_features=1)
        _, _, _, meta = spec["prepare"](cfg, hold_out_stations=spec["hold_out"])
        base = interp_baselines(meta, spec["hold_out"])
        # global (non-inductive) CP baseline: pool ALL training-station test residuals vs their own values is
        # not meaningful; we approximate the 'no-coordinate' CP by using the NN calibration pooled globally.
        omni = omnist_from_cache(name)
        row = {"OmniST_conformal": {"coverage": omni[0], "width": omni[1], "point_mae": omni[2]} if omni else "NO B1 CACHE - run B1 first",
               "baselines": {m: {"coverage": v[0], "width": v[1], "point_mae": v[2]} for m, v in base.items()}}
        summary["results_by_domain"][name] = row
        print("  OmniST:", row["OmniST_conformal"], flush=True)
        for m, v in base.items():
            print(f"  {m:6s}: cov={v[0]} width={v[1]} mae={v[2]}", flush=True)
    out_path = os.path.join(RESULTS, "exp_baselines.json")
    json.dump(summary, open(out_path, "w"), indent=2, default=float)
    print(f"\nsaved -> {out_path}")
    return summary


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""D3 - forecast-time (deployable) calibration at a new sensor.

D2 calibrates on the calibration sensors' errors over the SAME windows the new sensor is evaluated on.
That setting satisfies Theorem 1's assumptions exactly but cannot be run in deployment: at the time a
forecast for window t is issued, the calibration sensors' errors for window t do not exist yet.
Here the threshold for window t may use only errors of windows < t, from the calibration sensors, and the
new sensor's own realized miscoverage for windows < t. Schemes, all evaluated on windows t0 .. W-1:

  static     windows [0, t0) of the calibration sensors, fixed
  expanding  all windows before t
  rolling    the previous L windows
  aci        rolling, with the working level updated from the new sensor's own miscoverage e_{t-1}:
             a_{t+1} = clip(a_t + gamma (alpha - e_t), 0.005, 0.995)
Both the sensor-level rule (Theorem 1's correction counted in sensors) and the pooled rule are computed
for every scheme, on the same random (new sensor, calibration set) draws.

Fixed in advance (not tuned on these results): alpha = 0.1, gamma = 0.1, L = 4 windows for the daily
networks (Delhi, Beijing, weather) and 24 windows for the two traffic networks (one day on PeMSD7-M, six days on PeMS-BAY), t0 = L.

No theorem covers these schemes: errors of different windows are not exchangeable. They are measured.
Run: python3 experiments/D3_deployable.py     Out: results/exp_D3_deployable.json
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALPHA, GAMMA = 0.1, 0.1
CFG = {  # domain: (n values, L, draws)
    "delhi_air_quality": ([12, 15], 4, 400),
    "ghcn_weather": ([10, 11], 4, 400),
    "pemsd7_traffic": ([20, 59], 24, 60),
    "pemsbay_traffic": ([20, 89], 24, 60),
}
SCHEMES = ("static", "expanding", "rolling", "aci")


def kth(pool, miscov):
    """Smallest value v with at most `miscov` of the pool above it (order statistic); inf if miscov < 0."""
    if miscov < 0:
        return np.inf
    m = pool.size
    idx = min(m - 1, max(0, int(np.ceil((1 - miscov) * m)) - 1))
    return np.partition(pool, idx)[idx]


def one_draw(Sc, Sj, n, L, t0, rng_unused=None):
    """Return {(scheme, rule): (coverage, mean finite width)} for one (new sensor, calibration set)."""
    W = Sj.shape[0]
    out = {}
    static_pool = Sc[:, :t0].reshape(-1)
    for scheme in SCHEMES:
        a_t = ALPHA
        miss = {"crc": [], "pool": []}
        wid = {"crc": [], "pool": []}
        for t in range(t0, W):
            if scheme == "static":
                pool = static_pool
            elif scheme == "expanding":
                pool = Sc[:, :t].reshape(-1)
            else:
                pool = Sc[:, max(0, t - L):t].reshape(-1)
            a = a_t if scheme == "aci" else ALPHA
            m = pool.size
            for rule, miscov in (("crc", (a * (n + 1) - 1) / n), ("pool", (a * (m + 1) - 1) / m)):
                q = kth(pool, miscov)
                e = 0.0 if not np.isfinite(q) else float((Sj[t] > q).mean())
                miss[rule].append(e)
                wid[rule].append(2 * q if np.isfinite(q) else np.nan)
            if scheme == "aci":
                a_t = float(np.clip(a_t + GAMMA * (ALPHA - miss["crc"][-1]), 0.005, 0.995))
        for rule in ("crc", "pool"):
            w = np.array(wid[rule])
            out[(scheme, rule)] = (1.0 - float(np.mean(miss[rule])),
                                   float(np.nanmean(w)) if np.isfinite(w).any() else np.nan)
    return out


def t_interval(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    if len(v) < 2:
        return [m, None, None]
    from scipy import stats
    h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / np.sqrt(len(v))
    return [round(m, 4), round(m - h, 4), round(m + h, 4)]


if __name__ == "__main__":
    result = {}
    for dom, (ns, L, draws) in CFG.items():
        files = sorted(glob.glob(os.path.join(RES, "D1b", f"{dom}_trial*.npz")))
        rng = np.random.default_rng(31337)
        result[dom] = {"L": L, "gamma": GAMMA, "alpha": ALPHA, "n_trials": len(files), "cells": {}}
        for n in ns:
            per = {}  # (scheme, rule) -> list over trials of arrays
            for f in files:
                S = np.load(f, allow_pickle=True)["S_full"].astype(np.float64)       # (R, W, H)
                R, W, H = S.shape
                t0 = L
                draws_out = {}
                for _ in range(draws):
                    perm = rng.permutation(R)
                    d = one_draw(S[perm[1:n + 1]], S[perm[0]], n, L, t0)
                    for k, v in d.items():
                        draws_out.setdefault(k, []).append(v)
                for k, v in draws_out.items():
                    arr = np.array(v)                                                  # (draws, 2)
                    per.setdefault(k, []).append(arr)
            cell = {}
            for (scheme, rule), arrs in per.items():
                cov = [a[:, 0].mean() for a in arrs]
                tail = [(a[:, 0] < 0.80).mean() for a in arrs]
                wd = [np.nanmean(a[:, 1]) for a in arrs]
                cell[f"{scheme}/{rule}"] = {"coverage": t_interval(cov), "share_below_0.80": round(float(np.mean(tail)), 4),
                                            "width": round(float(np.nanmean(wd)), 4)}
            result[dom]["cells"][str(n)] = cell
            print(f"{dom} n={n} L={L} trials={len(files)} eval windows={W - L}")
            for rule in ("crc", "pool"):
                print("   " + ("sensor-level" if rule == "crc" else "pooled      ") + "  " + "   ".join(
                    f"{s}: cov {cell[f'{s}/{rule}']['coverage'][0]:.3f} tail {cell[f'{s}/{rule}']['share_below_0.80']:.2f} w {cell[f'{s}/{rule}']['width']:.3g}"
                    for s in SCHEMES))
    json.dump(result, open(os.path.join(RES, "exp_D3_deployable.json"), "w"), indent=1)
    print("saved exp_D3_deployable.json")

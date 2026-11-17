#!/usr/bin/env python3
"""D5b - (1) coverage over every block of k consecutive windows, per sensor; (2) distance groups.

(1) Corollary of Theorem 2. In per-sensor ACI the working level stays in [-gamma, 1 + gamma], so for every sensor
and every block of k consecutive windows,  | mean_{t in block} e_t - alpha | <= (1 + 2 gamma) / (gamma k).
k is fixed in advance as one week of windows on the traffic networks (PeMSD7-M: 1-hour windows on weekdays only,
5 x 24 = 120; PeMS-BAY: 6-hour windows, 28) and 6 windows on the daily networks (90 days for Beijing and weather,
180 days for Delhi, whose evaluation has only 6 windows). For each sensor case the worst block's coverage is
recorded, and the certificate is checked for every block.

(2) Distance groups, same-window setting (Theorem 1 applied within a group). In each split the held-out sensors are
divided into thirds by distance to the nearest training sensor, fixed before any error is seen. Within a third, each
sensor serves once as the new sensor with the rest of its third as calibration sensors: Theorem 1 then holds for a
randomly chosen sensor OF THAT THIRD, which needs at least 1/alpha - 1 = 9 calibration sensors per third, so this
is run on the two traffic networks only. Compared with calibrating on all other held-out sensors.

Run: python3 experiments/D5b_blocks_and_groups.py      Out: results/exp_D5b_blocks_groups.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from D5_per_sensor_guarantee import ALPHA, GAMMAS, L, RES, distances, q_level  # noqa: E402

BLOCK = {"delhi_air_quality": 6, "beijing_air_quality": 6, "ghcn_weather": 6, "pemsd7_traffic": 120,
         "pemsbay_traffic": 28}


def aci_errors(S, j, Lw, gamma):
    R, W, H = S.shape
    cal = np.array([i for i in range(R) if i != j])
    n, a_t, errs = cal.size, ALPHA, []
    for t in range(Lw, W):
        pool = np.sort(S[cal, max(0, t - Lw):t].reshape(-1))
        q = q_level(pool, a_t, n)
        e = 0.0 if q == np.inf else (1.0 if q < 0 else float((S[j, t] > q).mean()))
        errs.append(e)
        a_t += gamma * (ALPHA - e)
    return np.array(errs)


def blocks(e, k):
    if len(e) < k:
        return None
    c = np.concatenate([[0.0], np.cumsum(e)])
    return (c[k:] - c[:-k]) / k                     # miscoverage of every block of k consecutive windows


def same_window_cov(S, j, cal, t0):
    pool = np.sort(S[cal, t0:].reshape(-1))
    q = q_level(pool, ALPHA, cal.size)
    return 1.0 if not np.isfinite(q) else 1.0 - float((S[j, t0:] > q).mean()), (2 * q if np.isfinite(q) else np.inf)


if __name__ == "__main__":
    out = {}
    for dom in BLOCK:
        files = sorted(glob.glob(os.path.join(RES, "D1b", f"{dom}_trial*.npz")))
        k = BLOCK[dom]
        res = {"k": k, "eval_windows": None}
        for g in GAMMAS:
            worst, holds = [], True
            for f in files:
                S = np.load(f, allow_pickle=True)["S_full"].astype(np.float64)
                for j in range(S.shape[0]):
                    e = aci_errors(S, j, L[dom], g)
                    res["eval_windows"] = len(e)
                    b = blocks(e, k)
                    worst.append(1 - b.max())
                    holds &= bool(np.all(np.abs(b - ALPHA) <= (1 + 2 * g) / (g * k) + 1e-12))
            cert = 1 - ALPHA - (1 + 2 * g) / (g * k)
            res[f"aci-{g}"] = {"worst_block_cov_min": round(float(np.min(worst)), 4),
                               "worst_block_cov_mean": round(float(np.mean(worst)), 4),
                               "worst_block_cov_p05": round(float(np.percentile(worst, 5)), 4),
                               "certified_every_block": round(max(0.0, cert), 4), "certificate_holds": holds}
        if dom in ("pemsd7_traffic", "pemsbay_traffic"):
            groups = {name: {"group": [], "global": [], "w_group": [], "w_global": []} for name in ("near", "middle", "far")}
            for f in files:
                z = np.load(f, allow_pickle=True)
                S = z["S_full"].astype(np.float64)
                held, train = z["held"].astype(int), z["train"].astype(int)
                d = distances(dom, held, train)
                order = np.argsort(d)
                thirds = np.array_split(order, 3)
                t0 = L[dom]
                for name, members in zip(("near", "middle", "far"), thirds):
                    for j in members:
                        c_g, w_g = same_window_cov(S, j, np.array([m for m in members if m != j]), t0)
                        c_a, w_a = same_window_cov(S, j, np.array([m for m in range(S.shape[0]) if m != j]), t0)
                        groups[name]["group"].append(c_g); groups[name]["global"].append(c_a)
                        groups[name]["w_group"].append(w_g); groups[name]["w_global"].append(w_a)
            res["distance_groups"] = {
                name: {"n_cal_per_group": int(len(np.array_split(np.arange(S.shape[0]), 3)[0]) - 1),
                       "mean_cov_group_cal": round(float(np.mean(v["group"])), 4),
                       "mean_cov_all_cal": round(float(np.mean(v["global"])), 4),
                       "median_width_group_cal": round(float(np.median(v["w_group"])), 4),
                       "median_width_all_cal": round(float(np.median(v["w_global"])), 4)}
                for name, v in groups.items()}
        out[dom] = res
        print(f"{dom}: k={k} T={res['eval_windows']}", {g: res[f'aci-{g}'] for g in GAMMAS})
        if "distance_groups" in res:
            for name, v in res["distance_groups"].items():
                print("   ", name, v)
    json.dump(out, open(os.path.join(RES, "exp_D5b_blocks_groups.json"), "w"), indent=1)
    print("saved exp_D5b_blocks_groups.json")

#!/usr/bin/env python3
"""P1 - what contemporaneous calibration hides.

The paper's coverage experiments (D2) calibrate on the calibration sensors' errors over the SAME windows
that the new sensor is evaluated on. At forecast time those errors do not exist yet, so that protocol
cannot be run in deployment. This script compares, on the saved errors and with no retraining:

  contemporaneous  calibrate on windows [K, W) of the calibration sensors, evaluate the new sensor on [K, W)
  chronological    calibrate on windows [0, K) of the calibration sensors, evaluate the new sensor on [K, W)

K = W // 2. The same random (new sensor, calibration set) draws are used for both. Both rules are
reported: the sensor-level rule (Theorem 1) and the pooled rule.

Run: python3 experiments/P1_chronological_calibration.py     Out: results/exp_P1_chronological.json
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALPHA = 0.1
CELLS = {"delhi_air_quality": [12, 15], "ghcn_weather": [10, 11], "pemsd7_traffic": [20, 59],
         "pemsbay_traffic": [20, 89]}
DRAWS = {"delhi_air_quality": 1500, "ghcn_weather": 1500, "pemsd7_traffic": 400, "pemsbay_traffic": 400}


def q_of(pool_sorted, miscov):
    if miscov < 0:
        return np.inf
    m = pool_sorted.size
    return pool_sorted[min(m - 1, max(0, int(np.ceil((1 - miscov) * m)) - 1))]


def cell(S, K, n, draws, rng):
    R, W, H = S.shape
    early, late = S[:, :K].reshape(R, -1), S[:, K:].reshape(R, -1)
    res = {k: [] for k in ("chr_crc", "con_crc", "chr_pool", "con_pool")}
    a_n = (ALPHA * (n + 1) - 1) / n
    for _ in range(draws):
        perm = rng.permutation(R)
        j, cal = perm[0], perm[1:n + 1]
        for tag, src in (("chr", early), ("con", late)):
            pool = np.sort(src[cal].reshape(-1))
            m = pool.size
            a_m = (ALPHA * (m + 1) - 1) / m
            for name, a in (("crc", a_n), ("pool", a_m)):
                q = q_of(pool, a)
                res[f"{tag}_{name}"].append(1.0 if not np.isfinite(q) else float((late[j] <= q).mean()))
    return {k: np.array(v) for k, v in res.items()}


if __name__ == "__main__":
    out = {}
    for dom, ns in CELLS.items():
        files = sorted(glob.glob(os.path.join(RES, "D1b", f"{dom}_trial*.npz")))
        rng = np.random.default_rng(4242)
        out[dom] = {}
        for n in ns:
            per_trial = []
            for f in files:
                S = np.load(f, allow_pickle=True)["S_full"].astype(np.float64)
                per_trial.append(cell(S, S.shape[1] // 2, n, DRAWS[dom], rng))
            agg = {}
            for k in per_trial[0]:
                agg[k + "_cov"] = round(float(np.mean([t[k].mean() for t in per_trial])), 4)
                agg[k + "_share_below_0.80"] = round(float(np.mean([(t[k] < 0.80).mean() for t in per_trial])), 4)
            out[dom][str(n)] = agg
            print(dom, "n =", n, "trials =", len(files))
            print("   sensor-level: contemporaneous %.4f  chronological %.4f   | pooled: contemporaneous %.4f  chronological %.4f"
                  % (agg["con_crc_cov"], agg["chr_crc_cov"], agg["con_pool_cov"], agg["chr_pool_cov"]))
            print("   share of draws below 0.80 (pooled): contemporaneous %.3f  chronological %.3f"
                  % (agg["con_pool_share_below_0.80"], agg["chr_pool_share_below_0.80"]))
    json.dump(out, open(os.path.join(RES, "exp_P1_chronological.json"), "w"), indent=1)
    print("saved exp_P1_chronological.json")

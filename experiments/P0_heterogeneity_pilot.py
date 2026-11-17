#!/usr/bin/env python3
"""P0 - pilot: how much of the sensor-level price and the per-sensor tail is between-sensor heterogeneity?

Uses the per-sensor error arrays already on disk (results/D1b, OmniST with normalized coordinates).
No training. Three score definitions are compared under the SAME random (new sensor, calibration set)
draws and the same evaluation windows:

  plain    z = s                       (what the paper reports)
  oracle   z = s / m_i,  m_i = sensor i's mean error over ALL evaluation windows
           NOT deployable (uses the evaluation labels); an upper bound on what any scale estimate can do.
  burnin   z = s / m_i,  m_i = sensor i's mean error over the FIRST third of the windows; scores and
           coverage are measured on the remaining windows. Deployable: a sensor that has been running for
           a few windows has realized errors to estimate its own scale from. Applied identically to
           calibration and new sensors.

Interval for sensor j is  yhat +/- q * m_j  (plain: m_j = 1), so coverage is the share of z_j <= q and the
width is 2 q m_j. Both the sensor-level rule (Theorem 1) and the pooled rule are computed.
Validity: the normalizer is a fixed function of a sensor's own data that is applied identically to every
sensor, so the proof of Theorem 1 goes through with the loss defined on z; this pilot measures the gain.

Run: python3 experiments/P0_heterogeneity_pilot.py      Out: results/exp_P0_heterogeneity_pilot.json
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALPHA = 0.1
DOMAINS = {"delhi_air_quality": (12, 15), "ghcn_weather": (10, 11), "pemsd7_traffic": (20, 59),
           "pemsbay_traffic": (20, 89)}
DRAWS = {"delhi_air_quality": 1500, "ghcn_weather": 1500, "pemsd7_traffic": 400, "pemsbay_traffic": 400}


def order_stat(sorted_pool, level_miscov):
    m = sorted_pool.size
    if level_miscov < 0:
        return np.inf
    idx = int(np.ceil((1.0 - level_miscov) * m)) - 1
    return sorted_pool[min(m - 1, max(0, idx))]


def run_cell(z, scale, n, draws, rng):
    """z: (R, U) scores, scale: (R,) width multipliers. Returns per-draw coverage/width for both rules."""
    R, U = z.shape
    out = {k: [] for k in ("crc_cov", "crc_w", "pool_cov", "pool_w")}
    a_n = (ALPHA * (n + 1) - 1) / n
    a_m = (ALPHA * (n * U + 1) - 1) / (n * U)
    for _ in range(draws):
        perm = rng.permutation(R)
        j, cal = perm[0], perm[1:n + 1]
        pool = np.sort(z[cal].reshape(-1))
        for name, a in (("crc", a_n), ("pool", a_m)):
            q = order_stat(pool, a)
            out[f"{name}_cov"].append(1.0 if not np.isfinite(q) else float((z[j] <= q).mean()))
            out[f"{name}_w"].append(2.0 * q * scale[j] if np.isfinite(q) else np.inf)
    return {k: np.array(v) for k, v in out.items()}


def summarise(trial_cells):
    cov_c = np.mean([c["crc_cov"].mean() for c in trial_cells])
    cov_p = np.mean([c["pool_cov"].mean() for c in trial_cells])
    w_c = np.mean([c["crc_w"][np.isfinite(c["crc_w"])].mean() for c in trial_cells])
    w_p = np.mean([c["pool_w"].mean() for c in trial_cells])
    tail = np.mean([(c["pool_cov"] < ALPHA_TARGET - 0.10).mean() for c in trial_cells])
    p10 = np.mean([np.percentile(c["pool_cov"], 10) for c in trial_cells])
    return dict(crc_cov=round(float(cov_c), 4), crc_width=round(float(w_c), 4), pooled_cov=round(float(cov_p), 4),
                pooled_width=round(float(w_p), 4), price=round(float(w_c / w_p), 3),
                pooled_share_below_0p80=round(float(tail), 4), pooled_p10=round(float(p10), 3))


ALPHA_TARGET = 1 - ALPHA

if __name__ == "__main__":
    result = {}
    for dom, ns in DOMAINS.items():
        files = sorted(glob.glob(os.path.join(RES, "D1b", f"{dom}_trial*.npz")))
        rng = np.random.default_rng(777)
        result[dom] = {}
        for n in ns:
            cells = {"plain": [], "oracle": [], "burnin": []}
            for f in files:
                S = np.load(f, allow_pickle=True)["S_full"].astype(np.float64)       # (R, W, H)
                R, W, H = S.shape
                K = max(1, W // 3)
                ev = S[:, K:, :].reshape(R, -1)                                       # evaluation windows only
                m_oracle = S.reshape(R, -1).mean(axis=1)
                m_burn = S[:, :K, :].reshape(R, -1).mean(axis=1)
                seed_state = rng.bit_generator.state
                for name, scale in (("plain", np.ones(R)), ("oracle", m_oracle), ("burnin", m_burn)):
                    rng.bit_generator.state = seed_state                               # identical draws per variant
                    cells[name].append(run_cell(ev / scale[:, None], scale, n, DRAWS[dom], rng))
            result[dom][str(n)] = {k: summarise(v) for k, v in cells.items()}
            r = result[dom][str(n)]
            print(f"{dom} n={n} trials={len(files)}")
            for k in ("plain", "oracle", "burnin"):
                print(f"   {k:7s} {r[k]}")
    json.dump(result, open(os.path.join(RES, "exp_P0_heterogeneity_pilot.json"), "w"), indent=1)
    print("saved exp_P0_heterogeneity_pilot.json")

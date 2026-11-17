#!/usr/bin/env python3
"""D2 - sensor-level coverage analysis of the D1 score arrays (tests Theorem 1).

Setting. Within one trial the model is fixed and R sensors are held out. Theorem 1 is a statement
about a RANDOM assignment: one sensor of R is the new sensor, n others are calibration sensors. The
analysis therefore draws many random (test sensor, calibration subset) assignments from the saved
per-sensor errors and averages the test sensor's coverage. No retraining is needed.

Two ways of choosing the interval half-width q from the n calibration sensors' errors:
  crc    sensor-level correction (Theorem 1): smallest q with mean_i r_i(q) <= (a(n+1)-1)/n, where
         r_i(q) is the fraction of sensor i's (window, step) errors above q and n counts SENSORS.
         Infinite (vacuous) when n < 1/a - 1.
  naive  the usual split-conformal correction with m = n*U POINTS, which treats the n*U errors as
         exchangeable. They are not (errors within a sensor are dependent), so no guarantee.

q is searched on a fixed geometric grid, so each draw is a matrix mean and a first-crossing lookup.
The grid only discretises q upwards (conservative); it does not change the validity argument.

Also: paired point-accuracy comparisons across trials, and a drift re-run (static vs rolling vs
rolling-ACI) under the same sensor-level protocol.

Run:  python experiments/D2_analyse.py [domain ...]      Out: results/exp_D2_analysis.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALPHAS = (0.1, 0.2)
GRID = 3000
CAL_SIZES = {"delhi_air_quality": [9, 12, 15], "beijing_air_quality": [3, 4, 5],
             "ghcn_weather": [9, 10, 11], "pemsd7_traffic": [10, 20, 40, 59],
             "pemsbay_traffic": [10, 20, 45, 89]}
DRAWS = {"delhi_air_quality": 4000, "beijing_air_quality": 4000, "ghcn_weather": 4000,
         "pemsd7_traffic": 1500, "pemsbay_traffic": 1500}
FORECAST_METHODS = ["full", "no_gat", "ci", "full_legacy", "no_gat_legacy", "persist", "winmean"]
NOWCAST_METHODS = ["nn_conc", "idw_conc"]
DROP_METHODS = ["full_drop30", "full_drop50"]
# coverage/width cells are computed for these only (the rest enter through point accuracy)
COVERAGE_METHODS = ["full", "ci", "full_legacy", "persist", "nn_conc"] + DROP_METHODS


def t_ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return {"mean": float(v.mean()), "ci95": [None, None], "n": n}
    m, s = float(v.mean()), float(v.std(ddof=1))
    h = float(ss.t.ppf(0.975, n - 1)) * s / np.sqrt(n)
    return {"mean": round(m, 5), "sd": round(s, 5), "ci95": [round(m - h, 5), round(m + h, 5)],
            "min": round(float(v.min()), 5), "max": round(float(v.max()), 5), "n": n}


LEGACY_RENAME = {"full": "full_legacy", "no_gat": "no_gat_legacy",
                 "full_drop30": "full_legacy_drop30", "full_drop50": "full_legacy_drop50"}


def load_trials(domain):
    """Merge D1 (original Delhi coordinate constants, renamed *_legacy) with D1b (coordinates
    normalised from the training network, which take the plain names). Only trials present in both."""
    trials = []
    for f in sorted(glob.glob(os.path.join(RES, "D1", f"{domain}_trial*.npz"))):
        fb = f.replace(os.sep + "D1" + os.sep, os.sep + "D1b" + os.sep)
        if not os.path.exists(fb):
            continue
        z, zb = np.load(f, allow_pickle=True), np.load(fb, allow_pickle=True)
        t = {LEGACY_RENAME.get(k[2:], k[2:]): z[k] for k in z.files if k.startswith("S_")}
        assert np.array_equal(z["held"], zb["held"]), "D1 and D1b used different held-out sets"
        t.update({k[2:]: zb[k] for k in zb.files if k.startswith("S_")})
        trials.append(t)
    return trials


def risk_matrix(S, grid):
    """r_i(q_g): fraction of sensor i's errors strictly above q_g.  S: (R, W, H) -> (R, G)."""
    flat = np.sort(S.reshape(S.shape[0], -1), axis=1)
    U = flat.shape[1]
    return np.stack([1.0 - np.searchsorted(row, grid, side="right") / U for row in flat])


def make_grid(S):
    hi = float(S.max()) * 1.0001
    lo = max(float(np.median(S)) * 1e-3, 1e-6)
    return np.concatenate([[0.0], np.geomspace(lo, hi, GRID - 1)])


def first_crossing(mean_risk, alpha_eff):
    """Index of the smallest grid point with mean risk <= alpha_eff; -1 if none (alpha_eff < 0)."""
    if alpha_eff < 0:
        return np.full(mean_risk.shape[0], -1)
    ok = mean_risk <= alpha_eff + 1e-12
    idx = ok.argmax(axis=1)
    idx[~ok.any(axis=1)] = -1
    return idx


def coverage_draws(S, n, alpha, n_draws, rng):
    """For one trial/method: coverage and width per random draw, for crc and naive thresholds."""
    R, W, H = S.shape
    U = W * H
    grid = make_grid(S)
    risk = risk_matrix(S, grid)                                         # (R, G)
    order = np.argsort(rng.random((n_draws, R)), axis=1)
    test, cal = order[:, 0], order[:, 1:n + 1]
    mean_risk = risk[cal].mean(axis=1)                                  # (D, G)
    out = {}
    for name, a_eff in (("crc", (alpha * (n + 1) - 1) / n),
                        ("naive", (alpha * (n * U + 1) - 1) / (n * U))):
        idx = first_crossing(mean_risk, a_eff)
        finite = idx >= 0
        cov = np.where(finite, 1.0 - risk[test, np.where(finite, idx, 0)], 1.0)
        width = np.where(finite, 2.0 * grid[np.where(finite, idx, 0)], np.inf)
        out[name] = (cov, width)
    return out


def summarise_cells(per_trial, alpha):
    """per_trial: list of (cov (D,), width (D,)) -> across-trial summary."""
    trial_cov = [float(c.mean()) for c, _ in per_trial]
    trial_below = [float((c < 1 - alpha - 0.10).mean()) for c, _ in per_trial]
    trial_p10 = [float(np.percentile(c, 10)) for c, _ in per_trial]
    fin = [w[np.isfinite(w)] for _, w in per_trial]
    trial_w = [float(f.mean()) for f in fin if len(f)]
    return {"coverage": t_ci(trial_cov),
            "p10_draw_coverage": round(float(np.mean(trial_p10)), 4),
            "frac_draws_more_than_0.10_below_target": round(float(np.mean(trial_below)), 4),
            "width_finite": t_ci(trial_w) if trial_w else None,
            "frac_infinite_width": round(float(np.mean([(~np.isfinite(w)).mean() for _, w in per_trial])), 4)}


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    out = t_ci(d)
    out["trials_a_lower"] = int((d < 0).sum())
    return out


def drift_eval(S, n, alpha, n_draws, rng, L=4, gamma=0.1):
    """Static vs rolling vs rolling-ACI under the sensor-level protocol (windows are time-ordered).

    Calibration sensors keep reporting, so at window t they supply the errors of windows < t.
    static : first-half windows of the n calibration sensors, fixed.
    rolling: the previous L windows of the n calibration sensors.
    aci    : rolling, with the working level a_t updated from the new sensor's miscoverage e_t:
             a_{t+1} = clip(a_t + gamma (alpha - e_t)), gamma fixed in advance.
    Each threshold uses the sensor-level correction with n sensors. Evaluated on windows >= W//2.
    """
    R, W, H = S.shape
    half = W // 2
    late = (2 * W) // 3
    order = np.argsort(rng.random((n_draws, R)), axis=1)
    res = {m: {"all": [], "late": [], "width": []} for m in ("static", "rolling", "aci")}

    def thr(pool, a):
        a_eff = (a * (n + 1) - 1) / n
        if a_eff < 0:
            return np.inf
        pool = np.sort(pool.reshape(-1))
        m = pool.size
        return pool[min(m - 1, max(0, int(np.ceil((1 - a_eff) * m)) - 1))]

    for d in range(n_draws):
        j, cal = order[d, 0], order[d, 1:n + 1]
        q_static = thr(S[cal][:, :half], alpha)
        a_t = alpha
        miss = {m: [] for m in res}
        wid = {m: [] for m in res}
        for t in range(half, W):
            q_roll = thr(S[cal][:, max(0, t - L):t], alpha)
            q_aci = thr(S[cal][:, max(0, t - L):t], a_t)
            for name, q in (("static", q_static), ("rolling", q_roll), ("aci", q_aci)):
                e = float((S[j, t] > q).mean())
                miss[name].append(e)
                wid[name].append(2 * q if np.isfinite(q) else np.nan)
            a_t = float(np.clip(a_t + gamma * (alpha - miss["aci"][-1]), 0.005, 0.995))
        for name in res:
            e = np.array(miss[name])
            ts = np.arange(half, W)
            res[name]["all"].append(1 - e.mean())
            res[name]["late"].append(1 - e[ts >= late].mean())
            w = np.array(wid[name])
            res[name]["width"].append(np.nanmean(w) if np.isfinite(w).any() else np.nan)
    return {k: {"coverage_all": float(np.mean(v["all"])), "coverage_late": float(np.mean(v["late"])),
                "width": float(np.nanmean(v["width"])) if np.isfinite(v["width"]).any() else None,
                "windows_evaluated": int(W - half)} for k, v in res.items()}


def analyse(domain):
    trials = load_trials(domain)
    if not trials:
        return None
    methods = [m for m in FORECAST_METHODS + NOWCAST_METHODS + DROP_METHODS if m in trials[0]]
    R, W, H = trials[0]["full"].shape
    rng = np.random.default_rng(12345)
    out = {"n_trials": len(trials), "n_held_out": R, "windows": W, "horizon": H,
           "points_per_sensor": W * H, "methods": methods}

    mae = {m: [float(t[m].mean()) for t in trials] for m in methods}
    out["mae"] = {m: t_ci(v) for m, v in mae.items()}
    out["mae_paired_trials"] = {}
    for a, b in (("full", "ci"), ("full", "no_gat"), ("full", "persist"), ("full", "winmean"),
                 ("no_gat", "ci"), ("full_legacy", "ci"), ("full", "full_legacy")):
        if a in mae and b in mae:
            out["mae_paired_trials"][f"{a}_minus_{b}"] = paired(mae[a], mae[b])

    out["coverage"] = {}
    for alpha in ALPHAS:
        out["coverage"][str(alpha)] = {}
        for n in CAL_SIZES[domain]:
            if n > R - 1:
                continue
            cell = {}
            for m in [m for m in methods if m in COVERAGE_METHODS]:
                per = {"crc": [], "naive": []}
                for t in trials:
                    d = coverage_draws(t[m], n, alpha, DRAWS[domain], rng)
                    for k in per:
                        per[k].append(d[k])
                cell[m] = {k: summarise_cells(per[k], alpha) for k in per}
            out["coverage"][str(alpha)][str(n)] = cell
    drift_n = min(R - 1, 12)
    if W >= 8 and drift_n >= 9:
        out["drift"] = {"n_cal_sensors": drift_n, "alpha": 0.1, "L": 4, "gamma": 0.1}
        per = [drift_eval(t["full"], drift_n, 0.1, 150, rng) for t in trials]
        out["drift"]["results"] = {k: {kk: t_ci([p[k][kk] for p in per if p[k][kk] is not None])
                                       for kk in ("coverage_all", "coverage_late", "width")}
                                   for k in ("static", "rolling", "aci")}
    return out


if __name__ == "__main__":
    domains = sys.argv[1:] or list(CAL_SIZES)
    path = os.path.join(RES, "exp_D2_analysis.json")
    result = json.load(open(path)) if os.path.exists(path) else {}
    for dom in domains:
        r = analyse(dom)
        if r is None:
            print(f"[{dom}] no D1 trials yet")
            continue
        result[dom] = r
        print(f"[{dom}] trials={r['n_trials']} R={r['n_held_out']} U={r['points_per_sensor']}")
        for alpha in ALPHAS:
            for n, cell in r["coverage"][str(alpha)].items():
                f = cell["full"]
                print(f"  a={alpha} n={n}: crc cov={f['crc']['coverage']['mean']} "
                      f"w={f['crc']['width_finite']['mean'] if f['crc']['width_finite'] else 'inf'} | "
                      f"naive cov={f['naive']['coverage']['mean']} "
                      f"w={f['naive']['width_finite']['mean'] if f['naive']['width_finite'] else 'inf'}")
        json.dump(result, open(path, "w"), indent=1, default=float)
    print("saved", path)

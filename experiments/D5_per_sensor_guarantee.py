#!/usr/bin/env python3
"""D5 - a forecast-time guarantee for every individual new sensor, and a non-random new-sensor test.

Per-sensor adaptive conformal inference (ACI, Gibbs and Candes 2021) without clipping. For a new sensor j and
evaluation windows t = t0 .. W-1 (T = W - t0 windows), the interval half-width for window t is

    q_t = Q(alpha_t),   alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   alpha_{t0} = alpha,

where err_t in [0, 1] is the share of sensor j's (window t, horizon step) points outside the interval, known when
window t closes, which is the forecast origin of window t+1 (windows advance by the horizon). Q is the sensor-level
rule (Corollary 1) applied to the calibration sensors' errors of the previous L windows only, at working level
alpha_t:  Q = +inf if alpha_t (n+1) < 1,  Q = -1 (empty interval, err_t = 1) if alpha_t >= 1, and otherwise the
ceil((1 - a) m)-th smallest of the m pooled errors with a = (alpha_t (n+1) - 1) / n.

Q is non-increasing in alpha_t, is +inf below 1/(n+1) and gives err_t = 1 at alpha_t >= 1, so alpha_t stays in
[-gamma, 1 + gamma] and, telescoping the update, for EVERY sequence of errors

    | (1/T) sum_t err_t - alpha |  <=  (max(alpha, 1 - alpha) + gamma) / (gamma T).

No exchangeability, no random choice of the new sensor and no stationarity is needed; the statement is about the
sensor's own long-run coverage. This script checks the bound for every sensor in every split (it must hold).

Protocol: leave-one-sensor-out within each split's held-out set R; the calibration set is all other sensors of R
(n = |R| - 1), so no random draw is involved. Schemes compared on the same windows:
  rolling-pooled  pooled rule on the previous L windows (no guarantee)
  rolling-sensor  sensor-level rule on the previous L windows (no guarantee at forecast time)
  aci-0.1, aci-0.5  per-sensor ACI as above, gamma fixed in advance at 0.1 and 0.5
Same-window sensor-level (the Theorem 1 setting; not deployable) is reported for reference.

Non-random new sensor: in each split, the held-out sensor farthest from the training network (Haversine distance
to its nearest training sensor, in units of the training network's median nearest-neighbor spacing), chosen before
looking at any error.

Settings fixed before the results were seen: alpha = 0.1, L = 4 windows (daily networks) or 24 (5-minute traffic),
t0 = L, gamma in {0.1, 0.5}.
Run: python3 experiments/D5_per_sensor_guarantee.py      Out: results/exp_D5_per_sensor.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from D1_sensor_level_scores import DOMAINS, haversine_matrix  # noqa: E402
from src.data import DataConfig  # noqa: E402

RES = os.path.join(HERE, "..", "results")
ALPHA = 0.1
GAMMAS = (0.1, 0.5)
L = {"delhi_air_quality": 4, "beijing_air_quality": 4, "ghcn_weather": 4, "pemsd7_traffic": 24,
     "pemsbay_traffic": 24}


def q_level(pool_sorted, a_t, n):
    """Sensor-level threshold at working level a_t; +inf below 1/(n+1), empty (-1) at a_t >= 1."""
    if a_t >= 1.0:
        return -1.0
    miscov = (a_t * (n + 1) - 1.0) / n
    if miscov < 0:
        return np.inf
    m = pool_sorted.size
    idx = min(m - 1, max(0, int(np.ceil((1.0 - miscov) * m)) - 1))
    return float(pool_sorted[idx])


def q_pooled(pool_sorted):
    m = pool_sorted.size
    miscov = (ALPHA * (m + 1) - 1.0) / m
    idx = min(m - 1, max(0, int(np.ceil((1.0 - miscov) * m)) - 1))
    return float(pool_sorted[idx])


def run_sensor(S, j, Lw):
    """All schemes for new sensor j, calibration = all other sensors. S: (R, W, H)."""
    R, W, H = S.shape
    cal = np.array([i for i in range(R) if i != j])
    n = cal.size
    t0 = Lw
    T = W - t0
    res = {}
    # same-window reference (Theorem 1 setting) on the same evaluation windows
    pool_sw = np.sort(S[cal, t0:].reshape(-1))
    q = q_level(pool_sw, ALPHA, n)
    res["same-window"] = {"cov": 1.0 - float((S[j, t0:] > q).mean()) if np.isfinite(q) else 1.0,
                          "inf_share": float(not np.isfinite(q)), "width_med": 2 * q}
    pools = [np.sort(S[cal, max(0, t - Lw):t].reshape(-1)) for t in range(t0, W)]
    for name in ("rolling-pooled", "rolling-sensor"):
        errs, widths = [], []
        for k, t in enumerate(range(t0, W)):
            q = q_pooled(pools[k]) if name == "rolling-pooled" else q_level(pools[k], ALPHA, n)
            errs.append(0.0 if not np.isfinite(q) else float((S[j, t] > q).mean()))
            widths.append(2 * q)
        w = np.array(widths)
        res[name] = {"cov": 1.0 - float(np.mean(errs)), "inf_share": float(np.mean(~np.isfinite(w))),
                     "width_med": float(np.median(w)), "width_mean_finite": float(np.mean(w[np.isfinite(w)]))
                     if np.isfinite(w).any() else None}
    for g in GAMMAS:
        a_t, errs, widths, a_path = ALPHA, [], [], []
        for k, t in enumerate(range(t0, W)):
            q = q_level(pools[k], a_t, n)
            e = 0.0 if q == np.inf else (1.0 if q < 0 else float((S[j, t] > q).mean()))
            errs.append(e)
            widths.append(2 * q if q >= 0 else 0.0)
            a_path.append(a_t)
            a_t = a_t + g * (ALPHA - e)
        bound = (max(ALPHA, 1 - ALPHA) + g) / (g * T)
        miscov = float(np.mean(errs))
        w = np.array(widths)
        res[f"aci-{g}"] = {"cov": 1.0 - miscov, "bound": bound, "guaranteed_cov": max(0.0, 1 - ALPHA - bound),
                           "bound_holds": bool(abs(miscov - ALPHA) <= bound + 1e-12),
                           "inf_share": float(np.mean(~np.isfinite(w))), "empty_share": float(np.mean(w == 0)),
                           "width_med": float(np.median(w)),
                           "width_mean_finite": float(np.mean(w[np.isfinite(w)])) if np.isfinite(w).any() else None,
                           "alpha_range": [float(min(a_path)), float(max(a_path))]}
    return res, T


def distances(domain, held, train):
    spec = DOMAINS[domain]
    cfg = DataConfig(window_size=spec["window"], horizon=spec["horizon"], n_features=1)
    coords = spec["prepare"](cfg, hold_out_stations=[int(i) for i in held])[3]["coords_all"]
    D = haversine_matrix(coords)
    Dtt = D[np.ix_(train, train)].copy()
    np.fill_diagonal(Dtt, np.inf)
    spacing = float(np.median(Dtt.min(axis=1)))
    return D[np.ix_(held, train)].min(axis=1) / spacing


def summarise(per_sensor, key):
    covs = np.array([p[key]["cov"] for p in per_sensor])
    out = {"mean_cov": round(float(covs.mean()), 4), "min_cov": round(float(covs.min()), 4),
           "p05_cov": round(float(np.percentile(covs, 5)), 4), "share_below_0.80": round(float((covs < 0.80).mean()), 4),
           "share_below_0.85": round(float((covs < 0.85).mean()), 4),
           "inf_share": round(float(np.mean([p[key]["inf_share"] for p in per_sensor])), 4),
           "width_med": round(float(np.median([p[key]["width_med"] for p in per_sensor])), 4)}
    if key.startswith("aci"):
        out["bound"] = round(per_sensor[0][key]["bound"], 4)
        out["guaranteed_cov"] = round(per_sensor[0][key]["guaranteed_cov"], 4)
        out["bound_holds_all"] = all(p[key]["bound_holds"] for p in per_sensor)
        out["empty_share"] = round(float(np.mean([p[key]["empty_share"] for p in per_sensor])), 4)
    return out


if __name__ == "__main__":
    result = {}
    schemes = ["same-window", "rolling-pooled", "rolling-sensor"] + [f"aci-{g}" for g in GAMMAS]
    for dom in DOMAINS:
        files = sorted(glob.glob(os.path.join(RES, "D1b", f"{dom}_trial*.npz")))
        per_sensor, far, all_d, Tval = [], [], [], None
        for f in files:
            z = np.load(f, allow_pickle=True)
            S = z["S_full"].astype(np.float64)
            held, train = z["held"].astype(int), z["train"].astype(int)
            d = distances(dom, held, train)
            for j in range(S.shape[0]):
                r, Tval = run_sensor(S, j, L[dom])
                r["distance"] = float(d[j])
                per_sensor.append(r)
            far.append(per_sensor[-S.shape[0] + int(np.argmax(d))])
        out = {"n_trials": len(files), "n_sensor_cases": len(per_sensor), "eval_windows": Tval, "L": L[dom],
               "all_sensors": {s: summarise(per_sensor, s) for s in schemes},
               "farthest_sensor": {s: {"mean_cov": round(float(np.mean([p[s]["cov"] for p in far])), 4),
                                       "min_cov": round(float(np.min([p[s]["cov"] for p in far])), 4)} for s in schemes},
               "farthest_mean_distance": round(float(np.mean([p["distance"] for p in far])), 3),
               "all_mean_distance": round(float(np.mean([p["distance"] for p in per_sensor])), 3)}
        dist = np.array([p["distance"] for p in per_sensor])
        for s in ("rolling-sensor", "aci-0.1"):
            rho = stats.spearmanr(dist, [p[s]["cov"] for p in per_sensor])
            out[f"spearman_cov_vs_distance_{s}"] = [round(float(rho.statistic), 3), round(float(rho.pvalue), 4)]
        result[dom] = out
        print(f"\n{dom}: splits {len(files)}, sensor cases {len(per_sensor)}, evaluation windows T={Tval}")
        for s in schemes:
            a = out["all_sensors"][s]
            extra = (f" bound {a['bound']:.3f} (guaranteed {a['guaranteed_cov']:.3f}) holds={a['bound_holds_all']}"
                     f" empty {a['empty_share']:.3f}") if s.startswith("aci") else ""
            print(f"   {s:15s} mean {a['mean_cov']:.3f} min {a['min_cov']:.3f} p05 {a['p05_cov']:.3f} "
                  f"<0.80 {a['share_below_0.80']:.3f} inf {a['inf_share']:.3f} med-width {a['width_med']:.4g}{extra}")
        print("   farthest sensor:", {s: out["farthest_sensor"][s]["mean_cov"] for s in schemes},
              "distance", out["farthest_mean_distance"], "vs all", out["all_mean_distance"])
    json.dump(result, open(os.path.join(RES, "exp_D5_per_sensor.json"), "w"), indent=1)
    print("saved exp_D5_per_sensor.json")

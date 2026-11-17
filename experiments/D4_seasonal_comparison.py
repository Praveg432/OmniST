#!/usr/bin/env python3
"""D4 - OmniST, the coordinate-free GRU, persistence and seasonal-naive on the SAME windows.

Seasonal-naive (D1c) exists only for windows with enough history, so every method is restricted to windows
>= first_valid_window (0 for the daily networks, 20 for the 5-minute traffic networks). Reports mean absolute
error, paired differences against seasonal-naive across splits, and the interval width at the 90% target for
the largest calibration size, under the pooled rule and the sensor-level rule (same-window calibration).

Run: python3 experiments/D4_seasonal_comparison.py     Out: results/exp_D4_seasonal.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D2_analyse import CAL_SIZES, coverage_draws, t_ci  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ALPHA, DRAWS = 0.1, {"delhi_air_quality": 2000, "beijing_air_quality": 2000, "ghcn_weather": 2000,
                     "pemsd7_traffic": 600, "pemsbay_traffic": 600}
METHODS = ("full", "ci", "persist", "seasonal")


def load(domain):
    trials = []
    for fb in sorted(glob.glob(os.path.join(RES, "D1b", f"{domain}_trial*.npz"))):
        name = os.path.basename(fb)
        z1 = np.load(os.path.join(RES, "D1", name), allow_pickle=True)
        zb = np.load(fb, allow_pickle=True)
        zc = np.load(os.path.join(RES, "D1c", name), allow_pickle=True)
        w0 = int(zc["first_valid_window"])
        trials.append({"full": zb["S_full"][:, w0:], "ci": z1["S_ci"][:, w0:], "persist": z1["S_persist"][:, w0:],
                       "seasonal": zc["S_seasonal"][:, w0:], "w0": w0})
    return trials


if __name__ == "__main__":
    out = {}
    for dom in CAL_SIZES:
        trials = load(dom)
        n = max(CAL_SIZES[dom])
        R = trials[0]["full"].shape[0]
        n = min(n, R - 1)
        rng = np.random.default_rng(99)
        res = {"n_trials": len(trials), "first_valid_window": trials[0]["w0"], "n_cal": n,
               "windows_used": trials[0]["full"].shape[1], "methods": {}}
        mae = {m: [float(np.nanmean(t[m])) for t in trials] for m in METHODS}
        for m in METHODS:
            per = {"crc": [], "naive": []}
            for t in trials:
                d = coverage_draws(t[m].astype(np.float64), n, ALPHA, DRAWS[dom], rng)
                for k in per:
                    per[k].append(d[k])
            def w(rule):
                fin = [x[1][np.isfinite(x[1])] for x in per[rule]]
                return float(np.mean([f.mean() for f in fin])) if all(len(f) for f in fin) else None
            res["methods"][m] = {"mae": t_ci(mae[m]), "pooled_width": w("naive"), "sensor_width": w("crc"),
                                 "pooled_cov": float(np.mean([x[0].mean() for x in per["naive"]]))}
        res["paired_mae_vs_seasonal"] = {
            m: t_ci(np.array(mae[m]) - np.array(mae["seasonal"])) | {"splits_lower": int((np.array(mae[m]) < np.array(mae["seasonal"])).sum())}
            for m in ("full", "ci", "persist")}
        out[dom] = res
        s = res["methods"]
        print(f"{dom}: n={n} windows={res['windows_used']} (from {res['first_valid_window']})")
        for m in METHODS:
            print(f"   {m:9s} MAE {s[m]['mae']['mean']:.4g}   pooled width {s[m]['pooled_width']:.4g}   "
                  f"sensor-level width {s[m]['sensor_width'] if s[m]['sensor_width'] is None else round(s[m]['sensor_width'], 4)}")
        for m, v in res["paired_mae_vs_seasonal"].items():
            print(f"   MAE {m} - seasonal: {v['mean']:.3g} [{v['ci95'][0]:.3g}, {v['ci95'][1]:.3g}]  lower in {v['splits_lower']}/{v['n']}")
    json.dump(out, open(os.path.join(RES, "exp_D4_seasonal.json"), "w"), indent=1, default=float)
    print("saved exp_D4_seasonal.json")

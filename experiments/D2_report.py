#!/usr/bin/env python3
"""Print the numbers that the manuscript tables are built from, straight from exp_D2_analysis.json."""
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
d = json.load(open(os.path.join(RES, "exp_D2_analysis.json")))


def ci(v, nd=3):
    lo, hi = v["ci95"]
    return f"{v['mean']:.{nd}g} [{lo:.{nd}g},{hi:.{nd}g}]" if lo is not None else f"{v['mean']:.{nd}g}"


for dom, r in d.items():
    print(f"\n== {dom}: trials={r['n_trials']} held-out={r['n_held_out']} windows={r['windows']} "
          f"U={r['points_per_sensor']}")
    print(" MAE (mean over splits, 95% t-CI):")
    for m in ("full", "no_gat", "ci", "full_legacy", "persist", "winmean", "nn_conc", "idw_conc"):
        if m in r["mae"]:
            print(f"   {m:12s} {ci(r['mae'][m])}")
    print(" paired MAE differences (a - b), splits where a is lower:")
    for k, v in r["mae_paired_trials"].items():
        print(f"   {k:24s} {ci(v)}  a_lower {v['trials_a_lower']}/{v['n']}")
    n_big = str(max(int(n) for n in r["coverage"]["0.1"]))
    for alpha in ("0.1", "0.2"):
        cell = r["coverage"][alpha].get(n_big)
        if not cell:
            continue
        print(f" alpha={alpha}, n={n_big} calibration sensors:")
        for m, c in cell.items():
            crc, nv = c["crc"], c["naive"]
            wc = crc["width_finite"]["mean"] if crc["width_finite"] else float("nan")
            print(f"   {m:12s} sensor-level cov {crc['coverage']['mean']:.4f} w {wc:.4g} inf {crc['frac_infinite_width']:.2f}"
                  f" | pooled cov {nv['coverage']['mean']:.4f} [{nv['coverage']['ci95'][0]},{nv['coverage']['ci95'][1]}]"
                  f" w {nv['width_finite']['mean']:.4g} p10 {nv['p10_draw_coverage']:.3f}"
                  f" frac>0.10below {nv['frac_draws_more_than_0.10_below_target']:.3f}")
    if "drift" in r:
        print(f" drift (n_cal={r['drift']['n_cal_sensors']}, windows evaluated:",
              {k: v["coverage_all"]["mean"] for k, v in r["drift"]["results"].items()}, ")")
        for k, v in r["drift"]["results"].items():
            w = v["width"]
            print(f"   {k:8s} all {ci(v['coverage_all'])}  late {ci(v['coverage_late'])}  width {ci(w) if w else None}")

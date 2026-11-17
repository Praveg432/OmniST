#!/usr/bin/env python3
"""Print a summary table of the C1 rigorous-evaluation results (one row per domain).

Reads results/exp_C1_<domain>.json. Run: python experiments/C1_summarise.py
"""
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DOMAINS = ["delhi_air_quality", "beijing_air_quality", "ghcn_weather", "pemsd7_traffic", "pemsbay_traffic"]


def fmt(ci: dict) -> str:
    lo, hi = ci["ci95"]
    return f"{ci['mean']:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{ci['mean']:.3f}"


def main() -> None:
    print("| Dataset | Stations (train/cal/test) | Splits | Coverage, disjoint calibration | "
          "Coverage, training-station calibration | Per-station coverage: mean / 10th pct / min |")
    print("|---|---|:---:|---|---|---|")
    for dom in DOMAINS:
        path = os.path.join(RESULTS, f"exp_C1_{dom}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        a, s0 = d["aggregate"], d["splits"][0]
        dis, leg = a["disjoint_calibration"], a["legacy_train_station_calibration"]
        ps = dis["per_station_coverage"]
        print(f"| {dom} | {s0['n_train']}/{s0['n_cal']}/{s0['n_test']} | {len(d['splits'])} | "
              f"{fmt(dis['coverage_over_splits'])} | {fmt(leg['coverage_over_splits'])} | "
              f"{ps['mean']:.3f} / {ps['p10']:.3f} / {ps['min']:.3f} |")
    print()
    for dom in DOMAINS:
        path = os.path.join(RESULTS, f"exp_C1_{dom}.json")
        if not os.path.exists(path):
            continue
        a = json.load(open(path))["aggregate"]
        dis = a["disjoint_calibration"]
        line = (f"{dom}: sd over splits {dis['coverage_over_splits']['sd']}, range "
                f"[{dis['coverage_over_splits']['min']}, {dis['coverage_over_splits']['max']}], "
                f"splits at or above 0.90: {dis['fraction_of_splits_at_or_above_target']}, "
                f"width {dis['width_over_splits']['mean']}, test MAE {a['test_mae_over_splits']['mean']}")
        print(line)
        if "year_feature_set_to_latest_training_year" in a:
            y = a["year_feature_set_to_latest_training_year"]
            print(f"   year-feature check: coverage {y['coverage_over_splits']['mean']}, "
                  f"MAE {y['test_mae_over_splits']['mean']} (vs {a['test_mae_over_splits']['mean']} "
                  f"with the original encoding)")


if __name__ == "__main__":
    main()

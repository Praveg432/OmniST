#!/usr/bin/env python3
"""Check the numbers in the manuscript tables against the committed result files.

For each table in paper/jsen-submission/part3_results.tex the script parses the printed cells and
compares them with the value computed from results/exp_D2_analysis.json or results/exp_C1_*.json,
allowing only the rounding that the printed number itself implies. It exits non-zero on any mismatch.

Run: python3 paper/check_numbers.py
"""
import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RES = PAPER.parent / "results"
TEX = (PAPER / "jsen-submission" / "part3_results.tex").read_text()
FULLTEXT = (PAPER / "jsen-submission" / "OmniST-JSEN.tex").read_text()
D2 = json.load(open(RES / "exp_D2_analysis.json"))
DS = ["delhi_air_quality", "beijing_air_quality", "ghcn_weather", "pemsd7_traffic", "pemsbay_traffic"]
failures, checked = [], 0


def block(label):
    start = TEX.index("\\label{%s}" % label)
    ends = [i for i in (TEX.find("\\end{table}", start), TEX.find("\\end{table*}", start)) if i != -1]
    return TEX[start:min(ends)]


def rows(label):
    out = []
    for line in block(label).split("\n"):
        if line.rstrip().endswith("\\\\") and "&" in line and "\\multicolumn" not in line:
            out.append([c.strip().rstrip("\\").strip() for c in line.rstrip().rstrip("\\").split("&")])
    return out


def num(s):
    s = s.replace("$", "").replace("\\%", "").strip()
    return float(s)


def close(printed, value, what):
    """printed string vs true value, within the rounding the printed number implies."""
    global checked
    checked += 1
    dec = len(printed.split(".")[1].rstrip("%\\ ")) if "." in printed else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    if abs(num(printed) - value) > tol:
        failures.append(f"{what}: printed {printed}, computed {value:.6g}")


def cell(key, alpha, n, method="full"):
    return D2[key]["coverage"][alpha][str(n)][method]


# ---- Table: bound ---------------------------------------------------------------------------------
plan = [("delhi_air_quality", 9), ("delhi_air_quality", 15), ("beijing_air_quality", 5), ("ghcn_weather", 9),
        ("ghcn_weather", 11), ("pemsd7_traffic", 20), ("pemsd7_traffic", 59), ("pemsbay_traffic", 20),
        ("pemsbay_traffic", 89)]
largest = {k: max(int(n) for n in D2[k]["coverage"]["0.1"]) for k in DS}
bound_rows = [r for r in rows("tab:bound") if len(r) == 8 and r[1].isdigit()]
assert len(bound_rows) == len(plan), (len(bound_rows), len(plan))
for r, (key, n) in zip(bound_rows, plan):
    assert int(r[1]) == n, (r, n)
    c = cell(key, "0.1", n)
    what = f"bound {key} n={n}"
    if "vacuous" in r[2]:
        assert c["crc"]["frac_infinite_width"] > 0.999, what
    else:
        close(r[2], c["crc"]["coverage"]["mean"], what + " crc cov")
        close(r[3], c["crc"]["width_finite"]["mean"], what + " crc width")
        close(r[6], c["crc"]["width_finite"]["mean"] / c["naive"]["width_finite"]["mean"], what + " ratio")
    close(r[4], c["naive"]["coverage"]["mean"], what + " pooled cov")
    close(r[5], c["naive"]["width_finite"]["mean"], what + " pooled width")
    if n == largest[key]:
        close(r[7].replace("\\%", ""), 100 * c["naive"]["frac_draws_more_than_0.10_below_target"],
              what + " draws<0.80")

# ---- Table: accuracy ------------------------------------------------------------------------------
methods = [("full", "OmniST"), ("no_gat", "No attention"), ("ci", "Coordinate-free"), ("persist", "Persistence"),
           ("winmean", "Window mean"), ("full_legacy", "earlier code"), ("idw_conc", "IDW")]
acc_rows = [r for r in rows("tab:accuracy") if len(r) == 6 and re.fullmatch(r"[\d.]+|---", r[1])]
assert len(acc_rows) == len(methods), (len(acc_rows), len(methods))
for r, (m, name) in zip(acc_rows, methods):
    for key, cellv in zip(DS, r[1:]):
        if cellv.strip() == "---":
            assert m not in D2[key]["mae"], f"{m} exists for {key} but table shows a dash"
        else:
            close(cellv, D2[key]["mae"][m]["mean"], f"accuracy {m} {key}")

# ---- Table II: per-sensor forecast-time coverage (D5) ---------------------------------------------
D5 = json.load(open(RES / "exp_D5_per_sensor.json"))
dep_key = {"Delhi AQ": "delhi_air_quality", "Beijing AQ": "beijing_air_quality", "Weather": "ghcn_weather",
           "PeMSD7-M": "pemsd7_traffic", "PeMS-BAY": "pemsbay_traffic"}
dep_rows = [r for r in rows("tab:deployable") if len(r) == 14 and r[0] in dep_key]
assert len(dep_rows) == 5, len(dep_rows)
for r in dep_rows:
    key = dep_key[r[0]]
    A = D5[key]["all_sensors"]
    assert int(r[1]) == D5[key]["eval_windows"], (r[0], r[1])
    cols = [("rolling-pooled", "mean_cov", r[2]), ("rolling-pooled", "min_cov", r[3]),
            ("rolling-sensor", "mean_cov", r[4]), ("rolling-sensor", "min_cov", r[5])]
    for g, off in (("aci-0.1", 6), ("aci-0.5", 10)):
        cols += [(g, "mean_cov", r[off]), (g, "min_cov", r[off + 1]), (g, "inf_share", r[off + 2]),
                 (g, "guaranteed_cov", r[off + 3])]
        assert A[g]["bound_holds_all"], f"Theorem 2 bound violated for {key} {g}"
    for scheme, field, printed in cols:
        if printed in ("vac.", "---"):
            if printed == "vac.":
                assert A[scheme]["inf_share"] > 0.999, (key, scheme)
            else:
                assert A[scheme]["guaranteed_cov"] < 1e-9, (key, scheme)
            continue
        close(printed, A[scheme][field], f"Table II {key} {scheme} {field}")

# ---- Prose: forecast-time worst sensors, farthest sensors, earlier protocol ----------------------
FT = FULLTEXT
def need(phrase):
    if phrase not in FT:
        failures.append(f"prose: expected phrase not found: {phrase}")
far = {k: D5[k]["farthest_sensor"]["aci-0.1"] for k in D5}
for printed, val, what in (
        ("0.904", far["pemsd7_traffic"]["mean_cov"], "far mean D7"), ("0.897", far["pemsbay_traffic"]["mean_cov"], "far mean BAY"),
        ("0.822", far["delhi_air_quality"]["min_cov"], "far worst Delhi"), ("0.904", far["ghcn_weather"]["min_cov"], "far worst weather"),
        ("0.898", far["pemsd7_traffic"]["min_cov"], "far worst D7"), ("0.892", far["pemsbay_traffic"]["min_cov"], "far worst BAY")):
    close(printed, val, f"prose {what}")
ratios = [D5[k]["farthest_mean_distance"] / D5[k]["all_mean_distance"] for k in D5]
close("2.1", min(ratios), "far/all distance ratio min"); close("5.8", max(ratios), "far/all distance ratio max")
dists = [D5[k]["farthest_mean_distance"] for k in D5]
close("2.4", min(dists), "far distance min"); close("11.8", max(dists), "far distance max")
need("2.1 to 5.8 times"); need("2.4 to 11.8 median spacings")
pooled_worst = [D5[k]["all_sensors"]["rolling-pooled"]["min_cov"] for k in D5]
close("0.53", min(pooled_worst), "pooled worst min"); close("0.81", max(pooled_worst), "pooled worst max")
need("0.53 to 0.81")
sens_worst = [D5[k]["all_sensors"]["rolling-sensor"]["min_cov"] for k in D5
              if D5[k]["all_sensors"]["rolling-sensor"]["inf_share"] < 0.999]
close("0.58", min(sens_worst), "sensor worst min"); close("0.78", max(sens_worst), "sensor worst max")
need("0.58 to 0.78")
close("1.9", 100 * D5["delhi_air_quality"]["all_sensors"]["aci-0.1"]["share_below_0.80"], "Delhi ACI <0.80 share"); need("1.9\\% of cases")
c1 = {k: json.load(open(RES / f"exp_C1_{k}.json"))["aggregate"] for k in
      ("delhi_air_quality", "beijing_air_quality", "ghcn_weather", "pemsd7_traffic", "pemsbay_traffic")}
means = [v["disjoint_calibration"]["coverage_over_splits"]["mean"] for v in c1.values()]
worst = [v["disjoint_calibration"]["per_station_coverage"]["min"] for v in c1.values()]
close("0.876", min(means), "C1 mean min"); close("0.906", max(means), "C1 mean max")
close("0.40", min(worst), "C1 worst min"); close("0.86", max(worst), "C1 worst max")
need("0.876 to 0.906"); need("0.40 to 0.86")

# ---- Prose claims about seasonal-naive (D4) --------------------------------------------------------
D4 = json.load(open(RES / "exp_D4_seasonal.json"))
FULL = FULLTEXT
bay = D4["pemsbay_traffic"]["methods"]
claims = [
    ("MAE 3.56 against 4.68", [(bay["seasonal"]["mae"]["mean"], "3.56"), (bay["full"]["mae"]["mean"], "4.68")]),
    ("15.6 wide against 20.5", [(bay["seasonal"]["pooled_width"], "15.6"), (bay["full"]["pooled_width"], "20.5")]),
    ("intervals are 31\\% wider", [(100 * (bay["full"]["pooled_width"] / bay["seasonal"]["pooled_width"] - 1), "31")]),
    ("intervals 24\\% narrower", [(100 * (1 - bay["seasonal"]["pooled_width"] / bay["full"]["pooled_width"]), "24")]),
]
for phrase, pairs in claims:
    if phrase not in FULL:
        failures.append(f"prose: expected phrase not found: {phrase}")
        continue
    for value, printed in pairs:
        close(printed, value, f"prose '{phrase}'")
for dom, printed in (("delhi_air_quality", "$-9.5$"), ("beijing_air_quality", "$-15.2$"),
                     ("ghcn_weather", "$-0.24$"), ("pemsd7_traffic", "$-0.74$"), ("pemsbay_traffic", "$+1.12$")):
    v = D4[dom]["paired_mae_vs_seasonal"]["full"]["mean"]
    if printed not in TEX:
        failures.append(f"prose: {printed} for {dom} not found")
    close(printed.replace("$", "").replace("+", ""), v, f"seasonal paired {dom}")

print(f"checked {checked} printed numbers")
if failures:
    print(f"{len(failures)} MISMATCH(ES):")
    for f in failures:
        print("  ", f)
    sys.exit(1)
print("all checked numbers match the result files")

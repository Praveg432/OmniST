#!/usr/bin/env python3
"""Generate the supplementary tables from results/exp_D2_analysis.json (no number is typed by hand).

Output: paper/supplement_tables.tex, \\input by OmniST-Sensors-Supplement.tex.
Run:    python3 paper/make_supplement_tables.py
"""
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
d = json.load(open(PAPER.parent / "results" / "exp_D2_analysis.json"))

NAMES = [("delhi_air_quality", "Delhi AQ"), ("beijing_air_quality", "Beijing AQ"),
         ("ghcn_weather", "Weather"), ("pemsd7_traffic", "PeMSD7-M"), ("pemsbay_traffic", "PeMS-BAY")]
METHOD_LABEL = {"full": "OmniST", "ci": "Coordinate-free", "full_legacy": "OmniST, earlier code",
                "persist": "Persistence", "nn_conc": "Nearest neighbor, concurrent",
                "full_drop30": "OmniST, 30\\% removed", "full_drop50": "OmniST, 50\\% removed"}


def f(x, nd=4):
    return f"{x:.{nd}f}"


def width(cell):
    w = cell["width_finite"]
    return "$\\infty$" if (w is None or cell["frac_infinite_width"] > 0.999) else f"{w['mean']:.4g}"


def esc(s):
    return s.replace("_", "\\_")


out = []

# ---- S1: coverage of OmniST for every n, both alphas, both rules --------------------------------
out.append("\\begin{table*}[htbp]\n\\caption{Coverage and width of OmniST at every calibration size and both "
           "targets. Means over splits; the sensor-level rule is Theorem 1, the pooled rule the usual "
           "split-conformal quantile. $\\infty$: the sensor-level interval is vacuous.}\n"
           "\\label{tab:fullcov}\n\\centering\\small\n\\begin{tabular}{llrcccccc}\n\\toprule\n"
           " & & & \\multicolumn{2}{c}{\\textbf{Sensor-level}} & \\multicolumn{4}{c}{\\textbf{Pooled}} \\\\\n"
           "\\textbf{Dataset} & $1-\\alpha$ & $n$ & Cov. & Width & Cov. & Width & 10th pct. & Share $<$ target$-$0.10 \\\\\n"
           "\\midrule")
for key, label in NAMES:
    if key not in d:
        continue
    first = True
    for alpha in ("0.1", "0.2"):
        for n, cell in sorted(d[key]["coverage"][alpha].items(), key=lambda kv: int(kv[0])):
            c = cell["full"]
            out.append(f"{label if first else ''} & {1 - float(alpha):.1f} & {n} & "
                       f"{f(c['crc']['coverage']['mean'])} & {width(c['crc'])} & "
                       f"{f(c['naive']['coverage']['mean'])} & {c['naive']['width_finite']['mean']:.4g} & "
                       f"{f(c['naive']['p10_draw_coverage'], 3)} & "
                       f"{100 * c['naive']['frac_draws_more_than_0.10_below_target']:.1f}\\% \\\\")
            first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table*}\n")

# ---- S2: all methods at the largest n, alpha = 0.1 ---------------------------------------------
out.append("\\begin{table*}[htbp]\n\\caption{All methods at the largest calibration size, 90\\% target. "
           "Coverage and width under the sensor-level and the pooled rule. The concurrent interpolator uses "
           "the neighbors' values during the target window and is a nowcasting reference.}\n"
           "\\label{tab:allmethods}\n\\centering\\small\n\\begin{tabular}{llrcccc}\n\\toprule\n"
           " & & & \\multicolumn{2}{c}{\\textbf{Sensor-level}} & \\multicolumn{2}{c}{\\textbf{Pooled}} \\\\\n"
           "\\textbf{Dataset} & \\textbf{Method} & $n$ & Cov. & Width & Cov. & Width \\\\\n\\midrule")
for key, label in NAMES:
    if key not in d:
        continue
    n_big = str(max(int(n) for n in d[key]["coverage"]["0.1"]))
    cell = d[key]["coverage"]["0.1"][n_big]
    first = True
    for m, mlabel in METHOD_LABEL.items():
        if m not in cell:
            continue
        c = cell[m]
        out.append(f"{label if first else ''} & {mlabel} & {n_big} & {f(c['crc']['coverage']['mean'])} & "
                   f"{width(c['crc'])} & {f(c['naive']['coverage']['mean'])} & "
                   f"{c['naive']['width_finite']['mean']:.4g} \\\\")
        first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table*}\n")

# ---- S3: paired MAE differences ----------------------------------------------------------------
out.append("\\begin{table*}[htbp]\n\\caption{Paired differences in MAE between methods (first minus second), "
           "mean over splits with a 95\\% $t$-interval, and the number of splits in which the first method "
           "is lower. Negative means the first method is more accurate.}\n"
           "\\label{tab:paired}\n\\centering\\small\n\\begin{tabular}{llccc}\n\\toprule\n"
           "\\textbf{Dataset} & \\textbf{Comparison} & \\textbf{Difference} & \\textbf{95\\% interval} & \\textbf{First lower} \\\\\n"
           "\\midrule")
PAIR_LABEL = {"full_minus_ci": "OmniST $-$ coordinate-free", "full_minus_no_gat": "OmniST $-$ no attention",
              "no_gat_minus_ci": "No attention $-$ coordinate-free", "full_minus_persist": "OmniST $-$ persistence",
              "full_minus_winmean": "OmniST $-$ window mean", "full_legacy_minus_ci": "Earlier code $-$ coordinate-free",
              "full_minus_full_legacy": "OmniST $-$ earlier code"}
for key, label in NAMES:
    if key not in d:
        continue
    first = True
    for k, lab in PAIR_LABEL.items():
        v = d[key]["mae_paired_trials"].get(k)
        if v is None:
            continue
        lo, hi = v["ci95"]
        out.append(f"{label if first else ''} & {lab} & {v['mean']:.3g} & [{lo:.3g}, {hi:.3g}] & "
                   f"{v['trials_a_lower']}/{v['n']} \\\\")
        first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table*}\n")

# ---- S-removal: reference sensors removed at inference ------------------------------------------
out.append("\\begin{table}[htbp]\n\\caption{MAE and coverage after removing 30\\% or 50\\% of the reference sensors at "
           "inference (rescored and recalibrated on the survivors), largest calibration size, 90\\% target. Coverage: "
           "sensor-level rule / pooled rule.}\n\\label{tab:removalfull}\n\\centering\\small\n"
           "\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Dataset} & \\textbf{Removed} & \\textbf{MAE} & "
           "\\textbf{Sensor-level cov.} & \\textbf{Pooled cov.} & $n$ \\\\\n\\midrule")
for key, label in NAMES:
    if key not in d or "full_drop30" not in d[key]["mae"]:
        continue
    n_big = str(max(int(n) for n in d[key]["coverage"]["0.1"]))
    cell = d[key]["coverage"]["0.1"][n_big]
    if "full_drop30" not in cell:
        continue
    first = True
    for tag, m in (("0\\%", "full"), ("30\\%", "full_drop30"), ("50\\%", "full_drop50")):
        out.append(f"{label if first else ''} & {tag} & {d[key]['mae'][m]['mean']:.4g} & "
                   f"{f(cell[m]['crc']['coverage']['mean'], 3)} & {f(cell[m]['naive']['coverage']['mean'], 3)} & "
                   f"{n_big if first else ''} \\\\")
        first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table}\n")

# ---- S-seasonal: OmniST against seasonal-naive on the same windows ------------------------------
d4 = json.load(open(PAPER.parent / "results" / "exp_D4_seasonal.json"))
out.append("\\begin{table*}[htbp]\n\\caption{OmniST, the coordinate-free GRU, persistence and seasonal-naive on the same "
           "windows (those with a seasonal-naive forecast: all for the daily networks, windows 20 onward for PeMSD7-M, "
           "all for PeMS-BAY, whose lag is one day of 48 half-hour steps). MAE, and the 90\\% interval width under the pooled and the sensor-level rule "
           "(same-window calibration) at the largest calibration size. The last column is OmniST minus "
           "seasonal-naive MAE, mean and 95\\% interval over splits.}\n\\label{tab:seasonalfull}\n\\centering\\small\n"
           "\\begin{tabular}{llcccl}\n\\toprule\n\\textbf{Dataset} & \\textbf{Method} & \\textbf{MAE} & "
           "\\textbf{Pooled width} & \\textbf{Sensor-level width} & \\textbf{OmniST $-$ seasonal MAE} \\\\\n\\midrule")
ML = {"full": "OmniST", "ci": "Coordinate-free GRU", "persist": "Persistence", "seasonal": "Seasonal-naive"}
for key, label in NAMES:
    if key not in d4:
        continue
    r = d4[key]
    first = True
    for m, ml in ML.items():
        sw = r["methods"][m]["sensor_width"]
        pv = r["paired_mae_vs_seasonal"].get("full")
        last = (f"{pv['mean']:.3g} [{pv['ci95'][0]:.3g}, {pv['ci95'][1]:.3g}]" if (first and pv) else "")
        out.append(f"{label if first else ''} & {ml} & {r['methods'][m]['mae']['mean']:.4g} & "
                   f"{r['methods'][m]['pooled_width']:.4g} & {('$\\infty$' if sw is None else f'{sw:.4g}')} & {last} \\\\")
        first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table*}\n")

# ---- S-C1: earlier protocol vs random disjoint splits -------------------------------------------
out.append("\\begin{table}[htbp]\n\\caption{Coverage over random disjoint sensor splits (experiment C1, target 0.90, pooled "
           "rule, same-window calibration): mean with 95\\% interval over splits, the same trained models calibrated "
           "on training sensors, and the worst per-sensor coverage under disjoint calibration.}\n\\label{tab:legacyfull}\n"
           "\\centering\\small\n\\begin{tabular}{lrccc}\n\\toprule\n\\textbf{Dataset} & \\textbf{Splits} & "
           "\\textbf{Disjoint calibration} & \\textbf{Training-sensor calibration} & \\textbf{Worst sensor} \\\\\n\\midrule")
for key, label in NAMES:
    a = json.load(open(PAPER.parent / "results" / f"exp_C1_{key}.json"))["aggregate"]
    dc, lg = a["disjoint_calibration"], a["legacy_train_station_calibration"]
    c = dc["coverage_over_splits"]
    out.append(f"{label} & {c['n']} & {c['mean']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] & "
               f"{lg['coverage_over_splits']['mean']:.3f} & {dc['per_station_coverage']['min']:.3f} \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- S-D5: per-sensor forecast-time results in full ----------------------------------------------
d5 = json.load(open(PAPER.parent / "results" / "exp_D5_per_sensor.json"))
SCH = [("same-window", "Same-window sensor-level$^*$"), ("rolling-pooled", "Rolling, pooled"),
       ("rolling-sensor", "Rolling, sensor-level"), ("aci-0.1", "Per-sensor ACI, $\\gamma=0.1$"),
       ("aci-0.5", "Per-sensor ACI, $\\gamma=0.5$")]
out.append("\\begin{table*}[htbp]\n\\caption{Per-sensor coverage, every held-out sensor of every split as the new sensor "
           "(calibration: all other held-out sensors). Mean, 5th percentile and worst coverage over sensor cases, share of "
           "cases below 0.80, share of windows with an infinite interval, median width over sensor cases, Theorem~2's "
           "certified coverage, and mean coverage of the farthest held-out sensor of each split. $^*$Not deployable "
           "(calibrates on the evaluated windows); shown for reference.}\n\\label{tab:d5full}\n\\centering\\scriptsize\n"
           "\\begin{tabular}{llcccccccc}\n\\toprule\n\\textbf{Dataset} & \\textbf{Scheme} & \\textbf{Mean} & \\textbf{5th pct.} & "
           "\\textbf{Worst} & \\textbf{$<0.80$} & \\textbf{Inf.} & \\textbf{Median width} & \\textbf{Certified} & "
           "\\textbf{Farthest} \\\\\n\\midrule")
for key, label in NAMES:
    r = d5[key]
    first = True
    for sk, sl in SCH:
        a = r["all_sensors"][sk]
        cert = f"{a['guaranteed_cov']:.3f}" if sk.startswith("aci") and a["guaranteed_cov"] > 0 else ("vacuous" if sk.startswith("aci") else "---")
        wm = a["width_med"]
        wtxt = "$\\infty$" if wm == float("inf") or wm > 1e300 else f"{wm:.4g}"
        lab = f"{label} ($T={r['eval_windows']}$)" if first else ""
        out.append(f"{lab} & {sl} & {a['mean_cov']:.3f} & {a['p05_cov']:.3f} & {a['min_cov']:.3f} & "
                   f"{a['share_below_0.80']:.3f} & {a['inf_share']:.2f} & {wtxt} & {cert} & "
                   f"{r['farthest_sensor'][sk]['mean_cov']:.3f} \\\\")
        first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table*}\n")

# ---- S4: forecast-time calibration, every n, with widths ----------------------------------------
d3 = json.load(open(PAPER.parent / "results" / "exp_D3_deployable.json"))
out.append("\\begin{table*}[htbp]\n\\caption{Forecast-time calibration (target 0.90): mean coverage with 95\\% "
           "interval over splits, share of random new sensors covered below 0.80, and mean width, for each "
           "scheme, rule and calibration size. Windows before $t_0=L$ are not evaluated.}\n"
           "\\label{tab:deployfull}\n\\centering\\scriptsize\n\\begin{tabular}{llllcccc}\n\\toprule\n"
           "\\textbf{Dataset} & $n$ & \\textbf{Rule} & \\textbf{Scheme} & \\textbf{Coverage} & \\textbf{95\\% interval} & "
           "\\textbf{Share $<0.80$} & \\textbf{Width} \\\\\n\\midrule")
SCHEME_LABEL = {"static": "Static", "expanding": "Expanding", "rolling": "Rolling", "aci": "ACI"}
for key, label in NAMES:
    if key not in d3:
        continue
    first = True
    for n, cellj in d3[key]["cells"].items():
        for rule, rlabel in (("crc", "Sensor-level"), ("pool", "Pooled")):
            for sch, slabel in SCHEME_LABEL.items():
                c = cellj[f"{sch}/{rule}"]
                lo, hi = c["coverage"][1], c["coverage"][2]
                ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "---"
                out.append(f"{label if first else ''} & {n} & {rlabel} & {slabel} & {c['coverage'][0]:.3f} & {ci} & "
                           f"{c['share_below_0.80']:.2f} & {c['width']:.3g} \\\\")
                first = False
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table*}\n")


# ---- S-D5b: blocks of windows and distance groups --------------------------------------------------
d5b = json.load(open(PAPER.parent / "results" / "exp_D5b_blocks_groups.json"))
out.append("\\begin{table}[htbp]\n\\caption{Coverage of the worst block of $k$ consecutive windows, per sensor (experiment D5b). "
           "Minimum over all sensor cases of each sensor's worst block, and the coverage certified for every block by "
           "Corollary~2, $1-\\alpha-(1+2\\gamma)/(\\gamma k)$. The certificate held for every block.}\n"
           "\\label{tab:d5b}\n\\centering\\footnotesize\n\\begin{tabular}{lrrcccc}\n\\toprule\n"
           " & & & \\multicolumn{2}{c}{$\\gamma=0.1$} & \\multicolumn{2}{c}{$\\gamma=0.5$} \\\\\n"
           "\\textbf{Dataset} & $k$ & $T$ & Worst & Cert. & Worst & Cert. \\\\\n\\midrule")
for key, label in NAMES:
    r = d5b[key]
    a, b = r["aci-0.1"], r["aci-0.5"]
    ca = f"{a['certified_every_block']:.3f}" if a["certified_every_block"] > 0 else "vac."
    cb = f"{b['certified_every_block']:.3f}" if b["certified_every_block"] > 0 else "vac."
    out.append(f"{label} & {r['k']} & {r['eval_windows']} & {a['worst_block_cov_min']:.3f} & {ca} & "
               f"{b['worst_block_cov_min']:.3f} & {cb} \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

out.append("\\begin{table}[htbp]\n\\caption{Same-window coverage and median 90\\% width by distance third on the traffic networks "
           "(experiment D5b). Group: each sensor calibrated on the other sensors of its own third (Theorem~1 within the "
           "group). All: calibrated on all other held-out sensors.}\n\\label{tab:groups}\n\\centering\\footnotesize\n"
           "\\begin{tabular}{llcccccc}\n\\toprule\n\\textbf{Dataset} & \\textbf{Third} & $n_\\mathrm{cal}$ & \\multicolumn{2}{c}{Coverage} & "
           "\\multicolumn{2}{c}{Median width} \\\\\n & & & Group & All & Group & All \\\\\n\\midrule")
for key, label in (("pemsd7_traffic", "PeMSD7-M"), ("pemsbay_traffic", "PeMS-BAY")):
    for third, v in d5b[key]["distance_groups"].items():
        out.append(f"{label} & {third} & {v['n_cal_per_group']} & {v['mean_cov_group_cal']:.3f} & {v['mean_cov_all_cal']:.3f} & "
                   f"{v['median_width_group_cal']:.1f} & {v['median_width_all_cal']:.1f} \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- S-D6: sensors installed late ------------------------------------------------------------------
d6 = json.load(open(PAPER.parent / "results" / "exp_D6_late.json"))
out.append("\\begin{table}[htbp]\n\\caption{Sensors installed late (GHCN-Daily TMAX, experiment D6, target 0.90, $T="
           f"{d6['eval_windows']}$ windows). Late: {d6['n_late_cases']} station cases; control: {d6['n_control_cases']} "
           "established held-out station cases. Mean, worst coverage, share of cases below 0.80, share of infinite windows, "
           "median width ($^\\circ$C), and the worst coverage over any block of 6 windows (90 days). $^*$Not deployable.}\n"
           "\\label{tab:d6full}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{llccccc}\n\\toprule\n\\textbf{Stations} & \\textbf{Scheme} & \\textbf{Mean} & \\textbf{Worst} & "
           "\\textbf{$<0.80$} & \\textbf{Inf.} & \\textbf{Width} \\\\\n\\midrule")
for grp, glabel in (("late", "Late"), ("control", "Control")):
    first = True
    for sk, sl in SCH:
        a = d6[grp][sk]
        out.append(f"{glabel if first else ''} & {sl} & {a['mean_cov']:.3f} & {a['min_cov']:.3f} & "
                   f"{a['share_below_0.80']:.3f} & {a['inf_share']:.3f} & {a['width_med']:.1f} \\\\")
        first = False
    wb = d6[grp]["worst_block"]
    out.append(f" & Worst 6-window block, ACI $\\gamma=0.1$ / $0.5$ & \\multicolumn{{5}}{{l}}{{{wb['aci-0.1']['min']:.3f} / "
               f"{wb['aci-0.5']['min']:.3f}}} \\\\")
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\n\\end{table}\n")
(PAPER / "supplement_tables.tex").write_text("\n".join(out))
print("wrote", PAPER / "supplement_tables.tex", len("\n".join(out).split("\n")), "lines")

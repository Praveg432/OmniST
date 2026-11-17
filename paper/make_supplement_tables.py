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
           "windows (those with a seasonal-naive forecast: all for the daily networks, windows 20 onward for the "
           "5-minute traffic networks). MAE, and the 90\\% interval width under the pooled and the sensor-level rule "
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

(PAPER / "supplement_tables.tex").write_text("\n".join(out))
print("wrote", PAPER / "supplement_tables.tex", len("\n".join(out).split("\n")), "lines")

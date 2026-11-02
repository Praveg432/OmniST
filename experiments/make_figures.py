#!/usr/bin/env python3
"""Regenerate ALL paper figures from the REAL result JSONs (replaces the stale June synthetic PNGs).

Every figure is drawn strictly from a committed results/*.json so the paper's figures and numbers are
provably consistent. Figures are skipped (with a warning) if their source JSON is absent, so this can
be run partially now and fully once B1-B4 complete.

Outputs -> results/figures/ (overwrites the stale synthetic figures):
  fig_multidomain_coverage.png   inductive coverage at unseen nodes across 3 domain types + dropout   [exp_multidomain_v2.json | exp_multidomain_real.json]
  fig_temporal_drift.png         coverage over time: static collapse vs rolling-ACI fix                [exp_temporal_conformal.json]
  fig_mondrian.png               worst-group coverage marginal vs Mondrian per domain                   [exp_mondrian_conformal.json]
  fig_calibration.png            calibration curve (empirical vs target) per domain                     [exp_calibration.json]
  fig_scalable_knn.png           per-forward latency exact vs scalable k-NN across N                    [exp_approx_knn.json]

Run: python experiments/make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"  [skip] {name} not found"); return None
    return json.load(open(p))


def fig_multidomain():
    """Prefer the 3-type v2 (incl weather); fall back to the 2-type real multidomain."""
    d = _load("exp_multidomain_v2.json")
    src = "v2"
    if d is None:
        d = _load("exp_multidomain_real.json"); src = "real"
    if d is None:
        return
    rows = []
    if src == "v2":
        for name, r in d["results_by_domain"].items():
            cov = r.get("coverage_mean_std", [np.nan])[0]
            # dropout: average over per-seed
            d30 = np.mean([c["coverage_under_dropout"].get("drop_30", np.nan) for c in r.get("per_seed", []) if c.get("coverage_under_dropout")]) if r.get("per_seed") else np.nan
            d50 = np.mean([c["coverage_under_dropout"].get("drop_50", np.nan) for c in r.get("per_seed", []) if c.get("coverage_under_dropout")]) if r.get("per_seed") else np.nan
            rows.append((name, r.get("type", ""), cov, d30, d50))
    else:
        for name, r in d["results_by_domain"].items():
            cov = r["omnist_coverage_unseen_mean_std"][0]
            dd = r.get("coverage_under_dropout_mean", {})
            rows.append((name, "", cov, dd.get("drop_30pct", np.nan), dd.get("drop_50pct", np.nan)))
    labels = [r[0].replace("_", "\n") for r in rows]
    x = np.arange(len(rows)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, [r[2] for r in rows], w, label="full network")
    ax.bar(x,     [r[3] for r in rows], w, label="30% nodes dropped")
    ax.bar(x + w, [r[4] for r in rows], w, label="50% nodes dropped")
    ax.axhline(0.90, ls="--", c="k", lw=1, label="target 0.90")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("coverage at unseen nodes"); ax.set_ylim(0.80, 1.0)
    ax.set_title("Inductive coverage across domain types, under topology change")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_multidomain_coverage.png")); plt.close(fig)
    print("  [ok] fig_multidomain_coverage.png")


def fig_temporal():
    d = _load("exp_temporal_conformal.json")
    if d is None: return
    m = d["methods"]; order = ["static_split_cp", "rolling", "aci", "rolling_aci"]
    names = ["static", "rolling", "ACI", "rolling-ACI"]
    overall = [m[k]["coverage_mean_std"][0] for k in order]
    late = [m[k]["late_window_coverage_mean_std"][0] for k in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w/2, overall, w, label="overall coverage")
    ax.bar(x + w/2, late, w, label="late-window (worst drift)")
    ax.axhline(0.90, ls="--", c="k", lw=1, label="target 0.90")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("coverage over time"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Temporal-drift fix: rolling-ACI restores coverage (real Delhi)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_temporal_drift.png")); plt.close(fig)
    print("  [ok] fig_temporal_drift.png")


def fig_mondrian():
    d = _load("exp_mondrian_conformal.json")
    if d is None: return
    rows = []
    for name, r in d["results_by_domain"].items():
        rows.append((name.replace("_", "\n"), r["marginal_worst_group_coverage_mean"],
                     r["mondrian_worst_group_coverage_mean"]))
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w/2, [r[1] for r in rows], w, label="marginal CP (worst group)")
    ax.bar(x + w/2, [r[2] for r in rows], w, label="Mondrian CP (worst group)")
    ax.axhline(0.90, ls="--", c="k", lw=1, label="target 0.90")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("worst-group coverage"); ax.set_ylim(0.75, 0.97)
    ax.set_title("Conditional coverage: Mondrian lifts the hardest group")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_mondrian.png")); plt.close(fig)
    print("  [ok] fig_mondrian.png")


def fig_calibration():
    d = _load("exp_calibration.json")
    if d is None: return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.5, 1.0], [0.5, 1.0], ls="--", c="k", lw=1, label="ideal")
    for name, r in d["results_by_domain"].items():
        if not isinstance(r, dict): continue
        curve = r["calibration_curve"]
        ts = sorted(int(k.split("_")[1]) for k in curve)
        emp = [curve[f"target_{t}"] for t in ts]
        ax.plot([t/100 for t in ts], emp, marker="o", ms=4, label=name.replace("_", " "))
    ax.set_xlabel("target coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration: empirical vs target across domains")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_calibration.png")); plt.close(fig)
    print("  [ok] fig_calibration.png")


def fig_scalable_knn():
    d = _load("exp_approx_knn.json")
    if d is None: return
    ladder = d["speed_ladder_per_forward_B2"]
    Ns = [r["N"] for r in ladder]
    exact = [r.get("exact_ms") for r in ladder]
    scal = [r.get("scalable_ms") for r in ladder]
    exact = [e if isinstance(e, (int, float)) else np.nan for e in exact]
    scal = [s if isinstance(s, (int, float)) else np.nan for s in scal]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ns, exact, marker="o", label="exact k-NN (O(N^2))")
    ax.plot(Ns, scal, marker="s", label="scalable k-NN (cKDTree)")
    ax.set_xlabel("number of nodes N"); ax.set_ylabel("latency per forward (ms)")
    ax.set_title("Scalable k-NN: equivalent output, better scaling")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_scalable_knn.png")); plt.close(fig)
    print("  [ok] fig_scalable_knn.png")


def main():
    print("Regenerating figures from real result JSONs...")
    fig_multidomain(); fig_temporal(); fig_mondrian(); fig_calibration(); fig_scalable_knn()
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()

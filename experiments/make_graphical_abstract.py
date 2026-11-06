#!/usr/bin/env python3
"""Graphical abstract for the IEEE Sensors Journal submission.

Three panels:
  (a) schematic of the idea: a sensor network with failed and newly added sensors; OmniST forecasts at
      every sensor present and returns a coverage-guaranteed interval at the new one. This panel is an
      illustration (synthetic positions), not data, and is labelled as such.
  (b) coverage at unseen sensors, OmniST vs nearest neighbor, on four real datasets
      [results/exp_multidomain_v2.json, the source of the paper's multi-domain table].
  (c) coverage over time under seasonal drift, static split conformal vs rolling-ACI
      [results/exp_temporal_conformal.json, the source of the paper's drift table].

Every plotted value is read from a committed result file; nothing is typed in by hand.

Run: python experiments/make_graphical_abstract.py
Output: results/figures/graphical_abstract.png (and .pdf), the full three-panel version, and
        results/figures/graphical_abstract_jsen.png (and .pdf), the compact version that sits in the
        IEEE Sensors Journal abstract box at 3 in wide.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RESULTS, "figures")

TARGET = 0.90
DATASETS = [  # (key in exp_multidomain_v2.json, axis label)
    ("delhi_air_quality", "Delhi\nair quality"),
    ("beijing_air_quality", "Beijing\nair quality"),
    ("ghcn_weather", "NOAA\nweather"),
    ("pemsd7_traffic", "PeMSD7\ntraffic"),
]

# Colour-blind-safe pair (Okabe-Ito blue / orange) plus neutral grey for baselines.
C_OMNIST = "#0072B2"
C_BASE = "#E69F00"
C_GREY = "#7F7F7F"


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} is required to draw the graphical abstract; expected at {path}")
    with open(path) as fh:
        return json.load(fh)


def panel_schematic(ax):
    rng = np.random.default_rng(7)  # fixed seed: the layout is a drawing, kept stable across runs
    pts = rng.uniform(0.08, 0.92, size=(22, 2))
    failed = [3, 11, 17]
    new = np.array([0.55, 0.5])

    for i, (x, y) in enumerate(pts):
        if i in failed:
            ax.plot(x, y, marker="x", ms=8, mew=2, color=C_GREY, ls="none")
        else:
            ax.plot(x, y, marker="o", ms=7, color=C_OMNIST, ls="none")

    # k nearest active sensors of the new location, recomputed from coordinates.
    active = [i for i in range(len(pts)) if i not in failed]
    dist = np.linalg.norm(pts[active] - new, axis=1)
    for j in np.argsort(dist)[:4]:
        x, y = pts[active[j]]
        ax.plot([new[0], x], [new[1], y], color=C_OMNIST, lw=1, alpha=0.6, zorder=1)
    ax.plot(*new, marker="*", ms=17, color=C_BASE, mec="black", mew=0.6, ls="none", zorder=3)

    ax.plot([], [], marker="o", color=C_OMNIST, ls="none", label="trained sensor")
    ax.plot([], [], marker="x", color=C_GREY, mew=2, ls="none", label="failed sensor")
    ax.plot([], [], marker="*", ms=12, color=C_BASE, mec="black", mew=0.6, ls="none",
            label="new sensor: forecast\n+ guaranteed interval")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), fontsize=7.5, frameon=False, ncol=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) No global sensor count\n(schematic)", fontsize=9)
    for side in ax.spines.values():
        side.set_color("#BBBBBB")


def panel_unseen_coverage(ax):
    by_domain = load("exp_multidomain_v2.json")["results_by_domain"]
    omnist = [by_domain[k]["coverage_mean_std"][0] for k, _ in DATASETS]
    nn = [by_domain[k]["nn_coverage_mean_std"][0] for k, _ in DATASETS]
    x = np.arange(len(DATASETS))
    w = 0.38
    ax.bar(x - w / 2, omnist, w, color=C_OMNIST, label="OmniST + conformal")
    ax.bar(x + w / 2, nn, w, color=C_BASE, label="Nearest neighbor")
    ax.axhline(TARGET, ls="--", color="black", lw=1, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS], fontsize=7.5)
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Coverage at unseen sensors", fontsize=8.5)
    ax.set_title("(b) Coverage where no sensor\nwas trained", fontsize=9)
    # Legend below the axis, matching panel (a), so it never covers bars or tick labels.
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
    ax.tick_params(axis="y", labelsize=8)


def panel_drift(ax):
    methods = load("exp_temporal_conformal.json")["methods"]
    order = [("static_split_cp", "Static\nconformal", C_GREY), ("rolling_aci", "Rolling-ACI", C_OMNIST)]
    overall = [methods[k]["coverage_mean_std"][0] for k, _, _ in order]
    late = [methods[k]["late_window_coverage_mean_std"][0] for k, _, _ in order]
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, overall, w, color=[c for *_, c in order], alpha=0.45)
    ax.bar(x + w / 2, late, w, color=[c for *_, c in order], hatch="///", edgecolor="white", lw=0)
    ax.axhline(TARGET, ls="--", color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in order], fontsize=7.5)
    ax.set_ylim(0.40, 1.0)
    ax.set_ylabel("Coverage over time", fontsize=8.5)
    ax.set_title("(c) Coverage under\nseasonal drift (Delhi)", fontsize=9)
    # Neutral legend patches: the encoding is light = whole year, hatched = late window.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_GREY, alpha=0.45),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_GREY, hatch="///", edgecolor="white", lw=0),
        plt.Line2D([0], [0], ls="--", color="black", lw=1),
    ]
    ax.legend(handles, ["whole test year", "late-year window", "90% target"], fontsize=7.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
    ax.tick_params(axis="y", labelsize=8)


def main_jsen_box():
    """Compact two-panel version for the IEEE Sensors Journal abstract box (printed 3 in wide).

    Drawn at 6 x 2.9 in with large fonts, so that at the printed 3 in width text lands near 7 to 8 pt.
    Same data sources as the full version; the schematic panel is dropped to keep the plots legible.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3})
    by_domain = load("exp_multidomain_v2.json")["results_by_domain"]
    methods = load("exp_temporal_conformal.json")["methods"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    short = ["Delhi\nAQ", "Beijing\nAQ", "NOAA\nweather", "PeMSD7\ntraffic"]
    omnist = [by_domain[k]["coverage_mean_std"][0] for k, _ in DATASETS]
    nn = [by_domain[k]["nn_coverage_mean_std"][0] for k, _ in DATASETS]
    x = np.arange(len(DATASETS))
    w = 0.38
    ax1.bar(x - w / 2, omnist, w, color=C_OMNIST, label="OmniST")
    ax1.bar(x + w / 2, nn, w, color=C_BASE, label="Nearest neighbor")
    ax1.axhline(TARGET, ls="--", color="black", lw=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=12)
    ax1.set_ylim(0.85, 0.95)
    ax1.set_yticks([0.86, 0.90, 0.94])
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_title("Coverage at unseen sensors", fontsize=14)
    ax1.legend(fontsize=11, loc="upper right", ncol=2, frameon=False, handlelength=1.0,
               columnspacing=0.8, bbox_to_anchor=(1.02, 1.0))

    order = [("static_split_cp", "Static", C_GREY), ("rolling_aci", "Rolling\nACI", C_OMNIST)]
    late = [methods[k]["late_window_coverage_mean_std"][0] for k, _, _ in order]
    ax2.bar(np.arange(2), late, 0.6, color=[c for *_, c in order])
    ax2.axhline(TARGET, ls="--", color="black", lw=1.2)
    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels([label for _, label, _ in order], fontsize=12)
    ax2.set_ylim(0.4, 1.0)
    ax2.set_yticks([0.5, 0.7, 0.9])
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_title("Late-year coverage\nunder drift", fontsize=14)
    # Labels sit inside the bars so they never collide with the title.
    for i, v in enumerate(late):
        ax2.text(i, (0.4 + v) / 2, f"{v:.2f}", ha="center", va="center", fontsize=12, color="white",
                 fontweight="bold")

    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGS, f"graphical_abstract_jsen.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25, 0.95]})
    panel_schematic(axes[0])
    axes[0].grid(False)
    panel_unseen_coverage(axes[1])
    panel_drift(axes[2])
    fig.tight_layout(w_pad=1.5)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGS, f"graphical_abstract.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    main_jsen_box()

#!/usr/bin/env python3
"""Figures for the revised manuscript. Every plotted value is read from results/exp_D2_analysis.json.

  fig_sensor_level.png          coverage and width against the number of calibration SENSORS, for the
                                sensor-level rule (Theorem 1) and the pooled rule, alpha = 0.1
  graphical_abstract_jsen.png   compact two-panel version for the journal abstract box

Run: python experiments/make_figures_v2.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RES, "figures")
ALPHA = "0.1"
TARGET = 0.90
C_CRC, C_POOL, C_GREY = "#0072B2", "#E69F00", "#7F7F7F"
SHOW = [("delhi_air_quality", "Delhi (16 held out)"), ("ghcn_weather", "Weather (12)"),
        ("pemsd7_traffic", "PeMSD7-M (60)"), ("pemsbay_traffic", "PeMS-BAY (90)")]
SHORT = {"delhi_air_quality": "Delhi\nAQ", "ghcn_weather": "NOAA\nweather",
         "pemsd7_traffic": "D7-M\ntraffic", "pemsbay_traffic": "BAY\ntraffic"}


def cells(d, dom):
    out = []
    for n, cell in d[dom]["coverage"][ALPHA].items():
        c = cell["full"]
        w_crc = c["crc"]["width_finite"]
        out.append(dict(n=int(n), cov_crc=c["crc"]["coverage"]["mean"], cov_pool=c["naive"]["coverage"]["mean"],
                        w_crc=w_crc["mean"] if w_crc else np.nan,
                        w_pool=c["naive"]["width_finite"]["mean"]))
    return sorted(out, key=lambda r: r["n"])


def main():
    d = json.load(open(os.path.join(RES, "exp_D2_analysis.json")))
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3})

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.3))
    for j, (dom, title) in enumerate(SHOW):
        rows = cells(d, dom)
        n = [r["n"] for r in rows]
        ax = axes[j]
        ax.plot(n, [r["cov_crc"] for r in rows], "o-", color=C_CRC, label="sensor-level (Thm. 1)")
        ax.plot(n, [r["cov_pool"] for r in rows], "s--", color=C_POOL, label="pooled")
        ax.axhline(TARGET, ls=":", color="black", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.86, 1.01)
        ax.set_xlabel("calibration sensors $n$", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("Mean coverage at\na random new sensor", fontsize=8.5)
    axes[1].legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_sensor_level.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)

    # compact graphical abstract: largest n available per dataset
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 2.9))
    x = np.arange(len(SHOW))
    last = [cells(d, dom)[-1] for dom, _ in SHOW]
    w = 0.38
    a1.bar(x - w / 2, [r["cov_crc"] for r in last], w, color=C_CRC, label="sensor-level")
    a1.bar(x + w / 2, [r["cov_pool"] for r in last], w, color=C_POOL, label="pooled")
    a1.axhline(TARGET, ls="--", color="black", lw=1.2)
    a1.set_xticks(x)
    a1.set_xticklabels([SHORT[k] for k, _ in SHOW], fontsize=10)
    a1.set_ylim(0.85, 1.045)
    a1.set_yticks([0.86, 0.90, 0.94, 0.98])
    a1.tick_params(axis="y", labelsize=11)
    a1.set_title("Same-window calibration\n(theorem's setting)", fontsize=12)
    a1.legend(fontsize=9, loc="upper right", frameon=False, ncol=2, handlelength=1.0, columnspacing=0.6)
    d5 = json.load(open(os.path.join(RES, "exp_D5_per_sensor.json")))
    worst = [(d5[dom]["all_sensors"]["aci-0.1"]["min_cov"], d5[dom]["all_sensors"]["rolling-pooled"]["min_cov"])
             for dom, _ in SHOW]
    a2.bar(x - w / 2, [v[0] for v in worst], w, color=C_CRC, label="per-sensor ACI")
    a2.bar(x + w / 2, [v[1] for v in worst], w, color=C_POOL, label="pooled")
    a2.axhline(TARGET, ls="--", color="black", lw=1.2)
    a2.set_xticks(x)
    a2.set_xticklabels([SHORT[k] for k, _ in SHOW], fontsize=10)
    a2.set_ylim(0.5, 1.0)
    a2.set_yticks([0.6, 0.7, 0.8, 0.9])
    a2.tick_params(axis="y", labelsize=11)
    a2.legend(fontsize=9, loc="lower right", frameon=False, handlelength=1.0)
    a2.set_title("Forecast time:\nworst sensor", fontsize=12)
    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGS, f"graphical_abstract_jsen.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()

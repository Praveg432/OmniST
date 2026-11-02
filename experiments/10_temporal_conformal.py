#!/usr/bin/env python3
"""EX-TEMPORAL - the temporal-drift conformal fix on REAL data (resolves the OPT-4 negative).

Prior finding (OPT-4): on real Delhi PM2.5, a STATIC split-conformal interval calibrated on the first
half of the test year UNDER-covers badly on the second half (coverage ~0.75 vs target 0.90). The
spatial guarantee did not transfer to the temporal axis because seasonal drift breaks exchangeability
between the calibration window and later test points. A first naive ACI attempt did not recover it.

This experiment applies the principled online-conformal estimators in src/online_conformal.py:
  - STATIC split-CP            (baseline; expected to under-cover)
  - RollingConformal           (trailing-window quantile tracks the drifting residual scale)
  - AdaptiveConformal (ACI)    (Gibbs & Candes online level update; coverage guarantee under drift)
  - RollingACI                 (recommended: trailing window + ACI correction)

Metric: running empirical coverage of the time-ordered residual stream over the test period, plus
coverage over the LATE (final third) window where drift is worst. We also report mean interval width
so the reader can see the width OmniST pays to hold coverage.

Validated first on controlled synthetic drift (see /tmp/test_online_cp.py in the dev log):
static 0.61 -> RollingACI 0.90. Here we confirm on the real model+data.

Run: python experiments/10_temporal_conformal.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import DataConfig, collate_variable_nodes, SpatiotemporalDataset
from src.data_real import prepare_real_delhi
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model
from src.online_conformal import RollingConformal, AdaptiveConformal, RollingACI


def fine_test_dataset(meta, window, horizon, stride=3):
    """Rebuild the test dataset at a FINER stride so the temporal residual stream has enough steps
    for online adaptation (the default stride=horizon leaves only ~12 steps over the test year)."""
    test_all = meta["test_data_all_normalized"]           # (T, N_all, F), train-normalized
    train_idx = meta["train_stations"]
    data = test_all[:, train_idx, :]
    coords = meta["coords_all"][train_idx]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, coords, meta["test_dates"], window_size=w, horizon=horizon, stride=stride)

TARGET = 0.90
ALPHA = 1 - TARGET
N_SEEDS = 3
WINDOW, HORIZON = 60, 30


def time_ordered_residuals(model, test_ds, meta):
    """Per-step denormalized absolute residuals, in test-time order -> list of (n_nodes,) arrays."""
    mu = torch.tensor(float(np.asarray(meta["means"]).squeeze()))
    sd = torch.tensor(float(np.asarray(meta["stds"]).squeeze()))
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_nodes)
    seq = []
    model.eval()
    with torch.no_grad():
        for f, c, t, y in loader:
            p = model(f, c, t)
            r = (((p * sd + mu) - (y * sd + mu)).abs()).reshape(-1).numpy()
            seq.append(r)
    return seq


def static_split_cp(seq, alpha):
    """Calibrate on first half, evaluate running coverage on second half (the OPT-4 protocol)."""
    half = len(seq) // 2
    cal = np.concatenate(seq[:half]); n = len(cal)
    level = min(1.0, (1 - alpha) * (1 + 1 / max(n, 1)))
    q = float(np.quantile(cal, level))
    ev = seq[half:]
    cov = [float((r <= q).mean()) for r in ev if len(r)]
    width = 2 * q
    late = ev[len(ev) * 2 // 3:]
    late_cov = float(np.mean([float((r <= q).mean()) for r in late if len(r)])) if late else float("nan")
    return float(np.mean(cov)), float(width), late_cov


def online_method(ctor, seq, alpha, **kw):
    """Warm up on first half, run the estimator over the second half (same protocol as static)."""
    half = len(seq) // 2
    warmup = np.concatenate(seq[:half])
    est = ctor(alpha=alpha, warmup=warmup, **kw)
    widths = []
    ev = seq[half:]
    per_step_cov = []
    for r in ev:
        q, cov_t = est.update(r)
        widths.append(2 * q)
        per_step_cov.append(cov_t)
    late = per_step_cov[len(per_step_cov) * 2 // 3:]
    late_cov = float(np.nanmean(late)) if late else float("nan")
    return est.running_coverage, float(np.mean(widths)), late_cov


def run():
    t0 = time.time()
    cfg = DataConfig(window_size=WINDOW, horizon=HORIZON, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=25, patience=7, batch_size=8, loss_type="pot")

    methods = {
        "static_split_cp": None,
        "rolling": (RollingConformal, {"window": 20}),
        "aci": (AdaptiveConformal, {"gamma": 0.02}),
        "rolling_aci": (RollingACI, {"window": 20, "gamma": 0.05}),
    }
    acc = {k: {"cov": [], "width": [], "late": []} for k in methods}

    for seed in range(N_SEEDS):
        print(f"[seed {seed+1}/{N_SEEDS}] training OmniST on real Delhi...", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        tr, va, te, meta = prepare_real_delhi(cfg)
        model = OmniST(ModelConfig(n_features=1, horizon=HORIZON, d_model=64, k_neighbors=8))
        train_model(model, tr, va, tcfg, meta)
        te_fine = fine_test_dataset(meta, WINDOW, HORIZON, stride=3)
        seq = time_ordered_residuals(model, te_fine, meta)
        if len(seq) < 4:
            print(f"   seed {seed}: too few test steps ({len(seq)}), skipping", flush=True)
            continue

        c, w, lc = static_split_cp(seq, ALPHA)
        acc["static_split_cp"]["cov"].append(c); acc["static_split_cp"]["width"].append(w); acc["static_split_cp"]["late"].append(lc)
        for name, spec in methods.items():
            if spec is None:
                continue
            ctor, kw = spec
            c, w, lc = online_method(ctor, seq, ALPHA, **kw)
            acc[name]["cov"].append(c); acc[name]["width"].append(w); acc[name]["late"].append(lc)
        print(f"   static={acc['static_split_cp']['cov'][-1]:.3f} "
              f"rolling={acc['rolling']['cov'][-1]:.3f} aci={acc['aci']['cov'][-1]:.3f} "
              f"rolling_aci={acc['rolling_aci']['cov'][-1]:.3f} (n_steps={len(seq)})", flush=True)

    def ms(name, key):
        v = [x for x in acc[name][key] if x == x]
        return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v else [float("nan"), float("nan")]

    summary = {
        "experiment": "EX-TEMPORAL (online-conformal fix for temporal-drift under-coverage)",
        "data": "real CPCB Delhi PM2.5 (37 stations), OmniST model",
        "target_coverage": TARGET,
        "protocol": "calibrate on first half of test year; run online over second half (drift regime)",
        "n_seeds": N_SEEDS,
        "methods": {
            name: {
                "coverage_mean_std": ms(name, "cov"),
                "late_window_coverage_mean_std": ms(name, "late"),
                "mean_interval_width_mean_std": ms(name, "width"),
            } for name in methods
        },
        "prior_opt4_static_coverage": 0.7525,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp_temporal_conformal.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== TEMPORAL CONFORMAL SUMMARY ===")
    print(json.dumps(summary["methods"], indent=2, default=float))
    print(f"saved -> {out_path}")
    return summary


if __name__ == "__main__":
    run()

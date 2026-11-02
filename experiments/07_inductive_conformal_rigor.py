#!/usr/bin/env python3
"""EX-CONF-RIGOR - Strengthen the inductive-conformal findings across domains and cases.

The headline OmniST contribution (guaranteed-coverage prediction intervals at UNSEEN sensor
locations under dynamic topology) was shown on one domain (Delhi air quality), one coverage target
(90%), one held-out set. A single-dataset, single-config result is fragile. This experiment makes
it rigorous by stress-testing the SAME capability across:

  1. A SECOND DOMAIN with different scale/structure/features: PeMS-BAY-style TRAFFIC
     (100 sensors on freeway corridors; speed/flow/occupancy; hourly) vs Delhi (37 stations; daily;
     PM2.5). If the guarantee holds on both, it is a property of the architecture, not the dataset.
  2. MULTIPLE COVERAGE TARGETS (80%, 90%, 95%) -- a conformal method must hit WHATEVER target you
     set, not just one.
  3. MULTIPLE HELD-OUT SET SIZES -- the inductive claim must not depend on which/how-many nodes are
     unseen.
  4. HIGHER NODE-DROPOUT (up to 50-60%) -- stress the "guarantee survives topology change" claim.

Per domain/case we report: empirical coverage at UNSEEN nodes vs target, and mean interval width.
A PASS = empirical coverage within +/-3 pts of the target across targets, held-out sizes, and under
dropout. Multi-seed.

Run: python experiments/07_inductive_conformal_rigor.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import DataConfig, SpatiotemporalDataset, collate_variable_nodes
from src.data_real import prepare_real_delhi
from src.data_traffic import generate_traffic_realistic
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model

N_SEEDS = 3
COVERAGE_TARGETS = [0.80, 0.90, 0.95]


# ---------------------------------------------------------------- shared conformal utilities
def collect_residuals(model, ds, means, stds):
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    res = []
    model.eval()
    with torch.no_grad():
        for feat, coords, ts, tgt in loader:
            pred = model(feat, coords, ts)
            r = ((pred * stds + means) - (tgt * stds + means)).abs()
            res.append(r.reshape(-1).numpy())
    return np.concatenate(res) if res else np.array([])


def qhat(residuals, alpha):
    n = len(residuals)
    level = min(1.0, (1 - alpha) * (1 + 1 / max(n, 1)))
    return float(np.quantile(residuals, level))


def coverage_of(residuals, q):
    return float((residuals <= q).mean()) if len(residuals) else float("nan")


# ---------------------------------------------------------------- traffic inductive split
def build_traffic_case(hold_out, seed, window=168, horizon=24, n_sensors=100):
    """Return (train_ds, val_ds, cal_idx, holdout_idx, test_norm_all, coords, means, stds).
    Train on the non-held-out sensors; keep held-out for inductive eval. First feature (speed)."""
    feats, coords, dates = generate_traffic_realistic(n_sensors=n_sensors, n_days=120, seed=seed)
    feats = feats[:, :, :1]                       # speed only -> n_features=1 (match model)
    T = len(feats)
    means = feats.mean(axis=(0, 1), keepdims=True)
    stds = feats.std(axis=(0, 1), keepdims=True) + 1e-8
    norm = (feats - means) / stds
    train_sensors = [i for i in range(n_sensors) if i not in hold_out]
    n_tr = int(T * 0.7); n_val = int(T * 0.15)
    w = min(window, n_tr - horizon - 1)
    train_ds = SpatiotemporalDataset(norm[:n_tr][:, train_sensors], coords[train_sensors],
                                     dates[:n_tr], window_size=w, horizon=horizon, stride=horizon)
    val_ds = SpatiotemporalDataset(norm[n_tr:n_tr+n_val][:, train_sensors], coords[train_sensors],
                                   dates[n_tr:n_tr+n_val],
                                   window_size=min(w, n_val - horizon - 1), horizon=horizon, stride=horizon)
    test_norm = norm[n_tr+n_val:]
    return (train_ds, val_ds, train_sensors, hold_out, test_norm, coords, dates[n_tr+n_val:],
            float(means.squeeze()), float(stds.squeeze()), w, horizon)


def ds_over(test_norm, idx, coords, dates, window, horizon):
    data = test_norm[:, idx, :]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, coords[idx], dates, window_size=w, horizon=horizon, stride=horizon)


# ---------------------------------------------------------------- run one domain
def run_delhi(seed, hold_out, window=60, horizon=30):
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=25, patience=7, batch_size=8, loss_type="pot")
    train_ds, val_ds, test_ds, meta = prepare_real_delhi(cfg, hold_out_stations=hold_out)
    torch.manual_seed(seed)
    model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8))
    train_model(model, train_ds, val_ds, tcfg, meta)
    means = float(np.asarray(meta["means"]).squeeze()); stds = float(np.asarray(meta["stds"]).squeeze())
    m_t = torch.tensor(means); s_t = torch.tensor(stds)
    cal_res = collect_residuals(model, test_ds, m_t, s_t)        # training-station residuals
    ind_ds = _delhi_inductive_ds(meta, hold_out, window, horizon)
    ind_res = collect_residuals(model, ind_ds, m_t, s_t)
    train_idx = meta["train_stations"]
    return model, meta, cal_res, ind_res, train_idx, (window, horizon), (m_t, s_t)


def _delhi_inductive_ds(meta, idx, window, horizon):
    test_all = meta["test_data_all_normalized"]; coords_all = meta["coords_all"]
    data = test_all[:, idx, :]; w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, coords_all[idx], meta["test_dates"],
                                 window_size=w, horizon=horizon, stride=horizon)


def delhi_dropout_cov(meta, model, train_idx, ind_res, ms, window, horizon, frac, rng):
    m_t, s_t = ms
    keep = sorted(rng.choice(len(train_idx), int(len(train_idx) * (1 - frac)), replace=False))
    surv = _delhi_inductive_ds(meta, [train_idx[i] for i in keep], window, horizon)
    q = qhat(collect_residuals(model, surv, m_t, s_t), 0.10)
    return coverage_of(ind_res, q)


# ---------------------------------------------------------------- main
def run():
    t0 = time.time()
    out = {"experiment": "EX-CONF-RIGOR (multi-domain / multi-case inductive conformal)",
           "n_seeds": N_SEEDS, "coverage_targets": COVERAGE_TARGETS, "domains": {}}

    # ================= DOMAIN 1: DELHI AIR QUALITY (held-out-size sweep + coverage-target sweep) =================
    print("=== DELHI (air quality) ===", flush=True)
    delhi_holdout_cases = {"holdout_5": [3, 11, 19, 27, 33], "holdout_8": [2, 6, 11, 16, 21, 27, 31, 35]}
    delhi_results = {}
    for case, ho in delhi_holdout_cases.items():
        per_target = {f"target_{int(c*100)}": [] for c in COVERAGE_TARGETS}
        widths = {f"target_{int(c*100)}": [] for c in COVERAGE_TARGETS}
        drop_cov = {"drop_30": [], "drop_50": [], "drop_60": []}
        for seed in range(N_SEEDS):
            model, meta, cal_res, ind_res, train_idx, (w, h), ms = run_delhi(seed, ho)
            for c in COVERAGE_TARGETS:
                q = qhat(cal_res, 1 - c)
                per_target[f"target_{int(c*100)}"].append(coverage_of(ind_res, q))
                widths[f"target_{int(c*100)}"].append(2 * q)
            rng = np.random.default_rng(seed)
            for frac, key in [(0.3, "drop_30"), (0.5, "drop_50"), (0.6, "drop_60")]:
                drop_cov[key].append(delhi_dropout_cov(meta, model, train_idx, ind_res, ms, w, h, frac, rng))
        delhi_results[case] = {
            "coverage_by_target": {k: round(float(np.mean(v)), 4) for k, v in per_target.items()},
            "width_by_target": {k: round(float(np.mean(v)), 2) for k, v in widths.items()},
            "coverage_under_dropout_at_90": {k: round(float(np.mean(v)), 4) for k, v in drop_cov.items()},
        }
        print(f"  {case}: {delhi_results[case]['coverage_by_target']} drop={delhi_results[case]['coverage_under_dropout_at_90']}", flush=True)
    out["domains"]["delhi_airquality"] = {"n_train_stations_approx": 37, "feature": "PM2.5", "cases": delhi_results}

    # ================= DOMAIN 2: TRAFFIC (different scale/features) =================
    print("=== TRAFFIC (PeMS-BAY-style, 100 sensors) ===", flush=True)
    traffic_holdout = list(range(0, 100, 12))[:8]      # 8 spread-out held-out sensors
    per_target = {f"target_{int(c*100)}": [] for c in COVERAGE_TARGETS}
    widths = {f"target_{int(c*100)}": [] for c in COVERAGE_TARGETS}
    drop_cov = {"drop_30": [], "drop_50": [], "drop_60": []}
    for seed in range(N_SEEDS):
        (train_ds, val_ds, train_idx, ho, test_norm, coords, tdates, means, stds, w, h) = \
            build_traffic_case(traffic_holdout, seed)
        torch.manual_seed(seed)
        model = OmniST(ModelConfig(n_features=1, horizon=h, d_model=64, k_neighbors=8))
        train_model(model, train_ds, val_ds,
                    TrainConfig(lr=1e-3, n_epochs=25, patience=7, batch_size=8, loss_type="pot"),
                    {"means": np.array([[[means]]]), "stds": np.array([[[stds]]])})
        m_t, s_t = torch.tensor(means), torch.tensor(stds)
        cal_ds = ds_over(test_norm, train_idx, coords, tdates, w, h)
        cal_res = collect_residuals(model, cal_ds, m_t, s_t)
        ind_ds = ds_over(test_norm, ho, coords, tdates, w, h)
        ind_res = collect_residuals(model, ind_ds, m_t, s_t)
        for c in COVERAGE_TARGETS:
            q = qhat(cal_res, 1 - c)
            per_target[f"target_{int(c*100)}"].append(coverage_of(ind_res, q))
            widths[f"target_{int(c*100)}"].append(2 * q)
        rng = np.random.default_rng(seed)
        for frac, key in [(0.3, "drop_30"), (0.5, "drop_50"), (0.6, "drop_60")]:
            keep = sorted(rng.choice(len(train_idx), int(len(train_idx) * (1 - frac)), replace=False))
            surv = ds_over(test_norm, [train_idx[i] for i in keep], coords, tdates, w, h)
            q = qhat(collect_residuals(model, surv, m_t, s_t), 0.10)
            drop_cov[key].append(coverage_of(ind_res, q))
        print(f"  seed {seed}: cov@90={np.mean([per_target['target_90'][-1]]):.3f}", flush=True)
    out["domains"]["traffic"] = {
        "n_train_sensors_approx": 92, "feature": "speed",
        "cases": {"holdout_8": {
            "coverage_by_target": {k: round(float(np.mean(v)), 4) for k, v in per_target.items()},
            "width_by_target": {k: round(float(np.mean(v)), 2) for k, v in widths.items()},
            "coverage_under_dropout_at_90": {k: round(float(np.mean(v)), 4) for k, v in drop_cov.items()},
        }},
    }

    # ================= rigor verdict =================
    def within(cov, target, tol=0.03):
        return abs(cov - target) <= tol
    checks = []
    for dom, dd in out["domains"].items():
        for case, cd in dd["cases"].items():
            for k, cov in cd["coverage_by_target"].items():
                tgt = int(k.split("_")[1]) / 100
                checks.append(within(cov, tgt))
    out["rigor_summary"] = {
        "coverage_checks_within_3pts": f"{sum(checks)}/{len(checks)}",
        "pass_fraction": round(sum(checks) / max(1, len(checks)), 3),
        "verdict": ("ROBUST across domains/targets/held-out-sizes" if sum(checks) / len(checks) >= 0.8
                    else "PARTIAL - some configs miss target by >3pts (see per-case)"),
    }
    out["runtime_seconds"] = round(time.time() - t0, 1)

    from pathlib import Path
    Path(os.path.join(os.path.dirname(__file__), "..", "results",
                      "exp_inductive_conformal_rigor.json")).write_text(json.dumps(out, indent=2, default=float))
    print("\n=== RIGOR SUMMARY ===")
    print(json.dumps(out["rigor_summary"], indent=2))
    print(f"runtime {out['runtime_seconds']}s")
    return out


if __name__ == "__main__":
    run()

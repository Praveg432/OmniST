#!/usr/bin/env python3
"""EX-REAL-RIGOR - staged rigor experiments on the THREE REAL datasets. READY TO RUN; NOT auto-run.

This hardens the real-data evidence along the axes the readiness assessment flagged as remaining:

  A  Multi-seed hardening        -- raise seed counts on all 3 real domains (default 5; traffic 3)
                                     so every real-domain coverage number has a mean +/- std.
  B  Multi-target coverage sweep -- repeat inductive coverage at targets {80, 90, 95}% on all 3
                                     real domains (the "valid method hits whatever target" check),
                                     now on REAL traffic (supersedes the earlier synthetic sweep).
  C  Conditional-coverage diag.  -- report the SPREAD of per-held-out-node coverage (min/mean/max)
                                     at target 90%, quantifying how far marginal coverage is from
                                     conditional (per-node) coverage -- an honest limitation metric.

Design notes (compute realism, learned this project):
  - Traffic (228 nodes) is the expensive domain: non-overlapping train windows (loader already does
    this), fewer epochs, fewer seeds. Budget ~5-9 min/seed.
  - Everything reuses the validated harness (train_model / conformal helpers from exp 06/09).
  - Writes results/exp_realmultidomain_rigor.json. Incremental save per (domain,target) so a long
    run is never lost.

RUN ONLY ON EXPLICIT COMMAND:
    python experiments/11_realmultidomain_rigor.py            # full (all domains, all targets)
    python experiments/11_realmultidomain_rigor.py quick      # 1 seed, target 90 only (smoke)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.data import DataConfig, SpatiotemporalDataset
from src.data_real import prepare_real_delhi
from src.data_beijing import prepare_real_beijing
from src.data_traffic_real import prepare_real_traffic
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model

COVERAGE_TARGETS = [0.80, 0.90, 0.95]

# Per-domain config: window/horizon, held-out (unseen) nodes, epochs, seeds.
DOMAINS = {
    "delhi_air_quality": {
        "prepare": prepare_real_delhi, "window": 60, "horizon": 30,
        "hold_out": [3, 11, 19, 27, 33], "epochs": 25, "seeds": 5,
    },
    "beijing_air_quality": {
        "prepare": prepare_real_beijing, "window": 30, "horizon": 15,
        "hold_out": [2, 6, 9], "epochs": 25, "seeds": 5,
    },
    "pemsd7_traffic": {
        "prepare": prepare_real_traffic, "window": 48, "horizon": 12,
        "hold_out": [5, 40, 90, 140, 200], "epochs": 12, "seeds": 3,
    },
}


# ---- conformal helpers (identical logic to experiments 06/09) ----
def build_inductive_test(meta, station_idx, window, horizon):
    test_all = meta["test_data_all_normalized"]
    data = test_all[:, station_idx, :]
    coords = meta["coords_all"][station_idx]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, coords, meta["test_dates"], window_size=w, horizon=horizon, stride=horizon)


def collect_residuals(model, ds, meta, per_node=False):
    from torch.utils.data import DataLoader
    from src.data import collate_variable_nodes
    means = torch.tensor(meta["means"], dtype=torch.float32).squeeze()
    stds = torch.tensor(meta["stds"], dtype=torch.float32).squeeze()
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    flat, by_node = [], None
    model.eval()
    with torch.no_grad():
        for feat, coords, ts, tgt in loader:
            pred = model(feat, coords, ts)
            r = ((pred * stds + means) - (tgt * stds + means)).abs()   # (B, N, H, F)
            if per_node:
                rn = r.permute(1, 0, 2, 3).reshape(r.shape[1], -1).numpy()  # (N, B*H*F)
                by_node = rn if by_node is None else np.concatenate([by_node, rn], axis=1)
            flat.append(r.reshape(-1).numpy())
    flat = np.concatenate(flat) if flat else np.array([])
    return (flat, by_node) if per_node else flat


def split_conformal_qhat(residuals, alpha):
    n = len(residuals)
    level = min(1.0, (1 - alpha) * (1 + 1 / max(n, 1)))
    return float(np.quantile(residuals, level))


def run_domain_seed(spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    window, horizon, hold_out = spec["window"], spec["horizon"], spec["hold_out"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=spec["epochs"], patience=7, batch_size=8, loss_type="pot")
    train_ds, val_ds, test_ds, meta = spec["prepare"](cfg, hold_out_stations=hold_out)
    model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8))
    train_model(model, train_ds, val_ds, tcfg, meta)

    cal_res = collect_residuals(model, test_ds, meta)                 # training-station residuals
    ind_flat, ind_by_node = collect_residuals(model, build_inductive_test(meta, hold_out, window, horizon),
                                              meta, per_node=True)

    out = {"seed": seed, "coverage_by_target": {}, "width_by_target": {}}
    for tgt in COVERAGE_TARGETS:
        a = 1 - tgt
        q = split_conformal_qhat(cal_res, a)
        cov = float((ind_flat <= q).mean()) if len(ind_flat) else float("nan")
        out["coverage_by_target"][f"target_{int(tgt*100)}"] = round(cov, 4)
        out["width_by_target"][f"target_{int(tgt*100)}"] = round(2 * q, 3)
    # conditional-coverage diagnostic at 90%
    q90 = split_conformal_qhat(cal_res, 0.10)
    per_node_cov = [(row <= q90).mean() for row in ind_by_node] if ind_by_node is not None else []
    if per_node_cov:
        out["per_node_coverage_at90"] = {
            "min": round(float(np.min(per_node_cov)), 4),
            "mean": round(float(np.mean(per_node_cov)), 4),
            "max": round(float(np.max(per_node_cov)), 4),
        }
    return out


def run(quick=False):
    t0 = time.time()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp_realmultidomain_rigor.json")
    summary = {
        "experiment": "EX-REAL-RIGOR (multi-seed + multi-target + conditional-coverage on 3 REAL datasets)",
        "coverage_targets": COVERAGE_TARGETS, "quick": quick, "results_by_domain": {},
    }
    for name, spec in DOMAINS.items():
        seeds = 1 if quick else spec["seeds"]
        print(f"\n===== {name} ({seeds} seeds, epochs {spec['epochs']}) =====", flush=True)
        per_seed = []
        for s in range(seeds):
            print(f"  seed {s+1}/{seeds}...", flush=True)
            per_seed.append(run_domain_seed(spec, s))
            print(f"    cov90={per_seed[-1]['coverage_by_target'].get('target_90')} "
                  f"per_node={per_seed[-1].get('per_node_coverage_at90')}", flush=True)
        def agg(tgt):
            v = [p["coverage_by_target"][f"target_{int(tgt*100)}"] for p in per_seed]
            return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
        summary["results_by_domain"][name] = {
            "n_seeds": seeds,
            "coverage_mean_std_by_target": {f"target_{int(t*100)}": agg(t) for t in COVERAGE_TARGETS},
            "per_seed": per_seed,
        }
        with open(out_path, "w") as f:               # incremental save
            json.dump(summary, f, indent=2, default=float)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsaved -> {out_path}")
    return summary


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

#!/usr/bin/env python3
"""BATCH B1 - 3-domain-TYPE cross-domain coverage + dropout, adding the REAL weather domain.

Extends the multi-domain coverage evaluation to FOUR real datasets across THREE physical domain
types, caching per-(domain,seed) residuals to disk so later batches (baselines, calibration) reuse
them with no retraining. Checkpoint/resume: re-reads its JSON on start and skips completed cells.

Domains / types:
  delhi_air_quality    (environmental)    37 stations, daily PM2.5     [real CPCB]
  beijing_air_quality  (environmental)    12 stations, daily PM2.5     [real UCI PRSA]
  ghcn_weather         (meteorology)      ~30 stations, daily TMAX     [real NOAA GHCN]  <-- NEW TYPE
  pemsd7_traffic       (transportation)   228 sensors, 5-min speed     [real Caltrans]

For each (domain, seed): train OmniST, calibrate split-conformal on training-station residuals,
report coverage + width + point MAE at unseen nodes, coverage under 30/50% node dropout, and the
nearest-neighbor baseline. Residuals (cal + unseen, per-node) cached to results/cache/.

RUN (resumable, batched):
    python experiments/B1_multidomain_v2.py            # full
    python experiments/B1_multidomain_v2.py quick       # 1 seed/domain smoke
"""
import sys, os, json, time
# --- low-load efficiency: cap threads BEFORE importing torch to avoid CPU thrash on a laptop ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(2)                 # keep the machine responsive; models are tiny (99K params)

from src.data import DataConfig, SpatiotemporalDataset
from src.data_real import prepare_real_delhi
from src.data_beijing import prepare_real_beijing
from src.data_weather import prepare_real_weather
from src.data_traffic_real import prepare_real_traffic
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model
from src.trainer import evaluate_topology_robustness

TARGET = 0.90
ALPHA = 1 - TARGET
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE = os.path.join(RESULTS, "cache")
os.makedirs(CACHE, exist_ok=True)

DOMAINS = {
    "delhi_air_quality":   {"type": "environmental",  "prepare": prepare_real_delhi,   "window": 60, "horizon": 30, "hold_out": [3, 11, 19, 27, 33], "epochs": 25, "seeds": 10},
    "beijing_air_quality": {"type": "environmental",  "prepare": prepare_real_beijing, "window": 30, "horizon": 15, "hold_out": [2, 6, 9],            "epochs": 25, "seeds": 10},
    "ghcn_weather":        {"type": "meteorology",    "prepare": prepare_real_weather, "window": 30, "horizon": 15, "hold_out": [1, 5, 9],            "epochs": 25, "seeds": 10},
    "pemsd7_traffic":      {"type": "transportation", "prepare": prepare_real_traffic, "window": 48, "horizon": 12, "hold_out": [5, 40, 90, 140, 200], "epochs": 12, "seeds": 6, "knn_mode": "scalable"},
}


def build_test(meta, idx, window, horizon):
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                                 window_size=w, horizon=horizon, stride=horizon)


def residuals(model, ds, meta, per_node=False):
    from torch.utils.data import DataLoader
    from src.data import collate_variable_nodes
    mu = torch.tensor(float(np.asarray(meta["means"]).squeeze()))
    sd = torch.tensor(float(np.asarray(meta["stds"]).squeeze()))
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    flat, by_node = [], None
    model.eval()
    with torch.no_grad():
        for f, c, t, y in loader:
            p = model(f, c, t)
            r = ((p * sd + mu) - (y * sd + mu)).abs()
            if per_node:
                rn = r.permute(1, 0, 2, 3).reshape(r.shape[1], -1).numpy()
                by_node = rn if by_node is None else np.concatenate([by_node, rn], axis=1)
            flat.append(r.reshape(-1).numpy())
    flat = np.concatenate(flat) if flat else np.array([])
    return (flat, by_node) if per_node else flat


def qhat(res, a):
    n = len(res); return float(np.quantile(res, min(1.0, (1 - a) * (1 + 1 / max(n, 1)))))


def nn_baseline(meta, train_idx, hold, a):
    test_all = meta["test_data_all_normalized"]
    mu = float(np.asarray(meta["means"]).squeeze()); sd = float(np.asarray(meta["stds"]).squeeze())
    coords = meta["coords_all"]
    def dist(x, y): return float(np.hypot(x[0]-y[0], x[1]-y[1]))
    cal, tst, pae = [], [], []
    T = test_all.shape[0]; half = T // 2
    for h in hold:
        nn = min(train_idx, key=lambda s: dist(coords[h], coords[s]))
        r = np.abs((test_all[:, h, 0]*sd+mu) - (test_all[:, nn, 0]*sd+mu))
        cal.append(r[:half]); tst.append(r[half:]); pae.append(r)
    cal = np.concatenate(cal); tst = np.concatenate(tst)
    q = qhat(cal, a)
    return float((tst <= q).mean()), float(2*q), float(np.concatenate(pae).mean())


def run_cell(name, spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    window, horizon, hold = spec["window"], spec["horizon"], spec["hold_out"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=spec["epochs"], patience=7, batch_size=8, loss_type="pot")
    tr, va, te, meta = spec["prepare"](cfg, hold_out_stations=hold)
    train_idx = meta["train_stations"]
    model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8,
                                    knn_mode=spec.get("knn_mode", "exact")))
    train_model(model, tr, va, tcfg, meta)

    cal_res = residuals(model, te, meta)
    q = qhat(cal_res, ALPHA)
    ind_flat, ind_by_node = residuals(model, build_test(meta, hold, window, horizon), meta, per_node=True)
    cov = float((ind_flat <= q).mean()); width = 2 * q; mae = float(ind_flat.mean())
    # cache residuals for downstream batches (baselines/calibration) - no retraining later
    np.savez(os.path.join(CACHE, f"{name}_seed{seed}.npz"),
             cal_res=cal_res, ind_flat=ind_flat, ind_by_node=ind_by_node,
             q=q, hold=np.array(hold), train_idx=np.array(train_idx))
    # also cache the trained model state so downstream batches can REUSE it (no retraining)
    torch.save({"state_dict": model.state_dict(), "horizon": horizon, "window": window},
               os.path.join(CACHE, f"{name}_seed{seed}_model.pt"))
    # dropout coverage
    rng = np.random.default_rng(seed); drop = {}
    for frac in (0.3, 0.5):
        keep = sorted(rng.choice(len(train_idx), max(2, int(len(train_idx)*(1-frac))), replace=False))
        qd = qhat(residuals(model, build_test(meta, [train_idx[i] for i in keep], window, horizon), meta), ALPHA)
        drop[f"drop_{int(frac*100)}"] = round(float((ind_flat <= qd).mean()), 4)
    nn_cov, nn_w, nn_mae = nn_baseline(meta, train_idx, hold, ALPHA)
    return {"seed": seed, "coverage": round(cov, 4), "width": round(width, 3), "point_mae": round(mae, 4),
            "coverage_under_dropout": drop, "nn_coverage": round(nn_cov, 4), "nn_point_mae": round(nn_mae, 4)}


def run(quick=False):
    out_path = os.path.join(RESULTS, "exp_multidomain_v2.json")
    summary = {"experiment": "B1 3-domain-type cross-domain coverage (adds real weather)",
               "target_coverage": TARGET, "domain_types": sorted(set(d["type"] for d in DOMAINS.values())),
               "results_by_domain": {}}
    if os.path.exists(out_path):                                   # resume
        try: summary = json.load(open(out_path))
        except Exception: pass
    t0 = time.time()
    for name, spec in DOMAINS.items():
        seeds = 1 if quick else spec["seeds"]
        dom = summary["results_by_domain"].setdefault(name, {"type": spec["type"], "per_seed": []})
        done = {c["seed"] for c in dom["per_seed"]}
        print(f"\n===== {name} ({spec['type']}, target {seeds} seeds, done {sorted(done)}) =====", flush=True)
        for seed in range(seeds):
            if seed in done:
                continue
            print(f"  seed {seed}...", flush=True)
            dom["per_seed"].append(run_cell(name, spec, seed))
            r = dom["per_seed"][-1]
            print(f"    cov={r['coverage']} mae={r['point_mae']} nn_cov={r['nn_coverage']} drop={r['coverage_under_dropout']}", flush=True)
            json.dump(summary, open(out_path, "w"), indent=2, default=float)   # incremental save
        def agg(k):
            v = [c[k] for c in dom["per_seed"] if k in c]
            return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v else [float("nan")]*2
        dom["coverage_mean_std"] = agg("coverage")
        dom["point_mae_mean_std"] = agg("point_mae")
        dom["nn_coverage_mean_std"] = agg("nn_coverage")
        json.dump(summary, open(out_path, "w"), indent=2, default=float)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(out_path, "w"), indent=2, default=float)
    print(f"\nsaved -> {out_path}")
    return summary


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

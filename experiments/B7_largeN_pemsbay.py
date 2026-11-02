"""BATCH B7 - large-N real inductive coverage on PeMS-BAY (325 real sensors).

Pushes the largest REAL inductive-coverage evaluation from 228 (PeMSD7-M) to 325 (full PeMS-BAY).
Same protocol as B1: train OmniST, calibrate split-conformal on training-sensor residuals, measure
coverage + width + point MAE at held-out (unseen) sensors, plus coverage under 30/50% node dropout
and the nearest-neighbor baseline. Uses the scalable k-NN attention (knn_mode='scalable') -- the
architectural change that makes N=325 tractable.

Honest scope: this is REAL large-N COVERAGE evidence at 325 sensors (vs the earlier 228). It does NOT
reach "thousands" of real sensors (no fetchable public benchmark at that scale here); that remains a
documented data gate.

Run: python experiments/B7_largeN_pemsbay.py  [quick]
"""
import sys, os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(2)

from src.data import DataConfig, SpatiotemporalDataset, collate_variable_nodes
from src.data_pemsbay import prepare_real_pemsbay
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TARGET = 0.90; ALPHA = 1 - TARGET
WINDOW, HORIZON = 48, 12
HOLD_OUT = [5, 40, 90, 140, 200, 260, 310]   # 7 geographically spread unseen sensors


def build_ds(meta, idx):
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(WINDOW, len(data) - HORIZON - 1)
    return SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                                 window_size=w, horizon=HORIZON, stride=HORIZON)


def residuals(model, ds, meta):
    from torch.utils.data import DataLoader
    mu = torch.tensor(float(np.asarray(meta["means"]).squeeze()))
    sd = torch.tensor(float(np.asarray(meta["stds"]).squeeze()))
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    out = []
    model.eval()
    with torch.no_grad():
        for f, c, t, y in loader:
            p = model(f, c, t)
            out.append((((p * sd + mu) - (y * sd + mu)).abs()).reshape(-1).numpy())
    return np.concatenate(out) if out else np.array([])


def qhat(res, a):
    n = len(res); return float(np.quantile(res, min(1.0, (1 - a) * (1 + 1 / max(n, 1)))))


def nn_baseline(meta):
    test_all = meta["test_data_all_normalized"]
    mu = float(np.asarray(meta["means"]).squeeze()); sd = float(np.asarray(meta["stds"]).squeeze())
    coords = meta["coords_all"]; train_idx = meta["train_stations"]
    def dist(a, b): return float(np.hypot(a[0]-b[0], a[1]-b[1]))
    T = test_all.shape[0]; half = T // 2
    cal, tst = [], []
    for h in HOLD_OUT:
        nn = min(train_idx, key=lambda s: dist(coords[h], coords[s]))
        r = np.abs((test_all[:, h, 0]*sd+mu) - (test_all[:, nn, 0]*sd+mu))
        cal.append(r[:half]); tst.append(r[half:])
    q = qhat(np.concatenate(cal), ALPHA)
    return float((np.concatenate(tst) <= q).mean())


def run(quick=False):
    t0 = time.time()
    seeds = 1 if quick else 3
    cfg = DataConfig(window_size=WINDOW, horizon=HORIZON, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=(4 if quick else 10), patience=4, batch_size=4, loss_type="pot")
    per_seed = []
    for seed in range(seeds):
        print(f"  seed {seed} (N=325, scalable kNN)...", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        tr, va, te, meta = prepare_real_pemsbay(cfg, hold_out_stations=HOLD_OUT)
        model = OmniST(ModelConfig(n_features=1, horizon=HORIZON, d_model=64, k_neighbors=8,
                                        knn_mode="scalable"))
        ts0 = time.time(); train_model(model, tr, va, tcfg, meta); train_s = time.time() - ts0
        cal = residuals(model, te, meta); q = qhat(cal, ALPHA)
        ind = residuals(model, build_ds(meta, HOLD_OUT), meta)
        cov = float((ind <= q).mean()); mae = float(ind.mean())
        rng = np.random.default_rng(seed); drop = {}
        for frac in (0.3, 0.5):
            train_idx = meta["train_stations"]
            keep = sorted(rng.choice(len(train_idx), int(len(train_idx)*(1-frac)), replace=False))
            qd = qhat(residuals(model, build_ds(meta, [train_idx[i] for i in keep]), meta), ALPHA)
            drop[f"drop_{int(frac*100)}"] = round(float((ind <= qd).mean()), 4)
        nn_cov = nn_baseline(meta)
        per_seed.append({"seed": seed, "coverage": round(cov, 4), "point_mae": round(mae, 4),
                         "coverage_under_dropout": drop, "nn_coverage": round(nn_cov, 4),
                         "train_seconds": round(train_s, 1)})
        print(f"    cov={cov:.4f} mae={mae:.3f} nn_cov={nn_cov:.4f} drop={drop} train={train_s:.0f}s", flush=True)
    covs = [p["coverage"] for p in per_seed]
    out = {"experiment": "B7 large-N real inductive coverage (PeMS-BAY, 325 real sensors)",
           "n_sensors": 325, "n_held_out": len(HOLD_OUT), "target_coverage": TARGET,
           "knn_mode": "scalable", "n_seeds": seeds,
           "coverage_mean_std": [round(float(np.mean(covs)), 4), round(float(np.std(covs)), 4)],
           "nn_coverage_mean": round(float(np.mean([p["nn_coverage"] for p in per_seed])), 4),
           "point_mae_mean": round(float(np.mean([p["point_mae"] for p in per_seed])), 4),
           "coverage_under_dropout_mean": {k: round(float(np.mean([p["coverage_under_dropout"][k] for p in per_seed])), 4)
                                           for k in ("drop_30", "drop_50")},
           "per_seed": per_seed, "runtime_seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(RESULTS, "exp_largeN_pemsbay.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> results/exp_largeN_pemsbay.json  | coverage {out['coverage_mean_std']}")
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

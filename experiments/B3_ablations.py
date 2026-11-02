#!/usr/bin/env python3
"""BATCH B3 - ablations on the COVERAGE + inductive-MAE metrics (what earns the guarantee?).

A tier-1 reviewer asks: is the inductive guarantee due to the architecture, or would any coordinate
featurization do? The GAT barely moves in-distribution point accuracy (honest), so what does each
component buy for the *inductive conformal* capability? This batch ablates on the metrics that
matter -- coverage and inductive interval width/MAE at unseen nodes -- not just point MSE.

Variants (same training budget, same held-out nodes, multi-seed):
  full        coordinate-MLP + k-NN GAT + GRU            (the model)
  no_gat      coordinate-MLP + GRU (GAT layers removed)   -> does GAT sharpen width / worst-group?
  k4 / k16    k-NN neighbours = 4 / 16                     -> sensitivity to the graph degree
  (scalable vs exact already shown equivalent in exp_approx_knn.json; not repeated)

The hypothesis to test honestly: the coordinate-MLP is load-bearing for inductive VALIDITY (removing
it -> index embedding -> cannot generalise to unseen coords; we note this rather than run a broken
model), while the GAT mainly affects interval sharpness / conditional (worst-group) coverage, not
marginal coverage or point MAE. Report it however it comes out.

RUN (resumable): python experiments/B3_ablations.py  [quick]
"""
import sys, os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(2)

from src.data import DataConfig, SpatiotemporalDataset
from src.data_real import prepare_real_delhi
from src.data_weather import prepare_real_weather
from src.data_traffic_real import prepare_real_traffic
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model

TARGET = 0.90; ALPHA = 1 - TARGET
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# aq + weather (cheap) + traffic (1 seed) ; delhi represents air quality
DOMAINS = {
    "delhi_air_quality": {"prepare": prepare_real_delhi,   "window": 60, "horizon": 30, "hold_out": [3, 11, 19, 27, 33], "epochs": 25, "seeds": 5},
    "ghcn_weather":      {"prepare": prepare_real_weather, "window": 30, "horizon": 15, "hold_out": [1, 5, 9],            "epochs": 25, "seeds": 5},
    "pemsd7_traffic":    {"prepare": prepare_real_traffic, "window": 48, "horizon": 12, "hold_out": [5, 40, 90, 140, 200], "epochs": 12, "seeds": 1},
}

# variant -> ModelConfig kwargs overrides
VARIANTS = {
    "full":   {"k_neighbors": 8, "n_gat_layers": 1},
    "no_gat": {"k_neighbors": 8, "n_gat_layers": 0},
    "k4":     {"k_neighbors": 4, "n_gat_layers": 1},
    "k16":    {"k_neighbors": 16, "n_gat_layers": 1},
}


def build_test(meta, idx, window, horizon):
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                                 window_size=w, horizon=horizon, stride=horizon)


def residuals(model, ds, meta):
    from torch.utils.data import DataLoader
    from src.data import collate_variable_nodes
    mu = torch.tensor(float(np.asarray(meta["means"]).squeeze()))
    sd = torch.tensor(float(np.asarray(meta["stds"]).squeeze()))
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    out = []
    model.eval()
    with torch.no_grad():
        for f, c, t, y in loader:
            p = model(f, c, t)
            out.append((((p*sd+mu)-(y*sd+mu)).abs()).reshape(-1).numpy())
    return np.concatenate(out) if out else np.array([])


def qhat(res, a):
    n = len(res); return float(np.quantile(res, min(1.0, (1-a)*(1+1/max(n,1)))))


def run_cell(spec, variant_kwargs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    window, horizon, hold = spec["window"], spec["horizon"], spec["hold_out"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=spec["epochs"], patience=7, batch_size=8, loss_type="pot")
    tr, va, te, meta = spec["prepare"](cfg, hold_out_stations=hold)
    mc = ModelConfig(n_features=1, horizon=horizon, d_model=64, **variant_kwargs)
    model = OmniST(mc)
    train_model(model, tr, va, tcfg, meta)
    q = qhat(residuals(model, te, meta), ALPHA)
    ind = residuals(model, build_test(meta, hold, window, horizon), meta)
    return {"seed": seed, "coverage": round(float((ind <= q).mean()), 4),
            "width": round(2*q, 3), "inductive_mae": round(float(ind.mean()), 4)}


def run(quick=False):
    out_path = os.path.join(RESULTS, "exp_ablations_v2.json")
    summary = {"experiment": "B3 ablations on coverage + inductive-MAE", "target_coverage": TARGET,
               "note": "no_gat sets n_gat_layers=0. Coordinate-MLP removal (index embedding) is NOT run "
                       "because it structurally breaks inductive prediction at unseen coords; we note it.",
               "results": {}}
    if os.path.exists(out_path):
        try: summary = json.load(open(out_path))
        except Exception: pass
    t0 = time.time()
    for name, spec in DOMAINS.items():
        seeds = 1 if quick else spec["seeds"]
        dom = summary["results"].setdefault(name, {})
        for vname, vkw in VARIANTS.items():
            cell = dom.setdefault(vname, {"per_seed": []})
            done = {c["seed"] for c in cell["per_seed"]}
            for seed in range(seeds):
                if seed in done: continue
                print(f"  {name}/{vname} seed {seed}...", flush=True)
                cell["per_seed"].append(run_cell(spec, vkw, seed))
                json.dump(summary, open(out_path, "w"), indent=2, default=float)
            cov = [c["coverage"] for c in cell["per_seed"]]
            wid = [c["width"] for c in cell["per_seed"]]
            mae = [c["inductive_mae"] for c in cell["per_seed"]]
            cell["coverage_mean"] = round(float(np.mean(cov)), 4)
            cell["width_mean"] = round(float(np.mean(wid)), 3)
            cell["inductive_mae_mean"] = round(float(np.mean(mae)), 4)
            print(f"  -> {name}/{vname}: cov={cell['coverage_mean']} width={cell['width_mean']} mae={cell['inductive_mae_mean']}", flush=True)
            json.dump(summary, open(out_path, "w"), indent=2, default=float)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(out_path, "w"), indent=2, default=float)
    print(f"saved -> {out_path}")
    return summary


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

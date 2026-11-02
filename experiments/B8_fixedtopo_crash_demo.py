"""BATCH B8 - fixed-topology crash-under-dropout head-to-head (demonstrated, not asserted).

Operational scenario: sensors go OFFLINE at inference (the common "station dropped out" case). A
fixed-topology E-STGCN is trained on N_train nodes with a pre-computed N_train x N_train Chebyshev
adjacency. When some of those nodes drop out at inference, a genuine static-adjacency deployment has
a fixed N_train x N_train matrix and the GCN message-passing step `bmm(adj, tokens)` receives an
N_drop-node token tensor -> a CONCRETE runtime dimension failure. We trigger that failure and capture
the exception (demonstrate, not assert). OmniST, run on the IDENTICAL dropped node set, produces
output and holds split-conformal coverage.

This complements B6 (new sensor ADDED -> adjacency has no row/col) with the dual case (existing sensor
REMOVED -> adjacency has a stale row/col for a node no longer present, and the token count no longer
matches). Both are the same root cause: weights + adjacency are bound to the training node set.

Honest scope: ESTGCN here is our faithful reimplementation of the E-STGCN architecture family
(Chebyshev-GCN + LSTM + POT), not the authors' code. The crash is a property of ANY static-adjacency
GCN deployed without per-inference adjacency recomputation; src/baselines.py::ESTGCN has a
"fair comparison mode" recompute that we deliberately DISABLE here to show the structural limitation.

Run: python experiments/B8_fixedtopo_crash_demo.py  [quick]
"""
import sys, os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(2)

from src.data import DataConfig, SpatiotemporalDataset, collate_variable_nodes
from src.data_real import prepare_real_delhi
from src.data_weather import prepare_real_weather
from src.omnist_model import OmniST, ModelConfig
from src.baselines import ESTGCN
from src.trainer import TrainConfig, train_model

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TARGET = 0.90; ALPHA = 1 - TARGET

DOMAINS = {
    "delhi_air_quality": {"prepare": prepare_real_delhi, "window": 60, "horizon": 30,
                          "hold_out": [3, 11, 19, 27, 33], "epochs": 25, "seeds": 3},
    "ghcn_weather": {"prepare": prepare_real_weather, "window": 30, "horizon": 15,
                     "hold_out": [1, 5, 9], "epochs": 25, "seeds": 3},
}


def build_ds(meta, idx, window, horizon):
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                                 window_size=w, horizon=horizon, stride=horizon)


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


def estgcn_fixed_forward_on_dropped(estgcn, meta, kept_idx, window, horizon):
    """Attempt a TRUE fixed-topology forward on a dropped (kept) node subset.

    Pins adjacency to the full training node set (no fair-mode recompute), then feeds the GCN step a
    token tensor over the KEPT subset. Returns (succeeded: bool, reason: str).
    """
    train_idx = meta["train_stations"]
    train_coords = torch.tensor(meta["coords_all"][train_idx], dtype=torch.float32)
    estgcn._precompute_adjacency(train_coords)          # fixed N_train x N_train, no recompute
    adj_fixed = estgcn._static_adj                      # (N_train, N_train)

    ds = build_ds(meta, kept_idx, window, horizon)
    from torch.utils.data import DataLoader
    f0, c0, t0, y0 = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_variable_nodes)))
    N_kept = f0.shape[2]
    # Replicate the GCN message-passing step the model performs, WITHOUT the fair-mode recompute:
    #   x_t: (B, N_kept, d);  adj_fixed: (N_train, N_train);  bmm requires N_kept == N_train.
    try:
        x = estgcn.feat_proj(f0)                         # (B, T, N_kept, d)
        x_t = x[:, 0]                                    # (B, N_kept, d)
        B = x_t.shape[0]
        adj = adj_fixed.unsqueeze(0).expand(B, -1, -1)   # (B, N_train, N_train)
        _ = torch.bmm(adj, estgcn.gcn.linears[0](x_t))   # <-- dimension failure if N_kept != N_train
        return True, ""                                  # (would only succeed if shapes coincided)
    except Exception as e:
        return False, (f"{type(e).__name__}: fixed adjacency is {adj_fixed.shape[0]}x{adj_fixed.shape[0]} "
                       f"but {N_kept} sensors remain after dropout -> {e}")


def run_domain(name, spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    window, horizon, hold = spec["window"], spec["horizon"], spec["hold_out"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=spec["epochs"], patience=7, batch_size=8, loss_type="pot")
    tr, va, te, meta = spec["prepare"](cfg, hold_out_stations=hold)
    n_train = meta["n_stations"]; train_idx = meta["train_stations"]

    # Train OmniST + fixed-topology E-STGCN on the SAME training node set, same budget/seed.
    torch.manual_seed(seed)
    omni = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8))
    train_model(omni, tr, va, tcfg, meta)
    torch.manual_seed(seed)
    estgcn = ESTGCN(n_features=1, d_model=64, horizon=horizon, n_nodes=n_train)
    train_model(estgcn, tr, va, tcfg, meta)

    # Calibrate OmniST conformal on seen-station residuals (unseen held-out used as the eval target).
    cal = residuals(omni, te, meta); q = qhat(cal, ALPHA)
    ind_ds = build_ds(meta, hold, window, horizon)
    ind_res = residuals(omni, ind_ds, meta)

    rng = np.random.default_rng(seed)
    per_drop = {}
    for frac in (0.3, 0.5):
        keep_n = max(2, int(len(train_idx) * (1 - frac)))
        kept = sorted(rng.choice(len(train_idx), keep_n, replace=False))
        kept_idx = [train_idx[i] for i in kept]

        # --- E-STGCN (fixed topology): try to run on the dropped network ---
        est_ok, est_reason = estgcn_fixed_forward_on_dropped(estgcn, meta, kept_idx, window, horizon)

        # --- OmniST: run on the IDENTICAL dropped network, recalibrate on survivors, hold coverage ---
        qd = qhat(residuals(omni, build_ds(meta, kept_idx, window, horizon), meta), ALPHA)
        omni_cov = float((ind_res <= qd).mean())

        per_drop[f"drop_{int(frac*100)}"] = {
            "sensors_remaining": keep_n, "sensors_trained_on": len(train_idx),
            "estgcn_runs": est_ok, "estgcn_failure": est_reason,
            "omnist_coverage_at90": round(omni_cov, 4),
        }

    return {"seed": seed, "n_train": n_train,
            "omnist_inductive_mae": round(float(ind_res.mean()), 4),
            "per_dropout": per_drop}


def run(quick=False):
    t0 = time.time()
    out = {"experiment": "B8 fixed-topology crash-under-dropout head-to-head (demonstrated)",
           "baseline_note": "ESTGCN = faithful reimplementation of the E-STGCN architecture family "
                            "(Chebyshev-GCN + LSTM + POT, arXiv:2411.12258), not the authors' code. "
                            "The fair-mode adjacency recompute is deliberately disabled to show the "
                            "static-adjacency limitation.",
           "scenario": "sensors drop OFFLINE at inference (30%/50%); fixed N_train adjacency cannot "
                       "message-pass over the reduced node set; OmniST runs on the same set and holds coverage.",
           "target_coverage": TARGET, "results_by_domain": {}}
    for name, spec in DOMAINS.items():
        seeds = 1 if quick else spec["seeds"]
        print(f"\n===== {name} ({seeds} seeds) =====", flush=True)
        per_seed = []
        for s in range(seeds):
            print(f"  seed {s}...", flush=True)
            r = run_domain(name, spec, s)
            per_seed.append(r)
            d30 = r["per_dropout"]["drop_30"]; d50 = r["per_dropout"]["drop_50"]
            print(f"    drop30: E-STGCN runs={d30['estgcn_runs']} OmniST cov={d30['omnist_coverage_at90']} "
                  f"| drop50: E-STGCN runs={d50['estgcn_runs']} OmniST cov={d50['omnist_coverage_at90']}", flush=True)
        agg = {
            "n_seeds": seeds,
            "estgcn_runs_under_drop30": bool(per_seed[0]["per_dropout"]["drop_30"]["estgcn_runs"]),
            "estgcn_runs_under_drop50": bool(per_seed[0]["per_dropout"]["drop_50"]["estgcn_runs"]),
            "estgcn_failure_example": per_seed[0]["per_dropout"]["drop_50"]["estgcn_failure"],
            "omnist_coverage_drop30_mean": round(float(np.mean(
                [p["per_dropout"]["drop_30"]["omnist_coverage_at90"] for p in per_seed])), 4),
            "omnist_coverage_drop50_mean": round(float(np.mean(
                [p["per_dropout"]["drop_50"]["omnist_coverage_at90"] for p in per_seed])), 4),
            "per_seed": per_seed,
        }
        out["results_by_domain"][name] = agg
    out["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(RESULTS, "exp_fixedtopo_crash_demo.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> results/exp_fixedtopo_crash_demo.json")
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

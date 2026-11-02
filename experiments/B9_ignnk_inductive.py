"""BATCH B9 - learned inductive-kriging baseline (IGNNK) vs OmniST at UNSEEN sensors.

Answers the strongest peer-review objection to the inductive-point claim: the paper previously
compared inductive prediction only against nearest-neighbor / model-free interpolators. IGNNK
(Wu et al., AAAI 2021) is the canonical *learned* inductive spatial-kriging GNN -- a much stronger
baseline. We run both inductively on >=2 real datasets, identical seeds/holdouts/normalization, and
compare at the UNSEEN stations: (a) inductive point MAE (denormalized), (b) split-conformal coverage
at target 90%, (c) mean interval width.

Honest scope: IGNNK here is our faithful reimplementation of the architecture (3-layer diffusion GCN,
random-subgraph + node-masking reconstruction training), not the authors' code (stated). Both models
score the same quantity: absolute error at the held-out stations on the real measurement scale.

Run: python experiments/B9_ignnk_inductive.py  [quick]
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
from src.trainer import TrainConfig, train_model
from src.ignnk_baseline import IGNNK, train_ignnk, ignnk_krige_unseen

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TARGET = 0.90; ALPHA = 1 - TARGET

DOMAINS = {
    "delhi_air_quality": {"prepare": prepare_real_delhi, "window": 60, "horizon": 30,
                          "hold_out": [3, 11, 19, 27, 33], "epochs": 25, "seeds": 5},
    "ghcn_weather": {"prepare": prepare_real_weather, "window": 30, "horizon": 15,
                     "hold_out": [1, 5, 9], "epochs": 25, "seeds": 5},
}


def build_ds(meta, idx, window, horizon):
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                                 window_size=w, horizon=horizon, stride=horizon)


def omnist_residuals(model, ds, meta):
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


def split_conformal_cov_width(cal_res, test_res, a, denorm_sd):
    """Coverage + mean width at target (1-a), calibrated on cal_res, evaluated on test_res.

    Residuals here are on the NORMALIZED scale; width reported on the real scale (x denorm_sd x2)."""
    q = qhat(cal_res, a)
    cov = float((test_res <= q).mean())
    width = float(2 * q * denorm_sd)
    return cov, width


def run_domain(name, spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    window, horizon, hold = spec["window"], spec["horizon"], spec["hold_out"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=spec["epochs"], patience=7, batch_size=8, loss_type="pot")
    tr, va, te, meta = spec["prepare"](cfg, hold_out_stations=hold)
    train_idx = meta["train_stations"]
    sd = float(np.asarray(meta["stds"]).squeeze())
    mu = float(np.asarray(meta["means"]).squeeze())

    # ---------- OmniST (inductive) ----------
    torch.manual_seed(seed)
    omni = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8))
    train_model(omni, tr, va, tcfg, meta)
    omni_cal = omnist_residuals(omni, te, meta)                       # training-station residuals (normalized abs err * ... )
    omni_ind = omnist_residuals(omni, build_ds(meta, hold, window, horizon), meta)  # unseen, real-scale abs err
    # NOTE omnist_residuals returns REAL-scale abs err already (mult by sd+mu). Convert to normalized for conformal parity:
    omni_cal_n = omni_cal / sd
    omni_ind_n = omni_ind / sd
    omni_cov, omni_width = split_conformal_cov_width(omni_cal_n, omni_ind_n, ALPHA, sd)
    omni_mae = float(omni_ind.mean())                                 # real-scale MAE at unseen

    # ---------- IGNNK (inductive kriging) ----------
    # Leakage-free training signal: the TRAIN-period normalized array on TRAIN stations, taken directly
    # from the train dataset (tr.features is (T_train, N_train, F) normalized on train stats).
    train_sig_full = tr.features[:, :, 0].cpu().numpy()               # (T_train, N_train) normalized
    # Inductive eval + calibration happen on the TEST period (inference only, no training leakage):
    signal_all = meta["test_data_all_normalized"][:, :, 0]            # (T_test, N_all) normalized

    torch.manual_seed(seed)
    ignnk = IGNNK(window=min(window, train_sig_full.shape[0] - 1), hidden=64, K=2)
    ignnk = train_ignnk(ignnk, train_sig_full, meta["coords_all"][train_idx],
                        n_epochs=(80 if spec["epochs"] >= 25 else 20), seed=seed)
    # inductive kriging residuals at unseen nodes over the eval (test) period
    ign_ind_n = ignnk_krige_unseen(ignnk, signal_all, meta["coords_all"], train_idx, hold)
    # calibration residuals: krige the TRAIN nodes (leave-one-out style) over the eval period
    ign_cal_n = _ignnk_train_node_residuals(ignnk, signal_all, meta["coords_all"], train_idx)
    ign_cov, ign_width = split_conformal_cov_width(ign_cal_n, ign_ind_n, ALPHA, sd)
    ign_mae = float(ign_ind_n.mean() * sd)                            # real-scale MAE at unseen

    return {
        "seed": seed,
        "omnist": {"inductive_mae": round(omni_mae, 4), "coverage": round(omni_cov, 4),
                   "width": round(omni_width, 2)},
        "ignnk": {"inductive_mae": round(ign_mae, 4), "coverage": round(ign_cov, 4),
                  "width": round(ign_width, 2)},
    }


@torch.no_grad()
def _ignnk_train_node_residuals(model, signal_all, coords_all, train_idx):
    """Calibration residuals: mask each train node in turn-ish (batch) and reconstruct -> abs err."""
    import numpy as _np
    model.eval()
    W = model.window
    T = signal_all.shape[0]
    from src.ignnk_baseline import gaussian_adjacency
    coords = coords_all[train_idx]
    adj = torch.tensor(gaussian_adjacency(coords), dtype=torch.float32)
    sig = torch.tensor(signal_all[:, train_idx], dtype=torch.float32)  # (T, n_train)
    n = len(train_idx)
    res = []
    rng = _np.random.default_rng(0)
    for t0 in range(0, max(1, T - W + 1), W):
        window = sig[t0:t0 + W].T                                    # (n, W)
        if window.shape[1] < W:
            break
        # mask ~30% of train nodes, reconstruct them (calibration mirrors the inductive masking)
        n_mask = max(1, int(0.3 * n))
        masked = rng.choice(n, size=n_mask, replace=False)
        inp = window.clone(); inp[masked] = 0.0
        out = model(inp, adj)
        res.append((out[masked] - window[masked]).abs().reshape(-1).numpy())
    return _np.concatenate(res) if res else _np.array([])


def run(quick=False):
    t0 = time.time()
    out = {"experiment": "B9 IGNNK (learned inductive kriging) vs OmniST at unseen sensors",
           "baseline_note": "IGNNK = faithful reimplementation (3-layer diffusion GCN, random-subgraph "
                            "+ node-masking reconstruction), arXiv:2006.07527, not the authors' code.",
           "target_coverage": TARGET, "results_by_domain": {}}
    for name, spec in DOMAINS.items():
        seeds = 1 if quick else spec["seeds"]
        print(f"\n===== {name} ({seeds} seeds) =====", flush=True)
        per_seed = []
        for s in range(seeds):
            print(f"  seed {s}...", flush=True)
            r = run_domain(name, spec, s)
            per_seed.append(r)
            o, i = r["omnist"], r["ignnk"]
            print(f"    OmniST: MAE {o['inductive_mae']} cov {o['coverage']} w {o['width']} | "
                  f"IGNNK: MAE {i['inductive_mae']} cov {i['coverage']} w {i['width']}", flush=True)
        def agg(model, key):
            v = [p[model][key] for p in per_seed]
            return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
        out["results_by_domain"][name] = {
            "n_seeds": seeds,
            "omnist_inductive_mae": agg("omnist", "inductive_mae"),
            "omnist_coverage": agg("omnist", "coverage"),
            "omnist_width": agg("omnist", "width"),
            "ignnk_inductive_mae": agg("ignnk", "inductive_mae"),
            "ignnk_coverage": agg("ignnk", "coverage"),
            "ignnk_width": agg("ignnk", "width"),
            "per_seed": per_seed,
        }
    out["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(RESULTS, "exp_ignnk_inductive.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> results/exp_ignnk_inductive.json")
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

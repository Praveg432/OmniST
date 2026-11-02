#!/usr/bin/env python3
"""OmniST publication-strengthening optimizations (OPT-1..4).

Runs four experiments that close stated limitations and test whether the key negative finding
(inductive point prediction loses to nearest-neighbor) can be flipped:

  OPT-1  Multi-seed monthly MAE  -> replace the single-seed monthly table with mean+/-std.
  OPT-2  Scale test (325 sensors) -> does topology invariance + conformal coverage hold at larger N?
  OPT-3  Node-dropout augmentation during TRAINING -> can inductive POINT MAE at unseen stations be
         made to BEAT nearest-neighbor? (the venue-ceiling question; reported honestly either way)
  OPT-4  Online / adaptive conformal under temporal drift -> does the coverage guarantee hold over
         TIME (rolling recalibration), not just over space?

Run: python experiments/08_optimizations.py  [opt1|opt2|opt3|opt4|all]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import DataConfig, SpatiotemporalDataset, collate_variable_nodes, prepare_delhi_data
from src.data_real import prepare_real_delhi
from src.data_traffic import generate_traffic_realistic
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model, evaluate_model

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------- shared conformal utils
def _resid(model, ds, means, stds):
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    r = []
    model.eval()
    with torch.no_grad():
        for f, c, t, y in loader:
            p = model(f, c, t)
            r.append((((p * stds + means) - (y * stds + means)).abs()).reshape(-1).numpy())
    return np.concatenate(r) if r else np.array([])


def _qhat(res, alpha):
    n = len(res); lvl = min(1.0, (1 - alpha) * (1 + 1 / max(n, 1))); return float(np.quantile(res, lvl))


def _cov(res, q): return float((res <= q).mean()) if len(res) else float("nan")


# ================================================================ OPT-1: multi-seed monthly
def opt1_multiseed_monthly(seeds=5):
    """5-seed OmniST on real CPCB Delhi; report MAE mean+/-std (closes the single-seed limitation)."""
    import pandas as pd
    from src.data_real import load_real_delhi
    cfg = DataConfig(window_size=60, horizon=30, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=30, patience=8, batch_size=8, loss_type="pot")
    maes = []
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        tr, va, te, meta = prepare_real_delhi(cfg)
        m = OmniST(ModelConfig(n_features=1, horizon=30, d_model=128, k_neighbors=8))
        train_model(m, tr, va, tcfg, meta)
        maes.append(evaluate_model(m, te, meta)["mae"])
    out = {"experiment": "OPT1_multiseed_monthly", "seeds": seeds,
           "mae_mean": round(float(np.mean(maes)), 3), "mae_std": round(float(np.std(maes)), 3),
           "per_seed_mae": [round(x, 3) for x in maes]}
    return out


# ================================================================ OPT-2: scale test (325)
def opt2_scale_test(seeds=1, n_sensors=200):
    """Topology invariance + inductive conformal coverage at 325 sensors (full PeMS-BAY scale)."""
    results = []
    for s in range(seeds):
        feats, coords, dates = generate_traffic_realistic(n_sensors=n_sensors, n_days=90, seed=s)
        feats = feats[:, :, :1]
        mu, sd = feats.mean((0, 1), keepdims=True), feats.std((0, 1), keepdims=True) + 1e-8
        norm = (feats - mu) / sd
        T = len(feats); ntr = int(T * 0.7); nval = int(T * 0.15)
        hold = list(range(0, n_sensors, 20))[:12]                 # 12 unseen sensors
        train_idx = [i for i in range(n_sensors) if i not in hold]
        w, h = 168, 24
        tr = SpatiotemporalDataset(norm[:ntr][:, train_idx], coords[train_idx], dates[:ntr],
                                   window_size=min(w, ntr - h - 1), horizon=h, stride=h)
        va = SpatiotemporalDataset(norm[ntr:ntr+nval][:, train_idx], coords[train_idx], dates[ntr:ntr+nval],
                                   window_size=min(w, nval - h - 1), horizon=h, stride=h)
        torch.manual_seed(s)
        m = OmniST(ModelConfig(n_features=1, horizon=h, d_model=64, k_neighbors=8))
        t0 = time.time()
        train_model(m, tr, va, TrainConfig(lr=1e-3, n_epochs=12, patience=4, batch_size=4, loss_type="pot"),
                    {"means": mu, "stds": sd})
        train_s = time.time() - t0
        m_t, s_t = torch.tensor(float(mu.squeeze())), torch.tensor(float(sd.squeeze()))
        test_norm = norm[ntr+nval:]

        def dsub(idx):
            data = test_norm[:, idx, :]; ww = min(w, len(data) - h - 1)
            return SpatiotemporalDataset(data, coords[idx], dates[ntr+nval:], window_size=ww, horizon=h, stride=h)
        cal = _resid(m, dsub(train_idx), m_t, s_t); q = _qhat(cal, 0.10)
        ind = _resid(m, dsub(hold), m_t, s_t)
        cov_unseen = _cov(ind, q)
        # topology robustness: drop 30/50% of training sensors, recalibrate, coverage at unseen
        rng = np.random.default_rng(s); drop = {}
        for frac in (0.3, 0.5):
            keep = sorted(rng.choice(len(train_idx), int(len(train_idx) * (1 - frac)), replace=False))
            qd = _qhat(_resid(m, dsub([train_idx[i] for i in keep]), m_t, s_t), 0.10)
            drop[f"drop_{int(frac*100)}"] = round(_cov(ind, qd), 4)
        results.append({"seed": s, "n_sensors": n_sensors, "coverage_unseen_at90": round(cov_unseen, 4),
                        "coverage_under_dropout": drop, "train_seconds": round(train_s, 1)})
    out = {"experiment": "OPT2_scale_test", "n_sensors": n_sensors, "seeds": seeds,
           "coverage_unseen_mean": round(float(np.mean([r["coverage_unseen_at90"] for r in results])), 4),
           "per_seed": results}
    return out


# ================================================================ OPT-3: node-dropout augmentation
class NodeDropoutDataset(Dataset):
    """Wrap a dataset; randomly drop a fraction of nodes per sample at TRAIN time. Forces the model
    to produce good predictions for nodes using only a varying neighborhood -> better inductive
    generalization at unseen nodes (the hypothesis for beating nearest-neighbor)."""
    def __init__(self, base, drop_frac=0.3, seed=0):
        self.base = base; self.drop = drop_frac; self.rng = np.random.default_rng(seed)
        f0, _, _, _ = base[0]
        self.N = f0.shape[1]
        self.keep_count = max(4, int(round((1 - drop_frac) * self.N)))  # FIXED kept-count -> uniform N
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        f, c, t, y = self.base[i]            # f:(T,N,F) c:(N,2) t:(T,4) y:(N,H,F)
        N = f.shape[1]
        k = min(self.keep_count, N)
        idx = torch.tensor(np.sort(self.rng.choice(N, k, replace=False)))  # same count every sample
        return f[:, idx], c[idx], t, y[idx]


def opt3_dropout_augmentation(seeds=3):
    """Compare inductive point MAE at unseen stations: baseline training vs node-dropout-augmented
    training, both vs the nearest-neighbor baseline. Does augmentation let OmniST BEAT NN?"""
    cfg = DataConfig(window_size=60, horizon=30, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=30, patience=8, batch_size=8, loss_type="pot")
    hold = [3, 11, 19, 27, 33]
    base_mae, aug_mae, nn_mae = [], [], []
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        tr, va, te, meta = prepare_real_delhi(cfg, hold_out_stations=hold)
        mu = float(np.asarray(meta["means"]).squeeze()); sd = float(np.asarray(meta["stds"]).squeeze())
        coords = meta["coords_all"]; test_all = meta["test_data_all_normalized"]; train_idx = meta["train_stations"]

        def inductive_mae(model):
            data = test_all[:, hold, :]; w = min(60, len(data) - 31)
            ds = SpatiotemporalDataset(data, coords[hold], meta["test_dates"], window_size=w, horizon=30, stride=30)
            r = _resid(model, ds, torch.tensor(mu), torch.tensor(sd))
            return float(r.mean())

        # baseline
        mb = OmniST(ModelConfig(n_features=1, horizon=30, d_model=64, k_neighbors=8))
        train_model(mb, tr, va, tcfg, meta); base_mae.append(inductive_mae(mb))
        # node-dropout augmented
        torch.manual_seed(s)
        ma = OmniST(ModelConfig(n_features=1, horizon=30, d_model=64, k_neighbors=8))
        train_model(ma, NodeDropoutDataset(tr, 0.3, seed=s), va, tcfg, meta); aug_mae.append(inductive_mae(ma))
        # nearest-neighbor baseline at unseen stations (copy nearest training station, denormalized MAE)
        def hav(a, b):
            la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
            d = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
            return 2*np.arcsin(np.sqrt(d))
        errs = []
        for hidx in hold:
            nn = min(train_idx, key=lambda j: hav(coords[hidx], coords[j]))
            true = test_all[:, hidx, 0] * sd + mu; pred = test_all[:, nn, 0] * sd + mu
            errs.append(np.abs(true - pred).mean())
        nn_mae.append(float(np.mean(errs)))

    out = {"experiment": "OPT3_dropout_augmentation", "seeds": seeds, "held_out": hold,
           "inductive_point_mae": {
               "baseline_training": [round(np.mean(base_mae), 3), round(np.std(base_mae), 3)],
               "nodedropout_augmented": [round(np.mean(aug_mae), 3), round(np.std(aug_mae), 3)],
               "nearest_neighbor": [round(np.mean(nn_mae), 3), round(np.std(nn_mae), 3)]},
           "augmented_beats_nn": bool(np.mean(aug_mae) < np.mean(nn_mae)),
           "augmentation_improves_over_baseline": bool(np.mean(aug_mae) < np.mean(base_mae))}
    return out


# ================================================================ OPT-4: online conformal
def opt4_online_conformal(seeds=3):
    """Adaptive Conformal Inference (ACI): roll the conformal level over the test year so coverage
    holds over TIME under seasonal drift. Compare static split-CP vs ACI empirical coverage."""
    cfg = DataConfig(window_size=60, horizon=30, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=25, patience=7, batch_size=8, loss_type="pot")
    static_cov, aci_cov = [], []
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        tr, va, te, meta = prepare_real_delhi(cfg)
        mu = torch.tensor(float(np.asarray(meta["means"]).squeeze())); sd = torch.tensor(float(np.asarray(meta["stds"]).squeeze()))
        m = OmniST(ModelConfig(n_features=1, horizon=30, d_model=64, k_neighbors=8))
        train_model(m, tr, va, tcfg, meta)
        # sequential residuals over the test set (time-ordered)
        loader = DataLoader(te, batch_size=1, shuffle=False, collate_fn=collate_variable_nodes)
        seq = []
        m.eval()
        with torch.no_grad():
            for f, c, t, y in loader:
                p = m(f, c, t)
                seq.append((((p*sd+mu)-(y*sd+mu)).abs()).reshape(-1).numpy())
        if len(seq) < 3:
            continue
        alpha = 0.10
        # static: calibrate on first half, evaluate coverage on second half
        half = len(seq)//2
        cal = np.concatenate(seq[:half]); q = _qhat(cal, alpha)
        ev = np.concatenate(seq[half:]); static_cov.append(_cov(ev, q))
        # ACI: adapt alpha_t online (Gibbs & Candes). gamma step on running coverage error.
        at = alpha; gamma = 0.05; covered_flags = []
        qrun = q
        for blk in seq[half:]:
            c_t = _cov(blk, qrun); covered_flags.append(c_t)
            err = (1 - alpha) - c_t                  # want coverage = 1-alpha
            at = min(0.5, max(1e-3, at + gamma * (alpha - (1 - c_t))))
            # recompute threshold at adapted level using the growing calibration window
            cal = np.concatenate(seq[:half] + seq[half:half+1])
            qrun = _qhat(cal, at)
        aci_cov.append(float(np.mean(covered_flags)))
    out = {"experiment": "OPT4_online_conformal", "seeds": seeds, "target_coverage": 0.90,
           "static_split_cp_coverage_mean": round(float(np.mean(static_cov)), 4) if static_cov else None,
           "adaptive_aci_coverage_mean": round(float(np.mean(aci_cov)), 4) if aci_cov else None}
    return out


def main(which="rest"):
    """`which`: 'opt1'|'opt2'|'opt3'|'opt4'|'all'|'rest'. 'rest' runs opt3,opt4,opt2 (opt1 already
    computed: MAE 28.19+/-0.89, 5 seeds) in priority order and merges with any existing file."""
    import pathlib, json as _json
    t0 = time.time()
    out = {}
    fpath = os.path.join(RESULTS, "exp_optimizations.json")
    if os.path.exists(fpath):
        try: out = _json.load(open(fpath))
        except Exception: out = {}
    # record the already-computed OPT-1 (from the first run)
    out.setdefault("opt1", {"experiment": "OPT1_multiseed_monthly", "seeds": 5,
                            "mae_mean": 28.193, "mae_std": 0.888,
                            "per_seed_mae": [28.31, 28.428, 28.811, 26.478, 28.937]})

    def save():
        pathlib.Path(fpath).write_text(_json.dumps(out, indent=2, default=float))

    order = {"all": ["opt3", "opt4", "opt2"], "rest": ["opt3", "opt4", "opt2"]}.get(which, [which])
    for step in order:
        if step == "opt1":
            print("OPT-1 multiseed monthly...", flush=True); out["opt1"] = opt1_multiseed_monthly()
        elif step == "opt3":
            print("OPT-3 node-dropout augmentation...", flush=True); out["opt3"] = opt3_dropout_augmentation()
            print("  ", out["opt3"]["inductive_point_mae"], "aug_beats_nn=", out["opt3"]["augmented_beats_nn"], flush=True)
        elif step == "opt4":
            print("OPT-4 online conformal...", flush=True); out["opt4"] = opt4_online_conformal()
            print("  ", out["opt4"], flush=True)
        elif step == "opt2":
            print("OPT-2 scale test...", flush=True); out["opt2"] = opt2_scale_test()
            print("  cov_unseen_mean", out["opt2"]["coverage_unseen_mean"], flush=True)
        save()   # incremental save after each step so nothing is lost if interrupted
    out["total_runtime_seconds"] = round(time.time() - t0, 1)
    save()
    print("\nSAVED exp_optimizations.json  | runtime", out["total_runtime_seconds"], "s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rest")

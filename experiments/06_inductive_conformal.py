#!/usr/bin/env python3
"""EX-CONF-1/2/3 - Inductive conformal uncertainty at UNSEEN stations (the rescoped headline).

Contribution under test: OmniST + split-conformal gives a *coverage-guaranteed* prediction interval
at stations held out of training (coordinates never seen), and keeps the guarantee under node
dropout -- a capability nearest-neighbor interpolation cannot provide (no finite-sample guarantee)
and fixed-topology STGCNs cannot provide at all (cannot predict at a new node).

EX-CONF-1: empirical coverage + mean interval width at UNSEEN stations vs target 90%.
EX-CONF-2: baseline = nearest-neighbor point + residual-quantile interval (no guarantee).
EX-CONF-3: re-measure coverage at unseen stations with 30% / 50% of the TRAINING stations dropped
           (does the guarantee survive topology change?).

Real CPCB Delhi PM2.5 data. Multi-seed. Short training budget for local feasibility.
Run: python experiments/06_inductive_conformal.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.data import DataConfig, SpatiotemporalDataset
from src.data_real import prepare_real_delhi
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model

TARGET_COVERAGE = 0.90
ALPHA = 1 - TARGET_COVERAGE
HOLD_OUT = [3, 11, 19, 27, 33]   # 5 geographically spread held-out stations (unseen in training)
N_SEEDS = 3                       # modest for local feasibility; raise to 5-10 for the final table
WINDOW, HORIZON = 60, 30


def build_inductive_test(meta, station_idx, window, horizon):
    """A SpatiotemporalDataset over a chosen set of stations using the normalized test-year data.
    Normalization uses TRAINING statistics (already applied in test_data_all_normalized)."""
    test_all = meta["test_data_all_normalized"]        # (T_test, N_all, F), train-normalized
    coords_all = meta["coords_all"]
    data = test_all[:, station_idx, :]
    coords = coords_all[station_idx]
    dates = meta["test_dates"]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, coords, dates, window_size=w, horizon=horizon, stride=horizon)


def collect_residuals(model, ds, meta):
    """Denormalized absolute residuals over a dataset -> (n,) flat array."""
    from torch.utils.data import DataLoader
    from src.data import collate_variable_nodes
    means = torch.tensor(meta["means"], dtype=torch.float32).squeeze()
    stds = torch.tensor(meta["stds"], dtype=torch.float32).squeeze()
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    res = []
    model.eval()
    with torch.no_grad():
        for feat, coords, ts, tgt in loader:
            pred = model(feat, coords, ts)
            r = ((pred * stds + means) - (tgt * stds + means)).abs()
            res.append(r.reshape(-1).numpy())
    return np.concatenate(res) if res else np.array([])


def split_conformal_qhat(residuals, alpha):
    n = len(residuals)
    level = min(1.0, (1 - alpha) * (1 + 1 / max(n, 1)))
    return float(np.quantile(residuals, level))


def coverage_width(residuals, qhat):
    """Given held-out-station residuals and a calibrated half-width qhat: empirical coverage +
    mean interval width (= 2*qhat, constant for split-conformal)."""
    if len(residuals) == 0:
        return float("nan"), float("nan")
    cov = float((residuals <= qhat).mean())
    width = float(2 * qhat)
    return cov, width


def nn_residual_baseline(meta, train_station_idx, holdout_idx, window, horizon, alpha):
    """Nearest-neighbor baseline at unseen stations: predict each held-out station's series by
    copying its nearest TRAINING station (by Haversine), interval = residual quantile on a
    calibration split. No exchangeability guarantee -- included to show it mis-covers."""
    test_all = meta["test_data_all_normalized"]
    means = float(np.asarray(meta["means"]).squeeze())
    stds = float(np.asarray(meta["stds"]).squeeze())
    coords = meta["coords_all"]

    def haversine(a, b):
        la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
        d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 2 * np.arcsin(np.sqrt(d))

    # for each held-out station find nearest training station
    abs_res_cal, abs_res_test = [], []
    T = test_all.shape[0]
    cal_T = T // 2
    for h in holdout_idx:
        nn = min(train_station_idx, key=lambda s: haversine(coords[h], coords[s]))
        # "forecast" = nearest training station's value (denormalized); residual vs true held-out
        true = test_all[:, h, 0] * stds + means
        pred = test_all[:, nn, 0] * stds + means
        r = np.abs(true - pred)
        abs_res_cal.append(r[:cal_T]); abs_res_test.append(r[cal_T:])
    cal = np.concatenate(abs_res_cal); test = np.concatenate(abs_res_test)
    qhat = split_conformal_qhat(cal, alpha)   # residual-quantile interval (no guarantee transfer)
    cov = float((test <= qhat).mean())
    return cov, float(2 * qhat)


def run():
    t0 = time.time()
    cfg = DataConfig(window_size=WINDOW, horizon=HORIZON, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=30, patience=8, batch_size=8, loss_type="pot")

    seeds_out = []
    for seed in range(N_SEEDS):
        print(f"[seed {seed+1}/{N_SEEDS}] preparing + training (hold out {HOLD_OUT})...", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        train_ds, val_ds, test_ds, meta = prepare_real_delhi(cfg, hold_out_stations=HOLD_OUT)
        train_idx = meta["train_stations"]

        model = OmniST(ModelConfig(n_features=1, horizon=HORIZON, d_model=64, k_neighbors=8))
        train_model(model, train_ds, val_ds, tcfg, meta)

        # calibrate split-conformal on TRAINING-station residuals (the exchangeable calibration set)
        cal_res = collect_residuals(model, test_ds, meta)
        qhat = split_conformal_qhat(cal_res, ALPHA)

        # EX-CONF-1: coverage + width at UNSEEN (held-out) stations
        ind_ds = build_inductive_test(meta, HOLD_OUT, WINDOW, HORIZON)
        ind_res = collect_residuals(model, ind_ds, meta)
        cov_unseen, width_unseen = coverage_width(ind_res, qhat)

        # EX-CONF-2: NN-residual baseline at unseen stations
        cov_nn, width_nn = nn_residual_baseline(meta, train_idx, HOLD_OUT, WINDOW, HORIZON, ALPHA)

        # EX-CONF-3: coverage at unseen stations under training-node dropout (30%, 50%)
        drop_cov = {}
        rng = np.random.default_rng(seed)
        for frac in (0.3, 0.5):
            keep = sorted(rng.choice(len(train_idx), int(len(train_idx) * (1 - frac)), replace=False))
            # recalibrate on the surviving training stations, then measure unseen-station coverage
            surv_ds = build_inductive_test(meta, [train_idx[i] for i in keep], WINDOW, HORIZON)
            q_drop = split_conformal_qhat(collect_residuals(model, surv_ds, meta), ALPHA)
            c, _ = coverage_width(ind_res, q_drop)
            drop_cov[f"drop_{int(frac*100)}pct"] = round(c, 4)

        seeds_out.append({
            "seed": seed,
            "omnist_conformal_coverage_unseen": round(cov_unseen, 4),
            "omnist_conformal_width_unseen": round(width_unseen, 2),
            "nn_residual_coverage_unseen": round(cov_nn, 4),
            "nn_residual_width_unseen": round(width_nn, 2),
            "coverage_unseen_under_dropout": drop_cov,
        })
        print(f"   OmniST cov={cov_unseen:.3f} w={width_unseen:.1f} | NN cov={cov_nn:.3f} | drop={drop_cov}")

    def agg(key):
        vals = [s[key] for s in seeds_out]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    summary = {
        "experiment": "EX-CONF (inductive conformal UQ at unseen stations)",
        "target_coverage": TARGET_COVERAGE,
        "held_out_stations": HOLD_OUT,
        "n_seeds": N_SEEDS,
        "omnist_conformal": {
            "coverage_unseen_mean_std": agg("omnist_conformal_coverage_unseen"),
            "width_unseen_mean_std": agg("omnist_conformal_width_unseen"),
        },
        "nn_residual_baseline": {
            "coverage_unseen_mean_std": agg("nn_residual_coverage_unseen"),
            "width_unseen_mean_std": agg("nn_residual_width_unseen"),
        },
        "coverage_under_dropout_mean": {
            k: round(float(np.mean([s["coverage_unseen_under_dropout"][k] for s in seeds_out])), 4)
            for k in ("drop_30pct", "drop_50pct")
        },
        "per_seed": seeds_out,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp_inductive_conformal.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== EX-CONF SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

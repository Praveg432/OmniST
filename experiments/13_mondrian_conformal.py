#!/usr/bin/env python3
"""EX-MONDRIAN - group-conditional (Mondrian) conformal vs marginal split-CP, on 3 REAL datasets.

Motivation. Exp 11 quantified an honest limitation: marginal split conformal holds on average but
per-node coverage spreads widely on the irregular traffic network (0.73-0.98 at a 0.90 target). This
experiment tests whether Mondrian conformal -- a SEPARATE calibrated quantile per node-group --
lifts the WORST-group coverage toward target (the conditional-coverage improvement), and at what
width cost.

Grouping feature. Distance from each node to its nearest TRAINING node (unseen nodes far from the
training set are harder). We bin into 3 groups. Calibration groups are the training stations
themselves (grouped by their distance to the nearest OTHER training station, a within-set proxy);
test groups are the held-out unseen nodes.

Protocol per domain (reuses the validated harness):
  - train OmniST, collect per-node residuals on training stations (calibration) and unseen nodes;
  - MARGINAL: one quantile from all calibration residuals -> coverage overall + per-group;
  - MONDRIAN: per-group quantile -> coverage overall + per-group + worst-group;
  - report whether Mondrian lifts the worst-group coverage toward 0.90 and the width it pays.

RUN ONLY ON EXPLICIT COMMAND:
    python experiments/13_mondrian_conformal.py            # full (3 domains, multi-seed)
    python experiments/13_mondrian_conformal.py quick      # 1 seed, all domains
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
from src.online_conformal import MondrianConformal

ALPHA = 0.10
N_GROUPS = 3

DOMAINS = {
    "delhi_air_quality": {"prepare": prepare_real_delhi, "window": 60, "horizon": 30,
                          "hold_out": [3, 11, 19, 27, 33], "epochs": 25, "seeds": 3},
    "beijing_air_quality": {"prepare": prepare_real_beijing, "window": 30, "horizon": 15,
                            "hold_out": [2, 6, 9], "epochs": 25, "seeds": 3},
    "pemsd7_traffic": {"prepare": prepare_real_traffic, "window": 48, "horizon": 12,
                       "hold_out": [5, 40, 90, 140, 200], "epochs": 12, "seeds": 2,
                       "max_cal_nodes": 60},
}


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest_dist(node_coord, ref_coords, exclude_self_idx=None):
    best = np.inf
    for j, c in enumerate(ref_coords):
        if exclude_self_idx is not None and j == exclude_self_idx:
            continue
        d = _euclid(node_coord, c)
        if d < best:
            best = d
    return best


def per_node_residuals(model, ds, meta):
    """Per-node denormalized abs residuals -> (N_nodes, n_samples) array aligned to ds node order."""
    from torch.utils.data import DataLoader
    from src.data import collate_variable_nodes
    mu = torch.tensor(float(np.asarray(meta["means"]).squeeze()))
    sd = torch.tensor(float(np.asarray(meta["stds"]).squeeze()))
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    by_node = None
    model.eval()
    with torch.no_grad():
        for f, c, t, y in loader:
            p = model(f, c, t)
            r = (((p * sd + mu) - (y * sd + mu)).abs())            # (B, N, H, F)
            rn = r.permute(1, 0, 2, 3).reshape(r.shape[1], -1).numpy()
            by_node = rn if by_node is None else np.concatenate([by_node, rn], axis=1)
    return by_node


def build_test(meta, idx, window, horizon):
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(window, len(data) - horizon - 1)
    return SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                                 window_size=w, horizon=horizon, stride=horizon)


def run_domain_seed(spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    window, horizon, hold = spec["window"], spec["horizon"], spec["hold_out"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=spec["epochs"], patience=7, batch_size=8, loss_type="pot")
    train_ds, val_ds, test_ds, meta = spec["prepare"](cfg, hold_out_stations=hold)
    train_idx = meta["train_stations"]
    coords = meta["coords_all"]
    model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8))
    train_model(model, train_ds, val_ds, tcfg, meta)

    # calibration residuals on TRAINING stations (grouped by distance to nearest OTHER training st.)
    # Cap the number of calibration nodes for large networks (keep a geographically-spread subsample)
    # so the per-node residual pass stays tractable; coverage is unaffected (still thousands of resid).
    max_cal = spec.get("max_cal_nodes", len(train_idx))
    if len(train_idx) > max_cal:
        sub = np.linspace(0, len(train_idx) - 1, max_cal).astype(int)
        cal_node_local = [train_idx[i] for i in sub]
    else:
        cal_node_local = list(train_idx)
    cal_ds = build_test(meta, cal_node_local, window, horizon)
    cal_by_node = per_node_residuals(model, cal_ds, meta)                 # (n_cal, s)
    def nearest_excluding(node_global_idx):
        best = np.inf
        for j in train_idx:
            if j == node_global_idx:
                continue
            d = _euclid(coords[node_global_idx], coords[j])
            if d < best:
                best = d
        return best
    cal_feat = np.array([nearest_excluding(cal_node_local[i]) for i in range(len(cal_node_local))])
    # test residuals on UNSEEN nodes (grouped by distance to nearest TRAINING station)
    test_by_node = per_node_residuals(model, build_test(meta, hold, window, horizon), meta)  # (n_unseen, s)
    test_feat = np.array([nearest_dist(coords[h], [coords[j] for j in train_idx]) for h in hold])

    # group ids (shared binning learned on calibration feature so train/test use the same thresholds)
    qs = np.quantile(cal_feat, np.linspace(0, 1, N_GROUPS + 1)[1:-1]) if N_GROUPS > 1 else np.array([])
    cal_gid = np.searchsorted(qs, cal_feat, side="right")
    test_gid = np.searchsorted(qs, test_feat, side="right")
    # expand per-node group ids to per-residual (each node contributes s residuals)
    cal_gid_flat = np.repeat(cal_gid, cal_by_node.shape[1])
    cal_res_flat = cal_by_node.reshape(-1)
    test_gid_flat = np.repeat(test_gid, test_by_node.shape[1])
    test_res_flat = test_by_node.reshape(-1)

    # MARGINAL split-CP (one quantile)
    n = len(cal_res_flat); lvl = min(1.0, (1 - ALPHA) * (1 + 1 / max(n, 1)))
    q_marg = float(np.quantile(cal_res_flat, lvl))
    marg_cov = float((test_res_flat <= q_marg).mean())
    # per-group coverage under the MARGINAL quantile (to show the spread it leaves)
    marg_group = {}
    for g in range(N_GROUPS):
        m = test_gid_flat == g
        if m.any():
            marg_group[f"group_{g}"] = round(float((test_res_flat[m] <= q_marg).mean()), 4)

    # MONDRIAN (per-group quantile)
    mond = MondrianConformal(alpha=ALPHA, n_groups=N_GROUPS).fit(cal_gid_flat, cal_res_flat)
    mond_eval = mond.evaluate(test_gid_flat, test_res_flat)

    return {
        "seed": seed,
        "marginal": {"overall_coverage": round(marg_cov, 4), "width": round(2 * q_marg, 3),
                     "per_group_coverage": marg_group},
        "mondrian": mond_eval,
    }


def run(quick=False):
    t0 = time.time()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp_mondrian_conformal.json")
    summary = {"experiment": "EX-MONDRIAN (group-conditional conformal vs marginal, 3 real datasets)",
               "alpha": ALPHA, "n_groups": N_GROUPS, "target_coverage": 1 - ALPHA,
               "grouping_feature": "distance to nearest training node (unseen-node difficulty proxy)",
               "results_by_domain": {}}
    for name, spec in DOMAINS.items():
        seeds = 1 if quick else spec["seeds"]
        print(f"\n===== {name} ({seeds} seeds) =====", flush=True)
        per_seed = []
        for s in range(seeds):
            print(f"  seed {s+1}/{seeds}...", flush=True)
            r = run_domain_seed(spec, s)
            per_seed.append(r)
            print(f"    marginal overall={r['marginal']['overall_coverage']} worst_group(marg)="
                  f"{min(r['marginal']['per_group_coverage'].values()) if r['marginal']['per_group_coverage'] else 'NA'}"
                  f" | mondrian overall={r['mondrian']['overall_coverage']} "
                  f"worst_group={r['mondrian']['worst_group_coverage']}", flush=True)
        def mean_worst(key_fn):
            vals = [key_fn(p) for p in per_seed]; vals = [v for v in vals if v == v]
            return round(float(np.mean(vals)), 4) if vals else float("nan")
        summary["results_by_domain"][name] = {
            "n_seeds": seeds,
            "marginal_worst_group_coverage_mean": mean_worst(
                lambda p: min(p["marginal"]["per_group_coverage"].values()) if p["marginal"]["per_group_coverage"] else float("nan")),
            "mondrian_worst_group_coverage_mean": mean_worst(lambda p: p["mondrian"]["worst_group_coverage"]),
            "marginal_overall_coverage_mean": mean_worst(lambda p: p["marginal"]["overall_coverage"]),
            "mondrian_overall_coverage_mean": mean_worst(lambda p: p["mondrian"]["overall_coverage"]),
            "per_seed": per_seed,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsaved -> {out_path}")
    return summary


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))

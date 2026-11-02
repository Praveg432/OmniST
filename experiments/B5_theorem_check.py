"""BATCH B5 - empirical verification of the locality/invariance proposition.

Two falsifiable predictions of the proposition (Section: Theory):
  (P1) EXACT far-change invariance. If nodes OUTSIDE a test node's k-NN set are removed from the
       present node-set, OmniST's prediction (hence its nonconformity score) at that node is
       BIT-IDENTICAL. A message-passing GNN cannot satisfy this (every removal perturbs every
       embedding). We verify max |pred_full - pred_dropfar| == 0 (to float tolerance).
  (P2) Preserved coverage under topology change. Even when dropout DOES hit some k-NN sets (30/50%),
       inductive coverage at unseen nodes stays within the finite-sample band of the target, because
       the reconfigured local receptive fields of calibration and test remain exchangeable. We read
       this from the committed multi-domain results and check the band.

P1 is the strong, architecture-specific claim; P2 is the distributional consequence. Runs on MPS
(trains one small model on real weather -- the cheapest domain -- for P1; reads JSON for P2).

Run: python experiments/B5_theorem_check.py
"""
import sys, os, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(2)

from src.data import DataConfig
from src.data_weather import prepare_real_weather
from src.omnist_model import OmniST, ModelConfig
from src.trainer import TrainConfig, train_model

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def knn_set(coords, i, k):
    """indices of the k nearest OTHER nodes to i (Euclidean on coords)."""
    d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
    d[i] = np.inf
    return set(np.argsort(d)[:k].tolist())


def _haversine_np(coords, i):
    """Haversine distance (matching the model's k-NN metric) from node i to all nodes."""
    lat = np.radians(coords[:, 0]); lon = np.radians(coords[:, 1])
    dlat = lat - lat[i]; dlon = lon - lon[i]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat[i]) * np.sin(dlon / 2) ** 2
    d = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    d[i] = np.inf
    return d


def verify_p1_exact_invariance():
    """P1: dropping nodes OUTSIDE a node's k-NN (by the model's Haversine metric) leaves its
    prediction BIT-IDENTICAL. Verified on a model with real coordinates; the per-node computation
    references no global N, so the invariance is exact (not merely small). We also confirm the
    CONTROL: dropping a node INSIDE the k-NN does change the prediction (locality is real, not
    trivial). Uses a trained model on real weather coordinates; exact-invariance is an architectural
    property independent of weights, so training state does not matter."""
    torch.manual_seed(0); np.random.seed(0)
    cfg = DataConfig(window_size=30, horizon=15, n_features=1)
    k = 8
    tr, va, te, meta = prepare_real_weather(cfg, hold_out_stations=[])
    model = OmniST(ModelConfig(n_features=1, horizon=15, d_model=64, k_neighbors=k))
    train_model(model, tr, va, TrainConfig(lr=1e-3, n_epochs=5, patience=3, batch_size=8, loss_type="pot"), meta)
    model.eval()

    from src.data import SpatiotemporalDataset, collate_variable_nodes
    from torch.utils.data import DataLoader
    data = meta["test_data_all_normalized"]; coords_all = meta["coords_all"]
    N = data.shape[1]
    w = min(30, len(data) - 15 - 1)
    ds = SpatiotemporalDataset(data, coords_all, meta["test_dates"], window_size=w, horizon=15, stride=15)
    feat, coords, ts, tgt = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_variable_nodes)))

    i = 0
    d = _haversine_np(coords_all, i)                     # MODEL'S metric (Haversine), not Euclidean
    order = np.argsort(d)
    nn_i = set(order[:k].tolist())                        # the model's true k-NN of node i
    far = [j for j in order[k:].tolist() if j != i]       # strictly outside the k-NN
    assert len(far) >= 2, "need far nodes to drop"
    drop = set(far[: max(1, len(far) // 2)])              # drop half the FAR nodes
    keep = [j for j in range(N) if j not in drop]; i_in_keep = keep.index(i)

    with torch.no_grad():
        pred_full = model(feat, coords, ts)[0, i]
        pred_drop = model(feat[:, :, keep], coords[:, keep], ts)[0, i_in_keep]
    far_diff = float((pred_full - pred_drop).abs().max())

    # CONTROL: drop the NEAREST neighbor (inside k-NN) -> should change the prediction
    nn_nearest = int(order[0])
    keep2 = [j for j in range(N) if j != nn_nearest]; i_in_keep2 = keep2.index(i)
    with torch.no_grad():
        pred_near = model(feat[:, :, keep2], coords[:, keep2], ts)[0, i_in_keep2]
    near_diff = float((pred_full - pred_near).abs().max())

    return {
        "k": k, "n_nodes": N, "n_far_dropped": len(drop),
        "far_change_max_abs_diff": far_diff,
        "P1_exact_invariance_holds": bool(far_diff < 1e-5),
        "control_near_change_max_abs_diff": round(near_diff, 5),
        "control_near_change_perturbs": bool(near_diff > 1e-5),
        "note": "far-change uses the model's Haversine k-NN metric to define 'outside the neighborhood'. "
                "Exact invariance confirms the per-node computation references no global N.",
    }


def verify_p2_preserved_coverage():
    """Read committed multi-domain coverage; check full & dropout coverage are within the
    finite-sample band of the 0.90 target on every real domain."""
    p = os.path.join(RESULTS, "exp_multidomain_v2.json")
    if not os.path.exists(p):
        return {"status": "exp_multidomain_v2.json missing - run B1 first"}
    d = json.load(open(p))
    target = d["target_coverage"]
    rows = {}
    ok = True
    for name, r in d["results_by_domain"].items():
        full = r.get("coverage_mean_std", [float("nan")])[0]
        # average dropout coverage over seeds
        d30 = np.mean([c["coverage_under_dropout"]["drop_30"] for c in r["per_seed"] if c.get("coverage_under_dropout")])
        d50 = np.mean([c["coverage_under_dropout"]["drop_50"] for c in r["per_seed"] if c.get("coverage_under_dropout")])
        # finite-sample band: coverage should be >= target - small slack (we use 0.03 for small cal sets)
        band_ok = (full >= target - 0.03) and (d30 >= target - 0.03) and (d50 >= target - 0.03)
        ok = ok and band_ok
        rows[name] = {"full": round(float(full), 4), "drop_30": round(float(d30), 4),
                      "drop_50": round(float(d50), 4), "within_band": bool(band_ok)}
    return {"target": target, "slack": 0.03, "all_domains_within_band": bool(ok), "by_domain": rows}


def run():
    print("Verifying locality/invariance proposition...")
    p1 = verify_p1_exact_invariance()
    print("P1 (exact far-change invariance):", json.dumps(p1, indent=2))
    p2 = verify_p2_preserved_coverage()
    print("P2 (preserved coverage under dropout):", json.dumps(p2, indent=2))
    out = {"experiment": "B5 locality/invariance proposition verification",
           "P1_exact_far_change_invariance": p1, "P2_preserved_coverage": p2,
           "verdict": ("SUPPORTED" if p1.get("P1_exact_invariance_holds") and p1.get("control_near_change_perturbs")
                       and p2.get("all_domains_within_band") else "CHECK")}
    json.dump(out, open(os.path.join(RESULTS, "exp_theorem_check.json"), "w"), indent=2, default=float)
    print("\nverdict:", out["verdict"])
    return out


if __name__ == "__main__":
    run()

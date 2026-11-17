#!/usr/bin/env python3
"""C1 - Rigorous inductive-coverage evaluation: disjoint train / calibration / test STATIONS over
many random station splits.

Why this exists. The headline experiments (B1, B7) calibrate the conformal threshold on residuals of
the TRAINING stations and then measure coverage on a small, hard-coded set of held-out stations
(3 to 7 sensors, identical across seeds; seeds only change the initialisation). A review pointed out
two consequences: (1) training-station residuals are not exchangeable with unseen-station residuals,
so split-conformal's coverage argument does not apply; (2) the reported spread measures
initialisation noise on one fixed station set, not variation across sensor sets.

Protocol here, per random split s:
  1. Randomly partition the stations into TRAIN / CALIBRATION / TEST. The model never sees the
     calibration or test stations. Calibration and test stations are drawn from the same random
     pool, so their scores are exchangeable across stations (the assumption split conformal needs).
  2. Train one OmniST on the TRAIN stations (normalisation uses train-station statistics, as in the
     loaders).
  3. q_cal    = conformal quantile of |error| on the CALIBRATION stations (test period).
     q_legacy = conformal quantile of |error| on the TRAIN stations (the old B1/B7 protocol).
  4. Coverage at the TEST stations under both thresholds, per-station coverage, interval width.
The unit of inference is the SPLIT: mean, sd and 95% t-interval are taken over splits.

Delhi only: a sensitivity check for a quirk in SpatiotemporalDataset, which scales the year feature
against each dataset's own year range, so a single-year test set encodes as 0.0 while training spans
0..1. The same trained model is re-evaluated with the year feature set to 1.0 (latest training year).

Run:  python experiments/C1_rigorous_inductive_eval.py <domain|all> [quick]
Output: results/exp_C1_<domain>.json  (quick runs write results/C1_quick_<domain>.json instead and
never touch committed result files). Resumable: completed splits are skipped on restart.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats as ss
from torch.utils.data import DataLoader

torch.set_num_threads(2)

from src.data import DataConfig, SpatiotemporalDataset, collate_variable_nodes
from src.data_beijing import prepare_real_beijing
from src.data_pemsbay import prepare_real_pemsbay
from src.data_real import prepare_real_delhi
from src.data_traffic_real import prepare_real_traffic
from src.data_weather import prepare_real_weather
from src.omnist_model import ModelConfig, OmniST
from src.trainer import TrainConfig, train_model

TARGET = 0.90
ALPHA = 1.0 - TARGET
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

DOMAINS = {
    "delhi_air_quality": dict(prepare=prepare_real_delhi, window=60, horizon=30, epochs=25,
                              patience=7, batch=8, n_cal=8, n_test=8, splits=20, year_check=1.0),
    "beijing_air_quality": dict(prepare=prepare_real_beijing, window=30, horizon=15, epochs=25,
                                patience=7, batch=8, n_cal=3, n_test=3, splits=20),
    "ghcn_weather": dict(prepare=prepare_real_weather, window=30, horizon=15, epochs=25,
                         patience=7, batch=8, n_cal=6, n_test=6, splits=20),
    "pemsd7_traffic": dict(prepare=prepare_real_traffic, window=48, horizon=12, epochs=12,
                           patience=7, batch=8, n_cal=30, n_test=30, splits=6, knn_mode="scalable"),
    "pemsbay_traffic": dict(prepare=prepare_real_pemsbay, window=48, horizon=12, epochs=10,
                            patience=4, batch=4, n_cal=45, n_test=45, splits=6, knn_mode="scalable"),
}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(n, 1)))))


def score_array(model, meta, dataset) -> np.ndarray:
    """Absolute errors in original units, shape (windows, stations, horizon)."""
    mu = float(np.asarray(meta["means"]).squeeze())
    sd = float(np.asarray(meta["stds"]).squeeze())
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)
    chunks = []
    model.eval()
    with torch.no_grad():
        for feats, coords, stamps, targets in loader:
            pred = model(feats, coords, stamps)
            chunks.append(((pred * sd + mu) - (targets * sd + mu)).abs()[..., 0].numpy())
    return np.concatenate(chunks, axis=0)


def station_dataset(meta, idx, window, horizon, year_value=None) -> SpatiotemporalDataset:
    data = meta["test_data_all_normalized"][:, idx, :]
    w = min(window, len(data) - horizon - 1)
    ds = SpatiotemporalDataset(data, meta["coords_all"][idx], meta["test_dates"],
                               window_size=w, horizon=horizon, stride=horizon)
    if year_value is not None:
        ds.timestamps[:, 3] = year_value
    return ds


def coverage_summary(test_scores: np.ndarray, q: float) -> dict:
    per_station = (test_scores <= q).mean(axis=(0, 2))
    return {"coverage": round(float((test_scores <= q).mean()), 4),
            "width": round(2.0 * q, 3),
            "per_station_coverage": [round(float(c), 4) for c in per_station]}


def run_split(name: str, spec: dict, split: int, n_stations: int, quick: bool) -> dict:
    t0 = time.time()
    split_rng = np.random.default_rng(1000 + split)
    perm = split_rng.permutation(n_stations)
    test_idx = np.sort(perm[:spec["n_test"]])
    cal_idx = np.sort(perm[spec["n_test"]:spec["n_test"] + spec["n_cal"]])
    held = sorted(int(i) for i in np.concatenate([cal_idx, test_idx]))

    torch.manual_seed(split)
    np.random.seed(split)
    window, horizon = spec["window"], spec["horizon"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=2 if quick else spec["epochs"], patience=spec["patience"],
                       batch_size=spec["batch"], loss_type="pot")
    train_ds, val_ds, train_station_test_ds, meta = spec["prepare"](cfg, hold_out_stations=held)
    model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8,
                               knn_mode=spec.get("knn_mode", "exact")))
    train_model(model, train_ds, val_ds, tcfg, meta)

    cal_scores = score_array(model, meta, station_dataset(meta, cal_idx, window, horizon))
    test_scores = score_array(model, meta, station_dataset(meta, test_idx, window, horizon))
    legacy_scores = score_array(model, meta, train_station_test_ds)
    q_cal = conformal_quantile(cal_scores.reshape(-1), ALPHA)
    q_legacy = conformal_quantile(legacy_scores.reshape(-1), ALPHA)

    result = {
        "split": split,
        "n_train": len(meta["train_stations"]), "n_cal": len(cal_idx), "n_test": len(test_idx),
        "test_stations": [int(i) for i in test_idx], "cal_stations": [int(i) for i in cal_idx],
        "n_cal_scores": int(cal_scores.size),
        "test_mae": round(float(test_scores.mean()), 4),
        "disjoint_calibration": coverage_summary(test_scores, q_cal),
        "legacy_train_station_calibration": coverage_summary(test_scores, q_legacy),
    }

    year_value = spec.get("year_check")
    if year_value is not None:
        cal_y = score_array(model, meta, station_dataset(meta, cal_idx, window, horizon, year_value))
        test_y = score_array(model, meta, station_dataset(meta, test_idx, window, horizon, year_value))
        q_y = conformal_quantile(cal_y.reshape(-1), ALPHA)
        result["year_feature_set_to_latest_training_year"] = {
            "test_mae": round(float(test_y.mean()), 4), **coverage_summary(test_y, q_y)}
    result["seconds"] = round(time.time() - t0, 1)
    return result


def mean_ci(values) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return {"mean": round(float(v.mean()), 4), "sd": None, "ci95": [None, None], "n": n}
    m, s = float(v.mean()), float(v.std(ddof=1))
    h = float(ss.t.ppf(0.975, n - 1)) * s / np.sqrt(n)
    return {"mean": round(m, 4), "sd": round(s, 4), "ci95": [round(m - h, 4), round(m + h, 4)],
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n": n}


def aggregate(splits: list) -> dict:
    out = {}
    for key in ("disjoint_calibration", "legacy_train_station_calibration"):
        covs = [s[key]["coverage"] for s in splits]
        station_cov = np.concatenate([s[key]["per_station_coverage"] for s in splits])
        out[key] = {
            "coverage_over_splits": mean_ci(covs),
            "width_over_splits": mean_ci([s[key]["width"] for s in splits]),
            "fraction_of_splits_at_or_above_target": round(float(np.mean(np.array(covs) >= TARGET)), 3),
            "per_station_coverage": {
                "mean": round(float(station_cov.mean()), 4),
                "p10": round(float(np.percentile(station_cov, 10)), 4),
                "min": round(float(station_cov.min()), 4),
                "fraction_below_0.80": round(float((station_cov < 0.80).mean()), 3)},
        }
    out["test_mae_over_splits"] = mean_ci([s["test_mae"] for s in splits])
    if "year_feature_set_to_latest_training_year" in splits[0]:
        y = [s["year_feature_set_to_latest_training_year"] for s in splits]
        out["year_feature_set_to_latest_training_year"] = {
            "coverage_over_splits": mean_ci([s["coverage"] for s in y]),
            "test_mae_over_splits": mean_ci([s["test_mae"] for s in y])}
    return out


def run_domain(name: str, quick: bool) -> None:
    spec = DOMAINS[name]
    path = os.path.join(RESULTS, f"C1_quick_{name}.json" if quick else f"exp_C1_{name}.json")
    n_stations = spec["prepare"](DataConfig(window_size=spec["window"], horizon=spec["horizon"],
                                            n_features=1), hold_out_stations=[])[3]["coords_all"].shape[0]
    n_splits = 1 if quick else spec["splits"]
    state = {"experiment": "C1 rigorous inductive coverage: disjoint train/calibration/test stations",
             "domain": name, "n_stations": int(n_stations), "target_coverage": TARGET,
             "n_calibration_stations": spec["n_cal"], "n_test_stations": spec["n_test"],
             "splits": []}
    if os.path.exists(path):
        state = json.load(open(path))
    done = {s["split"] for s in state["splits"]}
    print(f"[{name}] N={n_stations}, target {n_splits} splits, done {sorted(done)}", flush=True)
    for split in range(n_splits):
        if split in done:
            continue
        res = run_split(name, spec, split, n_stations, quick)
        state["splits"].append(res)
        d, l = res["disjoint_calibration"], res["legacy_train_station_calibration"]
        print(f"  split {split}: disjoint cov={d['coverage']} | legacy cov={l['coverage']} "
              f"| mae={res['test_mae']} | {res['seconds']}s", flush=True)
        state["aggregate"] = aggregate(state["splits"])
        json.dump(state, open(path, "w"), indent=2, default=float)
    print(f"[{name}] saved -> {path}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "quick"]
    quick_mode = "quick" in sys.argv[1:]
    which = list(DOMAINS) if (not args or args[0] == "all") else [args[0]]
    for dom in which:
        run_domain(dom, quick_mode)

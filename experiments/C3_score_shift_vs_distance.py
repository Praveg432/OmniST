#!/usr/bin/env python3
"""C3 - Feasibility of a geometry-based coverage lower bound: how different are station score
distributions, and does the difference grow with distance?

A non-exchangeable conformal coverage bound (Barber, Candes, Ramdas and Tibshirani, 2023) loses
coverage equal to a weighted sum of total-variation (TV) distances between the score distribution at
the target site and at each calibration site. If TV is large even between neighbouring stations the
bound is vacuous and a "coverage-risk map" built on it would be uninformative. This experiment
measures that quantity directly, with a noise floor, before any method is built on it.

Per model (M random splits): hold out `n_hold` stations, train on the rest, compute the absolute-error
scores of every held-out station on the test year. For each pair of held-out stations: TV between
their score distributions (histogram estimate on shared decile bins) and their distance in network
spacings. Noise floor: TV between two random halves of one station's scores (pure estimation noise,
ignoring the time ordering) and between its first and second half in time (adds non-stationarity).

Also reports the loss a bound would charge a target station when its calibration weight is spread
uniformly over its 3 nearest other held-out stations (mean TV to them).

Run: python experiments/C3_score_shift_vs_distance.py <delhi_air_quality|ghcn_weather> [quick]
Output: results/exp_C3_<domain>.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import C1_rigorous_inductive_eval as c1  # noqa: E402
import C2_region_holdout_distance as c2  # noqa: E402
from src.data import DataConfig  # noqa: E402
from src.omnist_model import ModelConfig, OmniST  # noqa: E402
from src.trainer import TrainConfig, train_model  # noqa: E402
import torch  # noqa: E402

N_BINS = 10
HOLD = {"delhi_air_quality": 16, "ghcn_weather": 12}


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, N_BINS + 1)))
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def run(domain: str, quick: bool) -> None:
    spec = c1.DOMAINS[domain]
    plane, spacing = c2.get_plane_and_spacing(domain)
    n, n_hold = len(plane), HOLD[domain]
    n_models = 1 if quick else 5
    pairs, floors_iid, floors_time, target_losses = [], [], [], []
    for m in range(n_models):
        rng = np.random.default_rng(3000 + m)
        held = np.sort(rng.permutation(n)[:n_hold])
        torch.manual_seed(m)
        np.random.seed(m)
        window, horizon = spec["window"], spec["horizon"]
        cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
        tcfg = TrainConfig(lr=1e-3, n_epochs=2 if quick else spec["epochs"], patience=spec["patience"],
                           batch_size=spec["batch"], loss_type="pot")
        train_ds, val_ds, _, meta = spec["prepare"](cfg, hold_out_stations=[int(i) for i in held])
        model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8,
                                   knn_mode=spec.get("knn_mode", "exact")))
        train_model(model, train_ds, val_ds, tcfg, meta)
        scores = c1.score_array(model, meta, c1.station_dataset(meta, held, window, horizon))  # (W, n_hold, H)
        series = [scores[:, j, :].reshape(-1) for j in range(n_hold)]
        dist = np.linalg.norm(plane[held][:, None] - plane[held][None], axis=-1) / spacing

        for j in range(n_hold):
            half = len(series[j]) // 2
            floors_time.append(tv_distance(series[j][:half], series[j][half:]))
            perm = rng.permutation(len(series[j]))
            floors_iid.append(tv_distance(series[j][perm[:half]], series[j][perm[half:2 * half]]))
            for k in range(j + 1, n_hold):
                pairs.append((float(dist[j, k]), tv_distance(series[j], series[k])))
            others = [k for k in np.argsort(dist[j]) if k != j][:3]
            target_losses.append(float(np.mean([tv_distance(series[j], series[k]) for k in others])))
        print(f"  model {m}: {len(pairs)} pairs so far", flush=True)

    d, tv = np.array(pairs).T
    rho, p = ss.spearmanr(d, tv)
    near, far = d <= np.percentile(d, 25), d >= np.percentile(d, 75)
    out = {"experiment": "C3 station score-distribution shift (TV) versus distance", "domain": domain,
           "n_models": n_models, "n_held_out_per_model": n_hold, "n_pairs": int(len(d)),
           "distance_unit": "network spacings", "bins": N_BINS,
           "tv_pairs": {"median": round(float(np.median(tv)), 3), "p10": round(float(np.percentile(tv, 10)), 3),
                        "p90": round(float(np.percentile(tv, 90)), 3),
                        "nearest_quartile_median": round(float(np.median(tv[near])), 3),
                        "farthest_quartile_median": round(float(np.median(tv[far])), 3),
                        "spearman_rho_distance": round(float(rho), 3), "p_value": round(float(p), 4)},
           "noise_floor_random_halves_median": round(float(np.median(floors_iid)), 3),
           "noise_floor_time_halves_median": round(float(np.median(floors_time)), 3),
           "bound_loss_3_nearest_median": round(float(np.median(target_losses)), 3),
           "bound_loss_3_nearest_p90": round(float(np.percentile(target_losses, 90)), 3)}
    path = os.path.join(c1.RESULTS, f"C3_quick_{domain}.json" if quick else f"exp_C3_{domain}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "quick"]
    run(args[0] if args else "delhi_air_quality", "quick" in sys.argv[1:])

#!/usr/bin/env python3
"""C2 - Does coverage fall with distance from the training network? Contiguous-region holdout.

Question. When a new sensor is placed far from the sensors the model was trained on, is its coverage
worse than for a sensor near the network? If yes, calibrating on scattered stations that sit close to
the network overstates coverage at distant sites, and a geometry-aware calibration has a measured
motivation. If not, the random-split coverage result is stronger than feared. Either outcome is a
result.

Protocol, per split s (seed 2000 + s):
  1. Pick a random seed station; the TEST REGION is that station and its (n_test - 1) nearest
     stations, a contiguous cluster. This leaves a hole in the network.
  2. The CALIBRATION stations are drawn at random from the stations outside the region.
  3. The model trains on the remaining stations (never seeing the region or calibration stations).
  4. q = conformal quantile of |error| on the calibration stations. Per-station coverage and error
     are measured on the region stations.
  5. For every region and calibration station, record d1 = distance to the nearest TRAINING station
     and d3 = mean distance to its three nearest training stations, divided by the network spacing
     (the median nearest-neighbour distance among all stations), so datasets are comparable.

Analysis (`analyse` mode) pools the stations of all splits and reports, for the region arm and for the
scattered holdouts already saved by C1 (computed after the fact from their station indices):
Spearman correlation of per-station coverage with d1, coverage in distance tertiles (cluster
bootstrap over splits), and the distance gap between calibration and test stations.

Distances: Haversine-like (equirectangular, km) for the three lat/lon datasets; Euclidean in the
reconstructed MDS plane for the two traffic datasets (their coordinates are not measured positions).

Run:  python experiments/C2_region_holdout_distance.py <domain|all> [quick]
      python experiments/C2_region_holdout_distance.py analyse
Output: results/exp_C2_<domain>.json (quick writes results/C2_quick_<domain>.json).
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import C1_rigorous_inductive_eval as c1  # noqa: E402  (also puts the project root on sys.path)
from src.data import DataConfig  # noqa: E402
from src.omnist_model import ModelConfig, OmniST  # noqa: E402
from src.trainer import TrainConfig, train_model  # noqa: E402
import torch  # noqa: E402

RESULTS = c1.RESULTS
GEO_DOMAINS = {"delhi_air_quality", "beijing_air_quality", "ghcn_weather"}


def to_plane(coords: np.ndarray, domain: str) -> np.ndarray:
    """Project to a plane in kilometres (lat/lon datasets) or keep the MDS plane (traffic)."""
    if domain not in GEO_DOMAINS:
        return coords.astype(float)
    lat0 = np.radians(coords[:, 0].mean())
    return np.stack([np.radians(coords[:, 1]) * np.cos(lat0) * 6371.0,
                     np.radians(coords[:, 0]) * 6371.0], axis=1)


def network_spacing(plane: np.ndarray) -> float:
    d = np.linalg.norm(plane[:, None] - plane[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def distances_to_training(plane: np.ndarray, sites: np.ndarray, train: np.ndarray, spacing: float):
    d = np.linalg.norm(plane[sites][:, None] - plane[train][None], axis=-1)
    d.sort(axis=1)
    return d[:, 0] / spacing, d[:, :3].mean(axis=1) / spacing


def get_plane_and_spacing(domain: str):
    spec = c1.DOMAINS[domain]
    meta = spec["prepare"](DataConfig(window_size=spec["window"], horizon=spec["horizon"],
                                      n_features=1), hold_out_stations=[])[3]
    plane = to_plane(np.asarray(meta["coords_all"]), domain)
    return plane, network_spacing(plane)


def run_split(domain: str, spec: dict, split: int, plane: np.ndarray, spacing: float, quick: bool) -> dict:
    t0 = time.time()
    n = len(plane)
    rng = np.random.default_rng(2000 + split)
    seed_station = int(rng.integers(n))
    order = np.argsort(np.linalg.norm(plane - plane[seed_station], axis=1))
    region = np.sort(order[:spec["n_test"]])
    rest = np.setdiff1d(np.arange(n), region)
    cal = np.sort(rng.choice(rest, size=spec["n_cal"], replace=False))
    held = sorted(int(i) for i in np.concatenate([region, cal]))
    train = np.setdiff1d(np.arange(n), held)

    torch.manual_seed(split)
    np.random.seed(split)
    window, horizon = spec["window"], spec["horizon"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    tcfg = TrainConfig(lr=1e-3, n_epochs=2 if quick else spec["epochs"], patience=spec["patience"],
                       batch_size=spec["batch"], loss_type="pot")
    train_ds, val_ds, _, meta = spec["prepare"](cfg, hold_out_stations=held)
    model = OmniST(ModelConfig(n_features=1, horizon=horizon, d_model=64, k_neighbors=8,
                               knn_mode=spec.get("knn_mode", "exact")))
    train_model(model, train_ds, val_ds, tcfg, meta)

    cal_scores = c1.score_array(model, meta, c1.station_dataset(meta, cal, window, horizon))
    test_scores = c1.score_array(model, meta, c1.station_dataset(meta, region, window, horizon))
    q = c1.conformal_quantile(cal_scores.reshape(-1), c1.ALPHA)
    test_d1, test_d3 = distances_to_training(plane, region, train, spacing)
    cal_d1, _ = distances_to_training(plane, cal, train, spacing)
    return {
        "split": split, "seed_station": seed_station, "n_train": int(len(train)),
        "test_stations": [int(i) for i in region], "cal_stations": [int(i) for i in cal],
        "test_d1": [round(float(x), 4) for x in test_d1], "test_d3": [round(float(x), 4) for x in test_d3],
        "cal_d1": [round(float(x), 4) for x in cal_d1],
        "station_coverage": [round(float(x), 4) for x in (test_scores <= q).mean(axis=(0, 2))],
        "station_mae": [round(float(x), 4) for x in test_scores.mean(axis=(0, 2))],
        "coverage": round(float((test_scores <= q).mean()), 4), "width": round(2.0 * q, 3),
        "seconds": round(time.time() - t0, 1)}


def run_domain(domain: str, quick: bool) -> None:
    spec = c1.DOMAINS[domain]
    path = os.path.join(RESULTS, f"C2_quick_{domain}.json" if quick else f"exp_C2_{domain}.json")
    plane, spacing = get_plane_and_spacing(domain)
    state = {"experiment": "C2 contiguous-region holdout: coverage versus distance to training sensors",
             "domain": domain, "n_stations": int(len(plane)), "network_spacing": round(spacing, 4),
             "distance_unit": "km" if domain in GEO_DOMAINS else "MDS units",
             "n_calibration_stations": spec["n_cal"], "n_test_stations": spec["n_test"], "splits": []}
    if os.path.exists(path):
        state = json.load(open(path))
    done = {s["split"] for s in state["splits"]}
    n_splits = 1 if quick else spec["splits"]
    print(f"[{domain}] N={len(plane)}, spacing={spacing:.3f}, target {n_splits} splits, done {sorted(done)}",
          flush=True)
    for split in range(n_splits):
        if split in done:
            continue
        res = run_split(domain, spec, split, plane, spacing, quick)
        state["splits"].append(res)
        print(f"  split {split}: cov={res['coverage']} mean d1={np.mean(res['test_d1']):.2f} "
              f"(cal {np.mean(res['cal_d1']):.2f}) | {res['seconds']}s", flush=True)
        json.dump(state, open(path, "w"), indent=2, default=float)
    print(f"[{domain}] saved -> {path}", flush=True)


# ---------------------------------------------------------------- analysis (no training)
def tertile_table(d1, cov, split_ids, edges, n_boot=2000, seed=0):
    """Mean coverage in distance bins with a cluster bootstrap (resampling whole splits)."""
    rng = np.random.default_rng(seed)
    d1, cov, split_ids = map(np.asarray, (d1, cov, split_ids))
    bins = np.digitize(d1, edges)
    uniq = np.unique(split_ids)
    rows = []
    for b in range(len(edges) + 1):
        sel = bins == b
        if sel.sum() == 0:
            rows.append(None)
            continue
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            vals = np.concatenate([cov[(split_ids == s) & sel] for s in pick])
            if len(vals):
                boots.append(vals.mean())
        rows.append({"n": int(sel.sum()), "coverage": round(float(cov[sel].mean()), 3),
                     "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                              round(float(np.percentile(boots, 97.5)), 3)],
                     "mean_d1": round(float(d1[sel].mean()), 2)})
    return rows


def pooled_arm(domain, plane, spacing, arm):
    """Collect (d1, per-station coverage, split id, cal d1, station id) for one arm of a domain."""
    d1, cov, ids, cal_d1, stations = [], [], [], [], []
    if arm == "region":
        data = json.load(open(os.path.join(RESULTS, f"exp_C2_{domain}.json")))["splits"]
        for s in data:
            d1 += s["test_d1"]; cov += s["station_coverage"]; ids += [s["split"]] * len(s["test_d1"])
            cal_d1 += s["cal_d1"]; stations += s["test_stations"]
    else:
        data = json.load(open(os.path.join(RESULTS, f"exp_C1_{domain}.json")))["splits"]
        n = len(plane)
        for s in data:
            test, cal = np.array(s["test_stations"]), np.array(s["cal_stations"])
            train = np.setdiff1d(np.arange(n), np.concatenate([test, cal]))
            td, _ = distances_to_training(plane, test, train, spacing)
            cd, _ = distances_to_training(plane, cal, train, spacing)
            d1 += list(td); cov += s["disjoint_calibration"]["per_station_coverage"]
            ids += [s["split"]] * len(test); cal_d1 += list(cd); stations += [int(i) for i in test]
    return np.array(d1), np.array(cov), np.array(ids), np.array(cal_d1), np.array(stations)


def within_station_effect(arms, n_boot=2000, seed=0):
    """Effect of distance on coverage using only variation WITHIN a station.

    The same station is held out in several splits and sits at a different distance from the
    training network each time, so station identity (some stations are simply noisier) cannot
    explain a trend. Demeans distance and coverage per station and fits one slope; the 95% interval
    is a bootstrap over stations. Returns None if there is too little within-station variation.
    """
    by_station = {}
    for d1, cov, _, _, st in arms.values():
        for dist, c, s in zip(d1, cov, st):
            by_station.setdefault(int(s), []).append((dist, c))
    usable = {s: np.array(v) for s, v in by_station.items() if len(v) >= 4 and np.ptp(np.array(v)[:, 0]) > 0}
    if len(usable) < 4:
        return None

    def slope(keys):
        num = den = 0.0
        for k in keys:
            v = usable[k]
            dd, cc = v[:, 0] - v[:, 0].mean(), v[:, 1] - v[:, 1].mean()
            num += (dd * cc).sum(); den += (dd * dd).sum()
        return num / den if den > 0 else np.nan

    keys = list(usable)
    rng = np.random.default_rng(seed)
    boots = [slope(list(rng.choice(keys, size=len(keys), replace=True))) for _ in range(n_boot)]
    return {"n_stations": len(keys), "n_observations": int(sum(len(v) for v in usable.values())),
            "slope_per_spacing": round(float(slope(keys)), 4),
            "ci95": [round(float(np.nanpercentile(boots, 2.5)), 4), round(float(np.nanpercentile(boots, 97.5)), 4)]}


def analyse() -> None:
    out = {}
    for domain in c1.DOMAINS:
        if not os.path.exists(os.path.join(RESULTS, f"exp_C2_{domain}.json")):
            continue
        plane, spacing = get_plane_and_spacing(domain)
        arms = {a: pooled_arm(domain, plane, spacing, a) for a in ("scattered", "region")}
        edges = np.percentile(np.concatenate([arms[a][0] for a in arms]), [33.3, 66.7])
        out[domain] = {"edges_in_spacings": [round(float(e), 2) for e in edges]}
        print(f"\n== {domain}  (distance unit = network spacing; tertile edges {out[domain]['edges_in_spacings']})")
        out[domain]["within_station"] = within_station_effect(arms)
        ws = out[domain]["within_station"]
        print("  within-station slope of coverage per extra network spacing: " + (
            f"{ws['slope_per_spacing']:+.4f} [{ws['ci95'][0]:+.4f}, {ws['ci95'][1]:+.4f}] "
            f"({ws['n_stations']} stations, {ws['n_observations']} obs)" if ws else "too little within-station variation"))
        for arm, (d1, cov, ids, cal_d1, _stations) in arms.items():
            rho, p = ss.spearmanr(d1, cov)
            rows = tertile_table(d1, cov, ids, edges)
            out[domain][arm] = {"n_stations": int(len(d1)), "test_d1_mean": round(float(d1.mean()), 2),
                                "cal_d1_mean": round(float(cal_d1.mean()), 2),
                                "spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4),
                                "tertiles": rows}
            t = " | ".join(f"T{i + 1}: {r['coverage']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] n={r['n']}"
                           if r else f"T{i + 1}: -" for i, r in enumerate(rows))
            print(f"  {arm:9s} n={len(d1):3d} test d1 {d1.mean():.2f} vs cal d1 {cal_d1.mean():.2f} | "
                  f"rho={rho:+.3f} (p={p:.3f}) | {t}")
    json.dump(out, open(os.path.join(RESULTS, "exp_C2_analysis.json"), "w"), indent=2, default=float)
    print("\nsaved -> results/exp_C2_analysis.json")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "quick"]
    if argv and argv[0] == "analyse":
        analyse()
    else:
        quick_mode = "quick" in sys.argv[1:]
        which = list(c1.DOMAINS) if (not argv or argv[0] == "all") else [argv[0]]
        for dom in which:
            run_domain(dom, quick_mode)

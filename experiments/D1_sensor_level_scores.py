#!/usr/bin/env python3
"""D1 - per-sensor score collection under the sensor-level protocol used by Theorem 1.

What this produces. For each random split of the sensors into a TRAINING network T and a held-out
set R, a model is trained on T only. Every sensor in R is then scored by the SAME function: its own
history plus its k nearest sensors in T (reference-restricted neighbour search, Lemma 1). All of R
is scored in one joint pass, which Lemma 1 (and tests/test_reference_scoring.py) show is identical to
scoring each sensor alone. The output is, for every method and every sensor in R, the array of
absolute errors over (forecast window, horizon step). The analysis script (D2) draws random
calibration/test partitions of R from these arrays, so the theorem is tested without retraining.

Methods scored on identical windows:
  trained, inductive, use only the sensor's own history and its reference-network neighbours:
    full      coordinate embedding + neighbour attention + GRU (OmniST)
    no_gat    coordinate embedding + GRU
    ci        no coordinates, no attention: a shared channel-independent GRU (the control)
  no training, forecasting-feasible (own history only):
    persist   last observed value
    winmean   mean of the input window
  no training, NOT forecasting-feasible (use neighbours' CONCURRENT values, i.e. nowcasting):
    nn_conc   value of the nearest reference sensor in the target window
    idw_conc  inverse-distance-squared weighted mean of reference sensors in the target window
Robustness: the `full` model rescored with 30% / 50% of the reference sensors removed.

Run:  python experiments/D1_sensor_level_scores.py <domain|all> [quick]
Out:  results/D1/<domain>_trial<k>.npz (resumable; quick runs write to results/D1_quick/).
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# n_held = |R|. Sizes follow experiment C1 (calibration + test stations there).
DOMAINS = {
    "delhi_air_quality": dict(prepare=prepare_real_delhi, window=60, horizon=30, epochs=25,
                              patience=7, batch=8, n_held=16, trials=10,
                              variants=["full", "no_gat", "ci"]),
    "beijing_air_quality": dict(prepare=prepare_real_beijing, window=30, horizon=15, epochs=25,
                                patience=7, batch=8, n_held=6, trials=10,
                                variants=["full", "no_gat", "ci"]),
    "ghcn_weather": dict(prepare=prepare_real_weather, window=30, horizon=15, epochs=25,
                         patience=7, batch=8, n_held=12, trials=10,
                         variants=["full", "no_gat", "ci"]),
    "pemsd7_traffic": dict(prepare=prepare_real_traffic, window=48, horizon=12, epochs=12,
                           patience=7, batch=8, n_held=60, trials=4, knn_mode="scalable",
                           variants=["full", "ci"]),
    "pemsbay_traffic": dict(prepare=prepare_real_pemsbay, window=48, horizon=12, epochs=10,
                            patience=4, batch=4, n_held=90, trials=4, knn_mode="scalable",
                            variants=["full", "ci"]),
}


def variant_config(name: str, horizon: int, knn_mode: str, center=None, scale=None) -> ModelConfig:
    base = dict(n_features=1, horizon=horizon, d_model=64, k_neighbors=8, knn_mode=knn_mode)
    if center is not None:
        base.update(coord_center=tuple(float(c) for c in center),
                    coord_scale=tuple(float(s) for s in scale))
    if name == "full":
        return ModelConfig(**base)
    if name == "no_gat":
        return ModelConfig(**base, n_gat_layers=0)
    if name == "ci":
        return ModelConfig(**base, n_gat_layers=0, use_coords=False)
    raise ValueError(name)


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat, dlon = lat[:, None] - lat[None], lon[:, None] - lon[None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_loader(meta, window, horizon, batch=4):
    data = meta["test_data_all_normalized"]
    w = min(window, len(data) - horizon - 1)
    ds = SpatiotemporalDataset(data, meta["coords_all"], meta["test_dates"],
                               window_size=w, horizon=horizon, stride=horizon)
    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_variable_nodes)


def score_model(model, loader, held, ref_mask_np, sd):
    """|error| for sensors `held`, shape (len(held), windows, horizon), original units."""
    chunks = []
    model.eval()
    with torch.no_grad():
        for feats, coords, stamps, targets in loader:
            B, _, N, _ = feats.shape
            mask = torch.from_numpy(ref_mask_np).unsqueeze(0).expand(B, N).contiguous()
            pred = model(feats, coords, stamps, ref_mask=mask)           # (B, N, H, 1)
            err = (pred - targets).abs()[..., 0] * sd                    # (B, N, H)
            chunks.append(err[:, held].permute(1, 0, 2).numpy())         # (|R|, B, H)
    return np.concatenate(chunks, axis=1).astype(np.float32)


def score_baselines(loader, held, ref_idx, dist, sd):
    out = {k: [] for k in ("persist", "winmean", "nn_conc", "idw_conc")}
    nearest = {i: ref_idx[np.argmin(dist[i, ref_idx])] for i in held}
    w_idw = {}
    for i in held:
        w = 1.0 / (dist[i, ref_idx] ** 2 + 1e-9)
        w_idw[i] = w / w.sum()
    for feats, _, _, targets in loader:
        y = targets[..., 0].numpy()                                      # (B, N, H)
        x = feats[..., 0].numpy()                                        # (B, W, N)
        last, mean = x[:, -1, :], x.mean(axis=1)
        e = {k: [] for k in out}
        for i in held:
            e["persist"].append(np.abs(y[:, i] - last[:, i, None]))
            e["winmean"].append(np.abs(y[:, i] - mean[:, i, None]))
            e["nn_conc"].append(np.abs(y[:, i] - y[:, nearest[i]]))
            pred = np.tensordot(w_idw[i], y[:, ref_idx], axes=([0], [1]))   # (B, H)
            e["idw_conc"].append(np.abs(y[:, i] - pred))
        for k in out:
            out[k].append(np.stack(e[k], axis=0) * sd)                   # (|R|, B, H)
    return {k: np.concatenate(v, axis=1).astype(np.float32) for k, v in out.items()}


def run_trial(name, spec, trial, n_stations, outdir, quick):
    t0 = time.time()
    rng = np.random.default_rng(2000 + trial)
    perm = rng.permutation(n_stations)
    held = np.sort(perm[:spec["n_held"]])
    window, horizon = spec["window"], spec["horizon"]
    cfg = DataConfig(window_size=window, horizon=horizon, n_features=1)
    train_ds, val_ds, _, meta = spec["prepare"](cfg, hold_out_stations=[int(i) for i in held])
    train_idx = np.array(meta["train_stations"])
    sd = float(np.asarray(meta["stds"]).squeeze())
    loader = make_loader(meta, window, horizon)
    ref_mask = np.zeros(n_stations, dtype=bool)
    ref_mask[train_idx] = True
    dist = haversine_matrix(meta["coords_all"])

    arrays, timing = {}, {}
    center = scale = None
    variants = spec["variants"]
    if NORMALISED:
        # coordinate normalisation fixed from the TRAINING network only (never from R)
        tc = meta["coords_all"][train_idx]
        center, scale = tc.mean(axis=0), tc.std(axis=0) + 1e-6
        variants = [v for v in variants if v != "ci"]
    for variant in variants:
        torch.manual_seed(trial)
        np.random.seed(trial)
        model = OmniST(variant_config(variant, horizon, spec.get("knn_mode", "exact"), center, scale))
        tcfg = TrainConfig(lr=1e-3, n_epochs=2 if quick else spec["epochs"], patience=spec["patience"],
                           batch_size=spec["batch"], loss_type="pot")
        ts = time.time()
        train_model(model, train_ds, val_ds, tcfg, meta)
        timing[variant] = round(time.time() - ts, 1)
        arrays[variant] = score_model(model, loader, held, ref_mask, sd)
        if variant == "full":
            drop_rng = np.random.default_rng(5000 + trial)
            for frac in (0.3, 0.5):
                keep = drop_rng.choice(train_idx, size=int(round((1 - frac) * len(train_idx))),
                                       replace=False)
                m = np.zeros(n_stations, dtype=bool)
                m[keep] = True
                arrays[f"full_drop{int(frac * 100)}"] = score_model(model, loader, held, m, sd)
    if not NORMALISED:
        arrays.update(score_baselines(loader, held, train_idx, dist, sd))

    path = os.path.join(outdir, f"{name}_trial{trial}.npz")
    np.savez_compressed(path, held=held, train=train_idx, sd=sd, seconds=time.time() - t0,
                        timing=json.dumps(timing), **{f"S_{k}": v for k, v in arrays.items()})
    summary = " ".join(f"{k}={v.mean():.3g}" for k, v in arrays.items() if k in
                       ("full", "ci", "persist", "nn_conc"))
    print(f"  trial {trial}: |T|={len(train_idx)} |R|={len(held)} windows={arrays['full'].shape[1]} "
          f"MAE {summary} | {time.time() - t0:.0f}s", flush=True)


def run_domain(name, quick):
    spec = DOMAINS[name]
    outdir = os.path.join(ROOT, ("D1b" if NORMALISED else "D1") + ("_quick" if quick else ""))
    os.makedirs(outdir, exist_ok=True)
    n_stations = spec["prepare"](DataConfig(window_size=spec["window"], horizon=spec["horizon"],
                                            n_features=1), hold_out_stations=[])[3]["coords_all"].shape[0]
    n_trials = 1 if quick else spec["trials"]
    print(f"[{name}] N={n_stations} |R|={spec['n_held']} trials={n_trials} variants={spec['variants']}",
          flush=True)
    for trial in range(n_trials):
        if os.path.exists(os.path.join(outdir, f"{name}_trial{trial}.npz")):
            continue
        run_trial(name, spec, trial, n_stations, outdir, quick)
    print(f"[{name}] done", flush=True)


NORMALISED = "norm" in sys.argv[1:]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("quick", "norm")]
    quick_mode = "quick" in sys.argv[1:]
    which = list(DOMAINS) if (not args or args[0] == "all") else [args[0]]
    for dom in which:
        run_domain(dom, quick_mode)

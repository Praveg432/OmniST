#!/usr/bin/env python3
"""D6 - sensors that were actually installed late (GHCN-Daily TMAX, contiguous United States).

Natural experiment. 458 established stations report throughout 2014-2024. 70 "late" stations first reported
TMAX between 2019 and 2022 (selection rules fixed before download: experiments/fetch_ghcn_late.py). Training uses
2014-2018 data of established stations only, a period before any late station existed, so no late station and no
data from the evaluation period is seen in training. Evaluation is 2023-01-01 to 2024-12-31.

In each of 4 splits (rng seed 3000 + split): 60 random established stations are held out of training together with
all 70 late stations. The model is trained on the remaining 398 established stations, then every held-out station is
scored by the same function (own history + 8 nearest TRAINING stations; Lemma 1).
  control   each of the 60 held-out established stations as the new sensor, calibration = the other 59
  late      each of the 70 late stations as the new sensor, calibration = the 60 held-out established stations
The late stations are therefore genuinely new (never in training, installed after training ended) and calibrated
from stations that are not themselves in training. Schemes and settings are those of D5: alpha = 0.1, L = 4 windows,
t0 = L, gamma in {0.1, 0.5}; the block length is k = 6 windows (90 days).

Run: python3 experiments/D6_late_installed.py      Out: results/D6/trial*.npz, results/exp_D6_late.json
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import torch

torch.set_num_threads(2)

from D1_sensor_level_scores import haversine_matrix, make_loader, score_baselines, score_model, variant_config  # noqa: E402
from D5_per_sensor_guarantee import ALPHA, GAMMAS, run_sensor  # noqa: E402
from D5b_blocks_and_groups import aci_errors, blocks  # noqa: E402
from src.data import DataConfig  # noqa: E402
from src.data_ghcn_late import prepare_ghcn_late  # noqa: E402
from src.omnist_model import OmniST  # noqa: E402
from src.trainer import TrainConfig, train_model  # noqa: E402

RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "D6")
N_SPLITS, N_HELD_EST = 4, 60
WINDOW, HORIZON, EPOCHS, PATIENCE, BATCH = 30, 15, 25, 7, 4
LW, BLOCK = 4, 6
SCHEMES = ["same-window", "rolling-pooled", "rolling-sensor", "aci-0.1", "aci-0.5"]


def train_and_score(trial):
    path = os.path.join(OUT, f"trial{trial}.npz")
    if os.path.exists(path):
        return
    t0 = time.time()
    cfg = DataConfig(window_size=WINDOW, horizon=HORIZON, n_features=1)
    meta0 = prepare_ghcn_late(cfg, hold_out_stations=[])[3]
    is_late = meta0["is_late"]
    est = np.flatnonzero(~is_late)
    rng = np.random.default_rng(3000 + trial)
    held_est = np.sort(rng.choice(est, size=N_HELD_EST, replace=False))
    train_ds, val_ds, _, meta = prepare_ghcn_late(cfg, hold_out_stations=[int(i) for i in held_est])
    train_idx = np.array(meta["train_stations"])
    held = np.array(meta["hold_out_stations"])                 # held-out established + all late
    sd = float(np.asarray(meta["stds"]).squeeze())
    tc = meta["coords_all"][train_idx]                         # normalisation from training stations only
    center, scale = tc.mean(axis=0), tc.std(axis=0) + 1e-6
    torch.manual_seed(trial)
    np.random.seed(trial)
    model = OmniST(variant_config("full", HORIZON, "scalable", center, scale))
    train_model(model, train_ds, val_ds, TrainConfig(lr=1e-3, n_epochs=EPOCHS, patience=PATIENCE,
                                                     batch_size=BATCH, loss_type="pot"), meta)
    loader = make_loader(meta, WINDOW, HORIZON)
    ref = np.zeros(len(is_late), dtype=bool)
    ref[train_idx] = True
    dist = haversine_matrix(meta["coords_all"])
    S = score_model(model, loader, held, ref, sd)
    persist = score_baselines(loader, held, train_idx, dist, sd)["persist"]
    d_nn = dist[np.ix_(held, train_idx)].min(axis=1)
    Dtt = dist[np.ix_(train_idx, train_idx)].copy()
    np.fill_diagonal(Dtt, np.inf)
    spacing = float(np.median(Dtt.min(axis=1)))
    np.savez_compressed(path, S=S, persist=persist, held=held, train=train_idx, is_late=is_late[held],
                        distance=d_nn / spacing, spacing_km=spacing, sd=sd, seconds=time.time() - t0)
    print(f"  trial {trial}: |train|={len(train_idx)} |held|={len(held)} windows={S.shape[1]} "
          f"MAE {S.mean():.3f} persist {persist.mean():.3f} | {time.time() - t0:.0f}s", flush=True)


def summarise(cases, key):
    covs = np.array([c[key]["cov"] for c in cases])
    out = {"mean_cov": round(float(covs.mean()), 4), "min_cov": round(float(covs.min()), 4),
           "p05_cov": round(float(np.percentile(covs, 5)), 4),
           "share_below_0.80": round(float((covs < 0.80).mean()), 4),
           "inf_share": round(float(np.mean([c[key]["inf_share"] for c in cases])), 4),
           "width_med": round(float(np.median([c[key]["width_med"] for c in cases])), 4)}
    if key.startswith("aci"):
        out["bound"] = round(cases[0][key]["bound"], 4)
        out["guaranteed_cov"] = round(cases[0][key]["guaranteed_cov"], 4)
        out["bound_holds_all"] = all(c[key]["bound_holds"] for c in cases)
    return out


def analyse():
    control, late = [], []
    mae = {"control": [], "late": []}
    pers = {"control": [], "late": []}
    dist = {"control": [], "late": []}
    worst_block = {g: {"control": [], "late": []} for g in GAMMAS}
    block_ok = {g: True for g in GAMMAS}
    T = None
    for trial in range(N_SPLITS):
        z = np.load(os.path.join(OUT, f"trial{trial}.npz"))
        S, P, lt, d = z["S"].astype(np.float64), z["persist"], z["is_late"], z["distance"]
        e_idx, l_idx = np.flatnonzero(~lt), np.flatnonzero(lt)
        S_est = S[e_idx]
        for j in range(len(e_idx)):                              # control: leave one out among established
            r, T = run_sensor(S_est, j, LW)
            control.append(r)
            mae["control"].append(float(S_est[j].mean()))
            pers["control"].append(float(P[e_idx[j]].mean()))
            dist["control"].append(float(d[e_idx[j]]))
            for g in GAMMAS:
                b = blocks(aci_errors(S_est, j, LW, g), BLOCK)
                worst_block[g]["control"].append(1 - b.max())
                block_ok[g] &= bool(np.all(np.abs(b - ALPHA) <= (1 + 2 * g) / (g * BLOCK) + 1e-12))
        for j in l_idx:                                          # late: calibrate on all established held-out
            stack = np.concatenate([S_est, S[j][None]], axis=0)
            r, T = run_sensor(stack, len(e_idx), LW)
            late.append(r)
            mae["late"].append(float(S[j].mean()))
            pers["late"].append(float(P[j].mean()))
            dist["late"].append(float(d[j]))
            for g in GAMMAS:
                b = blocks(aci_errors(stack, len(e_idx), LW, g), BLOCK)
                worst_block[g]["late"].append(1 - b.max())
                block_ok[g] &= bool(np.all(np.abs(b - ALPHA) <= (1 + 2 * g) / (g * BLOCK) + 1e-12))
    out = {"n_splits": N_SPLITS, "eval_windows": T, "L": LW, "block_k": BLOCK, "n_control_cases": len(control),
           "n_late_cases": len(late)}
    for name, cases in (("control", control), ("late", late)):
        out[name] = {s: summarise(cases, s) for s in SCHEMES}
        out[name]["mae"] = round(float(np.mean(mae[name])), 4)
        out[name]["persistence_mae"] = round(float(np.mean(pers[name])), 4)
        out[name]["mean_distance_spacings"] = round(float(np.mean(dist[name])), 3)
        out[name]["worst_block"] = {f"aci-{g}": {"min": round(float(np.min(worst_block[g][name])), 4),
                                                 "mean": round(float(np.mean(worst_block[g][name])), 4)}
                                    for g in GAMMAS}
    out["block_certificate_holds"] = {f"aci-{g}": block_ok[g] for g in GAMMAS}
    # late stations are not all alike: sensors first reporting earlier vs later are not separated here, but the
    # per-split late-minus-control gap in MAE is reported with its spread over the 4 splits.
    gaps = []
    for trial in range(N_SPLITS):
        z = np.load(os.path.join(OUT, f"trial{trial}.npz"))
        S, lt = z["S"], z["is_late"]
        gaps.append(float(S[lt].mean() - S[~lt].mean()))
    out["mae_gap_late_minus_control_per_split"] = [round(g, 4) for g in gaps]
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for tr in range(N_SPLITS):
        train_and_score(tr)
    res = analyse()
    json.dump(res, open(os.path.join(RES, "exp_D6_late.json"), "w"), indent=1)
    print(f"\nevaluation windows T={res['eval_windows']}, control cases {res['n_control_cases']}, "
          f"late cases {res['n_late_cases']}")
    for name in ("control", "late"):
        r = res[name]
        print(f"{name}: MAE {r['mae']} (persistence {r['persistence_mae']}), distance {r['mean_distance_spacings']} spacings")
        for s in SCHEMES:
            a = r[s]
            extra = f" bound {a['bound']} guar {a['guaranteed_cov']} holds={a['bound_holds_all']}" if s.startswith("aci") else ""
            print(f"   {s:15s} mean {a['mean_cov']:.3f} min {a['min_cov']:.3f} <0.80 {a['share_below_0.80']:.3f} "
                  f"inf {a['inf_share']:.3f} width {a['width_med']:.3g}{extra}")
        print("   worst block", r["worst_block"])
    print("block certificate", res["block_certificate_holds"], "MAE gaps", res["mae_gap_late_minus_control_per_split"])

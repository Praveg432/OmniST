#!/usr/bin/env python3
"""Fetch the GHCN-Daily late-installation network (NOAA, public, no authentication).

Natural experiment: temperature stations that genuinely started reporting after the forecaster was trained.
All selection rules are fixed here, before any station's data is downloaded or looked at.

  Region       contiguous US: latitude 24 to 50, longitude -125 to -66, GHCN IDs starting with "US"
  Late         TMAX first year 2019 to 2022 (inventory) and last year 2025 or later
  Established  TMAX first year 2012 or earlier and last year 2025 or later; for each late station, its 8 nearest
               established stations (Haversine), de-duplicated
  Values       TMAX in tenths of degrees C -> degrees C; values with a non-empty quality flag are discarded
  Span         2014-01-01 to 2024-12-31
  Kept if      established: >= 90% of days present in 2014-2018 AND in 2023-2024
               late: >= 90% of days present in 2023-2024, no TMAX value before 2019-01-01, and a first value
                     on or before 2022-12-01 (so a 30-day input window exists before 2023)

Out: data/ghcn_late/ghcn_late.npz  (data (T, N) float32 with NaN, coords (N, 2), ids, is_late, first_date, dates)
     data/ghcn_late/selection.json (counts at each step, kept and dropped IDs with reasons)
Raw per-station files are cached in data/ghcn_late/raw/ (not committed).
Run: python3 experiments/fetch_ghcn_late.py
"""
import gzip
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ghcn_late")
RAW = os.path.join(ROOT, "raw")
BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
SPAN = pd.date_range("2014-01-01", "2024-12-31", freq="D")
K_NEAREST = 8


def inventory():
    path = os.path.join(ROOT, "ghcnd-inventory.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{BASE}/ghcnd-inventory.txt", path)
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 6 and p[3] == "TMAX" and p[0].startswith("US"):
            la, lo = float(p[1]), float(p[2])
            if 24 <= la <= 50 and -125 <= lo <= -66:
                rows.append((p[0], la, lo, int(p[4]), int(p[5])))
    return pd.DataFrame(rows, columns=["id", "lat", "lon", "first", "last"])


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fetch(sid):
    dest = os.path.join(RAW, f"{sid}.csv.gz")
    if not os.path.exists(dest):
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(f"{BASE}/by_station/{sid}.csv.gz", dest + ".part")
                os.replace(dest + ".part", dest)
                break
            except Exception as e:  # network errors: retry, then report
                err = e
        else:
            return sid, None, f"download failed: {err}"
    with gzip.open(dest, "rt") as fh:
        df = pd.read_csv(fh, header=None, usecols=[1, 2, 3, 5], names=["date", "el", "val", "qflag"],
                         dtype={"date": str, "el": str, "val": float, "qflag": str})
    df = df[(df.el == "TMAX") & df.qflag.isna()]
    first_any = pd.to_datetime(df.date.min(), format="%Y%m%d") if len(df) else None
    df = df[(df.date >= "20140101") & (df.date <= "20241231")]
    s = pd.Series(df.val.values / 10.0, index=pd.to_datetime(df.date, format="%Y%m%d"))
    s = s[~s.index.duplicated()].reindex(SPAN)
    return sid, (s.values.astype(np.float32), first_any), None


def coverage(x, start, end):
    m = (SPAN >= start) & (SPAN <= end)
    return float(np.mean(~np.isnan(x[m])))


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    inv = inventory()
    late = inv[(inv["first"].between(2019, 2022)) & (inv["last"] >= 2025)].reset_index(drop=True)
    est_all = inv[(inv["first"] <= 2012) & (inv["last"] >= 2025)].reset_index(drop=True)
    near = set()
    for _, r in late.iterrows():
        d = haversine(r.lat, r.lon, est_all.lat.values, est_all.lon.values)
        near.update(est_all.id.values[np.argsort(d)[:K_NEAREST]])
    est = est_all[est_all.id.isin(near)].reset_index(drop=True)
    print(f"candidates: late {len(late)}, established {len(est)}", flush=True)

    ids = list(late.id) + list(est.id)
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = {sid: (val, err) for sid, val, err in pool.map(fetch, ids)}

    keep, dropped = [], {}
    meta = pd.concat([late.assign(is_late=True), est.assign(is_late=False)]).set_index("id")
    for sid in ids:
        val, err = got[sid]
        if err:
            dropped[sid] = err
            continue
        x, first_any = val
        is_late = bool(meta.loc[sid, "is_late"])
        if is_late:
            ok_eval = coverage(x, "2023-01-01", "2024-12-31") >= 0.90
            ok_start = first_any is not None and first_any >= pd.Timestamp("2019-01-01")
            fv = np.flatnonzero(~np.isnan(x))
            ok_input = fv.size > 0 and SPAN[fv[0]] <= pd.Timestamp("2022-12-01")
            reasons = [n for n, ok in (("eval<90%", ok_eval), ("data before 2019", ok_start),
                                       ("starts after 2022-12-01", ok_input)) if not ok]
        else:
            reasons = [n for n, ok in (("train<90%", coverage(x, "2014-01-01", "2018-12-31") >= 0.90),
                                       ("eval<90%", coverage(x, "2023-01-01", "2024-12-31") >= 0.90)) if not ok]
        if reasons:
            dropped[sid] = ", ".join(reasons)
        else:
            keep.append((sid, x, is_late, str(first_any.date()) if first_any is not None else ""))

    data = np.stack([k[1] for k in keep], axis=1)
    coords = np.array([[meta.loc[k[0], "lat"], meta.loc[k[0], "lon"]] for k in keep])
    np.savez_compressed(os.path.join(ROOT, "ghcn_late.npz"), data=data, coords=coords,
                        ids=np.array([k[0] for k in keep]), is_late=np.array([k[2] for k in keep]),
                        first_date=np.array([k[3] for k in keep]), dates=SPAN.values.astype("datetime64[D]"))
    summary = {"candidates_late": len(late), "candidates_established": len(est),
               "kept_late": int(sum(k[2] for k in keep)), "kept_established": int(sum(not k[2] for k in keep)),
               "dropped": dropped}
    json.dump(summary, open(os.path.join(ROOT, "selection.json"), "w"), indent=1)
    print({k: v for k, v in summary.items() if k != "dropped"}, "dropped", len(dropped))
    from collections import Counter
    print(Counter(v for v in dropped.values()))

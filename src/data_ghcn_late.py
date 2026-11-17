"""GHCN-Daily late-installation network (NOAA, public): established stations plus stations that first reported
TMAX in 2019 to 2022. Built by experiments/fetch_ghcn_late.py (selection rules documented there).

Periods: training 2014-01-01 to 2018-06-30, validation 2018-07-01 to 2018-12-31 (before any late station existed),
evaluation 2023-01-01 to 2024-12-31. Missing values are filled within each period separately (forward fill, then
backward fill, then the station median of that period), so a late station's absent pre-installation history is
never back-filled into its evaluation inputs. Values outside [-40, 60] degC are treated as missing. Standardized with
training-station statistics. Contract matches the other loaders (see src/data_weather.py).
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data import DataConfig, SpatiotemporalDataset

NPZ = Path(__file__).parent.parent / "data" / "ghcn_late" / "ghcn_late.npz"
TRAIN = ("2014-01-01", "2018-06-30")
VAL = ("2018-07-01", "2018-12-31")
EVAL = ("2023-01-01", "2024-12-31")


def load_ghcn_late():
    z = np.load(NPZ, allow_pickle=True)
    return (z["data"].astype(np.float64), z["coords"].astype(np.float64), pd.DatetimeIndex(z["dates"]),
            [str(i) for i in z["ids"]], z["is_late"].astype(bool))


def _segment(data, dates, span):
    m = (dates >= span[0]) & (dates <= span[1])
    seg = data[m].copy()
    for s in range(seg.shape[1]):
        ser = pd.Series(seg[:, s])
        seg[:, s] = ser.ffill().bfill().fillna(ser.median()).values
    return seg[:, :, None], dates[m].values


def prepare_ghcn_late(config: DataConfig, hold_out_stations: Optional[List[int]] = None):
    data, coords, dates, ids, is_late = load_ghcn_late()
    data[(data < -40) | (data > 60)] = np.nan
    train, train_dates = _segment(data, dates, TRAIN)
    val, val_dates = _segment(data, dates, VAL)
    test, test_dates = _segment(data, dates, EVAL)

    hold = set(hold_out_stations or [])
    hold |= set(np.flatnonzero(is_late).tolist())          # late stations are never trained on
    train_stations = [i for i in range(len(ids)) if i not in hold]
    assert not np.isnan(train[:, train_stations]).any(), "training stations must be complete after filling"

    mu = train[:, train_stations].mean(keepdims=True).reshape(1, 1, 1)
    sd = train[:, train_stations].std(keepdims=True).reshape(1, 1, 1) + 1e-8
    norm = lambda x: (x - mu) / sd
    w = min(config.window_size, len(train) - config.horizon - 1)
    train_ds = SpatiotemporalDataset(norm(train[:, train_stations]), coords[train_stations], train_dates,
                                     window_size=w, horizon=config.horizon, stride=7)
    val_ds = SpatiotemporalDataset(norm(val[:, train_stations]), coords[train_stations], val_dates,
                                   window_size=min(w, len(val) - config.horizon - 1), horizon=config.horizon,
                                   stride=config.horizon)
    test_ds = SpatiotemporalDataset(norm(test[:, train_stations]), coords[train_stations], test_dates,
                                    window_size=min(w, len(test) - config.horizon - 1), horizon=config.horizon,
                                    stride=config.horizon)
    meta = {"means": mu, "stds": sd, "stations": ids, "train_stations": train_stations,
            "hold_out_stations": sorted(hold), "coords_all": coords, "coords_train": coords[train_stations],
            "n_stations": len(train_stations), "n_features": 1, "feature_cols": ["tmax_degC"],
            "domain": "ghcn_late", "test_data_all_normalized": norm(test), "test_dates": test_dates,
            "is_late": is_late}
    return train_ds, val_ds, test_ds, meta

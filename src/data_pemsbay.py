"""
Real PeMS-BAY Traffic Speed Data Loader (325 sensors) -- the large-N real benchmark.

Source: Caltrans PeMS Bay Area, the standard "PeMS-BAY" benchmark (Li et al., DCRNN, ICLR 2018):
  - 325 loop-detector stations, 5-minute average speed (mph), ~6 months.
  - A distance-kernel adjacency (325 x 325). As with PeMSD7-M, we reconstruct 2D sensor coordinates
    from the released adjacency via classical MDS (the speed series itself is the original real data).

This pushes OmniST's largest REAL inductive-coverage evaluation from 228 (PeMSD7-M) to 325 sensors.
Contract matches src/data_real.py / src/data_traffic_real.py.
"""
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, List, Optional
from src.data import SpatiotemporalDataset, DataConfig
from src.data_traffic_real import _coords_from_adjacency

DATA_DIR = Path(__file__).parent.parent / "data" / "real_pems"


def load_real_pemsbay(data_dir: Optional[Path] = None, subsample: int = 6
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load PeMS-BAY speeds + MDS-reconstructed coordinates.

    subsample: take every `subsample`-th 5-min step (default 6 -> 30-min cadence) to keep the
    52k-step series tractable while preserving daily/weekly structure. Returns (data (T,N,1), coords
    (N,2), dates, sensor_ids)."""
    if data_dir is None:
        data_dir = DATA_DIR
    vel = pd.read_csv(data_dir / "pemsbay_vel.csv", header=None).values.astype(np.float64)  # (T, N)
    vel = vel[::max(1, subsample)]
    T, N = vel.shape
    data = vel[:, :, None]
    adj = sp.load_npz(data_dir / "pemsbay_adj.npz").toarray()
    coords = _coords_from_adjacency(adj)
    dates = pd.date_range("2017-01-01", periods=T, freq=f"{5*subsample}min").values
    ids = [f"bay_{i}" for i in range(N)]
    return data, coords, dates, ids


def prepare_real_pemsbay(
    config: DataConfig,
    hold_out_stations: Optional[List[int]] = None,
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset, SpatiotemporalDataset, dict]:
    """Prepare real PeMS-BAY (325 sensors) with a temporal split, mirroring prepare_real_traffic."""
    data, coords, dates, ids = load_real_pemsbay()
    n_steps, n_stations, n_features = data.shape

    data[data <= 0] = np.nan
    for s in range(n_stations):
        ser = pd.Series(data[:, s, 0])
        data[:, s, 0] = ser.ffill().bfill().fillna(ser.median()).values

    test_len = int(n_steps * 0.20); val_len = int(n_steps * 0.10)
    test_data_full = data[-test_len:]; test_dates = dates[-test_len:]
    train_val = data[:-test_len]; train_val_dates = dates[:-test_len]
    val_data = train_val[-val_len:]; val_dates_arr = train_val_dates[-val_len:]
    train_data = train_val[:-val_len]; train_dates_arr = train_val_dates[:-val_len]

    if hold_out_stations is None:
        hold_out_stations = []
    train_stations = [i for i in range(n_stations) if i not in hold_out_stations]
    train_coords = coords[train_stations]

    train_means = np.nanmean(train_data[:, train_stations, :], axis=(0, 1), keepdims=True)
    train_stds = np.nanstd(train_data[:, train_stations, :], axis=(0, 1), keepdims=True) + 1e-8
    train_norm = (train_data[:, train_stations, :] - train_means) / train_stds
    val_norm = (val_data[:, train_stations, :] - train_means) / train_stds
    test_norm_train = (test_data_full[:, train_stations, :] - train_means) / train_stds
    test_norm_all = (test_data_full - train_means) / train_stds

    window = min(config.window_size, len(train_norm) - config.horizon - 1)
    stride = max(1, window)   # non-overlapping windows (large-N domain, keep tractable)
    train_ds = SpatiotemporalDataset(train_norm, train_coords, train_dates_arr,
                                     window_size=window, horizon=config.horizon, stride=stride)
    val_ds = SpatiotemporalDataset(val_norm, train_coords, val_dates_arr,
                                   window_size=min(window, len(val_norm) - config.horizon - 1),
                                   horizon=config.horizon, stride=config.horizon)
    test_ds = SpatiotemporalDataset(test_norm_train, train_coords, test_dates,
                                    window_size=min(window, len(test_norm_train) - config.horizon - 1),
                                    horizon=config.horizon, stride=config.horizon)
    metadata = {
        "means": train_means, "stds": train_stds, "stations": ids,
        "train_stations": train_stations, "hold_out_stations": hold_out_stations,
        "coords_all": coords, "coords_train": train_coords, "n_stations": len(train_stations),
        "n_features": n_features, "feature_cols": ["speed_mph"], "domain": "pemsbay_traffic",
        "test_data_all_normalized": test_norm_all, "test_dates": test_dates,
    }
    return train_ds, val_ds, test_ds, metadata

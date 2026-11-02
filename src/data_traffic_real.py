"""
Real PeMSD7-M Traffic Speed Data Loader.

Source: Caltrans Performance Measurement System (PeMS), District 7 (Los Angeles).
  The widely-used "PeMSD7(M)" benchmark (Yu, Yin & Zhu, STGCN, IJCAI 2018):
  - 228 loop-detector stations on LA-area freeways.
  - 5-minute average speed (mph), weekdays May-June 2012 -> 12,672 time steps.
  - A distance-derived Gaussian-kernel adjacency matrix (228 x 228).

This is REAL freeway sensor data (not synthetic) and is a genuinely different physical domain from
air quality (transportation vs environmental).

Coordinate handling. The public PeMSD7(M) release ships a distance-based adjacency W (W_ij =
exp(-d_ij^2 / sigma^2), thresholded) rather than raw (lat, lon). OmniST's graph is built from
coordinates, so we reconstruct a faithful 2D sensor embedding:
  1. recover pairwise distances d_ij = sqrt(-ln W_ij) for retained edges,
  2. complete the distance matrix via graph shortest paths (thresholded-out pairs),
  3. classical MDS -> 2D coordinates that preserve the sensor-network geometry.
These are embedding coordinates (not GPS), used only to build the same k-NN spatial graph OmniST
uses for every domain. The speed time series itself is fully real.

Contract matches src/data_real.py:
  load_real_traffic() -> (data (T,N,F), coords (N,2), dates (T,), station_names)
  prepare_real_traffic(cfg, hold_out_stations, ...) -> (train_ds, val_ds, test_ds, metadata)
"""
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from pathlib import Path
from typing import Tuple, List, Optional
from src.data import SpatiotemporalDataset, DataConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "real_pems"


def _coords_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """Reconstruct 2D sensor coordinates from the Gaussian-kernel adjacency via distance recovery +
    shortest-path completion + classical MDS. Returns (N, 2)."""
    n = adj.shape[0]
    W = adj.copy()
    np.fill_diagonal(W, 1.0)
    W = np.clip(W, 1e-12, 1.0)
    # recover distances where an edge exists (W>0); zeros = thresholded-out (unknown, far)
    with np.errstate(invalid="ignore"):
        D_local = np.sqrt(-np.log(W))          # d_ij for retained edges (0 on diagonal)
    edge = adj > 0
    sparse_d = np.where(edge, D_local, 0.0)
    np.fill_diagonal(sparse_d, 0.0)
    # complete via graph shortest paths (fills far pairs by summing recovered local hops)
    Dfull = shortest_path(sp.csr_matrix(sparse_d), method="D", directed=False)
    # any still-infinite (disconnected) -> set to max finite distance * 1.5
    finite_max = np.nanmax(Dfull[np.isfinite(Dfull)])
    Dfull[~np.isfinite(Dfull)] = finite_max * 1.5
    # classical MDS to 2D
    D2 = Dfull ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:2]
    L = np.clip(vals[idx], 0, None)
    coords = vecs[:, idx] * np.sqrt(L)
    # scale to a degree-like range so k-NN graph construction behaves like the geo datasets
    span = coords.max(0) - coords.min(0)
    span[span == 0] = 1.0
    coords = (coords - coords.min(0)) / span            # [0,1]
    coords = coords * np.array([0.5, 0.5]) + np.array([34.0, -118.3])  # LA-ish lat/lon box
    return coords.astype(np.float64)


def load_real_traffic(
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load PeMSD7-M real traffic speeds + reconstructed sensor coordinates.

    Returns:
        data: (T, N, 1) float64 speeds (mph)
        coords: (N, 2) embedding coordinates
        dates: (T,) datetime64 at 5-min cadence
        station_names: list[str] sensor ids
    """
    if data_dir is None:
        data_dir = DATA_DIR
    vel = pd.read_csv(data_dir / "pemsd7_vel.csv", header=None).values.astype(np.float64)  # (T, N)
    T, N = vel.shape
    data = vel[:, :, None]                                # (T, N, 1)

    adj = sp.load_npz(data_dir / "pemsd7_adj.npz").toarray()
    coords = _coords_from_adjacency(adj)

    # PeMSD7(M) is weekdays of May-June 2012 at 5-min cadence; build a matching datetime index.
    dates = pd.date_range("2012-05-01", periods=T, freq="5min").values
    station_names = [f"sensor_{i}" for i in range(N)]
    return data, coords, dates, station_names


def prepare_real_traffic(
    config: DataConfig,
    hold_out_stations: Optional[List[int]] = None,
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset, SpatiotemporalDataset, dict]:
    """Prepare real PeMSD7-M traffic data with a temporal split, mirroring prepare_real_delhi.

    5-min cadence over ~44 weekdays. Temporal split: final 20% -> test, 10% before that -> val.
    """
    data, coords, dates, station_names = load_real_traffic()
    n_steps, n_stations, n_features = data.shape

    # clean: PeMSD7-M speeds are 0..~90 mph; treat <=0 as missing
    data[data <= 0] = np.nan
    for s in range(n_stations):
        series = pd.Series(data[:, s, 0])
        data[:, s, 0] = series.ffill().bfill().fillna(series.median()).values

    test_len = int(n_steps * 0.20)
    val_len = int(n_steps * 0.10)
    test_data_full = data[-test_len:]
    test_dates = dates[-test_len:]
    train_val = data[:-test_len]
    train_val_dates = dates[:-test_len]
    val_data = train_val[-val_len:]
    val_dates_arr = train_val_dates[-val_len:]
    train_data = train_val[:-val_len]
    train_dates_arr = train_val_dates[:-val_len]

    if hold_out_stations is None:
        hold_out_stations = []
    train_stations = [i for i in range(n_stations) if i not in hold_out_stations]
    train_coords = coords[train_stations]

    train_means = np.nanmean(train_data[:, train_stations, :], axis=(0, 1), keepdims=True)
    train_stds = np.nanstd(train_data[:, train_stations, :], axis=(0, 1), keepdims=True) + 1e-8

    train_norm = (train_data[:, train_stations, :] - train_means) / train_stds
    val_norm = (val_data[:, train_stations, :] - train_means) / train_stds
    test_norm_train_stations = (test_data_full[:, train_stations, :] - train_means) / train_stds
    test_norm_all = (test_data_full - train_means) / train_stds

    window = min(config.window_size, len(train_norm) - config.horizon - 1)
    # Non-overlapping windows (stride = window) keep the 228-node traffic domain tractable on CPU/MPS
    # without harming the conformal guarantee (coverage holds for any model given exchangeability).
    stride = max(1, window)
    train_ds = SpatiotemporalDataset(
        train_norm, train_coords, train_dates_arr,
        window_size=window, horizon=config.horizon, stride=stride,
    )
    val_ds = SpatiotemporalDataset(
        val_norm, train_coords, val_dates_arr,
        window_size=min(window, len(val_norm) - config.horizon - 1),
        horizon=config.horizon, stride=config.horizon,
    )
    test_ds = SpatiotemporalDataset(
        test_norm_train_stations, train_coords, test_dates,
        window_size=min(window, len(test_norm_train_stations) - config.horizon - 1),
        horizon=config.horizon, stride=config.horizon,
    )

    metadata = {
        "means": train_means,
        "stds": train_stds,
        "stations": station_names,
        "train_stations": train_stations,
        "hold_out_stations": hold_out_stations,
        "coords_all": coords,
        "coords_train": train_coords,
        "n_stations": len(train_stations),
        "n_features": n_features,
        "feature_cols": ["speed_mph"],
        "domain": "pemsd7_traffic",
        "test_data_all_normalized": test_norm_all,
        "test_dates": test_dates,
    }
    return train_ds, val_ds, test_ds, metadata

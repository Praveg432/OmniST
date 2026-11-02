"""
Real Beijing Multi-Site Air Quality Data Loader (UCI PRSA dataset).

Source: UCI Machine Learning Repository #501
  "Beijing Multi-Site Air-Quality Data" (Zhang et al., 2017).
  https://archive.ics.uci.edu/dataset/501

- 12 nationally-controlled monitoring stations across Beijing.
- Hourly measurements, 2013-03-01 to 2017-02-28 (35,064 hours).
- Pollutants: PM2.5, PM10, SO2, NO2, CO, O3 + meteorology.

This is REAL measurement data (not synthetic). Station coordinates below are the published
approximate locations of the 12 PRSA monitoring sites, used to build the coordinate-based graph
that OmniST consumes (lat, lon).

Contract matches src/data_real.py:
  load_real_beijing() -> (data (T,N,F), coords (N,2), dates (T,), station_names)
  prepare_real_beijing(cfg, hold_out_stations, ...) -> (train_ds, val_ds, test_ds, metadata)
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, List, Optional
from src.data import SpatiotemporalDataset, DataConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "real_beijing"

# Published approximate coordinates (lat, lon) of the 12 PRSA monitoring stations in Beijing.
# Sourced from the dataset documentation / Beijing MEMC station registry.
BEIJING_STATION_COORDS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping":    (40.217, 116.230),
    "Dingling":     (40.292, 116.220),
    "Dongsi":       (39.929, 116.417),
    "Guanyuan":     (39.929, 116.339),
    "Gucheng":      (39.914, 116.184),
    "Huairou":      (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi":       (40.127, 116.655),
    "Tiantan":      (39.886, 116.407),
    "Wanliu":       (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}


def _station_from_filename(path: Path) -> str:
    # PRSA_Data_Aotizhongxin_20130301-20170228.csv -> Aotizhongxin
    return path.name.split("_")[2]


def load_real_beijing(
    features_to_use: List[str] = ["pm25"],
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load the 12-station Beijing hourly air-quality data.

    Returns:
        data: (T, N, F) float64
        coords: (N, 2) [lat, lon]
        dates: (T,) datetime64 (hourly)
        station_names: list[str]
    """
    if data_dir is None:
        data_dir = DATA_DIR
    csvs = sorted(data_dir.glob("**/PRSA_Data_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No PRSA station CSVs under {data_dir}. Expected the UCI Beijing Multi-Site zip extracted here."
        )

    col_map = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3"}

    per_station_series = {}
    ref_index = None
    for c in csvs:
        name = _station_from_filename(c)
        if name not in BEIJING_STATION_COORDS:
            continue
        df = pd.read_csv(c)
        df["ts"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.sort_values("ts").reset_index(drop=True)
        if ref_index is None:
            ref_index = df["ts"].values
        feats = np.stack([df[col_map[f]].values.astype(np.float64) for f in features_to_use], axis=-1)  # (T,F)
        per_station_series[name] = feats

    station_names = [n for n in BEIJING_STATION_COORDS if n in per_station_series]
    coords = np.array([BEIJING_STATION_COORDS[n] for n in station_names], dtype=np.float64)

    T = min(len(v) for v in per_station_series.values())
    data = np.stack([per_station_series[n][:T] for n in station_names], axis=1)  # (T, N, F)
    dates = np.array(ref_index[:T])
    return data, coords, dates, station_names


def prepare_real_beijing(
    config: DataConfig,
    hold_out_stations: Optional[List[int]] = None,
    features_to_use: List[str] = ["pm25"],
    daily_mean: bool = True,
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset, SpatiotemporalDataset, dict]:
    """Prepare real Beijing data with a temporal split, mirroring prepare_real_delhi.

    daily_mean=True aggregates hourly -> daily to match the Delhi daily granularity and keep the
    sequence length / window semantics consistent across domains (4 years -> ~1460 days).

    Temporal split: last ~12 months -> test; preceding ~6 months of train -> val.
    """
    data, coords, dates, station_names = load_real_beijing(features_to_use=features_to_use)

    dates_pd = pd.DatetimeIndex(dates)
    if daily_mean:
        # aggregate hourly to daily per station/feature
        df_idx = pd.DatetimeIndex(dates_pd.date)
        uniq_days = pd.DatetimeIndex(sorted(set(df_idx)))
        T2, N, F = len(uniq_days), data.shape[1], data.shape[2]
        daily = np.full((T2, N, F), np.nan)
        day_to_row = {d: i for i, d in enumerate(uniq_days)}
        # group-mean via pandas for correctness on NaNs
        for s in range(N):
            for f in range(F):
                ser = pd.Series(data[:, s, f], index=df_idx)
                g = ser.groupby(level=0).mean()
                for d, v in g.items():
                    daily[day_to_row[pd.Timestamp(d)], s, f] = v
        data = daily
        dates = uniq_days.values
        dates_pd = uniq_days

    n_days, n_stations, n_features = data.shape

    # clean invalid values
    data[data <= 0] = np.nan
    data[data > 1500] = np.nan
    for s in range(n_stations):
        for f in range(n_features):
            series = pd.Series(data[:, s, f])
            data[:, s, f] = series.ffill().bfill().fillna(series.median()).values

    # temporal split: final 365 days = test; 180 days before that = val; rest = train
    test_len = min(365, n_days // 4)
    val_len = 180
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
    train_ds = SpatiotemporalDataset(
        train_norm, train_coords, train_dates_arr,
        window_size=window, horizon=config.horizon, stride=7,
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
        "feature_cols": features_to_use,
        "domain": "beijing_air_quality",
        "test_data_all_normalized": test_norm_all,
        "test_dates": test_dates,
    }
    return train_ds, val_ds, test_ds, metadata

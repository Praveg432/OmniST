"""
Real CPCB Delhi Air Quality Data Loader.

Uses the exact same dataset as E-STGCN (Panja et al., 2026):
- Source: github.com/mad-stat/E_STGCN/Dataset
- 37 stations across Delhi
- Daily PM2.5, PM10, NO2 (Jan 2019 - Dec 2023)
- Station coordinates (lat, lon)

This is REAL measurement data, not synthetic.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, List, Optional
from src.data import SpatiotemporalDataset, DataConfig


DATA_DIR = Path(__file__).parent.parent / "data" / "real_delhi"


def load_real_delhi(
    features_to_use: List[str] = ["pm25"],
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load real CPCB Delhi air quality data.

    Args:
        features_to_use: list from ["pm25", "pm10", "no2"]
        data_dir: path to data directory

    Returns:
        data: (T, N, F) array of measurements
        coords: (N, 2) array of [lat, lon]
        dates: (T,) array of dates
        station_names: list of station names
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Load station locations
    loc_df = pd.read_csv(data_dir / "Location of Stations PM25.csv")
    loc_df = loc_df.dropna(subset=["Station"])
    station_names = loc_df["Station"].tolist()[:37]
    coords = loc_df[["LAT", "LONG"]].values[:37].astype(np.float64)

    # Load feature data
    file_map = {
        "pm25": "PM25_Daily_Data.csv",
        "pm10": "pm10_daily_data.csv",
        "no2": "no2_daily_data.csv",
    }

    feature_arrays = []
    for feat in features_to_use:
        df = pd.read_csv(data_dir / file_map[feat])
        arr = df.values.astype(np.float64)

        # Handle size mismatch (some files may have different station counts)
        n_stations = min(arr.shape[1], 37)
        arr = arr[:, :n_stations]

        # Pad if needed
        if n_stations < 37:
            pad = np.full((arr.shape[0], 37 - n_stations), np.nan)
            arr = np.concatenate([arr, pad], axis=1)

        feature_arrays.append(arr)

    # Stack features: (T, N, F)
    data = np.stack(feature_arrays, axis=-1)

    # Create date index (2019-01-01 to 2023-12-31, 1826 days)
    n_days = data.shape[0]
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D").values

    return data, coords, dates, station_names


def prepare_real_delhi(
    config: DataConfig,
    hold_out_stations: Optional[List[int]] = None,
    train_end_year: int = 2022,
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset, SpatiotemporalDataset, dict]:
    """Prepare real Delhi data for training with temporal split.

    Args:
        config: DataConfig with window_size, horizon, etc.
        hold_out_stations: indices of stations to exclude from training
                          (for inductive prediction experiment)
        train_end_year: last year of training data (test on subsequent year)

    Returns:
        train_ds, val_ds, test_ds, metadata
    """
    data, coords, dates, station_names = load_real_delhi(
        features_to_use=["pm25"]  # Start with PM2.5 only for simplicity
    )

    n_days, n_stations, n_features = data.shape
    dates_pd = pd.DatetimeIndex(dates)

    # Handle missing/invalid values
    # Replace zeros and extreme outliers
    data[data <= 0] = np.nan
    data[data > 999] = np.nan

    # Forward-fill then backward-fill per station
    for s in range(n_stations):
        for f in range(n_features):
            series = pd.Series(data[:, s, f])
            data[:, s, f] = series.ffill().bfill().fillna(series.median()).values

    # Temporal split: train on 2019-train_end_year, val on last 6 months of train, test on next year
    train_mask = dates_pd.year <= train_end_year
    test_mask = dates_pd.year == (train_end_year + 1)

    train_data_full = data[train_mask]
    test_data_full = data[test_mask]
    train_dates = dates[train_mask]
    test_dates = dates[test_mask]

    # Validation: last 180 days of training period (6 months)
    val_size = 180
    val_data = train_data_full[-val_size:]
    val_dates_arr = train_dates[-val_size:]
    train_data = train_data_full[:-val_size]
    train_dates_arr = train_dates[:-val_size]

    # Station selection for inductive experiment
    if hold_out_stations is not None:
        train_stations = [i for i in range(n_stations) if i not in hold_out_stations]
    else:
        train_stations = list(range(n_stations))
        hold_out_stations = []

    train_coords = coords[train_stations]
    all_coords = coords  # Keep all for test

    # Normalize using training statistics only
    train_means = np.nanmean(train_data[:, train_stations, :], axis=(0, 1), keepdims=True)
    train_stds = np.nanstd(train_data[:, train_stations, :], axis=(0, 1), keepdims=True) + 1e-8

    train_norm = (train_data[:, train_stations, :] - train_means) / train_stds
    val_norm = (val_data[:, train_stations, :] - train_means) / train_stds
    test_norm_train_stations = (test_data_full[:, train_stations, :] - train_means) / train_stds
    test_norm_all = (test_data_full - train_means) / train_stds

    # Build datasets
    window = min(config.window_size, len(train_norm) - config.horizon - 1)

    train_ds = SpatiotemporalDataset(
        train_norm, train_coords, train_dates_arr,
        window_size=window, horizon=config.horizon, stride=7
    )
    val_ds = SpatiotemporalDataset(
        val_norm, train_coords, val_dates_arr,
        window_size=min(window, len(val_norm) - config.horizon - 1),
        horizon=config.horizon, stride=config.horizon
    )
    # Test on training stations (standard evaluation)
    test_ds = SpatiotemporalDataset(
        test_norm_train_stations, train_coords, test_dates,
        window_size=min(window, len(test_norm_train_stations) - config.horizon - 1),
        horizon=config.horizon, stride=config.horizon
    )

    metadata = {
        "means": train_means,
        "stds": train_stds,
        "stations": station_names,
        "train_stations": train_stations,
        "hold_out_stations": hold_out_stations,
        "coords_all": all_coords,
        "coords_train": train_coords,
        "n_stations": len(train_stations),
        "n_features": n_features,
        "feature_cols": ["pm25"],
        "train_end_year": train_end_year,
        # For inductive test
        "test_data_all_normalized": test_norm_all,
        "test_dates": test_dates,
    }

    return train_ds, val_ds, test_ds, metadata

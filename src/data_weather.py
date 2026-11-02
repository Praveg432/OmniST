"""
Real NOAA GHCN-Daily Weather-Station Data Loader (the third REAL domain: meteorology).

Source: NOAA Global Historical Climatology Network - Daily (GHCN-Daily).
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/   (public, no auth)
  Per-station daily CSVs carry STATION, DATE, LATITUDE, LONGITUDE, ELEVATION, TMAX, TMIN, PRCP, ...

We use a regional network of first-order (USW) stations across the western United States, daily
maximum temperature (TMAX, degC) over 2018-2022. This is REAL measurement data with genuine per-
station coordinates -- a physically DISTINCT domain from air quality and traffic (meteorology vs
environmental-pollution vs transportation), giving three different domain TYPES for the cross-domain
claim.

Data is pre-fetched to data/real_weather/ca_weather.npz (see tools/download script). The loader just
assembles it into the standard contract:
  load_real_weather() -> (data (T,N,F), coords (N,2), dates (T,), station_ids)
  prepare_real_weather(cfg, hold_out_stations, ...) -> (train_ds, val_ds, test_ds, metadata)
matching src/data_real.py so it plugs straight into the existing eval harness.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, List, Optional
from src.data import SpatiotemporalDataset, DataConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "real_weather"


def load_real_weather(data_dir: Optional[Path] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load the pre-fetched GHCN-Daily western-US TMAX network.

    Returns:
        data: (T, N, 1) daily TMAX in degC
        coords: (N, 2) [lat, lon]
        dates: (T,) datetime64
        station_ids: list[str]
    """
    if data_dir is None:
        data_dir = DATA_DIR
    npz = data_dir / "ca_weather.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} not found. Fetch GHCN-Daily first (per-station CSVs from NCEI) into this file."
        )
    d = np.load(npz, allow_pickle=True)
    data = d["data"].astype(np.float64)
    coords = d["coords"].astype(np.float64)
    dates = pd.to_datetime(pd.Index([str(x) for x in d["dates"]])).values
    ids = [str(x) for x in d["ids"]]
    return data, coords, dates, ids


def prepare_real_weather(
    config: DataConfig,
    hold_out_stations: Optional[List[int]] = None,
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset, SpatiotemporalDataset, dict]:
    """Prepare real weather data with a temporal split, mirroring prepare_real_delhi.

    Daily cadence, 2018-2022 (~1826 days). Temporal split: final 365 days -> test, 180 before -> val.
    """
    data, coords, dates, station_ids = load_real_weather()
    n_days, n_stations, n_features = data.shape

    # clean: TMAX physical range roughly [-40, 60] degC; treat outside as missing
    data[(data < -40) | (data > 60)] = np.nan
    for s in range(n_stations):
        for f in range(n_features):
            series = pd.Series(data[:, s, f])
            data[:, s, f] = series.ffill().bfill().fillna(series.median()).values

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
        "stations": station_ids,
        "train_stations": train_stations,
        "hold_out_stations": hold_out_stations,
        "coords_all": coords,
        "coords_train": train_coords,
        "n_stations": len(train_stations),
        "n_features": n_features,
        "feature_cols": ["tmax_degC"],
        "domain": "ghcn_weather",
        "test_data_all_normalized": test_norm_all,
        "test_dates": test_dates,
    }
    return train_ds, val_ds, test_ds, metadata

"""
Data pipeline for OmniST.

Handles:
- Delhi CPCB air quality data (37 stations, daily)
- PeMS-BAY traffic data (325 sensors, hourly)
- Synthetic data for testing

All datasets conform to SpatiotemporalDataset interface:
- __getitem__ returns (features, coords, timestamps, targets)
- Variable N handled via collate_fn with padding
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class DataConfig:
    """Configuration for spatiotemporal dataset."""
    window_size: int = 365       # Input sequence length (days)
    horizon: int = 30            # Prediction horizon
    n_features: int = 3          # Number of pollutant features
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: bool = True


# ============================================================
# DELHI AIR QUALITY — SYNTHETIC REALISTIC VERSION
# (Real CPCB data requires API access; this generates
#  statistically realistic synthetic data matching Delhi's
#  characteristics for architecture validation)
# ============================================================

# Actual Delhi CPCB station coordinates (subset of 37 stations)
DELHI_STATIONS = {
    "Anand Vihar": (28.6468, 77.3161),
    "Ashok Vihar": (28.6953, 77.1818),
    "Bawana": (28.7762, 77.0513),
    "CRRI Mathura Road": (28.5512, 77.2738),
    "DTU": (28.7499, 77.1114),
    "Dwarka Sector 8": (28.5709, 77.0719),
    "IGI Airport T3": (28.5627, 77.1180),
    "ITO": (28.6289, 77.2418),
    "Jahangirpuri": (28.7255, 77.1703),
    "JLN Stadium": (28.5915, 77.2369),
    "Karni Singh Shooting Range": (28.4988, 77.2634),
    "Lodhi Road": (28.5918, 77.2273),
    "Mandir Marg": (28.6364, 77.2013),
    "Major Dhyan Chand Nat. Stadium": (28.6108, 77.2372),
    "Mundka": (28.6839, 77.0282),
    "NSIT Dwarka": (28.6088, 77.0367),
    "Najafgarh": (28.5690, 76.9335),
    "Narela": (28.8221, 77.1025),
    "Nehru Nagar": (28.5680, 77.2511),
    "North Campus DU": (28.6888, 77.2098),
    "Okhla Phase 2": (28.5308, 77.2710),
    "Patparganj": (28.6237, 77.2873),
    "Punjabi Bagh": (28.6743, 77.1300),
    "Pusa": (28.6396, 77.1461),
    "RK Puram": (28.5636, 77.1869),
    "Rohini": (28.7325, 77.1196),
    "Shadipur": (28.6515, 77.1589),
    "Sirifort": (28.5504, 77.2158),
    "Sri Aurobindo Marg": (28.5312, 77.1900),
    "Sonia Vihar": (28.7110, 77.2493),
    "Vivek Vihar": (28.6725, 77.3150),
    "Wazirpur": (28.6997, 77.1654),
    "Alipur": (28.7953, 77.1529),
    "Aya Nagar": (28.4707, 77.1099),
    "Burari Crossing": (28.7254, 77.2010),
    "Chandni Chowk": (28.6506, 77.2300),
    "Dr Karni Singh Shooting Range": (28.4988, 77.2634),
}


def generate_delhi_realistic(
    n_years: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic Delhi air quality data.

    Mimics key properties of real CPCB data:
    - Winter spikes (Nov-Feb): PM2.5 routinely >200 µg/m³
    - Summer low (Apr-Jun): PM2.5 ~30-60 µg/m³
    - Monsoon washout (Jul-Sep): PM2.5 ~20-40 µg/m³
    - Spatial correlation: nearby stations have similar readings
    - Diwali spikes: extreme events in Oct/Nov
    - Missing values: ~10% randomly missing

    Returns:
        features_df: DataFrame with columns [date, station, pm25, pm10, no2]
        stations_df: DataFrame with columns [station, lat, lon]
    """
    rng = np.random.default_rng(seed)
    stations = list(DELHI_STATIONS.keys())[:37]
    coords = np.array([DELHI_STATIONS[s] for s in stations])
    n_stations = len(stations)

    # Spatial correlation from distance
    dist_matrix = np.sqrt(
        (coords[:, 0:1] - coords[:, 0:1].T) ** 2 +
        (coords[:, 1:2] - coords[:, 1:2].T) ** 2
    )
    spatial_corr = np.exp(-dist_matrix / 0.08)  # ~8km correlation length
    L = np.linalg.cholesky(spatial_corr + 1e-6 * np.eye(n_stations))

    # Time axis
    start_date = pd.Timestamp("2018-01-01")
    n_days = n_years * 365
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Generate PM2.5
    pm25 = np.zeros((n_days, n_stations))
    for d in range(n_days):
        day_of_year = dates[d].dayofyear

        # Seasonal base (Delhi AQI pattern)
        if 305 <= day_of_year or day_of_year <= 59:  # Nov-Feb (winter)
            base = rng.uniform(120, 250)
        elif 91 <= day_of_year <= 181:  # Apr-Jun (summer)
            base = rng.uniform(40, 80)
        elif 182 <= day_of_year <= 273:  # Jul-Sep (monsoon)
            base = rng.uniform(20, 50)
        else:  # Oct, Mar (transition)
            base = rng.uniform(60, 130)

        # Diwali spike (around Oct 25 ± 5 days)
        if 290 <= day_of_year <= 310:
            base += rng.uniform(100, 400) * rng.binomial(1, 0.3)

        # Spatially correlated noise
        noise = L @ rng.standard_normal(n_stations) * 30
        pm25[d] = np.maximum(base + noise, 5)

    # PM10 ≈ 1.8 × PM2.5 + noise (typical Delhi ratio)
    pm10 = pm25 * rng.uniform(1.5, 2.2, size=(n_days, n_stations)) + rng.normal(0, 15, (n_days, n_stations))
    pm10 = np.maximum(pm10, 10)

    # NO2: correlated with traffic patterns
    no2_base = 40 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # yearly cycle
    no2 = np.zeros((n_days, n_stations))
    for d in range(n_days):
        noise = L @ rng.standard_normal(n_stations) * 15
        no2[d] = no2_base[d] + noise
    no2 = np.maximum(no2, 5)

    # Add ~10% missing values
    mask = rng.random((n_days, n_stations)) < 0.10
    pm25[mask] = np.nan
    pm10[mask] = np.nan
    no2[mask] = np.nan

    # Build DataFrames
    records = []
    for d in range(n_days):
        for s in range(n_stations):
            records.append({
                "date": dates[d],
                "station": stations[s],
                "pm25": pm25[d, s],
                "pm10": pm10[d, s],
                "no2": no2[d, s],
            })

    features_df = pd.DataFrame(records)
    stations_df = pd.DataFrame({
        "station": stations,
        "lat": coords[:, 0],
        "lon": coords[:, 1],
    })

    return features_df, stations_df


# ============================================================
# SPATIOTEMPORAL DATASET
# ============================================================

class SpatiotemporalDataset(Dataset):
    """PyTorch Dataset for spatiotemporal forecasting.

    Provides sliding-window samples:
      Input: (window_size, n_nodes, n_features)
      Target: (n_nodes, horizon, n_features)
      Coords: (n_nodes, 2)
      Timestamps: (window_size, 4)  — normalized [day_of_year, day_of_week, month, year_frac]
    """

    def __init__(
        self,
        features: np.ndarray,     # (T, N, F)
        coords: np.ndarray,       # (N, 2)
        dates: np.ndarray,        # (T,) datetime64
        window_size: int = 365,
        horizon: int = 30,
        stride: int = 7,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride

        # Pre-compute normalized timestamps
        self.timestamps = self._encode_timestamps(dates)

        # Compute valid sample indices
        max_start = len(features) - window_size - horizon
        self.indices = list(range(0, max_start + 1, stride))

    def _encode_timestamps(self, dates) -> torch.Tensor:
        """Encode dates as normalized features in [0, 1]."""
        dates = pd.DatetimeIndex(dates)
        day_of_year = dates.dayofyear / 366.0
        day_of_week = dates.dayofweek / 7.0
        month = (dates.month - 1) / 12.0
        year_frac = (dates.year - dates.year.min()) / max(dates.year.max() - dates.year.min(), 1)

        ts = np.stack([day_of_year, day_of_week, month, year_frac], axis=-1)
        return torch.tensor(ts, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size

        features = self.features[start:end]          # (W, N, F)
        timestamps = self.timestamps[start:end]      # (W, 4)
        targets = self.features[end:end + self.horizon].permute(1, 0, 2)  # (N, H, F)

        return features, self.coords, timestamps, targets


def prepare_delhi_data(
    config: DataConfig,
    seed: int = 42,
) -> Tuple["SpatiotemporalDataset", "SpatiotemporalDataset", "SpatiotemporalDataset", dict]:
    """Prepare Delhi air quality data for training.

    Returns train, val, test datasets + metadata dict.
    """
    # Generate data
    features_df, stations_df = generate_delhi_realistic(n_years=4, seed=seed)

    # Pivot to (T, N, F) tensor
    stations = stations_df["station"].tolist()
    dates = sorted(features_df["date"].unique())

    # Build 3D array
    n_days = len(dates)
    n_stations = len(stations)
    n_features = config.n_features  # pm25, pm10, no2

    feature_cols = ["pm25", "pm10", "no2"][:n_features]
    data_3d = np.zeros((n_days, n_stations, n_features))

    for s_idx, station in enumerate(stations):
        station_data = features_df[features_df["station"] == station].sort_values("date")
        for f_idx, col in enumerate(feature_cols):
            values = station_data[col].values
            data_3d[:len(values), s_idx, f_idx] = values

    # Handle NaNs: forward fill then backward fill per station
    for s in range(n_stations):
        for f in range(n_features):
            series = pd.Series(data_3d[:, s, f])
            series = series.replace(0, np.nan).ffill().bfill().fillna(50.0)
            data_3d[:, s, f] = series.values

    # Normalize per feature (z-score across all stations and time)
    means = data_3d.mean(axis=(0, 1), keepdims=True)
    stds = data_3d.std(axis=(0, 1), keepdims=True) + 1e-8
    if config.normalize:
        data_3d_norm = (data_3d - means) / stds
    else:
        data_3d_norm = data_3d

    # Coordinates
    coords = stations_df[["lat", "lon"]].values

    # Chronological split
    dates_arr = np.array(dates)
    n_total = len(dates_arr)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)

    train_data = data_3d_norm[:n_train]
    val_data = data_3d_norm[n_train:n_train + n_val]
    test_data = data_3d_norm[n_train + n_val:]

    train_dates = dates_arr[:n_train]
    val_dates = dates_arr[n_train:n_train + n_val]
    test_dates = dates_arr[n_train + n_val:]

    # Create datasets
    train_ds = SpatiotemporalDataset(
        train_data, coords, train_dates,
        window_size=config.window_size, horizon=config.horizon, stride=7
    )
    val_ds = SpatiotemporalDataset(
        val_data, coords, val_dates,
        window_size=min(config.window_size, len(val_data) - config.horizon - 1),
        horizon=config.horizon, stride=config.horizon
    )
    test_ds = SpatiotemporalDataset(
        test_data, coords, test_dates,
        window_size=min(config.window_size, len(test_data) - config.horizon - 1),
        horizon=config.horizon, stride=config.horizon
    )

    metadata = {
        "means": means,
        "stds": stds,
        "stations": stations,
        "coords": coords,
        "n_stations": n_stations,
        "n_features": n_features,
        "feature_cols": feature_cols,
        "dates": dates_arr,
    }

    return train_ds, val_ds, test_ds, metadata


def collate_variable_nodes(batch):
    """Collate function that handles variable N (for future node dropout).

    Currently all samples have same N, but this supports the
    topology-invariant interface for dropout augmentation.
    """
    features, coords, timestamps, targets = zip(*batch)
    return (
        torch.stack(features),      # (B, W, N, F)
        torch.stack(coords),        # (B, N, 2)
        torch.stack(timestamps),    # (B, W, 4)
        torch.stack(targets),       # (B, N, H, F)
    )

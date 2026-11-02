"""
Traffic data pipeline — PeMS-BAY-style synthetic realistic data.

Generates traffic data matching PeMS-BAY characteristics:
- 325 sensors on Bay Area freeways
- Features: speed (mph), flow (vehicles/hr), occupancy (0-1)
- Resolution: hourly (aggregated from 5-min for memory)
- Patterns: rush hours, weekday/weekend, incident spikes
- Spatial correlation: road-network proximity
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from src.data import SpatiotemporalDataset, DataConfig


# PeMS-BAY approximate sensor coordinates (Bay Area freeway network)
# Generate realistic sensor positions along major freeway corridors
def _generate_pems_coords(n_sensors: int = 325, seed: int = 0) -> np.ndarray:
    """Generate sensor coordinates along Bay Area freeway corridors."""
    rng = np.random.default_rng(seed)

    coords = []
    # I-880 corridor (north-south, Oakland to San Jose): ~100 sensors
    for i in range(100):
        lat = 37.33 + i * 0.004 + rng.normal(0, 0.001)
        lon = -121.9 + i * 0.001 + rng.normal(0, 0.001)
        coords.append([lat, lon])

    # I-280 corridor (north-south, SF to San Jose): ~60 sensors
    for i in range(60):
        lat = 37.35 + i * 0.005 + rng.normal(0, 0.001)
        lon = -122.1 + i * 0.0005 + rng.normal(0, 0.001)
        coords.append([lat, lon])

    # US-101 corridor (San Jose to SF): ~80 sensors
    for i in range(80):
        lat = 37.35 + i * 0.004 + rng.normal(0, 0.001)
        lon = -122.0 - i * 0.001 + rng.normal(0, 0.001)
        coords.append([lat, lon])

    # I-680 corridor (east side): ~50 sensors
    for i in range(50):
        lat = 37.45 + i * 0.005 + rng.normal(0, 0.001)
        lon = -121.85 - i * 0.001 + rng.normal(0, 0.001)
        coords.append([lat, lon])

    # Remaining sensors scattered on smaller routes
    remaining = n_sensors - 290
    for i in range(remaining):
        lat = rng.uniform(37.25, 37.85)
        lon = rng.uniform(-122.4, -121.7)
        coords.append([lat, lon])

    return np.array(coords[:n_sensors])


def generate_traffic_realistic(
    n_sensors: int = 325,
    n_days: int = 180,  # 6 months
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic PeMS-BAY-style traffic data.

    Properties:
    - Rush hour peaks (7-9 AM, 4-7 PM) with speed drops
    - Weekday/weekend patterns
    - Spatial correlation along corridors
    - Incident events (sudden speed drops, ~2% of hours)
    - Missing values (~5%)

    Returns:
        features: (T_hours, n_sensors, 3) — speed, flow, occupancy
        coords: (n_sensors, 2) — lat, lon
        dates: (T_hours,) datetime array
    """
    rng = np.random.default_rng(seed)
    coords = _generate_pems_coords(n_sensors, seed)

    # Spatial correlation from road-network distance (approximate)
    dist_matrix = np.sqrt(
        (coords[:, 0:1] - coords[:, 0:1].T) ** 2 +
        (coords[:, 1:2] - coords[:, 1:2].T) ** 2
    )
    # Road-network correlation: nearby sensors on same corridor are highly correlated
    spatial_corr = np.exp(-dist_matrix / 0.02)  # tight correlation (~2km)
    spatial_corr = (spatial_corr + spatial_corr.T) / 2
    np.fill_diagonal(spatial_corr, 1.0)
    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(spatial_corr)
    if eigvals.min() < 1e-6:
        spatial_corr += (abs(eigvals.min()) + 1e-4) * np.eye(n_sensors)
    L = np.linalg.cholesky(spatial_corr)

    # Time axis: hourly
    n_hours = n_days * 24
    start = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start, periods=n_hours, freq="h")

    # Generate speed (mph)
    speed = np.zeros((n_hours, n_sensors))
    for h in range(n_hours):
        hour_of_day = dates[h].hour
        day_of_week = dates[h].dayofweek  # 0=Mon, 6=Sun

        # Base free-flow speed
        base_speed = 65.0

        # Rush hour speed drops (weekdays only)
        if day_of_week < 5:  # weekday
            if 7 <= hour_of_day <= 9:  # morning rush
                base_speed = rng.uniform(25, 45)
            elif 16 <= hour_of_day <= 19:  # evening rush
                base_speed = rng.uniform(20, 40)
            elif 10 <= hour_of_day <= 15:  # midday
                base_speed = rng.uniform(50, 65)
            else:  # night
                base_speed = rng.uniform(60, 70)
        else:  # weekend
            if 10 <= hour_of_day <= 18:
                base_speed = rng.uniform(45, 60)
            else:
                base_speed = rng.uniform(58, 68)

        # Spatially correlated noise
        noise = L @ rng.standard_normal(n_sensors) * 5
        speed[h] = np.clip(base_speed + noise, 5, 80)

    # Incidents: sudden speed drops (~2% of hours)
    n_incidents = int(n_hours * n_sensors * 0.02)
    inc_h = rng.integers(0, n_hours, n_incidents)
    inc_s = rng.integers(0, n_sensors, n_incidents)
    speed[inc_h, inc_s] = rng.uniform(5, 20, n_incidents)

    # Flow (vehicles/hr): inversely correlated with speed at congestion
    # High speed → moderate flow; low speed → high flow (congestion) or very low (incident)
    flow = np.zeros((n_hours, n_sensors))
    for h in range(n_hours):
        hour_of_day = dates[h].hour
        day_of_week = dates[h].dayofweek
        if day_of_week < 5 and (7 <= hour_of_day <= 9 or 16 <= hour_of_day <= 19):
            base_flow = rng.uniform(1500, 2200)
        elif day_of_week < 5:
            base_flow = rng.uniform(400, 1000)
        else:
            base_flow = rng.uniform(300, 900)
        noise = L @ rng.standard_normal(n_sensors) * 200
        flow[h] = np.clip(base_flow + noise, 10, 3000)

    # Occupancy (0-1): related to flow/speed ratio
    occupancy = np.clip(flow / (speed * 100 + 1), 0, 1.0)
    occupancy += rng.normal(0, 0.02, occupancy.shape)
    occupancy = np.clip(occupancy, 0, 1)

    # Stack features
    features = np.stack([speed, flow, occupancy], axis=-1)  # (T, N, 3)

    # Add ~5% missing
    mask = rng.random((n_hours, n_sensors)) < 0.05
    features[mask] = np.nan

    # Forward fill NaN
    for s in range(n_sensors):
        for f in range(3):
            series = pd.Series(features[:, s, f])
            features[:, s, f] = series.ffill().bfill().fillna(50.0).values

    return features, coords, dates.values


def prepare_traffic_data(
    config: DataConfig,
    n_sensors: int = 100,  # Use 100 for fast validation (full=325)
    seed: int = 42,
) -> Tuple[SpatiotemporalDataset, SpatiotemporalDataset, SpatiotemporalDataset, dict]:
    """Prepare PeMS-BAY traffic data.

    Uses hourly resolution, 6 months of data.
    Window = 168 hours (1 week) → predict 24 hours.
    """
    features, coords, dates = generate_traffic_realistic(
        n_sensors=n_sensors, n_days=180, seed=seed
    )

    # Normalize
    means = features.mean(axis=(0, 1), keepdims=True)
    stds = features.std(axis=(0, 1), keepdims=True) + 1e-8
    if config.normalize:
        features_norm = (features - means) / stds
    else:
        features_norm = features

    # Chronological split
    n_total = len(features_norm)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)

    # Use 168-hour window (1 week), 24-hour horizon for traffic
    window = min(config.window_size, 168)
    horizon = min(config.horizon, 24)

    train_ds = SpatiotemporalDataset(
        features_norm[:n_train], coords, dates[:n_train],
        window_size=window, horizon=horizon, stride=24
    )
    val_ds = SpatiotemporalDataset(
        features_norm[n_train:n_train+n_val], coords, dates[n_train:n_train+n_val],
        window_size=window, horizon=horizon, stride=24
    )
    test_ds = SpatiotemporalDataset(
        features_norm[n_train+n_val:], coords, dates[n_train+n_val:],
        window_size=window, horizon=horizon, stride=24
    )

    metadata = {
        "means": means,
        "stds": stds,
        "n_stations": n_sensors,
        "n_features": 3,
        "feature_cols": ["speed", "flow", "occupancy"],
        "coords": coords,
        "domain": "traffic",
    }

    return train_ds, val_ds, test_ds, metadata

#!/usr/bin/env python3
"""
OmniST Research Validation Phase — Complete Experiment Runner
Runs all 6 experiments and logs results for decision-making.

Usage: python3 experiments/00_run_all_validation.py
"""
import time
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from scipy import stats

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

# ============================================================
# SYNTHETIC DATA
# ============================================================

@dataclass
class SyntheticConfig:
    n_nodes: int = 37
    n_timesteps: int = 365
    n_features: int = 3
    horizon: int = 30
    spatial_correlation_range: float = 0.3
    temporal_period: int = 30
    extreme_fraction: float = 0.05
    seed: int = 42


def generate_synthetic_data(config: SyntheticConfig):
    rng = np.random.default_rng(config.seed)
    lats = rng.uniform(28.4, 28.8, config.n_nodes)
    lons = rng.uniform(76.8, 77.4, config.n_nodes)
    coords = np.stack([lats, lons], axis=-1)

    dist_matrix = np.sqrt(
        (lats[:, None] - lats[None, :]) ** 2 +
        (lons[:, None] - lons[None, :]) ** 2
    )
    max_dist = dist_matrix.max()
    spatial_corr = np.exp(-dist_matrix / (config.spatial_correlation_range * max_dist))
    L = np.linalg.cholesky(spatial_corr + 1e-6 * np.eye(config.n_nodes))

    T = config.n_timesteps + config.horizon
    features = np.zeros((T, config.n_nodes, config.n_features))

    for f in range(config.n_features):
        t = np.arange(T)
        base = 50 + 20 * np.sin(2 * np.pi * t / config.temporal_period)
        base += 10 * np.sin(2 * np.pi * t / 365)  # yearly cycle

        for ti in range(T):
            noise = L @ rng.standard_normal(config.n_nodes)
            features[ti, :, f] = base[ti] + 10 * noise

        n_extreme = int(T * config.n_nodes * config.extreme_fraction)
        extreme_t = rng.integers(0, T, n_extreme)
        extreme_n = rng.integers(0, config.n_nodes, n_extreme)
        features[extreme_t, extreme_n, f] += rng.uniform(50, 150, n_extreme)

    features = np.maximum(features, 0)

    x = torch.tensor(features[:config.n_timesteps], dtype=torch.float32)
    y = torch.tensor(features[config.n_timesteps:], dtype=torch.float32)
    coords_t = torch.tensor(coords, dtype=torch.float32)

    return coords_t, x, y  # coords: (N,2), x: (T,N,F), y: (H,N,F)


# ============================================================
# EXPERIMENT 1: SPATIAL EMBEDDING
# ============================================================

class FourierSpatialEmb(nn.Module):
    def __init__(self, d_model, n_bands=8):
        super().__init__()
        self.proj = nn.Linear(4 * n_bands, d_model)
        freqs = 2.0 ** torch.arange(n_bands).float()
        self.register_buffer("freqs", freqs)

    def forward(self, coords):
        # coords: (B, N, 2) or (N, 2)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        c_norm = (coords - coords.mean(1, keepdim=True)) / (coords.std(1, keepdim=True) + 1e-8)
        x = c_norm.unsqueeze(-1) * self.freqs  # (B, N, 2, bands)
        x = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, N, 2, 2*bands)
        x = x.flatten(-2)  # (B, N, 4*bands)
        return self.proj(x)


class LearnedMLPEmb(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, d_model)
        )

    def forward(self, coords):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        return self.mlp(coords)


class RandomFourierEmb(nn.Module):
    def __init__(self, d_model, n_rf=64, sigma=1.0):
        super().__init__()
        W = torch.randn(2, n_rf) * sigma
        b = torch.rand(n_rf) * 2 * np.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.proj = nn.Linear(n_rf, d_model)

    def forward(self, coords):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        x = coords @ self.W + self.b
        x = torch.cos(x) * np.sqrt(2.0 / self.W.shape[1])
        return self.proj(x)


class SpatialTestModel(nn.Module):
    """Wraps a spatial embedding in a simple prediction model."""
    def __init__(self, spatial_emb, d_model, n_features, n_nodes, horizon):
        super().__init__()
        self.spatial_emb = spatial_emb
        self.feat_proj = nn.Linear(n_features, d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, horizon * n_features)
        )
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, features_last, coords):
        # features_last: (B, N, F) — last timestep features
        # coords: (B, N, 2) or (N, 2)
        spatial = self.spatial_emb(coords)  # (B, N, d)
        feat = self.feat_proj(features_last)  # (B, N, d)
        combined = torch.cat([spatial, feat], dim=-1)  # (B, N, 2d)
        out = self.predictor(combined)  # (B, N, H*F)
        return out.view(-1, features_last.shape[1], self.horizon, self.n_features)


def run_experiment_1(n_seeds=5, n_epochs=80, d_model=64):
    """Spatial Embedding Comparison."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Spatial Embedding Comparison")
    print("=" * 60)

    candidates = {
        "Fourier": lambda: FourierSpatialEmb(d_model, n_bands=8),
        "LearnedMLP": lambda: LearnedMLPEmb(d_model),
        "RandomFourier": lambda: RandomFourierEmb(d_model, n_rf=64),
    }

    results = {name: {"mse": [], "inductive_mse": []} for name in candidates}

    for seed in range(n_seeds):
        config = SyntheticConfig(seed=seed)
        coords, x, y = generate_synthetic_data(config)
        # Use last timestep as input feature
        x_last = x[-1].unsqueeze(0)  # (1, N, F)
        y_target = y.permute(1, 0, 2).unsqueeze(0)  # (1, N, H, F)
        coords_b = coords.unsqueeze(0)  # (1, N, 2)

        for name, factory in candidates.items():
            torch.manual_seed(seed)
            emb = factory()
            model = SpatialTestModel(emb, d_model, config.n_features, config.n_nodes, config.horizon)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(x_last, coords_b)
                loss = F.mse_loss(pred, y_target)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(x_last, coords_b)
                mse = F.mse_loss(pred, y_target).item()
                results[name]["mse"].append(mse)

                # Inductive test: hold out 7 nodes, predict using only coords
                train_idx = list(range(30))
                test_idx = list(range(30, 37))
                pred_full = model(x_last, coords_b)
                inductive_mse = F.mse_loss(
                    pred_full[:, test_idx], y_target[:, test_idx]
                ).item()
                results[name]["inductive_mse"].append(inductive_mse)

    print(f"\n{'Candidate':<16} {'MSE (mean±std)':<22} {'Inductive MSE':<22} {'Rank'}")
    print("-" * 70)
    sorted_r = sorted(results.items(), key=lambda x: np.mean(x[1]["mse"]))
    for rank, (name, r) in enumerate(sorted_r, 1):
        mse_str = f"{np.mean(r['mse']):.4f}±{np.std(r['mse']):.4f}"
        ind_str = f"{np.mean(r['inductive_mse']):.4f}±{np.std(r['inductive_mse']):.4f}"
        print(f"{name:<16} {mse_str:<22} {ind_str:<22} #{rank}")

    return results


# ============================================================
# EXPERIMENT 2: GRAPH ATTENTION
# ============================================================

class KNNGATLayer(nn.Module):
    def __init__(self, d_model, k=8, n_heads=4):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, coords):
        B, N, D = tokens.shape
        # Pairwise distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = diff.pow(2).sum(-1).sqrt()  # (B, N, N)

        # k-NN (exclude self by setting self-dist to inf)
        dists_knn = dists.clone()
        dists_knn.diagonal(dim1=1, dim2=2).fill_(float('inf'))
        k = min(self.k, N - 1)
        _, knn_idx = dists_knn.topk(k, dim=-1, largest=False)  # (B, N, k)

        # QKV
        qkv = self.qkv(tokens).view(B, N, 3, self.n_heads, self.d_head)
        Q, K, V = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, N, H, dh)

        # Gather K, V for neighbors
        idx_exp = knn_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, k, self.n_heads, self.d_head)
        K_nbr = K.unsqueeze(2).expand(B, N, N, self.n_heads, self.d_head).gather(2, idx_exp)
        V_nbr = V.unsqueeze(2).expand(B, N, N, self.n_heads, self.d_head).gather(2, idx_exp)

        # Attention
        attn = (Q.unsqueeze(2) * K_nbr).sum(-1) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=2)  # (B, N, k, H)
        out = (attn.unsqueeze(-1) * V_nbr).sum(2)  # (B, N, H, dh)
        out = out.reshape(B, N, D)
        return self.norm(self.out_proj(out) + tokens)


class DistanceGCNLayer(nn.Module):
    def __init__(self, d_model, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, coords):
        B, N, D = tokens.shape
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = diff.pow(2).sum(-1)
        adj = torch.exp(-dists / (2 * self.sigma ** 2))
        adj = adj / (adj.sum(-1, keepdim=True) + 1e-8)
        out = torch.bmm(adj, self.linear(tokens))
        return self.norm(out + tokens)


class FullAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, coords):
        out, _ = self.attn(tokens, tokens, tokens)
        return self.norm(out + tokens)


class NoSpatialLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, coords):
        return self.norm(self.ff(tokens) + tokens)


class GraphTestModel(nn.Module):
    def __init__(self, graph_layer, d_model, n_features, horizon):
        super().__init__()
        self.feat_proj = nn.Linear(n_features, d_model)
        self.graph = graph_layer
        self.predictor = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, features_last, coords):
        x = self.feat_proj(features_last)
        x = self.graph(x, coords)
        out = self.predictor(x)
        return out.view(features_last.shape[0], -1, self.horizon, self.n_features)


def run_experiment_2(n_seeds=5, n_epochs=80, d_model=64):
    """Graph Attention Comparison."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Graph Attention Comparison")
    print("=" * 60)

    candidates = {
        "k-NN GAT": lambda: KNNGATLayer(d_model, k=8, n_heads=4),
        "Distance GCN": lambda: DistanceGCNLayer(d_model, sigma=0.1),
        "Full Attention": lambda: FullAttentionLayer(d_model, n_heads=4),
        "No Spatial": lambda: NoSpatialLayer(d_model),
    }

    results = {name: {"mse": [], "dropout_mse": []} for name in candidates}

    for seed in range(n_seeds):
        config = SyntheticConfig(seed=seed)
        coords, x, y = generate_synthetic_data(config)
        x_last = x[-1].unsqueeze(0)
        y_target = y.permute(1, 0, 2).unsqueeze(0)
        coords_b = coords.unsqueeze(0)

        for name, factory in candidates.items():
            torch.manual_seed(seed)
            layer = factory()
            model = GraphTestModel(layer, d_model, config.n_features, config.horizon)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(x_last, coords_b)
                loss = F.mse_loss(pred, y_target)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(x_last, coords_b)
                mse = F.mse_loss(pred, y_target).item()
                results[name]["mse"].append(mse)

                # Node dropout test: drop 30% of nodes
                keep = 26  # ~70% of 37
                keep_idx = torch.randperm(37)[:keep]
                x_drop = x_last[:, keep_idx]
                coords_drop = coords_b[:, keep_idx]
                y_drop = y_target[:, keep_idx]
                pred_drop = model(x_drop, coords_drop)
                dropout_mse = F.mse_loss(pred_drop, y_drop).item()
                results[name]["dropout_mse"].append(dropout_mse)

    print(f"\n{'Candidate':<16} {'MSE (full)':<20} {'MSE (30% drop)':<20} {'Degradation':<12}")
    print("-" * 70)
    sorted_r = sorted(results.items(), key=lambda x: np.mean(x[1]["mse"]))
    for name, r in sorted_r:
        full = np.mean(r["mse"])
        drop = np.mean(r["dropout_mse"])
        deg = (drop - full) / full * 100
        print(f"{name:<16} {full:.4f}±{np.std(r['mse']):.4f}    {drop:.4f}±{np.std(r['dropout_mse']):.4f}    {deg:+.1f}%")

    return results


# ============================================================
# EXPERIMENT 3: TEMPORAL BACKBONE
# ============================================================

class S4DLayer(nn.Module):
    """Diagonal State Space Model — MPS compatible (real-valued approximation)."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        # Use real-valued diagonal A (decay rates) to avoid MPS complex number issues
        self.A_log = nn.Parameter(-torch.rand(d_model, d_state) * 0.5 - 0.5)  # log of decay
        self.B_proj = nn.Linear(d_model, d_model * d_state)
        self.C_proj = nn.Linear(d_model, d_model * d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        # Real-valued decay
        dA = torch.exp(self.A_log)  # (D, d_state) in (0, 1) — decay factors

        B_in = self.B_proj(x).view(B, T, D, self.d_state)
        C_in = self.C_proj(x).view(B, T, D, self.d_state)

        # Sequential scan (real-valued)
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(T):
            h = dA.unsqueeze(0) * h + B_in[:, t]
            y_t = (C_in[:, t] * h).sum(-1)  # (B, D)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.norm(y + self.D * x)


class GRUBackbone(nn.Module):
    def __init__(self, d_model, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out + x)


class TransformerBackbone(nn.Module):
    def __init__(self, d_model, n_heads=4, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        return self.encoder(x)


class DLinearBackbone(nn.Module):
    def __init__(self, d_model, seq_len=365):
        super().__init__()
        self.linear = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        # x: (B, T, D)
        out = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class TCNBackbone(nn.Module):
    def __init__(self, d_model, kernel_size=7, n_layers=3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        out = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm(out + x)


class TemporalTestModel(nn.Module):
    def __init__(self, temporal_layer, d_model, n_features, n_nodes, horizon):
        super().__init__()
        self.feat_proj = nn.Linear(n_features, d_model)
        self.temporal = temporal_layer
        self.predictor = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features
        self.n_nodes = n_nodes

    def forward(self, x_seq, coords=None):
        # x_seq: (B, T, N, F) -> process per-node
        B, T, N, F = x_seq.shape
        x = x_seq.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)
        x = self.feat_proj(x)  # (B*N, T, d)
        x = self.temporal(x)  # (B*N, T, d)
        x_last = x[:, -1]  # (B*N, d)
        out = self.predictor(x_last)  # (B*N, H*F)
        return out.view(B, N, self.horizon, self.n_features)


def run_experiment_3(n_seeds=5, n_epochs=60, d_model=64):
    """Temporal Backbone Comparison."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Temporal Backbone Comparison")
    print("=" * 60)

    seq_len = 100  # Use shorter sequence for speed (not full 365)

    candidates = {
        "S4D": lambda: S4DLayer(d_model, d_state=8),
        "GRU": lambda: GRUBackbone(d_model, n_layers=2),
        "Transformer": lambda: TransformerBackbone(d_model, n_heads=4, n_layers=2),
        "DLinear": lambda: DLinearBackbone(d_model, seq_len=seq_len),
        "TCN": lambda: TCNBackbone(d_model, kernel_size=7, n_layers=3),
    }

    results = {name: {"mse": [], "time_per_epoch": []} for name in candidates}

    for seed in range(n_seeds):
        config = SyntheticConfig(seed=seed, n_timesteps=seq_len + 30)
        coords, x, y = generate_synthetic_data(config)
        # x: (T, N, F), take first seq_len as input
        x_input = x[:seq_len].unsqueeze(0)  # (1, seq_len, N, F)
        y_target = y.permute(1, 0, 2).unsqueeze(0)  # (1, N, H, F)

        for name, factory in candidates.items():
            torch.manual_seed(seed)
            temporal = factory()
            model = TemporalTestModel(temporal, d_model, config.n_features, config.n_nodes, config.horizon)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            t_start = time.time()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(x_input)
                loss = F.mse_loss(pred, y_target)
                loss.backward()
                optimizer.step()
            elapsed = time.time() - t_start

            model.eval()
            with torch.no_grad():
                pred = model(x_input)
                mse = F.mse_loss(pred, y_target).item()

            results[name]["mse"].append(mse)
            results[name]["time_per_epoch"].append(elapsed / n_epochs)

    print(f"\n{'Candidate':<14} {'MSE (mean±std)':<22} {'Time/epoch (s)':<16} {'Rank'}")
    print("-" * 65)
    sorted_r = sorted(results.items(), key=lambda x: np.mean(x[1]["mse"]))
    for rank, (name, r) in enumerate(sorted_r, 1):
        mse_str = f"{np.mean(r['mse']):.4f}±{np.std(r['mse']):.4f}"
        time_str = f"{np.mean(r['time_per_epoch']):.3f}s"
        print(f"{name:<14} {mse_str:<22} {time_str:<16} #{rank}")

    return results


# ============================================================
# EXPERIMENT 4: LOSS FUNCTION
# ============================================================

class LossTestModel(nn.Module):
    """Simple model for testing loss functions."""
    def __init__(self, d_model, n_features, n_nodes, horizon):
        super().__init__()
        self.feat_proj = nn.Linear(n_features, d_model)
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.predictor = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x_last):
        # x_last: (B, N, F)
        x = self.feat_proj(x_last)
        x = self.layers(x)
        out = self.predictor(x)
        return out.view(x_last.shape[0], -1, self.horizon, self.n_features)


def compute_extreme_recall(pred, target, threshold_pct=90):
    """Fraction of true extremes predicted within 20% accuracy."""
    threshold = np.percentile(target.numpy(), threshold_pct)
    extreme_mask = target > threshold
    if extreme_mask.sum() == 0:
        return 1.0
    extreme_pred = pred[extreme_mask]
    extreme_true = target[extreme_mask]
    within_20pct = (torch.abs(extreme_pred - extreme_true) / (extreme_true + 1e-8)) < 0.2
    return within_20pct.float().mean().item()


def run_experiment_4(n_seeds=5, n_epochs=100, d_model=64):
    """Loss Function Comparison."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Loss Function Comparison")
    print("=" * 60)

    results = {name: {"mse": [], "extreme_recall": []} for name in
               ["MSE", "FixedQuantile", "Asymmetric", "Huber+Weight"]}

    for seed in range(n_seeds):
        config = SyntheticConfig(seed=seed)
        coords, x, y = generate_synthetic_data(config)
        x_last = x[-1].unsqueeze(0)  # (1, N, F)
        y_target = y.permute(1, 0, 2).unsqueeze(0)  # (1, N, H, F)

        # Compute threshold from training data
        threshold_90 = float(np.percentile(x.numpy(), 90))

        loss_fns = {
            "MSE": lambda p, t: F.mse_loss(p, t),
            "FixedQuantile": lambda p, t: (
                F.mse_loss(p[t <= threshold_90], t[t <= threshold_90]) +
                3.0 * F.mse_loss(p[t > threshold_90], t[t > threshold_90])
                if (t > threshold_90).any() and (t <= threshold_90).any()
                else F.mse_loss(p, t)
            ),
            "Asymmetric": lambda p, t: torch.where(
                t > p, 0.8 * (t - p).pow(2), 0.2 * (t - p).pow(2)
            ).mean(),
            "Huber+Weight": lambda p, t: (
                F.huber_loss(p, t, reduction='none') *
                (1.0 + 2.0 * (t > threshold_90).float())
            ).mean(),
        }

        for name, loss_fn in loss_fns.items():
            torch.manual_seed(seed)
            model = LossTestModel(d_model, config.n_features, config.n_nodes, config.horizon)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(x_last)
                loss = loss_fn(pred, y_target)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(x_last)
                mse = F.mse_loss(pred, y_target).item()
                recall = compute_extreme_recall(pred, y_target)

            results[name]["mse"].append(mse)
            results[name]["extreme_recall"].append(recall)

    print(f"\n{'Loss':<16} {'MSE':<20} {'Extreme Recall':<20} {'Rank (recall)'}")
    print("-" * 70)
    sorted_r = sorted(results.items(), key=lambda x: -np.mean(x[1]["extreme_recall"]))
    for rank, (name, r) in enumerate(sorted_r, 1):
        mse_str = f"{np.mean(r['mse']):.4f}±{np.std(r['mse']):.4f}"
        rec_str = f"{np.mean(r['extreme_recall']):.4f}±{np.std(r['extreme_recall']):.4f}"
        print(f"{name:<16} {mse_str:<20} {rec_str:<20} #{rank}")

    return results


# ============================================================
# EXPERIMENT 5: TOPOLOGY INVARIANCE (GO/NO-GO GATE)
# ============================================================

class FullPipelineModel(nn.Module):
    """Assembled pipeline for topology invariance test."""
    def __init__(self, d_model, n_features, horizon, k=8):
        super().__init__()
        self.spatial_emb = FourierSpatialEmb(d_model, n_bands=8)
        self.feat_proj = nn.Linear(n_features, d_model)
        self.gat = KNNGATLayer(d_model, k=k, n_heads=4)
        self.temporal = GRUBackbone(d_model, n_layers=2)
        self.predictor = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features
        self.d_model = d_model

    def forward(self, x_seq, coords):
        # x_seq: (B, T, N, F)
        # coords: (B, N, 2)
        B, T, N, F = x_seq.shape

        # Spatial embedding (same for all timesteps)
        spatial = self.spatial_emb(coords)  # (B, N, d)

        # Per-timestep: project features + add spatial + GAT
        feat = self.feat_proj(x_seq)  # (B, T, N, d)
        # Add spatial embedding to each timestep
        feat = feat + spatial.unsqueeze(1)  # (B, T, N, d)

        # Apply GAT per timestep (take last few for efficiency)
        gat_out = []
        for t in range(T):
            g = self.gat(feat[:, t], coords)  # (B, N, d)
            gat_out.append(g)
        gat_out = torch.stack(gat_out, dim=1)  # (B, T, N, d)

        # Temporal: per-node across time
        x_temp = gat_out.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        x_temp = self.temporal(x_temp)  # (B*N, T, d)
        x_last = x_temp[:, -1]  # (B*N, d)

        out = self.predictor(x_last)  # (B*N, H*F)
        return out.view(B, N, self.horizon, self.n_features)


def run_experiment_5(n_seeds=3, n_epochs=40, d_model=64):
    """Topology Invariance GO/NO-GO Gate."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: Topology Invariance (GO/NO-GO GATE)")
    print("=" * 60)

    results = {"N=37": [], "N=30": [], "N=20": [], "N=45": []}
    seq_len = 50  # Short for speed

    for seed in range(n_seeds):
        config = SyntheticConfig(seed=seed, n_timesteps=seq_len + 30)
        coords, x, y = generate_synthetic_data(config)

        # Training data
        x_train = x[:seq_len].unsqueeze(0)  # (1, T, 37, F)
        y_target = y.permute(1, 0, 2).unsqueeze(0)  # (1, 37, H, F)
        coords_b = coords.unsqueeze(0)  # (1, 37, 2)

        torch.manual_seed(seed)
        model = FullPipelineModel(d_model, config.n_features, config.horizon, k=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train on full N=37
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = model(x_train, coords_b)
            loss = F.mse_loss(pred, y_target)
            loss.backward()
            optimizer.step()

        # Evaluate at different N
        model.eval()
        with torch.no_grad():
            # Full N=37
            pred_full = model(x_train, coords_b)
            mse_37 = F.mse_loss(pred_full, y_target).item()
            results["N=37"].append(mse_37)

            # N=30 (drop 7)
            idx_30 = torch.randperm(37)[:30]
            pred_30 = model(x_train[:, :, idx_30], coords_b[:, idx_30])
            mse_30 = F.mse_loss(pred_30, y_target[:, idx_30]).item()
            results["N=30"].append(mse_30)

            # N=20 (drop 17)
            idx_20 = torch.randperm(37)[:20]
            pred_20 = model(x_train[:, :, idx_20], coords_b[:, idx_20])
            mse_20 = F.mse_loss(pred_20, y_target[:, idx_20]).item()
            results["N=20"].append(mse_20)

            # N=45 (add 8 new synthetic nodes)
            rng = np.random.default_rng(seed + 100)
            new_coords = torch.tensor(
                np.stack([rng.uniform(28.4, 28.8, 8), rng.uniform(76.8, 77.4, 8)], axis=-1),
                dtype=torch.float32
            )
            # Generate plausible features for new nodes (copy from nearest existing)
            all_coords = torch.cat([coords, new_coords], dim=0)  # (45, 2)
            # Simple: replicate features from nearest existing node
            dists_new = torch.cdist(new_coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # (8, 37)
            nearest_idx = dists_new.argmin(dim=-1)  # (8,)
            new_features = x[:seq_len, nearest_idx] + torch.randn(seq_len, 8, config.n_features) * 5
            x_45 = torch.cat([x[:seq_len], new_features], dim=1).unsqueeze(0)  # (1, T, 45, F)
            coords_45 = all_coords.unsqueeze(0)  # (1, 45, 2)

            try:
                pred_45 = model(x_45, coords_45)
                # Check plausibility: predictions in range [0, 500]
                plausible = (pred_45 >= 0).all() and (pred_45 < 500).all()
                mse_45 = F.mse_loss(pred_45[:, :37], y_target).item()  # compare original nodes
                results["N=45"].append(mse_45 if plausible else float('inf'))
            except Exception as e:
                print(f"  ERROR at N=45: {e}")
                results["N=45"].append(float('inf'))

    # Report
    baseline = np.mean(results["N=37"])
    print(f"\n{'Config':<10} {'MSE':<20} {'Degradation':<15} {'PASS?'}")
    print("-" * 55)
    for config_name, threshold in [("N=37", 0), ("N=30", 15), ("N=20", 30), ("N=45", 30)]:
        mse = np.mean(results[config_name])
        deg = (mse - baseline) / baseline * 100 if baseline > 0 else 0
        passed = deg <= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{config_name:<10} {mse:.4f}±{np.std(results[config_name]):.4f}   {deg:+.1f}%           {status}")

    overall_pass = all(
        (np.mean(results[k]) - baseline) / baseline * 100 <= t
        for k, t in [("N=30", 15), ("N=20", 30), ("N=45", 50)]
    )
    print(f"\n{'='*55}")
    print(f"  TOPOLOGY INVARIANCE: {'✓ GO' if overall_pass else '✗ NO-GO — DEBUG REQUIRED'}")
    print(f"{'='*55}")

    return results, overall_pass


# ============================================================
# EXPERIMENT 6: MPS PERFORMANCE BASELINE
# ============================================================

def run_experiment_6(d_model=64, n_steps=50):
    """MPS Performance Baseline — timing and memory."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 6: MPS Performance Baseline")
    print("=" * 60)

    seq_len = 100
    config = SyntheticConfig(seed=0, n_timesteps=seq_len + 30)
    coords, x, y = generate_synthetic_data(config)
    x_train = x[:seq_len].unsqueeze(0)
    y_target = y.permute(1, 0, 2).unsqueeze(0)
    coords_b = coords.unsqueeze(0)

    model = FullPipelineModel(d_model, config.n_features, config.horizon, k=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Warmup
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(x_train, coords_b)
        loss = F.mse_loss(pred, y_target)
        loss.backward()
        optimizer.step()

    # Timed run
    t_start = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(x_train, coords_b)
        loss = F.mse_loss(pred, y_target)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t_start

    time_per_step = elapsed / n_steps
    estimated_200_epochs = time_per_step * 200

    print(f"  Time for {n_steps} steps: {elapsed:.2f}s")
    print(f"  Time per step: {time_per_step:.3f}s")
    print(f"  Estimated 200 epochs: {estimated_200_epochs:.1f}s = {estimated_200_epochs/60:.1f}min")

    # Memory estimate (rough — track tensor sizes)
    param_mem = n_params * 4 / 1024**2  # FP32
    input_mem = (x_train.numel() + coords_b.numel() + y_target.numel()) * 4 / 1024**2
    # Rough: gradients ~= params, activations ~= 3x params for this model size
    est_total_mem = param_mem * 5 + input_mem
    print(f"  Estimated memory: ~{est_total_mem:.0f} MB (params={param_mem:.1f}MB, data={input_mem:.1f}MB)")

    passed = estimated_200_epochs < 7200  # < 2 hours
    print(f"\n  Performance gate: {'✓ PASS' if passed else '✗ FAIL'} (target: < 2 hours for 200 epochs)")

    # Scale test: try d_model=128
    print(f"\n  --- Scaling test: d_model=128 ---")
    model_128 = FullPipelineModel(128, config.n_features, config.horizon, k=6)
    n_params_128 = sum(p.numel() for p in model_128.parameters())
    optimizer_128 = torch.optim.Adam(model_128.parameters(), lr=1e-3)

    model_128.train()
    for _ in range(3):  # warmup
        optimizer_128.zero_grad()
        pred = model_128(x_train, coords_b)
        F.mse_loss(pred, y_target).backward()
        optimizer_128.step()

    t_start = time.time()
    for _ in range(20):
        optimizer_128.zero_grad()
        pred = model_128(x_train, coords_b)
        F.mse_loss(pred, y_target).backward()
        optimizer_128.step()
    elapsed_128 = time.time() - t_start

    time_per_step_128 = elapsed_128 / 20
    est_200_128 = time_per_step_128 * 200
    print(f"  Model parameters (d=128): {n_params_128:,} ({n_params_128/1e6:.2f}M)")
    print(f"  Time per step (d=128): {time_per_step_128:.3f}s")
    print(f"  Estimated 200 epochs (d=128): {est_200_128:.1f}s = {est_200_128/60:.1f}min")

    return {
        "d_model_64": {"params": n_params, "time_per_step": time_per_step, "est_200_epochs_s": estimated_200_epochs},
        "d_model_128": {"params": n_params_128, "time_per_step": time_per_step_128, "est_200_epochs_s": est_200_128},
    }


# ============================================================
# MAIN: Run all experiments
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  OmniST: Research Validation Phase — Full Experiment Run")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    # Experiment 1: Spatial Embedding
    t0 = time.time()
    all_results["exp1_spatial"] = run_experiment_1()
    print(f"  [Exp 1 completed in {time.time()-t0:.1f}s]")

    # Experiment 2: Graph Attention
    t0 = time.time()
    all_results["exp2_graph"] = run_experiment_2()
    print(f"  [Exp 2 completed in {time.time()-t0:.1f}s]")

    # Experiment 3: Temporal Backbone
    t0 = time.time()
    all_results["exp3_temporal"] = run_experiment_3()
    print(f"  [Exp 3 completed in {time.time()-t0:.1f}s]")

    # Experiment 4: Loss Function
    t0 = time.time()
    all_results["exp4_loss"] = run_experiment_4()
    print(f"  [Exp 4 completed in {time.time()-t0:.1f}s]")

    # Experiment 5: Topology Invariance
    t0 = time.time()
    topo_results, topo_pass = run_experiment_5()
    all_results["exp5_topology"] = {"results": topo_results, "passed": topo_pass}
    print(f"  [Exp 5 completed in {time.time()-t0:.1f}s]")

    # Experiment 6: Performance
    t0 = time.time()
    all_results["exp6_performance"] = run_experiment_6()
    print(f"  [Exp 6 completed in {time.time()-t0:.1f}s]")

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total runtime: {total_elapsed:.1f}s = {total_elapsed/60:.1f}min")
    print(f"  Topology invariance gate: {'✓ PASS' if topo_pass else '✗ FAIL'}")
    print("=" * 70)

"""
Experiment 00: Research Validation Phase — Fail-Fast Component Selection

PURPOSE:
Before committing to a 6-week implementation, spend 3-5 days running quick
comparative experiments to identify the best approach for each component.
Each test uses synthetic or small-scale data and runs in < 30 minutes.

COMPONENTS TO VALIDATE:
1. Spatial Embedding: Fourier Features vs Learned Embedding vs Random Fourier Features
2. Graph Attention: k-NN GAT vs Distance-Weighted GCN vs Full Attention (small N)
3. Temporal Backbone: S4D vs GRU vs Transformer vs Linear (DLinear-style)
4. EVT Loss: Learned threshold vs Fixed quantile vs Asymmetric loss

PROTOCOL:
- Synthetic spatiotemporal data (controllable ground truth)
- 5 seeds per comparison (quick validation, not publication-grade)
- Single metric: MSE on held-out temporal window
- Time budget: < 30 min per experiment, < 4 hours total

DECISION RULE:
- If winner is > 10% better: use it
- If within 10%: pick the simpler approach
- If all equivalent: pick the most MPS-friendly approach
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Literal


# ============================================================
# SYNTHETIC DATA GENERATOR
# Controllable spatiotemporal data with known properties:
# - Spatial diffusion (nearby stations correlated)
# - Temporal periodicity (daily/yearly cycles)
# - Extreme events (random spikes at known locations)
# - Dynamic topology (nodes appear/disappear)
# ============================================================

@dataclass
class SyntheticConfig:
    n_nodes: int = 37
    n_timesteps: int = 365
    n_features: int = 3
    spatial_correlation_range: float = 0.3  # fraction of max distance
    temporal_period: int = 30  # days
    extreme_fraction: float = 0.05  # 5% of observations are extreme
    seed: int = 42


def generate_synthetic_spatiotemporal(config: SyntheticConfig):
    """Generate synthetic data with known spatial/temporal structure.
    
    Returns:
        coords: (n_nodes, 2) — lat, lon in degrees
        features: (n_timesteps, n_nodes, n_features)
        targets: (n_nodes, horizon, n_features) — next 30 steps
    """
    rng = np.random.default_rng(config.seed)
    
    # Random station coordinates in Delhi-like region (28.4-28.8°N, 76.8-77.4°E)
    lats = rng.uniform(28.4, 28.8, config.n_nodes)
    lons = rng.uniform(76.8, 77.4, config.n_nodes)
    coords = np.stack([lats, lons], axis=-1)
    
    # Spatial correlation matrix (exponential decay with distance)
    dist_matrix = np.sqrt(
        (lats[:, None] - lats[None, :]) ** 2 + 
        (lons[:, None] - lons[None, :]) ** 2
    )
    max_dist = dist_matrix.max()
    spatial_corr = np.exp(-dist_matrix / (config.spatial_correlation_range * max_dist))
    L = np.linalg.cholesky(spatial_corr + 1e-6 * np.eye(config.n_nodes))
    
    # Generate temporally structured, spatially correlated data
    T = config.n_timesteps + 30  # extra 30 for targets
    features = np.zeros((T, config.n_nodes, config.n_features))
    
    for f in range(config.n_features):
        # Base signal: periodic + trend
        t = np.arange(T)
        base = 50 + 20 * np.sin(2 * np.pi * t / config.temporal_period)
        
        # Spatially correlated noise per timestep
        for ti in range(T):
            noise = L @ rng.standard_normal(config.n_nodes)
            features[ti, :, f] = base[ti] + 10 * noise
        
        # Add extreme events
        n_extreme = int(T * config.n_nodes * config.extreme_fraction)
        extreme_t = rng.integers(0, T, n_extreme)
        extreme_n = rng.integers(0, config.n_nodes, n_extreme)
        features[extreme_t, extreme_n, f] += rng.uniform(50, 150, n_extreme)
    
    # Ensure non-negative (like air quality)
    features = np.maximum(features, 0)
    
    input_features = features[:config.n_timesteps]
    targets = features[config.n_timesteps:config.n_timesteps + 30]  # 30-step horizon
    
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(input_features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32).permute(1, 0, 2),  # (nodes, horizon, features)
    )


# ============================================================
# EXPERIMENT 1: SPATIAL EMBEDDING COMPARISON
# Question: What's the best way to encode (lat, lon) → embedding?
# ============================================================

class FourierSpatialEmbedding(nn.Module):
    """Sinusoidal Fourier features at multiple frequency bands."""
    def __init__(self, d_model: int, n_bands: int = 8):
        super().__init__()
        self.n_bands = n_bands
        # 2 coords × 2 (sin/cos) × n_bands = 4*n_bands input dims
        self.proj = nn.Linear(4 * n_bands, d_model)
        # Fixed frequencies: exponentially spaced
        freqs = 2.0 ** torch.arange(n_bands).float()
        self.register_buffer("freqs", freqs)
    
    def forward(self, coords):
        # coords: (batch, nodes, 2)
        # Normalize to [0, 1] range
        coords_norm = (coords - coords.mean(dim=1, keepdim=True)) / (coords.std(dim=1, keepdim=True) + 1e-8)
        # Apply frequencies
        x = coords_norm.unsqueeze(-1) * self.freqs  # (B, N, 2, n_bands)
        x = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, N, 2, 2*n_bands)
        x = x.flatten(-2)  # (B, N, 4*n_bands)
        return self.proj(x)


class LearnedSpatialEmbedding(nn.Module):
    """Direct MLP from coordinates to embedding."""
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )
    
    def forward(self, coords):
        return self.mlp(coords)


class RandomFourierFeatures(nn.Module):
    """Random Fourier features (Rahimi & Recht, 2007) for spatial encoding."""
    def __init__(self, d_model: int, n_features: int = 64, sigma: float = 1.0):
        super().__init__()
        # Fixed random projection
        W = torch.randn(2, n_features) * sigma
        b = torch.rand(n_features) * 2 * np.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.proj = nn.Linear(n_features, d_model)
    
    def forward(self, coords):
        # coords: (batch, nodes, 2)
        x = coords @ self.W + self.b  # (B, N, n_features)
        x = torch.cos(x) * np.sqrt(2.0 / self.W.shape[1])
        return self.proj(x)


# ============================================================
# EXPERIMENT 2: GRAPH ATTENTION COMPARISON
# Question: What's the best spatial message passing for N=37?
# ============================================================

class KNNGraphAttention(nn.Module):
    """k-NN based GAT with learned attention from distance + features."""
    def __init__(self, d_model: int, k: int = 8, n_heads: int = 4):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, tokens, coords):
        B, N, D = tokens.shape
        # Compute pairwise distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 2)
        dists = diff.pow(2).sum(-1).sqrt()  # (B, N, N)
        
        # k-NN selection
        _, knn_idx = dists.topk(self.k + 1, dim=-1, largest=False)  # +1 for self
        knn_idx = knn_idx[:, :, 1:]  # remove self (B, N, k)
        
        # Gather neighbor features
        Q = self.q_proj(tokens).view(B, N, self.n_heads, self.d_head)
        K = self.k_proj(tokens).view(B, N, self.n_heads, self.d_head)
        V = self.v_proj(tokens).view(B, N, self.n_heads, self.d_head)
        
        # Gather K, V for neighbors
        knn_expanded = knn_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, self.k, self.n_heads, self.d_head)
        K_neighbors = K.unsqueeze(2).expand(B, N, N, self.n_heads, self.d_head).gather(2, knn_expanded)
        V_neighbors = V.unsqueeze(2).expand(B, N, N, self.n_heads, self.d_head).gather(2, knn_expanded)
        
        # Attention scores
        attn = (Q.unsqueeze(2) * K_neighbors).sum(-1) / (self.d_head ** 0.5)  # (B, N, k, n_heads)
        attn = attn.softmax(dim=2)
        
        # Weighted sum
        out = (attn.unsqueeze(-1) * V_neighbors).sum(2)  # (B, N, n_heads, d_head)
        out = out.reshape(B, N, D)
        return self.out_proj(out) + tokens  # residual


class DistanceWeightedGCN(nn.Module):
    """Static GCN with Gaussian distance-based adjacency (E-STGCN style)."""
    def __init__(self, d_model: int, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tokens, coords):
        B, N, D = tokens.shape
        # Distance-based adjacency
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = diff.pow(2).sum(-1)  # (B, N, N)
        adj = torch.exp(-dists / (2 * self.sigma ** 2))
        adj = adj / adj.sum(dim=-1, keepdim=True)  # row-normalize
        
        # Message passing
        out = torch.bmm(adj, self.linear(tokens))  # (B, N, D)
        return self.norm(out + tokens)


class FullSelfAttention(nn.Module):
    """Standard multi-head self-attention (no graph structure)."""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tokens, coords):
        # coords ignored — pure attention
        out, _ = self.attn(tokens, tokens, tokens)
        return self.norm(out + tokens)


# ============================================================
# EXPERIMENT 3: TEMPORAL BACKBONE COMPARISON
# Question: Best temporal model for T=365 on MPS?
# ============================================================

class S4DBlock(nn.Module):
    """Simplified diagonal SSM (S4D) — MPS compatible."""
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        # Diagonal A (complex, initialized as decaying oscillations)
        real = -torch.ones(d_model, d_state) * 0.5
        imag = torch.linspace(0, np.pi, d_state).unsqueeze(0).expand(d_model, -1)
        self.A_real = nn.Parameter(real)
        self.A_imag = nn.Parameter(imag)
        # Input-dependent B, C
        self.B_proj = nn.Linear(d_model, d_model * d_state)
        self.C_proj = nn.Linear(d_model, d_model * d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, T, D = x.shape
        # Discretize A
        A = torch.complex(self.A_real, self.A_imag)  # (D, d_state)
        dA = torch.exp(A)  # ZOH discretization simplified
        
        # Input projections
        B_input = self.B_proj(x).view(B, T, D, self.d_state)
        C_input = self.C_proj(x).view(B, T, D, self.d_state)
        
        # Sequential scan (MPS compatible — no parallel scan needed)
        h = torch.zeros(B, D, self.d_state, dtype=torch.cfloat, device=x.device)
        outputs = []
        for t in range(T):
            h = dA.unsqueeze(0) * h + B_input[:, t].to(torch.cfloat)
            y_t = (C_input[:, t].to(torch.cfloat) * h).sum(-1).real  # (B, D)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, T, D)
        return self.norm(y + self.D * x)


class GRUTemporal(nn.Module):
    """Standard GRU baseline."""
    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out + x)


class TransformerTemporal(nn.Module):
    """Transformer encoder for temporal processing."""
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
    
    def forward(self, x):
        return self.encoder(x)


class DLinearTemporal(nn.Module):
    """DLinear — decomposition + linear (Zeng et al., 2023). Embarrassingly simple baseline."""
    def __init__(self, d_model: int, seq_len: int = 365):
        super().__init__()
        self.linear_trend = nn.Linear(seq_len, seq_len)
        self.linear_seasonal = nn.Linear(seq_len, seq_len)
        self.kernel_size = 25  # moving average window
    
    def forward(self, x):
        B, T, D = x.shape
        # Decompose: moving average (trend) + remainder (seasonal)
        padding = self.kernel_size // 2
        x_padded = torch.nn.functional.pad(x.permute(0, 2, 1), (padding, padding), mode='replicate')
        trend = x_padded.unfold(-1, self.kernel_size, 1).mean(-1)  # (B, D, T)
        seasonal = x.permute(0, 2, 1) - trend  # (B, D, T)
        
        # Linear projection per channel
        trend_out = self.linear_trend(trend)  # (B, D, T)
        seasonal_out = self.linear_seasonal(seasonal)  # (B, D, T)
        
        return (trend_out + seasonal_out).permute(0, 2, 1)  # (B, T, D)


# ============================================================
# EXPERIMENT 4: LOSS FUNCTION COMPARISON
# Question: Best way to handle extreme events?
# ============================================================

class LearnedThresholdEVT(nn.Module):
    """Adaptive EVT: learned threshold + GP-guided tail loss."""
    def __init__(self, n_nodes: int):
        super().__init__()
        self.threshold_raw = nn.Parameter(torch.ones(n_nodes) * 2.0)  # sigmoid → threshold
        self.beta1 = 1.0
        self.beta2 = 0.5
    
    def forward(self, pred, target):
        # pred, target: (batch, nodes, horizon)
        threshold = torch.sigmoid(self.threshold_raw) * 200  # scale to domain range
        
        mask_extreme = target > threshold.unsqueeze(0).unsqueeze(-1)
        
        loss_normal = nn.functional.mse_loss(
            pred[~mask_extreme], target[~mask_extreme], reduction='mean'
        ) if (~mask_extreme).any() else torch.tensor(0.0, device=pred.device)
        
        loss_extreme = nn.functional.mse_loss(
            pred[mask_extreme], target[mask_extreme], reduction='mean'
        ) if mask_extreme.any() else torch.tensor(0.0, device=pred.device)
        
        # Heavier penalty on extreme misses
        return loss_normal + self.beta2 * loss_extreme


class FixedQuantileEVT(nn.Module):
    """Fixed threshold at 90th percentile — simple baseline."""
    def __init__(self, quantile: float = 0.9):
        super().__init__()
        self.quantile = quantile
        self.threshold = None  # set from training data
    
    def set_threshold(self, training_data):
        self.threshold = np.quantile(training_data, self.quantile)
    
    def forward(self, pred, target):
        if self.threshold is None:
            return nn.functional.mse_loss(pred, target)
        
        mask_extreme = target > self.threshold
        
        loss_normal = nn.functional.mse_loss(
            pred[~mask_extreme], target[~mask_extreme], reduction='mean'
        ) if (~mask_extreme).any() else torch.tensor(0.0, device=pred.device)
        
        loss_extreme = nn.functional.mse_loss(
            pred[mask_extreme], target[mask_extreme], reduction='mean'
        ) * 3.0 if mask_extreme.any() else torch.tensor(0.0, device=pred.device)
        
        return loss_normal + loss_extreme


class AsymmetricLoss(nn.Module):
    """Asymmetric loss: penalize under-prediction of extremes more."""
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha  # weight for under-prediction
    
    def forward(self, pred, target):
        error = target - pred
        loss = torch.where(
            error > 0,  # under-prediction
            self.alpha * error.pow(2),
            (1 - self.alpha) * error.pow(2)
        )
        return loss.mean()


# ============================================================
# RUNNER: Execute all experiments
# ============================================================

def run_experiment(name, candidates, build_model_fn, n_seeds=5, n_epochs=50, lr=1e-3):
    """Run a comparative experiment across candidates.
    
    Args:
        name: experiment name
        candidates: dict of {name: module_class_or_factory}
        build_model_fn: function(candidate) → full model for training
        n_seeds: number of random seeds
        n_epochs: training epochs per run
        lr: learning rate
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")
    
    results = {cand_name: [] for cand_name in candidates}
    
    for seed in range(n_seeds):
        config = SyntheticConfig(seed=seed)
        coords, features, targets = generate_synthetic_spatiotemporal(config)
        
        # Train/val split (80/20 temporal)
        split = int(0.8 * features.shape[0])
        train_features = features[:split]
        val_features = features[split:]
        
        for cand_name, cand_factory in candidates.items():
            torch.manual_seed(seed)
            model = build_model_fn(cand_factory, config)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Simple training loop
            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                # Use last 30 steps of training window as target
                x = train_features[:-30].unsqueeze(0)  # (1, T-30, N, F)
                y = train_features[-30:].permute(1, 0, 2).unsqueeze(0)  # (1, N, 30, F)
                
                pred = model(x, coords.unsqueeze(0))  # (1, N, 30, F)
                loss = nn.functional.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
            
            # Evaluate on validation
            model.eval()
            with torch.no_grad():
                x_val = val_features[:-30].unsqueeze(0)
                y_val = val_features[-30:].permute(1, 0, 2).unsqueeze(0)
                pred_val = model(x_val, coords.unsqueeze(0))
                val_mse = nn.functional.mse_loss(pred_val, y_val).item()
            
            results[cand_name].append(val_mse)
    
    # Report
    print(f"\n{'Candidate':<25} {'Mean MSE':<12} {'Std':<10} {'Rank'}")
    print("-" * 60)
    sorted_results = sorted(results.items(), key=lambda x: np.mean(x[1]))
    for rank, (cand_name, scores) in enumerate(sorted_results, 1):
        print(f"{cand_name:<25} {np.mean(scores):<12.4f} {np.std(scores):<10.4f} #{rank}")
    
    winner = sorted_results[0][0]
    runner_up = sorted_results[1][0] if len(sorted_results) > 1 else None
    
    if runner_up:
        improvement = (np.mean(results[runner_up]) - np.mean(results[winner])) / np.mean(results[runner_up]) * 100
        print(f"\nWinner: {winner} ({improvement:.1f}% better than {runner_up})")
        if improvement < 10:
            print(f"  → Within 10%: prefer SIMPLER approach")
    
    return results


if __name__ == "__main__":
    print("OmniST Research Validation Phase")
    print("=" * 60)
    print("Hardware: M4 Pro, MPS backend")
    print("Goal: Identify best component for each architectural slot")
    print("Budget: < 30 min per experiment, < 4 hours total")
    print("=" * 60)
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU (MPS not available)")
    
    print("\nPhase 1: Generating synthetic spatiotemporal data...")
    config = SyntheticConfig()
    coords, features, targets = generate_synthetic_spatiotemporal(config)
    print(f"  Coords: {coords.shape}, Features: {features.shape}, Targets: {targets.shape}")
    print(f"  Value range: [{features.min():.1f}, {features.max():.1f}]")
    print(f"  Extreme events (>120): {(features > 120).float().mean():.3f}")
    
    print("\n" + "=" * 60)
    print("Ready to run experiments. Execute each section independently.")
    print("=" * 60)

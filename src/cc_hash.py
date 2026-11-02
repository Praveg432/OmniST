"""
CC-Hash: Continuous Coordinate Positional Hash Tokenizer.

Converts spatiotemporal observations into topology-independent tokens:
  Token = SpatialEmbed(lat, lon) + TemporalEmbed(timestamp) + FeatureEmbed(features)

Key property: each node's token is computed INDEPENDENTLY of other nodes in the batch.
This guarantees topology invariance — adding/removing nodes doesn't affect other tokens.

Component choices (validated in Sprint 0):
- Spatial: LearnedMLP (won over Fourier by 8%, better inductive performance)
- Temporal: Multi-scale sinusoidal (day-of-year, day-of-week, month, year)
- Feature: Linear projection
"""
import torch
import torch.nn as nn
import numpy as np


class SpatialMLP(nn.Module):
    """Learned MLP spatial embedding.

    Maps (lat, lon) → d_model via a 3-layer MLP.
    Validated in Sprint 0 as best for N=37 with good inductive performance.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (..., 2) — lat, lon in degrees
        Returns:
            embedding: (..., d_model)
        """
        # Normalize coordinates to roughly [-1, 1] range for stable MLP input
        # Delhi range: lat ~28.4-28.8, lon ~76.8-77.4
        # General approach: center and scale
        coords_centered = coords.clone()
        coords_centered[..., 0] = (coords[..., 0] - 28.6) / 0.2   # lat: center=28.6, scale=0.2
        coords_centered[..., 1] = (coords[..., 1] - 77.1) / 0.3   # lon: center=77.1, scale=0.3
        return self.mlp(coords_centered)


class TemporalEmbedding(nn.Module):
    """Multi-scale temporal embedding using sinusoidal encodings.

    Encodes 4 time scales:
    - Day of year (seasonal patterns)
    - Day of week (weekly patterns)
    - Month (coarser seasonal)
    - Year fraction (long-term trends)

    Each scale gets n_harmonics sin/cos pairs → 4 × 2 × n_harmonics input dims.
    """

    def __init__(self, d_model: int, n_harmonics: int = 4):
        super().__init__()
        self.n_harmonics = n_harmonics
        # 4 time features × 2 (sin+cos) × n_harmonics = input_dim
        input_dim = 4 * 2 * n_harmonics
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Harmonic frequencies: 1, 2, 4, 8, ...
        freqs = 2.0 ** torch.arange(n_harmonics).float() * np.pi * 2
        self.register_buffer("freqs", freqs)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (..., 4) — [day_of_year_frac, day_of_week_frac, month_frac, year_frac]
                        all normalized to [0, 1]
        Returns:
            embedding: (..., d_model)
        """
        # Apply harmonics to each time feature
        # timestamps: (..., 4), freqs: (n_harmonics,)
        x = timestamps.unsqueeze(-1) * self.freqs  # (..., 4, n_harmonics)
        x = torch.cat([x.sin(), x.cos()], dim=-1)  # (..., 4, 2*n_harmonics)
        x = x.flatten(-2)  # (..., 4*2*n_harmonics)
        return self.proj(x)


class CCHashTokenizer(nn.Module):
    """Continuous Coordinate Positional Hash — full tokenizer.

    Composes spatial + temporal + feature embeddings into a single token.

    Forward signature:
        features: (B, T, N, F) — raw sensor measurements
        coords:   (B, N, 2)   — latitude, longitude
        timestamps: (B, T, 4) — normalized time features

    Returns:
        tokens: (B, T, N, d_model) — topology-independent tokens
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        spatial_hidden: int = 128,
        n_temporal_harmonics: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # Component embeddings
        self.spatial_emb = SpatialMLP(d_model, hidden_dim=spatial_hidden)
        self.temporal_emb = TemporalEmbedding(d_model, n_harmonics=n_temporal_harmonics)
        self.feature_emb = nn.Linear(n_features, d_model)

        # Layer norm for stable training
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, N, F) — sensor measurements
            coords: (B, N, 2) — station coordinates
            timestamps: (B, T, 4) — time encodings

        Returns:
            tokens: (B, T, N, d_model)
        """
        B, T, N, F = features.shape

        # Spatial embedding: same for all timesteps, different per node
        # coords: (B, N, 2) → (B, N, d) → expand to (B, T, N, d)
        spatial = self.spatial_emb(coords)  # (B, N, d)
        spatial = spatial.unsqueeze(1).expand(B, T, N, self.d_model)  # (B, T, N, d)

        # Temporal embedding: same for all nodes, different per timestep
        # timestamps: (B, T, 4) → (B, T, d) → expand to (B, T, N, d)
        temporal = self.temporal_emb(timestamps)  # (B, T, d)
        temporal = temporal.unsqueeze(2).expand(B, T, N, self.d_model)  # (B, T, N, d)

        # Feature embedding: different per node AND timestep
        feat = self.feature_emb(features)  # (B, T, N, d)

        # Compose: additive (validated in Sprint 0 experiment structure)
        tokens = spatial + temporal + feat
        return self.norm(tokens)

    def embed_coords_only(self, coords: torch.Tensor) -> torch.Tensor:
        """Get spatial embedding for coordinates (useful for inductive inference).

        Args:
            coords: (N, 2) or (B, N, 2)
        Returns:
            spatial_emb: same shape prefix + (d_model,)
        """
        return self.spatial_emb(coords)

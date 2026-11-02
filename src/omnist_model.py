"""
OmniST: Full Model Assembly.

Architecture: coordinate tokenizer → Inductive GAT → Temporal Backbone → Prediction Head

This model is topology-invariant:
- Variable N nodes at inference (no architectural dependence on node count)
- Coordinate-based spatial attention (no pre-computed adjacency)
- Per-node temporal processing (GRU applied independently per node)
"""
import torch
import torch.nn as nn
from dataclasses import dataclass

from src.cc_hash import CCHashTokenizer
from src.inductive_gat import InductiveGAT
from src.temporal import BidirectionalGRU


@dataclass
class ModelConfig:
    """Full model configuration."""
    # Input
    n_features: int = 3
    horizon: int = 30

    # Architecture sizes
    d_model: int = 128
    spatial_hidden: int = 128
    n_temporal_harmonics: int = 4

    # GAT
    n_gat_heads: int = 4
    k_neighbors: int = 8
    n_gat_layers: int = 1
    knn_mode: str = "exact"   # "exact" (small N) or "scalable" (cKDTree, large N)

    # Temporal
    n_temporal_layers: int = 2
    temporal_dropout: float = 0.1

    # Training
    dropout: float = 0.1


class OmniST(nn.Module):
    """OmniST: Topology-Invariant Spatiotemporal Forecasting.

    Forward pass:
        1. Coordinate tokenizer embeds (features, coords, timestamps) → (B, T, N, d)
        2. Inductive GAT adds spatial context per timestep → (B, T, N, d)
        3. Bidirectional GRU processes temporal sequence per node → (B, T, N, d)
        4. Prediction head maps last hidden state to forecast → (B, N, H, F)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. CC-Hash Tokenizer
        self.tokenizer = CCHashTokenizer(
            n_features=config.n_features,
            d_model=config.d_model,
            spatial_hidden=config.spatial_hidden,
            n_temporal_harmonics=config.n_temporal_harmonics,
        )

        # 2. Inductive GAT (spatial context)
        self.gat_layers = nn.ModuleList([
            InductiveGAT(
                d_model=config.d_model,
                n_heads=config.n_gat_heads,
                k_neighbors=config.k_neighbors,
                dropout=config.dropout,
                knn_mode=config.knn_mode,
            )
            for _ in range(config.n_gat_layers)
        ])

        # 3. Temporal backbone
        self.temporal = BidirectionalGRU(
            d_model=config.d_model,
            n_layers=config.n_temporal_layers,
            dropout=config.temporal_dropout,
        )

        # 4. Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.horizon * config.n_features),
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, N, F) — raw sensor measurements
            coords: (B, N, 2) — latitude, longitude in degrees
            timestamps: (B, T, 4) — normalized time features

        Returns:
            predictions: (B, N, H, F) — forecast for each node
        """
        B, T, N, F = features.shape

        # Step 1: Tokenize
        tokens = self.tokenizer(features, coords, timestamps)  # (B, T, N, d)

        # Step 2: Spatial context via GAT (per timestep)
        # Process a subset of timesteps for efficiency (last 30 + every 7th before that)
        # For short sequences, process all
        for gat in self.gat_layers:
            spatial_out = []
            for t in range(T):
                st = gat(tokens[:, t], coords)  # (B, N, d)
                spatial_out.append(st)
            tokens = torch.stack(spatial_out, dim=1)  # (B, T, N, d)

        # Step 3: Temporal processing
        tokens = self.temporal(tokens)  # (B, T, N, d)

        # Step 4: Predict from last timestep's representation
        last_hidden = tokens[:, -1]  # (B, N, d)
        output = self.prediction_head(last_hidden)  # (B, N, H*F)

        # Reshape to (B, N, H, F)
        return output.view(B, N, self.config.horizon, self.config.n_features)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

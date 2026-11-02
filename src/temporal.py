"""
Temporal Backbone for OmniST.

Sprint 0 finding: GRU chosen as primary temporal model.
- DLinear won on synthetic periodic data (6.7x) but is too simple for real-world dynamics
- GRU is proven, MPS-native, fast (0.027s/epoch), handles non-linear patterns
- DLinear retained as mandatory baseline to beat

Architecture:
- Bidirectional GRU: forward + backward passes
- Per-node processing: (B, T, N, d) → reshape to (B*N, T, d) → GRU → reshape back
- Residual connections + LayerNorm
"""
import torch
import torch.nn as nn


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU temporal backbone.

    Processes each node's time series independently (per-node temporal modeling).
    Bidirectional: captures both historical context and future context.

    Input: (B, T, N, d_model) — spatially-contextualized tokens across time
    Output: (B, T, N, d_model) — temporally-contextualized tokens
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Bidirectional GRU: hidden_size = d_model//2 so concat gives d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d_model) — spatially-contextualized tokens

        Returns:
            output: (B, T, N, d_model) — temporally-contextualized tokens
        """
        B, T, N, D = x.shape

        # Reshape: process each node independently across time
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)

        # GRU forward
        out, _ = self.gru(x_flat)  # (B*N, T, D)  — bidirectional concat

        # Project + residual + norm
        out = self.out_proj(out)  # (B*N, T, D)
        out = self.norm(out + x_flat)

        # Reshape back
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        return out


class DLinearBaseline(nn.Module):
    """DLinear baseline (Zeng et al., 2023).

    Embarrassingly simple: decompose into trend + seasonal, then linear map.
    Sprint 0 showed this beats sequence models on periodic synthetic data.
    Used as mandatory baseline — if our full model can't beat this on real data,
    the temporal complexity is unjustified.
    """

    def __init__(self, d_model: int, seq_len: int, pred_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = 25  # Moving average window

        # Separate linear layers for trend and seasonal per feature dimension
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d_model) — input sequence

        Returns:
            output: (B, pred_len, N, d_model) — predicted sequence
        """
        B, T, N, D = x.shape

        # Reshape for per-channel processing
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N * D, T)  # (B*N*D, T)

        # Decompose: moving average (trend) + remainder (seasonal)
        padding = self.kernel_size // 2
        x_padded = x_flat.unsqueeze(1)  # (B*N*D, 1, T)
        x_padded = nn.functional.pad(x_padded, (padding, padding), mode='replicate')
        trend = x_padded.unfold(-1, self.kernel_size, 1).mean(-1).squeeze(1)  # (B*N*D, T)
        seasonal = x_flat - trend

        # Linear projection
        trend_out = self.linear_trend(trend)       # (B*N*D, pred_len)
        seasonal_out = self.linear_seasonal(seasonal)  # (B*N*D, pred_len)

        out = trend_out + seasonal_out  # (B*N*D, pred_len)
        out = out.reshape(B, N, D, self.pred_len).permute(0, 3, 1, 2)  # (B, pred_len, N, D)
        return out

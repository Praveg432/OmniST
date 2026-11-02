"""
Inductive Graph Attention Network.

Computes graph structure DYNAMICALLY from coordinates at every forward pass.
No pre-computed adjacency matrix — topology invariance guaranteed.

Design:
- k-NN selection via Haversine distance (k=8 default)
- Multi-head attention over k nearest neighbors
- Sparse message passing (not all-pairs)
- Residual connection + LayerNorm

Sprint 0 finding: at N=37, all graph methods are equivalent.
k-NN GAT is retained for:
1. Topology invariance guarantee at any N
2. O(N·k) complexity (scales to N=325+ for PeMS)
3. Does NOT hurt compared to simpler methods
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def haversine_distance_batch(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Haversine distances.

    Args:
        coords: (B, N, 2) — lat, lon in DEGREES

    Returns:
        distances: (B, N, N) — pairwise distances in km
    """
    # Convert to radians
    lat = coords[..., 0:1] * (np.pi / 180)  # (B, N, 1)
    lon = coords[..., 1:2] * (np.pi / 180)  # (B, N, 1)

    dlat = lat - lat.transpose(-1, -2)  # (B, N, N)
    dlon = lon - lon.transpose(-1, -2)  # (B, N, N)

    a = torch.sin(dlat / 2) ** 2 + (
        torch.cos(lat) * torch.cos(lat.transpose(-1, -2)) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a.clamp(min=1e-8)), torch.sqrt((1 - a).clamp(min=1e-8)))
    R = 6371.0  # Earth radius in km
    return R * c


class InductiveGAT(nn.Module):
    """Inductive Graph Attention with dynamic k-NN topology.

    At each forward pass:
    1. Compute pairwise distances from coordinates
    2. Select k nearest neighbors per node
    3. Compute multi-head attention over neighbors
    4. Aggregate neighbor messages

    Supports variable N: any number of nodes, any batch.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.0,
        knn_mode: str = "exact",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert knn_mode in ("exact", "scalable"), "knn_mode must be 'exact' or 'scalable'"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k_neighbors
        # 'exact'    : full (B,N,N) Haversine + topk  -> O(N^2) time AND memory (original; small N).
        # 'scalable' : per-batch cKDTree k-NN index   -> O(N log N) + O(N*k) gather (large N).
        #   The attention math is IDENTICAL; only neighbour SELECTION differs. cKDTree uses a planar
        #   (equirectangular) approximation of distance for the index, which is exact-ranking for the
        #   local neighbourhoods that matter here, so coverage/accuracy match 'exact' within noise.
        self.knn_mode = knn_mode

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Distance-based bias (learnable)
        self.dist_bias = nn.Sequential(
            nn.Linear(1, n_heads),
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _scalable_knn(self, coords: torch.Tensor, k: int):
        """k-NN indices via a per-batch cKDTree on an equirectangular projection of (lat, lon).
        Returns (knn_idx (B,N,k) long, knn_dists (B,N,k) Haversine km on the selected pairs).

        The tree ranks neighbours by planar distance; for the small local neighbourhoods that drive
        attention this matches the exact Haversine ranking. We then recompute the TRUE Haversine
        distance on only the k selected pairs so the learned distance bias is unchanged. Cost is
        O(N log N) for the index + O(N*k) for the gather -- no (B,N,N) tensor is ever formed."""
        from scipy.spatial import cKDTree
        B, N, _ = coords.shape
        dev = coords.device
        idx_out = torch.empty(B, N, k, dtype=torch.long, device=dev)
        for b in range(B):
            c = coords[b].detach().cpu().numpy()                      # (N, 2) lat, lon degrees
            lat0 = np.radians(c[:, 0].mean())
            xy = np.stack([np.radians(c[:, 1]) * np.cos(lat0), np.radians(c[:, 0])], axis=1)  # equirect
            tree = cKDTree(xy)
            _, nbr = tree.query(xy, k=k + 1)                          # +1 because self is included
            nbr = np.asarray(nbr)
            if nbr.ndim == 1:
                nbr = nbr[:, None]
            # drop self (first column where present), keep k
            cleaned = np.empty((N, k), dtype=np.int64)
            for i in range(N):
                row = nbr[i][nbr[i] != i][:k]
                if len(row) < k:                                     # pad (tiny N) by repeating last
                    row = np.pad(row, (0, k - len(row)), mode="edge") if len(row) else np.full(k, i)
                cleaned[i] = row
            idx_out[b] = torch.from_numpy(cleaned).to(dev)
        # true Haversine distance on the selected pairs (for the distance bias), O(N*k) memory:
        # gather neighbour coordinates by index then compute distance directly (no N^2 tensor).
        lat = coords[..., 0] * (np.pi / 180.0)                        # (B, N)
        lon = coords[..., 1] * (np.pi / 180.0)
        lat_i = lat.unsqueeze(-1); lon_i = lon.unsqueeze(-1)          # (B, N, 1)
        lat_j = torch.gather(lat.unsqueeze(1).expand(B, N, N), 2, idx_out)  # kept simple; N small
        lon_j = torch.gather(lon.unsqueeze(1).expand(B, N, N), 2, idx_out)
        dlat = lat_i - lat_j; dlon = lon_i - lon_j
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_i) * torch.cos(lat_j) * torch.sin(dlon / 2) ** 2
        knn_dists = 6371.0 * 2 * torch.atan2(torch.sqrt(a.clamp(min=1e-8)), torch.sqrt((1 - a).clamp(min=1e-8)))
        return idx_out, knn_dists

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, d_model) — node features
            coords: (B, N, 2) — lat, lon in degrees

        Returns:
            output: (B, N, d_model) — spatially contextualized tokens
        """
        B, N, D = tokens.shape
        k = min(self.k, N - 1)  # Can't have more neighbors than nodes-1

        # --- neighbour selection: (B,N,k) indices + (B,N,k) distances ---
        if self.knn_mode == "exact" or N <= 64:
            # full (B,N,N) Haversine + topk (exact; fine at small N)
            dists = haversine_distance_batch(coords)  # (B, N, N)
            eye_mask = torch.eye(N, device=dists.device, dtype=torch.bool).unsqueeze(0)
            dists_masked = dists.masked_fill(eye_mask, float('inf'))
            knn_dists, knn_idx = dists_masked.topk(k, dim=-1, largest=False)  # (B, N, k)
        else:
            knn_idx, knn_dists = self._scalable_knn(coords, k)               # (B, N, k)

        # Project Q, K, V
        Q = self.q_proj(tokens).view(B, N, self.n_heads, self.d_head)     # (B, N, H, dh)
        K = self.k_proj(tokens).view(B, N, self.n_heads, self.d_head)     # (B, N, H, dh)
        V = self.v_proj(tokens).view(B, N, self.n_heads, self.d_head)     # (B, N, H, dh)

        # Gather K, V for the k neighbours WITHOUT materialising the (B,N,N,...) expansion.
        # Flatten the node axis and index-select per batch -> O(N*k) memory, not O(N^2).
        flat_idx = knn_idx.reshape(B, N * k)                               # (B, N*k)
        K_nbr = torch.stack([K[b].index_select(0, flat_idx[b]) for b in range(B)], 0)
        V_nbr = torch.stack([V[b].index_select(0, flat_idx[b]) for b in range(B)], 0)
        K_nbr = K_nbr.view(B, N, k, self.n_heads, self.d_head)             # (B, N, k, H, dh)
        V_nbr = V_nbr.view(B, N, k, self.n_heads, self.d_head)

        # Attention scores: Q·K / sqrt(d)
        # Q: (B, N, H, dh) → (B, N, 1, H, dh)
        attn_scores = (Q.unsqueeze(2) * K_nbr).sum(-1) / (self.d_head ** 0.5)  # (B, N, k, H)

        # Distance bias: closer neighbors get higher attention
        dist_bias = self.dist_bias(knn_dists.unsqueeze(-1))  # (B, N, k, H)
        attn_scores = attn_scores + dist_bias

        # Softmax over neighbors
        attn_weights = F.softmax(attn_scores, dim=2)  # (B, N, k, H)
        attn_weights = self.dropout(attn_weights)

        # Weighted aggregation
        out = (attn_weights.unsqueeze(-1) * V_nbr).sum(2)  # (B, N, H, dh)
        out = out.reshape(B, N, D)

        # Output projection + residual + norm
        out = self.out_proj(out)
        return self.norm(out + tokens)

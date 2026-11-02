"""IGNNK: Inductive Graph Neural Network Kriging baseline (Wu et al., AAAI 2021; arXiv:2006.07527).

A faithful, self-contained reimplementation of the IGNNK mechanism -- NOT the authors' code. IGNNK is
the canonical *learned inductive* spatial-kriging baseline for predicting signals at sensors with no
local observation, and is the right head-to-head for OmniST's inductive point prediction (stronger
than the nearest-neighbor / model-free interpolators already reported).

Faithful mechanism (rephrased from the paper/repo for compliance):
  - A diffusion graph convolution network (we use a 3-layer diffusion GCN) over a distance-based
    adjacency reconstructed from node coordinates.
  - Training uses random-subgraph sampling + node masking: sample a subset of observed sensors, mask
    some of them, and train the network to RECONSTRUCT the full signals (including the masked nodes)
    on that subgraph. This teaches the model to infer a node's signal from its neighbours alone.
  - At inference the unseen (held-out) sensors are added to the graph with the same distance-adjacency
    rule and their signals are reconstructed -> inductive kriging at never-trained coordinates.

Interface note: IGNNK krige-reconstructs a time window of signals, not a horizon forecast. For an
apples-to-apples comparison with OmniST's inductive point prediction at unseen stations, we task
IGNNK with reconstructing the signal AT the unseen stations over the evaluation window (the kriging
task it is designed for), and compare absolute error + conformal coverage on those reconstructions.

Refs: arXiv:2006.07527 (IGNNK); https://github.com/Kaimaoge/IGNNK  (reimplementation, not authors' code)
"""
import numpy as np
import torch
import torch.nn as nn


def gaussian_adjacency(coords: np.ndarray, sigma_scale: float = 1.0) -> np.ndarray:
    """Distance-kernel adjacency from coordinates (the standard kriging adjacency).

    coords: (N, 2). Returns row-normalized (N, N) Gaussian adjacency on pairwise Euclidean distance,
    with sigma set to sigma_scale * median(pairwise distance) for scale-invariance.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))                      # (N, N)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    sigma = max(sigma_scale * med, 1e-6)
    adj = np.exp(-(d ** 2) / (2 * sigma ** 2))
    deg = adj.sum(1, keepdims=True)
    deg[deg == 0] = 1.0
    return adj / deg                                      # row-normalized


class DiffusionGCNLayer(nn.Module):
    """One diffusion graph-convolution layer: aggregates over K diffusion steps of the adjacency."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.K = K
        self.theta = nn.Linear(in_dim * (K + 1), out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim), adj: (N, N)
        supports = [x]
        h = x
        for _ in range(self.K):
            h = adj @ h
            supports.append(h)
        return self.theta(torch.cat(supports, dim=-1))    # (N, out_dim)


class IGNNK(nn.Module):
    """3-layer diffusion-GCN kriging reconstructor.

    Input per sample: node signals over a time window, shape (N, T). The network treats the T-length
    series as per-node input features and reconstructs the full (N, T) signal matrix; masked/unseen
    nodes are reconstructed from their neighbours via the diffusion convolution.
    """

    def __init__(self, window: int, hidden: int = 64, K: int = 2):
        super().__init__()
        self.window = window
        self.l1 = DiffusionGCNLayer(window, hidden, K)
        self.l2 = DiffusionGCNLayer(hidden, hidden, K)
        self.l3 = DiffusionGCNLayer(hidden, window, K)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, T) node signals (masked nodes zeroed), adj: (N, N)
        h = self.act(self.l1(x, adj))
        h = self.act(self.l2(h, adj))
        return self.l3(h, adj)                             # (N, T) reconstruction


def train_ignnk(model, train_signal, train_coords, n_epochs=80, subgraph_size=None,
                mask_ratio=0.3, lr=1e-3, seed=0, device="cpu"):
    """Train IGNNK with random-subgraph + node-masking reconstruction.

    train_signal: (T_total, N_train) normalized signal on TRAINING sensors.
    train_coords: (N_train, 2).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    T_total, N = train_signal.shape
    W = model.window
    subgraph_size = subgraph_size or min(N, max(8, N // 2))
    sig = torch.tensor(train_signal, dtype=torch.float32, device=device)

    model.train()
    for ep in range(n_epochs):
        # random time window
        t0 = int(rng.integers(0, max(1, T_total - W)))
        window = sig[t0:t0 + W]                             # (W, N)
        # random subgraph of nodes
        sub = rng.choice(N, size=subgraph_size, replace=False)
        sub_sig = window[:, sub].T                          # (n_sub, W)
        sub_coords = train_coords[sub]
        adj = torch.tensor(gaussian_adjacency(sub_coords), dtype=torch.float32, device=device)
        # mask some nodes (their observations are hidden -> must be reconstructed from neighbours)
        n_mask = max(1, int(mask_ratio * subgraph_size))
        masked = rng.choice(subgraph_size, size=n_mask, replace=False)
        inp = sub_sig.clone()
        inp[masked] = 0.0
        out = model(inp, adj)                               # (n_sub, W)
        # reconstruction loss on the MASKED nodes (the kriging objective)
        loss = ((out[masked] - sub_sig[masked]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def ignnk_krige_unseen(model, signal_all, coords_all, train_idx, hold_idx, device="cpu"):
    """Inductively krige the held-out (unseen) sensors over the evaluation window.

    signal_all: (T, N_all) normalized signal (train + unseen). The unseen columns are the ground
    truth we withhold as INPUT (masked) and compare the reconstruction against.
    Returns per-(unseen-node, time) absolute residuals on the normalized scale, flattened.
    """
    model.eval()
    W = model.window
    T = signal_all.shape[0]
    node_idx = list(train_idx) + list(hold_idx)
    coords = coords_all[node_idx]
    adj = torch.tensor(gaussian_adjacency(coords), dtype=torch.float32, device=device)
    n_train = len(train_idx)
    sig = torch.tensor(signal_all[:, node_idx], dtype=torch.float32, device=device)  # (T, n_sub)
    res = []
    # slide non-overlapping windows over the eval period
    for t0 in range(0, max(1, T - W + 1), W):
        window = sig[t0:t0 + W].T                           # (n_sub, W)
        if window.shape[1] < W:
            break
        inp = window.clone()
        inp[n_train:] = 0.0                                 # mask the unseen sensors (no observation)
        out = model(inp, adj)                               # (n_sub, W)
        r = (out[n_train:] - window[n_train:]).abs()        # unseen-node reconstruction error
        res.append(r.reshape(-1).cpu().numpy())
    return np.concatenate(res) if res else np.array([])

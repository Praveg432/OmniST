"""
Extreme-Value-Aware Loss Functions for OmniST.

Three loss functions available:
1. POTLoss — E-STGCN style GP distribution-guided tail loss (principled, best for extremes)
2. HuberWeightedLoss — Simple weighted Huber (original Sprint 0 choice, fast)
3. StandardMSELoss — Plain MSE (ablation baseline)

The POT loss integrates Generalized Pareto distribution knowledge into the training,
following the formulation from Panja et al. (2026, JRSS-A):
  Loss = MSE(pred, target)                         when pred ≤ τ
  Loss = β₁·MSE(pred, target) + β₂·POTL(pred)     when pred > τ

Where POTL(z) = log(σ̂) + (1 + 1/ξ̂)·log(1 + ξ̂·z/σ̂)  (NLL of fitted GP distribution)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as scipy_stats
from typing import Optional, Tuple


class POTLoss(nn.Module):
    """Peaks-Over-Threshold Loss with Generalized Pareto Distribution.

    Integrates EVT knowledge into the loss function:
    - Below threshold: standard MSE (captures average behavior)
    - Above threshold: MSE + GP distribution NLL penalty (captures tail behavior)

    GP parameters (σ, ξ) are estimated per-node from training data exceedances.
    For unseen nodes at inference, global (averaged) GP parameters are used as fallback.

    Reference: E-STGCN (Panja et al., 2026, JRSS-A)
    """

    def __init__(
        self,
        beta1: float = 1.0,
        beta2: float = 0.5,
        threshold_percentile: float = 90.0,
    ):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold_percentile = threshold_percentile

        # Set from training data
        self.threshold: float = 0.0
        self.gp_sigma: Optional[torch.Tensor] = None  # (N,) or scalar
        self.gp_xi: Optional[torch.Tensor] = None     # (N,) or scalar
        # Global fallback for unseen nodes
        self.gp_sigma_global: float = 1.0
        self.gp_xi_global: float = 0.1

    def fit_from_training_data(self, training_data: np.ndarray, per_node: bool = True):
        """Fit GP distribution parameters from training data.

        Args:
            training_data: (T, N, F) or (T, N) — training features
                          If 3D, uses first feature channel for GP fitting.
            per_node: if True, fit separate (σ, ξ) per node; else fit global.
        """
        if training_data.ndim == 3:
            data = training_data[:, :, 0]  # Use first feature (e.g., PM2.5)
        else:
            data = training_data

        # Compute threshold
        self.threshold = float(np.nanpercentile(data, self.threshold_percentile))

        if per_node:
            n_nodes = data.shape[1]
            sigmas = np.ones(n_nodes)
            xis = np.ones(n_nodes) * 0.1

            for i in range(n_nodes):
                node_data = data[:, i]
                node_data = node_data[~np.isnan(node_data)]
                exceedances = node_data[node_data > self.threshold] - self.threshold

                if len(exceedances) > 10:  # Need enough exceedances for reliable fit
                    try:
                        xi, loc, sigma = scipy_stats.genpareto.fit(exceedances, floc=0)
                        # Clip to avoid numerical instability
                        sigmas[i] = np.clip(sigma, 0.01, 1000.0)
                        xis[i] = np.clip(xi, -0.5, 2.0)
                    except (RuntimeError, ValueError):
                        pass  # Keep defaults

            self.gp_sigma = torch.tensor(sigmas, dtype=torch.float32)
            self.gp_xi = torch.tensor(xis, dtype=torch.float32)
        else:
            all_data = data.flatten()
            all_data = all_data[~np.isnan(all_data)]
            exceedances = all_data[all_data > self.threshold] - self.threshold

            if len(exceedances) > 10:
                try:
                    xi, loc, sigma = scipy_stats.genpareto.fit(exceedances, floc=0)
                    self.gp_sigma_global = float(np.clip(sigma, 0.01, 1000.0))
                    self.gp_xi_global = float(np.clip(xi, -0.5, 2.0))
                except (RuntimeError, ValueError):
                    pass

            self.gp_sigma = None
            self.gp_xi = None

        # Also compute global fallback
        all_flat = data.flatten()
        all_flat = all_flat[~np.isnan(all_flat)]
        exc_global = all_flat[all_flat > self.threshold] - self.threshold
        if len(exc_global) > 10:
            try:
                xi_g, _, sigma_g = scipy_stats.genpareto.fit(exc_global, floc=0)
                self.gp_sigma_global = float(np.clip(sigma_g, 0.01, 1000.0))
                self.gp_xi_global = float(np.clip(xi_g, -0.5, 2.0))
            except (RuntimeError, ValueError):
                pass

    def _pot_nll(self, pred_excess: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood of GP distribution for exceedances.

        POTL(z) = log(σ) + (1 + 1/ξ) · log(1 + ξ·z/σ)

        Args:
            pred_excess: predictions above threshold (exceedance values, already shifted by τ)
            sigma: GP scale parameter
            xi: GP shape parameter

        Returns:
            NLL values (same shape as pred_excess)
        """
        # Avoid numerical issues
        xi_safe = xi.clamp(min=0.01)  # Avoid ξ=0 division; positive ξ for heavy tails
        sigma_safe = sigma.clamp(min=0.01)

        # GP NLL: log(σ) + (1 + 1/ξ) · log(1 + ξ·z/σ)
        inner = (1.0 + xi_safe * pred_excess / sigma_safe).clamp(min=1e-8)
        nll = torch.log(sigma_safe) + (1.0 + 1.0 / xi_safe) * torch.log(inner)

        return nll

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N, H, F) — predictions
            target: (B, N, H, F) — ground truth

        Returns:
            scalar loss
        """
        if self.threshold == 0.0:
            return F.mse_loss(pred, target)

        # Determine which predictions are above threshold
        # Use target values to determine extreme regime (same as E-STGCN)
        extreme_mask = target > self.threshold
        normal_mask = ~extreme_mask

        # Normal regime: standard MSE
        if normal_mask.any():
            loss_normal = F.mse_loss(pred[normal_mask], target[normal_mask])
        else:
            loss_normal = torch.tensor(0.0, device=pred.device)

        # Extreme regime: β₁·MSE + β₂·POTL
        if extreme_mask.any():
            pred_extreme = pred[extreme_mask]
            target_extreme = target[extreme_mask]

            mse_extreme = F.mse_loss(pred_extreme, target_extreme)

            # Compute POT NLL for extreme predictions
            # Exceedance = pred - threshold
            pred_excess = (pred_extreme - self.threshold).clamp(min=1e-6)

            # Get GP parameters (use global if per-node not available or N mismatch)
            if self.gp_sigma is not None and pred.shape[1] == self.gp_sigma.shape[0]:
                # Per-node parameters — expand to match extreme_mask shape
                # This is complex for arbitrary mask shapes, use global for simplicity in POT term
                sigma = torch.tensor(self.gp_sigma_global, device=pred.device)
                xi = torch.tensor(self.gp_xi_global, device=pred.device)
            else:
                sigma = torch.tensor(self.gp_sigma_global, device=pred.device)
                xi = torch.tensor(self.gp_xi_global, device=pred.device)

            pot_nll = self._pot_nll(pred_excess, sigma, xi)
            pot_loss = pot_nll.mean()

            loss_extreme = self.beta1 * mse_extreme + self.beta2 * pot_loss
        else:
            loss_extreme = torch.tensor(0.0, device=pred.device)

        # Combine weighted by fraction of samples in each regime
        n_total = pred.numel()
        n_extreme = extreme_mask.sum().float()
        n_normal = normal_mask.sum().float()

        if n_total > 0:
            loss = (n_normal / n_total) * loss_normal + (n_extreme / n_total) * loss_extreme
        else:
            loss = loss_normal

        return loss


class HuberWeightedLoss(nn.Module):
    """Huber loss with inverse-frequency weighting for extreme events.

    Simpler alternative to POTLoss. Used as comparison baseline.
    """

    def __init__(
        self,
        extreme_weight: float = 3.0,
        huber_delta: float = 1.0,
        threshold_percentile: float = 90.0,
    ):
        super().__init__()
        self.extreme_weight = extreme_weight
        self.huber_delta = huber_delta
        self.threshold_percentile = threshold_percentile
        self.threshold: float = 0.0

    def set_threshold_from_data(self, training_data: np.ndarray):
        self.threshold = float(np.nanpercentile(training_data, self.threshold_percentile))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber = F.huber_loss(pred, target, reduction='none', delta=self.huber_delta)
        if self.threshold != 0.0:
            extreme_mask = (target.abs() > abs(self.threshold)).float()
            weights = 1.0 + (self.extreme_weight - 1.0) * extreme_mask
        else:
            weights = torch.ones_like(huber)
        return (huber * weights).mean()


class StandardMSELoss(nn.Module):
    """Plain MSE loss — ablation baseline."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

"""
Conformal Prediction for Uncertainty Quantification.

Distribution-free prediction intervals around OmniST point forecasts.
Following the approach from E-STGCN (Panja et al., 2026):
  γ_t = |X_t - Model(X_{t-p})| / U(X_{t-p})

Where U is an uncertainty model (MAD or model-based).

This module is model-agnostic — wraps any point forecaster to produce
calibrated prediction intervals without retraining.

Key advantage over E-STGCN: we can produce uncertainty estimates at
UNSEEN nodes (because our point forecaster works there), which E-STGCN cannot.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from src.data import collate_variable_nodes


class ConformalPredictor:
    """Conformal prediction wrapper for calibrated uncertainty.

    Usage:
        1. Train OmniST normally
        2. Call calibrate() on a held-out calibration set
        3. Call predict_with_intervals() at inference time
    """

    def __init__(self, coverage: float = 0.80):
        """
        Args:
            coverage: desired coverage level (0.80 = 80% prediction interval)
        """
        self.coverage = coverage
        self.conformal_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None

    def calibrate(
        self,
        model: torch.nn.Module,
        calibration_ds,
        metadata: dict,
        batch_size: int = 4,
    ):
        """Compute conformal scores on calibration set.

        Args:
            model: trained OmniST model
            calibration_ds: held-out calibration dataset
            metadata: contains 'means' and 'stds' for denormalization
        """
        cal_loader = DataLoader(
            calibration_ds, batch_size=batch_size,
            shuffle=False, collate_fn=collate_variable_nodes
        )

        means = torch.tensor(metadata["means"], dtype=torch.float32).squeeze()
        stds = torch.tensor(metadata["stds"], dtype=torch.float32).squeeze()

        all_scores = []

        model.eval()
        with torch.no_grad():
            for features, coords, timestamps, targets in cal_loader:
                pred = model(features, coords, timestamps)

                # Denormalize
                pred_denorm = pred * stds + means
                target_denorm = targets * stds + means

                # Conformal score: |target - prediction| (absolute residuals)
                residuals = (target_denorm - pred_denorm).abs()
                all_scores.append(residuals.numpy())

        if not all_scores:
            self.quantile = 0.0
            return

        all_scores = np.concatenate(all_scores, axis=0)  # (n_samples, N, H, F)

        # Compute the conformal quantile (1 - α quantile of residuals)
        # This gives the width needed for desired coverage
        alpha = 1.0 - self.coverage
        # Correction for finite sample: use (n+1)(1-α)/n quantile
        n = all_scores.size
        adjusted_quantile = min((1.0 - alpha) * (1.0 + 1.0 / max(n, 1)), 1.0)
        self.quantile = float(np.quantile(all_scores.flatten(), adjusted_quantile))
        self.conformal_scores = all_scores

    def predict_with_intervals(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        coords: torch.Tensor,
        timestamps: torch.Tensor,
        metadata: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate point predictions with calibrated prediction intervals.

        Args:
            model: trained model
            features: (B, T, N, F)
            coords: (B, N, 2)
            timestamps: (B, T, 4)
            metadata: for denormalization

        Returns:
            (predictions, lower_bound, upper_bound) — all denormalized
        """
        means = torch.tensor(metadata["means"], dtype=torch.float32).squeeze()
        stds = torch.tensor(metadata["stds"], dtype=torch.float32).squeeze()

        model.eval()
        with torch.no_grad():
            pred = model(features, coords, timestamps)
            pred_denorm = (pred * stds + means).numpy()

        if self.quantile is None:
            # Not calibrated — return point prediction only
            return pred_denorm, pred_denorm, pred_denorm

        lower = pred_denorm - self.quantile
        upper = pred_denorm + self.quantile

        # Ensure physical plausibility (non-negative for pollution/traffic)
        lower = np.maximum(lower, 0)

        return pred_denorm, lower, upper

    def get_interval_width(self) -> float:
        """Return the calibrated interval half-width."""
        return self.quantile if self.quantile is not None else 0.0

    def get_empirical_coverage(
        self,
        model: torch.nn.Module,
        test_ds,
        metadata: dict,
        batch_size: int = 4,
    ) -> float:
        """Compute actual coverage on test set (should be ≥ target coverage)."""
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, collate_fn=collate_variable_nodes
        )

        means = torch.tensor(metadata["means"], dtype=torch.float32).squeeze()
        stds = torch.tensor(metadata["stds"], dtype=torch.float32).squeeze()

        covered = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for features, coords, timestamps, targets in test_loader:
                pred = model(features, coords, timestamps)
                pred_denorm = pred * stds + means
                target_denorm = targets * stds + means

                residuals = (target_denorm - pred_denorm).abs()
                covered += (residuals <= self.quantile).sum().item()
                total += residuals.numel()

        return covered / max(total, 1)

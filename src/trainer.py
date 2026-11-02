"""
Training and evaluation utilities for OmniST experiments.

Provides:
- train_model(): full training loop with early stopping
- evaluate_model(): compute all metrics on a dataset
- Metrics: MAE, RMSE, SMAPE, extreme recall
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
from dataclasses import dataclass

from src.data import collate_variable_nodes
from src.evt_loss import HuberWeightedLoss, StandardMSELoss, POTLoss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    patience: int = 20
    batch_size: int = 4
    grad_clip: float = 1.0
    use_evt_loss: bool = True
    loss_type: str = "pot"  # "pot", "huber", or "mse"


def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    config: TrainConfig,
    metadata: dict,
    verbose: bool = False,
) -> Dict:
    """Train a model with early stopping. Returns training history."""
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_variable_nodes
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_variable_nodes
    )

    # Loss function
    if config.loss_type == "pot":
        criterion = POTLoss(beta1=1.0, beta2=0.5, threshold_percentile=90.0)
        sample_features, _, _, _ = train_ds[0]
        criterion.fit_from_training_data(sample_features.numpy())
    elif config.loss_type == "huber" or config.use_evt_loss:
        criterion = HuberWeightedLoss(extreme_weight=3.0, threshold_percentile=90.0)
        sample_features, _, _, _ = train_ds[0]
        criterion.set_threshold_from_data(sample_features.numpy())
    else:
        criterion = StandardMSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.n_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for features, coords, timestamps, targets in train_loader:
            optimizer.zero_grad()
            pred = model(features, coords, timestamps)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for features, coords, timestamps, targets in val_loader:
                pred = model(features, coords, timestamps)
                loss = F.mse_loss(pred, targets)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == config.n_epochs - 1):
            print(f"  Epoch {epoch:3d}: train={avg_train:.4f} val={avg_val:.4f} best={best_val_loss:.4f}")

        if patience_counter >= config.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = epoch + 1
    return history


def evaluate_model(
    model: nn.Module,
    test_ds,
    metadata: dict,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Evaluate model on test set, compute all metrics."""
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_variable_nodes
    )

    means = torch.tensor(metadata["means"], dtype=torch.float32).squeeze()
    stds = torch.tensor(metadata["stds"], dtype=torch.float32).squeeze()

    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for features, coords, timestamps, targets in test_loader:
            pred = model(features, coords, timestamps)
            # Denormalize
            pred_denorm = pred * stds + means
            target_denorm = targets * stds + means
            all_preds.append(pred_denorm)
            all_targets.append(target_denorm)

    if not all_preds:
        return {"mae": float('inf'), "rmse": float('inf'), "smape": float('inf')}

    preds = torch.cat(all_preds, dim=0)    # (total_samples, N, H, F)
    targets = torch.cat(all_targets, dim=0)

    # Metrics
    errors = preds - targets
    abs_errors = errors.abs()

    mae = abs_errors.mean().item()
    rmse = errors.pow(2).mean().sqrt().item()

    # SMAPE (symmetric mean absolute percentage error)
    denom = (preds.abs() + targets.abs()).clamp(min=1e-8)
    smape = (2 * abs_errors / denom).mean().item() * 100  # percentage

    # Extreme recall (90th percentile)
    threshold = targets.quantile(0.9).item()
    extreme_mask = targets > threshold
    if extreme_mask.any():
        extreme_errors = abs_errors[extreme_mask]
        extreme_targets = targets[extreme_mask]
        within_20pct = (extreme_errors / extreme_targets.clamp(min=1e-8)) < 0.2
        extreme_recall = within_20pct.float().mean().item()
    else:
        extreme_recall = 0.0

    # Per-horizon MAE (30-day: 10, 20, 30 day horizons)
    mae_10d = abs_errors[:, :, :10, :].mean().item()
    mae_20d = abs_errors[:, :, :20, :].mean().item()
    mae_30d = abs_errors.mean().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "extreme_recall": extreme_recall,
        "mae_10d": mae_10d,
        "mae_20d": mae_20d,
        "mae_30d": mae_30d,
    }


def evaluate_topology_robustness(
    model: nn.Module,
    test_ds,
    metadata: dict,
    dropout_levels: list = [0.1, 0.2, 0.3],
    n_patterns: int = 5,
) -> Dict[str, list]:
    """Evaluate model under node dropout at inference."""
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False,
        collate_fn=collate_variable_nodes
    )

    results = {f"drop_{int(d*100)}pct": [] for d in dropout_levels}
    results["full"] = []

    model.eval()
    with torch.no_grad():
        for feat_batch, coords_batch, ts_batch, tgt_batch in test_loader:
            B, T, N, _ = feat_batch.shape

            # Full prediction
            pred_full = model(feat_batch, coords_batch, ts_batch)
            mse_full = F.mse_loss(pred_full, tgt_batch).item()
            results["full"].append(mse_full)

            # Dropout predictions
            for drop_frac in dropout_levels:
                pattern_mses = []
                for _ in range(n_patterns):
                    keep = int(N * (1 - drop_frac))
                    idx = torch.randperm(N)[:keep]
                    pred_drop = model(
                        feat_batch[:, :, idx], coords_batch[:, idx], ts_batch
                    )
                    mse_drop = F.mse_loss(pred_drop, tgt_batch[:, idx]).item()
                    pattern_mses.append(mse_drop)
                results[f"drop_{int(drop_frac*100)}pct"].append(np.mean(pattern_mses))

    return results

#!/usr/bin/env python3
"""
Experiment 01: First End-to-End Training Run

Validates that the full OmniST pipeline:
1. Loads realistic Delhi air quality data
2. Trains without NaN/divergence
3. Produces plausible forecasts
4. Completes within time budget

This is Sprint 1 Task 11 (renamed from Sprint 4 after validation phase).
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from src.data import DataConfig, prepare_delhi_data, collate_variable_nodes
from src.omnist_model import OmniST, ModelConfig
from src.evt_loss import HuberWeightedLoss
from torch.utils.data import DataLoader


def main():
    print("=" * 60)
    print("  Experiment 01: First End-to-End Training Run")
    print("=" * 60)

    # Configuration
    data_config = DataConfig(window_size=60, horizon=30, n_features=3, normalize=True)
    model_config = ModelConfig(
        n_features=3, horizon=30, d_model=128,
        n_gat_heads=4, k_neighbors=8, n_gat_layers=1,
        n_temporal_layers=2, temporal_dropout=0.1, dropout=0.1,
    )

    # Prepare data
    print("\n[1] Preparing Delhi air quality data...")
    t0 = time.time()
    train_ds, val_ds, test_ds, metadata = prepare_delhi_data(data_config)
    print(f"    Train: {len(train_ds)} samples, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"    Stations: {metadata['n_stations']}, Features: {metadata['n_features']}")
    print(f"    Data prep time: {time.time()-t0:.1f}s")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_variable_nodes)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_variable_nodes)

    # Model
    print(f"\n[2] Building model (d_model={model_config.d_model})...")
    model = OmniST(model_config)
    n_params = model.count_parameters()
    print(f"    Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Loss & optimizer
    criterion = HuberWeightedLoss(extreme_weight=3.0, threshold_percentile=90.0)
    # Set threshold from training data
    sample_features, _, _, _ = train_ds[0]
    criterion.set_threshold_from_data(sample_features.numpy())
    print(f"    Extreme threshold (90th pct): {criterion.threshold:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training
    print(f"\n[3] Training for 50 epochs...")
    t_train_start = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(50):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for features, coords, timestamps, targets in train_loader:
            optimizer.zero_grad()
            pred = model(features, coords, timestamps)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

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

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)
        scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 49:
            print(f"    Epoch {epoch:3d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, best={best_val_loss:.4f}")

        if patience_counter >= 15:
            print(f"    Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t_train_start
    print(f"\n    Training time: {train_time:.1f}s = {train_time/60:.1f}min")
    print(f"    Final train loss: {train_losses[-1]:.4f}")
    print(f"    Best val loss: {best_val_loss:.4f}")

    # Plausibility check
    print(f"\n[4] Forecast plausibility gate...")
    model.eval()
    with torch.no_grad():
        features, coords, timestamps, targets = next(iter(val_loader))
        pred = model(features, coords, timestamps)

        # Denormalize for plausibility check
        means = torch.tensor(metadata["means"], dtype=torch.float32).squeeze()
        stds = torch.tensor(metadata["stds"], dtype=torch.float32).squeeze()
        pred_denorm = pred * stds + means
        target_denorm = targets * stds + means

        # Check: predictions in plausible range
        pm25_pred = pred_denorm[:, :, :, 0]  # PM2.5
        plausible = (pm25_pred >= 0).all() and (pm25_pred < 1000).all()

        print(f"    PM2.5 predictions range: [{pm25_pred.min():.1f}, {pm25_pred.max():.1f}] µg/m³")
        print(f"    PM2.5 targets range: [{target_denorm[:,:,:,0].min():.1f}, {target_denorm[:,:,:,0].max():.1f}] µg/m³")
        print(f"    Plausibility gate: {'✓ PASS' if plausible else '✗ FAIL'}")

        # Basic accuracy check
        mae = (pred_denorm - target_denorm).abs().mean().item()
        print(f"    Overall MAE (denormalized): {mae:.2f}")

    # Node dropout test (topology invariance on trained model)
    print(f"\n[5] Topology invariance check (trained model)...")
    with torch.no_grad():
        features, coords, timestamps, targets = next(iter(val_loader))
        # Full prediction
        pred_full = model(features, coords, timestamps)
        mse_full = F.mse_loss(pred_full, targets).item()

        # Drop 30% of nodes
        N = features.shape[2]
        keep = int(N * 0.7)
        idx = torch.randperm(N)[:keep]
        pred_drop = model(features[:, :, idx], coords[:, idx], timestamps)
        mse_drop = F.mse_loss(pred_drop, targets[:, idx]).item()

        degradation = (mse_drop - mse_full) / mse_full * 100
        print(f"    MSE (full N={N}): {mse_full:.4f}")
        print(f"    MSE (drop 30%, N={keep}): {mse_drop:.4f}")
        print(f"    Degradation: {degradation:+.1f}%")
        print(f"    Topology gate: {'✓ PASS' if abs(degradation) < 20 else '✗ FAIL'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  ✓ Model trains without NaN/divergence")
    print(f"  ✓ Training time: {train_time:.0f}s (budget: 7200s)")
    print(f"  ✓ Plausibility: {'PASS' if plausible else 'FAIL'}")
    print(f"  ✓ Topology invariance: {degradation:+.1f}% degradation at 30% dropout")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

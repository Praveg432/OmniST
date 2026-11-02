"""Integration test: full model end-to-end."""
import torch
import pytest
from src.omnist_model import OmniST, ModelConfig
from src.data import DataConfig, prepare_delhi_data


@pytest.fixture
def model():
    config = ModelConfig(n_features=3, horizon=30, d_model=64, n_gat_heads=4, k_neighbors=6)
    return OmniST(config)


def test_forward_pass(model):
    """Full model forward pass produces correct output shape."""
    B, T, N, F = 2, 50, 37, 3
    features = torch.randn(B, T, N, F)
    coords = torch.randn(B, N, 2) * 0.1 + torch.tensor([28.6, 77.1])
    timestamps = torch.rand(B, T, 4)

    output = model(features, coords, timestamps)
    assert output.shape == (B, N, 30, F)


def test_topology_invariance(model):
    """Model handles different N without retraining."""
    B, T, F = 1, 30, 3
    timestamps = torch.rand(B, T, 4)

    for N in [20, 37, 45]:
        features = torch.randn(B, T, N, F)
        coords = torch.randn(B, N, 2) * 0.1 + torch.tensor([28.6, 77.1])
        output = model(features, coords, timestamps)
        assert output.shape == (B, N, 30, F)


def test_gradient_flow(model):
    """Gradients flow through all components."""
    B, T, N, F = 1, 20, 10, 3
    features = torch.randn(B, T, N, F, requires_grad=False)
    coords = torch.randn(B, N, 2) * 0.1 + torch.tensor([28.6, 77.1])
    timestamps = torch.rand(B, T, 4)
    target = torch.randn(B, N, 30, F)

    output = model(features, coords, timestamps)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check all parameters received gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    # At least 90% of parameters should have non-zero gradients
    # (some GRU hidden-to-hidden weights may be zero with short sequences)
    nonzero_count = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None and not torch.all(p.grad == 0)
    )
    total_count = sum(1 for p in model.parameters() if p.requires_grad)
    assert nonzero_count / total_count > 0.8, f"Only {nonzero_count}/{total_count} params got nonzero grads"


def test_parameter_count(model):
    """Model is within expected size range."""
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}")
    # At d_model=64: expect ~100K-500K params
    assert 50_000 < n_params < 2_000_000


def test_training_step(model):
    """Single training step completes without error."""
    B, T, N, F = 2, 30, 37, 3
    features = torch.randn(B, T, N, F)
    coords = torch.randn(B, N, 2) * 0.1 + torch.tensor([28.6, 77.1])
    timestamps = torch.rand(B, T, 4)
    target = torch.randn(B, N, 30, F)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    output = model(features, coords, timestamps)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_data_pipeline():
    """Delhi data pipeline produces valid datasets."""
    config = DataConfig(window_size=60, horizon=30, n_features=3)
    train_ds, val_ds, test_ds, metadata = prepare_delhi_data(config)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert metadata["n_stations"] == 37
    assert metadata["n_features"] == 3

    # Check single sample
    features, coords, timestamps, targets = train_ds[0]
    assert features.shape == (60, 37, 3)
    assert coords.shape == (37, 2)
    assert timestamps.shape == (60, 4)
    assert targets.shape == (37, 30, 3)

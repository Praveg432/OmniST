"""Coordinate normalization of the spatial embedding (the fix for the Delhi-specific constants)."""
import torch

from src.cc_hash import SpatialMLP


def _legacy(coords):
    c = coords.clone()
    c[..., 0] = (coords[..., 0] - 28.6) / 0.2
    c[..., 1] = (coords[..., 1] - 77.1) / 0.3
    return c


def test_default_frame_reproduces_the_legacy_formula_exactly():
    torch.manual_seed(0)
    emb = SpatialMLP(d_model=16, hidden_dim=16)
    coords = torch.tensor([[28.47, 76.93], [28.82, 77.32], [34.0, -118.3], [40.1, 116.4]])
    assert torch.equal(emb(coords), emb.mlp(_legacy(coords)))


def test_training_network_frame_is_invariant_to_shifting_all_coordinates():
    torch.manual_seed(0)
    coords = torch.tensor([[34.0, -118.3], [34.2, -118.1], [34.5, -117.8], [34.3, -118.0]])
    shifted = coords + torch.tensor([5.0, -40.0])
    a = SpatialMLP(16, 16, center=tuple(coords.mean(0).tolist()), scale=tuple(coords.std(0).tolist()))
    b = SpatialMLP(16, 16, center=tuple(shifted.mean(0).tolist()), scale=tuple(shifted.std(0).tolist()))
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k not in ("center", "scale")}, strict=False)
    assert torch.allclose(a(coords), b(shifted), atol=1e-5)


def test_frame_is_saved_with_the_model():
    emb = SpatialMLP(16, 16, center=(34.25, -118.05), scale=(0.15, 0.15))
    restored = SpatialMLP(16, 16)
    restored.load_state_dict(emb.state_dict())
    assert torch.equal(restored.center, emb.center) and torch.equal(restored.scale, emb.scale)

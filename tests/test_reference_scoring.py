"""Tests for reference-restricted scoring (Lemma 1 in the paper).

Claim under test: when neighbours are drawn only from a reference network R, a new sensor's output
depends only on (its own data, its k nearest reference sensors). Scoring many new sensors in one
joint pass is therefore identical to scoring each one alone, added to R. Architecture property, so it
is tested on randomly initialised weights.
"""
import numpy as np
import torch

from src.inductive_gat import haversine_distance_batch
from src.omnist_model import ModelConfig, OmniST


def _model(knn_mode="exact", **kw):
    torch.manual_seed(0)
    cfg = ModelConfig(n_features=1, horizon=3, d_model=16, spatial_hidden=16, n_gat_heads=4,
                      k_neighbors=5, knn_mode=knn_mode, **kw)
    return OmniST(cfg).eval()


def _data(n, T=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(1, T, n, 1, generator=g)
    coords = torch.stack([28.4 + 0.4 * torch.rand(1, n, generator=g),
                          76.8 + 0.6 * torch.rand(1, n, generator=g)], dim=-1)
    ts = torch.rand(1, T, 4, generator=g)
    return feats, coords, ts


def _pick(feats, coords, idx):
    return feats[:, :, idx], coords[:, idx]


def _joint_vs_isolated(model, n_ref, n_new, atol=1e-5):
    n = n_ref + n_new
    feats, coords, ts = _data(n)
    ref = list(range(n_ref))
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, ref] = True
    with torch.no_grad():
        joint = model(feats, coords, ts, ref_mask=mask)
        for i in range(n_ref, n):
            f, c = _pick(feats, coords, ref + [i])
            alone = model(f, c, ts)                    # ordinary pass on R plus this one sensor
            assert torch.allclose(joint[0, i], alone[0, -1], atol=atol), (
                i, float((joint[0, i] - alone[0, -1]).abs().max()))
        # non-interference the other way: reference outputs do not depend on the new sensors
        f, c = _pick(feats, coords, ref)
        ref_only = model(f, c, ts)
        assert torch.allclose(joint[0, ref], ref_only[0], atol=atol)


def test_default_path_unchanged_by_all_true_mask():
    m = _model()
    feats, coords, ts = _data(12)
    mask = torch.ones(1, 12, dtype=torch.bool)
    with torch.no_grad():
        a = m(feats, coords, ts)
        b = m(feats, coords, ts, ref_mask=mask)
    assert torch.equal(a, b)


def test_joint_scoring_equals_isolated_exact_knn():
    _joint_vs_isolated(_model("exact"), n_ref=14, n_new=4)


def test_joint_scoring_equals_isolated_scalable_knn():
    _joint_vs_isolated(_model("scalable"), n_ref=80, n_new=3)


def test_far_change_invariance_for_a_new_sensor():
    """Adding or removing reference sensors outside a new sensor's k-NN set leaves its output unchanged."""
    m = _model()
    n_ref = 20
    feats, coords, ts = _data(n_ref + 1)
    new = n_ref
    ref = list(range(n_ref))
    # use the model's own neighbour metric (Haversine), not Euclidean degrees
    d = haversine_distance_batch(coords)[0, new, :n_ref]
    order = torch.argsort(d).tolist()
    knn = set(order[:5])
    far = [j for j in ref if j not in knn][:6]
    kept = [j for j in ref if j not in far]
    with torch.no_grad():
        mask_full = torch.zeros(1, n_ref + 1, dtype=torch.bool); mask_full[0, ref] = True
        out_full = m(feats, coords, ts, ref_mask=mask_full)[0, new]
        f, c = _pick(feats, coords, kept + [new])
        mask_kept = torch.zeros(1, len(kept) + 1, dtype=torch.bool); mask_kept[0, :len(kept)] = True
        out_kept = m(f, c, ts, ref_mask=mask_kept)[0, -1]
        # control: removing a true neighbour DOES change the output
        near = order[0]
        kept2 = [j for j in ref if j != near]
        f2, c2 = _pick(feats, coords, kept2 + [new])
        mask2 = torch.zeros(1, len(kept2) + 1, dtype=torch.bool); mask2[0, :len(kept2)] = True
        out_near = m(f2, c2, ts, ref_mask=mask2)[0, -1]
    assert torch.allclose(out_full, out_kept, atol=1e-6)
    assert not torch.allclose(out_full, out_near, atol=1e-4)


def test_coordinate_free_control_ignores_coordinates():
    m = _model(use_coords=False)
    m.gat_layers = torch.nn.ModuleList()                 # channel-independent control: no attention
    feats, coords, ts = _data(10)
    with torch.no_grad():
        a = m(feats, coords, ts)
        b = m(feats, coords + 5.0, ts)
    assert torch.allclose(a, b, atol=1e-6)

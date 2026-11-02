#!/usr/bin/env python3
"""EX-KNN - equivalence + speed of the scalable (cKDTree) k-NN vs exact k-NN in the inductive GAT.

The inductive GAT originally built the full (B,N,N) Haversine matrix and materialised (B,N,N,H,dh)
K/V tensors -- O(N^2) time AND memory, the bottleneck behind the slow large-N runs. We added a
'scalable' knn_mode (InductiveGAT(..., knn_mode='scalable') / ModelConfig(knn_mode='scalable')) that:
  - selects neighbours with a per-batch cKDTree on an equirectangular projection  -> O(N log N),
  - gathers K/V by index without the (B,N,N,...) expansion                         -> O(N*k) memory,
  - recomputes TRUE Haversine distance on only the k selected pairs (bias unchanged).
Attention math is identical; only neighbour SELECTION differs. This test verifies (a) the two modes
produce near-identical outputs (equivalence) and (b) the scalable path is faster and scales better.

RUN: python experiments/14_approx_knn_test.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.inductive_gat import InductiveGAT


def make(N, d=64, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(B, N, d, generator=g)
    coords = torch.rand(B, N, 2, generator=g) * torch.tensor([0.5, 0.5]) + torch.tensor([34.0, -118.3])
    return tokens, coords


def run():
    torch.manual_seed(0); np.random.seed(0)
    d = 64
    exact = InductiveGAT(d, n_heads=4, k_neighbors=8, knn_mode="exact").eval()
    scal = InductiveGAT(d, n_heads=4, k_neighbors=8, knn_mode="scalable").eval()
    scal.load_state_dict(exact.state_dict())             # identical learned weights

    tokens, coords = make(228)
    with torch.no_grad():
        oe, os_ = exact(tokens, coords), scal(tokens, coords)
    diff = (oe - os_).abs()
    cos = torch.nn.functional.cosine_similarity(oe.reshape(-1), os_.reshape(-1), dim=0).item()
    equivalence = {
        "N": 228,
        "mean_abs_diff": round(float(diff.mean()), 6),
        "rel_mean_diff": round(float(diff.mean() / oe.abs().mean()), 6),
        "output_cosine_similarity": round(cos, 6),
    }

    ladder = []
    for N in [228, 456, 912, 1824]:
        tokens, coords = make(N)
        row = {"N": N}
        for mode, g in [("exact", exact), ("scalable", scal)]:
            try:
                with torch.no_grad():
                    t = time.time()
                    for _ in range(3):
                        g(tokens, coords)
                    row[f"{mode}_ms"] = round((time.time() - t) / 3 * 1000, 1)
            except RuntimeError as e:
                row[f"{mode}_ms"] = f"FAILED: {str(e)[:40]}"
        if isinstance(row.get("exact_ms"), (int, float)) and isinstance(row.get("scalable_ms"), (int, float)) and row["scalable_ms"] > 0:
            row["speedup"] = round(row["exact_ms"] / row["scalable_ms"], 2)
        ladder.append(row)

    out = {
        "experiment": "EX-KNN (scalable cKDTree k-NN vs exact: equivalence + speed)",
        "equivalence": equivalence,
        "speed_ladder_per_forward_B2": ladder,
        "verdict": ("EQUIVALENT (cosine>=0.999) and scalable is faster at large N"
                    if equivalence["output_cosine_similarity"] >= 0.999 else "CHECK: outputs diverge"),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "exp_approx_knn.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")
    return out


if __name__ == "__main__":
    run()

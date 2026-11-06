# OmniST: Topology-Invariant Spatiotemporal Forecasting with Guaranteed-Coverage Uncertainty at Unseen Sensor Locations

Reference implementation and full experiment suite for the OmniST paper.

**Author:** Praveg Vashishtha  
**Affiliation:** Department of Computer Science and Engineering, IIT Patna | Senior SDE, SBS India  
**Contact:** praveg_pa2503mth259@iitp.ac.in | omvashishtha432@gmail.com  
**LinkedIn:** [pravegvashishtha](https://www.linkedin.com/in/pravegvashishtha/)  
**Paper:** [arXiv:2611.04256](https://arxiv.org/abs/2611.04256)  
**Code:** [github.com/Praveg432/OmniST](https://github.com/Praveg432/OmniST)


OmniST is a spatiotemporal forecaster whose computation has **no structural dependency on the number
of sensors** `N`. Composed with inductive split-conformal calibration, it produces
**distribution-free, coverage-guaranteed prediction intervals at sensor coordinates never seen during
training**, and preserves that guarantee as sensors are added or removed. The contribution is a
*capability* (resilient, trustable inductive uncertainty under dynamic topology), demonstrated on four
real datasets across three physical domains — not a point-accuracy state of the art.

## Why it works (three design choices)

1. **Coordinate-based tokenization** — each sensor is embedded from its geographic coordinates via a
   learned spatial MLP, with no positional index or global node count.
2. **k-NN spatial attention** — neighbours are selected from coordinates at every forward pass, so the
   graph adapts to whatever sensors are present (no pre-computed `N x N` adjacency).
3. **Per-node temporal processing** — a bidirectional GRU runs independently per node; adding/removing
   a node changes no other node's representation.

A short **Proposition** (paper §Theory) formalizes the consequence: each node's nonconformity score
is *exactly invariant* to nodes outside its k-NN set, so the conformal guarantee is recalibration-free
under far topology changes.

## Key results (all from committed result files; see `RESULTS-INDEX.md`)

- **Guaranteed inductive coverage** at unseen sensors on **4 real datasets / 3 domains**: Delhi 0.914,
  Beijing 0.928, weather 0.919, PeMSD7-M traffic 0.915 (target 0.90), holding under 30–50% node
  dropout; the only method among four model-free interpolators to hold the target on all four.
- **Beats learned baselines at unseen stations**: inductive MAE better than nearest-neighbor (~12%)
  and the learned IGNNK kriging GNN (~21% on Delhi), while hitting the coverage target more precisely
  with tighter intervals.
- **Controlled comparison** vs a faithful E-STGCN reimplementation (identical data/seeds/budget):
  comparable transductive accuracy; the fixed-topology baseline structurally cannot predict at unseen
  nodes and crashes under node dropout, demonstrated (not asserted).
- **Scale**: coverage holds at 325 real sensors (full PeMS-BAY) with a scalable k-NN attention that is
  numerically equivalent to exact (cosine 0.99996) and removes the O(N^2) bottleneck.
- **Temporal drift**: a rolling adaptive-conformal layer restores coverage over time (0.89 overall /
  0.96 in the worst-drift window vs 0.50 for static), at an honest width cost.

## Install

```bash
python3 -m pip install -r requirements.txt   # pinned, verified environment
# or:  python3 -m pip install -e ".[dev]"
```

## Reproduce

```bash
pytest -q                      # component + integration tests (13 tests)
python run_all.py              # staged, resumable manifest of all experiments
python experiments/make_figures.py   # regenerate paper figures from result JSONs
```

Every number in the paper maps to a committed result JSON and the script that produced it — see
[`RESULTS-INDEX.md`](RESULTS-INDEX.md). Headline datasets (Delhi, weather) ship in full; the bulky
traffic/Beijing raw files are fetched from documented public sources — see
[`data/README.md`](data/README.md).

## Repository layout

```
src/
  omnist_model.py        full OmniST model (coordinate tokenizer + k-NN GAT + BiGRU + head)
  inductive_gat.py       coordinate-based k-NN attention (exact + scalable cKDTree paths)
  conformal.py           split-conformal calibration
  online_conformal.py    rolling-window + ACI + Mondrian (temporal / conditional coverage)
  evt_loss.py            Peaks-Over-Threshold loss (adopted from E-STGCN)
  trainer.py             training / evaluation loop
  baselines.py           E-STGCN (faithful reimpl), LSTM-only, DLinear
  ignnk_baseline.py      IGNNK learned inductive-kriging baseline (faithful reimpl)
  data_real.py / data_beijing.py / data_weather.py / data_traffic_real.py / data_pemsbay.py
                         real-dataset loaders (provenance documented inline)

experiments/
  06_inductive_conformal.py      headline inductive coverage (real CPCB)
  B1_multidomain_v2.py           cross-domain coverage (4 real datasets)
  B2_baselines.py                OmniST vs 4 model-free interpolators
  B3_ablations.py                ablation on the coverage metric
  B4_calibration.py              calibration curves / size-stratified
  B5_theorem_check.py            empirical verification of the locality Proposition
  B6_estgcn_controlled.py        controlled OmniST vs E-STGCN (identical conditions)
  B7_largeN_pemsbay.py           large-N coverage (325 real sensors)
  B8_fixedtopo_crash_demo.py     fixed-topology crash under node dropout
  B9_ignnk_inductive.py          OmniST vs learned IGNNK at unseen nodes
  10/11/12/13/14_*.py            temporal drift, multi-target, scalability, Mondrian, scalable k-NN
  make_figures.py                regenerate all paper figures

results/      committed result JSONs (one per experiment) + figures/
paper/        LaTeX source (IEEEtran) + references.bib
```

## Honest scope

The contribution is the *capability* (guaranteed inductive UQ under dynamic topology), not a
point-accuracy SOTA. Baselines (E-STGCN, IGNNK) are faithful reimplementations, not the authors'
original code. Coverage is certified to 325 real sensors; thousands-of-node certification awaits a
larger real benchmark (a data gate, not an architecture gate). The conformal guarantee is marginal,
with a partial Mondrian fix for conditional coverage.

## Citation

Paper: <https://arxiv.org/abs/2611.04256> (arXiv:2611.04256 [cs.LG]). LaTeX source is in `paper/`.
Repository: <https://github.com/Praveg432/OmniST>.

If you use this code or build on the method, please cite:

```bibtex
@misc{vashishtha2026omnist,
  title         = {OmniST: Topology-Invariant Spatiotemporal Forecasting with Guaranteed-Coverage
                   Uncertainty at Unseen Sensor Locations},
  author        = {Vashishtha, Praveg},
  year          = {2026},
  eprint        = {2611.04256},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.04256}
}
```

## License

MIT — see [`LICENSE`](LICENSE).

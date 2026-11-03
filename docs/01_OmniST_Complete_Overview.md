# OmniST — Project Overview

> Document 1 of 3. A top-level walkthrough of the method: the problem, the idea, the architecture at a
> glance, the experiments, the design decisions and how known limitations are addressed, and the
> findings. For the underlying concepts read `02_Foundational_Concepts.md`; for the math and code
> deep-dive read `03_Math_and_Implementation_Deep_Dive.md`.

---

## 0. One-sentence summary

OmniST is a forecasting model for sensor networks that (a) works for any number of sensors in any
spatial layout — including sensors absent from training — and (b) attaches a distribution-free,
coverage-guaranteed interval to every prediction, including at never-seen locations, and preserves
that guarantee when sensors are added or removed.

---

## 1. The problem

### 1.1 Spatiotemporal forecasting
A network of sensors (air-quality monitors, traffic loops, weather stations), each producing a time
series. The task is to forecast future values. The difficulty is joint **spatial + temporal**
structure: nearby sensors correlate; each sensor has its own dynamics.

### 1.2 The dominant approach and its structural constraint
Standard **STGCNs** (Spatio-Temporal Graph Convolutional Networks) model the network as a *fixed*
graph with a pre-computed `N x N` adjacency matrix `A`, and learn weights shaped around that `A`. This
**couples the model to a specific sensor count and layout**:
- Remove a sensor -> `A` has a stale row/column -> dimension mismatch.
- Add a sensor -> `A` has no row/column for it -> the model **structurally cannot produce output** for
  that location (not a poor output — no output).

Because real networks change continually (sensors fail, are added, or go offline), this is an
operational constraint, not only a theoretical one.

### 1.3 The uncertainty gap
A single point forecast at a new sensor does not tell an operator how much to trust it. The second
requirement is a calibrated error bar with a validity guarantee.

### 1.4 Scope of OmniST's claim
OmniST addresses both: an architecture with **zero structural dependence on `N`** (so it predicts
anywhere, under any topology) plus a calibration layer giving **distribution-free, coverage-guaranteed**
intervals at unseen locations, preserved under topology change.

OmniST does **not** claim higher point accuracy than STGCNs on the standard fixed-topology benchmark.
The contribution is a *capability* those models do not provide. Keeping this scope precise is a
deliberate design choice and governs how every result below is framed.

---

## 2. The architecture at a glance (4 stages + a wrapper)

Design invariant: *for any component f and node i, f(i) depends only on node i's own data and a
fixed-size k-NN neighborhood — never on the global node set.*

1. **Coordinate tokenizer** (`src/cc_hash.py`) — token = spatial(coords) + temporal(time) +
   feature(measurement). The spatial part is a learned MLP on `(lat, lon)`, so a new sensor obtains a
   valid embedding from its coordinates alone. Each token is computed independently of other nodes.
2. **Inductive k-NN attention / GAT** (`src/inductive_gat.py`) — the graph is built on the fly each
   forward pass: Haversine distances -> k nearest neighbors -> multi-head attention with a learned
   distance bias. No `N x N` matrix. Two numerically-equivalent paths: `exact` (small N) and
   `scalable` (cKDTree, large N).
3. **Per-node temporal backbone** (`src/temporal.py`) — a bidirectional GRU applied independently per
   node, so adding/removing a node changes no other node's temporal representation.
4. **Prediction head** — a 2-layer MLP to the forecast horizon.
5. **Conformal wrapper** (`src/conformal.py`, `src/online_conformal.py`) — turns the point model into
   guaranteed intervals (split conformal in space; rolling-ACI in time). Because the point model is
   inductive, the intervals are valid at unseen coordinates.

Model size: ~99K parameters at d=64 (~315K at d=128) — intentionally small, since the contribution is
the design rather than scale.

> Naming note: the repository directory is `omnist-mamba`. An earlier design explored a Mamba/S4D
> temporal backbone; the final model uses a GRU, and the temporal choice is discussed in Section 7 of
> `02_Foundational_Concepts.md`. The model class is `OmniST`.

---

## 3. The theory (the locality Proposition)

**Locality => exact score invariance.** Each node's nonconformity (calibration) score depends only on
its own coordinates/features and its k-NN set. Adding or removing nodes *outside* that set therefore
leaves the score **exactly unchanged**, so the conformal threshold stays valid with **no
recalibration** for such changes. This is stronger than the general message-passing-GNN case, where
any node change perturbs all embeddings and the calibration scores drift (requiring correction).

**Empirical verification (B5):** far-change prediction difference `8.9e-8` (floating-point zero =
exactly invariant); control (dropping a true neighbor) `~7e-3` (locality is substantive, not vacuous).
Verdict: SUPPORTED. See `03_Math_and_Implementation_Deep_Dive.md` Section 7 for statement, proof, and
the metric subtlety.

---

## 4. The experiments (the evidence arc)

Four real datasets / three physical domains: Delhi air quality (CPCB, 37), Beijing air quality
(UCI, 12), Western-US weather (NOAA GHCN, 30), Caltrans PeMSD7-M traffic (228); plus PeMS-BAY (325)
for scale. Every number maps to a committed JSON — see `RESULTS-INDEX.md`.

| # | Experiment (script) | What it shows | Headline number |
|---|---|---|---|
| 1 | Monthly accuracy (Delhi) | comparable to E-STGCN | 31.0 ug/m3 MAE |
| 2 | Sensor-failure resilience | graceful degradation | <= 1.1% at 40% removal |
| 3 | Inductive point (`tab:inductive`) | beats NN *and* learned IGNNK | 26.7 vs 30.3 vs 34.0 |
| 4 | Guaranteed coverage (`tab:conformal`) | valid, tighter, holds under dropout | 0.915 cov, 14% tighter |
| 5 | Cross-domain coverage (B1) | holds on all 4; only method to | 0.914/0.928/0.919/0.915 |
| 6 | Controlled E-STGCN (B6) | comparable transductive; E-STGCN cannot predict unseen | p=0.03 Delhi |
| 7 | Fixed-topology crash (B8) | E-STGCN RuntimeError under dropout; OmniST holds | demonstrated |
| 8 | Large-N scale (B7) | coverage holds at 325 real sensors | 0.921 |
| 9 | Learned baseline (B9, IGNNK) | OmniST beats learned kriging on MAE + coverage | ~21% MAE, 0.915 vs 0.891 |
| 10 | Ablation (B3) | the *construction* earns the guarantee, not GAT/k | coverage within +-0.01 |
| 11 | Calibration (B4) | curves track target; high stratum under-covers (reported) | — |
| 12 | Temporal drift (10) | rolling-ACI restores time coverage | 0.50 -> 0.96 late |
| 13 | Mondrian (13) | partial conditional-coverage fix | worst group 0.80 -> 0.86 |

Rigor: 10 seeds (air/weather), 6 (traffic); paired t-tests; mean+-std; fully reproducible from the
committed JSONs.

---

## 5. Design decisions and how limitations are addressed

Each item records a design choice, the reason, and how an associated limitation is handled.

1. **Temporal drift.** Split conformal assumes exchangeability, which seasonality breaks over a test
   year: a frozen calibration quantile under-covers later (static conformal reaches ~0.50 coverage in
   the late-drift window). *Resolution:* an online conformal layer — a rolling-window quantile (tracks
   the drifting residual scale) wrapped in Adaptive Conformal Inference (ACI). Rolling-ACI restores
   target coverage over time (0.89 overall, 0.96 late) at the cost of wider intervals (170 vs 95), an
   unavoidable and explicitly reported trade-off.
2. **Baseline strength.** Comparing inductive prediction only against nearest-neighbor interpolation
   would be weak. *Resolution:* a learned inductive baseline — a faithful reimplementation of **IGNNK**
   (diffusion-GCN kriging) — is run in a leakage-free head-to-head (B9). OmniST is more accurate at
   unseen stations and, unlike IGNNK, coverage-valid.
3. **E-STGCN comparison protocol.** Comparing against published numbers alone leaves a protocol
   confound. *Resolution:* a controlled reimplementation under identical data/seeds/budget (B6),
   reported in both directions (comparable transductive accuracy; E-STGCN marginally better on weather,
   OmniST on Delhi).
4. **Theory verification metric.** The locality Proposition must be checked with the metric the model
   actually uses. An initial check using Euclidean distance to define "far" gave a false nonzero
   result; using the model's Haversine metric yields the exact bit-identical invariance (B5).
5. **Scale certification.** Coverage is certified to 325 real sensors; no larger *real* benchmark was
   available. This is a **data limitation, not an architecture limitation** — the scalable k-NN path is
   verified numerically equivalent to exact (cosine 0.99996), so the method itself scales. Synthetic
   node duplication is used only for a runtime stress test and is explicitly not a coverage claim.
6. **Conditional coverage.** The guarantee is marginal (holds on average), which can hide per-node
   spread on irregular networks. *Partial resolution:* Mondrian group-conditional conformal lifts the
   worst traffic group from 0.80 to 0.86 — an improvement, stated as partial.
7. **Spatial-attention contribution.** Ablations show the coordinate-based inductive construction, not
   the GAT or a specific k, earns the marginal coverage guarantee. The GAT's role is enabling inductive
   prediction at all and mildly sharpening intervals; this is stated rather than overclaimed.

---

## 6. Findings

1. Topology invariance is achievable **by construction** — remove every reference to `N`, and the
   model handles any sensor set.
2. The **inductive conformal guarantee transfers to unseen coordinates** and survives topology change
   — the central result, demonstrated across 4 real datasets / 3 domains.
3. The **coordinate-based construction** earns the guarantee, not the GAT or a particular k
   (ablation-confirmed).
4. The guarantee is **marginal, not conditional**; Mondrian partially closes the gap.
5. **Space and time are both handled** (split conformal + rolling-ACI), with an explicit width cost in
   time.
6. There is **no point-accuracy SOTA claim**; the contribution is a capability neither fixed-topology
   STGCNs nor model-free interpolation provide.

---

## 7. Positioning and scope

The method is scoped precisely throughout: it *matches* rather than beats on transductive accuracy; it
*demonstrates* structural failures of fixed-topology baselines rather than asserting them; scale is a
*data* limit rather than an architectural one; and the conformal baselines (E-STGCN, IGNNK) are
faithful reimplementations, not the authors' original code. Combined with full reproducibility — every
reported number maps to a committed result file — this makes the contribution precise and verifiable.

---

## 8. Repository map (for navigation)

```
src/         model + conformal + losses + data loaders
experiments/ 06, B1-B9, 10-14, make_figures  (each -> a results/*.json)
results/     committed JSONs (evidence) + figures/
paper/       OmniST-Paper-v2.tex + references.bib
docs/        these three reference documents
RESULTS-INDEX.md  maps every reported number -> its JSON + script
data/README.md    dataset provenance + fetch instructions
```

## 9. Key terms

topology invariance; inductive vs transductive; split conformal prediction; nonconformity score;
exchangeability; marginal vs conditional coverage; coverage-vs-width trade-off; Haversine k-NN
attention; Adaptive Conformal Inference (ACI); the locality Proposition (exact far-change score
invariance).

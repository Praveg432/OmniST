# OmniST — Foundational Concepts (First Principles)

> Document 2 of 3. This explains every concept OmniST rests on, from the ground up, assuming only
> basic ML/linear-algebra background. Read `01_OmniST_Complete_Overview.md` first for the big
> picture; read `03_Math_and_Implementation_Deep_Dive.md` after this for the equations and code.

---

## 1. Spatiotemporal data and why it is hard

A spatiotemporal dataset is a set of `N` locations (sensors), each with coordinates `c_i = (lat, lon)`,
each producing a time series `x_i(t)`. Two coupled structures:

- **Temporal**: each series has trends, seasonality, autocorrelation.
- **Spatial**: values at nearby locations correlate (pollution drifts, traffic propagates, weather
  fronts move).

A good model must exploit *both*. Pure time-series models ignore space; pure spatial models ignore
time. Spatiotemporal models fuse them — the question is *how* they represent space.

---

## 2. Graphs and the adjacency matrix (the root of the problem)

A natural way to encode "which sensors are related" is a **graph**: nodes = sensors, edges = spatial
relationships, encoded in an **adjacency matrix** `A` of shape `N x N`, where `A[i][j]` is the
edge weight between sensors i and j (often a function of distance, e.g. a Gaussian kernel).

**Graph convolution** then mixes each node's features with its neighbors' via `A` (e.g. `H' = A H W`,
where `H` is node features and `W` is a learned weight). This is powerful but has a structural
consequence:

> `A` is `N x N`. The operation `A H` only makes sense when `H` has exactly `N` rows. The learned
> pipeline is shaped around a fixed `N`.

This is the coupling OmniST removes. If you *never form* an `N x N` matrix, you are never welded to a
fixed `N`.

### 2.1 Chebyshev / spectral GCN (what E-STGCN uses)
"Spectral" GCNs operate in the graph's Fourier domain via the graph Laplacian's eigenvectors.
Chebyshev GCNs approximate this with polynomials of `A` (orders 0..K), giving efficient K-hop
mixing. They are excellent on a *fixed* graph but inherit the fixed-`N` coupling and need the full
Laplacian — exactly what breaks under topology change.

---

## 3. Inductive vs transductive learning (the central distinction)

- **Transductive**: the model is trained and tested on the *same* set of entities. It can learn a
  per-entity parameter (e.g. a learned embedding per sensor ID). It cannot handle a *new* entity
  without retraining, because the new entity has no learned parameter.
- **Inductive**: the model learns a *function* of entity *features* (here, coordinates), so it
  generalizes to entities it never saw. A new sensor is handled instantly because it has coordinates.

OmniST is inductive because every node is represented by a function of its coordinates/features, never
by a learned per-node ID. This is the single most important design decision in the paper.

---

## 4. Attention and the Graph Attention Network (GAT)

### 4.1 Attention in one paragraph
Attention lets a token gather information from other tokens, weighted by learned relevance. For a
query `q_i` and keys/values `{k_j, v_j}`, the output is `sum_j softmax_j(q_i . k_j / sqrt(d)) v_j`.
"Softmax over j" makes the weights a probability distribution — the token decides how much to listen
to each other token.

### 4.2 GAT = attention on a graph
A Graph Attention Network restricts the "other tokens" to graph neighbors and learns the attention
weights (rather than fixing them in `A`). This is more flexible than fixed-weight graph convolution.

### 4.3 OmniST's twist: the neighborhood is computed from coordinates, live
Instead of neighbors defined by a stored `A`, OmniST computes each node's **k nearest neighbors from
coordinates at every forward pass** and attends only over those k. Consequences:
- No `N x N` matrix -> topology invariant.
- The neighbor set *adapts* to whoever is present (drop nodes -> remaining nodes re-find their k
  nearest; add a node -> it finds its k nearest and participates).
- Cost is `O(N . k)` in the neighborhood, not `O(N^2)` — and the `scalable` path makes neighbor
  *selection* `O(N log N)` via a spatial index.

---

## 5. Distance on a sphere: the Haversine metric

Sensors live on the Earth's surface; `(lat, lon)` are angles, not Cartesian coordinates. The correct
"nearness" is the **great-circle distance** (shortest path on a sphere), computed by the **Haversine
formula**. Using plain Euclidean distance on raw `(lat, lon)` distorts distances (a degree of
longitude is shorter near the poles). OmniST uses Haversine for neighbor selection and for the
learned distance bias.

> This mattered concretely: the theory-verification bug (Doc 1, Issue 4) came from using Euclidean
> distance to define "far" when the model's neighborhoods are Haversine-defined. "Far in Euclidean"
> did not equal "outside the Haversine k-NN set," producing a false result until the metric was
> matched.

---

## 6. Positional / coordinate encoding (how a location becomes a vector)

A model can't consume `(lat, lon)` as two raw numbers very effectively. Two standard ways to turn a
position into a rich vector:

- **Fourier / sinusoidal features**: project the coordinate through many sin/cos frequencies (as in
  Transformer positional encodings and NeRF). Good for high-frequency spatial detail.
- **Learned MLP**: feed the (normalized) coordinate through a small neural network that *learns* the
  spatial embedding. More flexible; better here for *generalizing to unseen coordinates* (validated
  early on; the MLP won over Fourier by ~8% on inductive performance).

OmniST uses the **learned MLP** for space, and **multi-scale sinusoidal** encodings for *time*
(day-of-year, day-of-week, month, year), capturing seasonality at several frequencies at once.

---

## 7. Recurrent networks and the GRU (the temporal backbone)

A **Recurrent Neural Network (RNN)** reads a sequence step by step, carrying a hidden state. Plain
RNNs forget long-range information (vanishing gradients). The **GRU (Gated Recurrent Unit)** adds
learned gates (update/reset) that control what to keep vs overwrite, handling longer dependencies
robustly. A **bidirectional** GRU runs two passes (forward + backward) and concatenates them, so each
step sees both past and future context within the window.

OmniST runs the GRU **per node** (each sensor's series independently). This is why "robustness to node
dropout" is almost automatic for this architecture: no node's temporal computation depends on any
other node.

> Why GRU and not Mamba/S4D? State-space models (Mamba, S4D) give O(T) memory and shine on *very long*
> sequences (T > 1000). This paper's windows are short (30-60 steps), so a GRU is simpler, proven, and
> MPS-friendly with no benefit left on the table. (The repo's legacy name reflects an earlier
> Mamba-based plan.)

---

## 8. Uncertainty quantification and conformal prediction (the statistical core)

### 8.1 Why point forecasts are not enough
A single predicted number gives no sense of reliability. Operators need an interval: "the true value
will be in [a, b] with 90% confidence." The hard part is making that 90% *actually true*.

### 8.2 Conformal prediction: guaranteed intervals, no distribution assumptions
Conformal prediction wraps ANY point model and produces intervals with a **finite-sample,
distribution-free coverage guarantee**. "Finite-sample" = the guarantee holds for your actual (finite)
calibration set, not only asymptotically. "Distribution-free" = no assumption about the data's
distribution or the model's form.

**Split conformal, step by step:**
1. Train the model on training data.
2. On a held-out **calibration set**, compute **nonconformity scores** — here, absolute residuals
   `s_i = |pred_i - actual_i|`.
3. Choose target coverage `1 - alpha` (e.g. 0.90, so `alpha = 0.10`).
4. Compute `q_hat` = the `ceil((n+1)(1-alpha)) / n` quantile of the calibration scores.
5. For any new input, output `[pred - q_hat, pred + q_hat]`.

**Guarantee:** if calibration and test scores are **exchangeable**, then
`P(actual in interval) >= 1 - alpha`. That's it — a rigorous probability bound from a quantile.

### 8.3 Exchangeability (the one assumption)
A sequence is **exchangeable** if its joint distribution is unchanged by reordering (weaker than
"i.i.d." but similar in spirit). Conformal's guarantee needs the calibration scores and the test score
to be exchangeable. Two ways this breaks:
- **Across space (inductive)**: calibration on training stations, test on an *unseen* station — are
  their residuals exchangeable? (OmniST's Proposition addresses this.)
- **Across time (drift)**: the residual distribution changes over the test horizon (seasonality), so
  an early-calibrated quantile under-covers later. (Rolling-ACI addresses this.)

### 8.4 Coverage vs width (the trade-off you must internalize)
- **Coverage** = fraction of intervals that actually contain the truth. Valid iff `>= target`.
- **Width** = how wide the intervals are. Narrower is more *useful*, but only if still valid.
- A narrow interval that under-covers (e.g. 0.80 at a 0.90 target) is **lying** about its confidence.
  A slightly wider valid interval is **honest**. Model-free interpolators in the paper are often
  tighter precisely because they under-cover — which is why tighter is not automatically better.

### 8.5 Marginal vs conditional coverage
- **Marginal**: coverage holds *on average* over all test points. This is what split conformal
  guarantees.
- **Conditional**: coverage holds for *every subgroup* (e.g. every station, every horizon). Much
  harder; not guaranteed by vanilla conformal. On irregular networks, marginal 0.92 can hide per-node
  spread of 0.73-0.98. **Mondrian conformal** (Section 10) partially restores conditional coverage.

### 8.6 Adaptive Conformal Inference (ACI) — coverage under drift
Gibbs & Candes (2021). Instead of a frozen `alpha`, ACI updates an *effective* level online:
`alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`, where `err_t = 1` if the last interval missed. If
you have been under-covering, `alpha_t` drops -> intervals widen; over-covering -> they shrink. This
drives long-run empirical miscoverage to `alpha` for **any** sequence — no exchangeability required —
which is exactly the property that survives temporal drift. OmniST combines a rolling-window base
quantile (tracks the drifting *scale*) with an ACI correction (removes residual *bias*): "Rolling-ACI."

---

## 9. The baselines (what OmniST is measured against, and why each)

- **Nearest-Neighbor (NN) interpolation**: predict an unseen sensor by copying its nearest training
  sensor's series. The simplest model-free inductive baseline.
- **IDW (Inverse-Distance Weighting)**: weighted average of nearby sensors, weight ~ 1/distance.
- **mean-k**: plain average of the k nearest sensors.
- **GP / kriging**: a Gaussian-process spatial interpolator — the classical statistical approach to
  "predict a field at an unobserved location."
- **E-STGCN** (Panja et al.): the fixed-topology deep baseline (Chebyshev-GCN + LSTM + POT loss); the
  method OmniST is positioned against. Valid transductively, structurally fails inductively.
- **IGNNK** (Wu et al., 2021): the canonical *learned* inductive-kriging GNN — a diffusion-GCN trained
  by masking out nodes and reconstructing them. The strongest inductive baseline; added during
  review-hardening. OmniST beats it on both accuracy and coverage.

All interpolation baselines are wrapped in the *same* split conformal, so the comparison is about the
underlying predictor's quality, not the UQ machinery.

---

## 10. Mondrian (group-conditional) conformal

To push toward conditional coverage: partition test nodes into `G` groups by a difficulty feature
(here, distance from an unseen node to its nearest *training* node — farther = harder), calibrate a
**separate quantile per group**, and apply each group's quantile to its own nodes. Under within-group
exchangeability each group gets `>= 1 - alpha`, so hard/sparse nodes get appropriately wider intervals
instead of free-riding on the easy-node average. In the paper it lifts the worst traffic group from
0.80 toward target (0.86) — an improvement, not a complete fix.

---

## 11. Extreme-Value Theory and the POT loss (a borrowed component)

Air-quality/traffic data have heavy tails (pollution spikes, congestion). **Extreme-Value Theory
(EVT)** models the tail: exceedances over a high threshold follow a **Generalized Pareto
Distribution (GPD)**. The **Peaks-Over-Threshold (POT) loss** (from E-STGCN) adds a GPD
negative-log-likelihood penalty for predictions above a threshold, on top of MSE below it. OmniST
*adopts* this loss for consistency with E-STGCN (so the controlled comparison differs only in the
spatial backbone). It is not a contribution of this paper, and coverage — the primary metric — is a
property of the conformal wrapper, independent of the loss choice.

---

## 12. How the concepts compose into OmniST

```
coordinates ---(learned MLP)------\
timestamps  ---(sinusoidal)--------> token  --(k-NN GAT, Haversine)--> spatial context
measurement ---(linear)-----------/                                         |
                                                                   (per-node BiGRU)
                                                                            |
                                                                    (MLP head) -> point forecast
                                                                            |
                                                      (split conformal in space, rolling-ACI in time)
                                                                            |
                                                   guaranteed interval at ANY coordinate, any topology
```

Every arrow is a concept above. Inductivity comes from the coordinate MLP; topology invariance from
the live k-NN + per-node processing; the guarantee from conformal; drift-robustness from ACI; the
sharper tails from POT. The Proposition explains *why* the guarantee survives topology change.

# OmniST — Math and Implementation Deep Dive

> Document 3 of 3. Equations and the actual implemented mechanics, keyed to the source files. Notation
> is ASCII-math. Read `02_Foundational_Concepts.md` first for the intuition behind each piece.

## Notation

- `B` batch, `T` input timesteps, `N` nodes (sensors), `F` input features, `d` model width,
  `H` forecast horizon, `k` neighbors, `n_h` attention heads, `d_h = d / n_h`.
- `c_i = (lat_i, lon_i)` degrees; `x_i(t)` measurement; `y_i` target; `hat_y_i` prediction.
- Tensors annotated with shapes, e.g. `features: (B, T, N, F)`.

---

## 1. Forward pass, end to end (shapes keyed to `src/omnist_model.py::OmniST.forward`)

```
Input:  features (B,T,N,F), coords (B,N,2) degrees, timestamps (B,T,4)
1. tokens      = Tokenizer(features, coords, timestamps)      -> (B,T,N,d)
2. for each GAT layer, for each t in 0..T-1:
       tokens[:,t] = GAT(tokens[:,t], coords)                 -> (B,N,d)   (restacked to (B,T,N,d))
3. tokens      = BiGRU(tokens)                                -> (B,T,N,d)
4. last_hidden = tokens[:, -1]                                -> (B,N,d)
   output      = Head(last_hidden)                            -> (B,N,H*F)
   return output.view(B, N, H, F)
```

Note the design consequence: **no step ever uses `N` as a learned dimension.** `N` only appears as a
*runtime* tensor extent, so any `N` is valid at inference. That is topology invariance, mechanically.

---

## 2. Stage 1 — Coordinate tokenizer (`src/cc_hash.py`)

Token = spatial + temporal + feature, then LayerNorm. Each term is computed per-node without
reference to other nodes.

### 2.1 Spatial embedding (`SpatialMLP`)
Normalize then a 3-layer MLP:
```
c_norm = ((lat - 28.6)/0.2, (lon - 77.1)/0.3)              # centering (Delhi-centered defaults)
s_i    = MLP(c_norm) in R^d,   MLP: 2 -> 128 -> 128 -> d  with GELU
```
Key property: `s_i` depends only on `c_i`. A never-seen coordinate gets a valid embedding -> the
source of inductivity. (`embed_coords_only(coords)` exposes this directly for inductive inference.)

### 2.2 Temporal embedding (`TemporalEmbedding`)
Four normalized time features `tau in [0,1]^4` (day-of-year, day-of-week, month, year), expanded over
harmonic frequencies `f_m = 2^m * 2*pi`, `m = 0..M-1`:
```
phi(tau) = concat_over_features_and_m [ sin(tau * f_m), cos(tau * f_m) ]   in R^{4*2*M}
temporal = MLP(phi(tau))  in R^d
```
Multi-scale sinusoids capture seasonality at several periods simultaneously.

### 2.3 Feature + compose
```
feat   = Linear_F->d(x)                       # per node, per timestep
token  = LayerNorm(spatial + temporal + feat) # additive composition, (B,T,N,d)
```

---

## 3. Stage 2 — Inductive k-NN attention (`src/inductive_gat.py::InductiveGAT`)

### 3.1 Haversine distance (used for neighbor selection + distance bias)
For nodes i, j with lat/lon in radians:
```
a_ij = sin^2((lat_i - lat_j)/2) + cos(lat_i) cos(lat_j) sin^2((lon_i - lon_j)/2)
d_ij = 2 R * atan2( sqrt(a_ij), sqrt(1 - a_ij) ),   R = 6371 km
```

### 3.2 Neighbor selection (two equivalent paths)
- `exact` (or `N <= 64`): form `D in R^{B x N x N}`, mask the diagonal with +inf, take the k smallest
  per row -> `knn_idx (B,N,k)`, `knn_dists (B,N,k)`. Cost: O(N^2) time and memory.
- `scalable`: per-batch `cKDTree` on an equirectangular projection
  `xy = (radians(lon) * cos(mean_lat), radians(lat))`, query k+1 nearest, drop self -> `knn_idx`.
  Then recompute TRUE Haversine on only the k selected pairs for the distance bias. Cost:
  O(N log N) index + O(N k) gather; no `N x N` tensor is ever formed.
  The attention math is identical; only *selection* differs. Verified equivalent: output cosine
  similarity 0.99996 at N=228 (B5/14), which is why coverage/accuracy match `exact` within noise.

### 3.3 Multi-head attention over the k neighbors
Project `Q, K, V = (W_q, W_k, W_v) token`, reshape to `(B,N,n_h,d_h)`. Gather neighbor K,V by index
(O(N k) memory, no `N x N` expansion). Then per node i, head h, neighbor j in its k-set:
```
score_{i,j,h} = (Q_{i,h} . K_{j,h}) / sqrt(d_h)  +  b_h(d_ij)      # b = learned Linear(1->n_h)
alpha_{i,.,h} = softmax_over_j(score_{i,.,h})                       # (B,N,k,n_h)
out_{i,h}     = sum_j alpha_{i,j,h} V_{j,h}
out_i         = Concat_h out_{i,h}  in R^d
output_i      = LayerNorm( W_o out_i + token_i )                    # residual + norm
```
The **distance bias** `b_h(d_ij)` lets each head learn how strongly proximity should matter.
Softmax over *neighbors* (dim=2) is what makes the k attention weights a distribution.

---

## 4. Stage 3 — Per-node bidirectional GRU (`src/temporal.py::BidirectionalGRU`)

```
x (B,T,N,d) --permute/reshape--> (B*N, T, d)        # each node's series, independent
out, _ = GRU(x)                                      # bidirectional: hidden d/2 each dir -> concat d
out    = LayerNorm( Linear(out) + x )                # residual + norm
--reshape--> (B,T,N,d)
```
Because the node axis is folded into the batch for the GRU, **no cross-node mixing** occurs in time.
This is the mechanical reason node dropout barely perturbs other nodes (Doc 1/2: near-tautological
robustness).

---

## 5. Stage 4 — Prediction head

```
last = tokens[:, -1]                        # (B,N,d)
out  = Linear_d->d; GELU; Dropout; Linear_d->(H*F)   (last)   # (B,N,H*F)
pred = out.view(B, N, H, F)
```

---

## 6. Split conformal prediction (`src/conformal.py`)

### 6.1 Nonconformity score and calibrated half-width
Score = absolute residual on the real (denormalized) scale: `s_i = |hat_y_i - y_i|`. Given `n`
calibration scores and miscoverage `alpha`:
```
level  = min(1, (1 - alpha) * (1 + 1/n))     # finite-sample inflation (the (n+1) correction)
q_hat  = Quantile(level) of { s_i }
interval(x) = [ hat_y(x) - q_hat , hat_y(x) + q_hat ]   (clip at 0 for non-negative quantities)
```
The `(1 + 1/n)` factor is the exact finite-sample correction that yields the coverage bound below.

### 6.2 The guarantee
If the calibration scores `s_1..s_n` and a test score `s_{n+1}` are **exchangeable**, then
```
P( y_test in interval ) >= 1 - alpha
```
Proof idea: by exchangeability, the rank of `s_{n+1}` among the `n+1` scores is uniform on
`{1,...,n+1}`. `y_test` is covered iff `s_{n+1} <= q_hat`, i.e. its rank is `<= ceil((n+1)(1-alpha))`.
The probability of that is `ceil((n+1)(1-alpha)) / (n+1) >= 1 - alpha`. No distributional assumption is
used anywhere — only exchangeability.

### 6.3 The inductive application (what makes OmniST's use nontrivial)
Calibration scores are computed on **training** stations; the test score is at an **unseen** station.
Validity therefore requires training- and unseen-station scores to be exchangeable. Section 7 (the
Proposition) is precisely the argument that OmniST's per-node construction keeps the score
distribution stable across node sets, so the guarantee transfers inductively and survives topology
change.

---

## 7. The locality Proposition (statement, proof, verification)

### 7.1 Statement
Let node i's nonconformity score be
```
s_i = g( c_i, x_i, { (c_j, x_j) : j in N_k(i; S) }, y_i )
```
where `N_k(i; S)` is i's k nearest *other* nodes in the present set `S` (by Haversine distance). For
any `S'` with `N_k(i; S') = N_k(i; S)` — in particular, adding/removing nodes OUTSIDE i's k-set —
`s_i(S') = s_i(S)` exactly. Hence a split-conformal threshold `q_hat` calibrated on training-station
scores stays valid with NO recalibration for any test node whose k-set is unchanged; and coverage is
preserved more broadly whenever the reconfigured local receptive fields remain exchangeable.

### 7.2 Proof sketch
Every OmniST component is a function of a node's own data and its k-NN receptive field only:
- the coordinate-MLP token depends on `(c_i, x_i)`;
- the k-NN attention aggregates exactly `N_k(i; S)`;
- the BiGRU and head act per node.
No operation references the global count `N` or a node index, and there is **no cross-node reduction**
(no batch-wise statistic over nodes). Therefore `s_i` is a fixed function of
`(c_i, x_i, N_k(i;S)-data, y_i)`; if that argument is unchanged, `s_i` is unchanged. Validity of
`q_hat` then follows from standard split conformal, since neither the calibration scores nor the
unchanged test score move. QED.

### 7.3 Why stronger than general inductive GNNs
Message-passing GNNs update every node's embedding when any node is added/removed, so all calibration
scores drift and coverage degrades unless corrected. OmniST has **no drift to correct** for far
changes — recalibration-free in the exact-invariance regime.

### 7.4 Empirical verification (`experiments/B5_theorem_check.py` -> `results/exp_theorem_check.json`)
- Far change (drop nodes OUTSIDE a test node's Haversine k-set): max abs prediction diff
  `= 8.94e-8` -> exactly invariant to float precision. `P1_exact_invariance_holds = true`.
- Control (drop a true neighbor): diff `= 6.9e-3` -> locality is real, not vacuous.
- Coverage band (P2): all four domains stay within +-0.03 of target under 30/50% dropout.
- Verdict: SUPPORTED.
- **Bug caught + fixed**: using Euclidean distance to define "far" gave a false nonzero diff
  (`0.0074`); switching to the model's Haversine metric gave the bit-identical `8.9e-8`.

---

## 8. Online conformal under temporal drift (`src/online_conformal.py`)

### 8.1 Rolling-window conformal (`RollingConformal`)
Recompute `q_hat` from only the residuals in the last `window` steps (optionally exponentially
weighted with `decay`). Tracks a drifting residual *scale* directly. Same finite-sample level
`(1-alpha)(1+1/n)` applied to the windowed residuals.

### 8.2 Adaptive Conformal Inference (`AdaptiveConformal`; Gibbs & Candes 2021)
Maintain a residual pool and an adaptive level `alpha_t`:
```
q_hat_t   = Quantile(1 - alpha_t) of pool
err_t     = 1 - covered_fraction_t            # realized miscoverage at step t
alpha_{t+1} = clip( alpha_t + gamma (alpha - err_t), 1e-3, 1-1e-3 )
```
Long-run empirical miscoverage -> `alpha` for ANY sequence (no exchangeability). If under-covering
(`err_t > alpha`), `alpha_t` falls -> the `(1-alpha_t)` quantile rises -> intervals widen, and
vice-versa.

### 8.3 Rolling-ACI (`RollingACI`, recommended)
Multiply a rolling base quantile by an ACI-controlled scale:
```
q_t   = rolling_quantile_{1-alpha}(recent residuals) * scale_t
scale_{t+1} = clip( scale_t * exp( gamma (err_t - alpha) / alpha ), 0.3, 5.0 )
```
Base tracks the drifting scale; the multiplicative ACI term removes residual coverage bias.

### 8.4 Measured (`results/exp_temporal_conformal.json`, Delhi, target 0.90)
| Method | Overall cov | Late-window cov | Width |
|---|---|---|---|
| Static split-CP | 0.758 | 0.498 | 95.4 |
| Rolling | 0.840 | 0.799 | 127.4 |
| ACI | 0.847 | 0.905 | 139.0 |
| **Rolling-ACI** | **0.890** | **0.962** | 170.1 |
Restoring late-window coverage from 0.50 to 0.96 costs width (95 -> 170) — the honest, unavoidable
trade-off: tracking a widening error distribution requires widening the interval.

---

## 9. Mondrian group-conditional conformal (`MondrianConformal`)

```
groups   = quantile_bin( difficulty_feature, G )      # feature = dist(unseen node, nearest train node)
q_hat[g] = Quantile_{(1-alpha)(1+1/n_g)}( calibration residuals in group g )   # per group
interval_i uses q_hat[ group(i) ]
```
Under within-group exchangeability each group gets `>= 1 - alpha`. Falls back to the global quantile
for groups with `< 20` calibration points. Lifts the worst traffic group 0.80 -> 0.86.

---

## 10. POT / EVT loss (`src/evt_loss.py::POTLoss`) — adopted from E-STGCN

Fit a Generalized Pareto Distribution to threshold exceedances (per node where enough data, else a
global fallback): `tau = percentile_90(data)`, exceedances `z = data - tau`,
`(xi, sigma) = GPD.fit(z)` (clipped for stability). Loss:
```
L = MSE(pred, target)                                   if target <= tau
L = beta1 * MSE + beta2 * GPD_NLL(pred - tau)           if target >  tau
GPD_NLL(z) = log(sigma) + (1 + 1/xi) * log(1 + xi z / sigma)
```
combined by the fraction of samples in each regime. Adopted so the controlled E-STGCN comparison
differs only in the spatial backbone; not a contribution. Coverage (the primary metric) is a property
of the conformal wrapper and is independent of this loss.

---

## 11. Scalable k-NN: why it is equivalent (and the complexity)

Attention math is identical for both paths; only neighbor *selection* differs. The cKDTree ranks by
planar (equirectangular) distance, which matches the exact Haversine ranking for the small local
neighborhoods that drive attention, then true Haversine is recomputed on just the k selected pairs for
the learned bias. Complexity: `exact` is O(N^2) time+memory; `scalable` is O(N log N) index + O(N k)
gather, never forming an `N x N` tensor. Equivalence is measured (cosine 0.99996 at N=228), and it is
what makes N=325 (and beyond) tractable (B7).

---

## 12. Training, config, and reproducibility

- Loss: POT (adopted); optimizer AdamW + cosine annealing + early stopping; `knn_mode` is a
  `ModelConfig` switch (`exact` default, `scalable` for large N).
- Defaults: `d=64` (~99K params) or `d=128` (~315K); `k=8`; `n_gat_heads=4`; `n_temporal_layers=2`.
- Seeds: 10 (air/weather), 6 (traffic, scalable path); paired t-tests; mean+-std reported.
- Every reported number -> a committed `results/*.json` via a specific `experiments/*.py`; see
  `RESULTS-INDEX.md`. Figures regenerate from the JSONs via `experiments/make_figures.py`.

---

## 13. Worked micro-example of split conformal (to cement intuition)

Suppose `n = 19` calibration residuals (ug/m3), sorted, and `alpha = 0.10`:
```
level = (1 - 0.10) * (1 + 1/19) = 0.90 * 1.0526 = 0.9474
q_hat = the 0.9474-quantile of the 19 residuals  (-> effectively the ~19th order statistic)
```
For a new station with prediction `hat_y = 120`, interval = `[120 - q_hat, 120 + q_hat]`. If, across
many unseen stations and days, the truth lands inside about 90% (or a bit more) of the time, coverage
is satisfied — which is exactly what `tab:conformal` reports (0.915 at target 0.90). The slight
over-coverage is the finite-sample `(1 + 1/n)` inflation doing its job.
```

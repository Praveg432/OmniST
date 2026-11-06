# Statement of Novelty and Significance

## What is new

The novelty is the combination, not any single ingredient. Conformal prediction, graph attention, and
coordinate-based embeddings all exist. What has not been shown before is that a forecaster can be built
so that its per-node computation never references a global node count, and that this single structural
property is enough to make an inductive conformal coverage guarantee (1) hold at sensor coordinates with
no training history, and (2) survive the addition or removal of sensors without recalibration in the
far-field case.

We make this precise with a locality proposition: because each node's nonconformity score depends only
on its own coordinates, features, and k-nearest neighbors, dropping or adding any non-neighbor leaves
the score exactly unchanged. The split-conformal threshold therefore stays valid with no recalibration
under such changes. This is a stronger statement than the drift-correction that message-passing GNNs
require, where any node change perturbs every embedding.

## How it differs from the closest prior work

- Fixed-topology STGCNs (e.g. the STGCN family and E-STGCN) cannot emit a prediction at an unseen node
  at all. We confirm this as a concrete runtime failure under identical conditions.
- Inductive kriging (IGNNK) predicts at unseen nodes but offers no validity guarantee; in our
  experiments OmniST is both more accurate at unseen stations and the only one of the two that holds
  its target coverage.
- The 2024-2026 line on conformal prediction for dynamic and temporal graphs establishes validity as a
  graph's edges or labels evolve, typically on a fixed node set. Our setting is different: coverage at
  spatially unseen coordinates, under node addition and removal.

## Why it matters

Operators of real sensor networks routinely face added, failed, or offline sensors. A forecast at a new
or recently restored sensor is only actionable if it carries a trustworthy error bar. OmniST provides
exactly that, with a finite-sample, distribution-free guarantee, and without retraining. The method is
validated on four real datasets spanning environmental pollution, meteorology, and transportation,
which indicates the guarantee is a property of the construction rather than of any one domain.

## Honest scope

The contribution is a capability, not a point-accuracy record. OmniST matches rather than beats a
comparable fixed-topology model on in-distribution accuracy. Coverage is certified to 325 real sensors;
reaching thousands awaits a larger real benchmark, not an architectural change. The guarantee is
marginal rather than fully conditional, with a partial Mondrian fix. The two learned baselines are
faithful reimplementations, not the original authors' code. All of this is stated in the manuscript.

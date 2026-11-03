# OmniST — Study & Reference Docs

Three progressively deeper documents explaining the OmniST paper, model, experiments, and the
concepts behind them. Read in order.

1. **[01_OmniST_Complete_Overview.md](01_OmniST_Complete_Overview.md)** — the project overview:
   the problem, the idea, the architecture at a glance, the experiment arc, the design decisions and
   how known limitations are addressed, and the findings. Start here.

2. **[02_Foundational_Concepts.md](02_Foundational_Concepts.md)** — every underlying concept from
   first principles: spatiotemporal data, graphs/adjacency, inductive vs transductive, attention/GAT,
   Haversine distance, coordinate encodings, GRUs, conformal prediction + exchangeability + ACI, the
   baselines, Mondrian, and EVT/POT. Read for *why* each piece works.

3. **[03_Math_and_Implementation_Deep_Dive.md](03_Math_and_Implementation_Deep_Dive.md)** — the
   equations and the actual implemented mechanics, keyed to the source files: full forward pass with
   tensor shapes, the attention math, split-conformal with the finite-sample quantile and its proof,
   the locality Proposition (statement + proof + verification + the metric bug), online conformal
   (rolling / ACI / rolling-ACI), Mondrian, POT loss, the scalable k-NN equivalence, and a worked
   numerical example.

These docs are internal study/reference material. The authoritative evidence map is the top-level
`RESULTS-INDEX.md`; the paper itself is in `paper/OmniST-Paper-v2.tex`.

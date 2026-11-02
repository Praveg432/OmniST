# OmniST — Results Index (every paper number → its source)

Maps each claim/table in the paper to the result JSON that produced it and the script that produced
that JSON. All numbers are reproducible: `pip install -r requirements.txt`, then run the script.
Headline datasets are real; see `src/data_*.py` for provenance.

## Datasets (all real)
| Domain | Type | Source | Loader |
|--------|------|--------|--------|
| Delhi air quality | environmental | CPCB (E-STGCN repo), 37 stations, daily PM2.5 | `src/data_real.py` |
| Beijing air quality | environmental | UCI PRSA, 12 stations, daily PM2.5 | `src/data_beijing.py` |
| GHCN weather | meteorology | NOAA GHCN-Daily, 30 western-US stations, daily TMAX | `src/data_weather.py` |
| PeMSD7-M traffic | transportation | Caltrans PeMS District 7, 228 sensors, 5-min speed | `src/data_traffic_real.py` |
| PeMS-BAY traffic (large-N) | transportation | Caltrans PeMS Bay Area, 325 sensors, 5-min speed | `src/data_pemsbay.py` |

## Claim → source map
| Claim / table | Value(s) | JSON | Script |
|---------------|----------|------|--------|
| Headline 5-seed inductive MAE (Delhi) | 28.2 ± 0.9 µg/m³ | `results/exp_optimizations.json` (opt1) | `experiments/08_optimizations.py` |
| Monthly MAE table (Delhi 2023) | avg 31.0 µg/m³ | `results/exp_real_cpcb_final.json` | — (single representative run) |
| Sensor-failure resilience (≤1.1% @ 40% removal) | see table | `results/exp_real_cpcb_final.json` | — |
| Inductive point MAE: OmniST vs NN vs aug | 26.7 / 30.3 / 27.1 | `results/exp_optimizations.json` (opt3) | `experiments/08_optimizations.py` |
| **Inductive OmniST vs learned IGNNK kriging (Delhi+weather, 5 seeds)** | Delhi MAE 26.7±1.1 vs 34.0±1.3, cov 0.915 vs 0.891, width 125.7 vs 154.4; weather MAE 4.99 vs 5.92, cov 0.917 vs 0.947 | `results/exp_ignnk_inductive.json` | `experiments/B9_ignnk_inductive.py` (model: `src/ignnk_baseline.py`) |
| Inductive conformal @ unseen (Delhi, 3-seed) | 0.915±0.006, width 128.5 vs NN 149.7 | `results/exp_inductive_conformal.json` | `experiments/06_inductive_conformal.py` |
| **3-type cross-domain coverage (4 real datasets)** | Delhi 0.914 / Beijing 0.928 / weather 0.919 / traffic 0.915 (6-seed) | `results/exp_multidomain_v2.json` | `experiments/B1_multidomain_v2.py` |
| **Fixed-topology crash-under-dropout head-to-head** | E-STGCN fails (RuntimeError) under 30/50% dropout on Delhi+weather; OmniST holds cov Delhi 0.914/0.913, weather 0.925/0.927 | `results/exp_fixedtopo_crash_demo.json` | `experiments/B8_fixedtopo_crash_demo.py` |
| **Baseline breadth (OmniST vs NN/IDW/meanK/GP)** | OmniST only method ≥0.90 on all 4 | `results/exp_baselines.json` | `experiments/B2_baselines.py` |
| **Calibration curves + size-stratified** | per-domain target-tracking; high-stratum under-covers | `results/exp_calibration.json` | `experiments/B4_calibration.py` |
| **Ablation (full/no-GAT/k4/k16)** | coverage + inductive-MAE | `results/exp_ablations_v2.json` | `experiments/B3_ablations.py` |
| Temporal-drift fix (rolling-ACI) | static 0.76/0.50 → rolling-ACI 0.89/0.96 | `results/exp_temporal_conformal.json` | `experiments/10_temporal_conformal.py` |
| Conditional coverage (Mondrian) | traffic worst-group 0.80 → 0.86 | `results/exp_mondrian_conformal.json` | `experiments/13_mondrian_conformal.py` |
| Scalable k-NN (equivalence + speed) | cosine 0.99996, 2.3× @ N=1824 | `results/exp_approx_knn.json` | `experiments/14_approx_knn_test.py` |
| Multi-target / conditional (3 real) | 9-cell target tracking | `results/exp_realmultidomain_rigor.json` | `experiments/11_realmultidomain_rigor.py` |
| Scalability cost ladder (honest caveat) | train 169→1106s at N=228→912 | `results/exp_scalability_stress.json` | `experiments/12_scalability_stress.py` |
| **Locality proposition (exact invariance + preserved coverage)** | far-change diff 8.9e-8; coverage in band 4/4 domains | `results/exp_theorem_check.json` | `experiments/B5_theorem_check.py` |
| E-STGCN comparison (contextual, published numbers) | MAE parity | `results/exp_final_5seed.json`, `results/exp04_comparison.json` | `experiments/04/05_*.py` |
| **Controlled E-STGCN head-to-head (real, 5 seeds, identical conditions)** | transductive MAE Delhi 28.6±1.2 vs 33.5±1.9 (p=0.03), weather 5.55±0.14 vs 5.30±0.29 (p=0.14 n.s.); inductive: E-STGCN structurally cannot predict at unseen nodes | `results/exp_estgcn_controlled.json` | `experiments/B6_estgcn_controlled.py` |
| **Large-N real inductive coverage (PeMS-BAY, 325 real sensors)** | coverage 0.921±0.000 (3 seeds); holds dropout 30/50% 0.921/0.922; NN 0.870; scalable k-NN | `results/exp_largeN_pemsbay.json` | `experiments/B7_largeN_pemsbay.py` |

## Figures (all regenerated from the JSONs above)
`python experiments/make_figures.py` → `results/figures/`:
- `fig_multidomain_coverage.png` ← `exp_multidomain_v2.json`
- `fig_temporal_drift.png` ← `exp_temporal_conformal.json`
- `fig_mondrian.png` ← `exp_mondrian_conformal.json`
- `fig_calibration.png` ← `exp_calibration.json`
- `fig_scalable_knn.png` ← `exp_approx_knn.json`

## Honest notes
- Traffic multidomain coverage is now 6-seed (0.915±0.008), all six seeds on the scalable k-NN path
  (homogeneous; the scalable path is numerically equivalent to exact, cosine 0.99996). Air quality +
  weather are 10-seed. The B2 baselines table (`exp_baselines.json`) traffic value (0.911) is a
  separate run and is not affected by the multidomain re-run.
- Traffic sensor coordinates are reconstructed via MDS from the released distance adjacency (the speed
  series itself is the original real data); see `src/data_traffic_real.py`.
- The scalability ladder's coverage column beyond N=228 is NOT a coverage claim (naive node duplication
  breaks exchangeability) — only the runtime scaling is cited.
- E-STGCN comparison exists at two levels: (a) their *published* numbers (contextual, monthly table),
  and (b) a *controlled* head-to-head on real data under identical seeds/preprocessing/budget against
  a faithful reimplementation of the E-STGCN architecture (Chebyshev-GCN + LSTM + POT) — not the
  authors' original code. See `experiments/B6_estgcn_controlled.py`.
- Large-N real coverage is certified to 325 real sensors (full PeMS-BAY); PeMS-BAY coordinates are
  reconstructed via MDS from the released distance adjacency (speed series is original). Certifying
  coverage at *thousands* of nodes still awaits a larger real benchmark (none fetchable here).

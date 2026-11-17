# OmniST: Topology-Invariant Forecasting for Dynamic Sensor Networks With Conformal Intervals at Unseen Sensors

> **Status (17 Nov 2026): research work, not thesis work.** The M.Tech thesis is `thesis-omnist/` (title and abstract submitted and finalised 17 Nov 2026). This is the standalone OmniST paper repository. Its work is also carried into the thesis as Phase 1 (`thesis-omnist/phase1-omnist/`); this repository itself is research and paper work.


Reference implementation and experiment suite.

**Author:** Praveg Vashishtha  
**Affiliation:** Department of Computer Science and Engineering, IIT Patna | Senior SDE, SBS India  
**Contact:** praveg_pa2503mth259@iitp.ac.in  
**Preprint (earlier version):** [arXiv:2611.04256](https://arxiv.org/abs/2611.04256)  
**Code:** [github.com/Praveg432/OmniST](https://github.com/Praveg432/OmniST)

> **Status of the claims.** This repository now accompanies a revised manuscript. The earlier version
> (arXiv:2611.04256 and the first journal submission) described the intervals as having a "guaranteed"
> coverage at unseen sensors. That claim is **withdrawn** as stated: the proposition offered for it showed
> only that a sensor's error is unchanged by far-away topology changes, which does not make calibration and
> new-sensor errors exchangeable. The revision replaces it with a theorem whose assumptions are stated,
> tests it on real data, and corrects several other points. See [`paper/REVISION-NOTES.md`](paper/REVISION-NOTES.md).

## What OmniST is

A forecaster with no structural dependence on the number of sensors `N`: each sensor is embedded from its
coordinates, spatial context comes from k-nearest-neighbor attention recomputed at every forward pass, and
each sensor's history goes through a GRU on its own. It forecasts at a sensor that was not in the training
network, provided that sensor has its own input window.

## What the revised paper claims

1. **Locality lemma.** With neighbors drawn from the training network only, a new sensor's forecast, and
   so its error, depends on its own data and its `k` nearest training-network sensors alone. Scoring many new
   sensors in one pass equals scoring each alone (`tests/test_reference_scoring.py`).
2. **Sensor-level coverage theorem.** If a new sensor and its calibration sensors are exchangeable by random
   assignment, a correction counted in calibration *sensors* (not forecasts) gives expected coverage of at
   least `1 - alpha` at the new sensor. The interval is finite only with at least `1/alpha - 1` calibration
   sensors, so a 12-station network cannot support it at 90%.
3. **Evidence, in two settings.** On five real networks (12 to 325 sensors), over repeated random sensor
   splits and thousands of random calibration/test assignments per model:
   - *Same-window calibration* (the theorem's setting): the bound held in every finite case at a width cost of
     1.1 to 1.6 times the pooled rule with 11 to 89 calibration sensors. The pooled rule has no guarantee; it
     averaged 0.896 to 0.901 at the 0.90 target, but for up to 13% of random new sensors it covered less than 0.80.
   - *Forecast-time calibration, per sensor* (errors of earlier windows only; deployable): every held-out sensor
     as the new sensor. Without adaptation the worst sensor is covered 0.53 to 0.81 (pooled rule). Per-sensor
     adaptive conformal inference (unclipped, Theorem 2: deterministic long-run bound for each individual sensor,
     checked for all 940 cases) raises the worst sensor to 0.893 / 0.887 on the traffic networks, certified at
     0.845 / 0.814, at no larger median width, with about 3% of windows infinite. It is weak with short histories:
     Delhi (6 windows) reaches 0.855 to 0.877. Sensors farthest from the training network fared like the rest
     (`experiments/D5_per_sensor_guarantee.py`).
   Prior art: treating the sensor as a group is hierarchical conformal prediction (Lee, Barber, Willett;
   arXiv:2306.06342). With equal points per sensor the sensor-level rule is exactly HCP's threshold
   (`tests/test_hcp_equivalence.py`), so the rule is not claimed as new. Generalized HCP (Mallick et al.,
   arXiv:2608.15500) and sequential whitening were not evaluated.
   Against seasonal-naive forecasting: OmniST is better on Delhi, Beijing and PeMSD7-M, ties on weather, and is
   worse on PeMS-BAY (MAE 4.68 against 3.56; `experiments/D4_seasonal_comparison.py`).
4. **Controls.** A coordinate-free GRU shared across sensors matches OmniST's accuracy at new sensors within
   about 2%, and persistence matches it on weather and PeMSD7-M. The ability to forecast at a new sensor comes
   from sharing parameters across sensors; coordinates and attention show no measured benefit.
5. **A defect, fixed.** The coordinate embedding used centering constants chosen for Delhi for every dataset.
   Normalizing from the training network removes a 0.7 to 6.0 unit accuracy loss on the other regions. The old
   behavior is the default of `ModelConfig.coord_center/coord_scale` so earlier results reproduce.

Withdrawn from the earlier version: the 12% and 21% accuracy gains over nearest-neighbor interpolation and the
IGNNK kriging network (those methods use neighbors' concurrent values, a different task), the "no
recalibration needed" statement, and the drift result of 0.50 to 0.96 (does not reproduce under the new
protocol).

## Install and reproduce

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests                                   # 28 tests: locality lemma, HCP identity, coordinate frame
python3 experiments/D1_sensor_level_scores.py all         # per-sensor errors, original coordinate constants
python3 experiments/D1_sensor_level_scores.py all norm    # same splits, coordinates normalized
python3 experiments/D2_analyse.py                         # same-window theorem check, accuracy -> results/exp_D2_analysis.json
python3 experiments/D3_deployable.py                      # static/expanding/rolling at forecast time -> results/exp_D3_deployable.json
python3 experiments/D5_per_sensor_guarantee.py            # per-sensor ACI, Theorem 2 check, farthest sensors -> results/exp_D5_per_sensor.json
python3 experiments/D1c_seasonal_naive.py all             # seasonal-naive errors on the same splits -> results/D1c
python3 experiments/D4_seasonal_comparison.py             # OmniST vs seasonal-naive -> results/exp_D4_seasonal.json
python3 experiments/P2_coordinate_range.py                # coordinate input ranges (docs/THESIS-COORDINATE-DEFECT.md)
python3 experiments/P0_heterogeneity_pilot.py             # pilot: per-sensor scale normalization (not in the paper)
python3 experiments/P1_chronological_calibration.py       # pilot: same-window vs earlier-window calibration
python3 experiments/make_figures_v2.py
python3 paper/make_supplement_tables.py && python3 paper/assemble_jsen.py
python3 paper/check_numbers.py                            # table numbers vs result files
python3 paper/build_preview.py                            # preview PDF
```

`paper/check_numbers.py` parses the printed tables in the manuscript and compares every number with the
result file it came from. The random-split evaluation (Table III) is `experiments/C1_*.py`; the distance
analysis is `C2_*.py` and `C3_*.py`. The raw per-sensor error arrays are in `results/D1` and `results/D1b`
(about 45 MB); they are the inputs to D2 and take roughly an hour to regenerate.

## Layout

```
src/                   model (coordinate tokenizer, k-NN attention with optional reference restriction, GRU),
                       loaders, trainer, conformal utilities
experiments/D1_*       per-sensor error collection under the sensor-level protocol
experiments/D2_*       theorem check, accuracy, drift, report
experiments/C1-C3_*    random-split evaluation and distance analyses
experiments/B*, 0*_    earlier-protocol experiments, kept as a record (see RESULTS-INDEX.md)
paper/                 manuscript parts, assembler, preview build, upload package, supplement
results/               result JSONs, per-sensor arrays, figures
```

## Honest scope

The guarantee is design-based: it is over which sensor is the new one, not for a named sensor, a time step,
or a sensor outside the network, and real new sensors are not random draws. Baselines in the earlier
experiments (E-STGCN, IGNNK) are the author's reimplementations. Targets include imputed values; traffic
coordinates are reconstructed from distance matrices; the largest network has 325 sensors. Delhi has 10
evaluation windows. A head-to-head comparison with other inductive forecasting methods under the forecasting
task has not been run.

## Citation

The current public preprint is arXiv:2611.04256, whose wording on guaranteed coverage is superseded by this
repository. Cite the preprint for the model and this repository for the corrected claims.

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

MIT, see [`LICENSE`](LICENSE).

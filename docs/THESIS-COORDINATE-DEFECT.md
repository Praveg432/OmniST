# Defect report: Delhi-specific coordinate normalization in `thesis-omnist`

Prepared 17 Nov 2026 from `omnist-mamba` (commit d6188b5 and later). For handling in `thesis-omnist` separately.
Nothing in `thesis-omnist` was changed.

## 1. The defect

The spatial embedding normalizes every station's coordinates with constants chosen for Delhi, whatever the
dataset:

```python
# thesis-omnist/phase1-omnist/src/cc_hash.py, SpatialMLP.forward, lines 48-49
coords_centered[..., 0] = (coords[..., 0] - 28.6) / 0.2   # lat: center=28.6, scale=0.2
coords_centered[..., 1] = (coords[..., 1] - 77.1) / 0.3   # lon: center=77.1, scale=0.3
```

The same two lines are in `thesis-omnist/phase2-omnist-scale/src/omnist_scale/omnist/cc_hash.py` (lines 48-49;
the file is identical to the Phase 1 copy). Phase 2's `scalable_model.py` imports this `CCHashTokenizer`, so every
Phase 2 model inherits it. Copies also exist in two other workspace projects, outside the thesis:
`adaptive-st-conformal/src/ast_conformal/omnist/cc_hash.py` (retired) and
`omnist-scale/src/omnist_scale/omnist/cc_hash.py`.

`thesis-omnist/phase1-omnist/src` (model, attention, temporal block, trainer, loss, all five loaders) is
byte-identical to `omnist-mamba/src` at commit a462ddd, the version the measurements below call "legacy". The
numbers below therefore transfer to the thesis code as it stands.

## 2. Why it matters: input range per network

Delhi maps to about [-1, 1], as intended. Every other network feeds the MLP inputs tens to hundreds of times
larger (`omnist-mamba/experiments/P2_coordinate_range.py`, `results/exp_P2_coordinate_range.json`):

| Network | Stations | Latitude | Longitude | Legacy MLP input, lat | Legacy MLP input, lon | Largest absolute input |
|---|---|---|---|---|---|---|
| Delhi PM2.5 | 37 | 28.47 to 28.82 | 76.93 to 77.32 | -0.6 to 1.1 | -0.6 to 0.7 | 1.1 |
| Beijing PM2.5 | 12 | 39.88 to 40.33 | 116.18 to 116.66 | 56.4 to 58.6 | 130.3 to 131.9 | 132 |
| NOAA weather | 30 | 32.58 to 44.56 | -119.69 to -110.71 | 19.9 to 79.8 | -656 to -626 | 656 |
| PeMSD7-M | 228 | 34.0 to 34.5 | -118.3 to -117.8 | 27.0 to 29.5 | -651 to -650 | 651 |
| PeMS-BAY | 325 | 34.0 to 34.5 | -118.3 to -117.8 | 27.0 to 29.5 | -651 to -650 | 651 |

Between-station differences survive (each axis is still affine), but they ride on a constant offset of hundreds of
units through a GELU MLP, followed by an additive combination with the time and feature embeddings and a
LayerNorm. Normalizing with the training stations' mean and standard deviation instead gives inputs of at most
about 2 to 3 in absolute value on every network.

**A separate oddity, not part of this defect.** The two traffic loaders reconstruct positions from the distance
matrix by multidimensional scaling and then place them in a box near Los Angeles (34.0, -118.3), including PeMS-BAY,
which is in the San Francisco Bay Area (`src/data_traffic_real.py` line 69, `_coords_from_adjacency`, which
`src/data_pemsbay.py` line 19 imports and reuses). Relative distances are what matter, so this is harmless once coordinates are normalized, but
traffic coordinates should not be described as geographic.

## 3. Measured effect

Same random sensor splits, seeds and training budget; only the coordinate normalization differs. "Normalized" uses
the mean and standard deviation of the training network's coordinates, fixed before any held-out sensor is scored.
Held-out sensors are scored with neighbors drawn from the training network; 10 splits (Delhi, Beijing, weather) or 4
(traffic). Source: `omnist-mamba/results/exp_D2_analysis.json` (from `results/D1` and `results/D1b`).

**Point accuracy at held-out sensors (MAE).** Paired difference is normalized minus legacy, with a 95% t-interval
over splits.

| Network | Legacy | Normalized | Normalized minus legacy | Legacy better in | Coordinate-free GRU | Persistence |
|---|---|---|---|---|---|---|
| Delhi PM2.5 | 28.05 | 28.53 | +0.48 [0.24, 0.73] | 9 of 10 | 27.97 | 37.98 |
| Beijing PM2.5 | 49.61 | 48.85 | -0.77 [-1.10, -0.43] | 0 of 10 | 48.43 | 52.73 |
| NOAA weather | 5.718 | 4.749 | -0.97 [-1.83, -0.10] | 0 of 10 | 4.816 | 4.650 |
| PeMSD7-M | 9.756 | 3.659 | -6.10 [-6.67, -5.53] | 0 of 4 | 3.715 | 3.570 |
| PeMS-BAY | 5.170 | 4.488 | -0.68 [-0.76, -0.61] | 0 of 4 | 4.447 | 5.140 |

With the legacy constants, OmniST is worse than a coordinate-free GRU on every network except Delhi, by 0.72 to
6.04 MAE units. The thesis's own C1 results show the same pattern: C1 test MAE is 10.33 on PeMSD7-M (persistence
3.57), 5.84 on weather (4.65) and 5.54 on PeMS-BAY (5.14), that is, worse than repeating the last value on three
networks.

**Intervals (90% target, pooled rule, largest calibration size, same-window calibration).**

| Network | Coverage, legacy | Coverage, normalized | Width, legacy | Width, normalized |
|---|---|---|---|---|
| Delhi PM2.5 | 0.8994 | 0.8997 | 125.7 | 127.6 |
| Beijing PM2.5 | 0.9004 | 0.9003 | 180.3 | 179.0 |
| NOAA weather | 0.8963 | 0.8965 | 24.56 | 20.28 |
| PeMSD7-M | 0.9006 | 0.9003 | 47.97 | 17.60 |
| PeMS-BAY | 0.8997 | 0.9010 | 20.97 | 19.38 |

Coverage is unchanged (within 0.0013): conformal calibration absorbs the extra error. The cost lands on width, 2.7
times wider on PeMSD7-M and 21% wider on weather.

## 4. What it affects in `thesis-omnist`

| Item | Affected | How |
|---|---|---|
| Submitted abstract: coverage 87.6% to 90.6% (C1) | Probably not | Coverage moved by at most 0.0013 in the measurements above. Not verified on C1 itself |
| C1 test MAE and interval widths (Beijing, weather, PeMSD7-M, PeMS-BAY) | Yes | Inflated as in section 3. Delhi is in range and changes little |
| C1b chronological re-run (in progress, uses the same models) | Yes for MAE and width; coverage probably not | Same mechanism |
| Any comparison against persistence, seasonal-naive or interpolators (plan S4, contribution D, gate G-B1) | Yes, materially | The model's error is inflated by up to 2.7 times on PeMSD7-M |
| Phase 2 finding "intervals 21% wider than seasonal-naive on PeMS-BAY" | Partly | In `omnist-mamba` the fix narrows the PeMS-BAY pooled width by 8% (20.97 to 19.38), and seasonal-naive is still narrower there (15.6 against 20.5 on the windows with a seasonal forecast). The finding probably survives in direction; its size must be re-measured on the Phase 2 model |
| Phase 2 V1, V3, V3b, V4 (scalable model, same tokenizer) | Accuracy and width yes; equivalence results (V2) no | V2 compares exact and fast attention on the same tokens, so it is unaffected |
| C2 (coverage against distance), C3 (score shift) | Possibly | Scores change scale per network; slopes and total-variation estimates should be re-checked on weather after the fix |
| `tools/check_submission_text.py`, `check_draft_abstract.py` | Only if figures are updated | They tie abstract numbers to C1 files |

## 5. The plan's done-criterion for S5 will fail

`THESIS-COMPLETION-PLAN.md` lists P1-3 / S5 (local coordinate frame) with "done when: tests pass; Phase 1 numbers
unchanged within noise", and runs Delhi and weather as a sanity check. On the evidence above, MAE and width will
**not** be unchanged on weather (MAE -17%, width -17%), and Delhi will get slightly worse (+0.48 MAE, 9 of 10
splits). Suggested criterion: coverage unchanged within 0.01 on every network; MAE and width reported before and
after, expected to fall on all networks except Delhi. Delhi and weather alone will not show the size of the problem;
include PeMSD7-M (the largest effect) in the sanity run if possible.

## 6. Reference fix (as implemented in `omnist-mamba`)

- `src/cc_hash.py`: `SpatialMLP(d_model, hidden_dim, center=(28.6, 77.1), scale=(0.2, 0.3))` stores `center` and
  `scale` as registered buffers and computes `(coords - center) / scale`. The defaults reproduce the legacy formula
  exactly. `CCHashTokenizer` passes `coord_center` and `coord_scale` through.
- `src/omnist_model.py`: `ModelConfig.coord_center` and `ModelConfig.coord_scale`, defaults equal to the legacy
  constants, passed to the tokenizer.
- Call site (`experiments/D1_sensor_level_scores.py`, `norm` mode): centre = mean, scale = standard deviation (plus
  1e-6) of the **training** stations' coordinates of each split. Calibration and test stations must not enter the
  statistics.
- Tests (`tests/test_coordinate_frame.py`): the default frame reproduces the legacy formula exactly; a frame set
  from the training network is invariant to shifting all coordinates; the frame is saved and restored with the model.

Porting notes:
1. The buffers are persistent, so a state dict saved before the change will not load with `strict=True` (missing
   `center` and `scale`). The thesis code only loads state dicts within a run (`trainer.py` line 121, early
   stopping) and in tests that build both models fresh, so nothing breaks today. If old checkpoints are ever
   reloaded, use `strict=False` and set the frame explicitly.
2. Keeping the legacy constants as the default keeps every committed result reproducible. Switch new runs to the
   training-network frame explicitly and record which frame produced each result file.
3. Port `phase1-omnist/src/cc_hash.py` and `phase2-omnist-scale/src/omnist_scale/omnist/cc_hash.py` together, so
   Phase 1 and Phase 2 stay comparable.
4. `phase1-omnist/tests/test_cc_hash.py` uses coordinates near Delhi, so it passes before and after and will not
   detect the defect. Add a test with non-Delhi coordinates.

## 7. Reproducing the numbers above

In `omnist-mamba`:
```bash
python3 experiments/P2_coordinate_range.py            # input ranges (section 2)
python3 experiments/D1_sensor_level_scores.py all     # legacy constants -> results/D1
python3 experiments/D1_sensor_level_scores.py all norm  # training-network frame, same splits -> results/D1b
python3 experiments/D2_analyse.py                     # results/exp_D2_analysis.json (section 3)
```
The per-sensor score arrays are committed, so `D2_analyse.py` alone regenerates section 3 in under a minute.

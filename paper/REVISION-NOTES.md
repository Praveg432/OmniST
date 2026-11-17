# Revision Notes: IEEE Sensors Journal Resubmission

Decision letter: Sensors-117291-2026, rejected. The Associate Editor's stated reason: the claim of guaranteed
coverage at unseen sensors is not established, because Proposition 1 proves only local score invariance, which
does not establish exchangeability between calibration and unseen-sensor scores. The journal will send a reworked
paper to review only if it shows noticeable change. The central claim, theory, protocol, baselines and several
headline numbers are different.

**The reviewers' individual comments were not available**; only the Associate Editor's summary. Section 6 has a
slot for them.

Source of truth: this repository (`omnist-mamba`, branch `main`). `thesis-omnist` was consulted for findings only.

## 1. Status

| Item | State |
|---|---|
| Manuscript | `paper/jsen-submission/OmniST-JSEN.tex`, assembled from parts by `paper/assemble_jsen.py`; 8 pages in the preview, within the journal's 8-page guideline (no overlength charge) |
| Supplement | `paper/OmniST-Sensors-Supplement.tex` / `.pdf`, 7 pages; tables generated from result files |
| Upload package | `paper/OmniST-JSEN-upload.zip` (9 files, 26 references) from `paper/make_jsen_upload.py` |
| Tests | 25 pass (`python3 -m pytest tests`) |
| Number check | `python3 paper/check_numbers.py`: 164 printed numbers (tables and key prose claims) match the result files. Mutation-tested; it also caught one rounding error in the text, now fixed |
| Preview build | `paper/build_preview.py` (Tectonic cannot embed the template's EPS logo, so the preview blanks it; the upload package keeps the original template files and the portal compiles them) |

## 2. The Associate Editor's point

| Problem | Change |
|---|---|
| Proposition 1 does not give exchangeability | Replaced by Lemma 1 (locality) and Theorem 1 (sensor-level coverage) with a proof. The rule equals hierarchical conformal prediction's (HCP) threshold for equal-size groups; this is stated in the paper and checked in `tests/test_hcp_equivalence.py`. The proof is the conformal risk control argument. Neither is claimed as new |
| Calibration sensors were the training sensors | Calibration and test sensors are held out of training and early stopping; neighbors are searched in the training network only (tested) |
| Points treated as exchangeable | Correction counted in calibration sensors; finite only with at least `1/alpha - 1` sensors |
| "Guaranteed" | Removed from title, abstract and contributions; assumptions A1 to A4 stated |
| "No recalibration needed" | Withdrawn |

## 3. Results

- **Same-window calibration (the theorem's setting):** bound held in every finite case at 90% (0.911 to 0.9996),
  at 1.1 to 1.6 times the pooled width with 11 to 89 calibration sensors; vacuous on the 12-station network.
  The pooled rule averaged 0.896 to 0.901 but covered up to 13% of random new sensors below 0.80.
- **Forecast-time calibration** (earlier windows only, deployable, no theorem): sensor-level rule with rolling
  recalibration 0.971 (weather), 0.913 (PeMSD7-M), 0.899 (PeMS-BAY), 0.853 (Delhi). Pooled rule short on three of
  four networks.
- **Accuracy:** coordinate-free shared GRU within about 2% of OmniST; persistence ties on weather and PeMSD7-M;
  seasonal-naive beats OmniST on PeMS-BAY (MAE 3.56 vs 4.68, OmniST intervals 31% wider), is worse on Delhi,
  Beijing and PeMSD7-M, and ties on weather.
- **Coordinate defect** (Delhi constants for every dataset) found, fixed, and reported; earlier-code numbers kept.

## 4. Withdrawn from the earlier version

12% and 21% gains over interpolation and IGNNK (different task); "no recalibration needed"; drift result 0.50 to
0.96 (does not reproduce); E-STGCN monthly table and controlled comparison (motivation, not evidence).

## 5. Open items (not blocking resubmission)

1. **Reviewer comments**: answer each when available (section 6).
2. **Other public versions still carry withdrawn claims**: arXiv:2611.04256, `paper/OmniST-Workshop.*`,
   `paper/OmniST-Paper-v2.tex`, `paper/OmniST.pdf`. Consider an arXiv v2 before or with resubmission.
3. **Not evaluated**: Generalized HCP, HCP's second-moment variant, sequential whitening; other inductive
   forecasters under the forecasting task. Stated as limitations.
4. **Thesis code** (`thesis-omnist`) still has the coordinate defect; its traffic conclusion against
   seasonal-naive was measured with it.
5. **Per-sensor score arrays** (`results/D1`, `D1b`, `D1c`, about 47 MB) are committed so every number can be
   re-derived without retraining.

## 6. Reviewer comments (to complete)

| # | Comment | Where addressed | Status |
|---|---|---|---|
| AE | Proposition 1 gives only local score invariance | Lemma 1, Theorem 1, Sections II-C, IV-A, IV-B | addressed |
| R1.. | *not available* | | |
| R2.. | *not available* | | |

## 7. Cover-letter text (draft)

> This manuscript is a substantially new version of Sensors-117291-2026. We accept the Associate Editor's point
> that local score invariance does not establish exchangeability, and we withdraw the claim of guaranteed
> coverage. The revision (i) states and proves a sensor-level coverage result with explicit assumptions and
> identifies it with hierarchical conformal prediction; (ii) tests it on five real networks over repeated random
> sensor splits, reporting its cost, where it is vacuous, and how it behaves when calibration is restricted to
> information available at forecast time; (iii) replaces comparisons that mixed forecasting with reconstruction by
> like-for-like baselines, including seasonal-naive forecasting and a coordinate-free control, which shows that the
> coordinates and attention add no measured accuracy; and (iv) reports and corrects a coordinate-normalization
> defect in our earlier code. Several earlier headline numbers are withdrawn.

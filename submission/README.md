# Journal Submission Package: OmniST to IEEE Sensors Journal

Target: **IEEE Sensors Journal**, regular paper, traditional (non-open-access) route.
Author guide: https://ieee-sensors.org/ieee-sensors-journal/for-authors/

## Files

| File | Purpose |
|---|---|
| `../paper/jsen-submission/OmniST-JSEN.tex` / `.pdf` | **Submission manuscript**, in the official IEEE Sensors Journal template (6 pages) |
| `../paper/OmniST-Sensors.tex` / `.pdf` | Same content in plain IEEEtran (5 pages); source for the JSEN version |
| `../paper/OmniST-Sensors-Supplement.tex` / `.pdf` | Supplementary material (3 pages): monthly table, ablation, calibration, Mondrian, cost scaling |
| `../paper/references.bib` | Bibliography (cited entries checked against arXiv/MDPI records on 2026-11-06) |
| `../results/figures/graphical_abstract_jsen.png` / `.pdf` | Mandatory graphical abstract (compact; sits in the template's abstract box), drawn from committed result JSONs by `../experiments/make_graphical_abstract.py` |
| `../results/figures/graphical_abstract.png` / `.pdf` | Full three-panel version (for the portal upload or slides) |
| `../results/figures/fig_multidomain_coverage.png`, `fig_temporal_drift.png` | The two figures used in the manuscript |
| `COVER-LETTER.md` | Cover letter for IEEE Sensors Journal |
| `HIGHLIGHTS.md`, `NOVELTY-STATEMENT.md` | Not required by IEEE; kept for reuse |

The arXiv version (`../paper/OmniST-Paper-v2.tex`) is unchanged in content and remains the long-form
reference.

## Journal rules that shaped this version

- **8 free pages.** Pages beyond 8 carry a mandatory charge of $175 each. The manuscript is 6 pages in
  the Sensors Journal template, which leaves room for revisions.
- **Voluntary page charge** of $110/page may be requested after acceptance. It is voluntary; decline it
  to keep the submission no-fee.
- **Abstract:** 150 to 250 words, one paragraph, no displayed math or tables. Current: 211 words.
- **Graphical abstract:** mandatory.
- **Template:** the journal-specific Sensors Journal template from the IEEE Template Selector.
- **Keywords:** pick one category from the journal's keyword list in the submission portal.
- **No parallel submission.** Duplicate submission leads to rejection and a one-year ban.

## Remaining steps

- [x] Official template downloaded and manuscript ported (`paper/jsen-submission/`).
- [ ] **Upload the LaTeX package.** The portal wants source: run `python3 paper/make_jsen_upload.py`
      to build `paper/OmniST-JSEN-upload.zip` (10 flat files: manuscript, `.bbl`, `.bib`, three figures, IEEE
      class/style/logo). It was test-compiled from an empty folder. The zip and folder are not in git,
      because they contain IEEE's template files.
- [ ] **Rebuild the final PDF with pdfLaTeX or Overleaf.** The PDF in the repo was made with a preview
      build (XeTeX, Times replaced by a clone, journal logo left blank) because this machine has no
      pdfLaTeX. Upload `paper/jsen-submission/` together with the IEEE template files from
      `paper/jsen-template/` to Overleaf and compile with pdfLaTeX; that PDF is the one to submit.
      Check that it is still at most 8 pages and that the header logo appears.
- [x] AI-use statement confirmed. [x] Funding statement confirmed (no specific grant).
- [ ] **Author bio:** IEEE prints a short bio; a photo can be added at the final-files stage.
- [ ] **Keyword category** in the portal: the closest fit is sensor networks / sensor data processing.
- [ ] **Suggested reviewers** (optional, can speed up review): researchers in conformal prediction or
      spatiotemporal sensor forecasting at other institutions, with no collaboration ties.
- [ ] **Submit** through the IEEE Author Portal with the manuscript PDF and source, supplementary
      material, graphical abstract, and cover letter.

## arXiv overlap

The preprint (arXiv:2611.04256) is cited in the cover letter as a longer version under a different
title. IEEE permits prior posting on arXiv. Do not submit this work to any other venue while it is
under review.

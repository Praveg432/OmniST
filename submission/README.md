# Journal Submission Package — OmniST

This folder holds everything needed to submit the OmniST manuscript to a journal. The manuscript
itself is `../paper/OmniST-Paper-v2.tex` (+ `references.bib` + the four figures).

## Files here
- `COVER-LETTER.md` — editor cover letter (fill in the bracketed journal name and date).
- `HIGHLIGHTS.md` — Elsevier-style highlights (5 bullets, each <=85 chars; verified).
- `NOVELTY-STATEMENT.md` — statement of novelty and significance (for the editor / significance field).

## Choose the venue first — it sets the format

The manuscript is currently in `IEEEtran` (IEEE journal class).

### Option A — IEEE journal (least work)
Targets: IEEE Internet of Things Journal, IEEE Trans. Intelligent Transportation Systems,
IEEE Trans. Neural Networks and Learning Systems (a stretch), IEEE Sensors Journal.
- Format: no change needed; the `.tex` already uses `\documentclass[journal]{IEEEtran}`.
- Submit: manuscript PDF + source; cover letter; (highlights optional).
- Check the specific journal's scope and page/overlength policy before submitting.

### Option B — Neurocomputing (Elsevier)
- Format change required: switch to `\documentclass[review]{elsarticle}`, move the author block to
  the elsarticle `\author`/`\affiliation` macros, and use `\begin{abstract}...\end{abstract}` +
  `\begin{keyword}...\end{keyword}` (elsarticle style). The body text, tables, equations, and
  bibliography carry over unchanged (elsarticle supports BibTeX; switch `\bibliographystyle` to a
  numeric Elsevier style such as `elsarticle-num`).
- Required extras: the Highlights file (here) and a declaration of interest statement (none).
- CRediT author statement: single author, so "Praveg Vashishtha: conceptualization, methodology,
  software, validation, writing."

Recommendation: if minimizing effort and risk, submit to an IEEE venue (Option A) — zero reformatting.
If targeting Neurocomputing specifically, I can produce the elsarticle version on request.

## arXiv overlap (both options)
A preprint exists at arXiv:2611.04256. Most IEEE and Elsevier journals permit preprints; state in the
cover letter (already drafted) that this submission extends the preprint, and cite it. Do not submit to
two venues at once.

## Pre-submission checklist
- [ ] Pick venue; if Elsevier, convert to elsarticle (ask and I will do it).
- [ ] Fill journal name + date in the cover letter.
- [ ] One `pdflatex` + `bibtex` + `pdflatex` x2 pass (or Overleaf) to produce the PDF and `.bbl`.
- [ ] Confirm all four figures render and citations resolve.
- [ ] Verify the `nonexchcptgnn2025` full author list in `references.bib` (currently "Wang, Zhang,
      and others"; confirmed submitter Tuo Wang, KDD 2025 — fill the complete list).
- [ ] Confirm the repository is public and the URL in the paper resolves.
- [ ] Suggested reviewers (optional field): consider authors active in conformal prediction and
      spatiotemporal GNNs; avoid anyone with a conflict.

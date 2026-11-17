# paper/

**Current manuscript:** `jsen-submission/OmniST-JSEN.tex` (IEEE Sensors Journal resubmission), assembled from
`jsen-submission/part*.tex`, `abstract.txt` and `setup_table.tex` by `assemble_jsen.py`. Supplement:
`OmniST-Sensors-Supplement.tex`. Upload package: `OmniST-JSEN-upload.zip` (built by `make_jsen_upload.py`).
What changed and why: `REVISION-NOTES.md`.

Build and check:
```bash
python3 paper/make_supplement_tables.py && python3 paper/assemble_jsen.py
python3 paper/check_numbers.py      # printed numbers vs result files
python3 paper/build_preview.py      # preview PDF (template logo blanked; the portal compiles the real one)
python3 paper/make_jsen_upload.py   # upload zip
```

**Superseded, kept as a record.** These carry claims that the current manuscript withdraws (guaranteed coverage,
12% and 21% gains over interpolation and IGNNK, "no recalibration needed", the drift result 0.50 to 0.96). Do not
quote them.

| File | What it is |
|---|---|
| `OmniST-Paper-v2.tex`, `OmniST.pdf` | Source and PDF of the arXiv preprint 2611.04256 |
| `OmniST-Workshop.tex`, `OmniST-Workshop.pdf` | Condensed workshop version of the preprint |
| `archive/OmniST-JSEN-v1-rejected.*` | First Sensors Journal submission (Sensors-117291-2026, rejected) |
| `archive/OmniST-Sensors-v1-IEEEtran.*`, `archive/OmniST-Sensors-Supplement-v1.tex` | Earlier journal drafts |
| `RESCOPE-FOR-PREPRINT.md`, `TIER1-FINALIZATION-PLAN.md`, `RELATED-WORK-LANDSCAPE.md` | Planning notes for the preprint |

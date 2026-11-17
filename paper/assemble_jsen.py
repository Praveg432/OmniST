#!/usr/bin/env python3
"""Assemble paper/jsen-submission/OmniST-JSEN.tex from its parts.

The manuscript is kept as parts so that a table or section can be edited in isolation:
  part1_front.tex      header, introduction, method, lemma, theorem        (@@ABSTRACT@@, @@RESULTS@@)
  abstract.txt         abstract text
  part2_protocol.tex   datasets and protocol                               (@@SETUP_TABLE@@)
  setup_table.tex      Table of networks and splits
  part3_results.tex    results                                             (@@LIMITS@@)
  part4_related.tex    related work
  part5_end.tex        limitations, conclusion, availability, bibliography

Run: python3 paper/assemble_jsen.py     Output: paper/jsen-submission/OmniST-JSEN.tex
"""
from pathlib import Path

D = Path(__file__).resolve().parent / "jsen-submission"


def read(name):
    return (D / name).read_text()


front = read("part1_front.tex").replace("@@ABSTRACT@@", read("abstract.txt").strip())
protocol = read("part2_protocol.tex").replace("@@SETUP_TABLE@@", read("setup_table.tex").strip())
results = read("part3_results.tex").replace("@@LIMITS@@", read("part4_related.tex") + "\n" + read("part5_end.tex"))
text = front.replace("@@RESULTS@@", protocol + "\n" + results)
leftover = [m for m in ("@@ABSTRACT@@", "@@RESULTS@@", "@@SETUP_TABLE@@", "@@LIMITS@@") if m in text]
assert not leftover, f"unreplaced markers: {leftover}"
(D / "OmniST-JSEN.tex").write_text(text)
print("wrote", D / "OmniST-JSEN.tex", len(text.split()), "words")

#!/usr/bin/env python3
"""Build the self-contained LaTeX upload package for the IEEE Sensors Journal portal.

The portal wants one flat set of source files. The repo layout keeps figures and the bibliography in
sibling folders, so this script gathers everything into paper/jsen-upload/ with paths rewritten, adds
the IEEE template files (downloaded separately into paper/jsen-template/, not stored in git), tries to
generate the .bbl, and zips the result.

Run from anywhere:  python3 paper/make_jsen_upload.py
Output:             paper/jsen-upload/  and  paper/OmniST-JSEN-upload.zip
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
FIGS = PAPER.parent / "results" / "figures"
TEMPLATE = PAPER / "jsen-template"
SRC = PAPER / "jsen-submission" / "OmniST-JSEN.tex"
OUT = PAPER / "jsen-upload"
ZIP = PAPER / "OmniST-JSEN-upload.zip"
MAIN = "OmniST-JSEN"

IEEE_FILES = ["ieeecolor.cls", "jsen.sty", "LOGO-jsen-web.eps", "IEEEtran.bst"]
FIGURES = ["graphical_abstract_jsen.png", "fig_multidomain_coverage.png", "fig_temporal_drift.png"]


def fail(msg):
    sys.exit(f"error: {msg}")


def main():
    for name in IEEE_FILES:
        if not (TEMPLATE / name).exists():
            fail(f"{name} missing from {TEMPLATE}. Download the Sensors Journal LaTeX template from "
                 "https://template-selector.ieee.org/ and unzip it there.")

    tex = SRC.read_text()
    tex, n1 = re.subn(r"\\graphicspath\{.*\}\n", "\\\\graphicspath{{./}}\n", tex)
    tex, n2 = re.subn(r"\\bibliography\{\.\./references\}", r"\\bibliography{references}", tex)
    if (n1, n2) != (1, 1):
        fail(f"expected to patch graphicspath and bibliography once each, got {n1} and {n2}")
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    unknown = used - set(FIGURES)
    if unknown:
        fail(f"manuscript uses figures not in the package list: {sorted(unknown)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for name in IEEE_FILES:
        shutil.copy(TEMPLATE / name, OUT / name)
    for name in FIGURES:
        shutil.copy(FIGS / name, OUT / name)
    shutil.copy(PAPER / "references.bib", OUT / "references.bib")
    (OUT / f"{MAIN}.tex").write_text(tex)

    make_bbl()
    make_zip()


def make_bbl():
    """Generate the .bbl with Tectonic (BibTeX output is engine independent). Optional."""
    if shutil.which("tectonic") is None:
        print("note: tectonic not found, so no .bbl was generated; the portal can build it from the .bib")
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in OUT.iterdir():
            shutil.copy(f, work / f.name)
        # Preview-only patches so XeTeX can compile: blank logo box, Times clone. The .bbl does not
        # depend on either, and the files in the package are left untouched.
        cls = (work / "ieeecolor.cls").read_text()
        cls = re.sub(r"\\includegraphics\[width=\\logowidth\]\{\\logoname\.eps\}", r"\\hbox to \\logowidth{\\hfil}", cls)
        (work / "ieeecolor.cls").write_text(cls)
        lines = (work / f"{MAIN}.tex").read_text().split("\n")
        i = next(k for k, line in enumerate(lines) if line.startswith("\\documentclass"))
        lines.insert(i + 1, "\\usepackage{fontspec}\\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,"
                            "BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]")
        (work / "_t.tex").write_text("\n".join(lines))
        res = subprocess.run(["tectonic", "-X", "compile", "_t.tex", "--keep-intermediates"],
                             cwd=work, capture_output=True, text=True)
        bbl = work / "_t.bbl"
        if res.returncode != 0 or not bbl.exists():
            print("warning: could not generate .bbl:\n" + res.stderr[-600:])
            return
        shutil.copy(bbl, OUT / f"{MAIN}.bbl")
        print(f"wrote {MAIN}.bbl ({bbl.read_text().count('bibitem')} references)")


def make_zip():
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f.name)
    names = sorted(f.name for f in OUT.iterdir())
    print(f"wrote {ZIP.name}: {len(names)} files")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a preview PDF of paper/jsen-submission/OmniST-JSEN.tex with Tectonic.

Tectonic cannot embed the template's EPS logo, so the preview is compiled from a temporary copy in
which the logo box is blank and a Times clone is used. The source files are not modified. The journal
portal compiles the real package (see make_jsen_upload.py), so page breaks in the preview can differ a
little from the final typeset PDF.

Run: python3 paper/build_preview.py      Output: paper/jsen-submission/OmniST-JSEN.pdf
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SRC = PAPER / "jsen-submission"
FIGS = PAPER.parent / "results" / "figures"
MAIN = "OmniST-JSEN"
FIGURES = ["graphical_abstract_jsen.png", "fig_sensor_level.png"]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("ieeecolor.cls", "jsen.sty", "IEEEtran.bst", "LOGO-jsen-web.eps"):
            shutil.copy(SRC / name, work / name)
        for name in FIGURES:
            shutil.copy(FIGS / name, work / name)
        shutil.copy(PAPER / "references.bib", work / "references.bib")
        cls = (work / "ieeecolor.cls").read_text()
        cls = re.sub(r"\\includegraphics\[width=\\logowidth\]\{\\logoname\.eps\}",
                     r"\\hbox to \\logowidth{\\hfil}", cls)
        (work / "ieeecolor.cls").write_text(cls)
        tex = (SRC / f"{MAIN}.tex").read_text()
        tex = re.sub(r"\\graphicspath\{.*\}\n", "\\\\graphicspath{{./}}\n", tex)
        tex = tex.replace("\\bibliography{../references}", "\\bibliography{references}")
        lines = tex.split("\n")
        i = next(k for k, line in enumerate(lines) if line.startswith("\\documentclass"))
        lines.insert(i + 1, "\\usepackage{fontspec}\\setmainfont{texgyretermes}[Extension=.otf,"
                            "UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,"
                            "BoldItalicFont=*-bolditalic]")
        (work / f"{MAIN}.tex").write_text("\n".join(lines))
        res = subprocess.run(["tectonic", "-X", "compile", f"{MAIN}.tex", "--keep-logs"],
                             cwd=work, capture_output=True, text=True)
        log = work / f"{MAIN}.log"
        pdf = work / f"{MAIN}.pdf"
        if log.exists():
            text = log.read_text(errors="ignore")
            for pat in ("Undefined control sequence", "LaTeX Error", "undefined", "Citation .* undefined",
                        "Reference .* undefined", "Missing"):
                hits = [l for l in text.split("\n") if re.search(pat, l)]
                if hits:
                    print(f"[{pat}] {len(hits)} line(s):", *hits[:6], sep="\n   ")
        if res.returncode != 0 or not pdf.exists():
            print(res.stderr[-1500:])
            sys.exit("build failed")
        shutil.copy(pdf, SRC / f"{MAIN}.pdf")
        print("wrote", SRC / f"{MAIN}.pdf")


if __name__ == "__main__":
    main()

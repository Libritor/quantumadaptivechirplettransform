"""Build main.pdf (pdflatex + bibtex) and report pages, overfull boxes, undefined refs/citations."""
import pathlib, re, subprocess, sys

PAPER = pathlib.Path(__file__).resolve().parents[1]


def run(*cmd):
    return subprocess.run(" ".join(cmd), cwd=PAPER, capture_output=True, text=True, errors="replace", shell=True)


steps = (("pdflatex", "-interaction=nonstopmode", "main"), ("bibtex", "main"),
         ("pdflatex", "-interaction=nonstopmode", "main"), ("pdflatex", "-interaction=nonstopmode", "main"))
for st in steps:
    r = run(*st)
    if st[0] == "pdflatex" and r.returncode != 0:
        log = (PAPER / "main.log").read_text(errors="replace")
        errs = [l for l in log.splitlines() if l.startswith("!")]
        print("pdflatex reported errors:", errs[:8])
        i = log.find("!")
        print(log[i:i + 1500])
        break
log = (PAPER / "main.log").read_text(errors="replace")
over = [l for l in log.splitlines() if l.startswith("Overfull")]
undef = [l for l in log.splitlines() if "undefined" in l.lower() and ("Warning" in l or "Citation" in l)]
pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
print("pages:", pages.group(1) if pages else "?", "| overfull:", len(over), "| undefined:", len(undef))
for l in over[:12]:
    print("  ", l[:110])
for l in undef[:20]:
    print("  ", l[:140])
try:
    from pypdf import PdfReader
    rd = PdfReader(str(PAPER / "main.pdf"))
    for i, pg in enumerate(rd.pages, 1):
        t = pg.extract_text() or ""
        first = t.strip().split("\n")[0][:60] if t.strip() else ""
        print(f"  page {i}: {len(t)} chars | {first}")
except Exception as e:
    print("pypdf:", e)

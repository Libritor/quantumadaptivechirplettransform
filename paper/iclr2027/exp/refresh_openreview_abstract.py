"""Regenerate the abstract text in openreview_fields.md from main.tex with macros expanded."""
import pathlib
import re

P = pathlib.Path(__file__).resolve().parents[1]
tex = (P / "main.tex").read_text(encoding="utf-8")
abs_ = tex[tex.index(r"\begin{abstract}") + len(r"\begin{abstract}"):tex.index(r"\end{abstract}")].strip()
num = {}
for f in ("numbers.tex", "numbers_denoise.tex"):
    if (P / f).exists():
        num.update(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", (P / f).read_text(encoding="utf-8")))
num.update({"qact": "QACT", "act": "ACT"})
txt = re.sub(r"\\([A-Za-z]+)(\\ |\{\})?", lambda m: num.get(m.group(1), m.group(0)) + (" " if m.group(2) == "\\ " else ""), abs_)
txt = txt.replace(r"\,", " ").replace("$", "").replace("--", "-").replace("~", " ").replace("\\emph{", "")
txt = re.sub(r"\\([a-zA-Z]+)", "", txt).replace("{", "").replace("}", "").replace("  ", " ")
p = P / "openreview_fields.md"
s = p.read_text(encoding="utf-8")
i = s.index("**Abstract.**")
j = s.index("**Authors (camera-ready only).**")
s = s[:i] + "**Abstract.** (generated from main.tex by exp/refresh_openreview_abstract.py)\n\n" + txt + "\n\n" + s[j:]
p.write_text(s, encoding="utf-8")
print(f"abstract refreshed: {len(txt.split())} words")
print(txt[:400])

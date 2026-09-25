#!/usr/bin/env python
"""Fold Khalil's denoising control arms into the paper.

Reads results/denoise_controls_iclr.json (his machine writes it; the schema is not fixed,
so this parser is tolerant), identifies the two arms
  A  classical fine-f + 3-point coordinate refiner
  B  classical fine-f, seed grid extended to 78 Hz
recomputes the paired Wilcoxon tests against QACT parity hw (1.213) and classical fine-f
(1.257) from per-subject scores when they are present, and writes numbers_denoise.tex, which
main.tex reads with \\InputIfFileExists. Nothing in main.tex needs editing.

    python paper/iclr2027/exp/fold_denoise_controls.py [path/to/json]
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "denoise_controls_iclr.json"
REF_Q, REF_C = 1.213, 1.257                       # QACT parity hw, classical fine-f (paper)


def fmt_p(p):
    if p is None:
        return "--"
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(math.floor(math.log10(max(p, 1e-300))))
    m = int(round(p / 10 ** e))
    if m == 10:
        m, e = 1, e + 1
    return f"{m}\\times10^{{{e}}}"


def walk(o, path=""):
    """Yield (path, dict) for every dict in the JSON tree."""
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")


def find_arm(d, patterns):
    """Find a dict whose path or 'name'/'engine'/'arm' matches all regexes."""
    hits = []
    for path, node in walk(d):
        label = " ".join(str(node.get(k, "")) for k in ("name", "engine", "arm", "label")) + " " + path
        if all(re.search(pt, label, re.I) for pt in patterns):
            hits.append((path, node))
    return hits


def scores_of(node):
    for k in ("per_subject", "scores", "score_per_subject", "held_out_scores", "subject_scores", "per_subject_score"):
        v = node.get(k)
        if isinstance(v, list) and len(v) >= 10 and all(isinstance(x, (int, float)) for x in v):
            return np.asarray(v, float)
        if isinstance(v, dict) and len(v) >= 10:
            try:
                return np.asarray([float(x) for x in v.values()])
            except Exception:
                pass
    return None


def mean_of(node, scores):
    for k in ("score", "mean_score", "mean", "tuning_score"):
        if isinstance(node.get(k), (int, float)):
            return float(node[k])
    return float(scores.mean()) if scores is not None else None


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} not found")
    d = json.load(open(SRC))
    arms = {
        "A": find_arm(d, [r"coord|hw|3.?point|three.?point"]),
        "B": find_arm(d, [r"78|extend|mains|band"]),
    }
    refs = {"Q": find_arm(d, [r"qact.*(parity|hw)"]), "C": find_arm(d, [r"classical.*(fine|0\.5)"])}
    for k, v in list(arms.items()) + list(refs.items()):
        print(k, "->", [p for p, _ in v][:4])
    macros = {}
    from scipy.stats import wilcoxon
    ref_scores = {k: (scores_of(v[0][1]) if v else None) for k, v in refs.items()}
    ref_means = {"Q": (mean_of(refs["Q"][0][1], ref_scores["Q"]) if refs["Q"] else REF_Q),
                 "C": (mean_of(refs["C"][0][1], ref_scores["C"]) if refs["C"] else REF_C)}
    for k, hits in arms.items():
        if not hits:
            print(f"arm {k}: NOT FOUND -- edit the patterns in this script")
            continue
        node = hits[0][1]
        s = scores_of(node)
        mu = mean_of(node, s)
        macros[f"denoiseArm{k}Score"] = f"{mu:.3f}" if mu is not None else "--"
        for r, refname in (("Q", "VsQ"), ("C", "VsC")):
            rs = ref_scores[r]
            if s is not None and rs is not None and len(s) == len(rs):
                better = int((s < rs).sum())
                p = float(wilcoxon(s, rs).pvalue) if np.any(s != rs) else 1.0
                macros[f"denoiseArm{k}{refname}Better"] = str(better)
                macros[f"denoiseArm{k}{refname}P"] = fmt_p(p)
                macros[f"denoiseArm{k}{refname}N"] = str(len(s))
            else:
                # fall back to whatever paired numbers the file carries
                for kk, vv in node.items():
                    if re.search(r"better", kk, re.I) and isinstance(vv, (int, float)):
                        macros.setdefault(f"denoiseArm{k}{refname}Better", str(int(vv)))
                    if re.search(r"^p$|p_?value|wilcoxon", kk, re.I) and isinstance(vv, (int, float)):
                        macros.setdefault(f"denoiseArm{k}{refname}P", fmt_p(float(vv)))
                macros.setdefault(f"denoiseArm{k}{refname}Better", "--")
                macros.setdefault(f"denoiseArm{k}{refname}P", "--")
                macros.setdefault(f"denoiseArm{k}{refname}N", "18")
        macros[f"denoiseArm{k}Delta"] = f"{mu - ref_means['Q']:+.3f}" if mu is not None else "--"
    macros["denoiseRefQ"] = f"{ref_means['Q']:.3f}"
    macros["denoiseRefC"] = f"{ref_means['C']:.3f}"
    macros["denoiseArmsPresent"] = "yes"
    out = PAPER / "numbers_denoise.tex"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"% generated by exp/fold_denoise_controls.py from {SRC.name}\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {out} with {len(macros)} macros:")
    for k, v in macros.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()

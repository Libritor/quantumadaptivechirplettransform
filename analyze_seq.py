#!/usr/bin/env python
"""Pre-registered analysis of train_seq.py results (see its docstring)."""
import glob, json
import numpy as np
from scipy.stats import wilcoxon

ALPHA = 0.05 / 4
MODELS = ("lstm", "qin_zz", "qin_z", "qlstm", "qlstm_twin")
R = [json.load(open(f)) for f in sorted(glob.glob("results/seq/chb*.json"))]
R = [r for r in R if all(m in r["models"] for m in MODELS)]
persons = [r["person"] for r in R]


def per_person(metric):
    out = {m: [] for m in MODELS}
    for r in R:
        for m in MODELS:
            v = [s[metric] for s in r["models"][m] if s[metric] is not None]
            out[m].append(np.mean(v) if v else np.nan)
    return {m: np.array(v) for m, v in out.items()}


skill, auc = per_person("skill"), per_person("auc")
print(f"{len(R)} persons: {', '.join(persons)}")
tot = lambda k: sum(r["info"][k] for r in R)
print(f"train {tot('train_h'):.0f} h / test {tot('test_h'):.0f} h; seizure onsets "
      f"train {tot('onsets_train')} / test {tot('onsets_test')}")
print(f"\n{'model':<11} {'forecast skill (mean / median)':>32}  {'risk AUC (mean, n)':>20}  "
      f"{'params':>7} {'core':>5}")
for m in MODELS:
    a = auc[m][~np.isnan(auc[m])]
    pr = np.mean([np.mean([s["params"] for s in r["models"][m]]) for r in R])
    core = R[0]["models"][m][0]["core_params"]
    print(f"{m:<11} {skill[m].mean():>+14.4f} / {np.median(skill[m]):+.4f}"
          f"{'':>8}  {a.mean() if len(a) else float('nan'):>12.3f} ({len(a):>2})  "
          f"{pr:>7.0f} {core if core is not None else '-':>5}")
print("\nforecast skill > 0 (beats persistence) for: " +
      ", ".join(f"{m} {int((skill[m] > 0).sum())}/{len(R)}" for m in MODELS))

print(f"\nPRE-REGISTERED DECISIONS (alpha = {ALPHA}):")
for q, c, label in (("qin_zz", "qin_z", "A: fixed circuit features"),
                    ("qlstm", "qlstm_twin", "B: trainable circuits in gates")):
    for name, M in (("forecast skill", skill), ("risk AUC", auc)):
        ok = ~np.isnan(M[q]) & ~np.isnan(M[c])
        d = M[q][ok] - M[c][ok]
        p = wilcoxon(M[q][ok], M[c][ok]).pvalue if ok.sum() > 1 and not np.allclose(d, 0) else 1.0
        win = d.mean() > 0 and p < ALPHA
        print(f"  {label:<31} {name:<15} quantum - twin = {d.mean():+.4f} "
              f"(better for {(d > 0).sum()}/{ok.sum()} persons), p = {p:.4f} -> "
              f"{'QUANTUM BENEFIT' if win else 'no demonstrated quantum benefit'}")

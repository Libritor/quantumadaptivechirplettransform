#!/usr/bin/env python
"""Iteration 4b: fairness check. In iteration 4 the quantum kernel's tuner chose the
SMALLEST bandwidth offered (s = 0.25) in 5 of 6 folds, so its optimum may lie below
the grid. Re-run Q with s in {0.03, 0.06, 0.12, 0.25} and compare, patient by patient,
with the saved accuracies of every other model (same split, same tuning)."""
import json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold
import qsearch_04_atomset_kernel as it4

atoms = np.load("results/atoms_chbmit.npy")
y = np.load("results/y_chbmit.npy")
g = np.load("results/g_chbmit.npy", allow_pickle=True)
subs = np.unique(g)
cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)
P, w = it4.atom_params(atoms)
kernels = {s: it4.set_kernel(P, w, "zz", s) for s in (0.03, 0.06, 0.12, 0.25)}
pred, chosen = it4.eval_precomputed(kernels, y, g, cv)
acc = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
prev = json.load(open("results/qsearch_04_atomset_kernel.json"))["acc"]
print(f"Q with extended bandwidth grid: {acc.mean() * 100:.2f}% per-patient; chosen {chosen}")
for k in ("Q", "Qp", "R", "F1", "F2", "F3"):
    b = np.array(prev[k])
    d = acc - b
    p = float(wilcoxon(acc, b).pvalue) if np.any(d != 0) else 1.0
    print(f"  Q(extended) - {k:3s} {d.mean() * 100:+6.2f} pts  better on {(d > 0).sum():2d}/23  p={p:.4f}")
json.dump(dict(acc=acc.tolist(), chosen=str(chosen)),
          open("results/qsearch_04b_bandwidth.json", "w"), indent=1)

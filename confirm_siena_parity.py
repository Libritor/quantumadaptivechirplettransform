#!/usr/bin/env python
"""Siena replication, rerun with the full-parity QACT engine.

confirm_siena.py compared classical ACT with the legacy QACT (plain matching
pursuit, unnormalised selection). QACT has since been brought to full functional
parity with the classical engine (OMP joint refit, exact-f update, backfit, the
post-selection-conditioned selection criterion). This reruns the same comparison
with the parity engine on the same epochs, same folds, same classifiers.

PRE-REGISTERED (written before running):
  cross-patient CV over the 14 Siena patients (StratifiedGroupKFold, 5 folds, seed 0),
  all features + regularisation, per-patient accuracy, paired Wilcoxon:
    T1  QACT parity           vs  classical ACT          does parity QACT beat classical?
    T2  QACT parity + amp     vs  classical ACT + amp    same, with amplitude features
    T3  QACT parity           vs  QACT legacy            did the parity work help?
  Two classifiers (logreg L2, svm_rbf) -> 6 tests, alpha = 0.05/6 = 0.00833.
  A quantum win needs T1 or T2 positive and significant.
"""
import json
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qbe.qact import QACT, features_batch

FS, N = 256.0, 512
ALPHA = 0.05 / 6
t0 = time.time()
X = np.load("results/Xraw_siena.npy")
y = np.load("results/y_siena.npy")
g = np.load("results/g_siena.npy", allow_pickle=True)
amp = np.load("results/amp_siena.npy")
E, C, _ = X.shape
print(f"[siena] {E} epochs x {C} channels, {len(np.unique(g))} patients", flush=True)

q = QACT(length=N, fs=FS, shots=256, device="cuda", seed=0, refine_steps=4,
         omp=True, backfit_passes=1, exact_f=True, norm_select=True)
flat = X.reshape(E * C, N).astype(np.float32)
feat = np.zeros((E * C, 13), np.float32)
for lo in range(0, len(flat), 256):
    par, err = q.transform(flat[lo:lo + 256], order=12, return_params=True)
    feat[lo:lo + len(par)] = features_batch(par, err, FS, N)
Xp = feat.reshape(E, C * 13)
np.save("results/X_siena_QACT_parity.npy", Xp)
print(f"[siena] parity QACT done ({time.time() - t0:.0f}s)", flush=True)

Xc = np.load("results/X_siena_classical_ACT.npy")
Xl = np.load("results/X_siena_QACT_sampled.npy")
SETS = {"classical ACT": Xc, "QACT legacy": Xl, "QACT parity": Xp,
        "classical ACT + amp": np.hstack([Xc, amp]), "QACT parity + amp": np.hstack([Xp, amp])}
MODELS = {"logreg L2": lambda: LogisticRegression(C=0.01, max_iter=5000),
          "svm_rbf": lambda: SVC(C=1.0, gamma="scale")}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
subs = np.unique(g)
acc = {}
for mname, mk in MODELS.items():
    for sname, XX in SETS.items():
        pipe = Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                         ("clf", mk())])
        pred = cross_val_predict(pipe, XX, y, groups=g, cv=cv)
        acc[(mname, sname)] = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
        print(f"{mname:<10} {sname:<20} per-patient {acc[(mname, sname)].mean() * 100:5.1f}%",
              flush=True)

print(f"\npaired Wilcoxon across {len(subs)} patients (alpha = {ALPHA:.5f}):")
tests = [("T1", "QACT parity", "classical ACT"),
         ("T2", "QACT parity + amp", "classical ACT + amp"),
         ("T3", "QACT parity", "QACT legacy")]
out = {}
for mname in MODELS:
    for tag, a_, b_ in tests:
        A, B = acc[(mname, a_)], acc[(mname, b_)]
        d = A - B
        p = float(wilcoxon(A, B).pvalue) if not np.allclose(d, 0) else 1.0
        verdict = ("SIGNIFICANT" if p < ALPHA and d.mean() > 0
                   else "(negative)" if d.mean() < 0 else "ns")
        print(f"  {mname:<10} {tag} {a_:<18} - {b_:<20} {d.mean() * 100:+5.2f} pts  "
              f"better {(d > 0).sum():>2}/{len(d)}  p={p:.4f}  {verdict}")
        out[f"{mname}:{tag}"] = dict(diff=float(d.mean()), p=p, better=int((d > 0).sum()))
json.dump({"per_patient": {f"{m}|{s}": v.tolist() for (m, s), v in acc.items()},
           "tests": out}, open("results/siena_parity.json", "w"), indent=2)
print(f"\nwrote results/siena_parity.json ({time.time() - t0:.0f}s)")

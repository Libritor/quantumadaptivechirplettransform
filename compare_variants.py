#!/usr/bin/env python
"""Do ideas #2 (proposal selection) and #3 (cubic/skew atoms) improve FEATURES?

They both improve ATOMS -- mean reconstruction error on planted signals falls
from 0.168 (quad, argmax) to 0.136 (cubic), 0.126 (skew), and 0.132 (proposal
k=32, which also beats argmax). This project has twice found that better
reconstruction does NOT imply better features, so that has to be tested, not
assumed.

Chirplet features only (no amplitude): amplitude features dominate selection and
would mask any difference between representations.

PRE-REGISTERED (written before running): per-patient accuracy, 6-fold
cross-patient CV, all 234 features with regularisation, paired Wilcoxon across
23 patients. Each new variant is tested against BOTH baselines:
    vs QACT quad argmax   -- does the new family/selection add anything?
    vs classical ACT      -- does it close the gap found in compare_qact_allfeat?
4 variants x 2 baselines x 2 classifiers = 16 tests, alpha = 0.05/16 = 0.003125.
"""
import json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ALPHA = 0.05 / 16
BASE = {"classical ACT": "chbmit", "QACT quad": "chbmitqactonly_ref_argmax"}
NEW = {"cubic": "chbmitqactonly_cubic", "skew": "chbmitqactonly_skew",
       "cubic+skew": "chbmitqactonly_cubicskew", "proposal k=32": "chbmitqactonly_prop32"}
MODELS = {"logreg L2": lambda: LogisticRegression(C=0.01, max_iter=5000),
          "svm_rbf": lambda: SVC(C=1.0, gamma="scale")}

y = np.load("results/y_chbmitamp.npy")
g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
subs = np.unique(g)
cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)
acc = {}

print(f"{'model':<10} {'feature set':<16} {'pooled':>8} {'per-patient':>12}")
for mname, mk in MODELS.items():
    for label, tag in {**BASE, **NEW}.items():
        X = np.load(f"results/X_{tag}.npy")
        assert (np.load(f"results/y_{tag}.npy") == y).all(), tag
        pipe = Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                         ("clf", mk())])
        pred = cross_val_predict(pipe, X, y, groups=g, cv=cv)
        acc[(mname, label)] = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
        print(f"{mname:<10} {label:<16} {(pred==y).mean()*100:>7.1f}% "
              f"{acc[(mname, label)].mean()*100:>11.1f}%", flush=True)

print(f"\npaired Wilcoxon across {len(subs)} patients (alpha = {ALPHA:.6f}):")
out = {}
for mname in MODELS:
    for nlabel in NEW:
        for blabel in BASE:
            A, B = acc[(mname, nlabel)], acc[(mname, blabel)]
            d = A - B
            p = wilcoxon(A, B).pvalue if not np.allclose(d, 0) else 1.0
            verdict = ("IMPROVES" if p < ALPHA and d.mean() > 0
                       else "(negative)" if d.mean() < 0 else "ns")
            print(f"  {mname:<10} {nlabel:<14} vs {blabel:<14} {d.mean()*100:+5.2f} pts "
                  f"better {(d>0).sum():>2}/{len(d)}  p={p:.4f}  {verdict}")
            out[f"{mname}:{nlabel}|{blabel}"] = dict(diff=float(d.mean()), p=float(p),
                                                     better=int((d > 0).sum()))
json.dump({"per_patient": {f"{m}|{l}": v.tolist() for (m, l), v in acc.items()},
           "tests": out}, open("results/qact_variants.json", "w"), indent=2)
print("\nwrote results/qact_variants.json")

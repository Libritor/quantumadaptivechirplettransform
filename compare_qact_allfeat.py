#!/usr/bin/env python
"""QACT vs classical ACT with ALL features and regularisation (no top-k selection).

Why this rerun: every earlier QACT-vs-ACT test used SelectKBest(k=20), and
`compare_feature_sets.py` already showed that univariate selection discards
chirplet features entirely (all 36 amplitude features outrank every chirplet
feature; the best chirplet feature ranks #37). Those tests therefore compared the
two transforms in the one regime where chirplet features barely matter. With all
234 features kept and regularisation doing the work, chirplets were worth up to
+8.6 points over amplitude alone -- so this is where a difference between the two
transforms should actually show up.

PRE-REGISTERED (written before running):
  per-patient accuracy, 6-fold cross-patient CV, 234 features, no selection;
  three regularised classifiers; paired Wilcoxon across 23 patients:
    A1  QACT sampled vs classical ACT        the headline question
    A2  QACT sampled vs QACT argmax          isolates quantum measurement
    A3  QACT argmax  vs classical ACT        isolates dictionary/formulation

PRE-REGISTERED ADDENDUM (written before running the parity engine, after QACT was
brought to full functional parity with the classical engine -- OMP joint refit,
exact-f update, backfit, two-width envelope, stopping rule, and the corrected
post-selection-conditioned selection criterion):
    A4  QACT parity sampled vs classical ACT      the headline question, with the
                                                  engine bookkeeping confound removed
    A5  QACT parity sampled vs QACT legacy sampled  does the parity work change
                                                  classification at all, or only
                                                  reconstruction
  alpha = 0.05 / 15 = 0.00333 (3 models x 5 comparisons).
  QSVC is excluded here because its qubit count equals its feature count, so
  "all features" is not a configuration it can take.
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

ALPHA = 0.05 / 15
SETS = {"classical ACT": "chbmit",
        "QACT argmax": "chbmitqactonly_ref_argmax",
        "QACT sampled": "chbmitqactonly_ref",
        "QACT parity argmax": "chbmitqactonly_parity_argmax",
        "QACT parity sampled": "chbmitqactonly_parity"}
MODELS = {
    "logreg L2": LogisticRegression(C=0.01, max_iter=5000),
    "logreg L1": LogisticRegression(C=0.05, penalty="l1", solver="saga", max_iter=3000),
    "svm_rbf": SVC(C=1.0, gamma="scale"),
}

y = np.load("results/y_chbmitamp.npy")
g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
subs = np.unique(g)
cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)

acc = {}
for mname, clf in MODELS.items():
    for sname, tag in SETS.items():
        X = np.load(f"results/X_{tag}.npy")
        assert (np.load(f"results/y_{tag}.npy") == y).all(), tag
        pipe = Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                         ("clf", clf)])
        pred = cross_val_predict(pipe, X, y, groups=g, cv=cv)
        acc[(mname, sname)] = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
        print(f"{mname:<10} {sname:<14} {X.shape[1]:>4} feats  pooled {(pred == y).mean()*100:5.1f}%"
              f"  per-patient {acc[(mname, sname)].mean()*100:5.1f}%", flush=True)

print(f"\npaired Wilcoxon across {len(subs)} patients (alpha = {ALPHA:.5f}):")
tests = [("A1", "QACT sampled", "classical ACT"), ("A2", "QACT sampled", "QACT argmax"),
         ("A3", "QACT argmax", "classical ACT"),
         ("A4", "QACT parity sampled", "classical ACT"),
         ("A5", "QACT parity sampled", "QACT sampled")]
out = {}
for mname in MODELS:
    for tag, a_, b_ in tests:
        A, B = acc[(mname, a_)], acc[(mname, b_)]
        d = A - B
        p = wilcoxon(A, B).pvalue if not np.allclose(d, 0) else 1.0
        verdict = ("SIGNIFICANT" if p < ALPHA and d.mean() > 0
                   else "(negative)" if d.mean() < 0 else "ns")
        print(f"  {mname:<10} {tag} {a_:<13} - {b_:<14} {d.mean()*100:+5.2f} pts  "
              f"better {(d>0).sum():>2}/{len(d)}  p={p:.4f}  {verdict}")
        out[f"{mname}:{tag}"] = dict(diff=float(d.mean()), p=float(p), better=int((d>0).sum()))
json.dump({"per_patient": {f"{m}|{s}": v.tolist() for (m, s), v in acc.items()}, "tests": out},
          open("results/qact_allfeat.json", "w"), indent=2)
print("\nwrote results/qact_allfeat.json")

#!/usr/bin/env python
"""Do cross-channel features add anything? (idea #5)

Per-channel features treat the 18 channels as independent. These 18 cross-channel
features measure spatial synchrony, spread and propagation -- the part of a
seizure's definition that per-channel aggregation cannot see.

PRE-REGISTERED (written before running): per-patient accuracy, 6-fold
cross-patient CV, all features with regularisation (no top-k selection, which is
known to discard chirplet features), paired Wilcoxon across 23 patients.

  C1  classical ACT + cross   vs  classical ACT
  C2  QACT + cross            vs  QACT
  C3  ACT + amplitude + cross vs  ACT + amplitude
  C4  QACT + amplitude + cross vs QACT + amplitude
Two classifiers (logreg L2, svm_rbf) -> 8 tests, alpha = 0.05/8 = 0.00625.
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

ALPHA = 0.05 / 8
PAIRS = [("C1", "chbmit_cc", "chbmit", "classical ACT"),
         ("C2", "chbmitqactonly_cc", "chbmitqactonly_ref", "QACT"),
         ("C3", "chbmitamp_cc", "chbmitamp", "classical ACT + amplitude"),
         ("C4", "chbmitqact_cc", "chbmitqact_ref", "QACT + amplitude")]
MODELS = {"logreg L2": lambda: LogisticRegression(C=0.01, max_iter=5000),
          "svm_rbf": lambda: SVC(C=1.0, gamma="scale")}

y = np.load("results/y_chbmitamp.npy")
g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
subs = np.unique(g)
cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)


def per_patient(tag, mk):
    X = np.load(f"results/X_{tag}.npy")
    assert (np.load(f"results/y_{tag}.npy") == y).all(), tag
    pipe = Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                     ("clf", mk())])
    pred = cross_val_predict(pipe, X, y, groups=g, cv=cv)
    return np.array([(pred[g == s] == y[g == s]).mean() for s in subs]), X.shape[1]


out = {}
print(f"{'model':<10} {'feature set':<28} {'feats':>6} {'per-patient':>12}")
acc = {}
for mname, mk in MODELS.items():
    for tag_, with_, without_, label in PAIRS:
        for tag in (without_, with_):
            if (mname, tag) not in acc:
                acc[(mname, tag)], nf = per_patient(tag, mk)
                print(f"{mname:<10} {tag:<28} {nf:>6} {acc[(mname, tag)].mean()*100:>11.1f}%",
                      flush=True)

print(f"\npaired Wilcoxon across {len(subs)} patients (alpha = {ALPHA:.5f}):")
for mname in MODELS:
    for tag_, with_, without_, label in PAIRS:
        A, B = acc[(mname, with_)], acc[(mname, without_)]
        d = A - B
        p = wilcoxon(A, B).pvalue if not np.allclose(d, 0) else 1.0
        verdict = ("CROSS-CHANNEL HELPS" if p < ALPHA and d.mean() > 0
                   else "(negative)" if d.mean() < 0 else "ns")
        print(f"  {mname:<10} {tag_} {label:<26} {d.mean()*100:+5.2f} pts  "
              f"better {(d>0).sum():>2}/{len(d)}  p={p:.4f}  {verdict}")
        out[f"{mname}:{tag_}"] = dict(diff=float(d.mean()), p=float(p),
                                      better=int((d > 0).sum()))
json.dump({"per_patient": {f"{m}|{t}": v.tolist() for (m, t), v in acc.items()},
           "tests": out}, open("results/crosschannel.json", "w"), indent=2)
print("\nwrote results/crosschannel.json")

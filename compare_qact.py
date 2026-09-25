#!/usr/bin/env python
"""Does the QUANTUM chirplet transform give better features than the classical one?

Pre-registered (written before the paired tests were run; the single logreg
number that motivated it -- 60.2% vs 57.7% -- is not the test):

  Per-patient accuracy from 6-fold cross-patient CV (StratifiedGroupKFold),
  for three classifiers x three feature sets, paired Wilcoxon across 23 patients:

    P1  QACT (sampled)  vs  classical ACT          the headline question
    P2  QACT (sampled)  vs  QACT (argmax)          isolates the QUANTUM step:
                                                   same dictionary, same atoms,
                                                   classical selection instead of
                                                   measurement sampling
    P3  QACT (argmax)   vs  classical ACT          isolates the DICTIONARY
                                                   (22,784 in-band atoms, no
                                                   off-grid refinement)

  alpha = 0.05 / 9 = 0.00556. A quantum benefit requires P1 AND P2 to favour the
  sampled version; if only P3 is significant, the gain is the dictionary, not the
  quantum measurement.
"""
import json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from qbe.quantum import make_pipeline

ALPHA = 0.05 / 9
import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", choices=("grid", "refined", "parity"), default="grid",
                 help="grid = dictionary only; refined = parameter-shift refined atoms "
                      "(the apples-to-apples comparison, since classical ACT also refines)")
_A = _ap.parse_args()
# parity = the same tests on features from the full-parity engine (OMP joint refit,
# exact-f, backfit, post-selection-conditioned selection), added after the parity
# work; the tests and alpha are unchanged
SETS = {"grid": {"classical ACT": "chbmit", "QACT argmax": "chbmitqactonly_argmax",
                 "QACT sampled": "chbmitqactonly"},
        "refined": {"classical ACT": "chbmit", "QACT argmax": "chbmitqactonly_ref_argmax",
                    "QACT sampled": "chbmitqactonly_ref"},
        "parity": {"classical ACT": "chbmit", "QACT argmax": "chbmitqactonly_parity_argmax",
                   "QACT sampled": "chbmitqactonly_parity"}}[_A.variant]
print(f"variant: {_A.variant}\n")
# (name, n features kept, kwargs). QSVC gets 8 because n_qubits == n features:
# 20 features would mean 20 qubits (a 2^20 statevector per sample), which is not
# a slow configuration but an infeasible one. 8 qubits is what every other QSVC
# in this project used. Its kernel runs through qbe.quantum_fast (GPU, verified
# identical to qiskit to 1e-14) rather than qiskit's CPU path.
MODELS = (("logreg", 20, dict(C=0.1)), ("svm_rbf", 20, dict(C=1.0)), ("qsvc", 8, dict(C=1.0)))

y = np.load("results/y_chbmitamp.npy")
g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
subs = np.unique(g)
cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)

acc = {}
def build(mname, k, kw):
    if mname != "qsvc":
        return make_pipeline(mname, k, **kw)
    from sklearn.feature_selection import VarianceThreshold, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from qbe.quantum import SafeSelectKBest
    from qbe.quantum_fast import FidelityKernelSVC
    return Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                     ("reduce", SafeSelectKBest(f_classif, k=k)),
                     ("angles", MinMaxScaler(feature_range=(0.0, 0.3 * np.pi))),
                     ("clf", FidelityKernelSVC(n_qubits=k, layers=2,
                                               entanglement="linear", **kw))])


for mname, k, kw in MODELS:
    for sname, tag in SETS.items():
        X = np.load(f"results/X_{tag}.npy")
        assert (np.load(f"results/y_{tag}.npy") == y).all(), tag
        pred = cross_val_predict(build(mname, k, kw), X, y, groups=g, cv=cv)
        acc[(mname, sname)] = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
        print(f"{mname:<8} {sname:<14} pooled {(pred == y).mean()*100:5.1f}%  "
              f"per-patient {acc[(mname, sname)].mean()*100:5.1f}%", flush=True)

print(f"\npaired Wilcoxon across {len(subs)} patients (alpha = {ALPHA:.5f}):")
tests = [("P1", "QACT sampled", "classical ACT"), ("P2", "QACT sampled", "QACT argmax"),
         ("P3", "QACT argmax", "classical ACT")]
out = {}
for mname, _k, _kw in MODELS:
    for tag, a_, b_ in tests:
        A, B = acc[(mname, a_)], acc[(mname, b_)]
        d = A - B
        p = wilcoxon(A, B).pvalue if not np.allclose(d, 0) else 1.0
        sig = "SIGNIFICANT" if p < ALPHA and d.mean() > 0 else ("(negative)" if d.mean() < 0 else "ns")
        print(f"  {mname:<8} {tag} {a_:<13} - {b_:<14} {d.mean()*100:+5.2f} pts  "
              f"better {(d>0).sum():>2}/{len(d)}  p={p:.4f}  {sig}")
        out[f"{mname}:{tag}"] = dict(diff=float(d.mean()), p=float(p),
                                     better=int((d > 0).sum()))
fn = f"results/qact_comparison_{_A.variant}.json"
json.dump({"per_patient": {f"{m}:{s}": v.tolist() for (m, s), v in acc.items()},
           "tests": out}, open(fn, "w"), indent=2)
print(f"\nwrote {fn}")

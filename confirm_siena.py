#!/usr/bin/env python
"""Independent confirmation on Siena: does the CHB-MIT ordering replicate?

CHB-MIT found, with all features and regularisation (the fair regime):
    classical ACT >= QACT argmax >= QACT sampled
and amplitude features worth ~+10 points on top of chirplets.

Siena is a different hospital, different patients, monopolar recording resampled
to the same 18 bipolar pairs -- nothing about it has been looked at while
developing any of this.

PRE-REGISTERED (written before running):
  cross-patient CV over the 14 Siena patients (StratifiedGroupKFold, 5 folds),
  all features + regularisation, per-patient accuracy, paired Wilcoxon:
    S1  classical ACT + amplitude  vs  amplitude only     does ACT add?
    S2  classical ACT              vs  QACT sampled       does the CHB-MIT
    S3  classical ACT + amplitude  vs  QACT + amplitude   ordering replicate?
  Two classifiers (logreg L2, svm_rbf) -> 6 tests, alpha = 0.05/6 = 0.00833.
  Replication is claimed only if the sign matches CHB-MIT; significance on 14
  patients is a bonus, not the bar.
"""
import json, time
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qbe.acquire import CHBMIT_CHANNELS
from qbe.features import feature_names
from qbe.gpu_features import extract_gpu
from qbe.qact import QACT, features_batch
from qbe.types import EpochSet

FS, N = 256.0, 512
t0 = time.time()
X = np.load("results/Xraw_siena.npy")
y = np.load("results/y_siena.npy")
g = np.load("results/g_siena.npy", allow_pickle=True)
amp = np.load("results/amp_siena.npy")
ep = EpochSet(X=X, y=y, channels=CHBMIT_CHANNELS, fs=FS,
              label_names=("interictal", "ictal"))
E, C, _ = X.shape
print(f"[siena] {E} epochs x {C} channels, {len(np.unique(g))} patients", flush=True)

Xc, _ = extract_gpu(ep, order=12, verbose=False)
print(f"[siena] classical ACT done ({time.time()-t0:.0f}s)", flush=True)
q = QACT(length=N, fs=FS, shots=256, device="cuda", seed=0, refine_steps=4)
flat = X.reshape(E * C, N).astype(np.float32)
feat = np.zeros((E * C, 13), np.float32)
for lo in range(0, len(flat), 256):
    par, err = q.transform(flat[lo:lo + 256], order=12, return_params=True)
    feat[lo:lo + len(par)] = features_batch(par, err, FS, N)
Xq = feat.reshape(E, C * 13)
print(f"[siena] QACT done ({time.time()-t0:.0f}s)", flush=True)

SETS = {"amplitude only": amp,
        "classical ACT": Xc,
        "QACT sampled": Xq,
        "classical ACT + amp": np.hstack([Xc, amp]),
        "QACT + amp": np.hstack([Xq, amp])}
for tag, XX in SETS.items():
    np.save(f"results/X_siena_{tag.replace(' ', '_').replace('+','plus')}.npy", XX)

MODELS = {"logreg L2": lambda: LogisticRegression(C=0.01, max_iter=5000),
          "svm_rbf": lambda: SVC(C=1.0, gamma="scale")}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
subs = np.unique(g)
acc = {}
print(f"\n{'model':<10} {'feature set':<22} {'feats':>6} {'pooled':>8} {'per-patient':>12}")
for mname, mk in MODELS.items():
    for sname, XX in SETS.items():
        pipe = Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                         ("clf", mk())])
        pred = cross_val_predict(pipe, XX, y, groups=g, cv=cv)
        acc[(mname, sname)] = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
        print(f"{mname:<10} {sname:<22} {XX.shape[1]:>6} {(pred==y).mean()*100:>7.1f}%"
              f" {acc[(mname, sname)].mean()*100:>11.1f}%", flush=True)

CHBMIT = {"S1": +10.4, "S2": +2.6, "S3": +2.5}   # signs to replicate
print(f"\npaired Wilcoxon across {len(subs)} patients (alpha = {0.05/6:.5f}); "
      f"CHB-MIT differences in brackets:")
out = {}
tests = [("S1", "classical ACT + amp", "amplitude only"),
         ("S2", "classical ACT", "QACT sampled"),
         ("S3", "classical ACT + amp", "QACT + amp")]
for mname in MODELS:
    for tag, a_, b_ in tests:
        A, B = acc[(mname, a_)], acc[(mname, b_)]
        d = A - B
        p = wilcoxon(A, B).pvalue if not np.allclose(d, 0) else 1.0
        same = "REPLICATES" if np.sign(d.mean()) == np.sign(CHBMIT[tag]) else "OPPOSITE SIGN"
        star = " *" if p < 0.05 / 6 else ""
        print(f"  {mname:<10} {tag} {a_:<21} - {b_:<18} {d.mean()*100:+5.2f} pts "
              f"[CHB-MIT {CHBMIT[tag]:+.1f}]  better {(d>0).sum():>2}/{len(d)}  "
              f"p={p:.4f}  {same}{star}")
        out[f"{mname}:{tag}"] = dict(diff=float(d.mean()), p=float(p),
                                     better=int((d > 0).sum()), replicates=same)
json.dump({"per_patient": {f"{m}|{s}": v.tolist() for (m, s), v in acc.items()},
           "tests": out}, open("results/siena_confirmation.json", "w"), indent=2)
print(f"\nwrote results/siena_confirmation.json ({time.time()-t0:.0f}s)")

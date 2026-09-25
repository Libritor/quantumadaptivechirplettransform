#!/usr/bin/env python
"""Iteration 5c: fairness check for 5b. The entangled 8-qubit kernel (Z8) lost to
everything at every n, but its bandwidth grid started at 0.1 and entangling maps
concentrate unless the encoding is gentle. Re-run Z8 alone with the grid extended
down to 0.01, on the SAME training draws (identical RNG sequence), and report the
median off-diagonal kernel value actually reached."""
import json
import os
os.environ.setdefault("QBE_DEVICE", "cuda")
import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import qsearch_05_allfeature_lowdata as it5

X = np.load("results/X_chbmitamp.npy")
y = np.load("results/y_chbmitamp.npy")
g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
subs = np.unique(g)
cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)
grid = (0.01, 0.02, 0.03, 0.05, 0.1, 0.25)
res = {n: [] for n in it5.NS}
meds, old_meds = {n: [] for n in it5.NS}, {n: [] for n in it5.NS}
rng = np.random.default_rng(0)
kf = lambda a, b, s: it5.zz_kernel(a, b, s, "zz")
for tr, te in cv.split(X, y, g):
    for n in it5.NS:
        for _ in range(it5.DRAWS):
            pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
            idx = np.concatenate([rng.choice(pos, n // 2, replace=False),
                                  rng.choice(neg, n // 2, replace=False)])
            sc = StandardScaler().fit(X[idx])
            Ztr, Zte = sc.transform(X[idx]), sc.transform(X[te])
            top8 = np.argsort(-np.nan_to_num(f_classif(Ztr, y[idx])[0]))[:8]
            Za, Zb = Ztr[:, top8], Zte[:, top8]
            p = it5.median_bandwidth(kf, Za, grid)
            K = kf(Za, Za, p)
            off = ~np.eye(len(K), dtype=bool)
            meds[n].append(float(np.median(K[off])))
            p_old = it5.median_bandwidth(kf, Za, (0.1, 0.25, 0.5, 1.0))
            old_meds[n].append(float(np.median(kf(Za, Za, p_old)[off])))
            clf = SVC(C=1.0, kernel="precomputed").fit(K, y[idx])
            res[n].append((te, clf.predict(kf(Zb, Za, p))))
prev = json.load(open("results/qsearch_05_allfeature_lowdata.json"))["part_b"]
out = {}
print("n    Z8 extended   (old Z8)   median K old -> new   vs Z8off        vs RBF          vs LR")
for n in it5.NS:
    hits, cnt = np.zeros(len(y)), np.zeros(len(y))
    for te, pr in res[n]:
        hits[te] += pr == y[te]
        cnt[te] += 1
    a = hits / np.maximum(cnt, 1)
    acc = np.array([a[g == s].mean() for s in subs])
    row = {}
    for m in ("Z8off", "RBF", "LR"):
        b = np.array(prev[m][str(n)])
        d = acc - b
        p = float(wilcoxon(acc, b).pvalue) if np.any(d != 0) else 1.0
        row[m] = (float(d.mean() * 100), int((d > 0).sum()), p)
    out[n] = dict(acc=acc.tolist(), vs=row, median_new=float(np.median(meds[n])),
                  median_old=float(np.median(old_meds[n])))
    print(f"{n:<4} {acc.mean() * 100:8.2f}%   ({np.mean(prev['Z8'][str(n)]) * 100:6.2f}%)   "
          f"{np.median(old_meds[n]):.3f} -> {np.median(meds[n]):.3f}   " + "   ".join(
              f"{row[m][0]:+5.2f} ({row[m][1]}/23 p={row[m][2]:.3f})" for m in ("Z8off", "RBF", "LR")))
json.dump(out, open("results/qsearch_05c_z8_bandwidth.json", "w"), indent=1)

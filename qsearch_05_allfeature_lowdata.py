#!/usr/bin/env python
"""Search iteration 5: two leads from the second research round.

5a  ALL-FEATURE QUANTUM KERNEL (after Delilbasic et al., arXiv:2605.17587, 2026:
    fidelity kernels on hyperspectral data using every band, one qubit per
    feature, no feature selection). With one Z-feature-map qubit per feature the
    fidelity kernel is exactly  K(x, x') = prod_i cos^2(s (z_i - z'_i)),  z the
    standardised features -- computed in closed form, no simulator needed. For
    small s it tends to an RBF kernel, so this tests whether the reported gain
    survives against a tuned RBF-SVM. (Entangled low-bond versions are what that
    paper simulates with tensor networks -- efficiently classical by design.)

5b  SMALL-DATA REGIME (Caro et al., Nat. Commun. 13:4919, 2022, and later
    experiments: QML models can generalise from very few samples). Learning
    curves with n in {20, 50, 100, 200} balanced training epochs drawn from the
    training patients, 10 draws per fold; tested on all epochs of the held-out
    patients. With so few labels nothing can be tuned fairly on labels, so EVERY
    kernel uses the same label-free bandwidth rule (median heuristic: median
    off-diagonal kernel value on the training draw = 0.5) and C = 1.

PRE-REGISTERED (written before running):
  data     CHB-MIT, 270 features (ACT + amplitude, the project's primary set),
           outer StratifiedGroupKFold(6, shuffle, seed 0) by patient, per-patient
           accuracy, paired Wilcoxon across 23 patients
  5a       A = all-feature quantum kernel (s tuned on the fold's 3,000-epoch
           subsample, as in iteration 4) vs tuned RBF-SVM and tuned gradient
           boosting; alpha = 0.05 / 2
  5b       at each n: Z8 = 8-qubit ZZ quantum kernel (full entanglement) on the 8
           features ranked best by ANOVA F on the training draw; Z8off = same,
           entanglement off; A = all-feature kernel; RBF = RBF-SVM; LR = logistic
           (C = 0.1). Primary: Z8 vs RBF and Z8 vs LR at each n (8 tests),
           alpha = 0.05 / 8. A quantum win needs Z8 significantly better than
           both RBF and LR at some n, and better than Z8off there.
"""
import json
import os
import time

os.environ.setdefault("QBE_DEVICE", "cuda")

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import qsearch_04_atomset_kernel as it4
from qbe.quantum_fast import statevectors

DEV = "cuda"
S_GRID = (0.01, 0.03, 0.1, 0.3)
NS = (20, 50, 100, 200)
DRAWS = 10


def allfeature_kernel(Za, Zb, s):
    """prod_i cos^2(s (za_i - zb_i)) for all pairs, on the GPU, in log space.
    Block size keeps the (blk, len(Zb), d) intermediate near 160 MB."""
    A = torch.as_tensor(Za, device=DEV, dtype=torch.float64)
    B = torch.as_tensor(Zb, device=DEV, dtype=torch.float64)
    blk = max(1, int(2e7 // (len(B) * A.shape[1])))
    out = torch.empty(len(A), len(B), device=DEV, dtype=torch.float64)
    for lo in range(0, len(A), blk):
        d = A[lo:lo + blk, None, :] - B[None, :, :]
        out[lo:lo + blk] = torch.log(torch.cos(s * d) ** 2 + 1e-300).sum(-1)
    return torch.exp(out).cpu().numpy()


def median_bandwidth(kfun, Z, grid):
    """Label-free: the bandwidth whose median off-diagonal kernel value is nearest 0.5."""
    best = None
    for p in grid:
        K = kfun(Z, Z, p)
        m = np.median(K[~np.eye(len(K), dtype=bool)])
        if best is None or abs(m - 0.5) < best[0]:
            best = (abs(m - 0.5), p)
    return best[1]


def zz_kernel(Za, Zb, s, kind):
    ang_a = np.pi * s * np.clip(Za, -3, 3) / 3
    ang_b = np.pi * s * np.clip(Zb, -3, 3) / 3
    Sa = statevectors(ang_a, 8, layers=2, entanglement="full", kind=kind)
    Sb = statevectors(ang_b, 8, layers=2, entanglement="full", kind=kind)
    return ((Sa @ Sb.conj().T).abs() ** 2).cpu().numpy()


def rbf(Za, Zb, gamma):
    d2 = ((Za[:, None, :] - Zb[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


def part_a(X, y, g, subs, cv):
    Z = StandardScaler().fit_transform(X)       # label-free, per-feature scale only
    kernels = {s: allfeature_kernel(Z, Z, s) for s in S_GRID}
    pred, chosen = it4.eval_precomputed(kernels, y, g, cv)
    acc = np.array([(pred[g == s] == y[g == s]).mean() for s in subs])
    return acc, chosen


def part_b(X, y, g, subs, cv):
    res = {m: {n: [] for n in NS} for m in ("Z8", "Z8off", "A", "RBF", "LR")}
    rng = np.random.default_rng(0)
    for fold, (tr, te) in enumerate(cv.split(X, y, g)):
        for n in NS:
            for _ in range(DRAWS):
                pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
                idx = np.concatenate([rng.choice(pos, n // 2, replace=False),
                                      rng.choice(neg, n // 2, replace=False)])
                sc = StandardScaler().fit(X[idx])
                Ztr, Zte = sc.transform(X[idx]), sc.transform(X[te])
                top8 = np.argsort(-np.nan_to_num(f_classif(Ztr, y[idx])[0]))[:8]
                preds = {}
                for m, kf, Za, Zb, grid in (
                        ("Z8", lambda a, b, s: zz_kernel(a, b, s, "zz"), Ztr[:, top8], Zte[:, top8], (0.1, 0.25, 0.5, 1.0)),
                        ("Z8off", lambda a, b, s: zz_kernel(a, b, s, "z"), Ztr[:, top8], Zte[:, top8], (0.1, 0.25, 0.5, 1.0)),
                        ("A", allfeature_kernel, Ztr, Zte, S_GRID),
                        ("RBF", rbf, Ztr, Zte, (1e-4, 1e-3, 1e-2, 1e-1))):
                    p = median_bandwidth(kf, Za, grid)
                    clf = SVC(C=1.0, kernel="precomputed").fit(kf(Za, Za, p), y[idx])
                    preds[m] = clf.predict(kf(Zb, Za, p))
                preds["LR"] = LogisticRegression(C=0.1, max_iter=5000).fit(Ztr, y[idx]).predict(Zte)
                for m, pr in preds.items():
                    res[m][n].append((te, pr))
        print(f"   5b fold {fold + 1}/6 done", flush=True)
    # per-patient accuracy at each n, averaged over draws
    acc = {m: {} for m in res}
    for m in res:
        for n in NS:
            hits = np.zeros(len(y))
            cnt = np.zeros(len(y))
            for te, pr in res[m][n]:
                hits[te] += pr == y[te]
                cnt[te] += 1
            a = hits / np.maximum(cnt, 1)
            acc[m][n] = np.array([a[g == s].mean() for s in subs])
    return acc


def main():
    t0 = time.time()
    X = np.load("results/X_chbmitamp.npy")
    y = np.load("results/y_chbmitamp.npy")
    g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
    subs = np.unique(g)
    cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)
    print(f"{len(y)} epochs, {len(subs)} patients, {X.shape[1]} features\n", flush=True)

    # ---- 5a ----
    accA, chosenA = part_a(X, y, g, subs, cv)
    base = {}
    for name, make, grid in (
            ("RBF-SVM", lambda C, gamma: it4.pipe(SVC(C=C, gamma=gamma)),
             [dict(C=c, gamma=gm) for c in (0.1, 1.0, 10.0) for gm in ("scale", 1e-3, 1e-2)]),
            ("grad. boosting", lambda lr, depth: it4.HistGradientBoostingClassifier(
                learning_rate=lr, max_depth=depth, max_iter=300, random_state=0),
             [dict(lr=lr, depth=d) for lr in (0.05, 0.1) for d in (3, None)])):
        pr, _ = it4.eval_flat(make, grid, X, y, g, cv)
        base[name] = np.array([(pr[g == s] == y[g == s]).mean() for s in subs])
    print(f"5a  all-feature quantum kernel {accA.mean() * 100:.2f}%  (bandwidths {chosenA})")
    outA = {}
    for name, b in base.items():
        d = accA - b
        p = float(wilcoxon(accA, b).pvalue) if np.any(d != 0) else 1.0
        outA[name] = dict(base_pct=float(b.mean() * 100), diff_pts=float(d.mean() * 100),
                          better=int((d > 0).sum()), p=p)
        print(f"    vs {name:<15} {b.mean() * 100:6.2f}%  diff {d.mean() * 100:+6.2f} pts  "
              f"better on {(d > 0).sum():2d}/23  p={p:.4f}", flush=True)

    # ---- 5b ----
    acc = part_b(X, y, g, subs, cv)
    print(f"\n5b  per-patient accuracy vs training-set size (mean of {DRAWS} draws x 6 folds):")
    print(f"    {'n':>5}" + "".join(f"{m:>9}" for m in ("Z8", "Z8off", "A", "RBF", "LR")))
    for n in NS:
        print(f"    {n:>5}" + "".join(f"{acc[m][n].mean() * 100:>8.2f}%"
                                      for m in ("Z8", "Z8off", "A", "RBF", "LR")))
    alpha = 0.05 / (2 * len(NS))
    print(f"    paired Wilcoxon, Z8 minus each (alpha = {alpha:.5f}):")
    outB, win = {}, False
    for n in NS:
        row = {}
        for m in ("RBF", "LR", "Z8off"):
            d = acc["Z8"][n] - acc[m][n]
            p = float(wilcoxon(acc["Z8"][n], acc[m][n]).pvalue) if np.any(d != 0) else 1.0
            row[m] = dict(diff_pts=float(d.mean() * 100), better=int((d > 0).sum()), p=p)
        outB[n] = row
        print(f"    n={n:>3}: " + "   ".join(
            f"vs {m} {row[m]['diff_pts']:+5.2f} ({row[m]['better']}/23, p={row[m]['p']:.4f})"
            for m in ("RBF", "LR", "Z8off")))
        if all(row[m]["diff_pts"] > 0 and row[m]["p"] < alpha for m in ("RBF", "LR")) \
                and row["Z8off"]["diff_pts"] > 0:
            win = True
    winA = all(v["diff_pts"] > 0 and v["p"] < 0.025 for v in outA.values())
    print(f"\nverdict: 5a {'QUANTUM-INSPIRED WIN' if winA else 'no win'}; "
          f"5b {'QUANTUM WIN' if win else 'no quantum win'}  ({time.time() - t0:.0f}s)")
    json.dump(dict(part_a=dict(acc=accA.tolist(), tests=outA, chosen=str(chosenA)),
                   part_b={m: {str(n): acc[m][n].tolist() for n in NS} for m in acc},
                   part_b_tests={str(n): v for n, v in outB.items()},
                   quantum_win_b=win, quantum_inspired_win_a=winA),
              open("results/qsearch_05_allfeature_lowdata.json", "w"), indent=1)


if __name__ == "__main__":
    main()

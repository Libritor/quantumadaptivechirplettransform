#!/usr/bin/env python
"""Search iteration 4: a quantum kernel on SETS of chirplet atoms.

Every EEG channel-epoch that ACT decomposes IS a set of atoms (time, frequency,
width, chirp rate), each with an energy. Flat summary features throw that set
structure away. Here each atom is embedded with an IQP-style quantum feature
map (Havlicek et al. 2019 ZZ map, all-to-all entangling, 8 qubits, 2 layers --
the family conjectured hard to simulate classically at scale), a window becomes
the energy-weighted MIXTURE of its atoms' states,

    rho = sum_k w_k |phi(a_k)><phi(a_k)|,   w = energy / total energy,

and windows are compared by K = Tr(rho rho') = sum_{k,l} w_k w'_l |<phi_k|phi'_l>|^2,
averaged over the 18 channels and cosine-normalised. States come from the
project's exact feature-map simulator (qbe/quantum_fast.py, verified against
qiskit to 1e-12).

PRE-REGISTERED (written before running):
  task     CHB-MIT seizure detection, 10,856 epochs, 23 patient groups; ACT-only
           information (no amplitude features) for every model
  split    outer StratifiedGroupKFold(6, shuffle, seed 0) by patient -- identical
           to compare_qact_allfeat.py; metric per-patient accuracy from
           out-of-fold predictions
  tuning   every model tunes on the same 3,000-epoch subsample of its outer
           training fold (GroupKFold(3) inside), then refits on the full fold
  models   Q    quantum atom-set kernel (zz map, full entanglement) + SVM;
                bandwidth s in {0.25, 0.5, 1, 2}, C in {0.1, 1, 10}
           Qp   the same map with entanglement OFF (product states)  [control]
           R    classical atom-set kernel: RBF on atom parameters, same set
                construction; gamma in {1, 4, 16, 64}, C as above
           F1   logistic regression on the 234 flat ACT features (C grid)
           F2   RBF-SVM on the flat ACT features (C x gamma grid)
           F3   HistGradientBoosting on the flat ACT features (small grid)
  primary  Q vs R, F1, F2, F3: paired Wilcoxon across patients, two-sided,
           Bonferroni alpha = 0.05 / 4 = 0.0125
  rule     QUANTUM WIN only if Q is significantly better than ALL FOUR. Q vs Qp
           is reported to show whether entanglement is doing the work.
"""
import json
import os
import time

os.environ.setdefault("QBE_DEVICE", "cuda")

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qbe.quantum_fast import statevectors

DEV = "cuda"
N_Q, LAYERS, BLK = 8, 2, 128
Q_SCALES = (0.25, 0.5, 1.0, 2.0)
RBF_GAMMAS = (1.0, 4.0, 16.0, 64.0)
C_GRID = (0.1, 1.0, 10.0)
INNER_SUB = 3000
ALPHA = 0.05 / 4


def atom_params(atoms):
    """(E, C, K, 5) atoms -> parameters in [0, 1] and energy weights.
    Label-free robust scaling (1st-99th percentile over all atoms)."""
    t, f, d, r, e = (atoms[..., i].astype(np.float64) for i in range(5))
    cols = [t, f, np.log(np.maximum(d, 1e-4)), np.sign(r) * np.log1p(np.abs(r))]
    live = e > 0
    P = []
    for x in cols:
        lo, hi = np.percentile(x[live], [1, 99])
        P.append(np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1))
    P = np.stack(P, -1)
    s = e.sum(-1, keepdims=True)
    w = np.where(s > 0, np.maximum(e, 0) / np.where(s > 0, s, 1), 0.0)
    return P.astype(np.float32), w.astype(np.float32)


def angles(X, scale):
    """8 angles per atom: the 4 parameters and their 4 cyclic pairwise products."""
    prods = [X[:, [0]] * X[:, [1]], X[:, [1]] * X[:, [2]],
             X[:, [2]] * X[:, [3]], X[:, [3]] * X[:, [0]]]
    return np.pi * scale * np.concatenate([X] + prods, 1)


def _reduce(W, Wb, M, b, K, E):
    return torch.einsum("bk,bkel,el->be", Wb, M.reshape(b, K, E, K), W)


def set_kernel(P, w, family, param):
    """Averaged-over-channels atom-set kernel, cosine-normalised. (E, E) float32."""
    E, C, K, _ = P.shape
    Kt = torch.zeros(E, E, device=DEV, dtype=torch.float32)
    for c in range(C):
        X = P[:, c].reshape(E * K, 4)
        W = torch.as_tensor(w[:, c], device=DEV)
        if family == "rbf":
            A = torch.as_tensor(X, device=DEV)
            sq = (A * A).sum(1)
        else:
            S = statevectors(angles(X, param), N_Q, layers=LAYERS, entanglement="full",
                             kind=family).to(torch.complex64)
        for lo in range(0, E, BLK):
            hi = min(E, lo + BLK)
            b = hi - lo
            if family == "rbf":
                Ab = A[lo * K:hi * K]
                D2 = (sq[lo * K:hi * K, None] + sq[None, :] - 2 * Ab @ A.T).clamp(min=0)
                M = torch.exp(-param * D2)
            else:
                G = S[lo * K:hi * K] @ S.conj().T
                M = G.real ** 2 + G.imag ** 2
            Kt[lo:hi] += _reduce(W, W[lo:hi], M, b, K, E)
        torch.cuda.synchronize()
    d = torch.sqrt(torch.clamp(torch.diagonal(Kt), min=1e-12))
    return (Kt / d[:, None] / d[None, :]).cpu().numpy()


def eval_precomputed(kernels, y, g, cv):
    """Outer CV with inner tuning of (kernel parameter, C) on a subsample."""
    pred = np.empty_like(y)
    chosen = []
    rng = np.random.default_rng(0)
    for tr, te in cv.split(np.zeros(len(y)), y, g):
        sub = rng.choice(tr, min(INNER_SUB, len(tr)), replace=False)
        best = None
        for p, Km in kernels.items():
            for C in C_GRID:
                sc = []
                for itr, ite in GroupKFold(3).split(sub, y[sub], g[sub]):
                    a, bb = sub[itr], sub[ite]
                    clf = SVC(C=C, kernel="precomputed").fit(Km[np.ix_(a, a)], y[a])
                    sc.append((clf.predict(Km[np.ix_(bb, a)]) == y[bb]).mean())
                if best is None or np.mean(sc) > best[0]:
                    best = (np.mean(sc), p, C)
        _, p, C = best
        chosen.append((p, C))
        Km = kernels[p]
        clf = SVC(C=C, kernel="precomputed").fit(Km[np.ix_(tr, tr)], y[tr])
        pred[te] = clf.predict(Km[np.ix_(te, tr)])
    return pred, chosen


def eval_flat(make, grid, X, y, g, cv):
    pred = np.empty_like(y)
    chosen = []
    rng = np.random.default_rng(0)
    for tr, te in cv.split(X, y, g):
        sub = rng.choice(tr, min(INNER_SUB, len(tr)), replace=False)
        best = None
        for params in grid:
            sc = []
            for itr, ite in GroupKFold(3).split(sub, y[sub], g[sub]):
                a, bb = sub[itr], sub[ite]
                m = make(**params).fit(X[a], y[a])
                sc.append((m.predict(X[bb]) == y[bb]).mean())
            if best is None or np.mean(sc) > best[0]:
                best = (np.mean(sc), params)
        chosen.append(best[1])
        pred[te] = make(**best[1]).fit(X[tr], y[tr]).predict(X[te])
    return pred, chosen


def pipe(clf):
    return Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()), ("clf", clf)])


def main():
    t0 = time.time()
    atoms = np.load("results/atoms_chbmit.npy")
    y = np.load("results/y_chbmit.npy")
    g = np.load("results/g_chbmit.npy", allow_pickle=True)
    Xf = np.load("results/X_chbmit.npy")
    subs = np.unique(g)
    cv = StratifiedGroupKFold(6, shuffle=True, random_state=0)
    P, w = atom_params(atoms)
    print(f"{len(y)} epochs, {len(subs)} patients, atoms {atoms.shape}, flat ACT "
          f"features {Xf.shape[1]}\n", flush=True)

    preds, chosen = {}, {}
    for name, family, grid in (("Q", "zz", Q_SCALES), ("Qp", "z", Q_SCALES),
                               ("R", "rbf", RBF_GAMMAS)):
        t1 = time.time()
        kernels = {p: set_kernel(P, w, family, p) for p in grid}
        t2 = time.time()
        preds[name], chosen[name] = eval_precomputed(kernels, y, g, cv)
        del kernels
        print(f"[{name}] kernels {t2 - t1:.0f}s, CV {time.time() - t2:.0f}s, chosen "
              f"{chosen[name]}", flush=True)

    flat = {
        "F1": (lambda C: pipe(LogisticRegression(C=C, max_iter=5000)),
               [dict(C=c) for c in (0.001, 0.01, 0.1, 1.0)]),
        "F2": (lambda C, gamma: pipe(SVC(C=C, gamma=gamma)),
               [dict(C=c, gamma=gm) for c in (0.1, 1.0, 10.0) for gm in ("scale", 1e-3, 1e-2)]),
        "F3": (lambda lr, depth: HistGradientBoostingClassifier(
                   learning_rate=lr, max_depth=depth, max_iter=300, random_state=0),
               [dict(lr=lr, depth=d) for lr in (0.05, 0.1) for d in (3, None)]),
    }
    for name, (make, grid) in flat.items():
        t1 = time.time()
        preds[name], chosen[name] = eval_flat(make, grid, Xf, y, g, cv)
        print(f"[{name}] CV {time.time() - t1:.0f}s, chosen {chosen[name]}", flush=True)

    acc = {k: np.array([(p[g == s] == y[g == s]).mean() for s in subs]) for k, p in preds.items()}
    labels = dict(Q="quantum atom-set kernel", Qp="quantum, entanglement off",
                  R="classical RBF atom-set kernel", F1="logistic, flat ACT",
                  F2="RBF-SVM, flat ACT", F3="gradient boosting, flat ACT")
    print(f"\nper-patient accuracy (mean over {len(subs)} patients):")
    for k in ("Q", "Qp", "R", "F1", "F2", "F3"):
        print(f"  {k:3s} {labels[k]:<32} {acc[k].mean() * 100:6.2f}%   pooled "
              f"{(preds[k] == y).mean() * 100:6.2f}%")
    print(f"\npaired Wilcoxon, Q minus each (alpha = {ALPHA:.4f}):")
    tests = {}
    for k in ("R", "F1", "F2", "F3", "Qp"):
        d = acc["Q"] - acc[k]
        p = float(wilcoxon(acc["Q"], acc[k]).pvalue) if np.any(d != 0) else 1.0
        tests[k] = dict(diff_pts=float(d.mean() * 100), better=int((d > 0).sum()), p=p)
        tag = "(control)" if k == "Qp" else ("BETTER *" if d.mean() > 0 and p < ALPHA else
                                               "WORSE *" if p < ALPHA else "no difference")
        print(f"  Q - {k:3s} {d.mean() * 100:+6.2f} pts  better on {(d > 0).sum():2d}/"
              f"{len(d)}  p={p:.4f}  {tag}")
    win = all(tests[k]["diff_pts"] > 0 and tests[k]["p"] < ALPHA for k in ("R", "F1", "F2", "F3"))
    print(f"\nverdict: {'QUANTUM WIN' if win else 'no quantum win'} under the "
          f"pre-registered rule  ({time.time() - t0:.0f}s)")
    json.dump(dict(acc={k: v.tolist() for k, v in acc.items()}, tests=tests,
                   chosen={k: str(v) for k, v in chosen.items()}, quantum_win=win),
              open("results/qsearch_04_atomset_kernel.json", "w"), indent=1)


if __name__ == "__main__":
    main()

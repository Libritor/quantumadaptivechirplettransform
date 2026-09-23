#!/usr/bin/env python
"""Can a quantum kernel beat the classical baselines on chirplet features?

PROTOCOL (fixed before looking at any result)
---------------------------------------------
The honest version of "beat classical" requires three things, and skipping any
one of them manufactures a win:

1. **Equal tuning budget.** Tuning the quantum model against untuned baselines
   is a rigged comparison. Every model here gets the same number of randomly
   sampled configurations from its own grid.
2. **Subject-wise splits everywhere.** Epochs from one person are near
   duplicates. An ungrouped split lets a model score by recognising the
   subject rather than the cognitive state. Outer split and inner folds are
   both grouped by subject, so the test subjects are people no model has seen.
3. **Selection and evaluation kept apart.** Hyperparameters are chosen by inner
   CV on the training subjects only. The held-out subjects are touched exactly
   once, at the end, by each model's single selected configuration.

The dominant quantum hyperparameter is `angle_scale` -- the kernel bandwidth.
Encoding into the full [0, pi] pushes states so far apart that the Gram matrix
concentrates toward the identity and nothing generalises; shrinking it is the
analogue of lowering gamma in an RBF kernel (Shaydulin & Wild, 2022).
"""
from __future__ import annotations

import argparse, json, time, warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

warnings.filterwarnings("ignore")
from qbe import quantum

RNG = np.random.default_rng(0)

QUANTUM_GRID = dict(
    angle_scale=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    reps=[1, 2],
    entanglement=["linear", "full"],
    feature_map=["zz", "z"],
    C=[0.1, 1.0, 10.0, 100.0],
    n_qubits=[4, 6, 8],
)
# Per-model grids. logreg ignores `gamma`, so including it there would let the
# sampler burn budget on configurations that are identical in effect -- which
# would quietly hand the quantum model an advantage. Each grid contains only
# the knobs that model actually responds to.
SVM_GRID = dict(
    C=[0.01, 0.1, 1.0, 10.0, 100.0],
    gamma=["scale", 0.01, 0.05, 0.1, 0.5],
    n_qubits=[4, 6, 8, 12, 20],      # = number of features kept by selectk
)
LOGREG_GRID = dict(
    C=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
    n_qubits=[4, 6, 8, 12, 16, 20, 30],
)
GRIDS = {"qsvc": QUANTUM_GRID, "vqc": QUANTUM_GRID,
         "svm_rbf": SVM_GRID, "logreg": LOGREG_GRID}


def sample(grid, n, rng):
    keys = list(grid)
    seen, out = set(), []
    for _ in range(n * 40):
        if len(out) >= n:
            break
        cfg = {k: grid[k][rng.integers(len(grid[k]))] for k in keys}
        key = tuple(str(cfg[k]) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def score_config(kind, X, y, g, cfg, folds, seed):
    cfg = dict(cfg)
    nq = cfg.pop("n_qubits")
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipe = quantum.make_pipeline(kind, nq, seed=seed, **cfg)
    s = cross_val_score(pipe, X, y, cv=cv, groups=g, scoring="accuracy", n_jobs=1)
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results", help="folder with X/y/g npy files")
    ap.add_argument("--tag", default="eegmat")
    ap.add_argument("--budget", type=int, default=25, help="configs per model")
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.33, help="fraction of SUBJECTS held out")
    ap.add_argument("--max-tune", type=int, default=900,
                    help="subsample this many training epochs for the search")
    ap.add_argument("--models", nargs="+",
                    default=["qsvc", "svm_rbf", "logreg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tuning.json")
    a = ap.parse_args()

    X = np.load(f"{a.data}/X_{a.tag}.npy")
    y = np.load(f"{a.data}/y_{a.tag}.npy")
    g = np.load(f"{a.data}/g_{a.tag}.npy", allow_pickle=True)
    print(f"[tune] X={X.shape} subjects={len(np.unique(g))} classes={np.bincount(y)}")

    # --- outer split, by SUBJECT ---
    rng = np.random.default_rng(a.seed)
    subs = np.unique(g)
    rng.shuffle(subs)
    n_test = max(2, int(round(len(subs) * a.test_frac)))
    test_subs = set(subs[:n_test].tolist())
    te = np.array([s in test_subs for s in g])
    tr = ~te
    print(f"[tune] train {tr.sum()} epochs / {len(subs)-n_test} subjects | "
          f"test {te.sum()} epochs / {n_test} subjects (held out)")

    # --- subsample the training set for the search only ---
    idx = np.flatnonzero(tr)
    if len(idx) > a.max_tune:
        idx = rng.choice(idx, a.max_tune, replace=False)
    Xs, ys, gs = X[idx], y[idx], g[idx]
    print(f"[tune] searching on {len(idx)} training epochs, "
          f"{len(np.unique(gs))} subjects, budget={a.budget}/model\n")

    report = {}
    for kind in a.models:
        grid = GRIDS[kind]
        cfgs = sample(grid, a.budget, np.random.default_rng(a.seed))
        best, best_s, t0 = None, -1.0, time.time()
        for i, cfg in enumerate(cfgs, 1):
            try:
                s = score_config(kind, Xs, ys, gs, cfg, a.inner_folds, a.seed)
            except Exception as e:
                print(f"  [{kind} {i}/{len(cfgs)}] FAILED {type(e).__name__}")
                continue
            if s > best_s:
                best, best_s = cfg, s
                print(f"  [{kind} {i}/{len(cfgs)}] inner={s*100:5.1f}%  {cfg}  <-- best")
            elif i % 10 == 0:
                print(f"  [{kind} {i}/{len(cfgs)}] inner={s*100:5.1f}%")
        search_s = time.time() - t0

        # --- refit on ALL training subjects, evaluate once on held-out ---
        cfg = dict(best); nq = cfg.pop("n_qubits")
        pipe = quantum.make_pipeline(kind, nq, seed=a.seed, **cfg)
        pipe.fit(X[tr], y[tr])
        test_acc = float(pipe.score(X[te], y[te]))
        # Per-subject accuracy. Epochs within a subject are not independent, so
        # the effective sample size for comparing models is the number of test
        # SUBJECTS, not epochs. Pooled accuracy alone would make a 1-point gap
        # look far more certain than it is.
        pred = pipe.predict(X[te])
        gte, yte = g[te], y[te]
        per_subj = {str(s_): float((pred[gte == s_] == yte[gte == s_]).mean())
                    for s_ in np.unique(gte)}
        report[kind] = dict(best=best, inner=best_s, test=test_acc,
                            per_subject=per_subj, search_seconds=search_s)
        print(f"[{kind}] inner {best_s*100:.1f}% -> HELD-OUT {test_acc*100:.1f}% "
              f"({search_s:.0f}s search)\n")

    maj = float(np.bincount(y[te]).max()) / int(te.sum())
    print("=" * 62)
    print(f"{'model':<10} {'inner CV':>9} {'HELD-OUT':>10}")
    print("-" * 62)
    for k, v in sorted(report.items(), key=lambda kv: -kv[1]["test"]):
        print(f"{k:<10} {v['inner']*100:>8.1f}% {v['test']*100:>9.1f}%")
    print(f"{'majority':<10} {'':>9} {maj*100:>9.1f}%")
    qn = [k for k in report if k in ("qsvc", "vqc")]
    cn = [k for k in report if k in ("svm_rbf", "logreg")]
    if qn and cn:
        bq = max(qn, key=lambda k: report[k]["test"])
        bc = max(cn, key=lambda k: report[k]["test"])
        d = (report[bq]["test"] - report[bc]["test"]) * 100
        print(f"\nbest quantum ({bq}) - best classical ({bc}) = {d:+.1f} points")
        # Paired across test subjects: the honest significance test here.
        # NB: do not name these `a`/`b` -- `a` is the argparse namespace.
        subs_ = sorted(report[bq]["per_subject"])
        qacc = np.array([report[bq]["per_subject"][s_] for s_ in subs_])
        cacc = np.array([report[bc]["per_subject"][s_] for s_ in subs_])
        wins = int((qacc > cacc).sum()); ties = int((qacc == cacc).sum())
        print(f"per-subject: quantum better on {wins}/{len(subs_)} "
              f"({ties} tied), mean diff {(qacc-cacc).mean()*100:+.1f} pts")
        try:
            from scipy.stats import wilcoxon
            if not np.allclose(qacc, cacc):
                print(f"paired Wilcoxon across subjects: "
                      f"p = {wilcoxon(qacc, cacc).pvalue:.3f}")
        except Exception:
            pass
    report["_meta"] = dict(test_subjects=sorted(test_subs), majority=maj,
                           n_train=int(tr.sum()), n_test=int(te.sum()))
    json.dump(report, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

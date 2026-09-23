#!/usr/bin/env python
"""Second attempt at quantum vs classical -- nested, subject-wise, pre-registered.

WIN CRITERION (fixed before this was run)
-----------------------------------------
Quantum wins only if the best quantum model has higher nested-CV accuracy than
the best classical model AND a paired Wilcoxon test across the 36 subjects gives
p < 0.025 (0.05 Bonferroni-corrected for the two normalisation conditions).
A lead that misses that bar is reported as "ahead, not significant", which is
not a win.

WHY NESTED CV THIS TIME
-----------------------
The first attempt held out 12 subjects and those results have now been seen.
Re-using that same held-out set while iterating on methods would be tuning to
the test set. Nested CV instead scores all 36 subjects exactly once, each by a
model whose hyperparameters were chosen using only the *other* subjects, and it
gives the paired test 36 subjects of power instead of 12.

WHAT IS NEW (each aimed at a measured failure of attempt one)
------------------------------------------------------------
* `pqk` -- projected quantum kernel (Huang et al. 2021). The fidelity kernel had
  the biggest inner->held-out drop (54.1% -> 50.9%), the signature of global
  kernel concentration; comparing single-qubit reduced states is the known fix.
* data re-uploading (Perez-Salinas et al. 2020). logreg won using 30 features
  while quantum was capped at 8; n qubits x L layers now reads n*L features.
* per-subject normalisation, applied to EVERY model including the classical
  baselines, so it cannot tilt the comparison. Each subject's features are
  z-scored using that subject's own epochs; labels are never used. Assumption,
  stated plainly: at deployment you have an unlabelled calibration batch from
  the new user containing a mix of mental states. EEGMAT's per-subject balance
  (28 rest / 28 arithmetic) makes this unusually favourable, so absolute
  accuracies under `subject` normalisation are optimistic. Because the same
  transform is given to every model, the *comparison* is unaffected.

Fairness rules carried over from attempt one: equal number of sampled
configurations per model, per-model grids containing only knobs that model
responds to, and the same outer folds for everyone.
"""
from __future__ import annotations

import argparse, json, time, warnings
import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
from qbe import quantum
from qbe.hybrid import make_hybrid_pipeline
from qbe.quantum_fast import FidelityKernelSVC, PQKFeatures

QUANTUM_MODELS = ("qsvc", "pqk", "qlr", "hybrid")
# Same hybrid with the non-entangling z map: its circuit outputs are exactly
# cos/sin of the inputs, i.e. classically trivial. The entanglement control.
CONTROL_MODELS = ("hybrid_z",)
CLASSICAL_MODELS = ("logreg", "svm_rbf")

_Q = dict(
    n_qubits=[4, 6, 8, 10],
    layers=[1, 2, 3],
    reupload=[False, True],
    entanglement=["linear", "full"],
    feature_map=["zz", "z"],
    angle_scale=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
)
GRIDS = {
    "qsvc": {**_Q, "C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "pqk": {**_Q, "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.1, 0.3, 1.0, 3.0]},
    # qlr: quantum features, linear readout. Attempt two showed the readout,
    # not the kernel, decides this problem: on weak near-linear signal a heavily
    # regularised logistic regression beat every SVM, classical RBF included.
    # So the quantum circuit here supplies the features -- single-qubit Pauli
    # expectations, which contain near-linear terms (<Y> ~ sin 2x), nonlinear
    # terms (<X> ~ cos 2x) and, under ZZ entanglement, pairwise interactions --
    # and the readout is the same model family as the classical winner.
    "qlr": {**_Q, "C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "hybrid": dict(
        k_classical=[8, 12, 16, 20, 30],
        n_qubits=[4, 6, 8, 10], layers=[1, 2, 3], reupload=[False, True],
        entanglement=["linear", "full"], feature_map=["zz"],
        angle_scale=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        C=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    ),
    "svm_rbf": dict(C=[0.01, 0.1, 1.0, 10.0, 100.0],
                    gamma=["scale", 0.01, 0.05, 0.1, 0.5],
                    n_qubits=[4, 6, 8, 12, 20, 30]),
    "logreg": dict(C=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
                   n_qubits=[4, 6, 8, 12, 16, 20, 30]),
}


def sample(grid, n, rng):
    keys, seen, out = list(grid), set(), []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        cfg = {k: grid[k][rng.integers(len(grid[k]))] for k in keys}
        if cfg.get("layers") == 1:
            cfg["reupload"] = False        # identical either way; avoid duplicates
        key = tuple(str(cfg[k]) for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def build(kind, cfg, seed):
    if kind in ("hybrid", "hybrid_z"):
        return make_hybrid_pipeline(**cfg)
    if kind in CLASSICAL_MODELS:
        return quantum.make_pipeline(kind, cfg["n_qubits"], C=cfg["C"],
                                     gamma=cfg.get("gamma", "scale"), seed=seed)
    n, L, ru = cfg["n_qubits"], cfg["layers"], cfg["reupload"]
    common = dict(n_qubits=n, layers=L, entanglement=cfg["entanglement"],
                  feature_map=cfg["feature_map"], reupload=ru)
    steps = [
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("reduce", quantum.SafeSelectKBest(f_classif, k=n * L if ru else n)),
        ("angles", MinMaxScaler(feature_range=(0.0, cfg["angle_scale"] * np.pi))),
    ]
    if kind == "qsvc":
        steps.append(("clf", FidelityKernelSVC(C=cfg["C"], **common)))
    elif kind == "qlr":
        steps += [("pqk", PQKFeatures(**common)),
                  ("std", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=3000, C=cfg["C"]))]
    else:
        steps += [("pqk", PQKFeatures(**common)),
                  ("clf", SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"]))]
    return Pipeline(steps)


def _score(kind, cfg, X, y, g, inner_folds, seed):
    """One inner-CV score; module-level so joblib workers can run it."""
    import warnings
    warnings.filterwarnings("ignore")
    try:
        cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        return float(cross_val_score(build(kind, cfg, seed), X, y, groups=g,
                                     cv=cv, n_jobs=1).mean())
    except Exception:
        return -1.0


def subject_normalize(X, g):
    """Z-score each feature within each subject, from that subject's own epochs.
    Label-free; no information crosses subjects."""
    Z = np.empty_like(X)
    for s in np.unique(g):
        m = g == s
        mu, sd = X[m].mean(0), X[m].std(0)
        Z[m] = (X[m] - mu) / np.where(sd > 1e-12, sd, 1.0)
    return Z


def run_condition(X, y, g, models, a, label):
    outer = StratifiedGroupKFold(n_splits=a.outer, shuffle=True, random_state=a.seed)
    splits = list(outer.split(X, y, g))
    cfgs = {m: sample(GRIDS[m], a.budget, np.random.default_rng(a.seed + i))
            for i, m in enumerate(models) if m != "hybrid_z"}
    if "hybrid_z" in models:
        base = cfgs.get("hybrid") or sample(GRIDS["hybrid"], a.budget,
                                            np.random.default_rng(a.seed + 99))
        cfgs["hybrid_z"] = [{**c, "feature_map": "z"} for c in base]
    preds = {m: np.full(len(y), -1) for m in models}
    chosen = {m: [] for m in models}
    t0 = time.time()
    from joblib import Parallel, delayed
    rng = np.random.default_rng(a.seed)
    for f, (tr, te) in enumerate(splits, 1):
        # Hyperparameters are chosen on at most `max_tune` training epochs;
        # the chosen model is then refit on ALL of them. Selection only.
        sel = tr if len(tr) <= a.max_tune else np.sort(rng.choice(tr, a.max_tune, replace=False))
        for m in models:
            scores = Parallel(n_jobs=a.jobs)(
                delayed(_score)(m, cfg, X[sel], y[sel], g[sel], a.inner, a.seed)
                for cfg in cfgs[m])
            k = int(np.argmax(scores))
            best, best_s = cfgs[m][k], float(scores[k])
            pipe = build(m, best, a.seed).fit(X[tr], y[tr])
            preds[m][te] = pipe.predict(X[te])
            acc = float((preds[m][te] == y[te]).mean())
            chosen[m].append(dict(fold=f, inner=best_s, outer=acc, cfg=best))
            print(f"  [{label}] fold {f}/{len(splits)} {m:<8} inner {best_s*100:5.1f}%"
                  f"  outer {acc*100:5.1f}%   {best}", flush=True)
        print(f"  [{label}] fold {f} done, {time.time()-t0:.0f}s", flush=True)

    subjects = np.unique(g)
    per_subj = {m: np.array([(preds[m][g == s] == y[g == s]).mean() for s in subjects])
                for m in models}
    pooled = {m: float((preds[m] == y).mean()) for m in models}
    return dict(pooled=pooled, per_subj=per_subj, chosen=chosen, preds=preds,
                subjects=subjects)


def report(res, label, alpha):
    pooled, ps = res["pooled"], res["per_subj"]
    qms = [m for m in QUANTUM_MODELS if m in pooled]
    cms = [m for m in CLASSICAL_MODELS if m in pooled]
    print(f"\n=== {label} normalisation: nested subject-wise CV, {len(res['subjects'])} subjects ===")
    print(f"{'model':<9} {'nested acc':>11} {'mean/subject':>13} {'sd':>6}")
    for m in sorted(pooled, key=lambda k: -pooled[k]):
        tag = ("quantum" if m in QUANTUM_MODELS
               else "control (no entanglement)" if m in CONTROL_MODELS
               else "classical")
        print(f"{m:<9} {pooled[m]*100:>10.1f}% {ps[m].mean()*100:>12.1f}% "
              f"{ps[m].std()*100:>5.1f}   {tag}")
    print("\npaired Wilcoxon across subjects (quantum vs each classical):")
    for q in qms:
        for c in cms:
            d = ps[q] - ps[c]
            p = wilcoxon(ps[q], ps[c]).pvalue if not np.allclose(d, 0) else 1.0
            print(f"  {q:<5} vs {c:<8} diff {d.mean()*100:+5.1f} pts  "
                  f"better on {(d>0).sum()}/{len(d)}  p = {p:.4f}")
    if "hybrid" in pooled and "hybrid_z" in pooled:
        d = ps["hybrid"] - ps["hybrid_z"]
        p_ = wilcoxon(ps["hybrid"], ps["hybrid_z"]).pvalue if not np.allclose(d, 0) else 1.0
        print(f"\nENTANGLEMENT CONTROL: hybrid (zz) vs hybrid_z (classically trivial): "
              f"diff {d.mean()*100:+.1f} pts, better on {(d>0).sum()}/{len(d)}, p = {p_:.4f}")
    bq = max(qms, key=lambda k: pooled[k])
    bc = max(cms, key=lambda k: pooled[k])
    d = ps[bq] - ps[bc]
    p = wilcoxon(ps[bq], ps[bc]).pvalue if not np.allclose(d, 0) else 1.0
    lead = (pooled[bq] - pooled[bc]) * 100
    if lead > 0 and p < alpha:
        verdict = "QUANTUM WINS"
    elif lead > 0:
        verdict = "quantum ahead, NOT significant -- not a win"
    else:
        verdict = "classical wins or ties"
    print(f"\nPRE-REGISTERED DECISION: best quantum {bq} vs best classical {bc}: "
          f"{lead:+.1f} pts, p = {p:.4f} (alpha {alpha}) -> {verdict}")
    return dict(best_quantum=bq, best_classical=bc, lead_points=lead, p=p,
                verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eegmat")
    ap.add_argument("--norm", choices=("none", "subject", "both"), default="both")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tuning2_eegmat.json")
    ap.add_argument("--max-tune", type=int, default=10**9,
                    help="subsample training epochs for the inner search only")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers for the inner search")
    ap.add_argument("--models", nargs="+",
                    default=list(QUANTUM_MODELS + CLASSICAL_MODELS))
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the significance threshold (use this to "
                         "correct for every test already run on the same data)")
    a = ap.parse_args()

    X = np.load(f"results/X_{a.tag}.npy")
    y = np.load(f"results/y_{a.tag}.npy")
    g = np.load(f"results/g_{a.tag}.npy", allow_pickle=True)
    models = tuple(a.models)
    conds = ["none", "subject"] if a.norm == "both" else [a.norm]
    alpha = a.alpha if a.alpha is not None else 0.05 / len(conds)
    print(f"[tune2] X={X.shape} subjects={len(np.unique(g))} budget={a.budget}/model "
          f"outer={a.outer} inner={a.inner} alpha={alpha}", flush=True)

    out = {}
    for c in conds:
        Xc = subject_normalize(X, g) if c == "subject" else X
        res = run_condition(Xc, y, g, models, a, c)
        dec = report(res, c, alpha)
        out[c] = dict(
            decision=dec,
            pooled=res["pooled"],
            per_subject={m: dict(zip(res["subjects"].tolist(), v.tolist()))
                         for m, v in res["per_subj"].items()},
            chosen=res["chosen"],
        )
        stem = a.out.rsplit("/", 1)[-1].replace(".json", "")
        np.savez(f"results/preds_{stem}_{c}.npz", y=y, g=g,
                 **{m: v for m, v in res["preds"].items()})
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

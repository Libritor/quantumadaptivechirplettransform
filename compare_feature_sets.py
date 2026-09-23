#!/usr/bin/env python
"""Does ACT add anything beyond amplitude on CHB-MIT?

Paired, per patient: every model's nested cross-patient accuracy with
ACT + amplitude features vs amplitude-only features. Both runs use identical
outer folds (same seed, labels and groups), so each patient is scored by the
same kind of model trained on the same other patients -- only the feature set
differs.

Decision rule (fixed before running): ACT pulls its weight for a model if
ACT+amplitude beats amplitude-only with paired Wilcoxon p < 0.05/14 = 0.0036
(7 models x 2 normalisation conditions).
"""
import json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

ALPHA = 0.05 / 14
both = json.load(open("results/tuning_chbmitamp.json"))
amp = json.load(open("results/tuning_chbmitampo.json"))

for cond in ("none", "subject"):
    print(f"\n=== {cond} normalisation: ACT+amplitude vs amplitude-only, 23 patients ===")
    print(f"{'model':<9} {'amp only':>9} {'ACT+amp':>9} {'diff':>7} {'better':>7} {'p':>8}  verdict")
    for m in both[cond]["pooled"]:
        pb, pa = both[cond]["per_subject"][m], amp[cond]["per_subject"][m]
        subs = sorted(pb)
        a = np.array([pa[s] for s in subs]); b = np.array([pb[s] for s in subs])
        d = b - a
        p = wilcoxon(b, a).pvalue if not np.allclose(d, 0) else 1.0
        v = ("ACT helps" if d.mean() > 0 and p < ALPHA
             else "ACT hurts" if d.mean() < 0 and p < ALPHA else "no detectable difference")
        print(f"{m:<9} {amp[cond]['pooled'][m]*100:>8.1f}% {both[cond]['pooled'][m]*100:>8.1f}% "
              f"{(both[cond]['pooled'][m]-amp[cond]['pooled'][m])*100:>+6.1f} "
              f"{(d>0).sum():>3}/{len(d)} {p:>8.4f}  {v}")

# Which features does the ANOVA selector reach for, from the combined pool?
X = np.load("results/X_chbmitamp.npy"); y = np.load("results/y_chbmitamp.npy")
names = json.load(open("results/names_chbmitamp.json"))
Z = StandardScaler().fit_transform(VarianceThreshold(0.0).fit_transform(X))
keep = VarianceThreshold(0.0).fit(X).get_support()
kept = [n for n, k in zip(names, keep) if k]
order = np.argsort(-np.nan_to_num(f_classif(Z, y)[0]))
print("\nshare of selected features that are ACT (rest are amplitude), full-data ranking:")
for k in (4, 6, 8, 10, 20, 30):
    top = [kept[i] for i in order[:k]]
    act = sum(not s.endswith(("_log_rms", "_log_linelength")) for s in top)
    print(f"   top {k:>2}: {act:>2} ACT / {k-act:>2} amplitude")
print("   best-ranked ACT feature is #%d overall" %
      (1 + next(i for i, j in enumerate(order)
                if not kept[j].endswith(("_log_rms", "_log_linelength")))))


# ---------------------------------------------------------------------------
# Follow-up: univariate selection never picks ACT, but ACT could still carry
# information that only helps in combination. Test that directly: no selection,
# all features, regularised models, same outer folds, same decision rule.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    g = np.load("results/g_chbmitamp.npy", allow_pickle=True)
    Xamp = X[:, 234:]
    import sys
    sys.path.insert(0, ".")
    from tune2 import subject_normalize

    def per_patient(model, XX):
        cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=0)
        pred = np.empty_like(y)
        for tr, te in cv.split(XX, y, g):
            pipe = Pipeline([("v", VarianceThreshold(0.0)), ("s", StandardScaler()),
                             ("c", model)]).fit(XX[tr], y[tr])
            pred[te] = pipe.predict(XX[te])
        subs = np.unique(g)
        return (pred == y).mean(), np.array([(pred[g == s] == y[g == s]).mean() for s in subs])

    print("\nALL features, no selection (does ACT help in combination?):")
    print(f"{'cond':<8} {'model':<22} {'amp only':>9} {'ACT+amp':>9} {'diff':>7} {'better':>7} {'p':>8}")
    for cond in ("none", "subject"):
        XA = subject_normalize(X, g) if cond == "subject" else X
        XO = subject_normalize(Xamp, g) if cond == "subject" else Xamp
        for name, mk in (("logreg C=0.01", lambda: LogisticRegression(C=0.01, max_iter=5000)),
                         ("logreg C=0.1", lambda: LogisticRegression(C=0.1, max_iter=5000)),
                         ("svm_rbf C=1", lambda: SVC(C=1.0, gamma="scale"))):
            ao, po = per_patient(mk(), XO)
            ab, pb = per_patient(mk(), XA)
            d = pb - po
            p = wilcoxon(pb, po).pvalue if not np.allclose(d, 0) else 1.0
            print(f"{cond:<8} {name:<22} {ao*100:>8.1f}% {ab*100:>8.1f}% {(ab-ao)*100:>+6.1f} "
                  f"{(d>0).sum():>3}/{len(d)} {p:>8.4f}")

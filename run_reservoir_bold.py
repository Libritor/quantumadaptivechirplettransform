#!/usr/bin/env python
"""Predict the UNSMOOTHED future: 30 two-second windows, not their average.

Every forecast so far predicted an averaged feature vector, so smoothness was
built into the question and the best possible answer was a shrunk, conservative
one (squared error is minimised by the conditional MEAN -- "boldness" is not a
free knob, it is a guaranteed loss under that metric).

This asks the harder question instead: given rich 60-s context (the K principal
components of the flattened 30 x 270 sub-window block, as in
run_reservoir_rich.py), predict all 30 two-second windows of the NEXT minute at
full detail -- 30 x 270 = 8100 numbers. Shrinking to the mean can no longer be
right, because the mean is not the target.

Baseline: persistence at the same resolution -- the last observed 2-s window
repeated 30 times.

Reported per model:
  skill        1 - MSE/MSE(persistence) on the unsmoothed future
  corr         mean correlation between predicted and actual fluctuation
               (scale-invariant, so it measures shape rather than size)
  skill_bold   the same predictions rescaled so their variance matches the
               training target's ("bold" outputs). This exists to show the
               trade-off explicitly: it should LOWER skill while leaving corr
               unchanged, which is what "make it bolder" costs under MSE.

Models: qrc, qrc_J0 (no couplings), esn (matched features), linear.
Readouts: per-person and pooled over all 23 people.
Pre-registered: qrc vs esn and qrc vs qrc_J0 on `skill`, paired Wilcoxon over
people, both readouts, alpha = 0.05/4 = 0.0125.
"""
import argparse, json, time
import numpy as np
import torch
from scipy.stats import wilcoxon

from qbe.reservoir import Standardizer
from qbe.sequence import make_samples
from run_reservoir_rich import (DEV, HOR, KMAX, STEP_S, W_STEP, load_rich, padded,
                                states, ALPHAS)
from train_seq import PERSONS

NSUB = W_STEP            # 30 sub-windows of 2 s in the next minute
NFEAT = 270


def prepare_bold(person, K):
    P = load_rich(person)
    n = len(P["fid"]); split = n // 2
    v = P["valid"]
    tr = np.arange(split)[v[:split]]
    Xr = P["rich"]                                   # (n, 30*270) raw sub-windows
    rmu, rsd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-6
    Xs = np.clip((Xr - rmu) / rsd, -10, 10).astype(np.float32)
    _, _, Vt = np.linalg.svd(Xs[tr] - Xs[tr].mean(0), full_matrices=False)
    u = (Xs - Xs[tr].mean(0)) @ Vt[:K].T
    u = (u / (u[tr].std(0) + 1e-6)).astype(np.float32); u[~v] = 0.0
    idx, y, elig = make_samples(P, history=KMAX, horizon=1, step_s=STEP_S)
    ok = v[idx] & v[idx + 1]
    trn = np.flatnonzero(idx + 1 < split); tst = np.flatnonzero(idx - KMAX + 1 >= split)
    fid_s = P["fid"][idx[trn]]
    onset_files = {k for k, _ in P["onsets"]}
    cand = np.array(sorted(set(fid_s.tolist()) - onset_files))
    rng = np.random.default_rng(0)
    nv = max(1, int(round(0.15 * len(set(fid_s.tolist())))))
    vf = set(rng.choice(cand, min(nv, len(cand)), replace=False).tolist()) if len(cand) else set()
    isval = np.isin(fid_s, list(vf))
    val, trn = trn[isval], trn[~isval]
    return dict(person=person, X=torch.as_tensor(Xs, device=DEV), u=u, fid=P["fid"],
                idx=idx, v=torch.as_tensor(v, device=DEV),
                trn=trn[ok[trn]], val=val[ok[val]], tst=tst[ok[tst]])


def design(P, sel, S):
    """Readout inputs: the K components of the current rich step, plus its last
    2-s window (so a persistence-like correction is expressible). The full 8100
    numbers are NOT fed in -- that would be 8676 features against ~500 samples
    per person. The targets still use the full detail."""
    t = torch.as_tensor(P["idx"][sel], device=DEV)
    last = P["X"][t].view(len(sel), NSUB, NFEAT)[:, -1]
    base = [torch.as_tensor(P["u"][P["idx"][sel]], device=DEV), last]
    if S is not None:
        base.append(S[t])
    return torch.cat(base, 1)


def target(P, sel):
    """Next step's 30 sub-windows, and persistence = last observed 2-s window."""
    t = torch.as_tensor(P["idx"][sel], device=DEV)
    Y = P["X"][t + 1]                                   # (B, 30*270)
    last = P["X"][t].view(len(sel), NSUB, NFEAT)[:, -1]  # (B, 270)
    pers = last[:, None, :].expand(-1, NSUB, -1).reshape(len(sel), -1)
    return Y, pers


def solve(G, Y, alphas):
    A = G.double().T @ G.double(); B = G.double().T @ Y.double()
    eye = torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
    return {a: torch.linalg.solve(A + a * eye, B).float() for a in alphas}, A, B


def metrics(pred, Y, pers):
    num = float(((pred - Y) ** 2).sum()); den = float(((pers - Y) ** 2).sum())
    d, p = (Y - pers), (pred - pers)
    dn = d - d.mean(0); pn = p - p.mean(0)
    corr = float((dn * pn).sum(0).div((dn.norm(dim=0) * pn.norm(dim=0)).clamp(min=1e-9)).mean())
    g = float(d.std() / p.std().clamp(min=1e-9))
    bold = pers + p * g
    return dict(skill=1 - num / max(den, 1e-12), corr=corr,
                skill_bold=1 - float(((bold - Y) ** 2).sum()) / max(den, 1e-12), gain=g)


def run_person(P, kind, K):
    S = states(kind, P, K)
    G = {k: Standardizer().fit(design(P, P["trn"], S)) for k in ("s",)}["s"]
    D = {k: G(design(P, P[k], S)) for k in ("trn", "val", "tst")}
    Ytr, ptr = target(P, P["trn"])
    Ws, _, _ = solve(D["trn"], Ytr - ptr, ALPHAS)
    Yv, pv = target(P, P["val"])
    best = min(ALPHAS, key=lambda a: float(((pv + D["val"] @ Ws[a] - Yv) ** 2).mean()))
    Yt, pt = target(P, P["tst"])
    return metrics(pt + D["tst"] @ Ws[best], Yt, pt) | {"alpha": best, "n_feat": D["trn"].shape[1]}


def run_pooled(people, kind, K):
    per, A, B = {}, None, None
    for P in people:
        S = states(kind, P, K)
        st = Standardizer().fit(design(P, P["trn"], S))
        D = {k: st(design(P, P[k], S)) for k in ("trn", "val", "tst")}
        Y, p0 = target(P, P["trn"])
        a = D["trn"].double().T @ D["trn"].double(); b = D["trn"].double().T @ (Y - p0).double()
        A = a if A is None else A + a
        B = b if B is None else B + b
        per[P["person"]] = D
    eye = torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
    best, bw = None, None
    for al in ALPHAS:
        W = torch.linalg.solve(A + al * eye, B).float()
        num = den = 0.0
        for P in people:
            Yv, pv = target(P, P["val"])
            pred = pv + per[P["person"]]["val"] @ W
            num += float(((pred - Yv) ** 2).sum()); den += float(((pv - Yv) ** 2).sum())
        sk = 1 - num / max(den, 1e-12)
        if best is None or sk > best:
            best, bw = sk, W
    return {P["person"]: metrics(target(P, P["tst"])[1] + per[P["person"]]["tst"] @ bw,
                                 *target(P, P["tst"])) for P in people}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=48); a = ap.parse_args()
    t0 = time.time(); K = a.K
    people = [prepare_bold(p, K) for p in PERSONS]
    print(f"prepared {len(people)} people, K={K}, target = {NSUB}x{NFEAT} unsmoothed "
          f"({time.time()-t0:.0f}s)", flush=True)
    models = ("qrc", "qrc_J0", "esn", "linear")
    out = {"per_person": {}, "pooled": {}}
    for m in models:
        out["per_person"][m] = {P["person"]: run_person(P, m, K) for P in people}
        out["pooled"][m] = run_pooled(people, m, K)
        print(f"[{m}] done ({time.time()-t0:.0f}s)", flush=True)
    for tag in ("per_person", "pooled"):
        sk = {m: np.array([out[tag][m][P["person"]]["skill"] for P in people]) for m in models}
        print(f"\n=== {tag} readout, K={K} ===")
        for m in models:
            r = [out[tag][m][P["person"]] for P in people]
            print(f"  {m:<8} skill {sk[m].mean():+.4f}  corr {np.mean([x['corr'] for x in r]):+.4f}"
                  f"  skill if made bold {np.mean([x['skill_bold'] for x in r]):+.4f}"
                  f"  (boldness x{np.mean([x['gain'] for x in r]):.2f})")
        for lab, q, c in (("qrc vs esn   ", "qrc", "esn"), ("qrc vs qrc_J0", "qrc", "qrc_J0")):
            d = sk[q] - sk[c]
            p = wilcoxon(sk[q], sk[c]).pvalue if not np.allclose(d, 0) else 1.0
            print(f"  {lab} diff {d.mean():+.4f} better {(d>0).sum()}/{len(d)} p={p:.4f} -> "
                  f"{'BENEFIT' if d.mean() > 0 and p < 0.0125 else 'none'}")
    json.dump(out, open(f"results/reservoir_bold_K{K}.json", "w"), indent=2, default=str)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

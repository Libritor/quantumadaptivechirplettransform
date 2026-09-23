#!/usr/bin/env python
"""Does richer per-step input revive the reservoirs at coarse resolution?

The 60-s run averaged 30 two-second windows into one step, which smoothed away
the within-step detail -- and both reservoirs (quantum and classical) then added
nothing over a plain linear readout. This keeps that detail instead of
discarding it: a 60-s step is the 30 windows FLATTENED (30 x 270 = 8100 numbers)
and reduced to its top K principal components, fitted on the training half.

    K = 8    same information as the averaged step (the previous setup)
    K = 24, 48, 100   progressively more within-step structure

The forecast target is unchanged -- the next step's averaged 270 features -- so
skill is directly comparable with the 60-s row of the resolution sweep.

Wider input needs more injection points, so the number of parallel 8-qubit
reservoirs scales with K (spatial multiplexing, Nakajima et al. 2019):
R = max(4, K // 4), each fed its own random projection through 2 input qubits.
The classical ESN is given N = 44*R nodes, so both readouts have the SAME number
of features at every K and a quantum win cannot come from extra width alone.

A 60-s step leaves only ~500 training samples per person, far too few for a
readout with hundreds of features, so every model is ALSO fitted with a POOLED
readout: one ridge trained on all 23 people's training halves at once
(~11,500 samples), each person standardised with their own training statistics.
The resolution sweep already showed pooling is worth +0.148 skill at 60 s.

PRE-REGISTERED (before any rich-input model was scored)
  at each K in {8, 24, 48, 100}, for BOTH readouts (per-person and pooled),
  paired Wilcoxon across the 23 people:
    A  qrc vs esn (matched features)   forecast skill
    B  qrc vs qrc_J0 (no couplings)    forecast skill
  alpha = 0.05 / 16 = 0.003125. Reservoir physics is fixed at the configuration
  the 60-s sweep already selected (h=3.0, W=0.05, dt=3.0) so that K is the only
  thing varying; ridge alpha is still chosen per person on validation.
"""
import json, time
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from qbe.reservoir import ESN, MultiReservoir, Standardizer, ridge_fit
from qbe.sequence import make_samples
from train_seq import PERSONS

DEV = "cuda"
W_STEP = 30                      # 30 x 2 s = 60 s steps
STEP_S = 2.0 * W_STEP
HOR = [1]                        # +60 s, as in the 60-s sweep row
KMAX = 24
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)
KS = (8, 24, 48, 100)
QRC_CFG = dict(h=3.0, W=0.05, dt=3.0)


def load_rich(person):
    """Per 60-s step: averaged features (targets) and the flattened sub-windows."""
    F, R, V, I, T, FID, onsets, ends = [], [], [], [], [], [], [], []
    off = 0
    for case in PERSONS[person]:
        z = np.load(f"results/continuous/{case}.npz")
        feats, valid, ictal, start, fid = z["feats"], z["valid"], z["ictal"], z["start_s"], z["file_id"]
        for k in np.unique(fid):
            m = np.flatnonzero(fid == k)
            ns = len(m) // W_STEP
            if ns == 0:
                continue
            idx = m[: ns * W_STEP].reshape(ns, W_STEP)
            v = valid[idx]
            avg = np.where(v[..., None], feats[idx], 0.0).sum(1) / np.maximum(v.sum(1), 1)[:, None]
            F.append(avg.astype(np.float32))
            R.append(np.where(v[..., None], feats[idx], 0.0).reshape(ns, -1).astype(np.float32))
            V.append(v.sum(1) >= int(np.ceil(0.6 * W_STEP)))
            I.append(ictal[idx].any(1)); T.append(start[idx[:, 0]])
            FID.append(np.full(ns, off + k))
        for k, s, e in z["seizures"]:
            onsets.append((off + int(k), s)); ends.append((off + int(k), e))
        off += int(fid.max()) + 1
    return dict(F=np.concatenate(F), rich=np.concatenate(R), valid=np.concatenate(V),
                ictal=np.concatenate(I), t=np.concatenate(T), fid=np.concatenate(FID),
                onsets=onsets, ends=ends)


def prepare_base(person, kmax=max(KS)):
    """Everything independent of K, including ONE truncated SVD per person.

    The K principal components are nested (the top 8 are a prefix of the top
    100), so this is computed once per person on the GPU and sliced per K.
    Recomputing a full 8100-dim CPU SVD per person per K kept 22 cores busy for
    hours and produced identical components.
    """
    P = load_rich(person)
    n = len(P["fid"]); split = n // 2
    v = P["valid"]
    tr = np.arange(split)[v[:split]]
    mu, sd = P["F"][tr].mean(0), P["F"][tr].std(0) + 1e-6
    Z = np.clip((P["F"] - mu) / sd, -10, 10).astype(np.float32); Z[~v] = 0.0
    # K principal components of the FLATTENED sub-windows, fitted on training half
    Xr = P["rich"]
    rmu, rsd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-6
    Xs = np.clip((Xr - rmu) / rsd, -10, 10).astype(np.float32)
    Xt = torch.as_tensor(Xs, device=DEV)
    trt = torch.as_tensor(tr, device=DEV)
    ctr = Xt[trt].mean(0)
    q = min(kmax + 10, len(tr) - 1, Xt.shape[1])
    _, _, Vt = torch.svd_lowrank(Xt[trt] - ctr, q=q, niter=4)
    proj = (Xt - ctr) @ Vt[:, :kmax]
    proj = (proj / (proj[trt].std(0) + 1e-6)).cpu().numpy().astype(np.float32)
    proj[~v] = 0.0
    del Xt
    idx, y, elig = make_samples(P, history=KMAX, horizon=max(HOR), step_s=STEP_S)
    fut = idx[:, None] + np.array(HOR)
    ok = v[idx] & v[fut].all(1)
    trn = np.flatnonzero(idx + max(HOR) < split); tst = np.flatnonzero(idx - KMAX + 1 >= split)
    fid_s = P["fid"][idx[trn]]
    onset_files = {k for k, _ in P["onsets"]}
    cand = np.array(sorted(set(fid_s.tolist()) - onset_files))
    rng = np.random.default_rng(0)
    nv = max(1, int(round(0.15 * len(set(fid_s.tolist())))))
    vf = set(rng.choice(cand, min(nv, len(cand)), replace=False).tolist()) if len(cand) else set()
    isval = np.isin(fid_s, list(vf))
    val, trn = trn[isval], trn[~isval]
    return dict(person=person, Z=torch.as_tensor(Z, device=DEV), v=torch.as_tensor(v, device=DEV),
                proj=proj, fid=P["fid"], idx=idx, y=y, elig=elig,
                trn=trn[ok[trn]], val=val[ok[val]], tst=tst)


def with_K(base, K):
    """A view of a prepared person using only the first K components."""
    return dict(base, u=base["proj"][:, :K])


def padded(P):
    fids = np.unique(P["fid"])
    spans = [np.flatnonzero(P["fid"] == k) for k in fids]
    T = max(len(s) for s in spans)
    U = np.zeros((len(spans), T, P["u"].shape[1]), np.float32)
    for i, s in enumerate(spans):
        U[i, :len(s)] = P["u"][s]
    return torch.as_tensor(U, device=DEV), spans


def states(kind, P, K):
    if kind == "linear":
        return None
    R = max(4, K // 4)
    U, spans = padded(P)
    if kind in ("qrc", "qrc_J0"):
        S = MultiReservoir(K, R=R, n=8, n_in=2, seed=0, J_zero=(kind == "qrc_J0"),
                           device=DEV, **QRC_CFG).run_continuous(U)
    elif kind == "esn":
        S = ESN(K, N=44 * R, seed=0, rho=0.5, leak=0.3, in_scale=0.5, device=DEV).run_continuous(U)
    else:
        raise ValueError(kind)
    out = torch.zeros(len(P["u"]), S.shape[-1], device=DEV)
    for i, sp in enumerate(spans):
        out[sp] = S[i, :len(sp)]
    return out


def design(P, sel, S):
    t = torch.as_tensor(P["idx"][sel], device=DEV)
    base = [P["Z"][t], P["v"][t][:, None].float(), torch.as_tensor(P["u"][P["idx"][sel]], device=DEV)]
    if S is not None:
        base.append(S[t])
    return torch.cat(base, 1)


def targets(P, sel):
    t = torch.as_tensor(P["idx"][sel], device=DEV)
    fut = t[:, None] + torch.as_tensor(HOR, device=DEV)
    last = P["Z"][t]
    return (P["Z"][fut] - last[:, None]).reshape(len(sel), -1), P["Z"][fut], P["v"][fut], last


def fit_eval(P, S):
    G = {k: design(P, P[k], S) for k in ("trn", "val", "tst")}
    st = Standardizer().fit(G["trn"]); G = {k: st(g) for k, g in G.items()}
    Y = targets(P, P["trn"])[0]
    A = G["trn"].double().T @ G["trn"].double(); B = G["trn"].double().T @ Y.double()
    eye = torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
    yv, fut_v, fv_v, last_v = targets(P, P["val"])
    best = None
    for a in ALPHAS:
        W = torch.linalg.solve(A + a * eye, B).float()
        e = ((G["val"] @ W - yv) ** 2).mean().item()
        if best is None or e < best[0]:
            best = (e, a, W)
    _, alpha, W = best
    _, fut_t, fv_t, last_t = targets(P, P["tst"])
    pt = last_t[:, None] + (G["tst"] @ W).view(len(P["tst"]), len(HOR), -1)
    m = fv_t[..., None].float()
    skill = 1 - float(((pt - fut_t) ** 2 * m).sum() / ((last_t[:, None] - fut_t) ** 2 * m).sum().clamp(min=1e-12))
    auc = None
    y, el = P["y"], P["elig"]
    etr = P["trn"][el[P["trn"]]]
    if y[etr].sum() > 0:
        pos = {k: i for i, k in enumerate(P["trn"])}
        rows = torch.as_tensor([pos[k] for k in etr], device=DEV)
        wr = ridge_fit(G["trn"][rows], torch.as_tensor(y[etr], device=DEV)[:, None].float(), alpha)
        et = el[P["tst"]]; yy = y[P["tst"]][et]
        if 0 < yy.sum() < len(yy):
            auc = float(roc_auc_score(yy, (G["tst"] @ wr)[:, 0].cpu().numpy()[et]))
    return dict(skill=skill, auc=auc, alpha=alpha, n_feat=int(G["trn"].shape[1]))


def pooled_eval(people, kind, K):
    """One ridge over all people's training halves; returns per-person test skill."""
    per, A, B = {}, None, None
    for P in people:
        S = states(kind, P, K)
        G = {k: design(P, P[k], S) for k in ("trn", "val", "tst")}
        st = Standardizer().fit(G["trn"]); G = {k: st(g) for k, g in G.items()}
        Y = targets(P, P["trn"])[0]
        a = G["trn"].double().T @ G["trn"].double(); b = G["trn"].double().T @ Y.double()
        A = a if A is None else A + a
        B = b if B is None else B + b
        per[P["person"]] = G
    eye = torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
    best = None
    for al in ALPHAS:
        W = torch.linalg.solve(A + al * eye, B).float()
        num = den = 0.0
        for P in people:
            _, fut_v, fv_v, last_v = targets(P, P["val"])
            pv = last_v[:, None] + (per[P["person"]]["val"] @ W).view(len(P["val"]), len(HOR), -1)
            m = fv_v[..., None].float()
            num += float(((pv - fut_v) ** 2 * m).sum()); den += float(((last_v[:, None] - fut_v) ** 2 * m).sum())
        sk = 1 - num / max(den, 1e-12)
        if best is None or sk > best[0]:
            best = (sk, al, W)
    _, al, W = best
    out = {}
    for P in people:
        _, fut_t, fv_t, last_t = targets(P, P["tst"])
        pt = last_t[:, None] + (per[P["person"]]["tst"] @ W).view(len(P["tst"]), len(HOR), -1)
        m = fv_t[..., None].float()
        out[P["person"]] = 1 - float(((pt - fut_t) ** 2 * m).sum() /
                                     ((last_t[:, None] - fut_t) ** 2 * m).sum().clamp(min=1e-12))
    return out


def main():
    t0 = time.time()
    ALPHA = 0.05 / 16
    res = {}
    bases = [prepare_base(p) for p in PERSONS]
    print(f"prepared {len(bases)} people ({time.time()-t0:.0f}s)", flush=True)
    for K in KS:
        t1 = time.time()
        people = [with_K(b, K) for b in bases]
        res[K] = {m: {} for m in ("qrc", "qrc_J0", "esn", "linear")}
        for m in res[K]:
            for P in people:
                res[K][m][P["person"]] = fit_eval(P, states(m, P, K))
        R = max(4, K // 4)
        sk = {m: np.array([res[K][m][P["person"]]["skill"] for P in people]) for m in res[K]}
        nf = res[K]["qrc"][people[0]["person"]]["n_feat"]
        print(f"\n=== K={K} inputs, R={R} reservoirs, {nf} readout features "
              f"({time.time()-t1:.0f}s) ===")
        for m in ("qrc", "qrc_J0", "esn", "linear"):
            au = np.array([res[K][m][P["person"]]["auc"] for P in people], dtype=float)
            au = au[~np.isnan(au)]
            print(f"  {m:<8} skill {sk[m].mean():+.4f} / median {np.median(sk[m]):+.4f}   "
                  f"risk AUC {au.mean() if len(au) else float('nan'):.3f} ({len(au)})")
        pool = {m: pooled_eval(people, m, K) for m in ("qrc", "qrc_J0", "esn", "linear")}
        skp = {m: np.array([pool[m][P["person"]] for P in people]) for m in pool}
        res[K]["_pooled"] = {m: pool[m] for m in pool}
        print("  pooled readout (all 23 people trained together):")
        for m in ("qrc", "qrc_J0", "esn", "linear"):
            print(f"    {m:<8} skill {skp[m].mean():+.4f} / median {np.median(skp[m]):+.4f}")
        for tag, M in (("per-person", sk), ("pooled    ", skp)):
            for lab, q, c in (("A qrc vs esn   ", "qrc", "esn"), ("B qrc vs qrc_J0", "qrc", "qrc_J0")):
                d = M[q] - M[c]
                p = wilcoxon(M[q], M[c]).pvalue if not np.allclose(d, 0) else 1.0
                print(f"  [{tag}] {lab} diff {d.mean():+.4f} better {(d>0).sum()}/{len(d)} "
                      f"p={p:.4f} -> {'BENEFIT' if d.mean() > 0 and p < ALPHA else 'none'}")
        json.dump({str(k): {m: {p: v for p, v in d.items()} for m, d in r.items()}
                   for k, r in res.items()},
                  open("results/reservoir_rich_chbmit.json", "w"), indent=2, default=str)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

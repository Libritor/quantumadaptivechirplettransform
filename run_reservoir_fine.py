#!/usr/bin/env python
"""Reservoirs at native 2-s resolution, run continuously along each recording.

WHY THIS DIFFERS FROM run_reservoir.py
--------------------------------------
Two things there could hide fast temporal structure from the reservoir:
  * features were averaged into 10-s steps, which removes fast dynamics;
  * the restart protocol re-prepared the reservoir for every prediction, so no
    memory could accumulate beyond the 5-minute window.
Here the reservoir is driven at the native 2-s rate and runs CONTINUOUSLY along
each recording file, carrying its state, with a readout after every step. That
is both the standard reservoir setup and cheaper (one step per prediction
instead of one window per prediction). The classical twins run continuously too,
so the protocol is identical for all of them.

Targets are unchanged from run_reservoir.py -- the same six points across the
next minute (+10..+60 s) -- so forecast skill is directly comparable; only the
input resolution and the memory protocol change.

Reading a quantum state without collapsing it is idealised. `qrc_shots` adds
1000-shot sampling noise; it does NOT model measurement back-action, which is
why run_reservoir.py's restart protocol (hardware-realistic, and there within
0.002 skill of the idealised readout) remains the realism check.

PRE-REGISTERED (written before any fine-resolution model was scored)
  models   qrc, qrc_shots, qrc_J0 (no couplings), esn, nvar, linear
  grids    12 configurations per family, selected on mean VALIDATION skill
  splits   identical people, halves and validation files as train_seq.py
  decisions, paired Wilcoxon across 23 people, alpha = 0.01 (5 tests):
    T1 qrc       vs best classical   forecast skill
    T2 qrc       vs best classical   risk AUC
    T3 qrc_shots vs best classical   forecast skill
    T4 qrc_shots vs best classical   risk AUC
    T5 qrc       vs qrc_J0           forecast skill
  secondary (descriptive): a POOLED readout trained on all people's training
  halves at once, to test whether the structure needs more data to be learnable.
"""
import itertools, json, time
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from qbe.reservoir import ESN, MultiReservoir, Standardizer, ridge_fit, shot_noise
from qbe.sequence import load_person, make_samples
from train_seq import PERSONS

DEV = "cuda"
# Resolution is a parameter: --step-windows W gives steps of 2*W seconds.
# Horizons always cover the next minute, as close to +10..+60 s as the step
# allows: 2-s steps -> [5,10,..,30]; 10-s -> [1..6]; 30-s -> [1,2].
# NOTE: skill is NOT comparable ACROSS resolutions -- a coarser step is a
# smoother target and a different persistence baseline. Only the comparisons
# within a resolution (quantum vs classical, entanglement control) are.
STEP_WINDOWS = 1
STEP_S = 2.0
KMAX = 24                     # longest NVAR tap span, also the sample guard
HOR = [5, 10, 15, 20, 25, 30]


def set_resolution(step_windows):
    global STEP_WINDOWS, STEP_S, HOR
    STEP_WINDOWS = int(step_windows)
    STEP_S = 2.0 * STEP_WINDOWS
    HOR = sorted({max(1, round(h / STEP_S)) for h in (10, 20, 30, 40, 50, 60)})
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)
GRIDS = {
    "qrc": [dict(h=h, W=W, dt=dt) for h, W, dt in itertools.product((0.3, 1.0, 3.0), (0.05, 1.0), (1.0, 3.0))],
    "esn": [dict(rho=r, leak=l, in_scale=s) for r, l, s in itertools.product((0.5, 0.9, 1.2), (0.1, 0.3), (0.5, 1.5))],
    "nvar": [dict(k=k, s=s) for k, s in itertools.product((2, 4, 6, 8), (1, 2, 3))],
}


def prepare(person):
    P = load_person("results/continuous", PERSONS[person], step_windows=STEP_WINDOWS)
    n = len(P["fid"]); split = n // 2
    v = P["valid"]
    tr = np.arange(split)[v[:split]]
    mu, sd = P["F"][tr].mean(0), P["F"][tr].std(0) + 1e-6
    Z = np.clip((P["F"] - mu) / sd, -10, 10).astype(np.float32); Z[~v] = 0.0
    Zc = Z[tr] - Z[tr].mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    u = (Z - Z[tr].mean(0)) @ Vt[:8].T
    u = (u / (u[tr].std(0) + 1e-6)).astype(np.float32); u[~v] = 0.0
    idx, y, elig = make_samples(P, history=KMAX, horizon=max(HOR), step_s=STEP_S)
    fut = idx[:, None] + np.array(HOR)
    ok = v[idx] & v[fut].all(1)
    in_tr = idx + max(HOR) < split
    in_te = idx - KMAX + 1 >= split
    trn, tst = np.flatnonzero(in_tr), np.flatnonzero(in_te)
    fid_s = P["fid"][idx[trn]]
    onset_files = {k for k, _ in P["onsets"]}
    cand = np.array(sorted(set(fid_s.tolist()) - onset_files))
    rng = np.random.default_rng(0)
    nv = max(1, int(round(0.15 * len(set(fid_s.tolist())))))
    vf = set(rng.choice(cand, min(nv, len(cand)), replace=False).tolist()) if len(cand) else set()
    isval = np.isin(fid_s, list(vf))
    val, trn = trn[isval], trn[~isval]
    return dict(person=person, Z=torch.as_tensor(Z, device=DEV), v=torch.as_tensor(v, device=DEV),
                u=u, fid=P["fid"], idx=idx, y=y, elig=elig, ok=ok,
                trn=trn[ok[trn]], val=val[ok[val]], tst=tst,
                info=dict(steps=n, files=len(np.unique(P["fid"])),
                          onsets=len(P["onsets"]), hours=round(n * STEP_S / 3600, 1)))


def padded_files(P):
    """(n_files, Tmax, 8) padded input tensor + per-file global indices."""
    fids = np.unique(P["fid"])
    spans = [np.flatnonzero(P["fid"] == k) for k in fids]
    T = max(len(s) for s in spans)
    U = np.full((len(spans), T, P["u"].shape[1]), 0.0, np.float32)
    for i, s in enumerate(spans):
        U[i, :len(s)] = P["u"][s]
    return torch.as_tensor(U, device=DEV), spans


def states(kind, cfg, P, seed=0):
    """Reservoir state after every step, as (n_steps, F); None for 'linear'."""
    if kind == "linear":
        return None
    if kind == "nvar":
        k, s = cfg["k"], cfg["s"]
        u = torch.as_tensor(P["u"], device=DEV)
        taps = [torch.roll(u, j * s, 0) for j in range(k)]
        lin = torch.cat(taps, 1)
        iu = torch.triu_indices(lin.shape[1], lin.shape[1])
        return torch.cat([lin, (lin[:, :, None] * lin[:, None, :])[:, iu[0], iu[1]]], 1)
    U, spans = padded_files(P)
    if kind in ("qrc", "qrc_shots", "qrc_J0"):
        mr = MultiReservoir(8, R=4, n=8, n_in=2, seed=seed, J_zero=(kind == "qrc_J0"), device=DEV, **cfg)
        S = mr.run_continuous(U)
    else:
        S = ESN(8, N=176, seed=seed, device=DEV, **cfg).run_continuous(U)
    out = torch.zeros(len(P["u"]), S.shape[-1], device=DEV)
    for i, sp in enumerate(spans):
        out[sp] = S[i, :len(sp)]
    if kind == "qrc_shots":
        g = torch.Generator(device=DEV); g.manual_seed(7)
        out = shot_noise(out, 1000, g)
    return out


def design(P, sel, S):
    t = torch.as_tensor(P["idx"][sel], device=DEV)
    last = torch.cat([P["Z"][t], P["v"][t][:, None].float()], 1)
    return last if S is None else torch.cat([last, S[t]], 1)


def targets(P, sel):
    t = torch.as_tensor(P["idx"][sel], device=DEV)
    fut = t[:, None] + torch.as_tensor(HOR, device=DEV)
    last = P["Z"][t]
    return (P["Z"][fut] - last[:, None]).reshape(len(sel), -1), P["Z"][fut], P["v"][fut], last


def fit_eval(P, S, final, ret_parts=False):
    G, Y = {}, {}
    for k in (["trn", "val"] + (["tst"] if final else [])):
        G[k] = design(P, P[k], S)
    st = Standardizer().fit(G["trn"])
    G = {k: st(g) for k, g in G.items()}
    Ytr = targets(P, P["trn"])[0]
    A = G["trn"].double().T @ G["trn"].double()
    B = G["trn"].double().T @ Ytr.double()
    eye = torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
    yv, fut_v, fv_v, last_v = targets(P, P["val"])
    best = None
    for a in ALPHAS:
        W = torch.linalg.solve(A + a * eye, B).float()
        e = ((G["val"] @ W - yv) ** 2).mean().item()
        if best is None or e < best[0]:
            best = (e, a, W)
    _, alpha, W = best
    pv = last_v[:, None] + (G["val"] @ W).view(len(P["val"]), len(HOR), -1)
    m = fv_v[..., None].float()
    out = dict(alpha=alpha, val_skill=1 - float(((pv - fut_v) ** 2 * m).sum() /
                                                ((last_v[:, None] - fut_v) ** 2 * m).sum().clamp(min=1e-12)))
    if ret_parts:
        out["parts"] = (G, A, B, alpha)
    if not final:
        return out
    _, fut_t, fv_t, last_t = targets(P, P["tst"])
    pt = last_t[:, None] + (G["tst"] @ W).view(len(P["tst"]), len(HOR), -1)
    m = fv_t[..., None].float()
    out["skill"] = 1 - float(((pt - fut_t) ** 2 * m).sum() / ((last_t[:, None] - fut_t) ** 2 * m).sum().clamp(min=1e-12))
    out["auc"] = None
    y, el = P["y"], P["elig"]
    etr = P["trn"][el[P["trn"]]]
    if y[etr].sum() > 0:
        pos = {k: i for i, k in enumerate(P["trn"])}
        rows = torch.as_tensor([pos[k] for k in etr], device=DEV)
        wr = ridge_fit(G["trn"][rows], torch.as_tensor(y[etr], device=DEV)[:, None].float(), alpha)
        et = el[P["tst"]]; yy = y[P["tst"]][et]
        if 0 < yy.sum() < len(yy):
            out["auc"] = float(roc_auc_score(yy, (G["tst"] @ wr)[:, 0].cpu().numpy()[et]))
    return out


def pooled(people, final, fam_cfg):
    """One readout trained on ALL people's training halves at once.

    Each person's design matrix is standardised with their own training stats
    first (so feature scales are comparable), then the normal equations are
    accumulated across people -- no need to hold every design matrix at once.
    """
    per = {}
    A = B = None
    for P in people:
        S = states(fam_cfg[0], fam_cfg[1], P)
        G = {k: design(P, P[k], S) for k in ("trn", "val", "tst")}
        st = Standardizer().fit(G["trn"])
        G = {k: st(g) for k, g in G.items()}
        Y = targets(P, P["trn"])[0]
        a = G["trn"].double().T @ G["trn"].double()
        b = G["trn"].double().T @ Y.double()
        A = a if A is None else A + a
        B = b if B is None else B + b
        per[P["person"]] = G
    eye = torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
    best = None
    for al in ALPHAS:
        W = torch.linalg.solve(A + al * eye, B).float()
        num = den = 0.0
        for P in people:
            yv, fut_v, fv_v, last_v = targets(P, P["val"])
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
    return out, al


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-windows", type=int, default=1, help="step = 2*W seconds")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    set_resolution(a.step_windows)
    tag = a.tag or f"{int(STEP_S)}s"
    out_json = f"results/reservoir_{tag}_chbmit.json"
    t0 = time.time()
    people = [prepare(p) for p in PERSONS]
    print(f"prepared {len(people)} people in {time.time()-t0:.0f}s; "
          f"{sum(P['info']['steps'] for P in people):,} steps of {STEP_S:.0f}s; "
          f"horizons {[int(h*STEP_S) for h in HOR]} s", flush=True)
    chosen, log = {}, {"select": {}}
    for fam, grid in GRIDS.items():
        sc = []
        for i, cfg in enumerate(grid):
            t1 = time.time()
            vs = [fit_eval(P, states(fam, cfg, P), final=False)["val_skill"] for P in people]
            sc.append(float(np.mean(vs)))
            print(f"  [select {fam} {i+1}/{len(grid)}] {cfg} val {sc[-1]:+.4f} ({time.time()-t1:.0f}s)", flush=True)
        chosen[fam] = (grid[int(np.argmax(sc))], max(sc))
        log["select"][fam] = [dict(cfg=c, val=s) for c, s in zip(grid, sc)]
        print(f"[{fam}] selected {chosen[fam][0]} (val {chosen[fam][1]:+.4f})", flush=True)
    bc = max(("esn", "nvar"), key=lambda f: chosen[f][1])
    print(f"best classical reservoir by VALIDATION: {bc}", flush=True)
    final = {"qrc": chosen["qrc"][0], "qrc_shots": chosen["qrc"][0], "qrc_J0": chosen["qrc"][0],
             "esn": chosen["esn"][0], "nvar": chosen["nvar"][0], "linear": {}}
    res = {m: {} for m in final}
    for m, cfg in final.items():
        for P in people:
            res[m][P["person"]] = fit_eval(P, states(m, cfg, P), final=True)
        print(f"[test {m}] done ({time.time()-t0:.0f}s)", flush=True)
    log.update(test=res, chosen=final, best_classical=bc,
               info={P["person"]: P["info"] for P in people})
    json.dump(log, open(out_json, "w"), indent=2, default=str)

    ps = [P["person"] for P in people]
    sk = {m: np.array([res[m][p]["skill"] for p in ps]) for m in res}
    au = {m: np.array([np.nan if res[m][p]["auc"] is None else res[m][p]["auc"] for p in ps]) for m in res}
    print(f"\n{'model':<10} {'forecast skill mean/median':>28} {'risk AUC (n)':>16}")
    for m in res:
        a = au[m][~np.isnan(au[m])]
        print(f"{m:<10} {sk[m].mean():>+14.4f} / {np.median(sk[m]):+.4f} "
              f"{a.mean() if len(a) else float('nan'):>12.3f} ({len(a)})")
    print(f"\nPRE-REGISTERED DECISIONS (alpha 0.01), best classical = {bc}:")
    for name, q, c, M, lab in (("T1", "qrc", bc, sk, "forecast"), ("T2", "qrc", bc, au, "risk AUC"),
                               ("T3", "qrc_shots", bc, sk, "forecast"), ("T4", "qrc_shots", bc, au, "risk AUC"),
                               ("T5", "qrc", "qrc_J0", sk, "forecast")):
        ok = ~np.isnan(M[q]) & ~np.isnan(M[c])
        d = M[q][ok] - M[c][ok]
        p = wilcoxon(M[q][ok], M[c][ok]).pvalue if ok.sum() > 1 and not np.allclose(d, 0) else 1.0
        print(f"  {name} {q:<10} vs {c:<9} {lab:<9} diff {d.mean():+.4f} better {(d>0).sum()}/{ok.sum()} "
              f"p={p:.4f} -> {'QUANTUM BENEFIT' if d.mean() > 0 and p < 0.01 else 'no demonstrated quantum benefit'}")
    # secondary, descriptive: one readout pooled over all people's training halves
    print("\nPOOLED readout (all people trained together), mean test skill:")
    pool = {}
    for m in ("qrc", "esn", "linear"):
        sk_p, al = pooled(people, True, (m, final[m]))
        pool[m] = sk_p
        arr = np.array([sk_p[p] for p in ps])
        d = arr - sk[m]
        print(f"  {m:<7} pooled {arr.mean():+.4f} vs per-person {sk[m].mean():+.4f} "
              f"(diff {d.mean():+.4f}, better for {(d>0).sum()}/{len(ps)}, alpha {al:g})")
    log["pooled"] = pool
    json.dump(log, open(out_json, "w"), indent=2, default=str)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

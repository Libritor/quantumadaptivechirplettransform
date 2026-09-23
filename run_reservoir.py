#!/usr/bin/env python
"""Quantum vs classical reservoirs for per-person EEG forecasting (CHB-MIT).

PRE-REGISTERED PROTOCOL (written before any reservoir was scored on EEG)
------------------------------------------------------------------------
Same task, people and splits as train_seq.py: per person, first half of the
recordings trains (random seizure-free files held out for validation), second
half tests. From the last 5 min (30 x 10-s steps) predict the next 1 min of
all 270 features and whether a seizure starts within 5 min.

Reservoir input: the step's first 8 principal components (fitted on the
person's training half). Every model's readout sees [last step's features,
reservoir features] and is the same ridge regression, alpha chosen per person
on validation.

Models, each family given EXACTLY 12 configurations; the configuration with the
best mean validation forecast skill across people is used for testing:
  qrc        4 parallel 8-qubit disordered-Ising reservoirs (176 features),
             grid h {0.3,1,3} x W {0.05,1} x dt {1,3}, exact expectations
  qrc_real   the selected qrc config, realistic readout: restart protocol plus
             1000-shot noise on every expectation value
  qrc_J0     the selected qrc config with all couplings removed (no entanglement)
  esn        176-node echo state network, rho {0.5,0.9,1.2} x leak {0.3,1} x
             input scale {0.5,1.5}
  nvar       NG-RC, taps k {2,4,6,8} x stride s {1,2,3}
  linear     readout on the last step's features only (no reservoir)

"Best classical reservoir" = whichever of esn / nvar has the higher mean
VALIDATION skill (chosen without looking at test).

Decisions (alpha = 0.01 = 0.05 / 5), paired Wilcoxon across people:
  T1 qrc      vs best classical   forecast skill
  T2 qrc      vs best classical   risk AUC
  T3 qrc_real vs best classical   forecast skill
  T4 qrc_real vs best classical   risk AUC
  T5 qrc      vs qrc_J0           forecast skill (does entanglement contribute?)
A quantum benefit is claimed only for a test with a higher mean and p < 0.01.
"""
import itertools, json, time
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from qbe.reservoir import ESN, NVAR, MultiReservoir, Standardizer, ridge_fit, shot_noise
from qbe.sequence import HISTORY, HORIZON
from train_seq import PERSONS, build

DEV = "cuda"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
GRIDS = {
    "qrc": [dict(h=h, W=W, dt=dt) for h, W, dt in itertools.product((0.3, 1.0, 3.0), (0.05, 1.0), (1.0, 3.0))],
    "esn": [dict(rho=r, leak=l, in_scale=s) for r, l, s in itertools.product((0.5, 0.9, 1.2), (0.3, 1.0), (0.5, 1.5))],
    "nvar": [dict(k=k, s=s) for k, s in itertools.product((2, 4, 6, 8), (1, 2, 3))],
}


def prepare(person):
    D = build(person, "results/continuous")
    Z, v = D["Z"], D["v"]
    n = len(Z); split = n // 2
    tr = np.arange(split)[v[:split]]
    Zc = Z[tr] - Z[tr].mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    pcs = (Z - Z[tr].mean(0)) @ Vt[:8].T
    pcs = pcs / (pcs[tr].std(0) + 1e-6)
    pcs[~v] = 0.0
    idx = D["idx"]
    # a sample is usable for fitting if its last history step and all targets are valid
    fut = idx[:, None] + np.arange(1, HORIZON + 1)
    ok = v[idx] & v[fut].all(1)
    return dict(person=person, D=D, pcs=torch.as_tensor(pcs, dtype=torch.float32, device=DEV),
                Z=torch.as_tensor(Z, device=DEV), v=torch.as_tensor(v, device=DEV),
                ok=ok, info=D["info"])


def windows(P, sel):
    t = torch.as_tensor(P["D"]["idx"][sel], device=DEV)
    return P["pcs"][t[:, None] + torch.arange(-HISTORY + 1, 1, device=DEV)]


def reservoir_features(kind, cfg, P, sel, seed=0):
    if kind == "linear":
        return None
    U = windows(P, sel)
    if kind in ("qrc", "qrc_real", "qrc_J0"):
        mr = MultiReservoir(8, R=4, n=8, n_in=2, seed=seed, J_zero=(kind == "qrc_J0"), device=DEV, **cfg)
        F = mr.run_windows(U)
        if kind == "qrc_real":
            g = torch.Generator(device=DEV); g.manual_seed(1234 + len(sel))
            F = shot_noise(F, 1000, g)
        return F
    if kind == "esn":
        return ESN(8, N=176, seed=seed, device=DEV, **cfg).run_windows(U)
    if kind == "nvar":
        return NVAR(**cfg).features(U)


def design(P, sel, F):
    t = torch.as_tensor(P["D"]["idx"][sel], device=DEV)
    last = torch.cat([P["Z"][t], P["v"][t][:, None].float()], 1)
    return last if F is None else torch.cat([last, F], 1)


def targets(P, sel):
    t = torch.as_tensor(P["D"]["idx"][sel], device=DEV)
    fut = t[:, None] + torch.arange(1, HORIZON + 1, device=DEV)
    last = P["Z"][t]
    return (P["Z"][fut] - last[:, None]).reshape(len(sel), -1), P["Z"][fut], P["v"][fut], last


def evaluate(kind, cfg, P, final):
    D = P["D"]
    trn = D["trn"][P["ok"][D["trn"]]]
    val = D["val"][P["ok"][D["val"]]]
    sets = {"trn": trn, "val": val}
    if final:
        sets["tst"] = D["tst"]
    Fs = {k: reservoir_features(kind, cfg, P, s) for k, s in sets.items()}
    X = {k: design(P, sets[k], Fs[k]) for k in sets}
    st = Standardizer().fit(X["trn"])
    G = {k: st(X[k]) for k in X}
    Ytr = targets(P, trn)[0]
    yv_d, fut_v, fv_v, last_v = targets(P, val)
    best = None
    for a in ALPHAS:
        W = ridge_fit(G["trn"], Ytr, a)
        e = ((G["val"] @ W - yv_d) ** 2).mean().item()
        if best is None or e < best[0]:
            best = (e, a, W)
    _, alpha, W = best
    pv = (last_v[:, None] + (G["val"] @ W).view(len(val), HORIZON, -1))
    m = fv_v[..., None].float()
    val_skill = 1 - float(((pv - fut_v) ** 2 * m).sum() / ((last_v[:, None] - fut_v) ** 2 * m).sum().clamp(min=1e-12))
    out = dict(val_skill=val_skill, alpha=alpha)
    if not final:
        return out
    tst = sets["tst"]
    _, fut_t, fv_t, last_t = targets(P, tst)
    pt = last_t[:, None] + (G["tst"] @ W).view(len(tst), HORIZON, -1)
    m = fv_t[..., None].float()
    out["skill"] = 1 - float(((pt - fut_t) ** 2 * m).sum() / ((last_t[:, None] - fut_t) ** 2 * m).sum().clamp(min=1e-12))
    # risk: ridge on the 0/1 onset label over eligible training samples, same alpha
    y, el = D["y"], D["elig"]
    etr = trn[el[trn]]
    ntr_pos = float(y[etr].sum())
    out["auc"] = None
    if ntr_pos > 0:
        pos_in = {k: i for i, k in enumerate(trn)}
        rows = torch.as_tensor([pos_in[k] for k in etr], device=DEV)
        wr = ridge_fit(G["trn"][rows], torch.as_tensor(y[etr], device=DEV)[:, None].float(), alpha)
        et = el[tst]; yy = y[tst][et]
        if 0 < yy.sum() < len(yy):
            score = (G["tst"] @ wr)[:, 0].cpu().numpy()[et]
            out["auc"] = float(roc_auc_score(yy, score))
    return out


def main():
    t0 = time.time()
    people = [prepare(p) for p in PERSONS]
    print(f"prepared {len(people)} people in {time.time()-t0:.0f}s", flush=True)
    log = {"select": {}, "test": {}}
    chosen = {}
    for fam, grid in GRIDS.items():
        scores = []
        for i, cfg in enumerate(grid):
            t1 = time.time()
            vs = [evaluate(fam, cfg, P, final=False)["val_skill"] for P in people]
            scores.append(float(np.mean(vs)))
            print(f"  [select {fam} {i+1}/{len(grid)}] {cfg}  mean val skill {scores[-1]:+.4f}  "
                  f"({time.time()-t1:.0f}s)", flush=True)
        chosen[fam] = (grid[int(np.argmax(scores))], max(scores))
        log["select"][fam] = [dict(cfg=c, val=s) for c, s in zip(grid, scores)]
        print(f"[{fam}] selected {chosen[fam][0]} (val {chosen[fam][1]:+.4f})", flush=True)
    best_classical = max(("esn", "nvar"), key=lambda f: chosen[f][1])
    print(f"best classical reservoir by VALIDATION: {best_classical}", flush=True)
    final = {"qrc": chosen["qrc"][0], "qrc_real": chosen["qrc"][0], "qrc_J0": chosen["qrc"][0],
             "esn": chosen["esn"][0], "nvar": chosen["nvar"][0], "linear": {}}
    res = {m: {} for m in final}
    for m, cfg in final.items():
        for P in people:
            res[m][P["person"]] = evaluate(m, cfg, P, final=True)
        print(f"[test {m}] done", flush=True)
    log["test"] = res; log["chosen"] = final; log["best_classical"] = best_classical
    json.dump(log, open("results/reservoir_chbmit.json", "w"), indent=2, default=str)

    persons = [P["person"] for P in people]
    sk = {m: np.array([res[m][p]["skill"] for p in persons]) for m in res}
    au = {m: np.array([np.nan if res[m][p]["auc"] is None else res[m][p]["auc"] for p in persons]) for m in res}
    print(f"\n{'model':<9} {'forecast skill mean/median':>28} {'beats persistence':>18} {'risk AUC (n)':>14}")
    for m in res:
        a = au[m][~np.isnan(au[m])]
        print(f"{m:<9} {sk[m].mean():>+14.4f} / {np.median(sk[m]):+.4f} {int((sk[m] > 0).sum()):>10}/{len(persons)}"
              f" {a.mean() if len(a) else float('nan'):>10.3f} ({len(a)})")
    print(f"\nPRE-REGISTERED DECISIONS (alpha 0.01), best classical = {best_classical}:")
    tests = [("T1", "qrc", best_classical, sk, "forecast skill"), ("T2", "qrc", best_classical, au, "risk AUC"),
             ("T3", "qrc_real", best_classical, sk, "forecast skill"), ("T4", "qrc_real", best_classical, au, "risk AUC"),
             ("T5", "qrc", "qrc_J0", sk, "forecast skill")]
    for name, q, c, M, lab in tests:
        ok = ~np.isnan(M[q]) & ~np.isnan(M[c])
        d = M[q][ok] - M[c][ok]
        p = wilcoxon(M[q][ok], M[c][ok]).pvalue if ok.sum() > 1 and not np.allclose(d, 0) else 1.0
        verdict = "QUANTUM BENEFIT" if d.mean() > 0 and p < 0.01 else "no demonstrated quantum benefit"
        print(f"  {name} {q:<8} vs {c:<7} {lab:<15} diff {d.mean():+.4f}  better {(d>0).sum()}/{ok.sum()}  "
              f"p = {p:.4f} -> {verdict}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

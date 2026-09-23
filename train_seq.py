#!/usr/bin/env python
"""Per-person sequence forecasting on CHB-MIT: train on the first half, test on the second.

PRE-REGISTERED PROTOCOL (written before any model was trained)
--------------------------------------------------------------
Task. Per person, 10-s steps in time order; first half trains (~15% of its
seizure-free files, chosen at random, are held out for early stopping), second
half tests. From the last 5 min, predict (a) the
next 1 min of all 270 standardised ACT+amplitude features and (b) whether a
seizure starts within the next 5 min.

Models (see qbe/sequence.py): lstm (reference), qin_zz vs its twin qin_z
(placement A: fixed circuit features as input), qlstm vs its twin qlstm_twin
(placement B: trainable circuits inside the gates). 3 seeds each; a person's
score for a model is the mean over its seeds.

Metrics, on the test half only.
  forecast skill = 1 - MSE(model) / MSE(persistence), over every valid target
                   entry; persistence = "the next minute equals the last step".
                   0 = no better than persistence, 1 = perfect.
  risk AUC       = ROC AUC of the seizure-onset probability over eligible test
                   samples, for persons with >= 1 onset in BOTH halves.

Decision. For each placement, the quantum model beats its twin on a metric iff
its mean over persons is higher AND a paired Wilcoxon test across persons gives
p < 0.0125 (0.05 over 2 placements x 2 metrics). Anything else is reported as
"no demonstrated quantum benefit".
"""
import argparse, json, os, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from qbe.sequence import (HISTORY, HORIZON, SeqModel, load_person, make_samples,
                          pauli_input_features)

MODELS = ("lstm", "qin_zz", "qin_z", "qlstm", "qlstm_twin")
PERSONS = {f"chb{i:02d}": [f"chb{i:02d}"] for i in range(1, 25) if i != 21}
PERSONS["chb01"] = ["chb01", "chb21"]          # same patient, 1.5 years apart


def build(person, cont_dir):
    P = load_person(cont_dir, PERSONS[person])
    n = len(P["fid"]); split = n // 2
    tr = np.arange(split)
    v = P["valid"]
    mu = P["F"][tr][v[tr]].mean(0); sd = P["F"][tr][v[tr]].std(0) + 1e-6
    Z = np.clip((P["F"] - mu) / sd, -10, 10).astype(np.float32)
    Z[~v] = 0.0
    idx, y, elig = make_samples(P)
    in_tr = idx + HORIZON < split
    in_te = idx - HISTORY + 1 >= split
    trn = np.flatnonzero(in_tr); tst = np.flatnonzero(in_te)
    # Early-stopping set: ~15% of the training-half FILES, drawn at random from
    # files with no seizure onset, so every training seizure stays in training
    # (holding out the last 15% by time can swallow a person's only training
    # seizure). Whole files, so no window overlaps between train and val.
    fid_s = P["fid"][idx[trn]]
    onset_files = {k for k, _ in P["onsets"]}
    cand = np.array(sorted(set(fid_s.tolist()) - onset_files))
    rng = np.random.default_rng(0)
    nv = max(1, int(round(0.15 * len(set(fid_s.tolist())))))
    vfiles = set(rng.choice(cand, min(nv, len(cand)), replace=False).tolist()) if len(cand) else set()
    isval = np.isin(fid_s, list(vfiles))
    val, trn = trn[isval], trn[~isval]
    # placement-A inputs: PCA fitted on valid training steps only
    Ztr = Z[tr][v[tr]]
    _, _, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)
    pca = Vt.T
    proj = Ztr @ pca[:, :8]
    lo, hi = np.percentile(proj, 1, 0), np.percentile(proj, 99, 0)
    extra = {}
    for kind, name in (("zz", "qin_zz"), ("z", "qin_z")):
        Q = pauli_input_features(Z, pca, lo, hi, kind)
        qm, qs = Q[tr][v[tr]].mean(0), Q[tr][v[tr]].std(0) + 1e-6
        Q = (Q - qm) / qs; Q[~v] = 0.0
        extra[name] = Q.astype(np.float32)
    base = np.concatenate([Z, v[:, None].astype(np.float32)], 1)
    # step index at which each seizure starts, to count onsets per half
    onset_step = []
    for k, s in P["onsets"]:
        m = np.flatnonzero((P["fid"] == k) & (P["t"] + 10.0 > s))
        if len(m):
            onset_step.append(int(m[0]))
    onset_step = np.array(onset_step)
    info = dict(steps=int(n), train_h=round(split * 10 / 3600, 1),
                test_h=round((n - split) * 10 / 3600, 1),
                onsets_train=int((onset_step < split).sum()),
                onsets_test=int((onset_step >= split).sum()),
                risk_pos_train=int(y[trn].sum() + y[val].sum()), risk_pos_test=int(y[tst].sum()))
    return dict(Z=Z, v=v, base=base, extra=extra, idx=idx, y=y, elig=elig,
                trn=trn, val=val, tst=tst, info=info)


def tensors(D, kind, dev):
    X = D["base"] if kind not in D["extra"] else np.concatenate([D["base"], D["extra"][kind]], 1)
    return (torch.as_tensor(X, device=dev), torch.as_tensor(D["Z"], device=dev),
            torch.as_tensor(D["v"], device=dev))


def batch(Xs, Zs, Vs, idx, sel, dev):
    t = torch.as_tensor(idx[sel], device=dev)
    hist = t[:, None] + torch.arange(-HISTORY + 1, 1, device=dev)
    fut = t[:, None] + torch.arange(1, HORIZON + 1, device=dev)
    return Xs[hist], Zs[t], Zs[fut], Vs[fut]


def run_model(D, kind, seed, dev, max_epochs=40, patience=5, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    Xs, Zs, Vs = tensors(D, kind, dev)
    model = SeqModel(kind, Xs.shape[1], Zs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.as_tensor(D["y"], device=dev); el = torch.as_tensor(D["elig"], device=dev)
    npos = float(D["y"][D["trn"]][D["elig"][D["trn"]]].sum())
    nneg = float((D["elig"][D["trn"]]).sum()) - npos
    use_risk = npos > 0
    bce = nn.BCEWithLogitsLoss(reduction="none",
                               pos_weight=torch.tensor(min(nneg / max(npos, 1), 200.0), device=dev))

    def loss_on(sel, train):
        x, last, fut, fv = batch(Xs, Zs, Vs, D["idx"], sel, dev)
        pred, logit = model(x, last)
        m = fv[..., None].float()
        lf = ((pred - fut) ** 2 * m).sum() / (m.sum() * fut.shape[-1]).clamp(min=1)
        e = el[torch.as_tensor(sel, device=dev)].float()
        lr_ = (bce(logit, y[torch.as_tensor(sel, device=dev)]) * e).sum() / e.sum().clamp(min=1)
        return lf + (lr_ if use_risk else 0.0)

    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = np.random.permutation(D["trn"])
        for i in range(0, len(perm), bs):
            opt.zero_grad(); loss_on(perm[i:i + bs], True).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(D["val"][i:i + 1024], False).item()
                                for i in range(0, len(D["val"]), 1024)]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    se_m = se_p = cnt = 0.0; probs = []
    with torch.no_grad():
        for i in range(0, len(D["tst"]), 1024):
            sel = D["tst"][i:i + 1024]
            x, last, fut, fv = batch(Xs, Zs, Vs, D["idx"], sel, dev)
            pred, logit = model(x, last)
            m = fv[..., None].float()
            se_m += float(((pred - fut) ** 2 * m).sum()); se_p += float(((last[:, None] - fut) ** 2 * m).sum())
            cnt += float(m.sum())
            probs.append(torch.sigmoid(logit).cpu().numpy())
    probs = np.concatenate(probs)
    te = D["tst"]; ee = D["elig"][te]; yy = D["y"][te][ee]
    auc = (float(roc_auc_score(yy, probs[ee])) if use_risk and 0 < yy.sum() < len(yy) else None)
    core = getattr(model.rnn, "core_params", lambda: None)()
    return dict(skill=1.0 - se_m / max(se_p, 1e-12), auc=auc, epochs=ep,
                params=sum(p.numel() for p in model.parameters()), core_params=core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--cont", default="results/continuous")
    ap.add_argument("--out", default="results/seq")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    D = build(a.person, a.cont)
    res = dict(person=a.person, info=D["info"], n_train=len(D["trn"]), n_val=len(D["val"]),
               n_test=len(D["tst"]), models={})
    print(f"[{a.person}] {D['info']} train={len(D['trn'])} val={len(D['val'])} "
          f"test={len(D['tst'])}  build {time.time()-t0:.0f}s", flush=True)
    for kind in a.models:
        res["models"][kind] = []
        for seed in range(a.seeds):
            t1 = time.time()
            r = run_model(D, kind, seed, dev)
            r["seconds"] = time.time() - t1
            res["models"][kind].append(r)
            print(f"[{a.person}] {kind:<11} seed {seed}: skill {r['skill']:+.4f}  "
                  f"auc {r['auc'] if r['auc'] is None else round(r['auc'],3)}  "
                  f"ep {r['epochs']}  {r['seconds']:.0f}s", flush=True)
        json.dump(res, open(f"{a.out}/{a.person}.json", "w"), indent=2)
    print(f"[{a.person}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

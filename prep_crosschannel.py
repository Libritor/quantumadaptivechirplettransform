#!/usr/bin/env python
"""Add cross-channel features to the CHB-MIT feature sets (classical ACT and QACT).

Per-channel features treat the 18 channels as independent; a seizure is partly
defined by spatial synchrony and spread, which that discards. This recomputes
both transforms while KEEPING the per-atom parameters, derives the 18
cross-channel features (qbe/crosschannel.py), and writes augmented sets:

    chbmit_cc            classical ACT 234 + cross 18
    chbmitqactonly_cc    QACT (refined, sampled) 234 + cross 18
    chbmitamp_cc         classical ACT 234 + amplitude 36 + cross 18
    chbmitqact_cc        QACT 234 + amplitude 36 + cross 18

Atom arrays are saved too (results/atoms_*.npy) so cross-channel features can be
redesigned without re-running either transform.
"""
import argparse, json, time
import numpy as np

from qbe import acquire
from qbe.crosschannel import CROSS_FEATURES, cross_channel_features, qact_params_to_atoms
from qbe.features import feature_names
from qbe.gpu_features import extract_gpu
from qbe.qact import QACT, features_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kc/datasets/chbmit")
    ap.add_argument("--order", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    ep, groups = acquire.load_chbmit(a.root, seed=a.seed, verify=False)
    E, C, N = ep.X.shape
    names = feature_names(ep.channels)
    cc_names = list(CROSS_FEATURES)
    print(f"[cc] {E} epochs x {C} channels ({time.time()-t0:.0f}s)", flush=True)

    # ---- classical ACT, keeping atoms ----
    t1 = time.time()
    Xc, _, atoms_c = extract_gpu(ep, order=a.order, verbose=False, return_atoms=True)
    cc_c = cross_channel_features(atoms_c)
    print(f"[cc] classical ACT + atoms in {time.time()-t1:.0f}s", flush=True)

    # ---- QACT (refined, sampled), keeping atoms ----
    t1 = time.time()
    q = QACT(length=N, fs=ep.fs, shots=a.shots, device="cuda", seed=a.seed,
             refine_steps=a.refine)
    flat = ep.X.reshape(E * C, N).astype(np.float32)
    feat = np.zeros((E * C, 13), np.float32)
    par_all = np.zeros((E * C, a.order, 6), np.float32)
    err_all = np.zeros(E * C)
    for lo in range(0, len(flat), a.batch):
        chunk = flat[lo:lo + a.batch]
        par, err = q.transform(chunk, order=a.order, return_params=True)
        feat[lo:lo + len(chunk)] = features_batch(par, err, ep.fs, N)
        par_all[lo:lo + len(chunk)] = par
        err_all[lo:lo + len(chunk)] = err
    Xq = feat.reshape(E, C * 13)
    atoms_q = qact_params_to_atoms(par_all, ep.fs, N).reshape(E, C, a.order, 5)
    cc_q = cross_channel_features(atoms_q)
    print(f"[cc] QACT + atoms in {time.time()-t1:.0f}s", flush=True)

    np.save("results/atoms_classical.npy", atoms_c.astype(np.float32))
    np.save("results/atoms_qact.npy", atoms_q.astype(np.float32))
    sets = {
        "chbmit_cc": (np.hstack([Xc, cc_c]), names + cc_names),
        "chbmitqactonly_cc": (np.hstack([Xq, cc_q]), names + cc_names),
        "chbmitamp_cc": (np.hstack([Xc, ep.extra, cc_c]), names + ep.extra_names + cc_names),
        "chbmitqact_cc": (np.hstack([Xq, ep.extra, cc_q]), names + ep.extra_names + cc_names),
    }
    for tag, (XX, nm) in sets.items():
        np.save(f"results/X_{tag}.npy", XX)
        np.save(f"results/y_{tag}.npy", ep.y)
        np.save(f"results/g_{tag}.npy", groups)
        json.dump(nm, open(f"results/names_{tag}.json", "w"))
        print(f"[cc] saved {tag}: {XX.shape}", flush=True)
    print(f"[cc] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

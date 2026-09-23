#!/usr/bin/env python
"""CHB-MIT features from the QUANTUM chirplet transform (see docs/QUANTUM_ACT.md).

Mirrors prep_chbmit.py exactly -- same epochs, same gate, same amplitude
features, same 13 features per channel -- but the chirplet decomposition is done
by qbe.qact.QACT (quantum-circuit atoms, QFT frequency axis, measurement-sampled
atom selection) instead of the classical gradient-refined engine.

Writes, aligned 1:1 with results/*_chbmitamp.npy so every downstream experiment
can be re-run unchanged:
    chbmitqactonly   234 QACT features (representation comparison vs chbmit)
    chbmitqact       234 QACT + 36 amplitude (vs chbmitamp)
"""
import argparse, json, time
import numpy as np

from qbe import acquire
from qbe.features import feature_names
from qbe.qact import QACT, features_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kc/datasets/chbmit")
    ap.add_argument("--order", type=int, default=12)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=("sample", "argmax", "proposal"), default="sample",
                    help="argmax = classical selection on the same dictionary (control)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--refine", type=int, default=0,
                    help="parameter-shift refinement steps per atom (0 = grid only)")
    ap.add_argument("--rates3", type=float, nargs="*", default=None,
                    help="cubic chirp rates in Hz/s^2 (idea #3)")
    ap.add_argument("--skews", type=float, nargs="*", default=None,
                    help="skew-normal envelope asymmetries (idea #3)")
    ap.add_argument("--legacy", action="store_true",
                    help="plain matching pursuit with the unnormalised selection "
                         "criterion -- reproduces the pre-parity reference bitwise")
    ap.add_argument("--backfit", type=int, default=1,
                    help="cyclic coordinate-descent passes over the chosen atoms")
    ap.add_argument("--asym", type=float, nargs="*", default=None,
                    help="envelope right/left width ratios (two-width envelope)")
    ap.add_argument("--refine-extra", action="store_true",
                    help="also refine c3, skew and the width ratio")
    ap.add_argument("--candidates", type=int, default=1,
                    help="proposal candidates per atom; >1 with --select proposal (idea #2)")
    a = ap.parse_args()

    t0 = time.time()
    ep, groups = acquire.load_chbmit(a.root, seed=a.seed, verify=False)
    E, C, N = ep.X.shape
    print(f"[prep] {E} epochs x {C} channels, {E*C} channel-epochs ({time.time()-t0:.0f}s)",
          flush=True)
    q = QACT(length=N, fs=ep.fs, shots=a.shots, device="cuda", seed=a.seed,
             refine_steps=a.refine, rates3_hz_s2=a.rates3, skews=a.skews,
             n_candidates=a.candidates,
             omp=not a.legacy, backfit_passes=0 if a.legacy else a.backfit,
             exact_f=not a.legacy, norm_select=not a.legacy,
             asym_ratios=a.asym, refine_extra=a.refine_extra)
    print(f"[prep] QACT: {q.n} qubits + 1 ancilla, {q.dict_size:,} in-band atoms, "
          f"shots={a.shots}, refine_steps={a.refine}"
          + (f" ({q.refiner.evals_per_step} circuit evaluations per step)"
             if q.refiner else ""), flush=True)
    flat = ep.X.reshape(E * C, N).astype(np.float32)
    out = np.zeros((E * C, 13), np.float32)
    t1 = time.time()
    for lo in range(0, len(flat), a.batch):
        chunk = flat[lo:lo + a.batch]
        # vectorised path: parameter arrays straight into features_batch, which
        # matches the per-row channel_features to 7e-15 (verify_features_batch)
        par, err = q.transform(chunk, order=a.order, select=a.select,
                               return_params=True)
        out[lo:lo + len(chunk)] = features_batch(par, err, ep.fs, N)
        if (lo // a.batch) % 50 == 0:
            done = lo + len(chunk)
            r = done / max(time.time() - t1, 1e-9)
            print(f"[prep] {done}/{E*C}  {r:.0f}/s  ~{(E*C-done)/r:.0f}s left", flush=True)
    X = out.reshape(E, C * 13)
    names = feature_names(ep.channels)
    Xa = np.hstack([X, ep.extra])
    for tag, XX, nm in ((f"chbmitqactonly{a.suffix}", X, names),
                        (f"chbmitqact{a.suffix}", Xa, names + ep.extra_names)):
        np.save(f"results/X_{tag}.npy", XX)
        np.save(f"results/y_{tag}.npy", ep.y)
        np.save(f"results/g_{tag}.npy", groups)
        json.dump(nm, open(f"results/names_{tag}.json", "w"))
    print(f"[prep] saved {X.shape} (QACT) and {Xa.shape} (QACT+amplitude); "
          f"post-selection success {q.post_select_frac:.3f}; "
          f"refinement circuit evaluations {q.refine_evals:,}; "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

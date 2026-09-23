#!/usr/bin/env python
"""CHB-MIT: load -> gate/normalise -> GPU chirplet features -> results/*_chbmit.npy

Run on the GPU machine:
    python prep_chbmit.py --root ~/datasets/chbmit
then evaluate with the same pre-registered protocol used for EEGMAT:
    QBE_DEVICE=cuda python tune2.py --tag chbmit --out results/tuning_chbmit.json

This is a FRESH dataset for this project: no model has been scored on it, so
tune2.py's default threshold (p < 0.025, i.e. 0.05 over its two normalisation
conditions) applies without further correction.
"""
import argparse, json, time
import numpy as np
from qbe import acquire
from qbe.gpu_features import extract_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kc/datasets/chbmit")
    ap.add_argument("--order", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    ep, groups = acquire.load_chbmit(a.root, seed=a.seed, verify=not a.no_verify)
    print(f"[prep] loaded in {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    X, names = extract_gpu(ep, order=a.order, batch=a.batch)
    print(f"[prep] GPU ACT on {ep.n_epochs*ep.n_channels} channel-epochs "
          f"in {time.time()-t1:.0f}s", flush=True)
    # Two feature sets, decided BEFORE any quantum model was scored on CHB-MIT:
    #   chbmit     ACT features only (unit-energy epochs -> fractions)
    #   chbmitamp  ACT + per-channel log RMS and log line length. PRIMARY.
    # Every model receives the same set, so the choice cannot tilt the
    # quantum-vs-classical comparison.
    Xa = np.hstack([X, ep.extra])
    for tag, XX, nm in (("chbmit", X, names),
                        ("chbmitamp", Xa, names + ep.extra_names)):
        np.save(f"results/X_{tag}.npy", XX)
        np.save(f"results/y_{tag}.npy", ep.y)
        np.save(f"results/g_{tag}.npy", groups)
        json.dump(nm, open(f"results/names_{tag}.json", "w"))
    json.dump(ep.meta, open("results/meta_chbmit.json", "w"))
    print(f"[prep] saved X={X.shape} (ACT) and {Xa.shape} (ACT+amplitude)  "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

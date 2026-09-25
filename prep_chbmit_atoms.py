#!/usr/bin/env python
"""CHB-MIT chirplet atoms, kept individually (search iteration 4).

Every earlier run stored only the 13-per-channel summary features. The atom-set
kernels need the atoms themselves: for each epoch and channel, the 12 atoms that
classical GPU ACT (the engine behind the headline CHB-MIT result) selects, as
(t_s, f_hz, dur_s, rate_hz_s, energy). Same loader, same seed, same epoch order as
prep_chbmit.py, so labels and patient groups line up with results/X_chbmit*.npy.
"""
import time

import numpy as np

from qbe import acquire
from qbe.gpu_features import extract_gpu


def main():
    t0 = time.time()
    ep, groups = acquire.load_chbmit("/home/kc/datasets/chbmit", seed=0, verify=False)
    print(f"[atoms] loaded {ep.n_epochs} epochs x {ep.n_channels} channels "
          f"({time.time() - t0:.0f}s)", flush=True)
    _, _, atoms = extract_gpu(ep, order=12, batch=4096, return_atoms=True)
    y_ref = np.load("results/y_chbmit.npy")
    g_ref = np.load("results/g_chbmit.npy")
    assert np.array_equal(ep.y, y_ref) and np.array_equal(groups, g_ref), \
        "epoch order differs from results/X_chbmit*.npy"
    np.save("results/atoms_chbmit.npy", atoms)
    print(f"[atoms] saved results/atoms_chbmit.npy {atoms.shape}; labels and groups "
          f"match the existing feature files  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

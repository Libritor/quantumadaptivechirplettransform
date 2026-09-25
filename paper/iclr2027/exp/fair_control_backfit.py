#!/usr/bin/env python
"""The last parity function, given back to the classical engine: backfit.

QACT's parity configuration re-refines every selected atom once against the
residual with its own contribution added back (a cyclic coordinate-descent pass
over atoms), then re-fits all coefficients. The classical act_gpu engine has no
such pass. This script adds it (Adam on the same objective, same step count) so
that the QACT-parity vs classical comparison of fair_control_refine.py is made at
equal function set, and reports the same paired statistics on the same windows.

    python paper/iclr2027/exp/fair_control_backfit.py [--subjects 20] [--orders 6,12]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qbe import act_gpu  # noqa: E402
from qbe.gpu_features import gpu_dictionary_grid  # noqa: E402
from qbe.qact import QACT  # noqa: E402
from fair_control_refine import FS, N, OUT, eeg_windows, timed  # noqa: E402


def decompose_backfit(X, D, max_atoms, steps=60, passes=1):
    R = X.clone()
    atoms = []
    for _ in range(max_atoms):
        p0 = D.best(R)
        atoms.append(act_gpu.refine_batch(R, p0, D.t, D.fs, steps=steps))
        _, recon = act_gpu._joint_refit(X, atoms, D.t)
        R = X - recon
    for _ in range(passes):
        for i in range(len(atoms)):
            gc, gs = act_gpu._quad(D.t, atoms[i][:, 0], atoms[i][:, 1], atoms[i][:, 2], atoms[i][:, 3])
            coef, recon = act_gpu._joint_refit(X, atoms, D.t)
            Ri = X - recon + coef[:, 2 * i][:, None] * gc + coef[:, 2 * i + 1][:, None] * gs
            atoms[i] = act_gpu.refine_batch(Ri, atoms[i], D.t, D.fs, steps=steps)
            _, recon = act_gpu._joint_refit(X, atoms, D.t)
            R = X - recon
    return (R.norm(dim=1) / X.norm(dim=1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--orders", default="6,12")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orders = [int(o) for o in args.orders.split(",")]
    X = eeg_windows(args.subjects)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    D = act_gpu.GPUDictionary(gpu_dictionary_grid(N, FS), N, FS, device=dev)
    from scipy.stats import wilcoxon

    def qact_parity(P):
        q = QACT(length=N, fs=FS, device=dev, seed=0, refine_steps=4, omp=True,
                 backfit_passes=1, exact_f=True, refine_mode="hw")
        return q.transform(X.astype(np.float32), order=P, select="argmax", return_params=True)[1]

    # the classical seed grid stops at fc = 40 Hz and its refinement stays in band; QACT's
    # exact frequency update reads the periodogram peak wherever it is, so 60 Hz mains is
    # representable there and not here. A grid extended to 78 Hz isolates that.
    g = gpu_dictionary_grid(N, FS)
    extra = g[np.isclose(g[:, 1], 1.0)].copy()
    g0 = np.concatenate([g] + [np.column_stack([extra[:, 0], np.full(len(extra), f), extra[:, 2], extra[:, 3]])
                               for f in np.arange(41.0, 79.0, 1.0)], 0)
    D0 = act_gpu.GPUDictionary(g0, N, FS, device=dev)
    engines = {
        "classical Adam-60": lambda P: decompose_backfit(Xt, D, P, 60, passes=0),
        "classical Adam-60 + backfit 1": lambda P: decompose_backfit(Xt, D, P, 60, passes=1),
        "classical Adam-60, fc to 78 Hz": lambda P: decompose_backfit(Xt, D0, P, 60, passes=0),
        "classical Adam-60, fc to 78 Hz + backfit 1": lambda P: decompose_backfit(Xt, D0, P, 60, passes=1),
        "QACT parity (hw-4, exact-f, backfit 1)": qact_parity,
    }
    # how much of each window's energy sits above 45 Hz (mains, which the 40 Hz grid cannot seed)
    spec = np.abs(np.fft.rfft(X, axis=1)) ** 2
    freqs = np.fft.rfftfreq(N, 1 / FS)
    res_drift = (spec[:, freqs >= 45.0].sum(1) / spec.sum(1)).tolist()
    res = {"n_windows": int(len(X)), "orders": orders, "engines": {}, "energy_above_45hz": res_drift}
    for P in orders:
        ref = None
        print(f"\norder {P}:  {'engine':<42}{'mean err':>10}{'vs ref':>9}{'better':>8}{'p':>11}{'ms/window':>11}",
              flush=True)
        for name, fn in engines.items():
            fn(2)
            err, dt = timed(lambda: fn(P))
            if ref is None:
                ref = err
            d = err - ref
            p = float(wilcoxon(err, ref).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
            res["engines"].setdefault(name, {})[str(P)] = dict(
                mean_err=float(err.mean()), median_err=float(np.median(err)),
                rel_vs_ref=float(err.mean() / ref.mean() - 1), better=int((err < ref).sum()),
                wilcoxon_p=p, ms_per_window=1e3 * dt / len(X), err=err.tolist())
            print(f"           {name:<42}{err.mean():>10.4f}{100 * (err.mean() / ref.mean() - 1):>+8.1f}%"
                  f"{(err < ref).sum():>4}/{len(X):<3}{p:>11.2e}{1e3 * dt / len(X):>11.3f}", flush=True)
    json.dump(res, open(OUT / "fair_control_backfit.json", "w"), indent=1)
    print("wrote", OUT / "fair_control_backfit.json")


if __name__ == "__main__":
    main()

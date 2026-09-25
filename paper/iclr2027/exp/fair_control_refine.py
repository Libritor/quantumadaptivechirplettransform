#!/usr/bin/env python
"""The control the denoising study named but did not run: give the CLASSICAL engine
the same 3-point coordinate search QACT's hardware refiner uses, so that the
refinement algorithm and the transform are no longer confounded.

Job: decompose public EEG windows (PhysioNet EEGMMIDB, 512 samples at 160 Hz,
zero-mean, unit-L2) into P atoms with orthogonal matching pursuit, and score the
relative residual norm per window. Every engine sees the same windows.

  classical  Adam-60           act_gpu as shipped: 20,400-atom grid, Adam, 60 steps  (reference)
  classical  Adam-4            the same with the refinement budget QACT uses (4 steps)
  classical  coord-4           the same grid, refinement = 3-point coordinate search,
                               4 sweeps x 4 coordinates x 2 evaluations (QACT's hw rule)
  classical  fine-f, Adam-60   the 0.5 Hz frequency grid (40,800 atoms): the dictionary control
  classical  fine-f, coord-4   both controls at once
  QACT       shift-4           parameter-shift refinement, 4 steps (argmax selection, OMP)
  QACT       hw-4              3-point coordinate search, 4 sweeps (the hardware refiner)
  QACT       parity            hw-4 + exact-f update + one backfit pass (the engine's default)

Paired Wilcoxon signed-rank tests across windows against the reference, at P = 6
and P = 12 atoms. Wall time per window is measured batched on the GPU.

    python paper/iclr2027/exp/fair_control_refine.py [--subjects 20] [--quick]
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

from qbe import act_gpu  # noqa: E402
from qbe.gpu_features import gpu_dictionary_grid  # noqa: E402
from qbe.qact import QACT  # noqa: E402

FS, N = 160.0, 512
CHANNELS = ("Fp1", "Fz", "C3", "Cz", "C4", "Pz", "O1", "O2")
OUT = ROOT / "paper" / "iclr2027" / "results"


def eeg_windows(n_subjects, data_dir=None):
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    data_dir = data_dir or str(Path.home() / "mne_data")
    out = []
    for s in range(1, n_subjects + 1):
        for run in (1, 2):
            f = eegbci.load_data(s, [run], path=data_dir, update_path=False, verbose=False)[0]
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            eegbci.standardize(raw)
            st = (raw.n_times - N) // 2
            X = raw.get_data(picks=list(CHANNELS))[:, st:st + N].astype(np.float64)
            X = X - X.mean(1, keepdims=True)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            out.append(X)
    return np.concatenate(out, 0)


def fine_grid(n, fs):
    T = n / fs
    tc = np.linspace(0.0, T, 17)
    fc = np.arange(1.0, 41.0, 0.5)
    dt = np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0])
    c = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    return np.stack(np.meshgrid(tc, fc, dt, c, indexing="ij"), -1).reshape(-1, 4)


# ----------------------------------------------------------------------------- coordinate search
def refine_coord_batch(X, p, t, fs, sweeps=4):
    """QACT's hardware refinement rule on the classical engine's objective.

    Per sweep, per coordinate (tc, fc, log dt, c), evaluate the captured energy at
    p - d and p + d and keep the best of the three. Step sizes are QACT's own,
    converted to physical units: tc one sample, fc half a bin, log dt 0.05, and the
    chirp step 2/(4N) in sample units = fs^2/N^2 Hz/s.
    """
    n = len(t)
    q = torch.stack([p[:, 0], p[:, 1], torch.log(p[:, 2].clamp(min=1e-3)), p[:, 3]], 1).clone()
    lo = torch.tensor([-0.1 * n / fs, 0.0, np.log(0.008), -1e4], device=X.device, dtype=X.dtype)
    hi = torch.tensor([1.1 * n / fs, 0.49 * fs, np.log(1.5), 1e4], device=X.device, dtype=X.dtype)
    steps = (1.0 / fs, 0.5 * fs / n, 0.05, fs ** 2 / n ** 2)

    def energy(qq):
        gc, gs = act_gpu._quad(t, qq[:, 0], qq[:, 1], torch.exp(qq[:, 2]), qq[:, 3])
        return act_gpu._energy(X, gc, gs)[0]

    e0 = energy(q)
    for _ in range(sweeps):
        for j, d in enumerate(steps):
            for sgn in (-1.0, 1.0):
                cand = q.clone()
                cand[:, j] = (cand[:, j] + sgn * d).clamp(lo[j], hi[j])
                e1 = energy(cand)
                take = e1 > e0
                q[take] = cand[take]
                e0 = torch.where(take, e1, e0)
    return torch.stack([q[:, 0], q[:, 1], torch.exp(q[:, 2]), q[:, 3]], 1)


def decompose_classical(X, D, max_atoms, mode, steps):
    """act_gpu.decompose_batch with a pluggable refiner (OMP on, no stopping rule)."""
    R = X.clone()
    out = []
    for _ in range(max_atoms):
        p0 = D.best(R)
        if mode == "adam":
            p = act_gpu.refine_batch(R, p0, D.t, D.fs, steps=steps)
        else:
            p = refine_coord_batch(R, p0, D.t, D.fs, sweeps=steps)
        out.append(p)
        _, recon = act_gpu._joint_refit(X, out, D.t)
        R = X - recon
    return (R.norm(dim=1) / X.norm(dim=1)).cpu().numpy()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--orders", default="6,12")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orders = [int(o) for o in args.orders.split(",")]
    if args.quick:
        args.subjects, orders = 2, [6]
    X = eeg_windows(args.subjects)
    print(f"{len(X)} windows from {args.subjects} subjects x 2 runs x {len(CHANNELS)} channels, "
          f"orders {orders}, device {dev}", flush=True)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    D = act_gpu.GPUDictionary(gpu_dictionary_grid(N, FS), N, FS, device=dev)
    Df = act_gpu.GPUDictionary(fine_grid(N, FS), N, FS, device=dev)
    engines = {
        "classical Adam-60": lambda P: decompose_classical(Xt, D, P, "adam", 60),
        "classical Adam-4": lambda P: decompose_classical(Xt, D, P, "adam", 4),
        "classical coord-4": lambda P: decompose_classical(Xt, D, P, "coord", 4),
        "classical fine-f Adam-60": lambda P: decompose_classical(Xt, Df, P, "adam", 60),
        "classical fine-f coord-4": lambda P: decompose_classical(Xt, Df, P, "coord", 4),
    }

    def qact(P, mode, exact_f=False, backfit=0):
        q = QACT(length=N, fs=FS, device=dev, seed=0, refine_steps=4, omp=True,
                 backfit_passes=backfit, exact_f=exact_f, refine_mode=mode)
        _, err = q.transform(X.astype(np.float32), order=P, select="argmax", return_params=True)
        return err

    engines.update({
        "QACT shift-4": lambda P: qact(P, "shift"),
        "QACT hw-4": lambda P: qact(P, "hw"),
        "QACT parity (hw-4, exact-f, backfit 1)": lambda P: qact(P, "hw", True, 1),
    })
    from scipy.stats import wilcoxon
    results = {"n_windows": int(len(X)), "fs": FS, "orders": orders, "engines": {}}
    for P in orders:
        ref = None
        print(f"\norder {P}:  {'engine':<42}{'mean err':>10}{'vs ref':>9}{'better':>8}"
              f"{'p':>11}{'ms/window':>11}", flush=True)
        for name, fn in engines.items():
            fn(min(P, 2))                                        # warm-up (kernels, caches)
            err, dt = timed(lambda: fn(P))
            if ref is None:
                ref = err
            d = err - ref
            p = float(wilcoxon(err, ref).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
            rel = float(err.mean() / ref.mean() - 1.0)
            results["engines"].setdefault(name, {})[str(P)] = dict(
                mean_err=float(err.mean()), rel_vs_ref=rel,
                better=int(np.sum(err < ref)), worse=int(np.sum(err > ref)),
                wilcoxon_p=p, ms_per_window=1e3 * dt / len(X), err=err.tolist())
            print(f"           {name:<42}{err.mean():>10.4f}{100 * rel:>+8.1f}%"
                  f"{np.sum(err < ref):>4}/{len(X):<3}{p:>11.2e}{1e3 * dt / len(X):>11.3f}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("fair_control_refine_quick.json" if args.quick else "fair_control_refine.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

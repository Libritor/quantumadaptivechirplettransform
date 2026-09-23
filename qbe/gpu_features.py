"""GPU chirplet feature extraction -- same 13 features per channel as `features`.

`features.extract` runs the Mann-lab `ACTv7` on the CPU at 0.2-0.7 s per
transform, which is hours for a corpus like CHB-MIT (~200k channel-epochs).
This module runs the batched GPU ACT from the sibling EEG-Memristor-ACT project
(`act_gpu.py`: dictionary search as one matmul, Adam off-grid refinement with
autograd, OMP joint refit) on thousands of channel-epochs at once, then computes
the per-channel features in the same tensors, without leaving the GPU.

WHAT IS THE SAME AS THE CPU PATH
  * the 13 features, their definitions and order (`features.PER_CHANNEL_FEATURES`)
  * the oscillatory-atom test (>= MIN_CYCLES cycles, centre in EEG_BAND,
    duration <= epoch, |sweep| <= MAX_RELATIVE_SWEEP * centre frequency)
  * OMP (all coefficients jointly refit after each new atom)

WHAT DIFFERS
  * The engine: Adam refinement instead of L-BFGS-B + Newton. The sibling
    project validated it against the CPU engine (validate_act_gpu.py).
  * The dictionary is far finer -- 20,400 atoms instead of 1,584 -- because on
    the GPU the search is one matmul and costs almost nothing extra.
  Measured on EEGMAT, the two paths do NOT agree feature-by-feature: median
  per-feature correlation 0.01-0.46 (highest for recon_err). That is not a bug
  -- on signals with a known answer the GPU features are exact (planted
  6/10/20 Hz bursts land in the right band within 0.1 Hz, +/-12 Hz/s chirps
  come back with the right sign and a sweep of rate x envelope). It is the
  non-uniqueness of greedy decomposition on 1/f EEG: many different atom sets
  reconstruct a noisy epoch about equally well, and two engines with different
  dictionaries pick different ones. The GPU features were not worse for
  classification (EEGMAT subject-wise logreg 61.3% vs 55.7%). Never mix
  features from the two paths in one model.

UNITS. The GPU engine is parameterised physically: tc and dt in seconds, fc in
Hz, and phase 2*pi*(fc*tau + c*tau^2/2), so `c` is directly the chirp rate in
Hz/s and the sweep across one envelope is c * dt.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import act_gpu
from .features import (BANDS, EEG_BAND, MAX_DURATION_EPOCHS, MAX_RELATIVE_SWEEP,
                       MIN_CYCLES, PER_CHANNEL_FEATURES, feature_names)
from .types import EpochSet


def gpu_dictionary_grid(n: int, fs: float) -> np.ndarray:
    """(M, 4) seed grid of (tc s, fc Hz, dt s, c Hz/s): 17 x 40 x 6 x 5 = 20,400."""
    T = n / fs
    tc = np.linspace(0.0, T, 17)
    fc = np.arange(1.0, 41.0, 1.0)
    dt = np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0])
    c = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    g = np.stack(np.meshgrid(tc, fc, dt, c, indexing="ij"), -1).reshape(-1, 4)
    return g


def _features_from_atoms(atoms, R, X, fs: float, n: int) -> torch.Tensor:
    """Vectorised version of `features.channel_features` over a batch."""
    t = torch.arange(n, device=X.device, dtype=X.dtype) / fs
    P = torch.stack([a["p"] for a in atoms], 1)                    # (B, K, 4)
    ac = torch.stack([a["ac"] for a in atoms], 1)                  # (B, K)
    as_ = torch.stack([a["as_"] for a in atoms], 1)
    keep = torch.stack([a["keep"] for a in atoms], 1)
    B, K, _ = P.shape
    gc, gs = act_gpu._quad(t, P[..., 0].reshape(-1), P[..., 1].reshape(-1),
                           P[..., 2].reshape(-1), P[..., 3].reshape(-1))
    w = ac.reshape(-1, 1) * gc + as_.reshape(-1, 1) * gs
    energy = (w * w).sum(1).reshape(B, K) * keep

    f = P[..., 1].abs()
    dur = P[..., 2]
    sweep = P[..., 3] * dur
    osc = ((f * dur >= MIN_CYCLES) & (f >= EEG_BAND[0]) & (f <= EEG_BAND[1])
           & (dur <= MAX_DURATION_EPOCHS * n / fs)
           & (sweep.abs() <= MAX_RELATIVE_SWEEP * f.clamp(min=1e-9)))
    tot = energy.sum(1).clamp(min=1e-12)
    eo = energy * osc
    osc_e = eo.sum(1)

    cols = [(eo * ((f >= lo) & (f < hi))).sum(1) / tot for lo, hi in BANDS.values()]
    cols.append(osc_e / tot)
    cols.append(1.0 - osc_e / tot)
    wsum = osc_e.clamp(min=1e-12)
    has = osc_e > 1e-12
    mean_f = (eo * f).sum(1) / wsum
    std_f = ((eo * (f - mean_f[:, None]) ** 2).sum(1) / wsum).clamp(min=0).sqrt()
    for v in (mean_f, std_f, (eo * sweep.abs()).sum(1) / wsum,
              (eo * sweep).sum(1) / wsum, (eo * dur).sum(1) / wsum):
        cols.append(torch.where(has, v, torch.zeros_like(v)))
    cols.append(R.norm(dim=1) / X.norm(dim=1).clamp(min=1e-12))
    return torch.stack(cols, 1)


def extract_gpu(
    epochs: EpochSet,
    *,
    order: int = 12,
    steps: int = 60,
    batch: int = 4096,
    device: str = "cuda",
    verbose: bool = True,
    return_atoms: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """GPU drop-in for `features.extract`. Returns (X, feature_names), and with
    `return_atoms` also the physical atom array (E, C, K, 5) that
    `crosschannel.cross_channel_features` consumes."""
    E, C, n = epochs.X.shape
    sig = torch.as_tensor(epochs.X.reshape(E * C, n), dtype=torch.float32)
    D = act_gpu.GPUDictionary(gpu_dictionary_grid(n, epochs.fs), n, epochs.fs,
                              device=device)
    out = np.empty((E * C, len(PER_CHANNEL_FEATURES)), dtype=np.float64)
    atom_out = np.empty((E * C, order, 5), dtype=np.float32) if return_atoms else None
    t0 = time.time()
    for lo in range(0, E * C, batch):
        Xb = sig[lo:lo + batch].to(device)
        atoms, R = act_gpu.decompose_batch(Xb, D, max_atoms=order, steps=steps,
                                           omp=True)
        with torch.no_grad():
            out[lo:lo + len(Xb)] = _features_from_atoms(atoms, R, Xb, epochs.fs,
                                                        n).double().cpu().numpy()
            if return_atoms:
                t = torch.arange(n, device=Xb.device, dtype=Xb.dtype) / epochs.fs
                P = torch.stack([a["p"] for a in atoms], 1)          # (B, K, 4)
                ac = torch.stack([a["ac"] for a in atoms], 1)
                as_ = torch.stack([a["as_"] for a in atoms], 1)
                keep = torch.stack([a["keep"] for a in atoms], 1)
                B, K, _ = P.shape
                gc, gs = act_gpu._quad(t, P[..., 0].reshape(-1), P[..., 1].reshape(-1),
                                      P[..., 2].reshape(-1), P[..., 3].reshape(-1))
                w = ac.reshape(-1, 1) * gc + as_.reshape(-1, 1) * gs
                energy = (w * w).sum(1).reshape(B, K) * keep
                phys = torch.stack([P[..., 0], P[..., 1].abs(), P[..., 2],
                                    P[..., 3], energy], -1)
                atom_out[lo:lo + len(Xb)] = phys.float().cpu().numpy()
        if verbose:
            done = min(lo + batch, E * C)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[gpu_features] {done}/{E*C} channel-epochs  {rate:.0f}/s  "
                  f"~{(E*C-done)/rate:.0f}s left", flush=True)
    X = out.reshape(E, C * len(PER_CHANNEL_FEATURES))
    if return_atoms:
        return X, feature_names(epochs.channels), atom_out.reshape(E, C, order, 5)
    return X, feature_names(epochs.channels)

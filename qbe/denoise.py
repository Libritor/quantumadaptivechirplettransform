"""Denoising with ACT / QACT: classify atoms as artifact, subtract, score vs clean EEG.

This is the task the sibling EEG-Memristor-ACT project built ACT for, and the one
place in this project with actual ground truth. Reconstruction error cannot
measure denoising -- leaving noise in the residual is what a denoiser should do --
so scoring needs a clean reference:

    clean EEG  ->  inject known artifacts  ->  ACT/QACT  ->  drop artifact atoms
                                                          ->  compare with clean

Reused from the sibling project (validated there, not reimplemented here):
  artifacts.inject      blinks, horizontal eye movement, EMG, the subject's own
                        ECG, electrode pops, mains, drift, with realistic
                        per-channel propagation
  metrics.cleaning_metrics / tuning_score
                        relative RMSE overall and on artifact-dominated samples,
                        plus per-band power error in dB. The band term matters:
                        without it a cleaner is free to delete genuine slow EEG to
                        buy time-domain error.

The rule FORM is also theirs -- an atom is artifact if it is mains-like, too
high-frequency, too low-frequency, or too short, each gated on amplitude relative
to that channel's robust sigma. Thresholds are tuned here on the first 18
subjects and frozen for the last 18, as they do.
"""
from __future__ import annotations

import numpy as np
import torch

WIN = 512          # 2 s at 256 Hz, the window both engines are built for
HOP = WIN // 2     # periodic Hann at 50% hop is a partition of unity

GRID = dict(F_HIGH=[22.0, 28.0], K_HIGH=[0.5, 1.0, 2.0], F_LOW=[3.0, 5.0],
            K_LOW=[1.5, 2.5, 4.0], DT_SHORT=[0.015, 0.03], K_SHORT=[3.0, 6.0])


def hann(n: int, device) -> torch.Tensor:
    return torch.hann_window(n, periodic=True, device=device)


def frame(x: np.ndarray) -> tuple[np.ndarray, int]:
    """(ch, n) -> (n_frames*ch, WIN) plus the frame count. Frames are NOT
    normalised: denoising needs the true amplitude."""
    ch, n = x.shape
    nf = 1 + max(0, (n - WIN) // HOP)
    idx = np.arange(nf)[:, None] * HOP + np.arange(WIN)[None, :]
    return x[:, idx].reshape(ch * nf, WIN), nf


def overlap_add(waves: torch.Tensor, ch: int, nf: int, n: int) -> np.ndarray:
    """(ch*nf, WIN) synthesised frames -> (ch, n), Hann-weighted overlap-add."""
    w = hann(WIN, waves.device)
    out = torch.zeros(ch, n, device=waves.device)
    W = (waves * w).reshape(ch, nf, WIN)
    for f in range(nf):
        out[:, f * HOP:f * HOP + WIN] += W[:, f]
    return out.cpu().numpy()


def artifact_mask(phys: dict, sigma: np.ndarray, R: dict) -> np.ndarray:
    """Rule form from the sibling project. `phys` holds per-atom physical
    parameters, `sigma` a robust scale per channel-frame row."""
    fc = np.abs(phys["f_hz"])
    dt = phys["dt_s"]
    amp = phys["amp"]
    s = sigma[:, None]
    sf = 1.0 / (2 * np.pi * np.maximum(dt, 1e-6))     # bandwidth of the envelope
    line = (np.abs(fc - 60.0) < 3.0) & (dt >= 0.1)    # EEGMAT is 50 Hz mains; both
    line |= (np.abs(fc - 50.0) < 3.0) & (dt >= 0.1)   # are treated as mains-like
    high = (fc - sf >= R["F_HIGH"]) & (amp >= R["K_HIGH"] * s)
    low = (fc <= R["F_LOW"]) & (amp >= R["K_LOW"] * s)
    short = (dt <= R["DT_SHORT"]) & (amp >= R["K_SHORT"] * s)
    return line | high | low | short


# ---------------------------------------------------------------------------
# engines: decompose a batch of frames, returning waveforms + physical params
# ---------------------------------------------------------------------------
def decompose_classical(frames: np.ndarray, fs: float, order: int, device="cuda",
                        batch=4096, asym=False, fine_f=False, refine="adam", fc_max=None):
    from . import act_gpu
    from .gpu_features import gpu_dictionary_grid
    grid = gpu_dictionary_grid(WIN, fs)
    if fine_f:
        # Control for the one structural asymmetry left: QACT's dictionary covers
        # EVERY QFT bin (fs/N = 0.5 Hz here) because one transform returns the
        # whole frequency axis, while the classical seed grid uses 1 Hz steps.
        # This rebuilds the classical grid at the same 0.5 Hz resolution, giving
        # it MORE atoms than QACT has.
        import numpy as _np
        T = WIN / fs
        g = _np.stack(_np.meshgrid(_np.linspace(0.0, T, 17),
                                   _np.arange(0.5, 45.5, 0.5),
                                   _np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0]),
                                   _np.array([-20.0, -10.0, 0.0, 10.0, 20.0]),
                                   indexing="ij"), -1).reshape(-1, 4)
        grid = g
    if fc_max is not None:
        # seed grid extended above the fine grid's 45 Hz in 1 Hz steps, so the
        # classical engine can seed a mains (50 Hz) atom the way QACT's exact-f
        # update can
        import numpy as _np
        assert fine_f, "fc_max extends the fine grid"
        T = WIN / fs
        g = _np.stack(_np.meshgrid(_np.linspace(0.0, T, 17),
                                   _np.concatenate([_np.arange(0.5, 45.5, 0.5),
                                                    _np.arange(46.0, fc_max + 1.0, 1.0)]),
                                   _np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0]),
                                   _np.array([-20.0, -10.0, 0.0, 10.0, 20.0]),
                                   indexing="ij"), -1).reshape(-1, 4)
        grid = g
    D = act_gpu.GPUDictionary(grid, WIN, fs, device=device)
    t = torch.arange(WIN, device=device, dtype=torch.float32) / fs
    M = frames.shape[0]
    waves = torch.empty(M, order, WIN, device=device)
    phys = {k: np.zeros((M, order), np.float32) for k in ("f_hz", "dt_s", "amp", "rate")}
    sig = torch.as_tensor(frames, dtype=torch.float32)
    for lo in range(0, M, batch):
        Xb = sig[lo:lo + batch].to(device)
        atoms, _ = act_gpu.decompose_batch(Xb, D, max_atoms=order,
                                           steps=4 if refine == "coord" else 60, omp=True,
                                           asym=asym, refine=refine)
        P = torch.stack([a["p"] for a in atoms], 1)
        ac = torch.stack([a["ac"] for a in atoms], 1)
        as_ = torch.stack([a["as_"] for a in atoms], 1)
        keep = torch.stack([a["keep"] for a in atoms], 1).float()
        B, K, _ = P.shape
        gc, gs = act_gpu._quad(t, P[..., 0].reshape(-1), P[..., 1].reshape(-1),
                               P[..., 2].reshape(-1), P[..., 3].reshape(-1),
                               P[..., 4].reshape(-1) if P.shape[-1] > 4 else None)
        w = (ac.reshape(-1, 1) * gc + as_.reshape(-1, 1) * gs).reshape(B, K, WIN)
        w = w * keep[..., None]
        waves[lo:lo + B] = w
        phys["f_hz"][lo:lo + B] = P[..., 1].abs().cpu().numpy()
        phys["dt_s"][lo:lo + B] = P[..., 2].cpu().numpy()
        phys["rate"][lo:lo + B] = P[..., 3].cpu().numpy()
        phys["amp"][lo:lo + B] = w.abs().amax(-1).cpu().numpy()
    return waves, phys


def decompose_qact(frames: np.ndarray, fs: float, order: int, device="cuda",
                   batch=512, select="argmax", **qkw):
    from .qact import QACT
    q = QACT(length=WIN, fs=fs, device=device, seed=0, refine_steps=4, **qkw)
    M = frames.shape[0]
    waves = torch.empty(M, order, WIN, device=device)
    phys = {k: np.zeros((M, order), np.float32) for k in ("f_hz", "dt_s", "amp", "rate")}
    for lo in range(0, M, batch):
        chunk = frames[lo:lo + batch].astype(np.float32)
        raw, _ = q.transform(chunk, order=order, select=select, return_raw=True)
        B = raw.shape[0]
        w = q.rebuild(raw.reshape(-1, raw.shape[-1])).reshape(B, order, WIN)
        waves[lo:lo + B] = w
        tc, ld, f, c, c3 = (raw[..., i] for i in range(5))
        phys["f_hz"][lo:lo + B] = np.abs(f + 2 * c * tc + 3 * c3 * tc ** 2) * fs / WIN
        phys["dt_s"][lo:lo + B] = np.exp(ld) / fs
        phys["rate"][lo:lo + B] = 2 * (c + 3 * c3 * tc) * fs * fs / WIN
        phys["amp"][lo:lo + B] = w.abs().amax(-1).cpu().numpy()
    return waves, phys, q.dict_size

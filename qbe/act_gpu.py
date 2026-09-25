"""Vendored from the sibling EEG-Memristor-ACT project (code/act_gpu.py). One change:
_joint_refit falls back to least squares for batch elements whose normal equations
are singular (e.g. duplicate atoms), instead of aborting the whole batch.

GPU-batched Adaptive Chirplet Transform (same algorithm as chirplet.py, all windows in parallel).

Per matching-pursuit iteration, for a batch of B windows at once:
  0. optional asymmetric envelope (separate left/right widths, seeded symmetric) as in Mann Lab ACTv9Asym
  1. dictionary search      E = (ss*bc^2 - 2*cs*bc*bs + cc*bs^2)/det     (one matmul, B x M)
  2. off-grid refinement    Adam on (tc, fc, log dt, c) maximizing the same 2-D least-squares
                            captured energy, with autograd through the atom construction
                            (CPU version uses L-BFGS-B on the identical objective)
  3. least-squares fit of the cos/sin quadrature pair, subtract, repeat
Windows whose best atom falls below min_amp stop updating (per-window active mask).
"""
import numpy as np
import torch

TWO_PI = 2*np.pi


def _quad(t, tc, fc, dt, c, dt_r=None):
    """Quadrature chirplet pair. dt_r given => asymmetric Gaussian envelope (left width dt, right width dt_r)."""
    tau = t[None, :] - tc[:, None]
    w = dt[:, None].clamp(min=1e-4)
    if dt_r is not None:
        w = torch.where(tau < 0, w, dt_r[:, None].clamp(min=1e-4))
    env = torch.exp(-0.5*(tau/w)**2)
    ph = TWO_PI*(fc[:, None]*tau + 0.5*c[:, None]*tau**2)
    return env*torch.cos(ph), env*torch.sin(ph)


def _energy(x, gc, gs):
    cc = (gc*gc).sum(1); ss = (gs*gs).sum(1); cs = (gc*gs).sum(1)
    bc = (x*gc).sum(1); bs = (x*gs).sum(1)
    det = cc*ss - cs*cs
    safe = det.abs() > 1e-9*torch.clamp(cc*ss, min=1e-30)
    E2 = (ss*bc**2 - 2*cs*bc*bs + cc*bs**2)/torch.where(safe, det, torch.ones_like(det))
    E1 = bc**2/cc.clamp(min=1e-30)
    E = torch.where(safe, E2, E1)
    ac = torch.where(safe, (ss*bc - cs*bs)/torch.where(safe, det, torch.ones_like(det)), bc/cc.clamp(min=1e-30))
    as_ = torch.where(safe, (cc*bs - cs*bc)/torch.where(safe, det, torch.ones_like(det)), torch.zeros_like(bs))
    return E, ac, as_


class GPUDictionary:
    def __init__(self, P, N, fs, device='cuda', dtype=torch.float32):
        self.dev, self.dtype, self.fs, self.N = device, dtype, float(fs), int(N)
        self.t = torch.arange(N, device=device, dtype=dtype)/fs
        self.P = torch.as_tensor(np.asarray(P), device=device, dtype=dtype)
        gc, gs = _quad(self.t, self.P[:, 0], self.P[:, 1], self.P[:, 2], self.P[:, 3])
        self.GC, self.GS = gc, gs
        self.cc = (gc*gc).sum(1); self.ss = (gs*gs).sum(1); self.cs = (gc*gs).sum(1)
        self.det = self.cc*self.ss - self.cs**2
        self.degen = self.det.abs() <= 1e-6*torch.clamp(self.cc*self.ss, min=1e-30)

    def best(self, R):
        bc = R @ self.GC.T; bs = R @ self.GS.T
        det = torch.where(self.degen, torch.ones_like(self.det), self.det)
        E = (self.ss*bc**2 - 2*self.cs*bc*bs + self.cc*bs**2)/det
        E = torch.where(self.degen, bc**2/self.cc.clamp(min=1e-30), E)
        return self.P[torch.argmax(torch.nan_to_num(E, nan=-1e30), dim=1)].clone()


def refine_batch(X, p, t, fs, steps=60, lr=0.05, asym=False, dt_max=1.5, c_max=1e4):
    """Adam on (tc, fc, log dt, c[, log dt_r]); returns refined params (B, 4) or (B, 5) when asym."""
    T = float(len(t))/fs
    cols = [p[:, 0], p[:, 1], torch.log(p[:, 2].clamp(min=1e-3)), p[:, 3]]
    if asym:
        cols.append(torch.log((p[:, 4] if p.shape[1] > 4 else p[:, 2]).clamp(min=1e-3)))   # seeded symmetric
    q = torch.stack(cols, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    lo = torch.tensor([-0.1*T, 0.0, np.log(0.008), -c_max] + ([np.log(0.008)] if asym else []), device=X.device, dtype=X.dtype)
    hi = torch.tensor([1.1*T, 0.49*fs, np.log(dt_max), c_max] + ([np.log(dt_max)] if asym else []), device=X.device, dtype=X.dtype)
    best_q = q.detach().clone()
    best_E = torch.full((X.shape[0],), -1e30, device=X.device, dtype=X.dtype)
    dr = (lambda qq: torch.exp(qq[:, 4])) if asym else (lambda qq: None)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        gc, gs = _quad(t, q[:, 0], q[:, 1], torch.exp(q[:, 2]), q[:, 3], dr(q))
        E, _, _ = _energy(X, gc, gs)
        (-E.sum()).backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(min=lo, max=hi)
            gc, gs = _quad(t, q[:, 0], q[:, 1], torch.exp(q[:, 2]), q[:, 3], dr(q))
            E2, _, _ = _energy(X, gc, gs)
            upd = E2 > best_E
            best_E = torch.where(upd, E2, best_E)
            best_q[upd] = q.detach()[upd]
    out = best_q.detach()
    cols = [out[:, 0], out[:, 1], torch.exp(out[:, 2]), out[:, 3]]
    if asym:
        cols.append(torch.exp(out[:, 4]))
    return torch.stack(cols, 1)


def refine_coord_batch(X, p, t, fs, sweeps=4, dt_max=1.5, c_max=1e4):
    """QACT's hardware refinement rule (3-point coordinate search) on the classical
    objective. Per sweep and coordinate (tc, fc, log dt, c): evaluate p - d and
    p + d, keep the best of the three. Steps are QACT's own in physical units: tc one
    sample, fc half a frequency bin, log dt 0.05, chirp rate fs^2/N^2 Hz/s (QACT's
    2/(4N) under this engine's phase convention pi*c*tau^2). Bounds are the Adam
    refiner's, so the only change is the search rule."""
    n = len(t)
    T = float(n)/fs
    q = torch.stack([p[:, 0], p[:, 1], torch.log(p[:, 2].clamp(min=1e-3)), p[:, 3]], 1).clone()
    lo = torch.tensor([-0.1*T, 0.0, np.log(0.008), -c_max], device=X.device, dtype=X.dtype)
    hi = torch.tensor([1.1*T, 0.49*fs, np.log(dt_max), c_max], device=X.device, dtype=X.dtype)
    steps = (1.0/fs, 0.5*fs/n, 0.05, fs**2/n**2)

    def energy(qq):
        gc, gs = _quad(t, qq[:, 0], qq[:, 1], torch.exp(qq[:, 2]), qq[:, 3])
        return _energy(X, gc, gs)[0]

    e0 = energy(q)
    for _ in range(sweeps):
        for j, d in enumerate(steps):
            for sgn in (-1.0, 1.0):
                cand = q.clone()
                cand[:, j] = (cand[:, j] + sgn*d).clamp(lo[j], hi[j])
                e1 = energy(cand)
                take = e1 > e0
                q[take] = cand[take]
                e0 = torch.where(take, e1, e0)
    return torch.stack([q[:, 0], q[:, 1], torch.exp(q[:, 2]), q[:, 3]], 1)


def _joint_refit(X, params, t):
    """Orthogonal-MP style joint least-squares over all selected atoms (batched 2K x 2K solve).
    params: list of (B,4) tensors. Returns coefficients (B, 2K) and the reconstruction (B, N)."""
    cols = []
    for p in params:
        gc, gs = _quad(t, p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4] if p.shape[1] > 4 else None)
        cols += [gc, gs]
    A = torch.stack(cols, 2)                      # (B, N, 2K)
    G = A.transpose(1, 2) @ A                     # (B, 2K, 2K)
    b = (A.transpose(1, 2) @ X[:, :, None])       # (B, 2K, 1)
    eye = torch.eye(G.shape[1], device=X.device, dtype=X.dtype)
    ridge = 1e-8*torch.diagonal(G, dim1=1, dim2=2).amax(1).clamp(min=1e-12)[:, None, None]*eye
    coef, info = torch.linalg.solve_ex(G + ridge, b)
    bad = info != 0
    if bool(bad.any()):                       # degenerate systems: least squares instead
        coef[bad] = torch.linalg.lstsq(A[bad], X[bad][:, :, None]).solution
    return coef[:, :, 0], (A @ coef)[:, :, 0]


def decompose_batch(X, D, max_atoms=30, min_amp=None, steps=60, omp=False, asym=False, dt_max=1.5, c_max=1e4,
                    refine="adam"):
    """X: (B, N) tensor. min_amp: (B,) stop threshold. Returns list of dicts of per-atom tensors."""
    R = X.clone()
    B = X.shape[0]
    active = torch.ones(B, dtype=torch.bool, device=X.device)
    if min_amp is None:
        min_amp = torch.zeros(B, device=X.device, dtype=X.dtype)
    out = []
    for _ in range(max_atoms):
        if not bool(active.any()):
            break
        p0 = D.best(R)
        if refine == "coord":                     # steps = sweeps of the 3-point search
            assert not asym, "coordinate refiner is symmetric-envelope only"
            p = refine_coord_batch(R, p0, D.t, D.fs, sweeps=steps, dt_max=dt_max, c_max=c_max)
        else:
            p = refine_batch(R, p0, D.t, D.fs, steps=steps, asym=asym, dt_max=dt_max, c_max=c_max)
        gc, gs = _quad(D.t, p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4] if asym else None)
        E, ac, as_ = _energy(R, gc, gs)
        w = ac[:, None]*gc + as_[:, None]*gs
        amp = w.abs().amax(1)
        keep = active & (amp >= min_amp) & (E > 0)
        out.append(dict(p=p.clone(), ac=ac.clone(), as_=as_.clone(), amp=amp.clone(), keep=keep.clone()))
        if omp:                                   # refit every selected atom jointly, then recompute residual
            coef, recon = _joint_refit(X, [a['p'] for a in out], D.t)
            for j, a in enumerate(out):
                a['ac'], a['as_'] = coef[:, 2*j].clone(), coef[:, 2*j+1].clone()
                gcj, gsj = _quad(D.t, a['p'][:, 0], a['p'][:, 1], a['p'][:, 2], a['p'][:, 3],
                                 a['p'][:, 4] if a['p'].shape[1] > 4 else None)
                a['amp'] = (a['ac'][:, None]*gcj + a['as_'][:, None]*gsj).abs().amax(1)
            R = X - recon
        else:
            R = R - torch.where(keep[:, None], w, torch.zeros_like(w))
        active = keep
    return out, R

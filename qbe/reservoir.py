"""Quantum and classical reservoirs with a shared linear readout.

QUANTUM RESERVOIR (Fujii & Nakajima 2017; Martinez-Pena et al. 2021)
--------------------------------------------------------------------
n qubits evolve under a disordered transverse-field Ising Hamiltonian

    H = sum_{i<j} J_ij X_i X_j + sum_i (h + D_i) Z_i,
    J_ij ~ U(-Js/2, Js/2),   D_i ~ U(-W, W),   Js = 1,

which conserves Z-parity (every term flips an even number of spins) -- the kind
of symmetry that keeps readouts from exponentially concentrating (2505.10062).
Martinez-Pena et al. find the best reservoir performance near its thermalisation
transition; `h`, `W` and the step time `dt` are the knobs.

Each input step:
  1. the n_in input qubits are REPLACED by the encoded state
     sqrt(1-s)|0> + sqrt(s)|1>  (s in [0,1]) -- a non-unitary channel, which is
     what gives the reservoir fading memory; a purely unitary circuit would
     remember everything forever and could not act as a reservoir;
  2. the whole system evolves: rho -> U rho U^dag, U = exp(-i H dt).
Readout after the last step: <Z_i>, <Z_i Z_j>, <X_i> -- n(n+3)/2 numbers.

This is simulated exactly as a density matrix (2^n x 2^n) in PyTorch on the GPU.

Two readout regimes:
  ideal      exact expectation values -- reading the state without disturbing
             it, which hardware cannot do.
  realistic  the restart protocol of Mujal et al. (2023): for every prediction
             the reservoir is re-prepared and fed only the recent window, then
             measured once, with finite-shot noise (`shots` per observable).
`run_windows` always uses the restart protocol (it is what makes batching over
predictions possible); `ideal`/`realistic` differ only in shot noise.

Spatial multiplexing (Nakajima et al. 2019): `MultiReservoir` runs R small
reservoirs with independent random couplings side by side, each fed its own
random projection of the input, and concatenates their readouts.

Control: `J_zero=True` removes every coupling. With no interactions nothing is
entangled and the memory qubits never see the input -- the Gotting et al. (2023)
"no entanglement" limit.

CLASSICAL TWINS
---------------
  ESN   echo state network (Jaeger 2001): leaky tanh recurrent net with a fixed
        random matrix, same restart window, readout on its node states.
  NVAR  next-generation reservoir computing (Gauthier et al. 2021): delayed
        copies of the input plus all their pairwise products. No reservoir at
        all -- often the strongest classical baseline.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# quantum reservoir
# ---------------------------------------------------------------------------
def _paulis(n):
    I = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1.0, -1.0])

    def op(single, q):                      # little-endian: qubit q = bit q
        out = np.array([[1.0]])
        for k in reversed(range(n)):
            out = np.kron(out, single if k == q else I)
        return out
    return [op(X, q) for q in range(n)], [op(Z, q) for q in range(n)]


def tfim_unitary(n, *, h, W, dt, seed, J_zero=False):
    from scipy.linalg import expm
    rng = np.random.default_rng(seed)
    Xs, Zs = _paulis(n)
    H = np.zeros((2 ** n, 2 ** n))
    if not J_zero:
        for i in range(n):
            for j in range(i + 1, n):
                H += rng.uniform(-0.5, 0.5) * Xs[i] @ Xs[j]
    D = rng.uniform(-W, W, n)
    for i in range(n):
        H += (h + D[i]) * Zs[i]
    return expm(-1j * H * dt)


class QuantumReservoir:
    def __init__(self, n=8, n_in=2, *, h=1.0, W=0.1, dt=10.0, seed=0,
                 J_zero=False, device="cuda"):
        self.n, self.n_in, self.dev = n, n_in, device
        self.D, self.R, self.I = 2 ** n, 2 ** (n - n_in), 2 ** n_in
        U = tfim_unitary(n, h=h, W=W, dt=dt, seed=seed, J_zero=J_zero)
        self.U = torch.as_tensor(U, dtype=torch.complex64, device=device)
        self.Ud = self.U.conj().T.contiguous()
        bits = ((torch.arange(self.D)[:, None] >> torch.arange(n)) & 1)
        self.zs = (1 - 2 * bits).float().to(device)                    # (D, n)
        iu = torch.triu_indices(n, n, 1)
        self.zz = (self.zs[:, iu[0]] * self.zs[:, iu[1]])               # (D, n(n-1)/2)
        idx = torch.arange(self.D)
        self.xflip = torch.stack([idx ^ (1 << q) for q in range(n)]).to(device)
        self.idx = idx.to(device)

    @property
    def n_features(self):
        return self.n * (self.n + 3) // 2

    def _inject(self, rho, s):
        """Replace the top n_in qubits by prod_k sqrt(1-s_k)|0> + sqrt(s_k)|1>."""
        B = rho.shape[0]
        rest = torch.diagonal(rho.view(B, self.I, self.R, self.I, self.R), dim1=1, dim2=3).sum(-1)
        amp = torch.stack([torch.sqrt(1 - s), torch.sqrt(s)], -1)       # (B, n_in, 2)
        phi = amp[:, 0]
        for k in range(1, self.n_in):                                   # kron, qubit order
            phi = (amp[:, k][:, :, None] * phi[:, None, :]).reshape(B, -1)
        phi = phi.to(torch.complex64)
        pp = phi[:, :, None] * phi.conj()[:, None, :]                   # (B, I, I)
        return torch.einsum("bik,brl->birkl", pp, rest).reshape(B, self.D, self.D)

    def _readout(self, rho):
        p = torch.diagonal(rho, dim1=1, dim2=2).real                    # (B, D)
        z = p @ self.zs
        zz = p @ self.zz
        x = torch.stack([rho[:, self.idx, self.xflip[q]].real.sum(1) for q in range(self.n)], 1)
        return torch.cat([z, zz, x], 1)

    def run_windows(self, S):
        """S: (B, T, n_in) inputs in [0,1]. Fresh |0..0> per window (restart protocol);
        returns the readout after the last step, (B, n_features)."""
        B, T, _ = S.shape
        rho = torch.zeros(B, self.D, self.D, dtype=torch.complex64, device=self.dev)
        rho[:, 0, 0] = 1.0
        for t in range(T):
            rho = self._inject(rho, S[:, t])
            rho = self.U @ rho @ self.Ud
        return self._readout(rho)

    def run_sequence(self, S):
        """S: (T, n_in). One continuous run, readout after every step (benchmarks)."""
        return self.run_continuous(S[None])[0]

    def run_continuous(self, S):
        """S: (B, T, n_in). B independent recordings run in parallel, each carrying
        its own state along the whole recording; readout after EVERY step.

        This is the reservoir used as a reservoir: memory accumulates over the
        recording instead of being truncated to a window, and it costs one step
        per prediction instead of one window per prediction. Reading the state
        without collapsing it is idealised; `shot_noise` adds the finite-sample
        part of a real measurement, but not its back-action.
        """
        B, T, _ = S.shape
        rho = torch.zeros(B, self.D, self.D, dtype=torch.complex64, device=self.dev)
        rho[:, 0, 0] = 1.0
        out = torch.empty(B, T, self.n_features, device=self.dev)
        for t in range(T):
            rho = self._inject(rho, S[:, t])
            rho = self.U @ rho @ self.Ud
            out[:, t] = self._readout(rho)
        return out


def shot_noise(F, shots, gen):
    """Finite-shot estimate of Pauli expectations: sd = sqrt((1 - <O>^2) / shots)."""
    sd = torch.sqrt(torch.clamp(1 - F ** 2, min=0) / shots)
    return torch.clamp(F + sd * torch.randn(F.shape, generator=gen, device=F.device), -1, 1)


class MultiReservoir:
    """R independent quantum reservoirs; each gets a fixed random projection of the input."""

    def __init__(self, d_in, R=4, n=8, n_in=2, seed=0, **kw):
        g = np.random.default_rng(seed + 1000)
        self.Win = torch.as_tensor(g.normal(0, 1 / math.sqrt(d_in), (R, d_in, n_in)),
                                   dtype=torch.float32, device=kw.get("device", "cuda"))
        self.res = [QuantumReservoir(n, n_in, seed=seed + r, **kw) for r in range(R)]

    @property
    def n_features(self):
        return sum(r.n_features for r in self.res)

    def run_continuous(self, U, chunk=64):
        """U: (B, T, d_in) -> (B, T, n_features), each row run continuously."""
        outs = []
        for lo in range(0, U.shape[0], chunk):
            u = U[lo:lo + chunk]
            outs.append(torch.cat([res.run_continuous(
                (0.5 + 0.5 * torch.tanh(u @ self.Win[r])).clamp(1e-6, 1 - 1e-6))
                for r, res in enumerate(self.res)], -1))
        return torch.cat(outs)

    def run_windows(self, U, chunk=1500):
        """U: (B, T, d_in) standardised inputs -> (B, n_features)."""
        outs = []
        for lo in range(0, U.shape[0], chunk):
            u = U[lo:lo + chunk]
            feats = []
            for r, res in enumerate(self.res):
                s = 0.5 + 0.5 * torch.tanh(u @ self.Win[r])             # (B, T, n_in) in (0,1)
                feats.append(res.run_windows(s.clamp(1e-6, 1 - 1e-6)))
            outs.append(torch.cat(feats, 1))
        return torch.cat(outs)


# ---------------------------------------------------------------------------
# classical twins
# ---------------------------------------------------------------------------
class ESN:
    def __init__(self, d_in, N=176, rho=0.9, leak=1.0, in_scale=1.0, seed=0, device="cuda"):
        g = np.random.default_rng(seed)
        Wr = g.normal(0, 1, (N, N)) * (g.random((N, N)) < 0.1)
        Wr *= rho / max(np.abs(np.linalg.eigvals(Wr)).max(), 1e-9)
        self.W = torch.as_tensor(Wr, dtype=torch.float32, device=device)
        self.Win = torch.as_tensor(g.uniform(-in_scale, in_scale, (d_in, N)) / math.sqrt(d_in),
                                   dtype=torch.float32, device=device)
        self.b = torch.as_tensor(g.uniform(-0.2, 0.2, N), dtype=torch.float32, device=device)
        self.leak, self.N = leak, N

    @property
    def n_features(self):
        return self.N

    def run_windows(self, U):
        x = torch.zeros(U.shape[0], self.N, device=U.device)
        for t in range(U.shape[1]):
            x = (1 - self.leak) * x + self.leak * torch.tanh(x @ self.W.T + U[:, t] @ self.Win + self.b)
        return x

    def run_sequence(self, U):
        return self.run_continuous(U[None])[0]

    def run_continuous(self, U):
        """U: (B, T, d_in) -> (B, T, N). Same continuous protocol as the quantum reservoir."""
        B, T, _ = U.shape
        x = torch.zeros(B, self.N, device=U.device)
        out = torch.empty(B, T, self.N, device=U.device)
        for t in range(T):
            x = (1 - self.leak) * x + self.leak * torch.tanh(x @ self.W.T + U[:, t] @ self.Win + self.b)
            out[:, t] = x
        return out


class NVAR:
    def __init__(self, k=2, s=1):
        self.k, self.s = k, s

    def features(self, U):
        """U: (B, T, d) -> linear taps of the last k steps (stride s) + their pairwise products."""
        taps = [U[:, -1 - j * self.s] for j in range(self.k)]
        lin = torch.cat(taps, 1)
        iu = torch.triu_indices(lin.shape[1], lin.shape[1])
        quad = (lin[:, :, None] * lin[:, None, :])[:, iu[0], iu[1]]
        return torch.cat([lin, quad], 1)

    run_windows = features


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------
def ridge_fit(F, Y, alpha):
    """F (N, p) standardised features with bias column; Y (N, q). Returns W (p, q).
    Solved in float64: reservoir features can be nearly collinear."""
    F64, Y64 = F.double(), Y.double()
    A = F64.T @ F64
    A = A + alpha * torch.eye(A.shape[0], device=F.device, dtype=torch.float64)
    return torch.linalg.solve(A, F64.T @ Y64).to(F.dtype)


class Standardizer:
    """Standardise columns and append a bias column. Constant columns (e.g. the
    idle memory qubits of the zero-coupling control) become exactly zero."""

    def fit(self, F):
        self.mu = F.mean(0); sd = F.std(0)
        self.keep = sd > 1e-6
        self.sd = torch.where(self.keep, sd, torch.ones_like(sd))
        return self

    def __call__(self, F):
        Z = (F - self.mu) / self.sd * self.keep
        return torch.cat([Z, torch.ones(len(Z), 1, device=Z.device, dtype=Z.dtype)], 1)

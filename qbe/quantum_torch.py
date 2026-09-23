"""Trainable variational quantum circuits in PyTorch (exact statevector, autograd).

Used by the QLSTM in `seq_models`. Architecture after Chen, Yoo & Fang,
"Quantum Long Short-Term Memory" (2020):

    |0>^n --H--RY(arctan z)--RZ(arctan z^2)--[ CNOT ring -- RY(a) RZ(b) ] x L -- measure <Z_k>

  * encoding: Hadamard, then RY(arctan z_k) and RZ(arctan z_k^2) on qubit k.
    arctan keeps any real input inside a bounded angle.
  * L variational layers: a ring of CNOTs (k -> k+1 mod n), then trainable
    RY and RZ on every qubit -- 2*n*L trainable angles per circuit.
  * readout: <Z_k> on every qubit, so the circuit maps R^n -> [-1, 1]^n.

Everything is ordinary differentiable tensor algebra, so gradients come from
autograd -- on hardware you would use the parameter-shift rule; on an exact
simulator the two are identical. `verify_against_qiskit` checks the forward
pass against qiskit's Statevector.

Batching: `VQC` holds G independent circuits (G = 4 LSTM gates) and evaluates
them all in one call on input of shape (G, B, n).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _apply_1q(psi: torch.Tensor, U: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """Apply a (per-sample) single-qubit gate. psi (..., 2^n); U (..., 2, 2).
    Qubit q is little-endian (qiskit convention): bit q of the basis index."""
    shp = psi.shape
    t = psi.reshape(*shp[:-1], 2 ** (n - 1 - q), 2, 2 ** q)
    t = torch.einsum("...ij,...ajb->...aib", U, t)
    return t.reshape(shp)


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)


def _rz(theta):
    e = torch.exp(-0.5j * theta.to(torch.complex64))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def _cnot_ring_perm(n: int) -> torch.Tensor:
    """Basis permutation implementing CNOT(0->1), CNOT(1->2), ..., CNOT(n-1->0)."""
    idx = torch.arange(2 ** n)
    perm = idx.clone()
    for c in range(n):
        t = (c + 1) % n
        if n == 1:
            break
        flip = ((perm >> c) & 1) << t
        perm = perm ^ flip
    # psi_out[j] = psi_in[perm^-1[j]]; build the inverse for gather-style indexing
    inv = torch.empty_like(perm)
    inv[perm] = idx
    return inv


class VQC(nn.Module):
    """G independent n-qubit variational circuits evaluated in one batched call."""

    def __init__(self, n_qubits: int = 6, layers: int = 2, n_circuits: int = 1):
        super().__init__()
        self.n, self.L, self.G = n_qubits, layers, n_circuits
        self.theta = nn.Parameter(0.1 * torch.randn(n_circuits, layers, n_qubits, 2))
        self.register_buffer("perm", _cnot_ring_perm(n_qubits))
        bits = ((torch.arange(2 ** n_qubits)[:, None] >> torch.arange(n_qubits)) & 1)
        self.register_buffer("zsign", (1 - 2 * bits).float())          # (2^n, n)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (G, B, n) real -> (G, B, n) expectations <Z_k> in [-1, 1]."""
        G, B, n = z.shape
        D = 2 ** n
        psi = torch.full((G, B, D), 1 / math.sqrt(D), dtype=torch.complex64, device=z.device)
        a = torch.atan(z)
        b = torch.atan(z * z)
        for q in range(n):
            psi = _apply_1q(psi, _ry(a[..., q]), q, n)
            psi = _apply_1q(psi, _rz(b[..., q]), q, n)
        for l in range(self.L):
            psi = psi[..., self.perm]
            for q in range(n):
                th = self.theta[:, l, q, :]                                    # (G, 2)
                psi = _apply_1q(psi, _ry(th[:, 0])[:, None], q, n)
                psi = _apply_1q(psi, _rz(th[:, 1])[:, None], q, n)
        probs = psi.real ** 2 + psi.imag ** 2
        return probs @ self.zsign

    def n_params(self) -> int:
        return self.theta.numel()


def verify_against_qiskit(n: int = 4, layers: int = 2, seed: int = 0) -> float:
    """Worst |<Z_k>| difference between this simulator and qiskit."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    torch.manual_seed(seed)
    vqc = VQC(n, layers, n_circuits=2)
    z = torch.randn(2, 3, n)
    mine = vqc(z).detach().numpy()
    worst = 0.0
    for g in range(2):
        for b in range(3):
            qc = QuantumCircuit(n)
            qc.h(range(n))
            for q in range(n):
                qc.ry(float(np.arctan(z[g, b, q])), q)
                qc.rz(float(np.arctan(z[g, b, q] ** 2)), q)
            for l in range(layers):
                for c in range(n):
                    qc.cx(c, (c + 1) % n)
                for q in range(n):
                    qc.ry(float(vqc.theta[g, l, q, 0].detach()), q)
                    qc.rz(float(vqc.theta[g, l, q, 1].detach()), q)
            sv = Statevector(qc)
            for q in range(n):
                lab = ["I"] * n; lab[n - 1 - q] = "Z"
                ref = sv.expectation_value(SparsePauliOp("".join(lab))).real
                worst = max(worst, abs(mine[g, b, q] - ref))
    return worst

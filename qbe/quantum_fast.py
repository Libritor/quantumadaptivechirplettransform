"""Exact vectorised simulation of Z/ZZ feature maps, plus the kernels built on them.

WHY THIS EXISTS
---------------
Qiskit's kernel classes simulate one circuit at a time. A nested, subject-wise
hyperparameter search needs thousands of kernel matrices, and at qiskit speed
one 25-configuration search took 445 s. But the Z and ZZ feature maps have
structure that makes exact simulation nearly free:

    each layer = H^{(x)n}  followed by  a DIAGONAL phase

The single-qubit gates are P(2 x_i) and the entangling blocks are CX-P-CX, all
diagonal in the computational basis, so one layer's action on a basis state
|b> is a phase:

    phase(b) = sum_i 2 x_i b_i  +  sum_{(i,j) in E} 2 (pi - x_i)(pi - x_j) (b_i XOR b_j)

A whole batch of samples is then a handful of dense matrix operations. This is
the same circuit, not an approximation -- `verify_against_qiskit` checks the
kernel matrix against qiskit's own `FidelityStatevectorKernel` to ~1e-12.

TWO KERNELS
-----------
`FidelityKernelSVC`  K(x,x') = |<psi(x)|psi(x')>|^2 -- the Havlicek et al.
                     kernel used so far, just faster.

`PQKFeatures`        the *projected* quantum kernel of Huang et al., "Power of
                     data in quantum machine learning" (Nat. Commun. 12, 2021).
                     Instead of the global overlap, it compares each qubit's
                     reduced state. Since ||rho_k - rho_k'||_F^2 is half the
                     squared distance between Pauli expectation vectors, the
                     projected kernel exp(-gamma * sum_k ||rho_k - rho_k'||^2) is
                     an RBF on the 3n expectations <X_k>, <Y_k>, <Z_k>. The
                     global fidelity kernel concentrates as the Hilbert space
                     grows, so almost every pair of states looks orthogonal; the
                     projected kernel is the standard remedy.

DATA RE-UPLOADING
-----------------
With `reupload=True`, layer l encodes a *different* block of n features
(Perez-Salinas et al., Quantum 4, 2020), so n qubits x L layers read n*L
features. That lets the circuit see as many features as the classical model
without needing 2^(n*L) amplitudes.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC

FAST_FEATURE_MAPS = ("zz", "z")

# Set QBE_DEVICE=cuda to run the simulation in PyTorch on the GPU. Results are
# the same circuit; only where the linear algebra runs changes.
DEVICE = os.environ.get("QBE_DEVICE", "cpu")


def _torch():
    import torch
    return torch


def _pairs(n: int, entanglement: str) -> list[tuple[int, int]]:
    if n < 2:
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "full":
        return list(combinations(range(n), 2))
    if entanglement == "circular":
        return [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    raise ValueError(f"unknown entanglement {entanglement!r}")


def _bits(n: int) -> np.ndarray:
    """(2^n, n) bit table, little-endian to match qiskit's qubit ordering."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    H = np.array([[1.0]])
    for _ in range(n):
        H = np.kron(H, h)
    return H


def statevectors(
    A: np.ndarray,
    n_qubits: int,
    layers: int = 2,
    entanglement: str = "linear",
    kind: str = "zz",
    reupload: bool = False,
) -> np.ndarray:
    """Exact feature-map statevectors for a batch of angle vectors.

    `A` is (N, n_qubits * layers) when `reupload`, else (N, n_qubits).
    Returns (N, 2^n_qubits) complex amplitudes.
    """
    if kind not in FAST_FEATURE_MAPS:
        raise ValueError(f"fast simulation supports {FAST_FEATURE_MAPS}, not {kind!r}")
    if DEVICE != "cpu":
        return _statevectors_torch(A, n_qubits, layers, entanglement, kind, reupload)
    A = np.asarray(A, dtype=np.float64)
    n, N = n_qubits, A.shape[0]
    need = n * layers if reupload else n
    if A.shape[1] != need:
        raise ValueError(f"expected {need} angle columns, got {A.shape[1]}")
    D = 2 ** n
    B = _bits(n)
    pairs = _pairs(n, entanglement) if kind == "zz" else []
    parity = [np.mod(B[:, i] + B[:, j], 2.0) for i, j in pairs]
    H = _hadamard(n)

    psi = np.full((N, D), 1.0 / np.sqrt(D), dtype=np.complex128)   # H^n |0>
    for layer in range(layers):
        x = A[:, layer * n:(layer + 1) * n] if reupload else A
        if layer > 0:
            psi = psi @ H                                           # H is symmetric
        phase = 2.0 * (x @ B.T)
        for (i, j), p in zip(pairs, parity):
            phase += 2.0 * ((np.pi - x[:, i]) * (np.pi - x[:, j]))[:, None] * p[None, :]
        psi *= np.exp(1j * phase)
    return psi


def _statevectors_torch(A, n, layers, entanglement, kind, reupload):
    """Same circuit as `statevectors`, as a GPU tensor (complex128)."""
    torch = _torch()
    A = torch.as_tensor(np.asarray(A), dtype=torch.float64, device=DEVICE)
    N, D = A.shape[0], 2 ** n
    B = torch.as_tensor(_bits(n), device=DEVICE)
    pairs = _pairs(n, entanglement) if kind == "zz" else []
    parity = [torch.remainder(B[:, i] + B[:, j], 2.0) for i, j in pairs]
    H = torch.as_tensor(_hadamard(n), dtype=torch.complex128, device=DEVICE)
    psi = torch.full((N, D), 1.0 / np.sqrt(D), dtype=torch.complex128, device=DEVICE)
    for layer in range(layers):
        x = A[:, layer * n:(layer + 1) * n] if reupload else A
        if layer > 0:
            psi = psi @ H
        phase = 2.0 * (x @ B.T)
        for (i, j), par in zip(pairs, parity):
            phase = phase + 2.0 * ((np.pi - x[:, i]) * (np.pi - x[:, j]))[:, None] * par[None, :]
        psi = psi * torch.exp(1j * phase)
    return psi


def fidelity_kernel(S1, S2) -> np.ndarray:
    if not isinstance(S1, np.ndarray):              # torch tensors on the GPU
        return (S1 @ S2.conj().T).abs().pow(2).cpu().numpy()
    return np.abs(S1 @ S2.conj().T) ** 2


def pauli_expectations(psi, n: int) -> np.ndarray:
    """(N, 3n) array of <X_k>, <Y_k>, <Z_k> -- the single-qubit reduced states."""
    if not isinstance(psi, np.ndarray):
        torch = _torch()
        N = psi.shape[0]
        cols = []
        for k in range(n):
            t = psi.reshape(N, 2 ** (n - 1 - k), 2, 2 ** k)
            a0 = t[:, :, 0, :].reshape(N, -1)
            a1 = t[:, :, 1, :].reshape(N, -1)
            r01 = (a0 * a1.conj()).sum(1)
            cols += [2.0 * r01.real, -2.0 * r01.imag,
                     (a0.abs() ** 2).sum(1) - (a1.abs() ** 2).sum(1)]
        return torch.stack(cols, 1).cpu().numpy()
    N = psi.shape[0]
    out = np.empty((N, 3 * n))
    for k in range(n):
        t = psi.reshape(N, 2 ** (n - 1 - k), 2, 2 ** k)
        a0 = t[:, :, 0, :].reshape(N, -1)
        a1 = t[:, :, 1, :].reshape(N, -1)
        r01 = (a0 * a1.conj()).sum(axis=1)
        out[:, 3 * k] = 2.0 * r01.real
        out[:, 3 * k + 1] = -2.0 * r01.imag
        out[:, 3 * k + 2] = (np.abs(a0) ** 2).sum(axis=1) - (np.abs(a1) ** 2).sum(axis=1)
    return out


class FidelityKernelSVC(ClassifierMixin, BaseEstimator):
    """QSVC with the exact fast simulator: SVM on K(x,x') = |<psi(x)|psi(x')>|^2."""

    def __init__(self, n_qubits=4, layers=2, entanglement="linear",
                 feature_map="zz", reupload=False, C=1.0):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.feature_map = feature_map
        self.reupload = reupload
        self.C = C

    def _states(self, X):
        return statevectors(X, self.n_qubits, self.layers, self.entanglement,
                            self.feature_map, self.reupload)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.train_states_ = self._states(X)
        K = fidelity_kernel(self.train_states_, self.train_states_)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        return self

    def predict(self, X):
        return self.svc_.predict(fidelity_kernel(self._states(X), self.train_states_))


class PQKFeatures(TransformerMixin, BaseEstimator):
    """Projected-quantum-kernel features: single-qubit Pauli expectations.

    Followed by an RBF SVC this is the projected quantum kernel. The features
    come out of the quantum circuit; the RBF is the kernel defined on them.
    """

    def __init__(self, n_qubits=4, layers=2, entanglement="linear",
                 feature_map="zz", reupload=False):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.feature_map = feature_map
        self.reupload = reupload

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        psi = statevectors(X, self.n_qubits, self.layers, self.entanglement,
                           self.feature_map, self.reupload)
        return pauli_expectations(psi, self.n_qubits)


def verify_against_qiskit(seed: int = 0, tol: float = 1e-9) -> float:
    """Check the fast simulator against qiskit. Returns the worst error seen."""
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in (2, 3, 4):
        for reps in (1, 2, 3):
            for kind, ent in (("zz", "linear"), ("zz", "full"), ("z", "linear")):
                A = rng.uniform(0, np.pi, (7, n))
                fm = (ZZFeatureMap(n, reps=reps, entanglement=ent) if kind == "zz"
                      else ZFeatureMap(n, reps=reps))
                K_ref = FidelityStatevectorKernel(feature_map=fm).evaluate(A)
                S = statevectors(A, n, reps, ent, kind)
                worst = max(worst, float(np.abs(fidelity_kernel(S, S) - K_ref).max()))
                # Single-qubit expectations against qiskit's statevector.
                sv = Statevector(fm.assign_parameters(A[0]))
                mine = pauli_expectations(S[:1], n)[0]
                for q in range(n):
                    for c, p in enumerate("XYZ"):
                        label = ["I"] * n
                        label[n - 1 - q] = p
                        ref = sv.expectation_value(SparsePauliOp("".join(label))).real
                        worst = max(worst, abs(mine[3 * q + c] - ref))
    if worst > tol:
        raise AssertionError(f"fast simulator disagrees with qiskit: {worst:.2e}")
    return worst

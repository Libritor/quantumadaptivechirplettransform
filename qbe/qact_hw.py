"""Hardware circuits for one QACT evaluation, assembled from Qiskit library parts.

Nothing here simulates anything. Circuits are built from Qiskit's own components
(`StatePreparation`, `UCRYGate`, `QFTGate`, `synth_qft_full`) and handed to the
standard tools: the transpiler for gate counts on a real device target, and Qiskit
Aer (with the device's calibrated noise model) for execution.

One QACT evaluation asks, for one (envelope, chirp) pair, for the distribution
over frequency k of

    P(k) = |sum_t s(t) e^{-i 2pi c t^2 / N} e^{-i 2pi k t / N}|^2 / N,
    s = env * r / ||env * r||

which, reweighted by the classical factor N ||env r||^2 / ||env||^2, is exactly
the post-selection-conditioned criterion `QACT._power` computes.

Four constructions of the same distribution:

  baseline      prepare r; envelope filter = multiplexed RY onto an ancilla,
                post-select ancilla = 0; chirp; inverse QFT (with swaps).
                What docs/QUANTUM_ACT.md describes.
  folded        prepare env * r directly. The window multiply is O(N) classical
                work but a 2^n-way multiplexed rotation plus post-selection
                quantumly, so it moves to the classical side of the interface:
                no ancilla, no multiplexor, no discarded shots.
  aqft          folded, with Qiskit's approximate QFT (drops controlled rotations
                below pi/2^(n-degree)). For devices without dynamic circuits.
  semiclassical folded, with the QFT replaced by mid-circuit measurement and
                classically controlled single-qubit phases (Griffiths & Niu,
                PRL 76, 3228, 1996). Exact, and it contains NO two-qubit gates:
                every controlled phase of the inverse QFT has one qubit that is
                already measured by the time the gate is reached.
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate, StatePreparation, UCRYGate
from qiskit.synthesis import synth_qft_full

from .qact import linear_phase_gates, quadratic_phase_gates

CONSTRUCTIONS = ("baseline", "folded", "aqft", "semiclassical", "windowed")


def window_register(env, n, sigmas=4.0):
    """(m, t0): the smallest register holding env's +-`sigmas` support, and its start.

    A Gaussian truncated at 4 sigma drops amplitudes below e^-8 = 3.4e-4.
    """
    N = 2 ** n
    keep = np.flatnonzero(env >= math.exp(-0.5 * sigmas ** 2) * env.max())
    lo, hi = int(keep[0]), int(keep[-1])
    m = min(n, max(1, math.ceil(math.log2(hi - lo + 1))))
    t0 = max(0, min(lo, N - 2 ** m))
    return m, t0


def target_distribution(r, env, c):
    """The distribution every construction must reproduce (numpy FFT)."""
    N = len(r)
    s = env * r
    s = s / np.linalg.norm(s)
    t = np.arange(N)
    z = s * np.exp(-1j * 2 * np.pi * c * t ** 2 / N)
    return np.abs(np.fft.fft(z)) ** 2 / N


def _chirp(qc, qubits, n, c, N, lin=0.0):
    """Conjugate chirp exp(-i 2pi (c t^2 + lin t) / N) from the verified gate lists.

    `n` is the register size and `N` the ORIGINAL signal length, so a window-local
    register applies the same physical chirp as the full one.
    """
    gates = quadratic_phase_gates(n, -c, N)
    if lin:
        gates = gates + linear_phase_gates(n, -lin, N)
    for g in gates:
        if g[0] == "p":
            qc.p(g[2], qubits[g[1]])
        else:
            qc.cp(g[3], qubits[g[1]], qubits[g[2]])


def _semiclassical_iqft(qc, t, ck):
    """Inverse QFT (input loaded bit-reversed) as H + measure + classically
    controlled single-qubit phases; bit j of the outcome lands in ck[j]."""
    n = len(t)
    for j in range(n):
        for k in range(j):
            with qc.if_test((ck[k], 1)):
                qc.p(-math.pi * 2.0 ** (k - j), t[j])
        qc.h(t[j])
        qc.measure(t[j], ck[j])


def build(kind, r, env, c, approx_degree=2, measure=True, offset=0):
    """One QACT evaluation as a Qiskit circuit.

    Measured frequency bits land in classical register 'k' (bit j = bit j of k);
    the baseline also measures its ancilla into register 'anc'.
    """
    N = len(r)
    n = int(round(math.log2(N)))
    t = QuantumRegister(n, "t")
    if kind == "baseline":
        a = QuantumRegister(1, "a")
        qc = QuantumCircuit(t, a)
        qc.append(StatePreparation(r / np.linalg.norm(r)), t)
        # |t>|0> -> env(t)/max env |t>|0> + ...|1>; UCRYGate takes the target first
        theta = 2 * np.arccos(np.clip(env / env.max(), 0.0, 1.0))
        qc.append(UCRYGate(list(theta)), [a[0]] + list(t))
        _chirp(qc, t, n, c, N)
        qc.append(QFTGate(n).inverse(), t)
        if measure:
            ck, ca = ClassicalRegister(n, "k"), ClassicalRegister(1, "anc")
            qc.add_register(ck, ca)
            qc.measure(t, ck)
            qc.measure(a, ca)
        return qc

    s = env * r
    s = s / np.linalg.norm(s)
    qc = QuantumCircuit(t)
    if kind in ("folded", "aqft"):
        qc.append(StatePreparation(s), t)
        _chirp(qc, t, n, c, N)
        if kind == "folded":
            qc.append(QFTGate(n).inverse(), t)
        else:
            qc.compose(synth_qft_full(n, approximation_degree=approx_degree,
                                      do_swaps=True, inverse=True), t, inplace=True)
        if measure:
            ck = ClassicalRegister(n, "k")
            qc.add_register(ck)
            qc.measure(t, ck)
        return qc

    if kind == "semiclassical":
        # The inverse QFT with swaps = swaps, then the swap-free inverse. The
        # leading swaps are a relabelling, so the INPUT is loaded bit-reversed
        # (state and chirp both see time bit k on qubit n-1-k) and the swap-free
        # inverse QFT follows: for j = 0..n-1, phases from every lower qubit,
        # then H(j), then measure j. Each lower qubit is already measured, so each
        # controlled phase becomes a classically controlled single-qubit phase.
        rev = list(t)[::-1]
        ck = ClassicalRegister(n, "k")
        qc.add_register(ck)
        qc.append(StatePreparation(s), rev)
        _chirp(qc, rev, n, c, N)
        _semiclassical_iqft(qc, t, ck)
        return qc

    if kind == "windowed":
        # Only the envelope's support is loaded, onto m <= n qubits starting at
        # t0. Shift covariance of the chirp, c(u+t0)^2 = c u^2 + 2 c t0 u + const,
        # folds the offset into a linear phase; `offset` j adds a further
        # frequency shift so that circuit j reads bins k' 2^(n-m) + j. Every
        # offset circuit is exact on its bins; together they cover all N.
        m, t0 = window_register(env, n)
        sw = s[t0:t0 + 2 ** m]
        sw = sw / np.linalg.norm(sw)
        tw = QuantumRegister(m, "t")
        qc = QuantumCircuit(tw)
        rev = list(tw)[::-1]
        ck = ClassicalRegister(m, "k")
        qc.add_register(ck)
        qc.append(StatePreparation(sw), rev)
        _chirp(qc, rev, m, c, N, lin=2 * c * t0 + offset)
        _semiclassical_iqft(qc, tw, ck)
        return qc

    raise ValueError(f"unknown construction {kind!r}")


def windowed_target(r, env, c, n, offset=0):
    """What the windowed circuit must return: the full target on bins
    k' 2^(n-m) + offset, renormalised, and the bin each outcome maps to."""
    m, _ = window_register(env, n)
    full = target_distribution(r, env, c)
    bins = np.arange(2 ** m) * 2 ** (n - m) + offset
    sub = full[bins]
    return sub / sub.sum(), bins, m


def distribution_from_counts(counts, n, postselect_ancilla=False):
    """Counts dict (Aer) -> probability vector over k, plus kept-shot fraction.

    Keys are space-separated registers, last-added register first, bit 0 rightmost.
    """
    p = np.zeros(2 ** n)
    kept = total = 0
    for key, v in counts.items():
        parts = key.split()
        total += v
        if postselect_ancilla:
            anc, kbits = parts[0], parts[1]
            if anc != "0":
                continue
        else:
            kbits = parts[-1]
        p[int(kbits, 2)] += v
        kept += v
    return p / max(kept, 1), kept / max(total, 1)


def two_qubit_stats(tq):
    """(two-qubit gate count, two-qubit depth, total depth) of a transpiled circuit,
    descending into control-flow blocks."""
    def walk(circ):
        n2 = 0
        for inst in circ.data:
            op = inst.operation
            if getattr(op, "blocks", None):
                for b in op.blocks:
                    n2 += walk(b)
            elif op.num_qubits == 2:
                n2 += 1
        return n2
    return (walk(tq), tq.depth(lambda i: i.operation.num_qubits == 2), tq.depth())

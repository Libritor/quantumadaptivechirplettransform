#!/usr/bin/env python
"""Search iteration 1: can fault-tolerant coherent dictionary search ever win?

Quantum: Durr-Hoyer maximum finding over the dictionary index, with each score
estimated coherently by amplitude estimation to precision eps. Cost per atom
selection, deliberately OPTIMISTIC for the quantum side:

    T_q(M) = c_DH * sqrt(M) * (1 / eps) * t_U

  c_DH = 1   (Durr & Hoyer prove 22.5; 1 is a lower bound)
  t_U        fault-tolerant runtime of ONE application of the oracle unitary U
             (load window -> conjugate chirp -> inverse QFT), from Microsoft's
             resource estimator at a per-call error budget of 1e-4; ignores that
             the budget must shrink as the call count grows, and ignores U^dagger
             and the reflections amplitude estimation also needs
  two cases  explicit state preparation (no QRAM), and QRAM treated as FREE
             (the circuit with state loading removed)

Classical: FFT-based matching pursuit, which scores all N/2 frequencies of an
(envelope, chirp) pair with one FFT. MEASURED per pair on this machine:

    T_c(M) = (M / (N/2)) * t_pair

Crossover dictionary size: M* = (c_DH * t_U * (N/2) / (eps * t_pair))^2.
Current QACT dictionary: M = 22,784 atoms.
"""
import json
import time

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, StatePreparation
from qdk.estimator import EstimatorParams, QECScheme, QubitParams
from qdk.qiskit import estimate

from qact_vs_act_speed import eeg_windows, env_of
from qbe.qact import QACT, quadratic_phase_gates

N, n = 512, 9
BASIS = ["cx", "rz", "sx", "x", "h"]
M_NOW = 22_784


def classical_t_pair():
    """Measured seconds per (envelope, chirp) pair: QACT's own FFT-based scoring."""
    q = QACT(length=N, fs=256.0, device="cuda")
    X = np.tile(eeg_windows(), (256, 1)).astype(np.float32)          # 1,024 windows
    R = torch.as_tensor(X, device="cuda")
    q._power(R)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        q._power(R)
    torch.cuda.synchronize()
    gpu = (time.perf_counter() - t0) / 5 / (len(X) * q.n_pairs_used)
    # CPU, one window at a time, numpy (no batching advantage)
    x = X[0].astype(np.float64)
    env = np.stack([np.exp(-((np.arange(N) - tc) ** 2) / (2 * np.exp(ld) ** 2))
                    for tc in range(0, N, 48) for ld in (1.5, 2.7, 3.9, 5.1)])
    chirp = np.exp(-2j * np.pi * 0.03125 * np.arange(N) ** 2 / N)
    t0 = time.perf_counter()
    for _ in range(20):
        np.fft.fft(env * x * chirp, axis=1)
    cpu = (time.perf_counter() - t0) / 20 / len(env)
    return gpu, cpu, q.n_pairs_used


def oracle(with_loading):
    """One application of U: [load window] -> conjugate chirp -> inverse QFT."""
    qc = QuantumCircuit(n)
    if with_loading:
        x = eeg_windows(1)[0]
        s = env_of(256.0, 3.9) * x
        qc.append(StatePreparation(s / np.linalg.norm(s)), range(n))
    else:
        qc.h(range(n))           # QRAM treated as free: only the rest of U is costed
    for g in quadratic_phase_gates(n, -0.03125, N):
        if g[0] == "p":
            qc.p(g[2], g[1])
        else:
            qc.cp(g[3], g[1], g[2])
    qc.append(QFTGate(n).inverse(), range(n))
    return transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)


def t_U(circ, tech):
    p = EstimatorParams()
    p.qubit_params.name = getattr(QubitParams, tech)
    p.qec_scheme.name = QECScheme.SURFACE_CODE
    p.error_budget = 1e-4
    r = estimate(circ, p)
    d = r.data() if hasattr(r, "data") else r
    return d["physicalCounts"]["runtime"] * 1e-9, d["physicalCounts"]["physicalQubits"]


def main():
    gpu, cpu, pairs = classical_t_pair()
    print(f"classical, per (envelope, chirp) pair = {N // 2} atoms scored: "
          f"GPU batched {gpu * 1e9:.1f} ns, CPU single window {cpu * 1e6:.2f} us "
          f"({pairs} pairs in the current dictionary)\n")
    out = dict(t_pair_gpu_s=gpu, t_pair_cpu_s=cpu, rows=[])
    print(f"{'oracle U':<22}{'technology':<12}{'t_U':>10}{'phys qubits':>13}"
          f"{'eps':>7}{'M* vs GPU':>12}{'M* vs CPU':>12}{'x current M':>13}")
    for load, label in ((True, "explicit loading"), (False, "QRAM for free")):
        circ = oracle(load)
        for tech in ("GATE_NS_E3", "GATE_NS_E4"):
            tu, phys = t_U(circ, tech)
            for eps in (0.1, 0.01):
                m_gpu = (tu * (N / 2) / (eps * gpu)) ** 2
                m_cpu = (tu * (N / 2) / (eps * cpu)) ** 2
                out["rows"].append(dict(oracle=label, tech=tech, t_U_s=tu, physical_qubits=phys,
                                        eps=eps, M_star_gpu=m_gpu, M_star_cpu=m_cpu))
                print(f"{label:<22}{tech:<12}{tu * 1e3:>8.2f}ms{phys:>13,}{eps:>7}"
                      f"{m_gpu:>12.1e}{m_cpu:>12.1e}{m_cpu / M_NOW:>13.1e}")
    best = min(r["M_star_cpu"] for r in out["rows"])
    print(f"\nmost favourable case for quantum (free QRAM, best qubits, eps 0.1, vs "
          f"single-window CPU): crossover at M* = {best:.1e} atoms = "
          f"{best / M_NOW:.1e} x the current dictionary")
    # for scale: the whole dictionary of all chirplets resolvable in a 512-sample window
    print("for scale: 512 positions x 512 frequencies x 100 widths x 1,000 chirp rates "
          f"= {512 * 512 * 100 * 1000:.1e} atoms")
    json.dump(out, open("results/qsearch_01_ft_crossover.json", "w"), indent=1)
    print("wrote results/qsearch_01_ft_crossover.json")


if __name__ == "__main__":
    main()

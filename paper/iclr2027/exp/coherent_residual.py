#!/usr/bin/env python
"""Coherent matching-pursuit residual for a gate-native chirplet dictionary: verification.

Claim (paper, Prop. on the coherent residual). Let P_k = I - |psi_k><psi_k| be the
orthogonal projector that removes atom k. With one ancilla, the linear combination
(I + R_k)/2 with R_k = I - 2|psi_k><psi_k| (a reflection, unitary) applied as
    ancilla H . controlled-R_k . ancilla H, post-select ancilla = |0>
maps |r> to P_k |r> (unnormalised) with success probability ||P_k r||^2 / ||r||^2.
Chaining K of them on the loaded signal |x> yields exactly the classical matching-
pursuit residual r_K with success probability ||r_K||^2 / ||x||^2 -- the residual
energy fraction, i.e. one minus the energy the K atoms explain. The reflection is
gate-native: R_k = U_k (I - 2|0><0|) U_k^dagger where U_k prepares the atom
(envelope preparation followed by the O(n^2) chirp phase gates), so no data is
loaded after the signal itself.

This script verifies the claim two ways on a real EEG window (N = 64, n = 6):
  1. numpy statevector: chained LCU projectors vs classical MP residuals (max |diff|)
  2. a qiskit circuit built from StatePreparation + phase gates + multi-controlled
     phase, checked with Statevector, and transpiled to count two-qubit gates.
It also tabulates the success probability on the 320-window public Set C of the
classical study (examples/eegmmidb_skew_replication_results.json of the ACT repo,
if present) as err^2 at orders 8/16/24/32.

    python paper/iclr2027/exp/coherent_residual.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from qbe.qact import linear_phase_gates, quadratic_phase_gates  # noqa: E402

OUT = ROOT / "paper" / "iclr2027" / "results"


def eeg_window(N):
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    f = eegbci.load_data(1, [1], path=str(Path.home() / "mne_data"), update_path=False,
                         verbose=False)[0]
    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
    eegbci.standardize(raw)
    st = (raw.n_times - N) // 2
    x = raw.get_data(picks=["O1"])[0, st:st + N].astype(np.float64)
    x = x - x.mean()
    return x / np.linalg.norm(x)


def atom(N, tc, logdt, f, c):
    t = np.arange(N)
    env = np.exp(-((t - tc) ** 2) / (2 * math.exp(logdt) ** 2))
    psi = env * np.exp(1j * 2 * np.pi * (c * t ** 2 + f * t) / N)
    return psi / np.linalg.norm(psi), env / np.linalg.norm(env)


def classical_mp(x, N, grid, K):
    r = x.astype(complex).copy()
    picked, resid = [], []
    for _ in range(K):
        best = max(grid, key=lambda th: abs(np.vdot(atom(N, *th)[0], r)) ** 2)
        psi = atom(N, *best)[0]
        r = r - np.vdot(psi, r) * psi
        picked.append(best)
        resid.append(r.copy())
    return picked, resid


def lcu_projector_chain(x, atoms):
    """numpy: apply (I + R_k)/2 in sequence, exactly as the LCU circuit does."""
    state = x.astype(complex) / np.linalg.norm(x)
    succ = 1.0
    for psi in atoms:
        R = np.eye(len(x)) - 2 * np.outer(psi, psi.conj())
        new = 0.5 * (state + R @ state)            # ancilla H, c-R, H, post-select |0>
        p = float(np.vdot(new, new).real)          # success probability of this step
        succ *= p
        state = new / math.sqrt(p)
    return state, succ


def qiskit_check(x, picked, N):
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    n = int(round(math.log2(N)))
    t = QuantumRegister(n, "t")
    a = QuantumRegister(len(picked), "a")
    qc = QuantumCircuit(t, a)
    qc.append(StatePreparation(x / np.linalg.norm(x)), t)

    def prep(qc_, th, inverse=False):
        """U_k: envelope state preparation then the chirp/linear phases (gate-native)."""
        tc, logdt, f, c = th
        _, env = atom(N, tc, logdt, f, c)
        gates = quadratic_phase_gates(n, c, N) + linear_phase_gates(n, f, N)
        if not inverse:
            qc_.append(StatePreparation(env), t)
            for g in gates:
                (qc_.p(g[2], t[g[1]]) if g[0] == "p" else qc_.cp(g[3], t[g[1]], t[g[2]]))
        else:
            for g in reversed(gates):
                (qc_.p(-g[2], t[g[1]]) if g[0] == "p" else qc_.cp(-g[3], t[g[1]], t[g[2]]))
            qc_.append(StatePreparation(env).inverse(), t)

    for k, th in enumerate(picked):
        qc.h(a[k])
        # controlled reflection about psi_k: U (I - 2|0><0|) U^dag, controlled on a[k].
        # (I - 2|0><0|) = X^n . (multi-controlled Z, phase pi on |1..1>) . X^n up to
        # the global sign, which the control makes relative: mcp(pi) with n+1 controls.
        prep(qc, th, inverse=True)
        qc.x(t)
        qc.mcp(math.pi, [a[k]] + list(t)[:-1], t[-1])
        qc.x(t)
        prep(qc, th)
        qc.h(a[k])
    sv = np.asarray(Statevector(qc).data).reshape([2] * (n + len(picked)))
    # post-select every ancilla on |0>: ancillas are the high-order qubits
    sub = sv
    for _ in range(len(picked)):
        sub = np.take(sub, 0, axis=0)
    amp = sub.reshape(-1)            # little-endian index over t
    succ = float(np.vdot(amp, amp).real)
    # cost of ONE controlled reflection at this n, on an all-to-all CZ basis
    one = QuantumCircuit(t, QuantumRegister(1, "a"))
    one.h(one.qubits[-1])
    tc, logdt, f, c = picked[0]
    _, env = atom(N, tc, logdt, f, c)
    gates = quadratic_phase_gates(n, c, N) + linear_phase_gates(n, f, N)
    for g in reversed(gates):
        (one.p(-g[2], t[g[1]]) if g[0] == "p" else one.cp(-g[3], t[g[1]], t[g[2]]))
    one.append(StatePreparation(env).inverse(), t)
    one.x(t)
    one.mcp(math.pi, [one.qubits[-1]] + list(t)[:-1], t[-1])
    one.x(t)
    one.append(StatePreparation(env), t)
    for g in gates:
        (one.p(g[2], t[g[1]]) if g[0] == "p" else one.cp(g[3], t[g[1]], t[g[2]]))
    one.h(one.qubits[-1])
    pm = generate_preset_pass_manager(optimization_level=3,
                                      basis_gates=["cz", "rz", "sx", "x"], seed_transpiler=7)
    tq = pm.run(one)
    n2 = sum(1 for inst in tq.data if inst.operation.num_qubits == 2)
    return amp, succ, n2


def main():
    N, K = 64, 4
    x = eeg_window(N)
    grid = [(tc, ld, f, c) for tc in (16, 32, 48) for ld in (1.5, 2.3) for f in range(1, 24)
            for c in (-0.05, 0.0, 0.05)]
    picked, resid = classical_mp(x, N, grid, K)
    atoms = [atom(N, *th)[0] for th in picked]
    state, succ = lcu_projector_chain(x, atoms)
    r_K = resid[-1]
    err_chain = float(np.abs(state - r_K / np.linalg.norm(r_K)).max())
    print(f"N={N}, K={K} atoms picked by classical MP: {picked}")
    print(f"numpy LCU chain vs classical residual r_K (normalised): max|diff| = {err_chain:.2e}")
    print(f"success probability of the chain: {succ:.6f}; ||r_K||^2/||x||^2 = "
          f"{float(np.vdot(r_K, r_K).real):.6f}")
    amp, succ_q, n2 = qiskit_check(x, picked, N)
    err_q = float(np.abs(amp / np.linalg.norm(amp) - r_K / np.linalg.norm(r_K)).max())
    print(f"qiskit circuit (StatePreparation + phase gates + mcp): max|diff| = {err_q:.2e}, "
          f"success {succ_q:.6f}; one controlled reflection at n={int(math.log2(N))}: "
          f"{n2} two-qubit gates (all-to-all CZ basis)")
    out = dict(N=N, K=K, picked=[list(map(float, p)) for p in picked],
               numpy_max_diff=err_chain, numpy_success=succ,
               residual_energy_fraction=float(np.vdot(r_K, r_K).real),
               qiskit_max_diff=err_q, qiskit_success=succ_q, reflection_two_qubit_gates=n2)
    setc = ROOT.parent / "adaptive-chirplet-transform" / "examples" / \
        "eegmmidb_skew_replication_results.json"
    if setc.exists():
        d = json.load(open(setc))
        tab = {o: {v: float(d["mean_err"][o][v]) ** 2 for v in ("F0", "v14", "v14full")}
               for o in d["mean_err"]}
        out["set_c_success_probability_err2"] = tab
        print("Set C (320 public EEG windows): mean err^2 = coherent-residual success probability")
        for o, row in tab.items():
            print(f"  order {o:>2}: " + ", ".join(f"{v} {p:.3f}" for v, p in row.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "coherent_residual.json", "w"), indent=1)
    print(f"wrote {OUT / 'coherent_residual.json'}")


if __name__ == "__main__":
    main()

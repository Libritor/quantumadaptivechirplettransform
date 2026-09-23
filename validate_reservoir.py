#!/usr/bin/env python
"""Validate qbe.reservoir before any EEG use.

1. Correctness: our density-matrix reservoir vs an independent qiskit
   implementation (partial_trace + tensor + evolve), several random steps.
2. Benchmarks (single reservoir, scalar input, continuous run):
     memory capacity  MC = sum_k r^2(u_{t-k}, prediction), k = 1..30
     NARMA-10         NMSE of one-step prediction (Atiya & Parlos 2000)
   against an ESN with the same number of readout features, NVAR, a
   zero-coupling quantum reservoir, and a linear (input-only) readout.
"""
import numpy as np, torch
from qbe.reservoir import (ESN, NVAR, QuantumReservoir, Standardizer, ridge_fit,
                           tfim_unitary)

dev = "cuda"


def check_against_qiskit(n=5, n_in=2, steps=4, seed=3):
    from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector, partial_trace
    rng = np.random.default_rng(seed)
    S = rng.uniform(0, 1, (steps, n_in))
    res = QuantumReservoir(n, n_in, h=0.8, W=0.3, dt=2.0, seed=seed, device=dev)
    mine = res.run_windows(torch.as_tensor(S[None], dtype=torch.float32, device=dev))[0].cpu().numpy()
    U = Operator(tfim_unitary(n, h=0.8, W=0.3, dt=2.0, seed=seed))
    rho = DensityMatrix.from_label("0" * n)
    for t in range(steps):
        rest = partial_trace(rho, list(range(n - n_in, n)))           # drop the top n_in qubits
        amps = [np.array([np.sqrt(1 - S[t, k]), np.sqrt(S[t, k])]) for k in range(n_in)]
        phi = Statevector(amps[0])
        for k in range(1, n_in):
            phi = Statevector(amps[k]).tensor(phi)                     # qubit k above qubit k-1
        rho = DensityMatrix(phi).tensor(rest).evolve(U)
    def ev(lbl): return rho.expectation_value(SparsePauliOp(lbl)).real
    def lab(ops): l = ["I"] * n; [l.__setitem__(n - 1 - q, p) for q, p in ops]; return "".join(l)
    ref = [ev(lab([(q, "Z")])) for q in range(n)]
    ref += [ev(lab([(i, "Z"), (j, "Z")])) for i in range(n) for j in range(i + 1, n)]
    ref += [ev(lab([(q, "X")])) for q in range(n)]
    return float(np.abs(mine - np.array(ref)).max())


def readout_score(F, y, ntr, nva):
    """Ridge with alpha chosen on a validation slice; returns test predictions."""
    F = torch.as_tensor(F, dtype=torch.float32, device=dev); y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    st = Standardizer().fit(F[:ntr]); G = st(F)
    best = None
    for a in (1e-6, 1e-4, 1e-2, 1, 100):
        Wt = ridge_fit(G[:ntr], y[:ntr], a)
        e = ((G[ntr:ntr + nva] @ Wt - y[ntr:ntr + nva]) ** 2).mean().item()
        if best is None or e < best[0]: best = (e, a)
    Wt = ridge_fit(G[:ntr + nva], y[:ntr + nva], best[1])
    return (G[ntr + nva:] @ Wt).cpu().numpy()


def features(kind, u, **kw):
    uu = torch.as_tensor(u, dtype=torch.float32, device=dev)
    if kind == "qrc":   return QuantumReservoir(8, 1, device=dev, **kw).run_sequence(uu[:, None]).cpu().numpy()
    if kind == "qrc_J0": return QuantumReservoir(8, 1, J_zero=True, device=dev, **kw).run_sequence(uu[:, None]).cpu().numpy()
    if kind == "esn":   return ESN(1, N=44, rho=0.9, leak=1.0, device=dev).run_sequence(uu[:, None] * 2 - 1).cpu().numpy()
    if kind == "linear": return u[:, None]
    if kind == "nvar":
        U = torch.as_tensor(np.stack([np.roll(u, j) for j in range(4)], 1)[:, ::-1].copy(), dtype=torch.float32, device=dev)
        return NVAR(k=4, s=1).features(U[:, :, None]).cpu().numpy()


if __name__ == "__main__":
    for n, k, st in ((3, 1, 3), (5, 2, 4), (6, 3, 5)):
        print(f"qiskit check n={n} n_in={k} steps={st}: worst |ours - qiskit| = {check_against_qiskit(n, k, st):.1e}")
    rng = np.random.default_rng(0)
    T, wash, ntr, nva = 4200, 200, 2400, 600
    kinds = {"qrc": dict(h=1.0, W=0.1, dt=10.0), "qrc_J0": dict(h=1.0, W=0.1, dt=10.0),
             "esn": {}, "nvar": {}, "linear": {}}
    # memory capacity
    u = rng.uniform(0, 1, T)
    print(f"\n{'model':<8} {'memory capacity':>16} {'NARMA-10 NMSE':>15}   (features)")
    un = rng.uniform(0, 0.5, T); yn = np.zeros(T)
    for t in range(9, T - 1):
        yn[t + 1] = 0.3 * yn[t] + 0.05 * yn[t] * yn[t - 9:t + 1].sum() + 1.5 * un[t - 9] * un[t] + 0.1
    for kind, kw in kinds.items():
        F = features(kind, u, **kw)[wash:]
        mc = 0.0
        for d in range(1, 31):
            y = np.roll(u, d)[wash:]
            p = readout_score(F, y, ntr, nva); yt = y[ntr + nva:]
            mc += np.corrcoef(p, yt)[0, 1] ** 2 if p.std() > 0 else 0.0
        Fn = features(kind, un / 0.5, **kw)[wash:]
        y = np.roll(yn, -1)[wash:]
        p = readout_score(Fn, y, ntr, nva); yt = y[ntr + nva:]
        nmse = ((p - yt) ** 2).mean() / yt.var()
        print(f"{kind:<8} {mc:>16.2f} {nmse:>15.3f}   ({F.shape[1]})")

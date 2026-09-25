#!/usr/bin/env python
"""Hybrid ACT at every qubit level: quality, QPU time, classical time, round trips.

Same job as qact_vs_act_speed.py / qact_hw_speedups.py: 4 real EEGMAT windows,
6 atoms each. Every level (qbe/hybrid_act.py) runs twice:

  heron   quantum routes executed in Qiskit Aer under IBM Heron's calibrated
          noise model (FakeTorino), on circuits transpiled to that device
  none    the same circuits noiseless, so noise damage is separable from
          shot noise

Quality: reconstruction error, and "selected-atom energy" -- the exact energy
of each atom chosen, as a fraction of the best atom in the dictionary.

Time per window:
  QPU        every circuit execution priced at its measured per-shot cost on
             Heron with zero repetition delay and multi-programming (k copies of
             that register size packed on the chip, transpiled and timed here)
  classical  measured wall time of the classical routes
  round trips  classical -> QPU -> classical turnarounds; NOT priced, because the
             per-job latency of a real service is not in any calibration, but a
             tightly interactive hybrid loop pays it every time
"""
import json
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qact_vs_act_speed import eeg_windows, env_of, width_for
from qbe.hybrid_act import LEVELS, HybridACT
from qbe.qact_hw import build, circuit_duration, two_qubit_stats

ORDER = 6
FILL = 126                     # usable qubits when packing (133 on the chip)


def packed_cost(backend, X, m):
    """Seconds per circuit-shot for m-qubit circuits, packed k-wide, zero delay."""
    k = FILL // m
    while k >= 1:
        circs = [build("windowed", X[i % len(X)], env_of(200.0 + 20 * (i % 5), width_for(m)),
                       0.03125) for i in range(k)]
        assert all(c.num_qubits == m for c in circs)
        big = QuantumCircuit(sum(c.num_qubits for c in circs),
                             sum(c.num_clbits for c in circs))
        oq = oc = 0
        for c in circs:
            big.compose(c, qubits=range(oq, oq + m), clbits=range(oc, oc + m), inplace=True)
            oq += m
            oc += m
        try:
            tq = transpile(big, backend=backend, optimization_level=3, seed_transpiler=7)
            dur = circuit_duration(tq, backend.target)
            return dict(m=m, k=k, packed_us=dur * 1e6, per_circuit_shot_us=dur * 1e6 / k,
                        cz_per_copy=two_qubit_stats(tq)[0] / k)
        except Exception:
            k -= 1
    raise RuntimeError(m)


def main():
    X = eeg_windows()
    K = len(X)
    backend = FakeTorino()
    print(f"job: {K} real EEGMAT windows x {ORDER} atoms\n", flush=True)

    print("multi-programming throughput per register size (FakeTorino, zero delay):",
          flush=True)
    cost = {}
    # refinement can shrink an atom until its window fits 5 qubits
    for m in (5, 6, 7, 8, 9):
        pc = packed_cost(backend, X, m)
        cost[m] = pc["per_circuit_shot_us"] * 1e-6
        print(f"   {m} qubits: {pc['k']:2d} copies on the chip, {pc['packed_us']:6.1f} us "
              f"per shot -> {pc['per_circuit_shot_us']:5.1f} us per circuit-shot, "
              f"{pc['cz_per_copy']:5.0f} CZ per copy", flush=True)

    results = []
    for noise in ("heron", "none"):
        for q_max in LEVELS:
            h = HybridACT(q_max, noise=noise, backend=backend)
            t0 = time.perf_counter()
            errs = [h.decompose(x, order=ORDER)[0] for x in X]
            wall = time.perf_counter() - t0
            shots = sum(s for _, s in h.qpu_log) / K
            # an unmeasured register size is priced as the next larger measured one
            price = lambda m: cost[min(x for x in cost if x >= m)]
            qpu = sum(s * price(m) for m, s in h.qpu_log) / K
            byq = {m: sum(s for mm, s in h.qpu_log if mm == m) / K for m in (5, 6, 7, 8, 9)}
            frac_q = h.evals["quantum"] / max(1, h.evals["quantum"] + h.evals["classical"])
            row = dict(noise=noise, q_max=q_max, recon_err=float(np.mean(errs)),
                       per_window_err=[float(e) for e in errs],
                       sel_quality=float(np.mean(h.sel_quality)),
                       sel_quality_min=float(np.min(h.sel_quality)),
                       qpu_s=qpu, classical_s=h.classical_s / K,
                       round_trips=h.round_trips / K, shots=shots,
                       shots_by_register=byq, quantum_eval_fraction=frac_q,
                       sim_wall_s=wall / K)
            results.append(row)
            print(f"[{noise:5s} q{q_max}] recon err {row['recon_err']:.3f}  selected-atom "
                  f"energy {row['sel_quality']:.2f} (worst {row['sel_quality_min']:.2f})  "
                  f"QPU {qpu:6.2f} s  classical {row['classical_s'] * 1e3:6.1f} ms  "
                  f"round trips {row['round_trips']:5.0f}  shots {shots:>9,.0f}  "
                  f"({wall / K:.0f}s sim per window)", flush=True)

    print(f"\n{'level':<8}{'quantum share':>14}{'recon err':>12}{'noiseless':>11}"
          f"{'atom energy':>13}{'QPU s':>9}{'classical':>11}{'round trips':>13}")
    for q_max in LEVELS:
        hn = next(r for r in results if r["noise"] == "heron" and r["q_max"] == q_max)
        nn = next(r for r in results if r["noise"] == "none" and r["q_max"] == q_max)
        print(f"q{q_max:<7}{hn['quantum_eval_fraction'] * 100:>12.0f}%"
              f"{hn['recon_err']:>12.3f}{nn['recon_err']:>11.3f}{hn['sel_quality']:>13.2f}"
              f"{hn['qpu_s']:>9.2f}{hn['classical_s'] * 1e3:>9.1f}ms{hn['round_trips']:>13.0f}")
    json.dump(dict(order=ORDER, windows=K, packing_cost_us={m: c * 1e6 for m, c in cost.items()},
                   results=results), open("results/hybrid_act_levels.json", "w"), indent=1)
    print("\nwrote results/hybrid_act_levels.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Error correction for the higher Hybrid ACT levels (q8, q9).

Once encoded in a surface code, a 9-qubit QACT circuit needs ~10^5 physical
qubits, and its arbitrary-angle rotations rule out stabilizer simulation. So
error correction is run the way that can actually be executed:

1. COST    Microsoft's resource estimator (qdk) sizes a surface code for the real
           QACT circuits of every register size: logical qubits, code distance,
           T-state factories, physical qubits, runtime per shot. Four qubit
           technologies: IBM Heron as calibrated (FakeTorino medians), Heron with
           readout improved to 1e-3 (hypothetical), and Microsoft's
           superconducting presets GATE_NS_E3 and GATE_NS_E4.
2. EFFECT  a code's job is to hand the algorithm a small logical error rate. The
           code is sized to an error budget (the probability that one shot
           fails), so the circuits are run in Aer at the LOGICAL level:
           all-to-all logical qubits, depolarizing error on every logical
           operation, scaled so a whole shot fails with probability = budget.
           a) per circuit, over a sweep of budgets, to find the cheapest code
              that still selects the right atom;
           b) end to end through Hybrid ACT (q8, q9) at that budget.
3. TIME    runtime per shot from (1) x shots per window from (2b).
"""
import json
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qdk.estimator import EstimatorParams, QECScheme, QubitParams
from qdk.qiskit import estimate

from qact_hardware import in_band
from qact_vs_act_speed import eeg_windows, env_of, width_for
from qbe.hybrid_act import HybridACT
from qbe.qact_hw import build, distribution_from_counts, windowed_target

BASIS = ["cx", "rz", "sx", "x", "h", "measure", "if_else"]   # logical gate set
BUDGETS = (0.3, 0.1, 0.03, 0.01)
REGS = (5, 6, 7, 8, 9)
C_RATE = 0.03125                     # 8 Hz/s chirp, sample units
SHOTS_SWEEP = 5000
ORDER = 6


def op_count(tq):
    """Logical operations per shot, descending into classically controlled blocks."""
    def walk(c):
        k = 0
        for inst in c.data:
            op = inst.operation
            if getattr(op, "blocks", None):
                k += sum(walk(b) for b in op.blocks[:1])
            elif op.name not in ("barrier",):
                k += 1
        return k
    return walk(tq)


def logical_noise(p):
    """Depolarizing error p on every logical gate, and on every logical readout."""
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["rz", "sx", "x", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def heron_medians():
    t = FakeTorino().target
    med = lambda op, a: float(np.median([getattr(v, a) for v in t[op].values()
                                         if v is not None and getattr(v, a) is not None]))
    return dict(cz=med("cz", "error"), sx=med("sx", "error"), meas=med("measure", "error"),
                t2q=med("cz", "duration") * 1e9, t1q=med("sx", "duration") * 1e9,
                tm=med("measure", "duration") * 1e9)


def params(tech, budget, h):
    p = EstimatorParams()
    p.qec_scheme.name = QECScheme.SURFACE_CODE
    p.error_budget = budget
    if tech in ("GATE_NS_E3", "GATE_NS_E4"):
        p.qubit_params.name = getattr(QubitParams, tech)
        return p
    q = p.qubit_params
    q.name = tech
    q.instruction_set = "GateBased"
    q.one_qubit_gate_time = f"{h['t1q']:.0f} ns"
    q.t_gate_time = f"{h['t1q']:.0f} ns"
    q.two_qubit_gate_time = f"{h['t2q']:.0f} ns"
    q.one_qubit_measurement_time = f"{h['tm']:.0f} ns"
    q.one_qubit_gate_error_rate = h["sx"]
    q.t_gate_error_rate = h["sx"]
    q.idle_error_rate = h["sx"]
    q.two_qubit_gate_error_rate = h["cz"]
    q.one_qubit_measurement_error_rate = 1e-3 if tech == "heron_readout_1e-3" else h["meas"]
    return p


def resource_table(X, h):
    techs = ("heron_as_calibrated", "heron_readout_1e-3", "GATE_NS_E3", "GATE_NS_E4")
    circ, table = {}, {}
    for m in REGS:
        qc = build("windowed", X[0], env_of(256.0, width_for(m)), C_RATE)
        circ[m] = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)
    for tech in techs:
        for m in REGS:
            for b in BUDGETS:
                try:
                    r = estimate(circ[m], params(tech, b, h))
                    d = r.data() if hasattr(r, "data") else r
                    pc, bd = d["physicalCounts"], d["physicalCounts"]["breakdown"]
                    table[(tech, m, b)] = dict(
                        ok=True, logical_qubits=bd["algorithmicLogicalQubits"],
                        distance=d["logicalQubit"]["codeDistance"],
                        physical_qubits=pc["physicalQubits"], runtime_s=pc["runtime"] * 1e-9,
                        t_states=bd["numTstates"], t_factories=bd["numTfactories"])
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:          # qdk's EstimatorError
                    msg = " ".join(str(exc).split())
                    table[(tech, m, b)] = dict(ok=False, error=msg[-160:])
    return circ, table


def sweep(X, G):
    """Selection quality of single long-atom circuits vs code budget."""
    backend = FakeTorino()
    heron = AerSimulator.from_backend(backend)
    cases = [(x, ld) for x in X for ld in (width_for(8), 3.9, 5.1)]
    conds = [("physical Heron, no QEC", None)] + [(f"budget {b}", b) for b in BUDGETS] \
        + [("perfect (noiseless)", 0.0)]
    rows = {name: [] for name, _ in conds}
    for x, ld in cases:
        env = env_of(256.0, ld)
        want, bins, m = windowed_target(x, env, C_RATE, 9)
        mask = in_band(bins, C_RATE, 256)
        best = want[mask].max()
        qc = build("windowed", x, env, C_RATE)
        lc = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)
        for name, b in conds:
            if b is None:
                tq = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=7)
                counts = heron.run(tq, shots=SHOTS_SWEEP, seed_simulator=3).result().get_counts()
            else:
                sim = AerSimulator(noise_model=logical_noise(b / G[m])) if b else AerSimulator()
                counts = sim.run(lc, shots=SHOTS_SWEEP, seed_simulator=3).result().get_counts()
            p, _ = distribution_from_counts(counts, m)
            sel = int(np.argmax(np.where(mask, p, -1)))
            rows[name].append(float(want[sel] / best))
    return rows


def main():
    t0 = time.time()
    X = eeg_windows()
    h = heron_medians()
    print(f"Heron medians: CZ {h['cz']:.2e} ({h['t2q']:.0f} ns), SX {h['sx']:.2e} "
          f"({h['t1q']:.0f} ns), readout {h['meas']:.2e} ({h['tm']:.0f} ns)\n", flush=True)

    # ---- 1. cost of the code ----
    circ, table = resource_table(X, h)
    G = {m: op_count(circ[m]) for m in REGS}
    print("logical operations per shot:", {m: G[m] for m in REGS}, flush=True)
    for tech in ("heron_as_calibrated", "heron_readout_1e-3", "GATE_NS_E3", "GATE_NS_E4"):
        print(f"\n{tech}: surface code for a 9-qubit QACT circuit")
        for b in BUDGETS:
            r = table[(tech, 9, b)]
            if r["ok"]:
                print(f"   budget {b:<5} d={r['distance']:<3} {r['logical_qubits']} logical -> "
                      f"{r['physical_qubits']:>9,} physical qubits, {r['runtime_s'] * 1e3:9.2f} ms "
                      f"per shot, {r['t_states']:,} T states", flush=True)
            else:
                print(f"   budget {b:<5} NOT POSSIBLE: {r['error']}", flush=True)

    # ---- 2a. how much protection is enough? ----
    print(f"\nselection quality of single long-atom circuits (6 windows x 3 widths = 8/9 "
          f"qubits; {SHOTS_SWEEP} shots):", flush=True)
    Xs = eeg_windows(6)
    rows = sweep(Xs, G)
    for name, v in rows.items():
        print(f"   {name:<26} mean {np.mean(v):.2f}  worst {np.min(v):.2f}", flush=True)
    ok = [b for b in BUDGETS if np.mean(rows[f"budget {b}"]) >= 0.95
          and np.min(rows[f"budget {b}"]) >= 0.8]
    chosen = max(ok) if ok else min(BUDGETS)
    print(f"-> cheapest code that selects like q7 on Heron (mean >= 0.95, worst >= 0.8): "
          f"budget {chosen}", flush=True)

    # ---- 2b. end to end through Hybrid ACT at that budget ----
    end = {}
    for q_max in (8, 9):
        hy = HybridACT(q_max, noise="logical", basis_gates=BASIS,
                       noise_model=logical_noise(chosen / G[9]))
        t1 = time.time()
        errs = [hy.decompose(x, order=ORDER)[0] for x in X]
        shots_m = {m: sum(s for mm, s in hy.qpu_log if mm == m) / len(X) for m in REGS}
        end[q_max] = dict(recon_err=float(np.mean(errs)), per_window=[float(e) for e in errs],
                          sel_quality=float(np.mean(hy.sel_quality)),
                          sel_worst=float(np.min(hy.sel_quality)), shots_by_register=shots_m,
                          round_trips=hy.round_trips / len(X),
                          classical_s=hy.classical_s / len(X))
        print(f"[QEC budget {chosen} q{q_max}] recon err {end[q_max]['recon_err']:.3f}  atom "
              f"quality {end[q_max]['sel_quality']:.2f} (worst {end[q_max]['sel_worst']:.2f})  "
              f"shots/window {sum(shots_m.values()):,.0f}  ({(time.time() - t1) / len(X):.0f}s "
              f"sim per window)", flush=True)

    # ---- 3. time per window on an error-corrected machine ----
    prev = {(r["noise"], r["q_max"]): r for r in
            json.load(open("results/hybrid_act_levels.json"))["results"]}
    print(f"\n{'level':<6}{'technology':<22}{'recon err':>10}{'physical qubits':>17}"
          f"{'time per window':>17}")
    print(f"{'q0':<6}{'classical':<22}{prev[('heron', 0)]['recon_err']:>10.3f}"
          f"{'—':>17}{prev[('heron', 0)]['classical_s'] * 1e3:>14.0f} ms")
    print(f"{'q7':<6}{'Heron, no QEC':<22}{prev[('heron', 7)]['recon_err']:>10.3f}"
          f"{'133':>17}{prev[('heron', 7)]['qpu_s']:>15.2f} s")
    for q_max in (8, 9):
        print(f"{'q' + str(q_max):<6}{'Heron, no QEC':<22}{prev[('heron', q_max)]['recon_err']:>10.3f}"
              f"{'133':>17}{prev[('heron', q_max)]['qpu_s']:>15.2f} s")
        for tech in ("heron_as_calibrated", "heron_readout_1e-3", "GATE_NS_E3", "GATE_NS_E4"):
            if not all(table[(tech, m, chosen)]["ok"] for m in REGS):
                print(f"{'':<6}{tech:<22}{'code above threshold — not possible':>44}")
                continue
            secs = sum(s * table[(tech, m, chosen)]["runtime_s"]
                       for m, s in end[q_max]["shots_by_register"].items())
            phys = max(table[(tech, m, chosen)]["physical_qubits"] for m in REGS
                       if end[q_max]["shots_by_register"][m] > 0)
            end[q_max].setdefault("time_s", {})[tech] = secs
            end[q_max].setdefault("physical_qubits", {})[tech] = phys
            t = f"{secs / 3600:.1f} h" if secs >= 3600 else f"{secs / 60:.1f} min"
            print(f"{'':<6}{tech:<22}{end[q_max]['recon_err']:>10.3f}{phys:>17,}{t:>17}")

    json.dump(dict(heron_medians=h, ops_per_shot=G, budgets=BUDGETS, chosen_budget=chosen,
                   resources={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in table.items()},
                   sweep=rows, end_to_end=end),
              open("results/qact_qec.json", "w"), indent=1)
    print(f"\nwrote results/qact_qec.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

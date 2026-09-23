#!/usr/bin/env python
"""QACT on real hardware: what one circuit evaluation costs, and how to cut it.

All simulation is Qiskit Aer; all gate counts come from the Qiskit transpiler
targeting IBM Heron (FakeTorino: heavy-hex, native CZ/RZ/SX/X, dynamic circuits);
all noise comes from that backend's calibrated noise model.

Signals: real EEGMAT windows (512 samples = 9 qubits). For each, the four
constructions in qbe/qact_hw.py are checked for
  1. correctness  noiseless Aer vs the exact target distribution
  2. cost         two-qubit gates / depth after transpiling to the device
  3. fidelity     Hellinger fidelity of the NOISY output to the ideal one, and
                  whether the ideal peak -- the atom QACT would select -- is
                  still the modal outcome
"""
import json
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qbe.acquire import read_eegmat_file
from qbe.qact_hw import (build, distribution_from_counts, target_distribution,
                         two_qubit_stats, windowed_target)

ROOT = "/home/kc/EEG-Memristor-ACT/data/eegmat"
FS, N = 256.0, 512
KINDS = ("baseline", "folded", "aqft", "semiclassical", "windowed")
DYNAMIC = ("semiclassical", "windowed")
ALL_TO_ALL = ["cz", "rz", "sx", "x", "measure", "if_else"]
SHOTS_EXACT = 400_000          # noiseless: tight check against the exact target
SHOTS_NOISY = 20_000


def windows(n_win=3, seed=0):
    rec = read_eegmat_file(f"{ROOT}/Subject00_1.edf", 0, target_fs=FS)
    x = rec.data
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_win):
        ch = rng.integers(x.shape[0])
        st = rng.integers(0, x.shape[1] - N)
        w = x[ch, st:st + N].astype(float)
        out.append(w - w.mean())
    return out


def envelope(tc, logdt):
    t = np.arange(N)
    return np.exp(-((t - tc) ** 2) / (2 * np.exp(logdt) ** 2))


def in_band(bins, c, tc, band=(0.5, 45.0)):
    """QACT's own dictionary mask: non-negative effective frequency whose centre
    frequency f + 2 c tc falls in band."""
    fc = (bins + 2 * c * tc) * FS / N
    return (bins < N // 2) & (fc >= band[0]) & (fc <= band[1])


def selection_quality(p_obs, want, mask):
    """Energy of the atom the (noisy) histogram would select, as a fraction of
    the best in-band atom's energy. 1.0 = the best atom; near-ties score ~1."""
    sel = int(np.argmax(np.where(mask, p_obs, -1)))
    return float(want[sel] / want[mask].max())


def fid(p, q):
    return float(np.sum(np.sqrt(p * q)) ** 2)


def main():
    backend = FakeTorino()
    ideal = AerSimulator()
    noisy = AerSimulator.from_backend(backend)
    c = 8.0 * N / (2 * FS ** 2)                    # an 8 Hz/s chirp, sample units
    cases = [(w, envelope(256, ld), ld) for w in windows() for ld in (1.5, 2.7, 3.9, 5.1)]
    rows, t0 = [], time.time()
    rng = np.random.default_rng(3)
    for ci, (r, env, ld) in enumerate(cases):
        full = target_distribution(r, env, c)
        # the in-band peak on the FULL grid: what the windowed coarse grid is
        # compared against (the unmasked argmax can sit at a negative frequency)
        full_peak = int(np.argmax(np.where(in_band(np.arange(N), c, 256), full, -1)))
        for kind in KINDS:
            qc = build(kind, r, env, c)
            post = kind == "baseline"
            if kind == "windowed":
                want, bins, nq = windowed_target(r, env, c, 9)
            else:
                want, bins, nq = full, np.arange(N), 9
            mask = in_band(bins, c, 256)
            peak = int(np.argmax(np.where(mask, want, -1)))
            # 1. correctness, noiseless. Circuits without mid-circuit measurement
            # get EXACT probabilities from Aer's statevector; the dynamic circuit
            # can only be sampled, so it is judged against the TVD that sampling
            # the exact target with the same number of shots produces.
            if kind in DYNAMIC:
                got, kept = distribution_from_counts(
                    ideal.run(transpile(qc, ideal), shots=SHOTS_EXACT,
                              seed_simulator=1).result().get_counts(), nq)
                floor = 0.5 * float(np.abs(
                    rng.multinomial(SHOTS_EXACT, want) / SHOTS_EXACT - want).sum())
            else:
                qs = build(kind, r, env, c, measure=False)
                qs.save_statevector()
                amp = np.asarray(ideal.run(transpile(qs, ideal, optimization_level=0))
                                 .result().get_statevector())
                pr = np.abs(amp) ** 2
                if post:                       # ancilla is the top qubit
                    kept = float(pr[:N].sum())
                    got = pr[:N] / kept
                else:
                    kept, got = 1.0, pr
                floor = 0.0
            tvd = 0.5 * float(np.abs(got - want).sum())
            # 2. cost on the device
            tq = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=7)
            n2, d2, d = two_qubit_stats(tq)
            n2_a2a = two_qubit_stats(transpile(qc, basis_gates=ALL_TO_ALL,
                                               optimization_level=3,
                                               seed_transpiler=7))[0]
            # 3. noisy execution
            pn, kept_n = distribution_from_counts(
                noisy.run(tq, shots=SHOTS_NOISY, seed_simulator=2).result().get_counts(),
                nq, postselect_ancilla=post)
            rows.append(dict(case=ci, logdt=ld, kind=kind, tvd_ideal=tvd,
                             tvd_floor=floor,
                             kept=kept, cz=n2, cz_a2a=n2_a2a, cz_depth=d2, depth=d,
                             qubits=nq, circuits_full_axis=2 ** (9 - nq),
                             grid_bin_err=abs(int(bins[peak]) - full_peak),
                             fidelity=fid(pn, want),
                             peak_ok=int(np.argmax(np.where(mask, pn, -1)) == peak),
                             sel_quality=selection_quality(pn, want, mask),
                             useful_shot_frac=kept_n))
            print(f"[{ci:2d} logDt={ld}] {kind:<13} TVD(ideal) {tvd:.2e} "
                  f"(floor {floor:.1e})  post-sel {kept:.3f}  "
                  f"q {nq}  CZ {n2:5d} (a2a {n2_a2a:4d})  CZ-depth {d2:5d}  "
                  f"fidelity {rows[-1]['fidelity']:.3f}  "
                  f"selected-atom energy {rows[-1]['sel_quality']:.2f}  "
                  f"useful shots {kept_n:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\nsummary over", len(cases), "EEG window x envelope cases; TVD is exact "
          "except semiclassical (sampled, floor in brackets):")
    print(f"{'construction':<14}{'max TVD':>18}{'CZ':>8}{'a2a CZ':>8}{'CZ depth':>10}"
          f"{'fidelity':>10}{'sel. energy':>12}{'useful shots':>14}")
    for kind in KINDS:
        R = [x for x in rows if x["kind"] == kind]
        fl = max(x["tvd_floor"] for x in R)
        tv = f"{max(x['tvd_ideal'] for x in R):.1e}" + (f" [{fl:.1e}]" if fl else "")
        print(f"{kind:<14}{tv:>18}"
              f"{np.mean([x['cz'] for x in R]):>8.0f}"
              f"{np.mean([x['cz_a2a'] for x in R]):>8.0f}"
              f"{np.mean([x['cz_depth'] for x in R]):>10.0f}"
              f"{np.mean([x['fidelity'] for x in R]):>10.3f}"
              f"{np.mean([x['sel_quality'] for x in R]):>12.2f}"
              f"{np.mean([x['useful_shot_frac'] for x in R]):>14.2f}")
    print("\nwindowed construction, by envelope width:")
    for ld in (1.5, 2.7, 3.9, 5.1):
        R = [x for x in rows if x["kind"] == "windowed" and x["logdt"] == ld]
        F = [x for x in rows if x["kind"] == "folded" and x["logdt"] == ld]
        print(f"  logDt {ld}: {R[0]['qubits']} qubits, CZ {np.mean([x['cz'] for x in R]):.0f}"
              f" vs folded {np.mean([x['cz'] for x in F]):.0f}, fidelity "
              f"{np.mean([x['fidelity'] for x in R]):.3f} vs "
              f"{np.mean([x['fidelity'] for x in F]):.3f}, selected-atom energy "
              f"{np.mean([x['sel_quality'] for x in R]):.2f} vs "
              f"{np.mean([x['sel_quality'] for x in F]):.2f}, full axis needs "
              f"{R[0]['circuits_full_axis']} circuit(s), coarse-grid peak error "
              f"{max(x['grid_bin_err'] for x in R)} bins")
    json.dump(rows, open("results/qact_hardware.json", "w"), indent=1)
    print("\nwrote results/qact_hardware.json")


if __name__ == "__main__":
    main()

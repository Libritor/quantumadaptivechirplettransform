#!/usr/bin/env python
"""QACT atom selection on a real IBM Heron QPU, certified against the exact distribution.

The windowed construction of qbe/qact_hw.py (envelope folded into the loaded state,
window-local register, chirp as O(m^2) phase gates, semiclassical inverse QFT) was
so far executed only under a fake backend's noise model. This script runs it on a
real device and asks the one question that matters for matching pursuit: does the
noisy histogram still select the atom the exact distribution selects?

Two variants of the same distribution are run:
  dyn   the semiclassical (Griffiths-Niu) inverse QFT: mid-circuit measurement and
        classically controlled phases, zero two-qubit gates in the QFT (dynamic circuit)
  uni   a unitary inverse QFT on the same window register (no feed-forward), which
        lets the runtime apply gate twirling as well as dynamical decoupling

For every circuit the classical reference is the exact target of qbe.qact_hw
(numpy FFT). Reported per circuit: total-variation distance and Hellinger fidelity
of the device histogram to the exact one, whether the in-band argmax coincides with
the exact in-band peak, and the selected-atom energy (energy of the atom the device
histogram selects as a fraction of the best in-band atom's). The same ISA circuits
are also run through a noiseless simulator (correctness) and through the fake
backend's calibrated noise model (prediction), so the device row can be read against
both. Job ids, backend, calibration date and options are written next to the numbers.

Signals: PhysioNet EEGMMIDB baseline runs (public; fetched by MNE on first use),
512 samples (3.2 s at 160 Hz), zero-mean, centred in the run.

    python paper/iclr2027/exp/qpu_selection.py --dry-run      # simulators only, $0
    python paper/iclr2027/exp/qpu_selection.py                 # + real device (~20 s QPU)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.circuit.library import QFTGate, StatePreparation  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

from qbe.qact import linear_phase_gates, quadratic_phase_gates  # noqa: E402
from qbe.qact_hw import (build, distribution_from_counts, target_distribution,  # noqa: E402
                         two_qubit_stats, window_register, windowed_target)

FS, N, n = 160.0, 512, 9
BAND = (0.5, 45.0)
TC = 256.0
OUT = ROOT / "paper" / "iclr2027" / "results"


# ----------------------------------------------------------------------------- data
def eeg_windows(subjects=(1, 2, 3), channels=("O1", "Cz"), data_dir=None):
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    data_dir = data_dir or str(Path.home() / "mne_data")
    out = []
    for s in subjects:
        f = eegbci.load_data(s, [1], path=data_dir, update_path=False, verbose=False)[0]
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        eegbci.standardize(raw)
        st = (raw.n_times - N) // 2
        for ch in channels:
            x = raw.get_data(picks=[ch])[0, st:st + N].astype(np.float64)
            x = x - x.mean()
            out.append(dict(subject=s, channel=ch, x=x / np.abs(x).max()))
    return out


def envelope(tc, logdt):
    t = np.arange(N)
    return np.exp(-((t - tc) ** 2) / (2 * np.exp(logdt) ** 2))


def in_band(bins, c, tc, band=BAND):
    fc = (bins + 2 * c * tc) * FS / N
    return (bins < N // 2) & (fc >= band[0]) & (fc <= band[1])


# ----------------------------------------------------------------------------- circuits
def _chirp(qc, qubits, m, c, offset):
    for g in quadratic_phase_gates(m, -c, N) + linear_phase_gates(m, -offset, N):
        if g[0] == "p":
            qc.p(g[2], qubits[g[1]])
        else:
            qc.cp(g[3], qubits[g[1]], qubits[g[2]])


def build_windowed_unitary(r, env, c, offset=0):
    """The windowed construction with a unitary inverse QFT (no mid-circuit measurement)."""
    s = env * r
    s = s / np.linalg.norm(s)
    m, t0 = window_register(env, n)
    sw = s[t0:t0 + 2 ** m]
    sw = sw / np.linalg.norm(sw)
    tw = QuantumRegister(m, "t")
    ck = ClassicalRegister(m, "k")
    qc = QuantumCircuit(tw, ck)
    qc.append(StatePreparation(sw), tw)
    _chirp(qc, list(tw), m, c, 2 * c * t0 + offset)
    qc.append(QFTGate(m).inverse(), tw)
    qc.measure(tw, ck)
    return qc


def make_case(w, logdt, c, variant):
    env = envelope(TC, logdt)
    want, bins, m = windowed_target(w["x"], env, c, n)
    qc = build("windowed", w["x"], env, c) if variant == "dyn" else \
        build_windowed_unitary(w["x"], env, c)
    mask = in_band(bins, c, TC)
    peak = int(np.argmax(np.where(mask, want, -1)))
    return dict(subject=w["subject"], channel=w["channel"], logdt=logdt, variant=variant,
                qubits=m, want=want, bins=bins, mask=mask, peak=peak, qc=qc)


# ----------------------------------------------------------------------------- scoring
def score(p, case):
    want, mask, peak = case["want"], case["mask"], case["peak"]
    sel = int(np.argmax(np.where(mask, p, -1)))
    return dict(tvd=0.5 * float(np.abs(p - want).sum()),
                fidelity=float(np.sum(np.sqrt(p * want)) ** 2),
                peak_ok=int(sel == peak),
                selected_atom_energy=float(want[sel] / want[mask].max()),
                p_peak_device=float(p[peak]), p_peak_exact=float(want[peak]))


def counts_to_p(counts, m):
    p = np.zeros(2 ** m)
    tot = 0
    for key, v in counts.items():
        p[int(key.split()[-1], 2)] += v
        tot += v
    return p / max(tot, 1)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="simulators only")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--widths", default="1.5,2.7", help="logDt values for every window")
    ap.add_argument("--long-width", type=float, default=3.9,
                    help="one 9-qubit long-atom case on the first window (0 = skip)")
    ap.add_argument("--rate-hz-s", type=float, default=8.0)
    ap.add_argument("--max-qpu-seconds", type=float, default=90.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)

    c = args.rate_hz_s * N / (2 * FS ** 2)
    wins = eeg_windows()
    cases = []
    for w in wins:
        for ld in (float(v) for v in args.widths.split(",")):
            for variant in ("dyn", "uni"):
                cases.append(make_case(w, ld, c, variant))
    if args.long_width > 0:
        for variant in ("dyn", "uni"):
            cases.append(make_case(wins[0], args.long_width, c, variant))
    print(f"{len(cases)} circuits: {len(wins)} windows x widths {args.widths} x 2 variants"
          + (f" + 2 long-atom (logDt {args.long_width})" if args.long_width > 0 else ""))

    # --- exact target vs noiseless simulation (correctness of the constructions)
    ideal = AerSimulator()
    for k, cs in enumerate(cases):
        tq = generate_preset_pass_manager(optimization_level=0, backend=ideal).run(cs["qc"])
        cnt = ideal.run(tq, shots=200_000, seed_simulator=1).result().get_counts()
        p = counts_to_p(cnt, cs["qubits"])
        cs["ideal_sim"] = score(p, cs)
        cs["ideal_sim"]["shots"] = 200_000
    print("noiseless simulator vs exact target: max TVD "
          f"{max(cs['ideal_sim']['tvd'] for cs in cases):.4f} "
          f"(sampling floor at 200k shots ~ {0.5 * math.sqrt(64 / 200_000):.4f}); "
          f"in-band peak reproduced in {sum(cs['ideal_sim']['peak_ok'] for cs in cases)}/{len(cases)}")

    # --- device target: transpile once, price, predict under the fake noise model
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fake = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=3, backend=fake, seed_transpiler=7)
    noisy = AerSimulator.from_backend(fake)
    est = 0.0
    for cs in cases:
        isa = pm.run(cs["qc"])
        cs["isa_fake"] = isa
        cz, czd, d = two_qubit_stats(isa)
        cs.update(cz=cz, cz_depth=czd, depth=d)
        try:
            dur = isa.estimate_duration(fake.target, unit="s") if cs["variant"] == "uni" else None
        except Exception:
            dur = None
        cs["duration_s"] = dur
        est += args.shots * ((dur or 1.2e-4) + 2.5e-4)
        cnt = noisy.run(isa, shots=args.shots, seed_simulator=2).result().get_counts()
        cs["fake_noisy"] = score(counts_to_p(cnt, cs["qubits"]), cs)
        print(f"  S{cs['subject']:03d} {cs['channel']:>3} logDt {cs['logdt']} {cs['variant']}: "
              f"{cs['qubits']} q, CZ {cz:4d} (depth {czd:4d}), fake-noise fidelity "
              f"{cs['fake_noisy']['fidelity']:.3f}, sel. energy "
              f"{cs['fake_noisy']['selected_atom_energy']:.2f}, peak "
              f"{'ok' if cs['fake_noisy']['peak_ok'] else 'LOST'}")
    print(f"estimated QPU time for {len(cases)} x {args.shots} shots: ~{est:.1f} s "
          f"(circuit + 250 us repetition delay)")

    rows = [{k: v for k, v in cs.items() if k not in ("qc", "isa_fake", "want", "bins", "mask")}
            for cs in cases]
    for r in rows:
        r["want_peak_index"] = r.pop("peak")
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        json.dump(dict(fs=FS, N=N, rate_hz_s=args.rate_hz_s, shots=args.shots, rows=rows),
                  open(OUT / f"qpu_selection_dryrun_{tag}.json", "w"), indent=1)
        print("dry run written")
        return

    # --- real device
    if est > args.max_qpu_seconds:
        sys.exit(f"estimated {est:.0f} s exceeds --max-qpu-seconds {args.max_qpu_seconds}")
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    if not tok:
        sys.exit("IBM_QUANTUM_TOKEN not set")
    svc0 = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
    free = [dict(i) for i in svc0.instances() if str(dict(i).get("plan", "")).lower() == "open"]
    if not free:
        sys.exit("no open-plan instance on this account")
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=free[0]["crn"])
    usage = svc.usage()
    remaining = float(usage.get("usage_remaining_seconds", 0))
    print(f"open-plan instance: {remaining:.0f} s remaining this period")
    if remaining < est + 60:
        sys.exit("not enough free quota; refusing to run")
    backend = svc.backend(args.backend)
    props_date = None
    try:
        props_date = str(backend.properties().last_update_date)
    except Exception:
        pass
    pm_dev = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    for cs in cases:
        cs["isa"] = pm_dev.run(cs["qc"])
        cz, czd, d = two_qubit_stats(cs["isa"])
        cs["cz_device"], cs["cz_depth_device"], cs["depth_device"] = cz, czd, d

    def run_job(sub, twirl):
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = args.shots
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        if twirl:
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.num_randomizations = 16
            sampler.options.twirling.shots_per_randomization = max(1, args.shots // 16)
        job = sampler.run([cs["isa"] for cs in sub])
        print(f"submitted job {job.job_id()} ({len(sub)} circuits, twirl={twirl}); waiting...",
              flush=True)
        t0 = time.time()
        res = job.result()
        print(f"  done in {time.time() - t0:.0f} s wall", flush=True)
        for cs, pr in zip(sub, res):
            cnt = pr.data.k.get_counts()
            cs["device"] = score(counts_to_p(cnt, cs["qubits"]), cs)
            cs["device"]["job_id"] = job.job_id()
            cs["device"]["counts"] = {k: int(v) for k, v in cnt.items()}
            cs["device"]["shots"] = int(sum(cnt.values()))
        try:
            md = res.metadata
            return job.job_id(), {k: str(v)[:200] for k, v in dict(md).items()} if md else {}
        except Exception:
            return job.job_id(), {}

    jobs = {}
    dyn = [cs for cs in cases if cs["variant"] == "dyn"]
    uni = [cs for cs in cases if cs["variant"] == "uni"]
    try:
        jobs["dyn"] = run_job(dyn, twirl=False)
    except Exception as e:  # dynamic circuits + DD may be refused; retry without DD
        print("dynamic job failed:", type(e).__name__, str(e)[:300])
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = args.shots
        job = sampler.run([cs["isa"] for cs in dyn])
        print(f"resubmitted dynamic job {job.job_id()} with default options", flush=True)
        res = job.result()
        for cs, pr in zip(dyn, res):
            cnt = pr.data.k.get_counts()
            cs["device"] = score(counts_to_p(cnt, cs["qubits"]), cs)
            cs["device"].update(job_id=job.job_id(), counts={k: int(v) for k, v in cnt.items()},
                                shots=int(sum(cnt.values())), options="default")
        jobs["dyn"] = (job.job_id(), {"options": "default"})
    jobs["uni"] = run_job(uni, twirl=True)
    usage_after = svc.usage()

    print("\nreal device vs exact target:")
    print(f"{'case':<28}{'q':>3}{'CZ':>6}{'fid sim':>9}{'fid dev':>9}{'sel.E sim':>11}"
          f"{'sel.E dev':>11}{'peak':>6}")
    for cs in cases:
        print(f"S{cs['subject']:03d} {cs['channel']:>3} ld{cs['logdt']:<4} {cs['variant']:<4}"
              f"{cs['qubits']:>3}{cs['cz_device']:>6}{cs['fake_noisy']['fidelity']:>9.3f}"
              f"{cs['device']['fidelity']:>9.3f}{cs['fake_noisy']['selected_atom_energy']:>11.2f}"
              f"{cs['device']['selected_atom_energy']:>11.2f}"
              f"{'ok' if cs['device']['peak_ok'] else 'LOST':>6}")
    rows = [{k: v for k, v in cs.items() if k not in ("qc", "isa", "isa_fake", "want", "bins", "mask")}
            for cs in cases]
    for r in rows:
        r["want_peak_index"] = r.pop("peak")
    json.dump(dict(backend=args.backend, calibration_last_update=props_date, fs=FS, N=N,
                   rate_hz_s=args.rate_hz_s, shots=args.shots, jobs=jobs,
                   quota_before_s=remaining,
                   quota_after_s=float(usage_after.get("usage_remaining_seconds", -1)),
                   rows=rows),
              open(OUT / f"qpu_selection_{tag}.json", "w"), indent=1)
    print(f"\nwrote {OUT / f'qpu_selection_{tag}.json'}; QPU seconds used: "
          f"{remaining - float(usage_after.get('usage_remaining_seconds', remaining)):.0f}")


if __name__ == "__main__":
    main()

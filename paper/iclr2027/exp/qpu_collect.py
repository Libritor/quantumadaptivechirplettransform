#!/usr/bin/env python
"""Companion to qpu_selection.py: submit the unitary-QFT job now (so both jobs queue in
parallel), wait for an already-submitted dynamic-circuit job by id, score everything
against the exact targets and write the final result file.

    python paper/iclr2027/exp/qpu_collect.py --dyn-job <job id> [--backend ibm_marrakesh]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from qpu_selection import (FS, N, OUT, counts_to_p, eeg_windows, make_case, score)  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qbe.qact_hw import two_qubit_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn-job", required=True)
    ap.add_argument("--uni-job", default="", help="reuse an already submitted unitary job")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--widths", default="1.5,2.7")
    ap.add_argument("--long-width", type=float, default=3.9)
    ap.add_argument("--rate-hz-s", type=float, default=8.0)
    ap.add_argument("--tag", default="marrakesh1")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
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
    # fake-backend prediction, as in qpu_selection.py
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fake = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=3, backend=fake, seed_transpiler=7)
    noisy = AerSimulator.from_backend(fake)
    for cs in cases:
        isa = pm.run(cs["qc"])
        cz, czd, d = two_qubit_stats(isa)
        cs.update(cz=cz, cz_depth=czd, depth=d)
        cnt = noisy.run(isa, shots=args.shots, seed_simulator=2).result().get_counts()
        cs["fake_noisy"] = score(counts_to_p(cnt, cs["qubits"]), cs)
    print("fake-backend predictions done", flush=True)

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    tok = os.environ["IBM_QUANTUM_TOKEN"]
    svc0 = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
    free = [dict(i) for i in svc0.instances() if str(dict(i).get("plan", "")).lower() == "open"][0]["crn"]
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=free)
    before = float(svc.usage().get("usage_remaining_seconds", 0))
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
    dyn = [cs for cs in cases if cs["variant"] == "dyn"]
    uni = [cs for cs in cases if cs["variant"] == "uni"]

    if args.uni_job:
        uni_job = svc.job(args.uni_job)
    else:
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = args.shots
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = 16
        sampler.options.twirling.shots_per_randomization = max(1, args.shots // 16)
        uni_job = sampler.run([cs["isa"] for cs in uni])
        print(f"submitted unitary job {uni_job.job_id()} ({len(uni)} circuits, DD + twirling)", flush=True)
    dyn_job = svc.job(args.dyn_job)

    def collect(job, sub, label):
        t0 = time.time()
        while True:
            st = str(job.status())
            if st in ("DONE", "ERROR", "CANCELLED") or "DONE" in st or "ERROR" in st or "CANCEL" in st:
                break
            print(f"  {label} {job.job_id()}: {st} ({time.time() - t0:.0f} s)", flush=True)
            time.sleep(60)
        res = job.result()
        for cs, pr in zip(sub, res):
            cnt = pr.data.k.get_counts()
            cs["device"] = score(counts_to_p(cnt, cs["qubits"]), cs)
            cs["device"].update(job_id=job.job_id(), counts={k: int(v) for k, v in cnt.items()},
                                shots=int(sum(cnt.values())))
        print(f"  {label} {job.job_id()} done", flush=True)

    collect(dyn_job, dyn, "dynamic")
    collect(uni_job, uni, "unitary")
    after = float(svc.usage().get("usage_remaining_seconds", -1))
    print("\nreal device vs exact target:")
    print(f"{'case':<28}{'q':>3}{'CZ':>6}{'fid sim':>9}{'fid dev':>9}{'sel.E sim':>11}{'sel.E dev':>11}{'peak':>6}")
    for cs in cases:
        print(f"S{cs['subject']:03d} {cs['channel']:>3} ld{cs['logdt']:<4} {cs['variant']:<4}"
              f"{cs['qubits']:>3}{cs['cz_device']:>6}{cs['fake_noisy']['fidelity']:>9.3f}"
              f"{cs['device']['fidelity']:>9.3f}{cs['fake_noisy']['selected_atom_energy']:>11.2f}"
              f"{cs['device']['selected_atom_energy']:>11.2f}{'ok' if cs['device']['peak_ok'] else 'LOST':>6}")
    rows = [{k: v for k, v in cs.items() if k not in ("qc", "isa", "want", "bins", "mask")} for cs in cases]
    for r in rows:
        r["want_peak_index"] = r.pop("peak")
    jobs = {"dyn": (dyn_job.job_id(), {"options": "default (DD refused for dynamic circuits)"}),
            "uni": (uni_job.job_id(), {"options": "DD XY4 + gate/measure twirling x16"})}
    json.dump(dict(backend=args.backend, calibration_last_update=props_date, fs=FS, N=N,
                   rate_hz_s=args.rate_hz_s, shots=args.shots, jobs=jobs, quota_before_s=before,
                   quota_after_s=after, rows=rows),
              open(OUT / f"qpu_selection_{args.tag}.json", "w"), indent=1)
    print(f"wrote {OUT / f'qpu_selection_{args.tag}.json'}; QPU seconds this session: {before - after:.0f}")


if __name__ == "__main__":
    main()

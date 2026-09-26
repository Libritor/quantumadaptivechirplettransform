"""Frozen submit / collect step for the preregistered fidelity-gap-law batch (PREREG_QPU_LAW.md).

    python paper/iclr2027/exp/qpu_law_submit.py --submit     # submit J1, J2, J3; do not wait
    python paper/iclr2027/exp/qpu_law_submit.py --collect    # later: fetch results, run void checks

--submit
  * loads results/qpu_law_batch_circuits.qpy as is (never re-transpiles) and refuses unless its
    sha256 equals the preregistered one and the circuit names equal the design json's qpy_order;
  * selects the open-plan instance explicitly by its CRN (suffix 6223ffe6126d) and refuses unless
    backend.name == 'ibm_marrakesh', the remaining quota is >= 78 s + MARGIN_S, and no physical
    qubit used by a circuit is marked faulty in the current calibration (the frozen circuits are
    never re-transpiled here; a faulty qubit stops the run for a human decision);
  * builds J1, J2, J3 in the design json's randomised PUB order (the 16k-shot PUBs reuse their
    primary circuit, `circuit_in_qpy`) with exactly the preregistered job_options:
      J1, J3: DD XY4, gate + measure twirling, 16 randomizations x 250 / 1000 shots
      J2:     default options (4000 shots; no DD attempt first)
  * submits the three jobs inside one Batch (batch execution mode); if the open plan refuses a
    Batch, falls back to three job-mode submissions and records that;
  * writes results/qpu_law_jobs.json (job ids, backend, submission time, calibration date, quota,
    the preregistered QPU estimate and the runtime's own usage estimate) and exits.

--collect
  * fetches the three jobs, reads counts from pr.data.k, and writes results/qpu_law_device.json
    = {'pubs': [{id, counts, shots, job_id}], backend, calibration date, qpy sha, quota before
    and after, void checks};
  * the void checks (notes/qpu_law_prereg_addendum.md) are computed here and again, independently,
    by exp/qpu_law_evaluate.py before any scoring.

The IBM token is read from IBM_QUANTUM_TOKEN and is never printed, logged or written.
"""
import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "paper" / "iclr2027" / "results"
DESIGN = RES / "qpu_law_batch_design.json"
QPY = RES / "qpu_law_batch_circuits.qpy"
JOBS_OUT = RES / "qpu_law_jobs.json"
DEVICE_OUT = RES / "qpu_law_device.json"

QPY_SHA256 = "d05153cc51738e87bee5d0126d9b7e4551f1967233c46d86f8a434fb54663486"
BACKEND = "ibm_marrakesh"
OPEN_CRN_SUFFIX = "6223ffe6126d"
EST_QPU_S = 78.0          # PREREG_QPU_LAW.md section 5
MARGIN_S = 60.0           # same margin as qpu_selection.py (est + 60)
SHOT_FRACTION_MIN = 0.90  # void rule
JOB_ORDER = ("J1_uni_twirl_4k", "J2_dyn_default_4k", "J3_uni_twirl_16k")
PRIMARY_JOB = "J1_uni_twirl_4k"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen():
    """Design json + qpy circuits, with the sha and name checks. Exits on any mismatch."""
    design = json.load(open(DESIGN))
    raw = QPY.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != QPY_SHA256 or design["qpy_sha256"] != QPY_SHA256:
        sys.exit(f"qpy sha256 mismatch: file {sha}, expected {QPY_SHA256}; refusing")
    from qiskit import qpy
    circs = qpy.load(io.BytesIO(raw))
    names = [c.name for c in circs]
    if names != design["qpy_order"]:
        sys.exit("qpy circuit names differ from design qpy_order; refusing")
    for c in circs:
        if [cr.name for cr in c.cregs] != ["k"]:
            sys.exit(f"circuit {c.name} does not measure into a single register 'k'; refusing")
    return design, {c.name: c for c in circs}, sha


def token():
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    if not tok:
        sys.exit("IBM_QUANTUM_TOKEN not set")
    return tok


def open_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = token()
    svc0 = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
    crns = [str(dict(i).get("crn", "")) for i in svc0.instances()]
    match = [c for c in crns if c.rstrip(":").endswith(OPEN_CRN_SUFFIX)]
    if len(match) != 1:
        sys.exit(f"open-plan instance (CRN suffix {OPEN_CRN_SUFFIX}) not found exactly once")
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=match[0])
    return svc, match[0]


def usage_summary(svc):
    u = dict(svc.usage())
    return dict(remaining_s=float(u.get("usage_remaining_seconds", -1)),
                consumed_s=float(u.get("usage_consumed_seconds", -1)),
                limit_s=float(u.get("usage_limit_seconds", -1)),
                limit_reached=bool(u.get("usage_limit_reached", False)),
                period=u.get("usage_period"))


def active_qubits(circ):
    used = set()
    for inst in circ.data:
        for q in inst.qubits:
            used.add(circ.find_bit(q).index)
    # control-flow blocks act on qubits already listed as operands of the if_else instruction
    return used


def build_pubs(design, circs):
    """{job: [(pub_id, circuit, shots)]} in the design's randomised order."""
    pubs = {p["id"]: p for p in design["pubs"]}
    out = {}
    for job in JOB_ORDER:
        rows = []
        for pid in design["jobs"][job]:
            P = pubs[pid]
            rows.append((pid, circs[P["circuit_in_qpy"]], int(P["shots"])))
        shots = {r[2] for r in rows}
        if shots != {int(design["job_options"][job]["default_shots"])}:
            sys.exit(f"{job}: PUB shots {shots} differ from the preregistered default_shots")
        out[job] = rows
    n = sum(len(v) for v in out.values())
    if n != len(design["pubs"]):
        sys.exit(f"built {n} PUBs, design has {len(design['pubs'])}")
    return out


def make_sampler(mode, job, opts):
    """Exactly the preregistered job_options."""
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=mode)
    s.options.default_shots = int(opts["default_shots"])
    if job in ("J1_uni_twirl_4k", "J3_uni_twirl_16k"):
        assert opts["dynamical_decoupling"] == "XY4"
        s.options.dynamical_decoupling.enable = True
        s.options.dynamical_decoupling.sequence_type = "XY4"
        s.options.twirling.enable_gates = bool(opts["twirling_gates"])
        s.options.twirling.enable_measure = bool(opts["twirling_measure"])
        s.options.twirling.num_randomizations = int(opts["num_randomizations"])
        s.options.twirling.shots_per_randomization = int(opts["shots_per_randomization"])
        assert opts["num_randomizations"] * opts["shots_per_randomization"] == opts["default_shots"]
    elif job == "J2_dyn_default_4k":
        pass  # default options: DD and twirling off
    else:
        raise ValueError(job)
    return s


def calibration_date(backend):
    try:
        return str(backend.properties().last_update_date)
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def submit(args):
    design, circs, sha = load_frozen()
    pubs_by_job = build_pubs(design, circs)
    total_shots = sum(s for rows in pubs_by_job.values() for _, _, s in rows)
    if total_shots != design["qpu_estimate"]["total_shots"]:
        sys.exit("total shots differ from the preregistered batch")
    svc, crn = open_service()
    quota_before = usage_summary(svc)
    print(f"open-plan instance ...{OPEN_CRN_SUFFIX}: {quota_before['remaining_s']:.0f} s remaining")
    if quota_before["limit_reached"] or quota_before["remaining_s"] < EST_QPU_S + MARGIN_S:
        sys.exit(f"remaining quota {quota_before['remaining_s']:.0f} s < {EST_QPU_S:.0f} + "
                 f"{MARGIN_S:.0f} s; refusing")
    backend = svc.backend(BACKEND)
    assert backend.name == BACKEND, backend.name
    cal = calibration_date(backend)
    faulty = []
    try:
        props = backend.properties()
        faulty = sorted(int(q) for q in props.faulty_qubits())
    except Exception:  # noqa: BLE001
        pass
    used = set()
    for c in circs.values():
        used |= active_qubits(c)
    bad = sorted(used & set(faulty))
    if bad:
        sys.exit(f"calibration marks used physical qubits {bad} faulty; not re-transpiling here, "
                 "refusing (human decision required, see PREREG section 4)")
    status = backend.status()
    print(f"backend {backend.name}: operational={status.operational} pending={status.pending_jobs} "
          f"calibration {cal}")
    if args.dry:
        print("dry: all checks passed; nothing submitted")
        return

    from qiskit_ibm_runtime import Batch
    record = dict(
        prereg="paper/iclr2027/PREREG_QPU_LAW.md",
        addendum="paper/iclr2027/notes/qpu_law_prereg_addendum.md",
        backend=backend.name, instance_crn_suffix=OPEN_CRN_SUFFIX,
        calibration_last_update=cal, faulty_qubits_at_submission=faulty,
        pending_jobs_at_submission=int(status.pending_jobs),
        qpy_file=str(QPY.relative_to(ROOT)).replace("\\", "/"), qpy_sha256=sha,
        design_sha256=sha256_file(DESIGN),
        submit_script_sha256=sha256_file(__file__),
        quota_before=quota_before,
        estimated_qpu_seconds=dict(prereg_estimate_s=EST_QPU_S,
                                   prereg_model_s=design["qpu_estimate"]["batch_model_s"],
                                   total_shots=total_shots),
        jobs={})
    mode_used = "batch"
    batch_id = None
    try:
        mode = Batch(backend=backend)
        batch_id = getattr(mode, "session_id", None)
    except Exception as e:  # noqa: BLE001
        print("Batch refused, falling back to job mode:", type(e).__name__, str(e)[:200])
        mode, mode_used = backend, "job"
    for job in JOB_ORDER:
        opts = design["job_options"][job]
        rows = pubs_by_job[job]
        sampler = make_sampler(mode, job, opts)
        try:
            j = sampler.run([c for _, c, _ in rows])
        except Exception as e:  # noqa: BLE001
            if mode_used == "batch" and job == JOB_ORDER[0]:
                print("Batch submission refused, falling back to job mode:", type(e).__name__,
                      str(e)[:200])
                mode, mode_used, batch_id = backend, "job", None
                sampler = make_sampler(mode, job, opts)
                j = sampler.run([c for _, c, _ in rows])
            else:
                raise
        t_sub = datetime.now(timezone.utc).isoformat(timespec="seconds")
        est = None
        try:
            ue = j.usage_estimation
            est = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(ue).items()}
        except Exception as e:  # noqa: BLE001
            est = f"unavailable ({type(e).__name__})"
        record["jobs"][job] = dict(job_id=j.job_id(), submitted_utc=t_sub, n_pubs=len(rows),
                                   pub_ids=[pid for pid, _, _ in rows],
                                   shots_per_pub=int(opts["default_shots"]),
                                   options=opts, runtime_usage_estimation=est)
        print(f"submitted {job}: {j.job_id()} ({len(rows)} PUBs)", flush=True)
    if batch_id is None and mode_used == "batch":
        batch_id = getattr(mode, "session_id", None)
    if mode_used == "batch":
        try:
            mode.close()  # no new jobs; queued jobs still run
        except Exception:  # noqa: BLE001
            pass
    record["execution_mode"] = mode_used
    record["batch_id"] = batch_id
    record["submitted_utc"] = record["jobs"][JOB_ORDER[0]]["submitted_utc"]
    json.dump(record, open(JOBS_OUT, "w"), indent=1)
    print(f"wrote {JOBS_OUT}")


def collect(args):
    design, circs, sha = load_frozen()
    rec = json.load(open(JOBS_OUT))
    svc, crn = open_service()
    pubs_meta = {p["id"]: p for p in design["pubs"]}
    out_pubs, checks = [], dict(qpy_sha256_ok=(sha == QPY_SHA256 == rec["qpy_sha256"]), jobs={})
    backends = set()
    for job in JOB_ORDER:
        jr = rec["jobs"][job]
        j = svc.job(jr["job_id"])
        st = str(j.status())
        jc = dict(job_id=jr["job_id"], status=st, backend=None, short_pubs=[], error=None)
        try:
            jc["backend"] = j.backend().name
        except Exception as e:  # noqa: BLE001
            jc["backend"] = f"unavailable ({type(e).__name__})"
        backends.add(jc["backend"])
        if st not in ("DONE", "JobStatus.DONE"):
            jc["error"] = f"job not done: {st}"
            try:
                jc["error_message"] = str(j.error_message())[:500]
            except Exception:  # noqa: BLE001
                pass
            checks["jobs"][job] = jc
            if st in ("QUEUED", "RUNNING", "INITIALIZING", "VALIDATING") and not args.force:
                sys.exit(f"{job} is still {st}; run --collect later")
            continue
        res = j.result()
        if len(res) != len(jr["pub_ids"]):
            jc["error"] = f"{len(res)} results for {len(jr['pub_ids'])} PUBs"
        for pid, pr in zip(jr["pub_ids"], res):
            cnt = {k: int(v) for k, v in pr.data.k.get_counts().items()}
            shots = int(sum(cnt.values()))
            want = int(pubs_meta[pid]["shots"])
            if shots < SHOT_FRACTION_MIN * want:
                jc["short_pubs"].append(dict(id=pid, shots=shots, preregistered=want))
            out_pubs.append(dict(id=pid, counts=cnt, shots=shots, job_id=jr["job_id"], job=job))
        checks["jobs"][job] = jc
    got = {p["id"] for p in out_pubs}
    prim = [p["id"] for p in design["pubs"] if p["arm"] == "primary"]
    j1 = checks["jobs"].get(PRIMARY_JOB, {})
    reasons = []
    if not checks["qpy_sha256_ok"]:
        reasons.append("qpy sha256 mismatch")
    if j1.get("error"):
        reasons.append(f"J1: {j1['error']}")
    if j1.get("backend") != BACKEND:
        reasons.append(f"J1 backend {j1.get('backend')} != {BACKEND}")
    missing = [pid for pid in prim if pid not in got]
    if missing:
        reasons.append(f"{len(missing)} primary PUBs missing")
    short_prim = [s for s in j1.get("short_pubs", []) if s["id"] in prim]
    if short_prim:
        reasons.append(f"{len(short_prim)} primary PUBs under {SHOT_FRACTION_MIN:.0%} of shots")
    checks["verdict_void"] = bool(reasons)
    checks["void_reasons"] = reasons
    checks["secondary_void_jobs"] = [
        job for job in JOB_ORDER[1:]
        if checks["jobs"].get(job, {}).get("error") or checks["jobs"].get(job, {}).get("short_pubs")
        or checks["jobs"].get(job, {}).get("backend") != BACKEND]
    backend = svc.backend(BACKEND)
    out = dict(backend=sorted(b for b in backends if b)[0] if len(backends) == 1 else sorted(
                   str(b) for b in backends),
               calibration_last_update_at_submission=rec.get("calibration_last_update"),
               calibration_last_update_at_collection=calibration_date(backend),
               qpy_sha256=sha, execution_mode=rec.get("execution_mode"), batch_id=rec.get("batch_id"),
               quota_before=rec.get("quota_before"), quota_after=usage_summary(svc),
               jobs={k: v["job_id"] for k, v in rec["jobs"].items()},
               void_checks=checks, pubs=out_pubs)
    json.dump(out, open(DEVICE_OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pubs"}, indent=1))
    print(f"wrote {DEVICE_OUT}" + ("  [VERDICT VOID]" if checks["verdict_void"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--collect", action="store_true")
    ap.add_argument("--dry", action="store_true", help="--submit: run every check, submit nothing")
    ap.add_argument("--force", action="store_true", help="--collect: write even if a job is pending")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    submit(args) if args.submit else collect(args)


if __name__ == "__main__":
    main()

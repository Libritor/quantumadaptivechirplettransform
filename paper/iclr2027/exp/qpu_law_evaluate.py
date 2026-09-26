"""Guarded evaluation of the fidelity-gap-law batch: void checks first, then the frozen analysis.

    python paper/iclr2027/exp/qpu_law_evaluate.py results/qpu_law_device.json

qpu_law_design.py --evaluate scores whatever histograms it is given. This wrapper enforces the
void rule of PREREG_QPU_LAW.md section 4 as scoped by notes/qpu_law_prereg_addendum.md, from the
device file alone and independently of qpu_law_submit.py's own checks:

  verdict VOID (nothing scored) if any of
    * the qpy file on disk, the design json and the device file do not all carry the
      preregistered sha256;
    * the backend recorded for J1 (or for the file) is not ibm_marrakesh;
    * J1 failed or did not finish;
    * any of the 28 primary PUBs is missing or has fewer than 90% of its preregistered shots.
  A failed, short or wrong-backend J2 or J3 voids only the secondary analyses that use it: its
  PUBs are removed before scoring and listed under `secondary_void`.

Otherwise it calls qpu_law_design.evaluate unchanged. Output:
results/qpu_law_evaluation_<device file stem>.json.
"""
import hashlib
import json
import sys
import warnings
import logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qpu_law_design as D  # noqa: E402

QPY_SHA256 = "d05153cc51738e87bee5d0126d9b7e4551f1967233c46d86f8a434fb54663486"
BACKEND = "ibm_marrakesh"
SHOT_FRACTION_MIN = 0.90
PRIMARY_JOB = "J1_uni_twirl_4k"
SECONDARY_JOBS = ("J2_dyn_default_4k", "J3_uni_twirl_16k")


def void_checks(design, dev):
    reasons = []
    disk_sha = hashlib.sha256(D.OUT_QPY.read_bytes()).hexdigest()
    if not (disk_sha == QPY_SHA256 == design.get("qpy_sha256") == dev.get("qpy_sha256")):
        reasons.append("qpy sha256 check failed (disk / design / device file)")
    vc = dev.get("void_checks", {})
    jobs = vc.get("jobs", {})
    j1 = jobs.get(PRIMARY_JOB)
    if j1 is None:
        reasons.append("no J1 record in device file")
    else:
        if j1.get("error"):
            reasons.append(f"J1 failed: {j1['error']}")
        if j1.get("backend") != BACKEND:
            reasons.append(f"J1 backend {j1.get('backend')!r} != {BACKEND}")
    b = dev.get("backend")
    if b != BACKEND and not (isinstance(b, list) and BACKEND in b and j1 and j1.get("backend") == BACKEND):
        reasons.append(f"device file backend {b!r} != {BACKEND}")
    pubs = {p["id"]: p for p in design["pubs"]}
    got = {p["id"]: p for p in dev["pubs"]}
    for p in design["pubs"]:
        if p["arm"] != "primary":
            continue
        g = got.get(p["id"])
        if g is None:
            reasons.append(f"primary PUB {p['id']} missing")
            continue
        n = int(sum(g["counts"].values()))
        if n != int(g.get("shots", n)):
            reasons.append(f"primary PUB {p['id']}: shots field {g.get('shots')} != counts sum {n}")
        if n < SHOT_FRACTION_MIN * p["shots"]:
            reasons.append(f"primary PUB {p['id']}: {n} shots < 90% of {p['shots']}")
        if g.get("job") not in (None, PRIMARY_JOB):
            reasons.append(f"primary PUB {p['id']} came from {g.get('job')}, not J1")
    # secondary jobs
    sec_void = {}
    for job in SECONDARY_JOBS:
        jr = jobs.get(job)
        why = []
        if jr is None:
            why.append("no job record")
        else:
            if jr.get("error"):
                why.append(jr["error"])
            if jr.get("backend") != BACKEND:
                why.append(f"backend {jr.get('backend')!r}")
        ids = design["jobs"][job]
        for pid in ids:
            g = got.get(pid)
            if g is None:
                why.append(f"{pid} missing")
            elif sum(g["counts"].values()) < SHOT_FRACTION_MIN * pubs[pid]["shots"]:
                why.append(f"{pid} short")
        if why:
            sec_void[job] = dict(reasons=why, removed_pubs=ids)
    return reasons, sec_void


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    path = Path(sys.argv[1])
    design = json.load(open(D.OUT_JSON))
    dev = json.load(open(path))
    reasons, sec_void = void_checks(design, dev)
    out_path = D.RES / f"qpu_law_evaluation_{path.stem}.json"
    if reasons:
        out = dict(label=str(path), verdict="VOID", void_reasons=reasons, secondary_void=sec_void,
                   note="void checks failed; nothing scored (PREREG section 4, addendum)")
    else:
        drop = {pid for v in sec_void.values() for pid in v["removed_pubs"]}
        device = {p["id"]: p for p in dev["pubs"] if p["id"] not in drop}
        out = D.evaluate(design, device, label=str(path))
        out["void_checks"] = dict(passed=True, secondary_void=sec_void)
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_pub"}, indent=1, default=float))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

# Addendum to PREREG_QPU_LAW.md (before submission)

Written on 2026-09-26, before any device data for the batch exist. `PREREG_QPU_LAW.md` is not edited. This note narrows two points and names the frozen submit and evaluation scripts.

## 1. Scope of the void rule

Section 4 of the prereg says the batch is void if "a job fails or returns fewer than 90% of its shots". That wording would let a failure in a secondary job void the primary test. The rule is scoped as follows.

- **The verdict (R1-R4) is void, and nothing is claimed, only if** one of these holds:
  - J1 (`J1_uni_twirl_4k`, which holds all 28 primary PUBs) fails or does not finish;
  - any primary PUB is missing or returns fewer than 90% of its preregistered 4000 shots;
  - J1 ran on a backend other than `ibm_marrakesh`;
  - the qpy sha256 check fails (the file on disk, the design json and the device results must all carry `d05153cc51738e87bee5d0126d9b7e4551f1967233c46d86f8a434fb54663486`).
- **A failed, short or wrong-backend J2 or J3 voids only the secondary analyses that use it.** J2 carries the dynamic twins (secondary (d)). J3 carries the 16k-shot re-runs (secondary (e)). Their PUBs are removed before scoring and listed as `secondary_void`. The primary verdict is unaffected.
- The faulty-qubit clause of section 4 stays as written, with one change: the submit script does not re-transpile. If the calibration at submission marks a used physical qubit as faulty, the script refuses to submit, and re-transpiling is left to a separate, reported decision.

## 2. Anchor predictions are in-sample

The two `anchor_sep25` circuits re-run Sep 25 circuits (`1|Cz|1.5|uni`, `2|Cz|2.7|uni`). Their predicted F comes from the `cz_fit_variant` fit, and that fit was made on all 26 Sep 25 rows, including those two. So the anchors' F predictions are in-sample. The anchors measure day-to-day drift (secondary (g)). They are not out-of-sample tests of the F predictor. All 28 primary circuits are new EEG windows (subjects 4-20) and none of them was in the fit.

## 3. Mechanical enforcement

- `exp/qpu_law_submit.py --submit` loads `results/qpu_law_batch_circuits.qpy` as is and never re-transpiles. It refuses to run unless all of these hold:
  - the qpy sha256 matches;
  - the circuit names equal `qpy_order`;
  - the open-plan instance (CRN suffix `6223ffe6126d`) is selected explicitly;
  - `backend.name == 'ibm_marrakesh'`;
  - the remaining quota is at least 78 s plus a 60 s margin.

  It then submits J1, J2 and J3 in the json's randomised PUB order, with exactly the preregistered `job_options`, and writes `results/qpu_law_jobs.json`.
- `exp/qpu_law_submit.py --collect` reads counts from `pr.data.k` and writes `results/qpu_law_device.json`. That file holds `{pubs: [{id, counts, shots, job_id}]}` plus the backend, the calibration dates, the qpy sha, the quota before and after, and the void checks above.
- `exp/qpu_law_evaluate.py <device json>` repeats the void checks from the device file alone. When they fail, it writes `verdict: "VOID"` and scores nothing. When they pass, it removes any voided secondary job's PUBs and then calls the unchanged `qpu_law_design.evaluate`. The pre-existing `qpu_law_design.py --evaluate` performs none of these checks and is not the scoring entry point for this batch.

  Before any submission, the guard was exercised on law-sampled mock histograms (never committed) with four cases:
  - a short primary PUB: VOID;
  - a wrong backend: VOID;
  - a failed J2: J2 was removed and the primary was scored;
  - all checks passing: scored.

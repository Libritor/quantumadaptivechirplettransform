#!/usr/bin/env python
"""The two classical controls that decide the one unexplained denoising number.

QACT parity hw (3-point coordinate refiner) scored 1.213 on EEGMAT denoising and
beat the frequency-matched classical engine (classical ACT fine-f, 1.257;
p = 0.0007). Two differences between them were never controlled:
  refiner    QACT's coordinate search vs the classical engine's 60 Adam steps
  seed band  the classical fine grid seeds 0.5-45 Hz, so it cannot seed a 50 Hz
             mains atom; QACT's exact-f update reads the periodogram peak anywhere

PRE-REGISTERED (written before running):
  arms      classical ACT fine-f coord  0.5 Hz grid, refinement = 3-point
                                        coordinate search, 4 sweeps
            classical ACT fine-f to78   0.5 Hz grid extended to 78 Hz (1 Hz steps
                                        above 45), Adam as shipped
  protocol  compare_denoise.py unchanged: thresholds tuned on subjects 0-7 and
            frozen, scored on the 18 held-out subjects 18-35, 180 s each, seed 7,
            order 12. The references (classical ACT, classical ACT fine-f, QACT
            parity hw) are rerun in the same process, and must reproduce the
            saved results/denoise_hw.json scores.
  tests     paired Wilcoxon across the 18 held-out subjects, alpha = 0.05/4:
              C1 fine-f coord - QACT parity hw     C2 fine-f to78 - QACT parity hw
              C3 fine-f coord - fine-f             C4 fine-f to78 - fine-f
  reading   if C1 or C2 is not significantly worse than QACT parity hw, that
            control explains the 1.213 vs 1.257 gap; if both stay significantly
            worse, the gap is still unexplained.
"""
import json
import subprocess
import sys

import numpy as np
from scipy.stats import wilcoxon

ARMS = ["classical ACT fine-f", "classical ACT fine-f coord", "classical ACT fine-f to78",
        "QACT parity hw"]
RUN = "results/denoise_controls_run.json"
subprocess.run([sys.executable, "compare_denoise.py", "--only", *ARMS, "--out", RUN],
               check=True)

run = json.load(open(RUN))["results"]
saved = json.load(open("results/denoise_hw.json"))["results"]
subs = sorted(run["classical ACT"]["per_subject"])
S = {k: np.array([run[k]["per_subject"][s] for s in subs]) for k in run}

repro = {}
for k in ("classical ACT", "classical ACT fine-f", "QACT parity hw"):
    old = np.array([saved[k]["per_subject"][s] for s in subs])
    repro[k] = dict(saved_mean=float(old.mean()), rerun_mean=float(S[k].mean()),
                    max_abs_diff=float(np.abs(old - S[k]).max()))
print("\nreproduction of saved scores:")
for k, v in repro.items():
    print(f"  {k:<22} saved {v['saved_mean']:.4f}  rerun {v['rerun_mean']:.4f}  "
          f"max |diff| {v['max_abs_diff']:.2e}")

ALPHA = 0.05 / 4
tests = {}
print(f"\npaired Wilcoxon, 18 held-out subjects, alpha = {ALPHA:.4f} "
      f"(negative = first arm cleans better):")
for tag, a_, b_ in (("C1", "classical ACT fine-f coord", "QACT parity hw"),
                    ("C2", "classical ACT fine-f to78", "QACT parity hw"),
                    ("C3", "classical ACT fine-f coord", "classical ACT fine-f"),
                    ("C4", "classical ACT fine-f to78", "classical ACT fine-f")):
    d = S[a_] - S[b_]
    p = float(wilcoxon(S[a_], S[b_]).pvalue) if not np.allclose(d, 0) else 1.0
    verdict = ("no difference" if p >= ALPHA else
               "first BETTER" if d.mean() < 0 else "first WORSE")
    tests[tag] = dict(a=a_, b=b_, mean_diff=float(d.mean()), a_better=int((d < 0).sum()),
                      p=p, verdict=verdict)
    print(f"  {tag} {a_:<27} - {b_:<21} {d.mean():+.4f}  better {(d < 0).sum():>2}/18  "
          f"p={p:.4f}  {verdict}")

out = dict(
    protocol="compare_denoise.py: EEGMAT, thresholds tuned on subjects 0-7 and frozen, "
             "18 held-out subjects 18-35, 180 s each, artifact seed 7, order 12",
    scores={k: dict(mean=float(S[k].mean()), median=float(np.median(S[k])),
                    config=run[k]["config"],
                    per_subject={s: float(v) for s, v in zip(subs, S[k])}) for k in S},
    tests=tests, alpha=ALPHA, reproduction=repro)
json.dump(out, open("results/denoise_controls_iclr.json", "w"), indent=2)
print("\nwrote results/denoise_controls_iclr.json")

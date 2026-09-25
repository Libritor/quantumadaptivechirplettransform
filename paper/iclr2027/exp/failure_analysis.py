#!/usr/bin/env python
"""Why does the classical engine leave >90% of the energy on some windows where QACT
parity does not? Inspect those windows: spectral content, the atoms QACT finds, and
whether widening the classical engine's width bound or its seed grid removes the failure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbe import act_gpu  # noqa: E402
from qbe.gpu_features import gpu_dictionary_grid  # noqa: E402
from qbe.qact import QACT  # noqa: E402
from fair_control_refine import FS, N, OUT, eeg_windows  # noqa: E402

dev = "cuda"
X = eeg_windows(20)
d = json.load(open(OUT / "fair_control_refine.json"))
ref = np.array(d["engines"]["classical Adam-60"]["12"]["err"])
q = np.array(d["engines"]["QACT parity (hw-4, exact-f, backfit 1)"]["12"]["err"])
fail = np.flatnonzero(ref > 0.9)
print(f"{len(fail)} failing windows (ref err > 0.9 at order 12); QACT parity err on them: "
      f"{q[fail].mean():.3f}; ref err on the others {ref[ref <= 0.9].mean():.3f}")
spec = np.abs(np.fft.rfft(X, axis=1)) ** 2
fr = np.fft.rfftfreq(N, 1 / FS)
for lo, hi in ((0, 1), (1, 4), (4, 13), (13, 45), (45, 80)):
    band = (fr >= lo) & (fr < hi)
    print(f"  energy in [{lo},{hi}) Hz: failing {100 * (spec[fail][:, band].sum(1) / spec[fail].sum(1)).mean():.1f}% "
          f"vs others {100 * (spec[~np.isin(np.arange(len(X)), fail)][:, band].sum(1) / spec[~np.isin(np.arange(len(X)), fail)].sum(1)).mean():.1f}%")
# QACT atoms on the failing windows
qa = QACT(length=N, fs=FS, device=dev, seed=0, refine_steps=4, omp=True, backfit_passes=1,
          exact_f=True, refine_mode="hw")
raw, err = qa.transform(X[fail].astype(np.float32), order=12, select="argmax", return_raw=True)
print("QACT first atoms on failing windows (tc s, width s, fc Hz, rate Hz/s):")
for i in range(min(5, len(fail))):
    tc, ld, f, c = raw[i, 0, 0], raw[i, 0, 1], raw[i, 0, 2], raw[i, 0, 3]
    fc = (f + 2 * c * tc) * FS / N
    print(f"  window {fail[i]}: tc {tc / FS:.2f} s, width {np.exp(ld) / FS:.2f} s, fc {fc:.2f} Hz, "
          f"rate {c * 2 * FS ** 2 / N:.2f} Hz/s; qact err {err[i]:.3f}")
# classical engine on the failing windows with wider width bound / more steps / lower fc
Xt = torch.as_tensor(X[fail], dtype=torch.float32, device=dev)
D = act_gpu.GPUDictionary(gpu_dictionary_grid(N, FS), N, FS, device=dev)
for label, kw in (("as shipped (dt_max 1.5 s)", dict(dt_max=1.5)),
                  ("dt_max 3.2 s", dict(dt_max=3.2)),
                  ("dt_max 3.2 s, c_max 60", dict(dt_max=3.2, c_max=60.0))):
    out, R = act_gpu.decompose_batch(Xt, D, max_atoms=12, steps=60, omp=True, **kw)
    e = (R.norm(dim=1) / Xt.norm(dim=1)).cpu().numpy()
    p0 = out[0]["p"].cpu().numpy()
    print(f"classical {label}: err on failing windows {e.mean():.3f}; first-atom widths "
          f"{np.round(p0[:5, 2], 2)} s, fc {np.round(p0[:5, 1], 2)} Hz")
json.dump(dict(failing=fail.tolist()), open(OUT / "failure_analysis.json", "w"))

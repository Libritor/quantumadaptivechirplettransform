#!/usr/bin/env python
"""Does Hybrid ACT at level q8 really beat classical ACT? A powered test.

On 4 windows (hybrid_act_levels.py, qact_qec.py), q8 under Heron noise and q8
error-corrected had slightly lower reconstruction error than classical q0 on 3/4
windows each, while NOISELESS q8 was worse on 3/4. Four windows cannot decide
anything (the smallest possible paired p-value is 0.125), and the pattern --
noisy better, noiseless worse -- suggests noise acting as randomised selection,
which a classical algorithm can do too.

PRE-REGISTERED (written before running):
  data     60 real EEGMAT windows (30 subjects x 2, fresh random draw), 6 atoms
  metric   per-window reconstruction error (lower is better), paired
  arms     q0              classical, every score exact (reference)
           q8 heron        quantum routes under IBM Heron noise (FakeTorino)
           q8 qec          quantum routes at the cheapest surface-code budget
                           (0.3), logical-level noise, as in qact_qec.py
           q8 noiseless    quantum routes sampled from their exact distributions
           mix 0.3         CLASSICAL control, no circuits: exact distributions
                           mixed 30% with uniform noise, same shots (matches the
                           code budget: ~30% of shots fail)
           mix 0.6         CLASSICAL control at a heavier mix (bare-Heron-like)
  primary  paired Wilcoxon, two-sided, each arm vs q0, for q8 heron, q8 qec,
           mix 0.3, mix 0.6; Bonferroni alpha = 0.05 / 4 = 0.0125
  rule     q8 "beats classical" only if (a) q8 heron or q8 qec is significantly
           better than q0 AND (b) it is also significantly better than its
           matched classical control (q8 qec vs mix 0.3; q8 heron vs mix 0.6).
           If a classical mix gets the same improvement, the benefit is
           randomised selection, not the quantum computer.
  speed    reported alongside: every arm's time per window.
"""
import json
import time

import numpy as np
from scipy.stats import wilcoxon

from qact_qec import BASIS, logical_noise
from qbe.acquire import read_eegmat_file
from qbe.hybrid_act import HybridACT

ROOT = "/home/kc/EEG-Memristor-ACT/data/eegmat"
N, ORDER, ALPHA = 512, 6, 0.05 / 4


def windows(n_sub=30, per_sub=2, seed=23):
    rng = np.random.default_rng(seed)
    out = []
    for sub in rng.choice(36, size=n_sub, replace=False):
        x = read_eegmat_file(f"{ROOT}/Subject{sub:02d}_1.edf", 0, target_fs=256.0).data
        for _ in range(per_sub):
            ch = rng.integers(x.shape[0])
            st = rng.integers(0, x.shape[1] - N)
            w = x[ch, st:st + N].astype(float)
            out.append(w - w.mean())
    return out


def main():
    X = windows()
    qec = json.load(open("results/qact_qec.json"))
    g9 = qec["ops_per_shot"]["9"]
    arms = {
        "q0": dict(q_max=0, noise="none"),
        "mix 0.3": dict(q_max=8, noise="classical_mix", mix=0.3),
        "mix 0.6": dict(q_max=8, noise="classical_mix", mix=0.6),
        "q8 noiseless": dict(q_max=8, noise="none"),
        "q8 qec": dict(q_max=8, noise="logical", basis_gates=BASIS,
                       noise_model=logical_noise(0.3 / g9)),
        "q8 heron": dict(q_max=8, noise="heron"),
    }
    print(f"{len(X)} windows x {ORDER} atoms; arms: {list(arms)}\n", flush=True)
    res = {}
    for name, kw in arms.items():
        h = HybridACT(**kw)
        t0 = time.perf_counter()
        errs = [h.decompose(x, order=ORDER)[0] for x in X]
        wall = (time.perf_counter() - t0) / len(X)
        res[name] = dict(err=[float(e) for e in errs], wall_s=wall,
                         shots=sum(s for _, s in h.qpu_log) / len(X),
                         shots_by_register={m: sum(s for mm, s in h.qpu_log if mm == m) / len(X)
                                            for m in (5, 6, 7, 8, 9)},
                         sel_quality=float(np.mean(h.sel_quality)))
        json.dump(res, open("results/q8_vs_classical.partial.json", "w"), indent=1)
        print(f"[{name:12s}] mean recon err {np.mean(errs):.4f}  atom quality "
              f"{res[name]['sel_quality']:.3f}  ({wall:.1f}s per window)", flush=True)

    ref = np.array(res["q0"]["err"])
    print(f"\npaired Wilcoxon vs q0 across {len(X)} windows (alpha = {ALPHA:.4f}); "
          f"negative = lower error than classical:")
    tests = {}
    for name in ("q8 heron", "q8 qec", "mix 0.3", "mix 0.6", "q8 noiseless"):
        a = np.array(res[name]["err"])
        d = a - ref
        p = float(wilcoxon(a, ref).pvalue) if np.any(d != 0) else 1.0
        primary = name != "q8 noiseless"
        tests[name] = dict(mean_diff=float(d.mean()), better=int((d < 0).sum()), p=p)
        verdict = ("BETTER *" if d.mean() < 0 and p < ALPHA else "WORSE *" if p < ALPHA
                   else "no difference") if primary else "(descriptive)"
        print(f"  {name:12s} {d.mean():+.4f}  better on {(d < 0).sum():2d}/{len(d)}  "
              f"p={p:.4f}  {verdict}")
    print("\nmatched controls (the decision rule, part b):")
    for q, c in (("q8 qec", "mix 0.3"), ("q8 heron", "mix 0.6")):
        a, b = np.array(res[q]["err"]), np.array(res[c]["err"])
        d = a - b
        p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
        tests[f"{q} vs {c}"] = dict(mean_diff=float(d.mean()), better=int((d < 0).sum()), p=p)
        print(f"  {q:9s} - {c:8s} {d.mean():+.4f}  better on {(d < 0).sum():2d}/{len(d)}  "
              f"p={p:.4f}")
    beats = any(tests[q]["mean_diff"] < 0 and tests[q]["p"] < ALPHA
                and tests[f"{q} vs {c}"]["mean_diff"] < 0 and tests[f"{q} vs {c}"]["p"] < ALPHA
                for q, c in (("q8 qec", "mix 0.3"), ("q8 heron", "mix 0.6")))
    print(f"\nverdict under the pre-registered rule: "
          f"{'q8 BEATS classical' if beats else 'q8 does NOT beat classical'}")
    json.dump(dict(results=res, tests=tests, verdict_q8_beats_classical=beats,
                   windows=len(X), order=ORDER, alpha=ALPHA),
              open("results/q8_vs_classical.json", "w"), indent=1)
    print("wrote results/q8_vs_classical.json")


if __name__ == "__main__":
    main()

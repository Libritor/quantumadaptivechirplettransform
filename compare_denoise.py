#!/usr/bin/env python
"""Denoising comparison: classical ACT vs QACT variants, against clean ground truth.

PROTOCOL (pre-registered; mirrors the sibling project's, which is why its
injection and metrics are reused rather than reimplemented)

  data      EEGMAT, already ICA-cleaned by its authors -> usable as ground truth.
            19 monopolar 10-20 channels, resampled to 256 Hz, first `--seconds`
            of run 1 per subject.
  contaminate  artifacts.inject: blinks, horizontal eye movement, EMG, the
            subject's OWN recorded ECG, electrode pops, mains, drift.
  clean     decompose 2-s frames (50% hop, periodic Hann), classify atoms as
            artifact by the sibling project's rule form, subtract them by
            overlap-add, leaving the rest untouched.
  score     metrics.cleaning_metrics -> relative RMSE overall and on
            artifact-dominated samples, per-band power error in dB;
            metrics.tuning_score = time fidelity + spectral fidelity (lower is
            better). The spectral term is what stops a cleaner from deleting
            genuine slow EEG to buy time-domain error.
  split     thresholds tuned on the first `--n-tune` subjects, FROZEN, then
            evaluated on the held-out subjects (subjects 18-35).

  DECISION: each QACT variant vs classical ACT on held-out tuning_score, paired
  Wilcoxon across held-out subjects; alpha = 0.05/(number of QACT arms). Lower is better, so
  an improvement means a NEGATIVE difference.
"""
import argparse, itertools, json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/kc/EEG-Memristor-ACT/code")   # validated injection + metrics
import artifacts as ART
import metrics as MET

from qbe.acquire import EEGMAT_CHANNELS, read_eegmat_file
from qbe.denoise import (GRID, HOP, WIN, artifact_mask, decompose_classical,
                         decompose_qact, frame, overlap_add)
from qbe.edf import read_edf

FS = 256.0
# The PRIMARY comparison is "QACT parity" vs "classical ACT": the two engines now
# perform the same core functions (OMP joint refit, off-grid refinement of every
# parameter, two-width envelope, per-window stopping rule, normalised selection),
# so the difference isolates the quantum formulation rather than engine
# bookkeeping. "QACT legacy" is retained to quantify how much of the previously
# reported gap was bookkeeping. The cubic and skew arms are secondary: they are
# atom families the classical engine does not have.
ENGINES = {
    "classical ACT": dict(kind="classical"),
    # the classical engine's own two-width envelope, so the asymmetric QACT arm
    # is not being compared against a family the classical side was denied
    "classical ACT asym": dict(kind="classical", asym=True),
    # decisive control: classical engine at QACT's frequency resolution
    "classical ACT fine-f": dict(kind="classical", fine_f=True),
    "classical ACT fine+asym": dict(kind="classical", fine_f=True, asym=True),
    # refinement control: the 0.5 Hz classical engine with QACT's hardware refiner
    # (3-point coordinate search, 4 sweeps) in place of 60 Adam steps
    "classical ACT fine-f coord": dict(kind="classical", fine_f=True, refine="coord"),
    # seed-band control: the 0.5 Hz classical grid extended to 78 Hz (1 Hz steps
    # above 45), so a 50 Hz mains atom can be seeded
    "classical ACT fine-f to78": dict(kind="classical", fine_f=True, fc_max=78.0),
    "QACT legacy": dict(kind="qact", omp=False, backfit_passes=0, exact_f=False,
                        norm_select=False),
    "QACT parity": dict(kind="qact"),
    # Control: EXACTLY the classical engine's function set -- OMP joint refit and
    # the post-selection-conditioned criterion, but neither of the two things
    # QACT has that act_gpu does not (the exact-f update and backfit). If this
    # arm also beats classical, those two extras are not what did it.
    "QACT matched-fn": dict(kind="qact", backfit_passes=0, exact_f=False),
    # hardware-economical refinement: 8 circuit evaluations per step instead of
    # 112, same step sizes and band semantics (qact_hardware.py, README)
    "QACT parity hw": dict(kind="qact", refine_mode="hw"),
    "QACT parity+asym": dict(kind="qact", asym_ratios=[1.0, 2.0, 4.0],
                             refine_extra=True),
    "QACT cubic": dict(kind="qact", rates3_hz_s2=[-40, -20, 0, 20, 40],
                       refine_extra=True),
    "QACT skew": dict(kind="qact", skews=[-4, -2, 0, 2, 4], refine_extra=True),
}
PRIMARY = "QACT parity"


def load_subject(root, sub, seconds):
    rec = read_eegmat_file(f"{root}/{sub}_1.edf", 0, target_fs=FS)
    eeg = rec.data[:, :int(seconds * FS)]
    sigs, rates, _ = read_edf(f"{root}/{sub}_1.edf")
    ecg = sigs.get("ECG ECG")
    if ecg is None:
        ecg = np.zeros(eeg.shape[1])
    else:                                  # 500 Hz -> 256 Hz, then trim
        from fractions import Fraction
        from scipy.signal import resample_poly
        r = Fraction(FS / float(rates["ECG ECG"])).limit_denominator(1000)
        ecg = resample_poly(ecg, r.numerator, r.denominator)[:eeg.shape[1]]
        if len(ecg) < eeg.shape[1]:
            ecg = np.pad(ecg, (0, eeg.shape[1] - len(ecg)))
    return eeg, ecg


def prepare(root, sub, seconds, seed, engine, order):
    clean, ecg = load_subject(root, sub, seconds)
    cont, parts = ART.inject(clean, ecg, FS, seed=seed)
    art = np.sum([parts[k] for k in parts], axis=0)
    frames, nf = frame(cont)
    ch = clean.shape[0]
    kw = {k: v for k, v in engine.items() if k != "kind"}
    if engine["kind"] == "classical":
        waves, phys = decompose_classical(frames, FS, order, **kw)
        dsize = None
    else:
        waves, phys, dsize = decompose_qact(frames, FS, order, **kw)
    sigma = 1.4826 * np.median(np.abs(cont - np.median(cont, axis=1, keepdims=True)),
                               axis=1)
    return dict(clean=clean, cont=cont, art=art, waves=waves, phys=phys, nf=nf,
                ch=ch, n=clean.shape[1], sigma=np.repeat(sigma, nf), dsize=dsize)


def score(P, R):
    mask = artifact_mask(P["phys"], P["sigma"], R)
    m = torch.as_tensor(mask.astype(np.float32), device=P["waves"].device)
    est_frames = (P["waves"] * m[..., None]).sum(1)
    est = overlap_add(est_frames, P["ch"], P["nf"], P["n"])
    cleaned = P["cont"] - est
    return MET.cleaning_metrics(P["clean"], cleaned, FS, P["art"]), float(mask.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kc/EEG-Memristor-ACT/data/eegmat")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--order", type=int, default=12)
    ap.add_argument("--n-tune", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these engines ('classical ACT' is always kept "
                         "as the paired reference)")
    ap.add_argument("--out", default="results/denoise_comparison.json")
    a = ap.parse_args()
    subs = [f"Subject{i:02d}" for i in range(36)]
    tune_subs, test_subs = subs[:a.n_tune], subs[18:]
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"[denoise] {len(cfgs)} rule configs; tune on {len(tune_subs)} subjects, "
          f"evaluate on {len(test_subs)} held out; {a.seconds:.0f}s per subject", flush=True)

    t0 = time.time()
    # ---- baselines that do not depend on the engine ----
    base = {}
    for sub in test_subs:
        clean, ecg = load_subject(a.root, sub, a.seconds)
        cont, parts = ART.inject(clean, ecg, FS, seed=a.seed)
        art = np.sum([parts[k] for k in parts], axis=0)
        base[sub] = MET.tuning_score(MET.cleaning_metrics(clean, cont, FS, art))
    print(f"[denoise] no cleaning (contaminated as-is): median score "
          f"{np.median(list(base.values())):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    results, chosen = {}, {}
    if a.only:
        keep = set(a.only) | {"classical ACT"}
        for k in [k for k in ENGINES if k not in keep]:
            del ENGINES[k]
    for ename, engine in ENGINES.items():
        t1 = time.time()
        # tune
        scores = np.zeros((len(tune_subs), len(cfgs)))
        for i, sub in enumerate(tune_subs):
            P = prepare(a.root, sub, a.seconds, a.seed, engine, a.order)
            for j, R in enumerate(cfgs):
                m, _ = score(P, R)
                scores[i, j] = MET.tuning_score(m)
            del P
            torch.cuda.empty_cache()
        best = cfgs[int(np.argmin(np.median(scores, axis=0)))]
        chosen[ename] = best
        print(f"[{ename}] tuned in {time.time()-t1:.0f}s -> {best}", flush=True)
        # evaluate frozen
        per_sub, detail = {}, {}
        for sub in test_subs:
            P = prepare(a.root, sub, a.seconds, a.seed, engine, a.order)
            m, frac = score(P, best)
            per_sub[sub] = MET.tuning_score(m)
            detail[sub] = dict(m, artifact_atom_frac=frac)
            del P
            torch.cuda.empty_cache()
        results[ename] = dict(per_subject=per_sub, detail=detail, config=best)
        arr = np.array(list(per_sub.values()))
        med = lambda k: np.median([detail[s][k] for s in test_subs])
        print(f"[{ename}] held-out score {arr.mean():.3f} (median {np.median(arr):.3f})"
              f"  rrmse {med('rrmse'):.3f}  alpha_dB {med('bp_err_db_alpha'):.2f}"
              f"  delta_dB {med('bp_err_db_delta'):.2f}"
              f"  atoms called artifact {med('artifact_atom_frac')*100:.0f}%"
              f"  ({time.time()-t1:.0f}s)", flush=True)

    from scipy.stats import wilcoxon
    ref = np.array([results["classical ACT"]["per_subject"][s] for s in test_subs])
    print(f"\nno cleaning: {np.mean([base[s] for s in test_subs]):.3f}   "
          f"(lower is better throughout)")
    print(f"\npaired Wilcoxon across {len(test_subs)} held-out subjects "
          f"(alpha = {0.05/(len(ENGINES)-1):.4f}); negative difference = cleans BETTER "
          f"than classical ACT:")
    for ename in ENGINES:
        if ename == "classical ACT":
            continue
        arr = np.array([results[ename]["per_subject"][s] for s in test_subs])
        d = arr - ref
        p = wilcoxon(arr, ref).pvalue if not np.allclose(d, 0) else 1.0
        sig = p < 0.05 / (len(ENGINES) - 1)
        verdict = ("BETTER *" if d.mean() < 0 and sig
                   else "better (ns)" if d.mean() < 0
                   else "WORSE *" if sig else "worse (ns)")
        print(f"  {ename:<20} - classical ACT  {d.mean():+.4f}  "
              f"better on {(d<0).sum():>2}/{len(d)}  p={p:.4f}  {verdict}")
    json.dump({"results": results, "no_cleaning": base, "chosen": chosen},
              open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

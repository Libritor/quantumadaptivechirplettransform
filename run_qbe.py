#!/usr/bin/env python
"""Quantum Brain Encoding -- end-to-end Muse EEG -> chirplets -> Qiskit.

Examples
--------
Run everything on synthetic data (no headband needed):
    python run_qbe.py --source synthetic

Analyse your own Mind Monitor CSV exports, arranged one folder per class:
    data/eyes_open/*.csv   data/eyes_closed/*.csv
    python run_qbe.py --source csv --data data

Record live from a running `muselsl stream`:
    python run_qbe.py --source lsl --seconds 60 --blocks 4

Inspect the chirplet decomposition of a single epoch:
    python run_qbe.py --source synthetic --explain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from qbe import acquire, features, preprocess, quantum
from qbe.types import EPOCH_SAMPLES


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Muse EEG -> Adaptive Chirplet Transform -> Qiskit classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("data source")
    src.add_argument("--source",
                     choices=("synthetic", "csv", "eegmat", "lsl", "brainflow"),
                     default="synthetic", help="where the EEG comes from")
    src.add_argument("--good-only", action="store_true",
                     help="EEGMAT: keep only the 26 subjects the dataset marks "
                          "as having counted well")
    src.add_argument("--max-per-recording", type=int, default=None,
                     help="cap epochs per recording (balances EEGMAT, whose "
                          "rest runs are ~3x longer than its arithmetic runs)")
    src.add_argument("--data", type=Path, default=Path("data"),
                     help="root folder for --source csv (one subfolder per class)")
    src.add_argument("--labels", nargs=2, default=("eyes_open", "eyes_closed"),
                     metavar=("CLASS0", "CLASS1"), help="the two class names")
    src.add_argument("--seconds", type=float, default=60.0,
                     help="seconds per recording block")
    src.add_argument("--blocks", type=int, default=6,
                     help="recording blocks per class (synthetic/live)")
    src.add_argument("--board", default="MUSE_S_BLED",
                     help="BrainFlow board name for --source brainflow")
    src.add_argument("--seed", type=int, default=0, help="random seed")

    pre = p.add_argument_group("preprocessing")
    pre.add_argument("--bandpass", nargs=2, type=float, default=None,
                     metavar=("LO", "HI"),
                     help="OPT-IN zero-phase band-pass. Off by default: it "
                          "measurably worsened class separation and smears the "
                          "chirp onsets ACT exists to resolve.")
    pre.add_argument("--notch", type=float, default=None,
                     help="OPT-IN mains notch in Hz (e.g. 60). Usually "
                          "unnecessary; the chirplet dictionary excludes it.")
    pre.add_argument("--no-artifact-gate", action="store_true",
                     help="keep epochs containing blinks / jaw clenches")

    act = p.add_argument_group("chirplet transform")
    act.add_argument("--order", type=int, default=12,
                     help="chirplet atoms per channel")
    act.add_argument("--engine", choices=("v7", "v9", "v14"), default="v7",
                     help="ACT engine; only v7 emits per-atom coefficients")
    act.add_argument("--jobs", type=int, default=1,
                     help="parallel worker processes for the transform")

    qc = p.add_argument_group("quantum classifier")
    qc.add_argument("--qubits", type=int, default=4,
                    help="qubits = PCA components kept")
    qc.add_argument("--reps", type=int, default=2, help="feature-map repetitions")
    qc.add_argument("--reduce", default="selectk", choices=list(quantum.REDUCERS),
                    help="how to get from 52 features down to --qubits. "
                         "selectk (ANOVA F-test) keeps the features that "
                         "separate the classes; pca keeps the highest-variance "
                         "directions, which on these features is measurably "
                         "worse (best component d=0.47 vs d=0.80).")
    qc.add_argument("--entanglement", default="linear",
                    choices=("linear", "full", "circular"))
    qc.add_argument("--models", nargs="+", default=list(quantum.CLASSIFIERS),
                    choices=list(quantum.CLASSIFIERS),
                    help="which classifiers to cross-validate")
    qc.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    qc.add_argument("--maxiter", type=int, default=60, help="VQC optimiser steps")

    out = p.add_argument_group("output")
    out.add_argument("--explain", action="store_true",
                     help="print the chirplet decomposition of one epoch and exit")
    out.add_argument("--save", type=Path, default=None,
                     help="write features, labels and results to this folder")
    return p.parse_args(argv)


def load_recordings(args) -> list:
    labels = tuple(args.labels)
    if args.source == "synthetic":
        print(f"[main] generating synthetic Muse data "
              f"({args.blocks} blocks/class x {args.seconds:.0f}s)")
        return acquire.synthetic_dataset(
            n_per_class=args.blocks, seconds=args.seconds, seed=args.seed,
            label_names=labels,
        )
    if args.source == "csv":
        return acquire.load_csv_directory(args.data, labels)
    if args.source == "eegmat":
        return acquire.load_eegmat(args.data, good_quality_only=args.good_only)
    if args.source in ("lsl", "brainflow"):
        recs = []
        for label, name in enumerate(labels):
            for b in range(args.blocks):
                input(f"\n>>> Block {b + 1}/{args.blocks} for '{name}'. "
                      f"Get ready, then press Enter to record "
                      f"{args.seconds:.0f}s ...")
                if args.source == "lsl":
                    rec = acquire.record_lsl(label, args.seconds,
                                             name=f"{name}-{b}")
                else:
                    rec = acquire.record_brainflow(label, args.seconds,
                                                   board=args.board,
                                                   name=f"{name}-{b}")
                recs.append(rec)
        return recs
    raise ValueError(args.source)


def explain_one(epochs, args) -> None:
    """Print the chirplet atoms of a single epoch, in physical units."""
    eng = features.build_engine(EPOCH_SAMPLES, args.engine)
    idx = 0
    ep = epochs.X[idx]
    print(f"\nChirplet decomposition of epoch {idx} "
          f"(class = {epochs.label_names[epochs.y[idx]]})")
    for c, ch in enumerate(epochs.channels):
        atoms, cert = eng.transform(ep[c], order=args.order)
        print(f"\n  {ch}: reconstruction error {cert.reconstruction_error:.3f}, "
              f"{cert.energy_explained_fraction * 100:.0f}% energy explained, "
              f"{cert.dictionary_size} atom dictionary")
        print(f"    {'f_Hz':>7} {'dur_s':>7} {'sweep_Hz':>9} "
              f"{'t_s':>6} {'energy%':>8}  kind")
        phys = [
            features.atom_to_physical(
                a, epochs.fs, EPOCH_SAMPLES,
                a.coeff_real ** 2 + a.coeff_imag ** 2)
            for a in atoms
        ]
        tot = sum(p.energy for p in phys) or 1.0
        for p in sorted(phys, key=lambda q: -q.energy):
            kind = "rhythm" if p.oscillatory else "transient/drift"
            print(f"    {p.f_center_hz:>7.2f} {p.duration_s:>7.3f} "
                  f"{p.sweep_hz:>9.2f} {p.t_center_s:>6.2f} "
                  f"{100 * p.energy / tot:>7.1f}%  {kind}")


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    labels = tuple(args.labels)

    recordings = load_recordings(args)
    if args.source == "eegmat":
        labels = acquire.EEGMAT_LABELS
    epochs = preprocess.epochs_from_recordings(
        recordings, labels,
        bandpass=tuple(args.bandpass) if args.bandpass else None,
        notch=args.notch,
        max_abs_z=None if args.no_artifact_gate else preprocess.DEFAULT_MAX_ABS_Z,
        # EEGMAT is in microvolts, so the ADC-rail test has no meaning there.
        check_rails=(args.source != "eegmat"),
        max_per_recording=(
            args.max_per_recording
            if args.max_per_recording is not None
            else (28 if args.source == "eegmat" else None)
        ),
    )
    print(f"[main] {epochs}")

    # Multi-subject corpora must be cross-validated by subject, or the model
    # can score by recognising the person instead of the mental state.
    groups = None
    if args.source == "eegmat":
        groups = np.array([acquire.subject_of(m["rec"]) for m in epochs.meta])

    if args.explain:
        explain_one(epochs, args)
        return 0

    if min(np.bincount(epochs.y)) < 2:
        print("[main] ERROR: need at least 2 epochs per class to cross-validate.",
              file=sys.stderr)
        return 1

    X, names = features.extract(
        epochs, order=args.order, engine=args.engine, n_jobs=args.jobs
    )

    if args.qubits > X.shape[1]:
        print(f"[main] --qubits {args.qubits} exceeds the {X.shape[1]} available "
              f"features; clamping.", file=sys.stderr)
        args.qubits = X.shape[1]
    if args.reduce == "pca":
        max_pca = min(X.shape[0], X.shape[1])
        if args.qubits > max_pca:
            print(f"[main] clamping --qubits to {max_pca} (PCA limit at "
                  f"{X.shape[0]} epochs)", file=sys.stderr)
            args.qubits = max_pca

    results = quantum.run_comparison(
        X, epochs.y, args.qubits, kinds=tuple(args.models), folds=args.folds,
        seed=args.seed, reps=args.reps, entanglement=args.entanglement,
        maxiter=args.maxiter, reduce=args.reduce, groups=groups,
    )

    best = max(results, key=lambda r: r.mean)
    print(f"\n[main] best: {best.name} at {best.mean * 100:.1f}%")
    q = [r for r in results if r.name in ("qsvc", "vqc")]
    cl = [r for r in results if r.name in ("svm_rbf", "logreg")]
    if q and cl:
        bq, bc = max(q, key=lambda r: r.mean), max(cl, key=lambda r: r.mean)
        delta = (bq.mean - bc.mean) * 100
        verdict = "beats" if delta > 0 else ("ties" if abs(delta) < 1e-9 else "loses to")
        print(f"[main] best quantum ({bq.name}) {verdict} best classical "
              f"({bc.name}) by {delta:+.1f} points")
    print(f"[main] total {time.time() - t0:.1f}s")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        np.save(args.save / "features.npy", X)
        np.save(args.save / "labels.npy", epochs.y)
        (args.save / "feature_names.json").write_text(json.dumps(names, indent=2))
        (args.save / "results.json").write_text(json.dumps(
            [{"model": r.name, "mean": r.mean, "std": r.std,
              "scores": r.scores.tolist(), "seconds": r.seconds, **r.extra}
             for r in results], indent=2))
        print(f"[main] saved to {args.save}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

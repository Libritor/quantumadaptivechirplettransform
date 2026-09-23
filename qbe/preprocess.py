"""Epoch gating and normalisation for raw Muse EEG.

WHAT THE HEADBAND ALREADY DOES, AND WHAT IT DOES NOT
----------------------------------------------------
The Muse streams two different kinds of data and only one of them is cleaned:

  * The *derived* streams -- per-band absolute/relative power (``Delta_TP9``,
    ``Alpha_AF7``, ...) and the headband-fit indicator -- are computed on-device
    with Muse's own filtering, FFT and artifact blanking. Those are cleaned.
  * The *raw* channels (``TP9_RAW``..``TP10_RAW``) are essentially ADC output.
    What is applied is analog, not algorithmic: an anti-alias band-limit ahead
    of the 12-bit ADC, and the DRL/reference circuit doing common-mode
    rejection (which suppresses much of the mains hum as a side effect).
    There is no digital notch, no artifact rejection and no packet-loss repair.

This pipeline uses the raw channels, because the Adaptive Chirplet Transform
needs an actual time-domain waveform. Muse's on-device band powers cannot be
chirplet-decomposed, and using them would discard the entire reason to choose
ACT: chirp rate, atom timing and non-stationarity.

WHY THERE IS NO FILTERING HERE BY DEFAULT
-----------------------------------------
Two reasons, the second measured rather than assumed:

  1. The chirplet dictionary *is* the band-pass. The ``fc`` grid spans 0.5-40 Hz
     at this epoch length and sample rate, so an atom outside that band cannot
     be constructed and out-of-band noise -- mains included -- lands in the
     residual instead of the features. Pre-filtering is largely redundant.

  2. Filtering measurably *hurt* class separation in this pipeline. On
     synthetic eyes-open/closed epochs, alpha-fraction separability was
     Cohen's d = +1.16 unfiltered, +0.50 with a 1 Hz zero-phase high-pass and
     +0.84 with a 2 Hz one. The high-pass produces *more* oscillatory atoms but
     *worse* discrimination, because it also lifts alpha-band content in the
     eyes-open condition (alpha fraction 0.000 -> 0.039), creating a
     false-positive floor. Separately, an IIR band-pass smears exactly the
     transient onsets ACT exists to resolve, biasing the chirp-rate estimate.

``bandpass``/``notch`` are therefore opt-in and off by default. They are kept
because a recording made next to bad mains, or one with heavy drift, may still
need them -- but turn them on deliberately, and re-measure separability.

The gate below is taken verbatim from the upstream ACT repository's validated
real-Muse protocol (``examples/muse_eeg_skew_replication.py``).
"""
from __future__ import annotations

import numpy as np

from .types import EPOCH_SAMPLES, EpochSet, Recording

# --- Upstream protocol constants (do not tune casually) ---------------------
# A 512-sample window off a healthy 256 Hz packet stream spans ~1.95 s. A lossy
# stream stretches, so anything outside this band had dropped packets.
SPAN_MS: tuple[float, float] = (1500.0, 2200.0)
# Mind Monitor raw exports sit in 0..1649 ADC units; samples pinned at either
# rail are saturated, not signal.
RAIL_LO, RAIL_HI = 0.0, 1649.0
# Reject the window if at least this fraction of samples are railed.
RAIL_FRAC = 0.01

# --- Additional artifact gate (not upstream; see `max_abs_z`) ---------------
# Blinks and jaw clenches are IN-BAND and enormous on AF7/AF8, which sit
# directly above the eyes: hundreds of uV against ~10-50 uV of real EEG.
# Upstream only cared about reconstruction error, where a blink is simply a
# large feature to fit. Here a blink would monopolise the atom budget and
# swamp the spectral features, so we drop windows containing one.
DEFAULT_MAX_ABS_Z = 8.0


def _optional_filter(
    x: np.ndarray,
    fs: float,
    bandpass: tuple[float, float] | None,
    notch: float | None,
) -> np.ndarray:
    """Apply opt-in zero-phase filtering. Off by default -- see module docstring."""
    if bandpass is None and notch is None:
        return x
    from scipy.signal import butter, filtfilt, iirnotch

    y = x
    if bandpass is not None:
        lo, hi = bandpass
        b, a = butter(2, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="bandpass")
        y = filtfilt(b, a, y)
    if notch is not None:
        bn, an = iirnotch(notch, 30.0, fs)
        y = filtfilt(bn, an, y)
    return y


def gate_window(
    w: np.ndarray,
    span_ms: float,
    *,
    check_rails: bool = True,
    max_abs_z: float | None = DEFAULT_MAX_ABS_Z,
) -> str | None:
    """Return a rejection reason for one raw window, or None if it passes.

    `w` is raw, pre-normalisation, in ADC units. `span_ms` is the measured
    elapsed time across the window.
    """
    if np.any(~np.isfinite(w)):
        return "nonfinite"
    if not (SPAN_MS[0] <= span_ms <= SPAN_MS[1]):
        return "packet_loss"
    if check_rails and float(np.mean((w <= RAIL_LO) | (w >= RAIL_HI))) >= RAIL_FRAC:
        return "adc_rail"
    # Upstream's near-constant check: a flat window is a disconnected electrode.
    if float(np.std(w)) < 1e-3 * (1.0 + abs(float(np.mean(w)))):
        return "near_constant"
    if max_abs_z is not None:
        sd = float(np.std(w))
        if sd > 0 and float(np.max(np.abs(w - np.mean(w)))) / sd > max_abs_z:
            return "artifact"
    return None


def normalise(w: np.ndarray) -> np.ndarray | None:
    """Zero-mean and unit-L2 normalise, per the upstream protocol.

    Returns None if the window has no energy. This normalisation is why
    absolute amplitude carries no information downstream and every spectral
    feature is expressed as a *fraction* of epoch energy.
    """
    w = w - float(np.mean(w))
    nrm = float(np.linalg.norm(w))
    if nrm < 1e-6:
        return None
    return w / nrm


def epochs_from_recordings(
    recordings: list[Recording],
    label_names: tuple[str, ...],
    *,
    n_samples: int = EPOCH_SAMPLES,
    require_all_channels: bool = True,
    bandpass: tuple[float, float] | None = None,
    notch: float | None = None,
    check_rails: bool = True,
    max_abs_z: float | None = DEFAULT_MAX_ABS_Z,
    max_per_recording: int | None = None,
    verbose: bool = True,
) -> EpochSet:
    """Cut non-overlapping gated epochs from raw recordings.

    An epoch is kept only if *every* channel passes the gate when
    `require_all_channels` is set, because the downstream feature vector
    concatenates all four channels and a single bad channel corrupts it.
    """
    if not recordings:
        raise ValueError("no recordings supplied")

    channels = recordings[0].channels
    fs = recordings[0].fs
    X: list[np.ndarray] = []
    y: list[int] = []
    meta: list[dict] = []
    rejects: dict[str, int] = {}

    for rec in recordings:
        if rec.channels != channels:
            raise ValueError(
                f"{rec.name}: channels {rec.channels} != {channels}"
            )
        if rec.label is None:
            raise ValueError(f"{rec.name}: unlabelled recording")
        kept_here = 0
        for start in range(0, rec.n_samples - n_samples + 1, n_samples):
            stop = start + n_samples
            span = float(rec.ms[stop - 1] - rec.ms[start])
            raw = rec.data[:, start:stop]

            reasons = [
                gate_window(
                    raw[c], span, check_rails=check_rails, max_abs_z=max_abs_z
                )
                for c in range(raw.shape[0])
            ]
            bad = [r for r in reasons if r is not None]
            if bad and (require_all_channels or len(bad) == len(reasons)):
                rejects[bad[0]] = rejects.get(bad[0], 0) + 1
                continue

            chans = []
            ok = True
            for c in range(raw.shape[0]):
                w = _optional_filter(raw[c], fs, bandpass, notch)
                nw = normalise(w)
                if nw is None:
                    ok = False
                    break
                chans.append(nw)
            if not ok:
                rejects["zero_energy"] = rejects.get("zero_energy", 0) + 1
                continue

            X.append(np.stack(chans))
            y.append(int(rec.label))
            meta.append({"rec": rec.name, "start": start, "span_ms": span})
            kept_here += 1
            if max_per_recording is not None and kept_here >= max_per_recording:
                break

    if not X:
        raise RuntimeError(
            f"every window was rejected: {rejects}. "
            "Check the recording length, sample rate and channel scaling."
        )

    es = EpochSet(
        X=np.stack(X), y=np.asarray(y), channels=channels, fs=fs,
        label_names=label_names, meta=meta,
    )
    if verbose:
        total = len(X) + sum(rejects.values())
        print(f"[preprocess] kept {len(X)}/{total} epochs  {es.counts()}")
        if rejects:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(rejects.items()))
            print(f"[preprocess] rejected: {detail}")
    return es

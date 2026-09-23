"""Adaptive Chirplet Transform features for Muse EEG.

WHY CHIRPLETS
-------------
A chirplet atom is a Gaussian-windowed linear-FM (chirp) waveform described by
four continuous parameters -- centre time `tc`, centre frequency `fc`,
log-duration `logDt`, and chirp rate `c`. A Gabor/wavelet atom is the special
case `c = 0`. That extra parameter is the whole point: real EEG transients
drift in frequency, and a stationary atom can only represent a drifting burst
by smearing it across several frequency bins. The chirp rate is information a
Welch spectrum structurally cannot report.

The decomposition is greedy: orthogonal matching pursuit picks the dictionary
atom best correlated with the residual, refines its four parameters off-grid
with an analytic-gradient L-BFGS-B step plus a Newton polish, re-fits all
coefficients, and repeats `order` times.

WHAT THE ATOMS ACTUALLY LOOK LIKE ON EEG
----------------------------------------
Measured, not assumed: on realistic 1/f EEG only a minority of atoms come back
oscillatory. Most collapse toward `fc -> 0`, `c -> 0` and a short duration --
slow Gaussian bumps. That is not a bug. Pink noise genuinely carries most of
its energy at low frequency, so greedy OMP correctly spends its first atoms
there, and the off-grid refinement is free to drive `fc` below the dictionary's
lowest grid line.

So this module splits the atoms rather than fighting them:

  * **Oscillatory** atoms -- at least `MIN_CYCLES` cycles within their own
    envelope, and inside the EEG band -- carry the rhythm features.
  * Everything else is pooled into a single `transient_frac` feature. Drift and
    residual blink energy are *informative*, not merely noise, so they get one
    feature instead of contaminating twelve.

FEATURE SCALE
-------------
Epochs arrive zero-mean and unit-L2 normalised, so absolute amplitude carries
no information. Every energy feature is therefore a *fraction* of total atom
energy, which is what makes features comparable across epochs and channels.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .types import EPOCH_SAMPLES, EpochSet

# The dictionary grid validated on real Muse recordings upstream. In
# sample-domain units at length 512: 11 tc x 9 fc x 4 logDt x 4 c = 1584 atoms.
# `fc` here is cycles-per-window, so 1..80 maps to 0.5..40 Hz at 256 Hz -- the
# dictionary is itself the band-pass, which is why no filter is applied first.
MUSE_ACT_GRID = dict(
    tc_info=(0, EPOCH_SAMPLES, 48),
    fc_info=(1.0, 80.0, 9.0),
    logDt_info=(1.5, 6.0, 1.2),
    c_info=(-12.0, 12.0, 7.0),
)

# Classic EEG bands, in Hz.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

EEG_BAND = (0.5, 40.0)
# An atom must fit at least this many cycles inside its own envelope to count
# as a rhythm rather than a spike.
MIN_CYCLES = 0.5
# The off-grid refinement is bounded only locally around its dictionary seed,
# so it can overshoot the grid ceiling and return atoms wider than the epoch
# itself (observed: a 17.9 s atom on a 2.0 s epoch, sweeping 9155 Hz). Such an
# atom is a fitting artifact, not a rhythm -- it is not time-localised and its
# chirp rate is meaningless. Atoms longer than this multiple of the epoch are
# classed as non-oscillatory.
MAX_DURATION_EPOCHS = 1.0
# A chirp rate is only identifiable if the atom actually oscillates, so a wild
# sweep on a sub-cycle atom is a fitting artifact too. Reject atoms whose
# instantaneous frequency changes by more than this multiple of their centre
# frequency across the envelope. Chosen by measurement, not taste: on synthetic
# eyes-open/closed data, a bound of 0.5-1.0 preserved the best class separation
# (Cohen's d = 0.91), while a hard in-band test on the swept frequency rejected
# genuine alpha atoms and cost ~0.15 d.
MAX_RELATIVE_SWEEP = 1.0

PER_CHANNEL_FEATURES: tuple[str, ...] = (
    *(f"frac_{b}" for b in BANDS),
    "osc_frac",
    "transient_frac",
    "mean_fc_hz",
    "std_fc_hz",
    "mean_abs_sweep_hz",
    "mean_signed_sweep_hz",
    "mean_dur_s",
    "recon_err",
)


@dataclass
class AtomPhysical:
    """One chirplet atom converted from sample-domain to physical units."""

    t_center_s: float
    f_center_hz: float
    duration_s: float
    chirp_rate_hz_s: float   # instantaneous df/dt
    sweep_hz: float          # signed frequency traversed across the envelope
    energy: float            # |coefficient|^2, unnormalised
    oscillatory: bool


def atom_to_physical(atom, fs: float, n: int, energy: float) -> AtomPhysical:
    """Convert a raw ACT atom to physical units.

    The atom's phase is ``2*pi/N * (c*(t-tc)^2 + fc*(t-tc))`` with ``t`` in
    samples, so instantaneous frequency in cycles/sample is
    ``(2c(t-tc) + fc)/N``. Converting to Hz and differentiating in seconds:

        f_centre = fc * fs / N                 [Hz]
        df/dt    = 2 * c * fs**2 / N           [Hz/s]

    `sweep_hz` is that rate times the atom's own duration -- the frequency
    actually traversed while the atom has appreciable amplitude, which is far
    more interpretable than the raw rate.

    An atom counts as oscillatory only if it is a plausible rhythm: enough
    cycles inside its envelope, a centre frequency in band, a duration that
    fits within the epoch, and a sweep that is modest relative to its own
    centre frequency. The last two tests reject the degenerate wide and
    fast-sweeping atoms the bounded off-grid refinement can produce.
    """
    f_hz = abs(float(atom.fc)) * fs / n
    dur_s = float(np.exp(atom.logDt)) / fs
    rate = 2.0 * float(atom.c) * fs * fs / n
    sweep = rate * dur_s
    cycles = f_hz * dur_s
    epoch_s = n / fs
    osc = bool(
        cycles >= MIN_CYCLES
        and EEG_BAND[0] <= f_hz <= EEG_BAND[1]
        and dur_s <= MAX_DURATION_EPOCHS * epoch_s
        and abs(sweep) <= MAX_RELATIVE_SWEEP * max(f_hz, 1e-9)
    )
    return AtomPhysical(
        t_center_s=float(atom.tc) / fs, f_center_hz=f_hz, duration_s=dur_s,
        chirp_rate_hz_s=rate, sweep_hz=sweep, energy=energy, oscillatory=osc,
    )


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.sum()
    return float((values * weights).sum() / w) if w > 1e-12 else 0.0


def channel_features(atoms, cert, fs: float, n: int) -> np.ndarray:
    """Aggregate one channel's atoms into a permutation-invariant vector.

    Aggregation matters: the raw atom list is ordered by greedy selection, so
    two near-equal atoms can swap places between epochs. Every feature here is
    an energy-weighted statistic over the whole atom set, so it is invariant to
    that ordering.
    """
    phys = [
        atom_to_physical(
            a, fs, n, float(a.coeff_real) ** 2 + float(a.coeff_imag) ** 2
        )
        for a in atoms
    ]
    total = sum(p.energy for p in phys)
    out: list[float] = []

    if total <= 1e-12:
        return np.zeros(len(PER_CHANNEL_FEATURES))

    osc = [p for p in phys if p.oscillatory]
    osc_e = sum(p.energy for p in osc)

    # Band fractions, over all atom energy so they sum to <= 1 alongside
    # transient_frac.
    for lo, hi in BANDS.values():
        e = sum(p.energy for p in osc if lo <= p.f_center_hz < hi)
        out.append(e / total)

    out.append(osc_e / total)
    out.append(1.0 - osc_e / total)

    if osc:
        f = np.array([p.f_center_hz for p in osc])
        d = np.array([p.duration_s for p in osc])
        s = np.array([p.sweep_hz for p in osc])
        w = np.array([p.energy for p in osc])
        mean_f = _weighted(f, w)
        out.append(mean_f)
        out.append(float(np.sqrt(max(_weighted((f - mean_f) ** 2, w), 0.0))))
        out.append(_weighted(np.abs(s), w))
        out.append(_weighted(s, w))
        out.append(_weighted(d, w))
    else:
        out.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    out.append(float(cert.reconstruction_error) if cert is not None else 0.0)
    return np.asarray(out, dtype=np.float64)


def feature_names(channels: tuple[str, ...]) -> list[str]:
    return [f"{ch}_{f}" for ch in channels for f in PER_CHANNEL_FEATURES]


def build_engine(length: int = EPOCH_SAMPLES, engine: str = "v7", **overrides):
    """Construct an ACT engine.

    v7 is the default because it is the only engine in the lineage that
    populates per-atom coefficients, which every energy-weighted feature here
    depends on. v9/v14 reconstruct better but return zero coefficients, so
    their atoms have to be weighted uniformly -- worse features despite better
    reconstruction. They are exposed for comparison, not recommended.

    Note also that the library's own `for_eeg_v7` factory is unusable at this
    epoch length: its default grids build ~281,600 atoms (~2.3 GB) and get the
    process OOM-killed. `MUSE_ACT_GRID` is the 1,584-atom validated alternative.
    """
    grid = {**MUSE_ACT_GRID, **overrides}
    if engine == "v7":
        from act import ACTv7

        return ACTv7(length=length, **grid)
    if engine == "v9":
        from act import ACTv9Asym

        return ACTv9Asym(length=length, diagonal_seed=True)
    if engine == "v14":
        from act import ACTv14Skew

        return ACTv14Skew(
            length=length, analytic_grad=True, newton=True, diagonal_seed=False
        )
    raise ValueError(f"unknown ACT engine {engine!r}; use v7, v9 or v14")


_WORKER: dict = {}


def _init_worker(length, engine, order, fs):
    _WORKER["engine"] = build_engine(length, engine)
    _WORKER["order"] = order
    _WORKER["fs"] = fs
    _WORKER["n"] = length


def _one_epoch(epoch: np.ndarray) -> np.ndarray:
    eng, order = _WORKER["engine"], _WORKER["order"]
    fs, n = _WORKER["fs"], _WORKER["n"]
    vecs = []
    for ch in range(epoch.shape[0]):
        atoms, cert = eng.transform(epoch[ch], order=order)
        vecs.append(channel_features(atoms, cert, fs, n))
    return np.concatenate(vecs)


def extract(
    epochs: EpochSet,
    *,
    order: int = 12,
    engine: str = "v7",
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Run ACT on every channel of every epoch and return (X, feature_names).

    `order` is the number of atoms per channel. More atoms explain more of the
    epoch but cost linearly more time; 12 is a reasonable balance at this
    dictionary size.
    """
    n_ep, n_ch, n_samp = epochs.X.shape
    if engine in ("v9", "v14"):
        warnings.warn(
            f"ACT engine {engine!r} does not emit per-atom coefficients, so "
            "all atoms are weighted equally and the energy-weighted features "
            "degrade. Use engine='v7' unless you are comparing engines.",
            stacklevel=2,
        )
    if verbose:
        print(
            f"[features] ACT {engine} order={order} on {n_ep} epochs "
            f"x {n_ch} channels ({n_ep * n_ch} transforms, n_jobs={n_jobs})"
        )

    if n_jobs == 1:
        _init_worker(n_samp, engine, order, epochs.fs)
        rows = []
        for i in range(n_ep):
            rows.append(_one_epoch(epochs.X[i]))
            if verbose and (i + 1) % 25 == 0:
                print(f"[features]   {i + 1}/{n_ep}")
        X = np.stack(rows)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_jobs, initializer=_init_worker,
            initargs=(n_samp, engine, order, epochs.fs),
        ) as pool:
            X = np.stack(pool.map(_one_epoch, list(epochs.X), chunksize=4))

    names = feature_names(epochs.channels)
    if X.shape[1] != len(names):
        raise AssertionError(f"{X.shape[1]} features but {len(names)} names")
    if verbose:
        print(f"[features] X = {X.shape}")
    return X, names

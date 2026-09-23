"""Cross-channel features: what the per-channel pipeline throws away.

Every feature set so far was computed per channel and concatenated -- 18
independent views of the same moment. But a seizure is defined partly by
*spatial synchrony*: the same time-frequency cell lighting up across many
channels at once, and spreading between them. None of that survives
per-channel aggregation, which makes it the largest untapped signal in the
pipeline, independent of anything quantum.

Input is a physical atom array, shared by every engine (classical GPU ACT and
QACT alike):

    atoms[epoch, channel, k] = (t_center_s, f_hz, duration_s, rate_hz_s, energy)

Output is 18 features per epoch:

  spatial spread (3)
    energy_participation   (sum E)^2 / (n * sum E^2): 1 = every channel equally
                           active, 1/n = one channel only
    energy_entropy         normalised entropy of the per-channel energy share
    profile_coherence      mean pairwise cosine similarity of the 5-band energy
                           profile -- high when channels share a rhythm
  per band, 5 bands (10)
    cooccur_<band>         fraction of channel pairs where both channels have a
                           band atom within COINCIDENCE_S of each other
    lag_<band>             mean |dt| among those co-occurring pairs (propagation
                           delay), 0 when none
  recruitment and agreement (5)
    recruitment            fraction of channels with an atom matching the
                           epoch's strongest atom in both frequency and time
    freq_disagreement      mean pairwise |difference| of energy-weighted mean
                           frequency across channels (low = synchronised)
    freq_spread            std across channels of that mean frequency
    sweep_disagreement     same, for energy-weighted chirp rate
    top_band_agreement     fraction of channels whose strongest atom falls in the
                           same band as the epoch's strongest atom
"""
from __future__ import annotations

import numpy as np

from .features import BANDS

COINCIDENCE_S = 0.25      # two atoms count as simultaneous within this
FREQ_TOL_HZ = 2.0         # ... and as the same rhythm within this

CROSS_FEATURES: tuple[str, ...] = (
    "energy_participation", "energy_entropy", "profile_coherence",
    *(f"cooccur_{b}" for b in BANDS),
    *(f"lag_{b}" for b in BANDS),
    "recruitment", "freq_disagreement", "freq_spread", "sweep_disagreement",
    "top_band_agreement",
)


def cross_channel_features(atoms: np.ndarray) -> np.ndarray:
    """atoms: (E, C, K, 5) -> (E, len(CROSS_FEATURES))."""
    atoms = np.asarray(atoms, dtype=np.float64)
    E, C, K, _ = atoms.shape
    t, f, _dur, rate, en = (atoms[..., i] for i in range(5))
    out = np.zeros((E, len(CROSS_FEATURES)))
    col = 0

    # --- spatial spread of energy ---
    Ec = en.sum(2)                                            # (E, C)
    tot = Ec.sum(1)
    live = tot > 1e-12
    safe = np.where(live, tot, 1.0)
    share = Ec / safe[:, None]
    out[:, col] = (tot ** 2) / (C * np.maximum((Ec ** 2).sum(1), 1e-30)); col += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(share * np.log(np.where(share > 0, share, 1.0))).sum(1) / np.log(C)
    out[:, col] = np.nan_to_num(ent); col += 1

    # --- band energy profile per channel, and its pairwise coherence ---
    prof = np.zeros((E, C, len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS.values()):
        prof[..., bi] = (en * ((f >= lo) & (f < hi))).sum(2)
    pn = prof / np.maximum(np.linalg.norm(prof, axis=2, keepdims=True), 1e-12)
    sim = np.einsum("ecb,edb->ecd", pn, pn)                   # (E, C, C) cosine
    iu = np.triu_indices(C, 1)
    out[:, col] = sim[:, iu[0], iu[1]].mean(1); col += 1

    # --- per-band co-occurrence and propagation lag ---
    lag_cols = []
    for bi, (lo, hi) in enumerate(BANDS.values()):
        inb = (f >= lo) & (f < hi)
        w = en * inb
        has = w.sum(2) > 1e-12                                # (E, C)
        tb = (w * t).sum(2) / np.where(has, np.maximum(w.sum(2), 1e-30), 1.0)
        dt = np.abs(tb[:, :, None] - tb[:, None, :])
        both = has[:, :, None] & has[:, None, :]
        co = both & (dt <= COINCIDENCE_S)
        cop = co[:, iu[0], iu[1]]
        out[:, col + bi] = cop.mean(1)
        lag = np.where(cop, dt[:, iu[0], iu[1]], 0.0).sum(1) / np.maximum(cop.sum(1), 1)
        lag_cols.append(lag)
    col += len(BANDS)
    for bi in range(len(BANDS)):
        out[:, col + bi] = lag_cols[bi]
    col += len(BANDS)

    # --- recruitment around the epoch's strongest atom ---
    flat = en.reshape(E, C * K)
    top = flat.argmax(1)
    tc_top = t.reshape(E, C * K)[np.arange(E), top]
    f_top = f.reshape(E, C * K)[np.arange(E), top]
    match = ((np.abs(f - f_top[:, None, None]) <= FREQ_TOL_HZ)
             & (np.abs(t - tc_top[:, None, None]) <= COINCIDENCE_S)
             & (en > 0)).any(2)                               # (E, C)
    out[:, col] = match.mean(1); col += 1

    # --- agreement of per-channel weighted means ---
    wsum = np.maximum(Ec, 1e-30)
    fbar = (en * f).sum(2) / wsum
    rbar = (en * rate).sum(2) / wsum
    for arr in (fbar, rbar):
        d = np.abs(arr[:, :, None] - arr[:, None, :])[:, iu[0], iu[1]]
        if arr is fbar:
            out[:, col] = d.mean(1); col += 1
            out[:, col] = arr.std(1); col += 1
        else:
            out[:, col] = d.mean(1); col += 1
    # top-atom band agreement
    top_ch = en.argmax(2)                                     # (E, C)
    f_topch = np.take_along_axis(f, top_ch[..., None], 2)[..., 0]
    def band_of(x):
        b = np.full(x.shape, -1)
        for bi, (lo, hi) in enumerate(BANDS.values()):
            b = np.where((x >= lo) & (x < hi), bi, b)
        return b
    out[:, col] = (band_of(f_topch) == band_of(f_top)[:, None]).mean(1); col += 1

    out[~live] = 0.0
    assert col == len(CROSS_FEATURES), (col, len(CROSS_FEATURES))
    return out


def qact_params_to_atoms(params: np.ndarray, fs: float, N: int) -> np.ndarray:
    """QACT (E*C, K, 6) -> physical (E*C, K, 5) = t_s, f_hz, dur_s, rate, energy."""
    tc, logDt, fc, c, cr, ci = (params[..., i] for i in range(6))
    return np.stack([tc / fs, np.abs(fc) * fs / N, np.exp(logDt) / fs,
                     2.0 * c * fs * fs / N, cr ** 2 + ci ** 2], -1)

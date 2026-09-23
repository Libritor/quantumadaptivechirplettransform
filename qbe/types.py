"""Shared data containers for the Quantum Brain Encoding pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The Muse EEG montage. Order is fixed and matches the Mind Monitor CSV export
# and the muselsl / BrainFlow channel order.
MUSE_CHANNELS: tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10")

# Nominal Muse sample rate. The hardware clock drifts and Bluetooth drops
# packets, which is why `preprocess` gates on the *measured* timestamp span
# rather than trusting this number.
MUSE_FS: float = 256.0

# Epoch length in samples. 512 @ 256 Hz = 2.0 s, the epoch size the upstream
# ACT repository validated its real-Muse protocol on. Changing this invalidates
# the dictionary grids in `features.MUSE_ACT_GRID`.
EPOCH_SAMPLES: int = 512


@dataclass
class Recording:
    """A single continuous block of raw Muse EEG, before any gating.

    `data` is in raw ADC units (nominally 0..1649 for Mind Monitor exports),
    NOT microvolts, because the rail-detection gate in `preprocess` works in
    ADC units. `ms` is elapsed milliseconds per sample and is used to detect
    Bluetooth packet loss -- it is not assumed to be uniformly spaced.
    """

    ms: np.ndarray                    # (n_samples,) elapsed ms
    data: np.ndarray                  # (n_channels, n_samples) raw ADC units
    channels: tuple[str, ...]
    fs: float
    label: int | None = None          # class index, or None for unlabelled
    name: str = "recording"

    def __post_init__(self) -> None:
        self.ms = np.asarray(self.ms, dtype=np.float64)
        self.data = np.atleast_2d(np.asarray(self.data, dtype=np.float64))
        if self.data.shape[1] != self.ms.shape[0]:
            raise ValueError(
                f"{self.name}: {self.data.shape[1]} samples but "
                f"{self.ms.shape[0]} timestamps"
            )
        if self.data.shape[0] != len(self.channels):
            raise ValueError(
                f"{self.name}: {self.data.shape[0]} rows but "
                f"{len(self.channels)} channel names"
            )

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


@dataclass
class EpochSet:
    """Gated, normalised epochs ready for chirplet decomposition.

    Each epoch has been zero-meaned and unit-L2 normalised per channel, per the
    upstream protocol. That normalisation is why absolute amplitude is not a
    usable feature here and all spectral features are *fractions* of epoch
    energy -- see `features`.
    """

    X: np.ndarray                     # (n_epochs, n_channels, n_samples)
    y: np.ndarray                     # (n_epochs,) int labels
    channels: tuple[str, ...]
    fs: float
    label_names: tuple[str, ...]
    meta: list[dict] = field(default_factory=list)
    # Optional per-epoch features computed BEFORE normalisation (e.g. amplitude),
    # which the unit-energy epochs in X can no longer express.
    extra: np.ndarray | None = None
    extra_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=int)

    @property
    def n_epochs(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    def counts(self) -> dict[str, int]:
        return {
            name: int((self.y == i).sum())
            for i, name in enumerate(self.label_names)
        }

    def __repr__(self) -> str:
        return (
            f"EpochSet(n={self.n_epochs}, ch={len(self.channels)}, "
            f"samples={self.X.shape[2] if self.X.size else 0}, "
            f"counts={self.counts()})"
        )

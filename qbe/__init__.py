"""Quantum Brain Encoding: Muse EEG -> Adaptive Chirplet Transform -> Qiskit.

Pipeline stages, each in its own module:

    acquire     Muse data sources (synthetic, Mind Monitor CSV, LSL, BrainFlow)
    preprocess  epoch gating and normalisation (deliberately no filtering)
    features    Adaptive Chirplet Transform -> energy-weighted feature vector
    quantum     Qiskit QSVC / VQC plus classical baselines
"""
from . import acquire, features, preprocess, quantum
from .types import MUSE_CHANNELS, MUSE_FS, EPOCH_SAMPLES, EpochSet, Recording

__version__ = "0.1.0"
__all__ = [
    "acquire", "features", "preprocess", "quantum",
    "EpochSet", "Recording", "MUSE_CHANNELS", "MUSE_FS", "EPOCH_SAMPLES",
]

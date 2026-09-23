"""Minimal EDF/EDF+ reader (numpy only).

Adapted from the sibling EEG-Memristor-ACT project's reader so this package
stays self-contained. EDF stores 16-bit integers per data record plus a
per-signal linear calibration, which is all this needs to handle.
"""
from __future__ import annotations

import numpy as np


def read_edf_start_seconds(path) -> float:
    """Recording start time from the EDF header, as seconds since midnight.

    Bytes 168..176 hold 'hh.mm.ss'. Needed for datasets that annotate seizures in
    wall-clock time (Siena) rather than seconds from file start (CHB-MIT).
    """
    with open(path, "rb") as f:
        head = f.read(184)
    txt = head[168:176].decode("latin1").strip()
    parts = [p for p in txt.replace(":", ".").split(".") if p]
    h, m, sec = (int(parts[0]), int(parts[1]), int(parts[2])) if len(parts) >= 3 else (0, 0, 0)
    return h * 3600 + m * 60 + sec


def read_edf(path) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, str]]:
    """Return ({label: signal}, {label: sample_rate}, {label: unit})."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256])
        n_records = int(head[236:244])
        record_secs = float(head[244:252])
        hs = f.read(256 * ns)

        def field(off: int, width: int) -> list[str]:
            return [
                hs[off + i * width: off + (i + 1) * width].decode("latin1").strip()
                for i in range(ns)
            ]

        o = 0
        labels = field(o, 16); o += 16 * ns
        o += 80 * ns                                    # transducer type
        units = field(o, 8); o += 8 * ns
        pmin = np.array(field(o, 8), float); o += 8 * ns
        pmax = np.array(field(o, 8), float); o += 8 * ns
        dmin = np.array(field(o, 8), float); o += 8 * ns
        dmax = np.array(field(o, 8), float); o += 8 * ns
        o += 80 * ns                                    # prefiltering
        n_samp = np.array(field(o, 8), int)
        body = f.read()
        # A truncated file can end mid-sample; drop the dangling byte. Whole
        # truncated records are dropped below. Callers that need an intact
        # file should verify its checksum first (see acquire.load_chbmit).
        raw = np.frombuffer(body[: len(body) - len(body) % 2], dtype="<i2")

    per_record = int(n_samp.sum())
    n_records = min(
        n_records if n_records > 0 else raw.size // per_record,
        raw.size // per_record,
    )
    raw = raw[: n_records * per_record].reshape(n_records, per_record)

    gain = (pmax - pmin) / (dmax - dmin)
    offset = pmin - gain * dmin

    signals, rates = {}, {}
    col = 0
    for i, label in enumerate(labels):
        block = raw[:, col: col + n_samp[i]].reshape(-1).astype(float)
        col += n_samp[i]
        if label.startswith("EDF Annotations"):
            continue
        signals[label] = block * gain[i] + offset[i]
        rates[label] = n_samp[i] / record_secs
    return signals, rates, dict(zip(labels, units))

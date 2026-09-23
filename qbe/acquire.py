"""Muse data sources: synthetic, Mind Monitor CSV, LSL, and BrainFlow.

Every source returns `Recording` objects in raw ADC units so that the
rail-detection gate in `preprocess` works identically regardless of origin.

Channel order is always (TP9, AF7, AF8, TP10) -- left ear, left forehead,
right forehead, right ear. That montage matters: AF7/AF8 sit directly above
the eyes and carry large blink artifacts, while alpha rhythm is strongest at
the posterior TP9/TP10 pair.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .types import MUSE_CHANNELS, MUSE_FS, EpochSet, Recording

# Mind Monitor exports raw channels with a _RAW suffix and elapsed time in ms.
CSV_RAW_COLUMNS = ("TP9_RAW", "AF7_RAW", "AF8_RAW", "TP10_RAW")
CSV_TIME_COLUMN = "ms_ELAPSED"

# Muse's 12-bit ADC spans ~1682 uV, so roughly 1 ADC unit per uV, centred
# near mid-scale. Synthetic data is generated in these units.
ADC_BASELINE = 820.0


# --------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------
def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f background -- the dominant component of real resting EEG."""
    f = np.fft.rfftfreq(n, 1.0)
    amp = np.zeros_like(f)
    amp[1:] = 1.0 / f[1:]
    phase = rng.uniform(0, 2 * np.pi, f.shape[0])
    x = np.fft.irfft(amp * np.exp(1j * phase), n=n)
    sd = float(np.std(x))
    return x / sd if sd > 0 else x


def synthetic_recording(
    label: int,
    condition: str,
    *,
    seconds: float = 60.0,
    fs: float = MUSE_FS,
    alpha_uv: float = 18.0,
    seed: int | None = None,
    blink_rate_hz: float = 0.25,
    line_noise_uv: float = 0.0,
    name: str | None = None,
) -> Recording:
    """Generate one realistic synthetic Muse recording.

    This is a *stand-in for the headband*, not a simulation of neurophysiology.
    It exists so the whole pipeline runs and is testable before any hardware is
    paired. It deliberately includes the things that break naive pipelines:
    1/f background, blink transients localised to the frontal pair, dropped
    packets in the timestamps, and frequency-sweeping bursts (which is what
    makes the chirplet representation earn its keep over a plain FFT).

    `condition` is "closed" (strong posterior alpha) or "open" (suppressed).
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    n_ch = len(MUSE_CHANNELS)
    sig = np.zeros((n_ch, n))

    # 1/f background, independent per channel.
    for c in range(n_ch):
        sig[c] = 12.0 * _pink(n, rng)

    # Alpha. Eyes-closed drives a large posterior (TP9/TP10) increase; the
    # frontal pair sees a weaker version. This is the effect the classifier
    # is meant to find.
    posterior = {"TP9": 1.0, "TP10": 1.0, "AF7": 0.35, "AF8": 0.35}
    amp = alpha_uv if condition == "closed" else alpha_uv * 0.18
    n_bursts = int(seconds * (0.9 if condition == "closed" else 0.5))
    for _ in range(max(n_bursts, 1)):
        t0 = rng.uniform(0, seconds)
        dur = rng.uniform(0.35, 1.1)
        f0 = rng.uniform(9.0, 11.5)
        env = np.exp(-((t - t0) ** 2) / (2 * (dur / 2.5) ** 2))
        ph = rng.uniform(0, 2 * np.pi)
        for c, ch in enumerate(MUSE_CHANNELS):
            sig[c] += amp * posterior[ch] * env * np.sin(2 * np.pi * f0 * t + ph)

    # Frequency-sweeping bursts. Real EEG transients drift in frequency; a
    # stationary Gabor atom cannot represent this but a chirplet can, so this
    # is the component the chirp-rate features are built to pick up.
    for _ in range(max(int(seconds * 0.4), 1)):
        t0 = rng.uniform(0, seconds)
        dur = rng.uniform(0.25, 0.7)
        f0 = rng.uniform(6.0, 20.0)
        rate = rng.uniform(-14.0, 14.0)
        env = np.exp(-((t - t0) ** 2) / (2 * (dur / 2.5) ** 2))
        sweep = np.sin(2 * np.pi * (f0 * (t - t0) + 0.5 * rate * (t - t0) ** 2))
        c = rng.integers(0, n_ch)
        sig[c] += rng.uniform(4.0, 9.0) * env * sweep

    # Low-amplitude beta, everywhere.
    for c in range(n_ch):
        sig[c] += 3.0 * np.sin(
            2 * np.pi * rng.uniform(16, 24) * t + rng.uniform(0, 6.28)
        )

    # Blinks: large, slow, frontal. These are what the artifact gate removes.
    for _ in range(max(int(seconds * blink_rate_hz), 0)):
        t0 = rng.uniform(0, seconds)
        env = np.exp(-((t - t0) ** 2) / (2 * 0.08**2))
        mag = rng.uniform(150.0, 320.0)
        for c, ch in enumerate(MUSE_CHANNELS):
            sig[c] += mag * env * (1.0 if ch in ("AF7", "AF8") else 0.15)

    if line_noise_uv > 0:
        sig += line_noise_uv * np.sin(2 * np.pi * 60.0 * t)[None, :]

    data = np.clip(ADC_BASELINE + sig, 0.0, 1649.0)

    # Timestamps with occasional Bluetooth stalls, so the packet-loss gate has
    # something real to catch.
    dt = np.full(n, 1000.0 / fs)
    for _ in range(max(int(seconds * 0.05), 1)):
        i = int(rng.integers(0, n))
        dt[i] += rng.uniform(250.0, 650.0)
    ms = np.cumsum(dt) - dt[0]

    return Recording(
        ms=ms, data=data, channels=MUSE_CHANNELS, fs=fs, label=label,
        name=name or f"synthetic-{condition}-{seed}",
    )


def synthetic_dataset(
    *,
    n_per_class: int = 6,
    seconds: float = 60.0,
    seed: int = 0,
    label_names: tuple[str, str] = ("eyes_open", "eyes_closed"),
    **kwargs,
) -> list[Recording]:
    """A balanced set of synthetic recordings, one label per condition."""
    conds = {"eyes_open": "open", "eyes_closed": "closed"}
    recs = []
    for label, name in enumerate(label_names):
        for k in range(n_per_class):
            recs.append(
                synthetic_recording(
                    label=label, condition=conds[name], seconds=seconds,
                    seed=seed + 1000 * label + k, name=f"{name}-{k}", **kwargs,
                )
            )
    return recs


# --------------------------------------------------------------------------
# Mind Monitor CSV
# --------------------------------------------------------------------------
def read_mind_monitor_csv(path: str | Path, label: int | None = None) -> Recording | None:
    """Read one Mind Monitor CSV export. Returns None if it lacks raw columns.

    Mind Monitor can export with raw channels disabled, in which case only the
    on-device band powers are present -- those cannot be chirplet-decomposed,
    so such files are skipped rather than silently half-used.
    """
    path = Path(path)
    ms: list[float] = []
    cols: dict[str, list[float]] = {c: [] for c in CSV_RAW_COLUMNS}
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr, None)
        if not hdr:
            return None
        idx = {h: i for i, h in enumerate(hdr)}
        if any(c not in idx for c in CSV_RAW_COLUMNS) or CSV_TIME_COLUMN not in idx:
            return None
        need = [idx[c] for c in CSV_RAW_COLUMNS] + [idx[CSV_TIME_COLUMN]]
        for row in rdr:
            if len(row) <= max(need):
                continue
            try:
                vals = [float(row[i]) for i in need[:-1]]
                tv = float(row[need[-1]])
            except ValueError:
                continue  # blank raw cells are normal in Mind Monitor exports
            ms.append(tv)
            for c, v in zip(CSV_RAW_COLUMNS, vals):
                cols[c].append(v)
    if len(ms) < 2:
        return None
    return Recording(
        ms=np.asarray(ms),
        data=np.stack([np.asarray(cols[c]) for c in CSV_RAW_COLUMNS]),
        channels=MUSE_CHANNELS, fs=MUSE_FS, label=label, name=path.stem,
    )


def load_csv_directory(
    root: str | Path, label_names: tuple[str, ...]
) -> list[Recording]:
    """Load `<root>/<label_name>/*.csv`, taking the class from the folder name."""
    root = Path(root)
    recs: list[Recording] = []
    for label, name in enumerate(label_names):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"expected a folder per class; missing {folder}. "
                f"Create {root}/{'/'.join(label_names)} and put CSVs inside."
            )
        found = 0
        for p in sorted(folder.glob("*.csv")):
            rec = read_mind_monitor_csv(p, label=label)
            if rec is None:
                print(f"[acquire] skipped {p.name}: no raw EEG columns")
                continue
            recs.append(rec)
            found += 1
        print(f"[acquire] {name}: {found} recording(s)")
    if not recs:
        raise RuntimeError(f"no usable CSVs under {root}")
    return recs


# --------------------------------------------------------------------------
# Live capture
# --------------------------------------------------------------------------
def record_lsl(
    label: int, seconds: float, *, name: str = "lsl", timeout: float = 15.0
) -> Recording:
    """Capture from a running muselsl stream.

    Start the stream first, in a separate terminal:
        muselsl stream --name <your-muse>
    """
    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError as exc:
        raise ImportError(
            "pylsl is required for LSL capture: pip install muselsl pylsl"
        ) from exc

    streams = resolve_byprop("type", "EEG", timeout=timeout)
    if not streams:
        raise RuntimeError(
            "no LSL EEG stream found. Run `muselsl stream` in another terminal."
        )
    inlet = StreamInlet(streams[0], max_chunklen=12)
    fs = float(inlet.info().nominal_srate()) or MUSE_FS
    samples: list[list[float]] = []
    stamps: list[float] = []
    print(f"[acquire] recording {seconds:.0f}s from LSL at {fs:.0f} Hz ...")
    while len(samples) < int(seconds * fs):
        chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=32)
        if chunk:
            samples.extend(chunk)
            stamps.extend(ts)
    arr = np.asarray(samples, dtype=np.float64).T
    # muselsl exposes 5 channels (the 5th is AUX/right-aux); keep the first 4.
    arr = arr[: len(MUSE_CHANNELS)]
    ms = (np.asarray(stamps) - stamps[0]) * 1000.0
    return Recording(
        ms=ms, data=arr, channels=MUSE_CHANNELS, fs=fs, label=label, name=name
    )


def record_brainflow(
    label: int,
    seconds: float,
    *,
    board: str = "MUSE_S_BLED",
    serial_port: str = "",
    mac_address: str = "",
    name: str = "brainflow",
) -> Recording:
    """Capture via BrainFlow.

    BrainFlow board support for specific Muse models changes between releases;
    `board` is passed through by name so you can select whatever your installed
    version exposes (e.g. MUSE_2_BOARD, MUSE_S_BOARD, MUSE_S_BLED_BOARD).
    Run `python -c "import brainflow; print([b for b in dir(brainflow.BoardIds)
    if 'MUSE' in b])"` to list them.
    """
    try:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    except ImportError as exc:
        raise ImportError(
            "brainflow is required for this source: pip install brainflow"
        ) from exc

    key = board if board.endswith("_BOARD") else f"{board}_BOARD"
    available = [b for b in dir(BoardIds) if "MUSE" in b]
    if not hasattr(BoardIds, key):
        raise ValueError(
            f"unknown BrainFlow board {key!r}. Available Muse boards in your "
            f"brainflow build: {available}"
        )
    board_id = getattr(BoardIds, key).value

    params = BrainFlowInputParams()
    params.serial_port = serial_port
    params.mac_address = mac_address
    shim = BoardShim(board_id, params)
    shim.prepare_session()
    shim.start_stream()
    print(f"[acquire] recording {seconds:.0f}s from BrainFlow {key} ...")
    import time as _time

    _time.sleep(seconds)
    raw = shim.get_board_data()
    shim.stop_stream()
    shim.release_session()

    eeg_rows = BoardShim.get_eeg_channels(board_id)[: len(MUSE_CHANNELS)]
    fs = float(BoardShim.get_sampling_rate(board_id))
    data = raw[eeg_rows]
    try:
        ts_row = BoardShim.get_timestamp_channel(board_id)
        ms = (raw[ts_row] - raw[ts_row][0]) * 1000.0
    except Exception:
        ms = np.arange(data.shape[1]) * (1000.0 / fs)
    return Recording(
        ms=ms, data=data, channels=MUSE_CHANNELS, fs=fs, label=label, name=name
    )


# --------------------------------------------------------------------------
# PhysioNet EEGMAT (EEG During Mental Arithmetic Tasks)
# --------------------------------------------------------------------------
# Zyma et al., "Electroencephalograms during Mental Arithmetic Task
# Performance", Data 4(1), 2019. 36 subjects, two runs each:
#   run 1 = rest / baseline   (~180 s)
#   run 2 = mental arithmetic (~60 s)
# 19-channel 10-20 montage + ECG at 500 Hz, already ICA-cleaned by the authors.
#
# This is real EEG on a genuine 2-class cognitive contrast, which makes it a far
# better test than the synthetic Muse generator. Three things differ from Muse
# and are handled below:
#   * 500 Hz, not 256 -- resampled so the validated ACT dictionary grid still
#     applies unchanged.
#   * microvolts, not ADC counts -- so the ADC-rail gate must be disabled.
#   * every file ends in ~1-2 s of near-zero padding, which is trimmed.
EEGMAT_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
)
EEGMAT_FS = 500.0
EEGMAT_LABELS = ("rest", "arithmetic")


def read_eegmat_file(
    path: str | Path,
    label: int,
    *,
    channels: tuple[str, ...] = EEGMAT_CHANNELS,
    target_fs: float = MUSE_FS,
) -> Recording:
    """Read one EEGMAT EDF, trim its zero padding, and resample to `target_fs`."""
    from .edf import read_edf

    path = Path(path)
    sigs, rates, _ = read_edf(path)
    missing = [c for c in channels if f"EEG {c}" not in sigs]
    if missing:
        raise KeyError(f"{path.name}: missing channels {missing}")
    data = np.stack([sigs[f"EEG {c}"] for c in channels])
    fs_in = float(rates[f"EEG {channels[0]}"])

    # Trim the trailing near-zero padding (|x| ~ 1e-3 uV) present in every file.
    alive = np.flatnonzero(np.abs(data).max(axis=0) > 0.05)
    if alive.size == 0:
        raise ValueError(f"{path.name}: all-zero recording")
    data = data[:, : alive[-1] + 1]

    if abs(fs_in - target_fs) > 1e-9:
        from fractions import Fraction
        from scipy.signal import resample_poly

        ratio = Fraction(target_fs / fs_in).limit_denominator(1000)
        data = resample_poly(data, ratio.numerator, ratio.denominator, axis=1)

    n = data.shape[1]
    ms = np.arange(n) * (1000.0 / target_fs)
    return Recording(
        ms=ms, data=data, channels=channels, fs=target_fs, label=label,
        name=path.stem,
    )


def load_eegmat(
    root: str | Path,
    *,
    channels: tuple[str, ...] = EEGMAT_CHANNELS,
    subjects: list[str] | None = None,
    good_quality_only: bool = False,
    target_fs: float = MUSE_FS,
    verbose: bool = True,
) -> list[Recording]:
    """Load the EEGMAT corpus as labelled recordings.

    `good_quality_only` keeps the 26 subjects the dataset marks as having
    counted well (`Count quality` = 1 in subject-info.csv). The other 10 were
    poor counters, so their "arithmetic" run is a weaker instance of the
    intended cognitive state -- a defensible but optional restriction.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"EEGMAT folder not found: {root}")

    quality: dict[str, int] = {}
    info = root / "subject-info.csv"
    if info.is_file():
        rows = info.read_text().strip().splitlines()[1:]
        for r in rows:
            parts = r.split(",")
            quality[parts[0]] = int(parts[-1])

    if subjects is None:
        subjects = sorted({p.stem.rsplit("_", 1)[0] for p in root.glob("Subject*_*.edf")})
    if good_quality_only:
        if not quality:
            raise FileNotFoundError(f"need {info} for --good-only")
        subjects = [s for s in subjects if quality.get(s, 0) == 1]

    recs: list[Recording] = []
    for subj in subjects:
        for label, run in ((0, 1), (1, 2)):
            p = root / f"{subj}_{run}.edf"
            if not p.is_file():
                continue
            recs.append(
                read_eegmat_file(p, label, channels=channels, target_fs=target_fs)
            )
    if not recs:
        raise RuntimeError(f"no EEGMAT EDF files under {root}")
    if verbose:
        secs = sum(r.n_samples for r in recs) / target_fs
        print(f"[acquire] EEGMAT: {len(subjects)} subjects, {len(recs)} runs, "
              f"{secs / 60:.0f} min, {len(channels)} channels @ {target_fs:.0f} Hz")
    return recs


def subject_of(recording_name: str) -> str:
    """Group key for subject-wise cross-validation ('Subject07_2' -> 'Subject07')."""
    return recording_name.rsplit("_", 1)[0]


# --------------------------------------------------------------------------
# PhysioNet CHB-MIT Scalp EEG Database (seizure detection)
# --------------------------------------------------------------------------
# Shoeb, "Application of machine learning to epileptic seizure onset detection
# and treatment", MIT PhD thesis, 2009. 23 cases from 22 pediatric patients with
# intractable seizures, 256 Hz, bipolar 10-20 montage, 198 annotated seizures.
# Task here: ictal (inside an annotated seizure) vs interictal (from files with
# no seizure at all), cross-patient.
CHBMIT_CHANNELS: tuple[str, ...] = (
    # The 18-channel "double banana" bipolar montage present in nearly every
    # file. Some cases change montage mid-recording; files missing any of these
    # are skipped rather than zero-filled.
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)
CHBMIT_FS = 256.0
CHBMIT_LABELS = ("interictal", "ictal")


def chbmit_subject(case: str) -> str:
    """Group key. chb21 is chb01 recorded 1.5 years later (same patient), so
    they must share a group or one brain would sit on both sides of a split."""
    return "chb01" if case == "chb21" else case


def parse_chbmit_summary(path) -> dict[str, list[tuple[float, float]]]:
    """{edf file name: [(seizure start s, end s), ...]}; [] for seizure-free files.

    Handles both annotation styles used across cases: 'Seizure Start Time: N
    seconds' and 'Seizure 3 Start Time: N seconds'.
    """
    import re

    out: dict[str, list[tuple[float, float]]] = {}
    cur, starts = None, []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+\.edf)", line)
        if m:
            cur = m.group(1); out[cur] = []; starts = []
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)", line)
        if m and cur:
            starts.append(float(m.group(1)))
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)", line)
        if m and cur and starts:
            out[cur].append((starts.pop(0), float(m.group(1))))
    return out


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_chbmit(
    root: str | Path,
    *,
    n_samples: int = 512,
    interictal_ratio: float = 1.0,
    max_interictal_files: int = 12,
    cases: list[str] | None = None,
    verify: bool = True,
    seed: int = 0,
    verbose: bool = True,
):
    """Build a balanced, gated, normalised ictal/interictal EpochSet.

    Returns (EpochSet, groups) with one group per *patient*.

    * Ictal epochs: every non-overlapping window lying fully inside an
      annotated seizure.
    * Interictal epochs: the same number per patient (x `interictal_ratio`),
      drawn at random from files containing no seizure at all, spread across up
      to `max_interictal_files` files so no single hour dominates.
    * Gate: only non-finite windows and flat (disconnected) channels are
      rejected. The blink/artifact z-gate used for EEGMAT is deliberately NOT
      applied: a seizure is itself a burst of large, spiky activity, so that
      gate would preferentially discard ictal epochs and bias the data.
    * `verify` checks each file's SHA-256 against PhysioNet's SHA256SUMS.txt,
      so a truncated or corrupted download can never enter the data.
    """
    from .edf import read_edf
    from .preprocess import normalise

    root = Path(root)
    sums = {}
    if verify:
        for line in (root / "SHA256SUMS.txt").read_text().splitlines():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = h
    rng = np.random.default_rng(seed)
    with_seizures = set()
    rws = root / "RECORDS-WITH-SEIZURES"
    if rws.is_file():
        with_seizures = {ln.strip().split("/")[-1] for ln in rws.read_text().splitlines()
                         if ln.strip()}
    all_cases = sorted(p.name for p in root.glob("chb[0-9][0-9]") if p.is_dir())
    cases = cases or all_cases

    X, y, groups, meta, amp = [], [], [], [], []
    skipped: dict[str, int] = {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    def load(case, fname):
        path = root / case / fname
        if not path.is_file():
            skip("missing file"); return None
        if verify and sums.get(f"{case}/{fname}") != _sha256(path):
            skip("checksum mismatch (incomplete download?)"); return None
        sigs, _, _ = read_edf(path)
        if any(c not in sigs for c in CHBMIT_CHANNELS):
            skip("montage lacks a standard channel"); return None
        return np.stack([sigs[c] for c in CHBMIT_CHANNELS])

    def take(data, start, label, case, fname):
        w = data[:, start:start + n_samples]
        if w.shape[1] < n_samples or not np.isfinite(w).all():
            skip("non-finite/short window"); return False
        if (w.std(axis=1) < 1e-3).any():
            skip("flat channel"); return False
        chans = [normalise(w[c]) for c in range(w.shape[0])]
        if any(c is None for c in chans):
            skip("flat channel"); return False
        # Amplitude is the strongest seizure cue and the unit-energy
        # normalisation above erases it, so record it first: log RMS and log
        # line length (Esteller et al. 2001) per channel, from the raw uV.
        amp.append(np.concatenate([np.log(w.std(axis=1) + 1e-6),
                                   np.log(np.abs(np.diff(w, axis=1)).mean(axis=1) + 1e-6)]))
        X.append(np.stack(chans)); y.append(label)
        groups.append(chbmit_subject(case))
        meta.append({"rec": fname, "start": int(start), "case": case})
        return True

    for case in cases:
        ann = parse_chbmit_summary(root / case / f"{case}-summary.txt")
        n_ictal = 0
        for fname, seizures in ann.items():
            if not seizures:
                continue
            data = load(case, fname)
            if data is None:
                continue
            for s, e in seizures:
                a = int(round(s * CHBMIT_FS))
                while a + n_samples <= int(round(e * CHBMIT_FS)):
                    n_ictal += take(data, a, 1, case, fname)
                    a += n_samples
        want = int(round(n_ictal * interictal_ratio))
        # Seizure-free = every EDF in the folder that is not a seizure file.
        # Not "files the summary lists with zero seizures": chb24's summary
        # lists only its seizure files, which would leave it no interictal data.
        seizure_files = {f for f, sz in ann.items() if sz} | with_seizures
        free = sorted(p.name for p in (root / case).glob(f"{case}_*.edf")
                      if p.name not in seizure_files)
        rng.shuffle(free)
        free = free[:max_interictal_files]
        got = 0
        for k, fname in enumerate(free):
            if got >= want:
                break
            data = load(case, fname)
            if data is None:
                continue
            quota = int(np.ceil((want - got) / (len(free) - k)))
            starts = rng.choice(np.arange(0, data.shape[1] - n_samples, n_samples),
                                size=min(quota, (data.shape[1] - n_samples) // n_samples),
                                replace=False)
            for a in sorted(starts):
                if got >= want:
                    break
                got += take(data, int(a), 0, case, fname)
        if verbose:
            print(f"[chbmit] {case}: {n_ictal} ictal, {got} interictal epochs", flush=True)

    es = EpochSet(X=np.stack(X).astype(np.float32), y=np.asarray(y),
                  channels=CHBMIT_CHANNELS, fs=CHBMIT_FS,
                  label_names=CHBMIT_LABELS, meta=meta,
                  extra=np.asarray(amp),
                  extra_names=[f"{c}_log_rms" for c in CHBMIT_CHANNELS]
                              + [f"{c}_log_linelength" for c in CHBMIT_CHANNELS])
    if verbose:
        print(f"[chbmit] total {es.n_epochs} epochs {es.counts()}, "
              f"{len(set(groups))} patients", flush=True)
        if skipped:
            print("[chbmit] skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()))
    return es, np.asarray(groups)


# --------------------------------------------------------------------------
# PhysioNet Siena Scalp EEG Database (independent confirmation set)
# --------------------------------------------------------------------------
# Detti P., "Siena Scalp EEG Database" (PhysioNet, 2020). 14 patients, 47
# seizures, ~128 recording hours, 512 Hz, 10-20 MONOPOLAR montage.
#
# Used to confirm CHB-MIT findings on data the models have never seen. Three
# differences from CHB-MIT, all handled here so the feature pipeline is
# unchanged and results are directly comparable:
#   * monopolar recording -> the same 18 bipolar pairs are derived by subtraction
#   * 512 Hz -> resampled to 256 Hz
#   * seizures annotated in WALL-CLOCK time in Seizures-list-PNxx.txt -> converted
#     to seconds from file start using the EDF header's own start time
SIENA_LABELS = ("interictal", "ictal")


def _siena_bipolar(sigs: dict[str, np.ndarray]) -> np.ndarray | None:
    """Derive CHBMIT_CHANNELS (18 bipolar pairs) from monopolar Siena signals."""
    lut = {}
    for k, v in sigs.items():
        name = k.replace("EEG", "").strip().upper()
        lut[name] = v
    rows = []
    for pair in CHBMIT_CHANNELS:
        a, b = pair.split("-")
        # Siena writes T3/T4/T5/T6 as T7/T8/P7/P8 in some files, and vice versa
        alt = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
               "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
        va = lut.get(a) if a in lut else lut.get(alt.get(a, ""))
        vb = lut.get(b) if b in lut else lut.get(alt.get(b, ""))
        if va is None or vb is None:
            return None
        n = min(len(va), len(vb))
        rows.append(va[:n] - vb[:n])
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows])


def parse_siena_seizures(path) -> dict[str, list[tuple[float, float]]]:
    """{edf name: [(seizure start, end)]} in seconds from that file's start.

    The 14 annotation files use five different layouts, all handled here:

      1. block header "Seizure n 1"                     (PN00, PN03)
      2. header with a trailing colon "Seizure n 1:"    (most patients)
      3. header with a parenthetical, "Seizure n 1 (in sleep):"   (PN13, PN16)
      4. "Start time:" / "End time:" instead of "Seizure start/end time:" (PN01)
      5. "File name:" and "Registration start time:" declared ONCE in a preamble
         before the first block, or omitted from later blocks of the same file

    Times are wall clock, so each file's registration start is subtracted;
    recordings crossing midnight are wrapped. The EDF header's own start time is
    NOT used -- it disagrees with the registration line (every file reports
    01.01.16, PN00-1's registration says 19.39.33).

    One seizure is dropped: PN00-3 lists an end time after its registration end
    (18.57.13 vs 19.29.29, almost certainly a typo for 18.29.29). Guessing the
    intent would be worse than losing 1 of 47 seizures.
    """
    import re

    def secs(tok):
        parts = [q for q in tok.replace(":", ".").split(".") if q.strip().isdigit()]
        if len(parts) < 3:
            return None
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    text = Path(path).read_text(errors="replace")
    HEADER = r"(?im)^[ \t]*Seizure\s+n\s*\d+\b.*$"
    parts = re.split(HEADER, text)
    preamble, blocks = parts[0], parts[1:]

    out: dict[str, list[tuple[float, float]]] = {}
    reg_of: dict[str, int] = {}
    cur_file = None
    m = re.search(r"File name:\s*(\S+?\.edf)", preamble, re.I)
    if m:
        cur_file = m.group(1)
        m2 = re.search(r"Registration start time:\s*([\d.:]+)", preamble, re.I)
        if m2 and secs(m2.group(1)) is not None:
            reg_of[cur_file] = secs(m2.group(1))

    for blk in blocks:
        m_file = re.search(r"File name:\s*(\S+?\.edf)", blk, re.I)
        if m_file:
            cur_file = m_file.group(1)
        m_reg = re.search(r"Registration start time:\s*([\d.:]+)", blk, re.I)
        if m_reg and cur_file and secs(m_reg.group(1)) is not None:
            reg_of[cur_file] = secs(m_reg.group(1))
        # Anchored to line start: an unanchored "start time:" also matches inside
        # "Registration start time:", which silently substitutes the registration
        # time for the seizure time and loses almost every seizure.
        m_s = re.search(r"(?im)^[ \t]*(?:Seizure\s+)?start time:\s*([\d.:]+)", blk)
        m_e = re.search(r"(?im)^[ \t]*(?:Seizure\s+)?end time:\s*([\d.:]+)", blk)
        if not (cur_file and m_s and m_e):
            continue
        t0 = reg_of.get(cur_file)
        s0, s1 = secs(m_s.group(1)), secs(m_e.group(1))
        if t0 is None or s0 is None or s1 is None:
            continue
        rel_s, rel_e = s0 - t0, s1 - t0
        if rel_s < 0:
            rel_s += 86400
            rel_e += 86400
        if rel_e < rel_s:
            rel_e += 86400
        if rel_e - rel_s > 1200:          # see PN00-3 note above
            continue
        out.setdefault(cur_file, []).append((float(rel_s), float(rel_e)))
    return out


def load_siena(
    root: str | Path,
    *,
    n_samples: int = 512,
    interictal_ratio: float = 1.0,
    max_interictal_files: int = 12,
    target_fs: float = MUSE_FS,
    guard_s: float = 600.0,
    seed: int = 0,
    verbose: bool = True,
):
    """Balanced ictal/interictal EpochSet from Siena. Returns (EpochSet, groups)."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    from .edf import read_edf, read_edf_start_seconds
    from .preprocess import normalise

    root = Path(root)
    rng = np.random.default_rng(seed)
    X, y, groups, meta, amp = [], [], [], [], []
    skipped: dict[str, int] = {}

    def skip(w):
        skipped[w] = skipped.get(w, 0) + 1

    def load(path):
        sigs, rates, _ = read_edf(path)
        data = _siena_bipolar(sigs)
        if data is None:
            skip("montage incomplete"); return None, None
        fs_in = float(next(iter(rates.values())))
        if abs(fs_in - target_fs) > 1e-9:
            r = Fraction(target_fs / fs_in).limit_denominator(1000)
            data = resample_poly(data, r.numerator, r.denominator, axis=1)
        return data, read_edf_start_seconds(path)

    def take(data, start, label, case, fname):
        w = data[:, start:start + n_samples]
        if w.shape[1] < n_samples or not np.isfinite(w).all():
            skip("non-finite/short"); return False
        if (w.std(axis=1) < 1e-3).any():
            skip("flat channel"); return False
        chans = [normalise(w[c]) for c in range(w.shape[0])]
        if any(c is None for c in chans):
            skip("flat channel"); return False
        amp.append(np.concatenate([np.log(w.std(axis=1) + 1e-6),
                                   np.log(np.abs(np.diff(w, axis=1)).mean(axis=1) + 1e-6)]))
        X.append(np.stack(chans)); y.append(label); groups.append(case)
        meta.append({"rec": fname, "start": int(start), "case": case})
        return True

    def resolve(case, fname):
        """Annotation names do not always match the files on disk: the published
        set contains PNO6-*.edf (letter O for zero) and PN11-.edf (trailing dash),
        and PN01's annotation says PN01.edf while the file is PN01-1.edf."""
        direct = root / case / fname
        if direct.is_file():
            return direct
        norm = lambda x: x.lower().replace("o", "0").rstrip("-.")
        stem = fname[:-4]
        on_disk = sorted((root / case).glob("*.edf"))
        for p_ in on_disk:                      # exact after normalisation
            if norm(p_.stem) == norm(stem):
                return p_
        cands = [p_ for p_ in on_disk if norm(p_.stem).startswith(norm(stem))]
        if len(cands) == 1:
            return cands[0]
        return on_disk[0] if len(on_disk) == 1 else None

    cases = sorted(p.name for p in root.glob("PN*") if p.is_dir())
    for case in cases:
        lst = list((root / case).glob("Seizures-list-*.txt"))
        ann = parse_siena_seizures(lst[0]) if lst else {}
        n_ictal = 0
        seizure_files = {k for k, v in ann.items() if v}
        for fname in sorted(seizure_files):
            path = resolve(case, fname)
            if path is None:
                skip("annotation names a file not on disk"); continue
            data, _ = load(path)
            if data is None:
                continue
            for s, e in ann[fname]:        # already seconds from file start
                if e - s > 1200:               # >20 min is not a seizure; skip
                    skip("implausible seizure length"); continue
                a0 = int(round(s * target_fs))
                while a0 + n_samples <= int(round(e * target_fs)):
                    n_ictal += take(data, a0, 1, case, fname)
                    a0 += n_samples
        want = int(round(n_ictal * interictal_ratio))
        # Unlike CHB-MIT, nearly every Siena file contains a seizure, so there are
        # almost no seizure-free files to draw from. Interictal windows therefore
        # come from the same files, excluding `guard_s` either side of every
        # seizure (standard practice, and it keeps pre/post-ictal EEG out).
        files = sorted(p.name for p in (root / case).glob("*.edf"))
        rng.shuffle(files)
        got = 0
        for k, fname in enumerate(files[:max_interictal_files]):
            if got >= want:
                break
            data, _ = load(root / case / fname)
            if data is None:
                continue
            res = resolve(case, fname)
            sz = []
            for ann_name, spans in ann.items():
                if resolve(case, ann_name) == res:
                    sz += spans
            avail = []
            for a0 in range(0, data.shape[1] - n_samples, n_samples):
                t_s = a0 / target_fs
                if all(t_s + n_samples / target_fs < s - guard_s or t_s > e + guard_s
                       for s, e in sz):
                    avail.append(a0)
            if not avail:
                continue
            quota = int(np.ceil((want - got) / max(min(len(files), max_interictal_files) - k, 1)))
            pick = rng.choice(np.asarray(avail), size=min(quota, len(avail)), replace=False)
            for a0 in sorted(pick):
                if got >= want:
                    break
                got += take(data, int(a0), 0, case, fname)
        if verbose:
            print(f"[siena] {case}: {n_ictal} ictal, {got} interictal", flush=True)

    if not X:
        raise RuntimeError(f"no usable Siena epochs under {root}: {skipped}")
    es = EpochSet(X=np.stack(X).astype(np.float32), y=np.asarray(y),
                  channels=CHBMIT_CHANNELS, fs=target_fs, label_names=SIENA_LABELS,
                  meta=meta, extra=np.asarray(amp),
                  extra_names=[f"{c}_log_rms" for c in CHBMIT_CHANNELS]
                              + [f"{c}_log_linelength" for c in CHBMIT_CHANNELS])
    if verbose:
        print(f"[siena] total {es.n_epochs} epochs {es.counts()}, "
              f"{len(set(groups))} patients", flush=True)
        if skipped:
            print("[siena] skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()))
    return es, np.asarray(groups)

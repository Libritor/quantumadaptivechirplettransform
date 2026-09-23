#!/usr/bin/env python
"""Raw-signal input for the reservoirs: all 276,480 samples of each 60-s step.

Every experiment so far fed the reservoirs ACT features -- 12 chirplet atoms per
channel per 2 s, a lossy summary. This keeps the raw waveform instead: a 60-s
step is 18 channels x 15,360 samples = 276,480 numbers, reduced by a FIXED
random projection to `--dims` values per step.

A random projection is used rather than PCA on purpose: it is fitted on nothing,
so it cannot leak test information, and the Johnson-Lindenstrauss lemma says it
preserves the geometry of the data. Each channel gets the same projection matrix
(15,360 -> dims_per_channel), so the output keeps per-channel structure.

Amplitude is preserved: channels are scaled by a per-channel constant estimated
on the person's TRAINING half only (first half of files, in time order), not
per-step, so loud steps stay loud.

Output: results/rawproj/<case>.npz with proj (n_steps, 18*dims_per_channel),
valid, ictal, start_s, file_id, files, seizures -- aligned 1:1 with the 60-s
steps of results/continuous/<case>.npz.
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch

from qbe.acquire import CHBMIT_CHANNELS, CHBMIT_FS, parse_chbmit_summary
from qbe.edf import read_edf
from prep_chbmit_continuous import file_order

W_STEP = 30                       # 30 x 2 s = 60 s
SAMPLES = int(W_STEP * 2 * CHBMIT_FS)      # 15,360 per channel per step
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kc/datasets/chbmit")
    ap.add_argument("--out", default="results/rawproj")
    ap.add_argument("--per-channel", type=int, default=32,
                    help="projected dims per channel (18 x this = input width)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = torch.Generator(device=DEV).manual_seed(a.seed)
    # one fixed projection, shared by every channel, person and step
    P = torch.randn(SAMPLES, a.per_channel, generator=g, device=DEV) / np.sqrt(SAMPLES)

    t0 = time.time()
    for case in sorted(p.name for p in root.glob("chb[0-9][0-9]")):
        dst = out / f"{case}.npz"
        if dst.exists():
            print(f"[{case}] exists, skipping", flush=True); continue
        ann = parse_chbmit_summary(root / case / f"{case}-summary.txt")
        files = sorted((p.name for p in (root / case).glob("*.edf")), key=file_order)
        # pass 1: per-channel scale from the training half (first half of files)
        half = max(1, len(files) // 2)
        acc, n_acc = np.zeros(len(CHBMIT_CHANNELS)), 0
        for fname in files[:half]:
            sigs, _, _ = read_edf(root / case / fname)
            if any(c not in sigs for c in CHBMIT_CHANNELS):
                continue
            d = np.stack([sigs[c] for c in CHBMIT_CHANNELS])
            acc += np.nanstd(d, axis=1); n_acc += 1
        scale = (acc / max(n_acc, 1)) + 1e-6
        # pass 2: project every step
        F, V, I, S, FID, names, onsets, skipped = [], [], [], [], [], [], [], []
        for fname in files:
            sigs, _, _ = read_edf(root / case / fname)
            if any(c not in sigs for c in CHBMIT_CHANNELS):
                skipped.append(fname); continue
            d = np.stack([sigs[c] for c in CHBMIT_CHANNELS])
            ns = d.shape[1] // SAMPLES
            if ns == 0:
                continue
            X = d[:, : ns * SAMPLES].reshape(len(CHBMIT_CHANNELS), ns, SAMPLES)
            finite = np.isfinite(X).all(axis=(0, 2))
            flat = (np.nanstd(X, axis=2) < 1e-3).any(axis=0)
            X = np.nan_to_num(X) / scale[:, None, None]
            T = torch.as_tensor(X, dtype=torch.float32, device=DEV)     # (18, ns, S)
            Z = torch.einsum("cns,sk->nck", T, P).reshape(ns, -1)       # (ns, 18*k)
            k = len(names); names.append(fname)
            F.append(Z.cpu().numpy()); V.append(finite & ~flat)
            starts = np.arange(ns) * SAMPLES / CHBMIT_FS
            ict = np.zeros(ns, bool)
            for s, e in ann.get(fname, []):
                ict |= (starts < e) & (starts + SAMPLES / CHBMIT_FS > s)
            I.append(ict); S.append(starts); FID.append(np.full(ns, k, np.int32))
            onsets += [(k, s, e) for s, e in ann.get(fname, [])]
            del T, Z
        np.savez_compressed(
            dst, proj=np.concatenate(F).astype(np.float32), valid=np.concatenate(V),
            ictal=np.concatenate(I), start_s=np.concatenate(S),
            file_id=np.concatenate(FID), files=np.array(names),
            seizures=np.array(onsets, float).reshape(-1, 3), skipped=np.array(skipped))
        n = sum(len(f) for f in F)
        print(f"[{case}] {len(names)} files, {n} steps of 60 s, "
              f"{n * 276480 / 1e9:.1f}G raw values -> {n * 18 * a.per_channel / 1e6:.1f}M projected "
              f"| {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

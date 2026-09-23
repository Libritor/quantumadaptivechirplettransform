#!/usr/bin/env python
"""Continuous CHB-MIT features for sequence models: every 2-s window, in time order.

For each patient case, every EDF is processed in chronological order and every
non-overlapping 2-s window becomes a 270-dim feature vector:
    234 GPU-ACT features (13 per channel x 18) on the unit-energy window, then
     36 amplitude features (log RMS, log line length per channel) from raw uV,
in the same column order as results/names_chbmitamp.json.

Nothing is dropped. Windows with a non-finite sample or a flat (disconnected)
channel are kept in place and flagged invalid, so the time axis stays intact
and downstream models can mask them. Each window is also tagged ictal (overlaps
an annotated seizure) or not.

Output: results/continuous/<case>.npz, resumable (existing cases are skipped).
Files were already verified against PhysioNet's SHA256SUMS in full, so this
script does not re-hash them.
"""
import argparse, os, re, time
from pathlib import Path
import numpy as np
import torch

from qbe import act_gpu
from qbe.acquire import CHBMIT_CHANNELS, CHBMIT_FS, parse_chbmit_summary
from qbe.edf import read_edf
from qbe.gpu_features import _features_from_atoms, gpu_dictionary_grid

N = 512


def file_order(name: str):
    """chb02_16.edf, chb02_16+.edf, chb02_17.edf ... ('+' = continuation file).
    Plain sorting puts '16+' before '16' because '+' < '.', which is wrong."""
    m = re.match(r"chb\d+[a-z]?_(\d+)(\+?)\.edf", name)
    return (int(m.group(1)), m.group(2) == "+") if m else (10**6, False)


def process_file(path, seizures, D, batch):
    sigs, _, _ = read_edf(path)
    if any(c not in sigs for c in CHBMIT_CHANNELS):
        return None
    data = np.stack([sigs[c] for c in CHBMIT_CHANNELS])
    nw = data.shape[1] // N
    W = data[:, : nw * N].reshape(len(CHBMIT_CHANNELS), nw, N).transpose(1, 0, 2)
    finite = np.isfinite(W).all(axis=(1, 2))
    W = np.nan_to_num(W)
    sd = W.std(axis=2)
    valid = finite & (sd >= 1e-3).all(axis=1)
    amp = np.concatenate([np.log(sd + 1e-6),
                          np.log(np.abs(np.diff(W, axis=2)).mean(axis=2) + 1e-6)], axis=1)
    Z = W - W.mean(axis=2, keepdims=True)
    nrm = np.linalg.norm(Z, axis=2, keepdims=True)
    Z = np.where(nrm > 1e-6, Z / np.maximum(nrm, 1e-12), 0.0)
    flat = Z.reshape(-1, N)
    # Only windows that belong to a valid epoch go through ACT; the rest keep
    # zero features and stay flagged invalid, so the timeline is intact.
    use = np.repeat(valid, len(CHBMIT_CHANNELS))
    rows = np.flatnonzero(use)
    act = np.zeros((flat.shape[0], 13), dtype=np.float32)
    sig = torch.as_tensor(flat[rows], dtype=torch.float32)
    for lo in range(0, len(rows), batch):
        Xb = sig[lo:lo + batch].cuda()
        atoms, R = act_gpu.decompose_batch(Xb, D, max_atoms=12, steps=60, omp=True)
        with torch.no_grad():
            act[rows[lo:lo + len(Xb)]] = _features_from_atoms(atoms, R, Xb, CHBMIT_FS, N).cpu().numpy()
    feats = np.concatenate([act.reshape(nw, -1), amp.astype(np.float32)], axis=1)
    starts = np.arange(nw) * N / CHBMIT_FS
    ictal = np.zeros(nw, bool)
    for s, e in seizures:
        ictal |= (starts < e) & (starts + N / CHBMIT_FS > s)
    return feats, valid, ictal, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kc/datasets/chbmit")
    ap.add_argument("--out", default="results/continuous")
    ap.add_argument("--batch", type=int, default=16384)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = act_gpu.GPUDictionary(gpu_dictionary_grid(N, CHBMIT_FS), N, CHBMIT_FS)
    t0 = time.time()
    for case in sorted(p.name for p in root.glob("chb[0-9][0-9]")):
        dst = out / f"{case}.npz"
        if dst.exists():
            print(f"[{case}] exists, skipping", flush=True); continue
        ann = parse_chbmit_summary(root / case / f"{case}-summary.txt")
        files = sorted((p.name for p in (root / case).glob("*.edf")), key=file_order)
        F, V, I, S, FID, names, onsets, skipped = [], [], [], [], [], [], [], []
        tc = time.time()
        for fname in files:
            r = process_file(root / case / fname, ann.get(fname, []), D, a.batch)
            if r is None:
                skipped.append(fname); continue
            feats, valid, ictal, starts = r
            k = len(names); names.append(fname)
            F.append(feats); V.append(valid); I.append(ictal); S.append(starts)
            FID.append(np.full(len(feats), k, np.int32))
            onsets += [(k, s, e) for s, e in ann.get(fname, [])]
        np.savez_compressed(
            dst, feats=np.concatenate(F), valid=np.concatenate(V), ictal=np.concatenate(I),
            start_s=np.concatenate(S), file_id=np.concatenate(FID),
            files=np.array(names), seizures=np.array(onsets, float).reshape(-1, 3),
            skipped=np.array(skipped))
        nw = sum(len(f) for f in F)
        print(f"[{case}] {len(names)} files, {nw} windows ({nw*2/3600:.1f} h), "
              f"{len(onsets)} seizures, {len(skipped)} skipped, {time.time()-tc:.0f}s  "
              f"| total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Sequence data and models: forecast a person's future brain-signal features.

DATA
----
Continuous 2-s windows (prep_chbmit_continuous.py) are averaged into 10-s
steps (a step is valid if >= 3 of its 5 windows are valid). Per person, steps
are split chronologically: the first half trains (its last 15% is held back for
early stopping), the second half tests. chb01 and chb21 are the same person, so
they form one sequence (chb01 first).

A sample is a history of L steps and the next H steps, all within one recording
file (files have gaps between them). From the history the model predicts:
  * the next H steps of all 270 standardised features (forecast), and
  * the probability that a seizure STARTS within the next 5 minutes (risk).
Risk samples are only used where the last history step is neither ictal nor
within 10 min after a seizure ended (post-ictal EEG makes onset trivial).

MODELS (all LSTMs; each quantum model has a classical twin)
-------------------------------------------------------------
  lstm        plain classical LSTM on the 270 features (reference)
  qin_zz      LSTM whose input adds 24 measurements of a fixed ZZ-entangled
              circuit (8 qubits) applied to the step's first 8 principal comps
  qin_z       its twin: identical, but the non-entangling Z circuit, whose
              outputs are just cos/sin of the inputs (classically trivial)
  qlstm       QLSTM (Chen et al. 2020): every LSTM gate passes through a trainable
              6-qubit variational circuit
  qlstm_twin  its twin: each circuit replaced by a classical tanh(Linear) layer
              of the same width

All models predict the forecast as a correction to the last observed step, so
"nothing changes" (persistence) is representable exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .quantum_torch import VQC

STEP_WINDOWS = 5                 # 5 x 2 s = 10-s steps
HISTORY = 30                     # 5 minutes of context
HORIZON = 6                      # forecast the next 1 minute
RISK_S = 300.0                   # seizure onset within the next 5 minutes
POSTICTAL_S = 600.0              # exclude 10 min after a seizure ends from risk


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_person(cont_dir, cases, step_windows=STEP_WINDOWS):
    """Concatenate a person's cases in order into steps of `step_windows` x 2 s.

    step_windows=5 gives the 10-s steps used by the LSTM experiments;
    step_windows=1 keeps the native 2-s resolution (finer temporal structure,
    noisier per-step features)."""
    F, V, I, T, FID, onsets, ends = [], [], [], [], [], [], []
    fid_off = 0
    for case in cases:
        z = np.load(f"{cont_dir}/{case}.npz")
        feats, valid, ictal = z["feats"], z["valid"], z["ictal"]
        start, fid = z["start_s"], z["file_id"]
        thresh = max(1, int(np.ceil(0.6 * step_windows)))
        for k in np.unique(fid):
            m = np.flatnonzero(fid == k)
            ns = len(m) // step_windows
            if ns == 0:
                continue
            idx = m[: ns * step_windows].reshape(ns, step_windows)
            v = valid[idx]
            f = np.where(v[..., None], feats[idx], 0.0).sum(1) / np.maximum(v.sum(1), 1)[:, None]
            F.append(f.astype(np.float32)); V.append(v.sum(1) >= thresh)
            I.append(ictal[idx].any(1)); T.append(start[idx[:, 0]])
            FID.append(np.full(ns, fid_off + k))
        for k, s, e in z["seizures"]:
            onsets.append((fid_off + int(k), s)); ends.append((fid_off + int(k), e))
        fid_off += int(fid.max()) + 1
    return dict(F=np.concatenate(F), valid=np.concatenate(V), ictal=np.concatenate(I),
                t=np.concatenate(T), fid=np.concatenate(FID), onsets=onsets, ends=ends)


def make_samples(P, history=HISTORY, horizon=HORIZON, step_s=STEP_WINDOWS * 2.0):
    """Indices t (last history step) with history+horizon inside one file,
    plus risk labels and eligibility."""
    fid, t = P["fid"], P["t"]
    n = len(fid)
    ok = np.zeros(n, bool)
    for i in range(history - 1, n - horizon):
        a, b = i - history + 1, i + horizon
        ok[i] = fid[a] == fid[b]
    idx = np.flatnonzero(ok)
    tend = t[idx] + step_s
    y = np.zeros(len(idx), np.float32)
    elig = ~P["ictal"][idx]
    for k, s in P["onsets"]:
        m = fid[idx] == k
        y[m & (s > tend) & (s <= tend + RISK_S)] = 1.0
    for k, e in P["ends"]:
        m = fid[idx] == k
        elig &= ~(m & (tend > e) & (tend <= e + POSTICTAL_S))
    return idx, y, elig


def pauli_input_features(Z, pca, lo, hi, kind, n_q=8, scale=0.3):
    """Fixed-circuit measurements (placement A): first n_q principal components,
    min-max scaled with training-half bounds to [0, scale*pi], then the ZZ (or Z)
    feature map, then <X>,<Y>,<Z> per qubit."""
    from . import quantum_fast as qf
    A = np.clip((Z @ pca[:, :n_q] - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * scale * np.pi
    out = np.empty((len(A), 3 * n_q), np.float32)
    for s in range(0, len(A), 20000):
        psi = qf.statevectors(A[s:s + 20000], n_q, 2, "linear", kind)
        out[s:s + 20000] = qf.pauli_expectations(psi, n_q)
    return out


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class GatedCellLSTM(nn.Module):
    """LSTM whose gates pass through a narrow 'core': a variational quantum circuit
    (quantum) or tanh(Linear) of the same width (classical twin).

        v = [x_proj, h];  z = W_in v  (4 gates x n_q);  c = core(z);  g = W_out c
    """

    def __init__(self, d_in, hidden=32, d_proj=16, n_q=6, layers=3, quantum=True):
        super().__init__()
        self.hidden, self.n_q = hidden, n_q
        self.proj = nn.Linear(d_in, d_proj)
        self.w_in = nn.Linear(d_proj + hidden, 4 * n_q)
        if quantum:
            self.core = VQC(n_q, layers, n_circuits=4)
        else:
            self.core_w = nn.Parameter(torch.randn(4, n_q, n_q) / math.sqrt(n_q))
            self.core_b = nn.Parameter(torch.zeros(4, 1, n_q))
            self.core = None
        self.w_out = nn.Parameter(torch.randn(4, n_q, hidden) / math.sqrt(n_q))
        self.b_out = nn.Parameter(torch.zeros(4, 1, hidden))

    def _core(self, z):
        if self.core is not None:
            return self.core(z)
        return torch.tanh(torch.einsum("gbi,gij->gbj", z, self.core_w) + self.core_b)

    def core_params(self):
        return self.core.n_params() if self.core is not None else \
            self.core_w.numel() + self.core_b.numel()

    def forward(self, x):                                   # x (B, L, d_in)
        B, L, _ = x.shape
        h = x.new_zeros(B, self.hidden); c = x.new_zeros(B, self.hidden)
        xp = self.proj(x)
        for s in range(L):
            z = self.w_in(torch.cat([xp[:, s], h], 1)).view(B, 4, self.n_q).transpose(0, 1)
            g = torch.einsum("gbi,gih->gbh", self._core(z), self.w_out) + self.b_out
            i, f, o, u = torch.sigmoid(g[0]), torch.sigmoid(g[1]), torch.sigmoid(g[2]), torch.tanh(g[3])
            c = f * c + i * u
            h = o * torch.tanh(c)
        return h


class SeqModel(nn.Module):
    def __init__(self, kind, d_in, n_feat, horizon=HORIZON):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "qin_zz", "qin_z"):
            self.rnn = nn.LSTM(d_in, 64, batch_first=True); hid = 64
        elif kind in ("qlstm", "qlstm_twin"):
            self.rnn = GatedCellLSTM(d_in, quantum=(kind == "qlstm")); hid = 32
        else:
            raise ValueError(kind)
        self.n_feat, self.horizon = n_feat, horizon
        self.fc = nn.Linear(hid, horizon * n_feat)
        self.risk = nn.Linear(hid, 1)

    def forward(self, x, last):
        h = self.rnn(x)[0][:, -1] if isinstance(self.rnn, nn.LSTM) else self.rnn(x)
        delta = self.fc(h).view(-1, self.horizon, self.n_feat)
        return last[:, None, :] + delta, self.risk(h)[:, 0]

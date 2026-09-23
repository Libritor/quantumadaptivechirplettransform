"""Quantum Adaptive Chirplet Transform (QACT). See docs/QUANTUM_ACT.md.

A chirplet's phase is exactly 2-local in the binary time encoding:

    exp(i 2pi c t^2 / N),  t = sum_k b_k 2^k,  b_k^2 = b_k
      => t^2 = sum_k b_k 2^{2k} + 2 sum_{j<k} b_j b_k 2^{j+k}
      => n single-qubit phase gates + n(n-1)/2 controlled-phase gates, EXACT.

and a shifted chirp needs no extra phase machinery, because
    c(t-tc)^2 + fc(t-tc) = c t^2 + (fc - 2 c tc) t + const,
so `tc` enters only through the Gaussian envelope (a diagonal filter, realised
with one ancilla and post-selection) and the frequency axis is produced in one
shot by a QFT: after the envelope and the conjugate chirp, amplitude f of the
QFT is the correlation <psi_(tc,f,dt,c) | x>, so MEASURING samples atoms with
probability proportional to |<psi|x>|^2 -- exactly what matching pursuit wants.

`verify_gates` checks the decomposition against the diagonal it should produce
and against qiskit. `QACT.transform` runs the full matching pursuit with
sampling-based (stochastic) atom selection and returns atoms in the same form as
the classical engines, so features.channel_features works unchanged.

Selection is stochastic by construction: with `shots` per dictionary triple an
atom is chosen by sampling, converging to the classical argmax as shots -> inf.
`post_select_frac` reports the envelope filter's success probability, which is
the hardware measurement overhead a real device would pay.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .features import PER_CHANNEL_FEATURES  # noqa: F401  (feature contract)


# ---------------------------------------------------------------------------
# gate-level pieces (exactness checks; the batched transform uses their action)
# ---------------------------------------------------------------------------
def quadratic_phase_gates(n: int, c: float, N: int | None = None):
    """Gate list for exp(i 2pi c t^2 / N): [('p', k, angle)] and [('cp', j, k, angle)]."""
    N = N or 2 ** n
    single = [("p", k, 2 * math.pi * c * (2 ** (2 * k)) / N) for k in range(n)]
    pairs = [("cp", j, k, 2 * math.pi * c * (2 ** (j + k + 1)) / N)
             for j in range(n) for k in range(j + 1, n)]
    return single + pairs


def cubic_phase_gates(n: int, c3: float, N: int | None = None):
    """Gate list for exp(i 2pi c3 t^3 / N) -- exact, and 3-local.

    t^3 = (sum_k b_k 2^k)^3 = sum_{i,j,k} b_i b_j b_k 2^{i+j+k}, and b^2 = b, so
    grouping by how many indices coincide:

        i=j=k          ->  sum_i b_i 2^{3i}                       (1-qubit phase)
        exactly two    ->  3 sum_{i != k} b_i b_k 2^{2i+k}        (2-qubit phase)
        all distinct   ->  6 sum_{i<j<k} b_i b_j b_k 2^{i+j+k}    (3-qubit phase)

    Cost n + n(n-1)/2 + n(n-1)(n-2)/6 gates -- 129 at n=9. A cubic phase is an
    atom whose chirp RATE itself evolves, the quantum-native analogue of the
    classical v10/v11 families.
    """
    N = N or 2 ** n
    k2 = 2 * math.pi * c3 / N
    single = [("p", k, k2 * (2 ** (3 * k))) for k in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # both orderings of the "exactly two coincide" term act on {i, j}
            w = 3 * (2 ** (2 * i + j) + 2 ** (2 * j + i))
            pairs.append(("cp", i, j, k2 * w))
    triples = [("ccp", i, j, k, k2 * 6 * (2 ** (i + j + k)))
               for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    return single + pairs + triples


def linear_phase_gates(n: int, f: float, N: int | None = None):
    """Gate list for exp(i 2pi f t / N)."""
    N = N or 2 ** n
    return [("p", k, 2 * math.pi * f * (2 ** k) / N) for k in range(n)]


def apply_gate_list(diag: np.ndarray, gates, n: int) -> np.ndarray:
    """Apply a phase-gate list to a diagonal (length 2^n) of unit amplitudes."""
    idx = np.arange(len(diag))
    bits = [((idx >> k) & 1) for k in range(n)]
    out = diag.copy()
    for g in gates:
        if g[0] == "p":
            _, k, a = g
            out = out * np.exp(1j * a * bits[k])
        elif g[0] == "cp":
            _, j, k, a = g
            out = out * np.exp(1j * a * (bits[j] & bits[k]))
        else:
            _, i, j, k, a = g
            out = out * np.exp(1j * a * (bits[i] & bits[j] & bits[k]))
    return out


def verify_gates(n: int = 6, c: float = 3.7, f: float = 11.3, c3: float = 0.9) -> dict:
    """Check the decompositions against the diagonals they must equal, and qiskit."""
    N = 2 ** n
    t = np.arange(N)
    ref_q = np.exp(1j * 2 * np.pi * c * t ** 2 / N)
    ref_l = np.exp(1j * 2 * np.pi * f * t / N)
    ref_c = np.exp(1j * 2 * np.pi * c3 * t ** 3 / N)
    got_q = apply_gate_list(np.ones(N, complex), quadratic_phase_gates(n, c, N), n)
    got_l = apply_gate_list(np.ones(N, complex), linear_phase_gates(n, f, N), n)
    got_c = apply_gate_list(np.ones(N, complex), cubic_phase_gates(n, c3, N), n)
    out = {"quadratic_vs_formula": float(np.abs(got_q - ref_q).max()),
           "linear_vs_formula": float(np.abs(got_l - ref_l).max()),
           "cubic_vs_formula": float(np.abs(got_c - ref_c).max())}
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(n)
        qc.h(range(n))                                   # uniform superposition
        for g in (quadratic_phase_gates(n, c, N) + linear_phase_gates(n, f, N)
                  + cubic_phase_gates(n, c3, N)):
            if g[0] == "p":
                qc.p(g[2], g[1])
            elif g[0] == "cp":
                qc.cp(g[3], g[1], g[2])
            else:
                qc.mcp(g[4], [g[1], g[2]], g[3])         # 3-qubit controlled phase
        sv = np.asarray(Statevector(qc).data)
        want = (ref_q * ref_l * ref_c) / math.sqrt(N)
        out["vs_qiskit"] = float(np.abs(sv - want).max())
        out["gate_counts"] = dict(
            quadratic=len(quadratic_phase_gates(n, c, N)),
            cubic=len(cubic_phase_gates(n, c3, N)))
    except Exception as exc:                              # pragma: no cover
        out["vs_qiskit"] = f"unavailable: {type(exc).__name__}"
    return out


# ---------------------------------------------------------------------------
# batched QACT
# ---------------------------------------------------------------------------
class QACTAtom:
    """Same fields the classical engines emit, so features.channel_features works."""

    __slots__ = ("tc", "fc", "logDt", "c", "coeff_real", "coeff_imag")

    def __init__(self, tc, fc, logDt, c, coeff_real, coeff_imag):
        self.tc, self.fc, self.logDt, self.c = tc, fc, logDt, c
        self.coeff_real, self.coeff_imag = coeff_real, coeff_imag


class QACT:
    """Matching pursuit with quantum-sampled atom selection.

    The dictionary is (tc, logDt, chirp rate) x every QFT frequency bin. Chirp
    rates are given in PHYSICAL Hz/s and converted internally: the sample-domain
    coefficient is c = rate * N / (2 * fs**2), so the phase 2pi c t^2 / N matches
    an instantaneous frequency sweeping at `rate`. (Specifying the grid in
    sample-domain units is how the first draft ended up with +-3072 Hz/s atoms,
    none of which describe EEG.)

    The QFT returns the EFFECTIVE frequency f_eff, and the atom's centre
    frequency is fc = f_eff + 2*c*tc, which can fall outside the band or above
    Nyquist. Such atoms are aliases, not signals, so they are masked out of the
    dictionary before sampling.
    """

    def __init__(self, length=512, fs=256.0, tc=None, logDt=None, rates_hz_s=None,
                 band_hz=(0.5, 45.0), shots=256, device="cuda", seed=0,
                 refine_steps=0, rates3_hz_s2=None, skews=None, n_candidates=1,
                 omp=True, backfit_passes=1, min_amp=0.0, exact_f=True,
                 asym_ratios=None, refine_extra=False, norm_select=True,
                 exact_ls=False):
        self.N = int(length)
        self.n = int(round(math.log2(self.N)))
        assert 2 ** self.n == self.N, "length must be a power of two (log2 N qubits)"
        self.fs = float(fs)
        self.tc = np.asarray(tc if tc is not None else np.arange(0, self.N, 48), float)
        self.logDt = np.asarray(logDt if logDt is not None else [1.5, 2.7, 3.9, 5.1], float)
        self.rates = np.asarray(rates_hz_s if rates_hz_s is not None
                                else [-40, -20, -8, 0, 8, 20, 40], float)
        self.c = self.rates * self.N / (2 * self.fs ** 2)      # sample-domain chirp
        # CUBIC phase: chirp rate that itself evolves, in Hz/s^2. Instantaneous
        # frequency is (3 c3 t^2 + 2 c t + f) fs / N, so c3 = rate3 N / (6 fs^3).
        self.rates3 = np.asarray(rates3_hz_s2 if rates3_hz_s2 is not None else [0.0], float)
        self.c3 = self.rates3 * self.N / (6 * self.fs ** 3)
        # SKEW envelope: Gaussian x (1 + erf(skew d / (sqrt2 Dt))), a smooth
        # onset/decay asymmetry; skew = 0 is the symmetric Gaussian bit-for-bit.
        self.skews = np.asarray(skews if skews is not None else [0.0], float)
        self.n_candidates = int(n_candidates)
        # --- functional parity with the classical engine ---
        # omp            re-fit EVERY selected coefficient after each new atom, by
        #                least squares. The Gram matrix entries are the atom-atom
        #                overlaps <psi_i|psi_j>, i.e. exactly what a swap test
        #                measures, so this reuses the fidelity primitive; the k x k
        #                solve itself is classical because k <= order is tiny.
        # backfit_passes cyclic coordinate descent over the selected atoms: each is
        #                re-refined against the residual with its own contribution
        #                added back, then all coefficients are re-fitted.
        # min_amp        per-window stopping rule, as the classical engine's
        #                active mask (0 disables, matching how it was run here).
        # exact_f        the objective as a function of f is |sum env x e^{-2pi i f t/N}|^2,
        #                the periodogram -- one QFT gives every f, so the optimal f
        #                is read off directly instead of gradient-stepped, with
        #                parabolic interpolation for sub-bin precision.
        self.omp = bool(omp)
        self.backfit_passes = int(backfit_passes)
        self.min_amp = float(min_amp)
        self.exact_f = bool(exact_f)
        # asym_ratios   right/left width ratio of the envelope, the classical
        #               engine's ACTv9Asym two-width Gaussian: width dt for
        #               t < tc, dt * ratio for t >= tc. [1.0] is symmetric.
        # refine_extra  also refine the parameters that were carried as fixed
        #               grid context -- c3 by the parameter-shift rule (its gates
        #               are 3-local, so singles, pairs AND triples contribute),
        #               skew and the width ratio by central differences.
        self.asym = np.asarray(asym_ratios if asym_ratios is not None else [1.0], float)
        self.refine_extra = bool(refine_extra)
        # norm_select   condition the measured distribution on the envelope
        #               filter succeeding, i.e. divide |<psi|r>|^2 by ||psi||^2.
        #               The raw histogram is the JOINT probability of (filter
        #               succeeds, frequency k), so it over-ranks wide, high-norm
        #               envelopes; the conditional distribution is what an
        #               experiment keeping only post-selected shots reports, and
        #               it is the same quantity the refiner maximizes and the
        #               classical engine's least-squares criterion normalizes by.
        self.norm_select = bool(norm_select)
        # exact_ls      the classical engine scores an atom by the EXACT 2-D least
        #               squares over its quadrature pair, which needs <gc,gc>,
        #               <gs,gs> and the cross term <gc,gs>. All three follow from
        #               one quantity: the atom's second-harmonic self-overlap
        #               H = <psi|psi*> = sum env^2 e^{-2i phi}, a swap test between
        #               the state and its complex conjugate. H does not depend on
        #               the data, so it is a table built once at construction and
        #               applied to the measured counts as post-processing.
        #               MEASURED: on planted chirplets this changes reconstruction
        #               error by <1e-4 (0.17233 -> 0.17221) for 1.8x the search
        #               cost, because an oscillating atom's cos and sin
        #               quadratures are already near-orthogonal and equal-norm.
        #               Implemented for completeness; off by default for that
        #               reason, not because it is unavailable.
        self.exact_ls = bool(exact_ls)
        self.shots, self.dev = int(shots), device
        self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)
        self.fmax = self.N // 2
        t = torch.arange(self.N, device=device, dtype=torch.float32)
        self.t = t
        tcs = torch.as_tensor(self.tc, device=device, dtype=torch.float32)
        lds = torch.as_tensor(self.logDt, device=device, dtype=torch.float32)
        sks = torch.as_tensor(self.skews, device=device, dtype=torch.float32)
        ars = torch.as_tensor(self.asym, device=device, dtype=torch.float32)
        # envelope grid over (tc, logDt, skew, width ratio). With a single unit
        # ratio the flattened order is identical to the three-way grid.
        TC, LD, SK, AR = torch.meshgrid(tcs, lds, sks, ars, indexing="ij")
        self.env_tc, self.env_logDt, self.env_skew, self.env_asym = (
            TC.reshape(-1), LD.reshape(-1), SK.reshape(-1), AR.reshape(-1))
        self.env = self._envelope(self.env_tc, self.env_logDt, self.env_skew,
                                  self.env_asym)
        # ||psi||^2 per envelope = the envelope filter's success probability
        self.env_norm2 = (self.env ** 2).sum(1).clamp(min=1e-12)
        # chirp grid over (c, c3)
        C2, C3 = torch.meshgrid(torch.as_tensor(self.c, device=device, dtype=torch.float32),
                                torch.as_tensor(self.c3, device=device, dtype=torch.float32),
                                indexing="ij")
        self.c_t, self.c3_t = C2.reshape(-1), C3.reshape(-1)
        # Float-multiplication is not associative: grouping this as
        # (-2*pi*i*(c t^2 + c3 t^3))/N instead of ((-2*pi*i*c) t^2)/N changes the
        # last bits of the phase even when c3 == 0 exactly, and because selection
        # is stochastic sampling that flips ~7% of draws. Keep the original
        # expression whenever there is no cubic term, so the verified reference
        # configuration stays bit-for-bit reproducible.
        if not np.any(self.c3):
            self.chirp = torch.exp(-1j * 2 * math.pi * self.c_t[:, None]
                                   * (t[None, :] ** 2) / self.N)
        else:
            self.chirp = torch.exp(-1j * 2 * math.pi
                                   * (self.c_t[:, None] * (t[None, :] ** 2)
                                      + self.c3_t[:, None] * (t[None, :] ** 3)) / self.N)
        # band mask over (E, C, F). With a cubic term the centre frequency is
        # f_bin + 2 c tc + 3 c3 tc^2 (the instantaneous frequency at tc).
        f_bin = torch.arange(self.fmax, device=device, dtype=torch.float32)
        fc = (f_bin[None, None, :]
              + 2 * self.c_t[None, :, None] * self.env_tc[:, None, None]
              + 3 * self.c3_t[None, :, None] * self.env_tc[:, None, None] ** 2) \
            * self.fs / self.N
        self.mask = ((fc >= band_hz[0]) & (fc <= band_hz[1])).float()
        self.dict_size = int(self.mask.sum().item())
        self.post_select_frac = None
        # the exact-f update is judged with the refiner's energy functional, so it
        # only applies when refinement is enabled
        self.exact_f = self.exact_f and refine_steps > 0
        # Refinement stays inside the dictionary's declared support: c3, skew and
        # the width ratio are bounded by the extent of their own grids, so
        # off-grid polishing never invents an atom family the dictionary did not
        # claim to cover.
        self.refiner = (QACTRefiner(
            self.N, self.fs, device=device, steps=refine_steps, band_hz=band_hz,
            c3_max=(float(np.abs(self.c3).max()) if self.refine_extra else 0.0),
            skew_max=(float(np.abs(self.skews).max()) if self.refine_extra else 0.0),
            ratio_range=((float(self.asym.min()), float(self.asym.max()))
                         if self.refine_extra else (1.0, 1.0))
        ) if refine_steps else None)
        self.refine_evals = 0
        # (envelope, chirp) pairs with no in-band frequency at all contribute
        # nothing but still cost a QFT, so they are skipped. The flat dictionary
        # layout is unchanged (their slots stay zero), so measurement sampling
        # sees exactly the same distribution.
        pair_ok = (self.mask.sum(-1) > 0)
        ei, ci = torch.nonzero(pair_ok, as_tuple=True)
        self.pair_e, self.pair_c = ei, ci
        # env and chirp are applied separately, in the same order as the reference
        # implementation: (X*env)*chirp. Folding them into one product would save
        # a negligible multiply but change floating-point rounding, and because
        # atom selection is stochastic that flips a minority of draws.
        self.pair_env = self.env[ei].contiguous()
        self.pair_chirp = self.chirp[ci].contiguous()
        self.n_pairs_used = int(len(ei))
        if self.exact_ls:
            # H(f) = sum_t env^2 e^{-2i phi(t)} is the FFT of env^2 * chirp^2
            # sampled at bin 2f, so the whole table is one transform per pair and
            # costs nothing per iteration -- it does not involve the data.
            H = torch.fft.fft((self.pair_env ** 2).to(torch.complex64)
                              * self.pair_chirp ** 2, dim=-1)
            two_f = (2 * torch.arange(self.fmax, device=device)) % self.N
            H = H[:, two_f]                                    # (pairs, fmax)
            E2 = (self.pair_env ** 2).sum(1)[:, None]           # ||psi||^2
            self.ls_cc = (E2 + H.real) / 2
            self.ls_ss = (E2 - H.real) / 2
            self.ls_cs = -H.imag / 2
            self.ls_det = self.ls_cc * self.ls_ss - self.ls_cs ** 2
            self.ls_degen = self.ls_det <= 1e-12 * (E2 ** 2)

    def _envelope(self, tc, logdt, skew, ratio=None):
        """Gaussian envelope with two optional asymmetries.

        `ratio` gives the right half a different width (the classical two-width
        form); `skew` multiplies by a skew-normal factor. Both neutral values
        return the plain Gaussian by the identical expression, which the bitwise
        reference configuration depends on.
        """
        d = self.t[None, :] - tc[:, None]
        w = torch.exp(logdt)[:, None]
        g = torch.exp(-(d ** 2) / (2 * w ** 2))
        if ratio is not None and torch.count_nonzero(ratio - 1.0) > 0:
            wr = w * ratio[:, None]
            g = torch.where(d >= 0, torch.exp(-(d ** 2) / (2 * wr ** 2)), g)
        if torch.count_nonzero(skew) == 0:
            return g
        return g * (1.0 + torch.erf(skew[:, None] * d / (math.sqrt(2.0) * w)))

    def _power(self, R, pair_chunk=128):
        """|<psi|r>|^2 over (E, C, F), masked to in-band atoms. R: (B, N).

        Only (envelope, chirp) pairs that contain at least one in-band frequency
        are transformed; the rest stay zero. env*chirp is precomputed, so each
        pair costs one complex multiply and one simulated QFT.
        """
        B = R.shape[0]
        X = R.to(torch.complex64)
        E, C = self.env.shape[0], self.chirp.shape[0]
        P = torch.zeros(B, E, C, self.fmax, device=self.dev)
        for lo in range(0, self.n_pairs_used, pair_chunk):
            sl = slice(lo, lo + pair_chunk)
            z = (X[:, None, :] * self.pair_env[None, sl, :]) * self.pair_chirp[None, sl, :]
            F = torch.fft.fft(z, dim=-1)[..., :self.fmax]
            if self.exact_ls:
                # exact 2-D least squares over the real quadrature pair, the same
                # criterion act_gpu.GPUDictionary.best uses
                bc, bs = F.real, -F.imag
                cc, ss = self.ls_cc[None, sl], self.ls_ss[None, sl]
                cs, det = self.ls_cs[None, sl], self.ls_det[None, sl]
                E = ((ss * bc ** 2 - 2 * cs * bc * bs + cc * bs ** 2)
                     / torch.where(self.ls_degen[None, sl], torch.ones_like(det), det))
                E = torch.where(self.ls_degen[None, sl],
                                bc ** 2 / cc.clamp(min=1e-30), E)
                P[:, self.pair_e[sl], self.pair_c[sl]] = torch.nan_to_num(E, nan=0.0)
            else:
                P[:, self.pair_e[sl], self.pair_c[sl]] = F.abs() ** 2
        # envelope filter success probability (the post-selection overhead)
        w = (self.env[None] * X[:, None, :]).abs() ** 2
        succ = float((w.sum(-1) / (X.abs() ** 2).sum(-1, keepdim=True)
                      .clamp(min=1e-12)).mean().item())
        if self.norm_select and not self.exact_ls:
            # exact_ls already normalises by the atom's own Gram
            P = P / self.env_norm2[None, :, None, None]
        return P * self.mask[None], succ

    def _joint_refit(self, X, bases):
        """OMP step: least-squares re-fit of every selected atom's quadrature pair.

        `bases` is a list of (B, N) tensors. Returns (coef (B, 2k), residual (B, N)).
        The normal-equation matrix is the Gram of atom overlaps -- the swap-test
        observable -- with a ridge and a least-squares fallback for the degenerate
        systems duplicate atoms produce.
        """
        A = torch.stack(bases, 2)                              # (B, N, 2k)
        G = A.transpose(1, 2) @ A
        rhs = A.transpose(1, 2) @ X[:, :, None]
        eye = torch.eye(G.shape[1], device=X.device, dtype=X.dtype)
        ridge = 1e-8 * torch.diagonal(G, dim1=1, dim2=2).amax(1).clamp(min=1e-12)
        coef, info = torch.linalg.solve_ex(G + ridge[:, None, None] * eye, rhs)
        bad = info != 0
        if bool(bad.any()):
            coef[bad] = torch.linalg.lstsq(A[bad], X[bad][:, :, None]).solution
        return coef[:, :, 0], X - (A @ coef)[:, :, 0]

    def _accept_f(self, R, tc, logdt, f, c, c3, skew, ratio=None):
        """Take the periodogram-optimal f only where it raises captured energy."""
        f_new = self._exact_f(R, tc, logdt, c, c3, skew, ratio)
        e_old = self.refiner.energy(R, tc, logdt, f, c, c3=c3, skew=skew, ratio=ratio)
        e_new = self.refiner.energy(R, tc, logdt, f_new, c, c3=c3, skew=skew, ratio=ratio)
        return torch.where(e_new > e_old, f_new, f)

    def _exact_f(self, R, tc, logdt, c, c3, skew, ratio=None):
        """Optimal frequency for fixed envelope and chirp, straight from one QFT.

        E(f) = |sum_t env(t) r(t) e^{-i 2pi f t / N}|^2 is the periodogram, so the
        whole f axis comes from a single transform; a 3-point parabolic fit around
        the peak gives sub-bin resolution.
        """
        env = self._envelope(tc, logdt, skew, ratio)
        t = self.t[None, :]
        ph = 2 * math.pi * (c3[:, None] * t ** 3 + c[:, None] * t ** 2) / self.N
        z = (env * R).to(torch.complex64) * torch.exp(-1j * ph)
        P = torch.fft.fft(z, dim=-1)[:, :self.fmax].abs() ** 2
        k = P.argmax(1)
        km1 = (k - 1).clamp(min=0); kp1 = (k + 1).clamp(max=self.fmax - 1)
        rows = torch.arange(len(R), device=R.device)
        y0, y1, y2 = P[rows, km1], P[rows, k], P[rows, kp1]
        # standard three-point parabolic peak interpolation. At a peak the
        # denominator is NEGATIVE, so it must not be clamped by magnitude -- only
        # the degenerate (flat) case needs guarding.
        den = y0 - 2 * y1 + y2
        safe = torch.where(den.abs() > 1e-20, den, torch.full_like(den, 1e-20))
        delta = torch.where(den.abs() > 1e-20, 0.5 * (y0 - y2) / safe,
                            torch.zeros_like(den))
        return k.to(torch.float32) + delta.clamp(-0.5, 0.5)

    def rebuild(self, raw):
        """Sum the atom waveforms named by `raw` exactly as they were fitted.

        Columns: tc, logDt, f, c, c3, skew, a, b, and a 9th width ratio when the
        asymmetric envelope is in use. Returns (M, N).
        """
        T = lambda col: torch.as_tensor(raw[:, col], dtype=torch.float32, device=self.dev)
        ratio = T(8) if raw.shape[1] > 8 else None
        bc, bs, _ = self._atom_basis_cont(T(0), T(1), T(2), T(3), c3=T(4), skew=T(5),
                                          ratio=ratio)
        return T(6)[:, None] * bc + T(7)[:, None] * bs

    def _atom_basis_cont(self, tc, logdt, f, c, c3=None, skew=None, ratio=None):
        """Quadrature pair for continuous (possibly refined) parameters."""
        z = torch.zeros_like(tc)
        env = self._envelope(tc, logdt, z if skew is None else skew, ratio)
        t = self.t[None, :]
        if c3 is None or not bool(torch.any(c3)):       # see note on associativity
            phase = 2 * math.pi * (c[:, None] * t ** 2 + f[:, None] * t) / self.N
        else:
            phase = 2 * math.pi * (c3[:, None] * t ** 3 + c[:, None] * t ** 2
                                   + f[:, None] * t) / self.N
        return env * torch.cos(phase), env * torch.sin(phase), c

    def _atom_basis(self, e, c_i, f_i):
        """Real quadrature pair of the selected atom, (B, N) each."""
        c_val = self.c_t[c_i][:, None]
        phase = 2 * math.pi * (c_val * self.t[None, :] ** 2
                               + f_i.to(torch.float32)[:, None] * self.t[None, :]) / self.N
        env = self.env[e]
        return env * torch.cos(phase), env * torch.sin(phase), c_val[:, 0]

    def transform(self, signal, order=12, emit_certificate=True, select="sample",
                  return_params=False, return_raw=False):
        """Atom selection, three modes:

        "sample"    quantum measurement -- atoms drawn ~ |<psi|r>|^2, modal
                    outcome of `shots` shots.
        "argmax"    the deterministic classical choice on the SAME dictionary --
                    the control isolating measurement from dictionary.
        "proposal"  quantum measurement used as a PROPOSAL distribution: draw
                    `n_candidates` distinct atoms, refine each by parameter
                    shift, and keep whichever has the highest REFINED captured
                    energy. Selecting on the same quantity that was sampled
                    could never beat argmax (argmax is that quantity's maximum);
                    selecting on the refined energy can, because greedy
                    correlation is myopic -- the best atom before refinement is
                    not always the best after it. Requires refine_steps > 0.
        """
        if select == "proposal" and self.refiner is None:
            raise ValueError("select='proposal' needs refine_steps > 0")
        single = signal.ndim == 1
        R0 = torch.as_tensor(np.atleast_2d(signal), dtype=torch.float32, device=self.dev)
        R = R0.clone()
        B = R.shape[0]
        picks = [[] for _ in range(B)]
        sel = []                 # per atom: (tc, logDt, f, c, c3, skew, c_val)
        bases = []               # per atom, two entries: cos and sin quadratures
        coefs = []               # per atom: (a, b); replaced wholesale by OMP
        succ_hist = []
        active = torch.ones(B, dtype=torch.bool, device=self.dev)
        rows = torch.arange(B, device=self.dev)
        for _ in range(order):
            P, succ = self._power(R)
            succ_hist.append(succ)
            flat = P.reshape(B, -1)
            tot = flat.sum(1, keepdim=True)
            if float(tot.min()) <= 0:
                break
            C, F = P.shape[2], P.shape[3]

            def decode(idx):
                e_ = idx // (C * F); ci_ = (idx % (C * F)) // F; fi_ = idx % F
                return (self.env_tc[e_], self.env_logDt[e_], self.env_skew[e_],
                        fi_.to(torch.float32), self.c_t[ci_], self.c3_t[ci_],
                        self.env_asym[e_])

            if select == "argmax":
                best = flat.argmax(dim=1)
                tc_sel, ld_sel, sk_sel, f_sel, c_sel, c3_sel, ar_sel = decode(best)
            elif select == "proposal":
                # k distinct candidates per row, drawn from the measurement
                # distribution, then refined and judged on refined energy
                k = self.n_candidates
                cand = torch.multinomial(flat / tot, k, replacement=False,
                                         generator=self.gen)            # (B, k)
                pr = [x.reshape(-1) for x in decode(cand.reshape(-1))]
                Rk = R.repeat_interleave(k, dim=0)
                tcf, ldf, ff, cf = self.refiner.refine(Rk, pr[0], pr[1], pr[3], pr[4],
                                                       c3=pr[5], skew=pr[2], ratio=pr[6])
                self.refine_evals += k * self.refiner.steps * self.refiner.evals_per_step
                en = self.refiner.energy(Rk, tcf, ldf, ff, cf, c3=pr[5], skew=pr[2],
                                         ratio=pr[6])
                pick = en.view(B, k).argmax(1) + torch.arange(B, device=self.dev) * k
                tc_sel, ld_sel, f_sel, c_sel = (tcf[pick], ldf[pick], ff[pick], cf[pick])
                sk_sel, c3_sel, ar_sel = pr[2][pick], pr[5][pick], pr[6][pick]
            else:
                samp = torch.multinomial(flat / tot, self.shots, replacement=True,
                                         generator=self.gen)
                best = torch.mode(samp, dim=1).values
                tc_sel, ld_sel, sk_sel, f_sel, c_sel, c3_sel, ar_sel = decode(best)
            if self.refiner is not None and select != "proposal":
                if self.refine_extra:
                    (tc_sel, ld_sel, f_sel, c_sel, c3_sel, sk_sel, ar_sel
                     ) = self.refiner.refine(R, tc_sel, ld_sel, f_sel, c_sel,
                                             c3=c3_sel, skew=sk_sel, ratio=ar_sel,
                                             full=True)
                    self.refine_evals += (self.refiner.steps
                                          * self.refiner.evals_per_step_full)
                else:
                    # the width ratio is fixed context here, but it MUST be seen:
                    # refining against a symmetric envelope and then building the
                    # atom with an asymmetric one fits the wrong objective
                    tc_sel, ld_sel, f_sel, c_sel = self.refiner.refine(
                        R, tc_sel, ld_sel, f_sel, c_sel, c3=c3_sel, skew=sk_sel,
                        ratio=ar_sel)
                    self.refine_evals += self.refiner.steps * self.refiner.evals_per_step
            if self.exact_f:
                f_sel = self._accept_f(R, tc_sel, ld_sel, f_sel, c_sel, c3_sel, sk_sel,
                                       ar_sel)
            bc, bs, c_val = self._atom_basis_cont(tc_sel, ld_sel, f_sel, c_sel,
                                                  c3=c3_sel, skew=sk_sel, ratio=ar_sel)
            # classical `keep & (E > 0)`: an atom that captures nothing ends the
            # window rather than being appended with a zero coefficient
            active = active & (((R * bc).sum(1) ** 2 + (R * bs).sum(1) ** 2) > 0)
            if not bool(active.all()):        # stopping rule: frozen rows take no atom
                bc = bc * active[:, None]; bs = bs * active[:, None]
            sel.append((tc_sel, ld_sel, f_sel, c_sel, c3_sel, sk_sel, c_val, ar_sel))
            if self.omp:
                # OMP: re-fit EVERY selected coefficient against the signal. The
                # normal matrix is the Gram of atom overlaps <psi_i|psi_j> -- the
                # swap-test observable -- so no primitive beyond the ones the
                # circuit already provides is needed.
                bases += [bc, bs]
                coef, R = self._joint_refit(R0, bases)
                coefs = [(coef[:, 2 * i], coef[:, 2 * i + 1])
                         for i in range(len(bases) // 2)]
                a, b = coefs[-1]
            else:
                # plain matching pursuit: project once, never revisit
                a = torch.zeros(B, device=self.dev); b = torch.zeros(B, device=self.dev)
                for basis, coef in ((bc, a), (bs, b)):
                    nrm = basis.norm(dim=1).clamp(min=1e-12)
                    u = basis / nrm[:, None]
                    proj = (R * u).sum(1)
                    coef.copy_(proj / nrm)
                    R = R - proj[:, None] * u
                bases += [bc, bs]
                coefs.append((a, b))
            if self.min_amp > 0:
                amp = torch.sqrt(a ** 2 + b ** 2)
                active = active & (amp >= self.min_amp * R0.norm(dim=1)
                                   / math.sqrt(self.N))
        # ---- backfit: cyclic coordinate descent over the chosen atoms ----
        # Greedy selection fixes each atom against a residual that later atoms
        # then change. A backfit pass re-refines every atom against the residual
        # with its own contribution added back, then re-fits all coefficients.
        if self.refiner is not None and self.omp and self.backfit_passes > 0 and sel:
            for _ in range(self.backfit_passes):
                for i in range(len(sel)):
                    tc_i, ld_i, f_i, c_i, c3_i, sk_i, _, ar_i = sel[i]
                    ai, bi = coefs[i]
                    Ri = R + ai[:, None] * bases[2 * i] + bi[:, None] * bases[2 * i + 1]
                    if self.refine_extra:
                        (tc_i, ld_i, f_i, c_i, c3_i, sk_i, ar_i) = self.refiner.refine(
                            Ri, tc_i, ld_i, f_i, c_i, c3=c3_i, skew=sk_i, ratio=ar_i,
                            full=True)
                        self.refine_evals += (self.refiner.steps
                                              * self.refiner.evals_per_step_full)
                    else:
                        tc_i, ld_i, f_i, c_i = self.refiner.refine(
                            Ri, tc_i, ld_i, f_i, c_i, c3=c3_i, skew=sk_i, ratio=ar_i)
                        self.refine_evals += (self.refiner.steps
                                              * self.refiner.evals_per_step)
                    if self.exact_f:
                        f_i = self._accept_f(Ri, tc_i, ld_i, f_i, c_i, c3_i, sk_i, ar_i)
                    bc, bs, c_val = self._atom_basis_cont(tc_i, ld_i, f_i, c_i,
                                                          c3=c3_i, skew=sk_i, ratio=ar_i)
                    sel[i] = (tc_i, ld_i, f_i, c_i, c3_i, sk_i, c_val, ar_i)
                    bases[2 * i], bases[2 * i + 1] = bc, bs
                    coef, R = self._joint_refit(R0, bases)
                    coefs = [(coef[:, 2 * j], coef[:, 2 * j + 1])
                             for j in range(len(bases) // 2)]

        # ---- emit ----
        # One device->host transfer per atom. Reading these tensors element by
        # element cost ~6 GPU syncs per atom per row, which profiling showed to
        # be about two thirds of total runtime.
        param_rows, raw_rows = [], []
        asym_used = self.asym.size > 1 or self.refine_extra
        for i, (tc_sel, ld_sel, f_sel, c_sel, c3_sel, sk_sel, c_val, ar_sel) in enumerate(sel):
            a, b = coefs[i]
            if return_raw:
                cols = [tc_sel, ld_sel, f_sel, c_sel, c3_sel, sk_sel, a, b]
                if asym_used:                 # 9th column: envelope width ratio
                    cols.append(ar_sel)
                raw_rows.append(torch.stack(cols, 1).cpu().numpy())
            rec = torch.stack([tc_sel, ld_sel, f_sel, c_val, a, b, c3_sel], 1).cpu().numpy()
            # promote before deriving fc: the per-row path computed it in Python
            # float64, and doing it in float32 here shifted features by ~4e-6
            r64 = rec.astype(np.float64)
            # centre frequency = instantaneous frequency at tc, which the cubic
            # term contributes 3 c3 tc^2 to
            fc_col = (r64[:, 2] + 2.0 * r64[:, 3] * r64[:, 0]
                      + 3.0 * r64[:, 6] * r64[:, 0] ** 2)
            # report the chirp rate AT tc so features.atom_to_physical stays exact:
            # d/dt of (3 c3 t^2 + 2 c t) at tc is 2(c + 3 c3 tc)
            r64[:, 3] = r64[:, 3] + 3.0 * r64[:, 6] * r64[:, 0]
            param_rows.append(np.stack([r64[:, 0], r64[:, 1], fc_col, r64[:, 3],
                                        r64[:, 4], r64[:, 5]], 1))
            if return_params:
                continue          # skip building per-atom objects entirely
            for k in range(B):
                picks[k].append(QACTAtom(
                    tc=float(r64[k, 0]), fc=float(fc_col[k]), logDt=float(r64[k, 1]),
                    c=float(r64[k, 3]), coeff_real=float(r64[k, 4]),
                    coeff_imag=float(r64[k, 5])))
        err = (R.norm(dim=1) / R0.norm(dim=1).clamp(min=1e-12)).cpu().numpy()
        self.post_select_frac = float(np.mean(succ_hist)) if succ_hist else 0.0
        certs = [_Cert(float(err[k]), self.dict_size, self.post_select_frac) for k in range(B)]
        if return_raw:
            return np.stack(raw_rows, 1).astype(np.float32), err.astype(np.float64)
        if return_params:
            # (B, order, 6) plus the reconstruction errors; feed to features_batch
            return np.stack(param_rows, 1).astype(np.float64), err.astype(np.float64)
        return (picks[0], certs[0]) if single else (picks, certs)


class QACTRefiner:
    """Off-grid refinement of (tc, logDt, f_eff, c) by the PARAMETER-SHIFT RULE.

    The objective is the captured energy  E(theta) = |<psi_theta|r>|^2 / ||psi||^2.
    Each phase gate enters the overlap linearly in exp(i a), so E is exactly
    A + B cos a + C sin a in every gate angle, and the two-point shift rule is
    exact -- the same rule used on hardware:

        dE/da = [ E(a + pi/2) - E(a - pi/2) ] / 2

    Chain rule back to the physical parameters:
        f_eff  ->  n gates, weights 2*pi*2^k / N
        c      ->  n single-qubit gates (2*pi*2^{2k}/N)
                   + n(n-1)/2 controlled gates (2*pi*2^{j+k+1}/N)
    The envelope parameters (tc, logDt) are not gate angles -- they set the
    ancilla rotation angles of the diagonal filter -- so they use two-point
    central differences, which is also what a device would do.

    In simulation the two shifted evaluations are obtained in closed form.
    Writing the overlap as O = S0 + S1, where S1 sums the basis states the gate
    acts on, the shifted values are |S0 -+ i S1|^2, so

        dE/da = -2 * Im( S0 * conj(S1) ) / ||psi||^2

    exactly. `evals_per_step` reports the number of circuit evaluations the same
    gradient would cost on hardware, which is the honest price of this step.
    """

    def __init__(self, N, fs, device="cuda", steps=4, lr_f=0.5, lr_c=None,
                 lr_tc=1.0, lr_logdt=0.05, band_hz=(0.5, 45.0),
                 logdt_range=(1.0, 6.0), rate_max_hz_s=60.0,
                 c3_max=0.0, skew_max=0.0, ratio_range=(1.0, 1.0)):
        self.N, self.fs, self.dev, self.steps = int(N), float(fs), device, int(steps)
        self.n = int(round(math.log2(N)))
        self.t = torch.arange(N, device=device, dtype=torch.float32)
        idx = torch.arange(N, device=device)
        self.bit = torch.stack([((idx >> k) & 1).float() for k in range(self.n)])
        self.pairs = [(j, k) for j in range(self.n) for k in range(j + 1, self.n)]
        self.pair_mask = torch.stack([self.bit[j] * self.bit[k] for j, k in self.pairs]) \
            if self.pairs else torch.zeros(0, N, device=device)
        # complex copies: the masked partial sums multiply a complex vector
        self.bit_c = self.bit.to(torch.complex64)
        self.pair_mask_c = self.pair_mask.to(torch.complex64)
        # chain-rule weights
        self.w_f = torch.tensor([2 * math.pi * (2 ** k) / N for k in range(self.n)],
                                device=device)
        self.w_c1 = torch.tensor([2 * math.pi * (2 ** (2 * k)) / N for k in range(self.n)],
                                 device=device)
        self.w_c2 = torch.tensor([2 * math.pi * (2 ** (j + k + 1)) / N for j, k in self.pairs],
                                 device=device)
        # Cubic phase is 3-local, so its chain rule collects three groups of gate
        # angles -- exactly the decomposition `cubic_phase_gates` emits.
        self.triples = [(i, j, k) for i in range(self.n) for j in range(i + 1, self.n)
                        for k in range(j + 1, self.n)] if c3_max > 0 else []
        self.trip_mask_c = (torch.stack([self.bit[i] * self.bit[j] * self.bit[k]
                                         for i, j, k in self.triples]).to(torch.complex64)
                            if self.triples else torch.zeros(0, N, device=device,
                                                             dtype=torch.complex64))
        self.w_c3_1 = torch.tensor([2 * math.pi * (2 ** (3 * k)) / N for k in range(self.n)],
                                   device=device)
        self.w_c3_2 = torch.tensor([2 * math.pi * 3 * (2 ** (2 * j + k) + 2 ** (2 * k + j)) / N
                                    for j, k in self.pairs], device=device)
        self.w_c3_3 = torch.tensor([2 * math.pi * 6 * (2 ** (i + j + k)) / N
                                    for i, j, k in self.triples], device=device)
        self.c3_max, self.skew_max, self.ratio_range = float(c3_max), float(skew_max), ratio_range
        # step sizes for the parameters that used to be fixed grid context
        self.lr_c3 = (c3_max / 8.0) if c3_max > 0 else 0.0
        self.lr_skew = (skew_max / 8.0) if skew_max > 0 else 0.0
        self.lr_logr = 0.05 if ratio_range[1] > ratio_range[0] else 0.0
        self.lr = dict(f=lr_f, c=lr_c if lr_c is not None else 2.0 / (N * 4),
                       tc=lr_tc, logdt=lr_logdt)
        self.band = band_hz
        self.logdt_range = logdt_range
        self.c_max = rate_max_hz_s * N / (2 * fs ** 2)
        self.evals_per_step = 2 * self.n + 2 * (self.n + len(self.pairs)) + 4
        # refining c3 costs its own 3-local shift gradients; skew and the width
        # ratio cost one central difference each
        self.evals_per_step_full = (self.evals_per_step
                                    + 2 * (self.n + len(self.pairs) + len(self.triples))
                                    + (4 if self.lr_skew > 0 else 0)
                                    + (4 if self.lr_logr > 0 else 0))

    def _env_phase(self, tc, logdt, f, c, c3=None, skew=None, ratio=None):
        """Envelope and phase for the full parameter set.

        `c3`, `skew` and `ratio` are refined when `refine(..., full=True)` is used
        and carried as fixed dictionary context otherwise. Both neutral
        asymmetries reproduce the plain Gaussian by the identical expression.
        """
        d = self.t[None, :] - tc[:, None]
        w = torch.exp(logdt)[:, None]
        env = torch.exp(-(d ** 2) / (2 * w ** 2))
        if ratio is not None and torch.count_nonzero(ratio - 1.0) > 0:
            wr = w * ratio[:, None]
            env = torch.where(d >= 0, torch.exp(-(d ** 2) / (2 * wr ** 2)), env)
        if skew is not None and torch.count_nonzero(skew) > 0:
            env = env * (1.0 + torch.erf(skew[:, None] * d / (math.sqrt(2.0) * w)))
        t = self.t[None, :]
        if c3 is None or not bool(torch.any(c3)):       # see note on associativity
            phase = 2 * math.pi * (c[:, None] * t ** 2 + f[:, None] * t) / self.N
        else:
            phase = 2 * math.pi * (c3[:, None] * t ** 3 + c[:, None] * t ** 2
                                   + f[:, None] * t) / self.N
        return env, phase

    def energy(self, R, tc, logdt, f, c, c3=None, skew=None, ratio=None):
        env, phase = self._env_phase(tc, logdt, f, c, c3, skew, ratio)
        O = ((env * R) * torch.exp(-1j * phase)).sum(1)
        return (O.abs() ** 2) / (env ** 2).sum(1).clamp(min=1e-12)

    def refine(self, R, tc, logdt, f, c, c3=None, skew=None, ratio=None, full=False):
        """Gradient ascent on captured energy. R: (B, N) residual; params (B,).

        `full=True` also refines c3, skew and the envelope width ratio, and
        returns all seven parameters instead of four.
        """
        tc, logdt, f, c = (x.clone() for x in (tc, logdt, f, c))
        if full:
            c3 = None if c3 is None else c3.clone()
            skew = None if skew is None else skew.clone()
            ratio = None if ratio is None else ratio.clone()
        for _ in range(self.steps):
            env, phase = self._env_phase(tc, logdt, f, c, c3, skew, ratio)
            w = (env * R) * torch.exp(-1j * phase)            # (B, N)
            norm2 = (env ** 2).sum(1).clamp(min=1e-12)
            tot = w.sum(1)
            # parameter-shift gradients w.r.t. every gate angle, in closed form
            S1_s = w @ self.bit_c.T                            # (B, n)
            g_s = -2 * (tot[:, None] * S1_s.conj()).imag / norm2[:, None]
            if len(self.pairs):
                S1_p = w @ self.pair_mask_c.T                  # (B, n(n-1)/2)
                g_p = -2 * (tot[:, None] * S1_p.conj()).imag / norm2[:, None]
            else:
                g_p = torch.zeros(len(R), 0, device=self.dev)
            grad_f = (g_s * self.w_f).sum(1)
            grad_c = (g_s * self.w_c1).sum(1) + (g_p * self.w_c2).sum(1)
            if full and c3 is not None and self.lr_c3 > 0:
                S1_t = w @ self.trip_mask_c.T                  # (B, n(n-1)(n-2)/6)
                g_t = -2 * (tot[:, None] * S1_t.conj()).imag / norm2[:, None]
                grad_c3 = ((g_s * self.w_c3_1).sum(1) + (g_p * self.w_c3_2).sum(1)
                           + (g_t * self.w_c3_3).sum(1))
            else:
                grad_c3 = None
            # envelope: two-point central differences (hardware would do the same).
            # The phase does not depend on tc or logDt, so exp(-i*phase) is reused
            # instead of recomputed: 5 phase exponentials per step become 2.
            expph = torch.exp(-1j * phase)

            def _energy_env(tc2, ld2, sk2=None, ra2=None):
                d2 = self.t[None, :] - tc2[:, None]
                w2 = torch.exp(ld2)[:, None]
                env2 = torch.exp(-(d2 ** 2) / (2 * w2 ** 2))
                ra2 = ratio if ra2 is None else ra2
                if ra2 is not None and torch.count_nonzero(ra2 - 1.0) > 0:
                    wr2 = w2 * ra2[:, None]
                    env2 = torch.where(d2 >= 0, torch.exp(-(d2 ** 2) / (2 * wr2 ** 2)), env2)
                sk2 = skew if sk2 is None else sk2
                if sk2 is not None and torch.count_nonzero(sk2) > 0:
                    env2 = env2 * (1.0 + torch.erf(sk2[:, None] * d2
                                                   / (math.sqrt(2.0) * w2)))
                O2 = ((env2 * R) * expph).sum(1)
                return (O2.abs() ** 2) / (env2 ** 2).sum(1).clamp(min=1e-12)

            e0 = self.energy(R, tc, logdt, f, c, c3, skew, ratio)
            gtc = _energy_env(tc + 0.5, logdt) - _energy_env(tc - 0.5, logdt)
            gld = (_energy_env(tc, logdt + 0.05) - _energy_env(tc, logdt - 0.05)) / 0.1
            new = (tc + self.lr["tc"] * gtc / (gtc.abs().max().clamp(min=1e-9)),
                   (logdt + self.lr["logdt"] * torch.sign(gld)).clamp(*self.logdt_range),
                   f + self.lr["f"] * grad_f / (grad_f.abs().max().clamp(min=1e-9)),
                   (c + self.lr["c"] * torch.sign(grad_c)).clamp(-self.c_max, self.c_max))
            cand_tc, cand_ld, cand_f, cand_c = new
            cand_tc = cand_tc.clamp(0, self.N - 1)
            # keep the centre frequency in band
            c3v = 0.0 if c3 is None else c3
            fc_hz = (cand_f + 2 * cand_c * cand_tc
                     + (0.0 if c3 is None else 3 * c3v * cand_tc ** 2)) * self.fs / self.N
            bad = (fc_hz < self.band[0]) | (fc_hz > self.band[1])
            cand_f = torch.where(bad, f, cand_f)
            cand_c3, cand_sk, cand_ra = c3, skew, ratio
            if full:
                if grad_c3 is not None:
                    cand_c3 = (c3 + self.lr_c3 * torch.sign(grad_c3)).clamp(
                        -self.c3_max, self.c3_max)
                if skew is not None and self.lr_skew > 0:
                    h = max(self.lr_skew, 1e-6)
                    gsk = (_energy_env(tc, logdt, sk2=skew + h)
                           - _energy_env(tc, logdt, sk2=skew - h)) / (2 * h)
                    cand_sk = (skew + self.lr_skew * torch.sign(gsk)).clamp(
                        -self.skew_max, self.skew_max)
                if ratio is not None and self.lr_logr > 0:
                    lr_ = torch.log(ratio.clamp(min=1e-6))
                    up, dn = torch.exp(lr_ + 0.05), torch.exp(lr_ - 0.05)
                    gra = (_energy_env(tc, logdt, ra2=up)
                           - _energy_env(tc, logdt, ra2=dn)) / 0.1
                    cand_ra = torch.exp(lr_ + self.lr_logr * torch.sign(gra)).clamp(
                        *self.ratio_range)
            e1 = self.energy(R, cand_tc, cand_ld, cand_f, cand_c, cand_c3, cand_sk, cand_ra)
            take = e1 > e0                                     # accept only improvements
            tc = torch.where(take, cand_tc, tc)
            logdt = torch.where(take, cand_ld, logdt)
            f = torch.where(take, cand_f, f)
            c = torch.where(take, cand_c, c)
            if full:
                if cand_c3 is not None and c3 is not None:
                    c3 = torch.where(take, cand_c3, c3)
                if cand_sk is not None and skew is not None:
                    skew = torch.where(take, cand_sk, skew)
                if cand_ra is not None and ratio is not None:
                    ratio = torch.where(take, cand_ra, ratio)
        return (tc, logdt, f, c, c3, skew, ratio) if full else (tc, logdt, f, c)


def features_batch(params, recon_err, fs, N):
    """Vectorised equivalent of features.channel_features for a batch of atom sets.

    params: (B, P, 6) float64 array of (tc, logDt, fc, c, coeff_real, coeff_imag);
    recon_err: (B,). Returns (B, 13) in the same order as
    features.PER_CHANNEL_FEATURES. Computed in float64 so it matches the
    per-row Python path; `verify_features_batch` checks that.
    """
    from .features import (BANDS, EEG_BAND, MAX_DURATION_EPOCHS, MAX_RELATIVE_SWEEP,
                           MIN_CYCLES)
    fc, logDt, c = params[..., 2], params[..., 1], params[..., 3]
    cr, ci = params[..., 4], params[..., 5]
    energy = cr ** 2 + ci ** 2
    f_hz = np.abs(fc) * fs / N
    dur = np.exp(logDt) / fs
    sweep = (2.0 * c * fs * fs / N) * dur
    cycles = f_hz * dur
    osc = ((cycles >= MIN_CYCLES) & (f_hz >= EEG_BAND[0]) & (f_hz <= EEG_BAND[1])
           & (dur <= MAX_DURATION_EPOCHS * N / fs)
           & (np.abs(sweep) <= MAX_RELATIVE_SWEEP * np.maximum(f_hz, 1e-9)))
    tot = energy.sum(1)
    eo = energy * osc
    osc_e = eo.sum(1)
    B = params.shape[0]
    out = np.zeros((B, 13))
    live = tot > 1e-12
    safe_tot = np.where(live, tot, 1.0)
    col = 0
    for lo, hi in BANDS.values():
        out[:, col] = (eo * ((f_hz >= lo) & (f_hz < hi))).sum(1) / safe_tot
        col += 1
    out[:, col] = osc_e / safe_tot; col += 1
    out[:, col] = 1.0 - osc_e / safe_tot; col += 1
    w = np.where(osc_e > 1e-12, osc_e, 1.0)
    mean_f = (eo * f_hz).sum(1) / w
    std_f = np.sqrt(np.maximum((eo * (f_hz - mean_f[:, None]) ** 2).sum(1) / w, 0.0))
    has = osc_e > 1e-12
    for v in (mean_f, std_f, (eo * np.abs(sweep)).sum(1) / w,
              (eo * sweep).sum(1) / w, (eo * dur).sum(1) / w):
        out[:, col] = np.where(has, v, 0.0); col += 1
    out[:, col] = recon_err
    out[~live] = 0.0
    return out


def verify_features_batch(seed=0, B=64, order=12, N=512, fs=256.0, device="cuda"):
    """Check features_batch against the per-row features.channel_features path."""
    from .features import channel_features
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, N)).astype(np.float32)
    X = X - X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = QACT(length=N, fs=fs, device=device, seed=seed, refine_steps=2)
    picks, certs = q.transform(X, order=order)
    loop = np.stack([channel_features(a, c, fs, N) for a, c in zip(picks, certs)])
    par = np.stack([[[a.tc, a.logDt, a.fc, a.c, a.coeff_real, a.coeff_imag] for a in A]
                    for A in picks])
    vec = features_batch(par, np.array([c.reconstruction_error for c in certs]), fs, N)
    return float(np.abs(loop - vec).max())


class _Cert:
    __slots__ = ("reconstruction_error", "dictionary_size", "energy_explained_fraction",
                 "post_select_frac")

    def __init__(self, err, dsize, succ):
        self.reconstruction_error = err
        self.dictionary_size = dsize
        self.energy_explained_fraction = 1.0 - err ** 2
        self.post_select_frac = succ

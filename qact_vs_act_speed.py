#!/usr/bin/env python
"""QACT vs classical ACT: how fast is each -- measured, and on a real QPU.

Same job for every engine: decompose K real EEGMAT windows (2 s, 512 samples)
into P chirplet atoms, OMP coefficients, no backfit.

Measured here (wall clock):
  classical ACT          act_gpu engine exactly as in the denoising study
                         (OMP, 60 refinement steps), on GPU and on CPU
  QACT exact simulation  qbe.qact on GPU (the FFT is the QFT's output), in the
                         same configuration the circuit run uses
  QACT circuits          every selection and refinement step is a hardware
                         circuit (qbe.qact_hw 'windowed': envelope folded into the
                         loaded state, window-local register, semiclassical QFT),
                         executed shot by shot in Qiskit Aer; branch-and-bound
                         selection, 3-point refinement; OMP classical

Estimated for IBM Heron (FakeTorino calibration: CZ 68 ns, measure 1.56 us,
default repetition delay 250 us): every circuit the Aer run ACTUALLY executed,
priced at shots x (device-transpiled circuit duration + repetition delay). The
originally drafted design is priced the same way for comparison. Queueing,
compilation and upload latency are excluded, so these are lower bounds.
"""
import json
import math
import time

import numpy as np
import torch
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qbe import act_gpu
from qbe.acquire import read_eegmat_file
from qbe.gpu_features import gpu_dictionary_grid
from qbe.qact import QACT
from qbe.qact_hw import build, circuit_duration, window_register

ROOT = "/home/kc/EEG-Memristor-ACT/data/eegmat"
FS, N, n = 256.0, 512, 9
K, P = 4, 6                    # windows, atoms per window
SHOTS = 2000
STEPS = 4                      # refinement steps (as in every QACT run)
BATCH_ENV = 4                  # envelopes per Aer job during branch-and-bound
REP_DELAY = 250e-6             # FakeTorino default_rep_delay (range 0-500 us)
BAND = (0.5, 45.0)
t_ax = np.arange(N)


def eeg_windows(k=K, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for sub in rng.choice(36, size=k, replace=False):
        rec = read_eegmat_file(f"{ROOT}/Subject{sub:02d}_1.edf", 0, target_fs=FS)
        ch = rng.integers(rec.data.shape[0])
        st = rng.integers(0, rec.data.shape[1] - N)
        w = rec.data[ch, st:st + N].astype(np.float64)
        out.append(w - w.mean())
    return np.stack(out)


def env_of(tc, ld):
    return np.exp(-((t_ax - tc) ** 2) / (2 * np.exp(ld) ** 2))


def basis(tc, ld, f, c):
    e = env_of(tc, ld)
    ph = 2 * np.pi * (c * t_ax ** 2 + f * t_ax) / N
    return e * np.cos(ph), e * np.sin(ph)


class CircuitQACT:
    """QACT whose every energy evaluation is a circuit run in Aer."""

    def __init__(self, q):
        self.sim = AerSimulator()
        self.env_tc = q.env_tc.cpu().numpy()
        self.env_ld = q.env_logDt.cpu().numpy()
        self.c_t = q.c_t.cpu().numpy()
        pe, pc = q.pair_e.cpu().numpy(), q.pair_c.cpu().numpy()
        self.chirps_of = {e: sorted(pc[pe == e].tolist()) for e in np.unique(pe)}
        self.lr_c = 2.0 / (N * 4)          # the refiner's own step sizes
        self.c_max = 60.0 * N / (2 * FS ** 2)
        self.log = []                      # (register size m, shots) per circuit
        self.aer_time = 0.0

    def _run(self, circuits):
        tq = transpile(circuits, self.sim, optimization_level=0)
        res = self.sim.run(tq, shots=SHOTS, seed_simulator=len(self.log)).result()
        self.aer_time += res.time_taken
        out = []
        for i, qc in enumerate(circuits):
            m = qc.num_qubits
            self.log.append((m, SHOTS))
            p = np.zeros(2 ** m)
            for key, v in res.get_counts(i).items():
                p[int(key, 2)] += v
            out.append(p / SHOTS)
        return out

    @staticmethod
    def _scale(r, env):
        """criterion(k) = 2^m P_m(k') ||(env r)_window||^2 / ||env||^2"""
        m, t0 = window_register(env, n)
        wr = (env * r)[t0:t0 + 2 ** m]
        return m, 2 ** m * float(wr @ wr) / float(env @ env)

    def select(self, r):
        live = sorted(self.chirps_of)
        E = np.stack([env_of(self.env_tc[e], self.env_ld[e]) for e in live])
        bound = (E @ np.abs(r)) ** 2 / (E ** 2).sum(1)
        order = [live[i] for i in np.argsort(-bound)]
        bnd = dict(zip(live, bound))
        best, i = (-1.0, None), 0
        while i < len(order) and bnd[order[i]] > best[0]:
            batch = [e for e in order[i:i + BATCH_ENV] if bnd[e] > best[0]]
            i += BATCH_ENV
            jobs, meta = [], []
            for e in batch:
                env = env_of(self.env_tc[e], self.env_ld[e])
                for ci in self.chirps_of[e]:
                    jobs.append(build("windowed", r, env, self.c_t[ci], loader="initialize"))
                    meta.append((e, ci, env))
            for p, (e, ci, env) in zip(self._run(jobs), meta):
                m, sc = self._scale(r, env)
                step = 2 ** (n - m)
                k = np.arange(2 ** m) * step
                tc, c = self.env_tc[e], self.c_t[ci]
                fc = (k + 2 * c * tc) * FS / N
                ok = (k < N // 2) & (fc >= BAND[0]) & (fc <= BAND[1])
                if not ok.any():
                    continue
                crit = np.where(ok, p * sc, -1.0)
                j = int(np.argmax(crit))
                if crit[j] > best[0]:
                    best = (float(crit[j]), (float(tc), float(self.env_ld[e]),
                                             float(k[j]), float(c)))
        return best[1]

    def energies(self, r, params):
        """Captured energy of each (tc, ld, f, c), one circuit each: the frequency
        offset puts f exactly on a measured bin."""
        jobs, meta = [], []
        for tc, ld, f, c in params:
            env = env_of(tc, ld)
            m, _ = window_register(env, n)
            step = 2 ** (n - m)
            kq = math.floor(f / step)
            jobs.append(build("windowed", r, env, c, offset=f - kq * step,
                              loader="initialize"))
            meta.append((kq % 2 ** m, env))
        return [p[kq] * self._scale(r, env)[1]
                for p, (kq, env) in zip(self._run(jobs), meta)]

    def refine(self, r, x):
        x = list(x)
        e0 = self.energies(r, [x])[0]
        deltas = (1.0, 0.05, 0.5, self.lr_c)             # tc, logDt, f, c
        for _ in range(STEPS):
            for j, d in enumerate(deltas):
                cands = []
                for sgn in (-1.0, 1.0):
                    y = list(x)
                    y[j] += sgn * d
                    y[0] = min(max(y[0], 0.0), N - 1.0)
                    y[1] = min(max(y[1], 1.0), 6.0)
                    y[3] = min(max(y[3], -self.c_max), self.c_max)
                    cands.append(y)
                for y, e1 in zip(cands, self.energies(r, cands)):
                    fc = (y[2] + 2 * y[3] * y[0]) * FS / N
                    if e1 > e0 and (j != 2 or BAND[0] <= fc <= BAND[1]):
                        x, e0 = y, e1
        return x

    def decompose(self, x):
        r, atoms = x.copy(), []
        for _ in range(P):
            atoms.append(self.refine(r, self.select(r)))
            A = np.stack([b for a in atoms for b in basis(*a)], 1)
            coef = np.linalg.lstsq(A, x, rcond=None)[0]            # OMP, classical
            r = x - A @ coef
        return float(np.linalg.norm(r) / np.linalg.norm(x))


def timed(fn, sync=False):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync:
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def classical_act(X, device):
    D = act_gpu.GPUDictionary(gpu_dictionary_grid(N, FS), N, FS, device=device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    act_gpu.decompose_batch(Xt, D, max_atoms=P, steps=60, omp=True)         # warm-up
    (_, R), dt = timed(lambda: act_gpu.decompose_batch(Xt, D, max_atoms=P, steps=60,
                                                       omp=True), sync=device == "cuda")
    err = (R.norm(dim=1) / Xt.norm(dim=1)).cpu().numpy()
    return err, dt / len(X)


def exact_qact(X):
    q = QACT(length=N, fs=FS, device="cuda", seed=0, refine_steps=STEPS,
             omp=True, backfit_passes=0, exact_f=False, refine_mode="hw")
    q.transform(X.astype(np.float32), order=P, select="argmax", return_params=True)
    (_, err), dt = timed(lambda: q.transform(X.astype(np.float32), order=P,
                                             select="argmax", return_params=True),
                         sync=True)
    return err, dt / len(X), q


def width_for(m):
    """A logDt whose +-4 sigma window needs exactly m qubits."""
    for ld in np.linspace(0.5, 6.5, 1201):
        if window_register(env_of(256.0, ld), n)[0] == m:
            return ld
    raise ValueError(m)


def main():
    X = eeg_windows()
    XB = np.tile(X, (256, 1))                       # 1,024 windows, batched
    print(f"job: {K} real EEGMAT windows x {P} atoms  (512 samples = 9 qubits each)\n",
          flush=True)
    res = []                                        # name, s/window, s/window batched, err

    e, dt = classical_act(X, "cuda"); _, dtb = classical_act(XB, "cuda")
    res.append(("classical ACT, GPU", dt, dtb, float(e.mean())))
    e, dt = classical_act(X, "cpu")
    res.append(("classical ACT, CPU", dt, None, float(e.mean())))
    e, dt, q = exact_qact(X); _, dtb, _ = exact_qact(XB)
    res.append(("QACT exact simulation, GPU", dt, dtb, float(e.mean())))
    for name, dt, dtb, err in res:
        print(f"{name:<32} {dt * 1e3:10.2f} ms/window"
              + (f"  ({dtb * 1e3:.3f} ms/window batched x1024)" if dtb else "")
              + f"   recon err {err:.3f}", flush=True)

    cq = CircuitQACT(q)
    t0 = time.perf_counter()
    errs = []
    for i, x in enumerate(X):
        errs.append(cq.decompose(x))
        print(f"   circuit QACT window {i + 1}/{K}: recon err {errs[-1]:.3f}, "
              f"{len(cq.log):,} circuits so far ({time.perf_counter() - t0:.0f}s)",
              flush=True)
    dt = (time.perf_counter() - t0) / K
    res.append(("QACT circuits in Aer (simulator)", dt, None, float(np.mean(errs))))
    print(f"QACT circuits in Aer: {dt:.1f} s/window (Aer execution {cq.aer_time / K:.1f} s "
          f"of it); {len(cq.log) / K:,.0f} circuits and {sum(s for _, s in cq.log) / K:,.0f}"
          f" shots per window; recon err {np.mean(errs):.3f}", flush=True)

    # ---- real-device estimate for the circuits that were actually run ----
    backend = FakeTorino()
    ms = sorted({m for m, _ in cq.log})
    dur, cnt = {}, {}
    for m in ms:
        tq = transpile(build("windowed", X[0], env_of(256.0, width_for(m)), 0.03125),
                       backend=backend, optimization_level=3, seed_transpiler=7)
        dur[m] = circuit_duration(tq, backend.target)
        cnt[m] = sum(1 for mm, _ in cq.log if mm == m) / K
    heron = sum(s * (dur[m] + REP_DELAY) for m, s in cq.log) / K
    heron_min = sum(s * dur[m] for m, s in cq.log) / K
    print("\nIBM Heron, per shot (device-transpiled; feed-forward latency excluded):")
    for m in ms:
        print(f"   {m}-qubit window: {dur[m] * 1e6:6.1f} us + {REP_DELAY * 1e6:.0f} us "
              f"repetition delay   x {cnt[m]:5.0f} circuits/window x {SHOTS} shots")

    # the originally drafted design, priced the same way
    tb = transpile(build("baseline", X[0], env_of(256.0, 2.7), 0.03125), backend=backend,
                   optimization_level=3, seed_transpiler=7)
    d_base = tb.estimate_duration(backend.target, unit="s")
    post = float(np.mean([1 / 0.027, 1 / 0.057, 1 / 0.181, 1 / 0.57]))
    circ_base = P * (q.n_pairs_used + 1 + STEPS * 112)
    base_s = circ_base * SHOTS * post * (d_base + REP_DELAY)
    res += [("QACT on IBM Heron, optimised", heron, None, None),
            ("   same, repetition delay 0", heron_min, None, None),
            ("QACT on IBM Heron, original design", base_s, None, None)]

    same_job = res[1][1]                     # classical CPU on the same 4 windows
    thru = res[0][2]                         # classical GPU, batched throughput
    fmt = lambda x: f"{x * 1e3:,.2f} ms" if x < 1 else (
        f"{x:,.1f} s" if x < 3600 else f"{x / 3600:,.1f} h")
    rel = lambda x: (f"{x:,.0f}x slower" if x >= 1.5 else
                     f"{1 / x:,.1f}x faster" if x <= 1 / 1.5 else "~same")
    print(f"\n{'engine':<36}{'per window':>12}{'vs classical CPU':>22}"
          f"{'vs classical GPU batched':>28}")
    for name, dt, _, _ in res:
        est = "  (estimate)" if "Heron" in name or name.startswith("   ") else ""
        print(f"{name:<36}{fmt(dt):>12}{rel(dt / same_job):>22}{rel(dt / thru):>28}{est}")
    print(f"\noriginal design: {circ_base / P:,.0f} circuits/atom, {post:.1f}x shots lost "
          f"to post-selection, {d_base * 1e6:.0f} us/shot.  optimised: "
          f"{len(cq.log) / K / P:,.0f} circuits/atom, no shots lost.")

    json.dump(dict(rows=[dict(engine=a, seconds_per_window=b,
                              batched_seconds_per_window=c, recon_err=d)
                         for a, b, c, d in res],
                   circuits_per_window=len(cq.log) / K, shots=SHOTS,
                   duration_us={str(m): dur[m] * 1e6 for m in ms},
                   circuits_by_register={str(m): cnt[m] for m in ms},
                   rep_delay_us=REP_DELAY * 1e6, baseline_duration_us=d_base * 1e6,
                   baseline_circuits_per_atom=circ_base / P),
              open("results/qact_vs_act_speed.json", "w"), indent=1)
    print("\nwrote results/qact_vs_act_speed.json")


if __name__ == "__main__":
    main()

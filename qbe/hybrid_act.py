"""Hybrid ACT: each part of the transform runs where it is better, quantum or classical.

The parts that could go either way are routed by one number, the LEVEL `q_max`:
an atom whose +-4 sigma window fits in <= q_max qubits is scored and refined by
QACT circuits; a wider atom is scored and refined by the exact classical FFT.

    q_max = 0   fully classical: the same algorithm with every score exact
    q_max = 6   only the shortest atoms (windows <= 64 samples) on the QPU
    q_max = 7   short atoms (<= 128 samples): the circuit sizes that selected the
                right atom under IBM Heron noise in qact_hardware.py
    q_max = 8   adds mid-length windows (<= 256 samples)
    q_max = 9   everything quantum (QACT as in qact_hw_speedups.py)

Fixed routes, whatever the level:

    classical  window multiply env * r       O(N) arithmetic; on a QPU a 2^n-way
                                             multiplexed rotation + post-selection
    classical  branch-and-bound bounds       O(N) per envelope, rigorous
    classical  OMP coefficients, residual    the atom Gram matrix is
                                             data-independent and the right-hand
                                             side is O(kN): a circuit buys nothing
    quantum    scoring + refinement of atoms that fit the level: window-local
               register, semiclassical QFT, adaptive shots (successive
               elimination; refinement comparisons raced)

Selection mixes the two pools honestly: classical candidates are exact, so the
best of them seeds branch-and-bound and prunes quantum circuits outright;
quantum candidates carry confidence bounds and are eliminated against it.
"""
from __future__ import annotations

import math
import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .qact import QACT
from .qact_hw import build, window_register, windowed_distribution

LEVELS = (0, 6, 7, 8, 9)


class HybridACT:
    def __init__(self, q_max, noise="heron", N=512, fs=256.0, steps=4, s0=250,
                 s_max=2000, z=2.0, band=(0.5, 45.0), batch_env=4, device="cuda",
                 backend=None, opt_level=3, seed=0, noise_model=None,
                 basis_gates=None, mix=0.0):
        """noise: "heron" (IBM Heron physical noise, device-transpiled),
        "none" (noiseless), "logical" (an error-corrected machine seen at the
        logical level: all-to-all logical qubits, `noise_model` on the
        `basis_gates` of the logical circuit), or "classical_mix" -- NO circuits:
        each quantum-route evaluation is the circuit's exact output distribution
        (computed classically), mixed with a uniform distribution at weight `mix`
        and sampled with the same shot count. The classical control for any
        benefit that noisy selection might bring."""
        self.q_max, self.N, self.fs = int(q_max), int(N), float(fs)
        self.n = int(round(math.log2(N)))
        q = QACT(length=N, fs=fs, device=device, seed=0, refine_steps=steps,
                 omp=True, backfit_passes=0, exact_f=False, refine_mode="hw")
        self.env_tc = q.env_tc.cpu().numpy()
        self.env_ld = q.env_logDt.cpu().numpy()
        self.c_t = q.c_t.cpu().numpy()
        pe, pc = q.pair_e.cpu().numpy(), q.pair_c.cpu().numpy()
        self.chirps_of = {int(e): sorted(pc[pe == e].tolist()) for e in np.unique(pe)}
        self.t = np.arange(N, dtype=float)
        self.lr_c = 2.0 / (N * 4)                 # the refiner's own step sizes
        self.c_max = 60.0 * N / (2 * fs ** 2)
        self.steps, self.s0, self.s_max, self.z = steps, s0, s_max, z
        self.band, self.batch_env, self.opt_level = band, batch_env, opt_level
        self.noise = noise
        self.basis_gates = basis_gates
        if noise == "heron":
            from qiskit_ibm_runtime.fake_provider import FakeTorino
            self.backend = backend or FakeTorino()
            self.sim = AerSimulator.from_backend(self.backend)
        elif noise == "logical":
            self.backend, self.sim = None, AerSimulator(noise_model=noise_model)
        elif noise == "classical_mix":
            self.backend, self.sim = None, None
            self.mix = float(mix)
            self._rng = np.random.default_rng(1000 + seed)
        else:
            self.backend, self.sim = None, AerSimulator()
        self._seed = 1000 + seed
        self.reset_stats()

    # ------------------------------------------------------------------ stats
    def reset_stats(self):
        self.qpu_log = []              # (register size, shots) per circuit execution
        self.round_trips = 0           # classical -> QPU -> classical turnarounds
        self.classical_s = 0.0         # wall time of the classical routes only
        self.evals = {"quantum": 0, "classical": 0}
        self.sel_quality = []          # exact energy of chosen atom / best available

    # ---------------------------------------------------------------- helpers
    def env(self, tc, ld):
        return np.exp(-((self.t - tc) ** 2) / (2 * np.exp(ld) ** 2))

    def is_quantum(self, env):
        return window_register(env, self.n)[0] <= self.q_max

    def _exact(self, r, tc, ld, f, c):
        e = self.env(tc, ld)
        ph = 2 * np.pi * (c * self.t ** 2 + f * self.t) / self.N
        return float(abs(np.sum(e * r * np.exp(-1j * ph))) ** 2 / (e @ e))

    def exact_energy(self, r, tc, ld, f, c):
        """Classical route: exact captured energy of one atom."""
        t0 = time.perf_counter()
        v = self._exact(r, tc, ld, f, c)
        self.classical_s += time.perf_counter() - t0
        self.evals["classical"] += 1
        return v

    def _exact_grid(self, r, e, ci):
        """Exact criterion over the full frequency grid for one (envelope, chirp)."""
        tc, c = self.env_tc[e], self.c_t[ci]
        env = self.env(tc, self.env_ld[e])
        z = env * r * np.exp(-2j * np.pi * c * self.t ** 2 / self.N)
        P = np.abs(np.fft.fft(z)) ** 2 / (env @ env)
        k = np.arange(self.N)
        fc = (k + 2 * c * tc) * self.fs / self.N
        ok = (k < self.N // 2) & (fc >= self.band[0]) & (fc <= self.band[1])
        return P, ok

    def _scale(self, r, env):
        m, t0 = window_register(env, self.n)
        wr = (env * r)[t0:t0 + 2 ** m]
        return 2 ** m * float(wr @ wr) / float(env @ env)

    # ---------------------------------------------------------- quantum route
    def _compile(self, qcs):
        if self.noise == "classical_mix":
            return qcs                              # specs, not circuits
        if self.noise == "heron":
            return transpile(qcs, backend=self.backend, optimization_level=self.opt_level,
                             seed_transpiler=7)
        if self.noise == "logical":
            return transpile(qcs, basis_gates=self.basis_gates,
                             optimization_level=self.opt_level, seed_transpiler=7)
        return transpile(qcs, self.sim, optimization_level=0)

    def _circuit(self, r, env, c, offset=0.0):
        if self.noise == "classical_mix":
            return dict(r=r, env=env, c=c, offset=offset)
        # noise must act on real gates, so noisy runs synthesise the state loading
        loader = "initialize" if self.noise == "none" else "stateprep"
        return build("windowed", r, env, c, offset=offset, loader=loader)

    def _exec(self, tqs, shots):
        if self.noise == "classical_mix":
            out = []
            for spec in tqs:
                P, m = windowed_distribution(spec["r"], spec["env"], spec["c"],
                                             spec["offset"])
                P = (1 - self.mix) * P + self.mix / 2 ** m
                self.qpu_log.append((m, shots))
                self.evals["quantum"] += 1
                out.append(self._rng.multinomial(shots, P / P.sum()).astype(float))
            return out
        res = self.sim.run(tqs, shots=shots, seed_simulator=self._seed).result()
        self._seed += 1
        self.round_trips += 1
        out = []
        for i, tq in enumerate(tqs):
            m = tq.num_clbits
            self.qpu_log.append((m, shots))
            self.evals["quantum"] += 1
            cnt = np.zeros(2 ** m)
            for key, v in res.get_counts(i).items():
                cnt[int(key.replace(" ", ""), 2)] += v
            out.append(cnt)
        return out

    def _bounds(self, d):
        h, s, sc, ok = d["counts"], d["shots"], d["sc"], d["ok"]
        pt = (h + 1) / (s + 2)
        crit = sc * h / s
        sig = sc * np.sqrt(pt * (1 - pt) / s)
        crit_m = np.where(ok, crit, -np.inf)
        d["j"] = int(np.argmax(crit_m))
        d["best"] = float(crit_m[d["j"]])
        d["lcb"] = float(np.where(ok, crit - self.z * sig, -np.inf).max())
        d["ucb"] = float(np.where(ok, crit + self.z * sig, -np.inf).max())

    # -------------------------------------------------------------- selection
    def select(self, r):
        live = sorted(self.chirps_of)
        t0 = time.perf_counter()
        E = np.stack([self.env(self.env_tc[e], self.env_ld[e]) for e in live])
        bound = (E @ np.abs(r)) ** 2 / (E ** 2).sum(1)           # classical, rigorous
        bnd = dict(zip(live, bound))
        qpool = [e for e in live if self.is_quantum(E[live.index(e)])]
        cpool = [e for e in live if e not in qpool]
        # classical pool: exact scores on the full grid; the best seeds the search
        best_c, best_lcb = (-1.0, None), -1.0
        for e in cpool:
            for ci in self.chirps_of[e]:
                P, ok = self._exact_grid(r, e, ci)
                self.evals["classical"] += 1
                if ok.any():
                    j = int(np.argmax(np.where(ok, P, -1)))
                    if P[j] > best_c[0]:
                        best_c = (float(P[j]), (float(self.env_tc[e]), float(self.env_ld[e]),
                                                float(j), float(self.c_t[ci])))
        best_lcb = best_c[0]
        self.classical_s += time.perf_counter() - t0
        # quantum pool: branch-and-bound screen, then successive elimination
        order = sorted(qpool, key=lambda e: -bnd[e])
        cands, i = [], 0
        while i < len(order) and bnd[order[i]] > best_lcb:
            batch = [e for e in order[i:i + self.batch_env] if bnd[e] > best_lcb]
            i += self.batch_env
            new = []
            for e in batch:
                env = self.env(self.env_tc[e], self.env_ld[e])
                m, _ = window_register(env, self.n)
                sc = self._scale(r, env)
                k = np.arange(2 ** m) * 2 ** (self.n - m)
                for ci in self.chirps_of[e]:
                    c, tc = self.c_t[ci], self.env_tc[e]
                    fc = (k + 2 * c * tc) * self.fs / self.N
                    new.append(dict(e=e, ci=ci, sc=sc, k=k,
                                    ok=(k < self.N // 2) & (fc >= self.band[0])
                                    & (fc <= self.band[1]),
                                    qc=self._circuit(r, env, c)))
            if not new:
                continue
            tqs = self._compile([d["qc"] for d in new])
            for d, tq, cnt in zip(new, tqs, self._exec(tqs, self.s0)):
                d.update(tq=tq, counts=cnt, shots=self.s0)
                if d["ok"].any():
                    self._bounds(d)
                    best_lcb = max(best_lcb, d["lcb"])
                    cands.append(d)
        alive = [d for d in cands if d["ucb"] >= best_lcb]
        while len(alive) > 1 and alive[0]["shots"] < self.s_max:
            add = min(alive[0]["shots"], self.s_max - alive[0]["shots"])
            for d, cnt in zip(alive, self._exec([d["tq"] for d in alive], add)):
                d["counts"] = d["counts"] + cnt
                d["shots"] += add
                self._bounds(d)
            best_lcb = max([best_c[0]] + [d["lcb"] for d in alive])
            alive = [d for d in alive if d["ucb"] >= best_lcb]
        pick = best_c
        if alive:
            d = max(alive, key=lambda d: d["best"])
            if d["best"] > best_c[0]:
                pick = (d["best"], (float(self.env_tc[d["e"]]), float(self.env_ld[d["e"]]),
                                    float(d["k"][d["j"]]), float(self.c_t[d["ci"]])))
        self._record_quality(r, pick[1])
        return pick[1]

    def _record_quality(self, r, atom):
        """Diagnostic only (not timed): true energy of the chosen atom as a
        fraction of the best atom anywhere in the dictionary."""
        best = 0.0
        for e in self.chirps_of:
            for ci in self.chirps_of[e]:
                P, ok = self._exact_grid(r, e, ci)
                if ok.any():
                    best = max(best, float(P[ok].max()))
        self.sel_quality.append(self._exact(r, *atom) / best if best > 0 else 1.0)

    # ------------------------------------------------------------- refinement
    def refine(self, r, x):
        cache = {}

        def ensure(keys, S):
            fresh = [key for key in keys if key not in cache]
            qkeys, qcs = [], []
            for key in fresh:
                tc, ld, f, c = key
                env = self.env(tc, ld)
                if self.is_quantum(env):
                    m, _ = window_register(env, self.n)
                    step = 2 ** (self.n - m)
                    kq = math.floor(f / step)
                    cache[key] = dict(kind="q", kq=kq % 2 ** m, sc=self._scale(r, env),
                                      hits=0.0, shots=0)
                    qkeys.append(key)
                    qcs.append(self._circuit(r, env, c, offset=f - kq * step))
                else:
                    cache[key] = dict(kind="c", val=self.exact_energy(r, *key))
            if qcs:
                for key, tq in zip(qkeys, self._compile(qcs)):
                    cache[key]["tq"] = tq
            by_add = {}
            for key in keys:
                if cache[key]["kind"] == "q" and S - cache[key]["shots"] > 0:
                    by_add.setdefault(S - cache[key]["shots"], []).append(key)
            for add, ks in by_add.items():
                for key, cnt in zip(ks, self._exec([cache[q]["tq"] for q in ks], add)):
                    cache[key]["hits"] += cnt[cache[key]["kq"]]
                    cache[key]["shots"] += add

        def est(key):
            d = cache[key]
            if d["kind"] == "c":
                return d["val"], 0.0
            h, s = d["hits"], d["shots"]
            pt = (h + 1) / (s + 2)
            return d["sc"] * h / s, d["sc"] * math.sqrt(pt * (1 - pt) / s)

        x = tuple(x)
        deltas = (1.0, 0.05, 0.5, self.lr_c)                  # tc, logDt, f, c
        for _ in range(self.steps):
            for j, dlt in enumerate(deltas):
                arms = [x]
                for sgn in (-1.0, 1.0):
                    y = list(x)
                    y[j] += sgn * dlt
                    y[0] = min(max(y[0], 0.0), self.N - 1.0)
                    y[1] = min(max(y[1], 1.0), 6.0)
                    y[3] = min(max(y[3], -self.c_max), self.c_max)
                    fc = (y[2] + 2 * y[3] * y[0]) * self.fs / self.N
                    if j != 2 or self.band[0] <= fc <= self.band[1]:
                        arms.append(tuple(y))
                S = self.s0
                while True:
                    ensure(arms, S)
                    e = {a: est(a) for a in arms}
                    top = max(arms, key=lambda a: e[a][0])
                    clear = all(e[top][0] - e[a][0] > self.z * math.hypot(e[top][1], e[a][1])
                                for a in arms if a != top)
                    if clear or S >= self.s_max:
                        break
                    S = min(2 * S, self.s_max)
                x = top
        return x

    # ------------------------------------------------------------- decompose
    def basis(self, tc, ld, f, c):
        e = self.env(tc, ld)
        ph = 2 * np.pi * (c * self.t ** 2 + f * self.t) / self.N
        return e * np.cos(ph), e * np.sin(ph)

    def decompose(self, x, order=6):
        r, atoms = x.copy(), []
        for _ in range(order):
            atoms.append(self.refine(r, self.select(r)))
            t0 = time.perf_counter()                             # OMP: classical
            A = np.stack([b for a in atoms for b in self.basis(*a)], 1)
            coef = np.linalg.lstsq(A, x, rcond=None)[0]
            r = x - A @ coef
            self.classical_s += time.perf_counter() - t0
        return float(np.linalg.norm(r) / np.linalg.norm(x)), atoms

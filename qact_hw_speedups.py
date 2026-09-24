#!/usr/bin/env python
"""Three published speed-ups applied to QACT's hardware circuits (REFERENCES.md).

Same job and same EEG windows as qact_vs_act_speed.py (4 real EEGMAT windows x 6
atoms), whose fixed-shot circuit run -- 2,000 shots on every circuit -- is the
reference.

1. Zero repetition delay   IBM Heron allows rep_delay 0-500 us (default 250 us).
2. Multi-programming       Niu & Todri-Sanial, Quantum 7:925 (2023); Ohkura et al.,
                           IEEE TQE 3 (2022). k independent circuits share one
                           chip. MEASURED here by transpiling k copies of a 9-qubit
                           QACT circuit onto FakeTorino and timing the packed
                           circuit. NOT measured: crosstalk -- the fake backends'
                           noise models do not contain it, and both papers report
                           a fidelity cost.
3. Adaptive shots          successive elimination, Huang & Izmaylov,
                           arXiv:2509.14917 (2025). Run for real in Aer:
                           selection screens every candidate circuit with S0 shots
                           and keeps adding shots (doubling, up to 2,000) only to
                           candidates whose upper confidence bound can still beat
                           the leader's lower bound; each 3-point refinement
                           comparison is raced the same way and stops once the
                           winner is clear. Undecided comparisons fall back to the
                           reference rule at 2,000 shots.
"""
import json
import math
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino

import qact_vs_act_speed as ref
from qact_vs_act_speed import (BAND, BATCH_ENV, FS, K, N, P, REP_DELAY, STEPS,
                               CircuitQACT, eeg_windows, env_of, n, width_for)
from qbe.qact import QACT
from qbe.qact_hw import build, circuit_duration, two_qubit_stats, window_register

S0, S_MAX, Z = 250, 2000, 2.0          # first-look shots, cap (= reference), z-score


class AdaptiveCircuitQACT(CircuitQACT):
    def __init__(self, q):
        super().__init__(q)
        self._seed = 10_000

    def _exec(self, tqs, shots):
        """Run pre-transpiled circuits; returns raw count vectors."""
        res = self.sim.run(tqs, shots=shots, seed_simulator=self._seed).result()
        self._seed += 1
        self.aer_time += res.time_taken
        out = []
        for i, tq in enumerate(tqs):
            m = tq.num_clbits
            self.log.append((m, shots))
            c = np.zeros(2 ** m)
            for key, v in res.get_counts(i).items():
                c[int(key, 2)] += v
            out.append(c)
        return out

    @staticmethod
    def _bounds(d):
        """Point estimate, LCB and UCB of the best in-band atom of candidate d."""
        h, s, sc, ok = d["counts"], d["shots"], d["sc"], d["ok"]
        pt = (h + 1) / (s + 2)                         # keeps zero-count bins honest
        crit = sc * h / s
        sig = sc * np.sqrt(pt * (1 - pt) / s)
        crit_m = np.where(ok, crit, -np.inf)
        d["j"] = int(np.argmax(crit_m))
        d["best"] = float(crit_m[d["j"]])
        d["lcb"] = float(np.where(ok, crit - Z * sig, -np.inf).max())
        d["ucb"] = float(np.where(ok, crit + Z * sig, -np.inf).max())

    def select(self, r):
        live = sorted(self.chirps_of)
        E = np.stack([env_of(self.env_tc[e], self.env_ld[e]) for e in live])
        bound = (E @ np.abs(r)) ** 2 / (E ** 2).sum(1)
        order = [live[i] for i in np.argsort(-bound)]
        bnd = dict(zip(live, bound))
        cands, best_lcb, i = [], -1.0, 0
        # screen: branch-and-bound over envelopes, pruning against the leader's
        # LOWER confidence bound so a noisy early estimate cannot prune the winner
        while i < len(order) and bnd[order[i]] > best_lcb:
            batch = [e for e in order[i:i + BATCH_ENV] if bnd[e] > best_lcb]
            i += BATCH_ENV
            new = []
            for e in batch:
                env = env_of(self.env_tc[e], self.env_ld[e])
                m, sc = self._scale(r, env)
                k = np.arange(2 ** m) * 2 ** (n - m)
                for ci in self.chirps_of[e]:
                    c, tc = self.c_t[ci], self.env_tc[e]
                    fc = (k + 2 * c * tc) * FS / N
                    new.append(dict(e=e, ci=ci, sc=sc, k=k,
                                    ok=(k < N // 2) & (fc >= BAND[0]) & (fc <= BAND[1]),
                                    qc=build("windowed", r, env, c, loader="initialize")))
            tqs = transpile([d["qc"] for d in new], self.sim, optimization_level=0)
            for d, tq, cnt in zip(new, tqs, self._exec(tqs, S0)):
                d.update(tq=tq, counts=cnt, shots=S0)
                if d["ok"].any():
                    self._bounds(d)
                    best_lcb = max(best_lcb, d["lcb"])
                    cands.append(d)
        # successive elimination: double the shots of every candidate that could
        # still be the best, drop the rest
        alive = [d for d in cands if d["ucb"] >= best_lcb]
        while len(alive) > 1 and alive[0]["shots"] < S_MAX:
            add = min(alive[0]["shots"], S_MAX - alive[0]["shots"])
            for d, cnt in zip(alive, self._exec([d["tq"] for d in alive], add)):
                d["counts"] = d["counts"] + cnt
                d["shots"] += add
                self._bounds(d)
            best_lcb = max(d["lcb"] for d in alive)
            alive = [d for d in alive if d["ucb"] >= best_lcb]
        d = max(alive, key=lambda d: d["best"])
        return (float(self.env_tc[d["e"]]), float(self.env_ld[d["e"]]),
                float(d["k"][d["j"]]), float(self.c_t[d["ci"]]))

    def refine(self, r, x):
        cache = {}

        def ensure(keys, S):
            fresh = [key for key in keys if key not in cache]
            if fresh:
                qcs = []
                for tc, ld, f, c in fresh:
                    env = env_of(tc, ld)
                    m, _ = window_register(env, n)
                    step = 2 ** (n - m)
                    kq = math.floor(f / step)
                    qcs.append(build("windowed", r, env, c, offset=f - kq * step,
                                     loader="initialize"))
                    cache[(tc, ld, f, c)] = dict(kq=kq % 2 ** m, sc=self._scale(r, env)[1],
                                                 hits=0.0, shots=0)
                for key, tq in zip(fresh, transpile(qcs, self.sim, optimization_level=0)):
                    cache[key]["tq"] = tq
            by_add = {}
            for key in keys:
                need = S - cache[key]["shots"]
                if need > 0:
                    by_add.setdefault(need, []).append(key)
            for add, ks in by_add.items():
                for key, cnt in zip(ks, self._exec([cache[q]["tq"] for q in ks], add)):
                    cache[key]["hits"] += cnt[cache[key]["kq"]]
                    cache[key]["shots"] += add

        def est(key):
            h, s, sc = cache[key]["hits"], cache[key]["shots"], cache[key]["sc"]
            pt = (h + 1) / (s + 2)
            return sc * h / s, sc * math.sqrt(pt * (1 - pt) / s)

        x = tuple(x)
        deltas = (1.0, 0.05, 0.5, self.lr_c)             # tc, logDt, f, c
        for _ in range(STEPS):
            for j, dlt in enumerate(deltas):
                arms = [x]
                for sgn in (-1.0, 1.0):
                    y = list(x)
                    y[j] += sgn * dlt
                    y[0] = min(max(y[0], 0.0), N - 1.0)
                    y[1] = min(max(y[1], 1.0), 6.0)
                    y[3] = min(max(y[3], -self.c_max), self.c_max)
                    fc = (y[2] + 2 * y[3] * y[0]) * FS / N
                    if j != 2 or BAND[0] <= fc <= BAND[1]:  # same band rule as reference
                        arms.append(tuple(y))
                S = S0
                while True:
                    ensure(arms, S)
                    e = {a: est(a) for a in arms}
                    top = max(arms, key=lambda a: e[a][0])
                    clear = all(e[top][0] - e[a][0] > Z * math.hypot(e[top][1], e[a][1])
                                for a in arms if a != top)
                    if clear or S >= S_MAX:
                        break
                    S = min(2 * S, S_MAX)
                x = top
        return x


def pack(backend, X, ks):
    """Transpile k copies of a 9-qubit QACT circuit onto one chip and time it."""
    rows = []
    for k in ks:
        # every copy a genuine 9-qubit circuit (a centred logDt 3.9 envelope
        # needs the full register), on different EEG windows
        circs = [build("windowed", X[i % len(X)], env_of(200.0 + 20 * (i % 5), 3.9),
                       0.03125) for i in range(k)]
        assert all(c.num_qubits == 9 for c in circs)
        tot_q = sum(c.num_qubits for c in circs)
        tot_c = sum(c.num_clbits for c in circs)
        big = QuantumCircuit(tot_q, tot_c)
        oq = oc = 0
        for c in circs:
            big.compose(c, qubits=range(oq, oq + c.num_qubits),
                        clbits=range(oc, oc + c.num_clbits), inplace=True)
            oq += c.num_qubits
            oc += c.num_clbits
        t0 = time.time()
        try:
            tq = transpile(big, backend=backend, optimization_level=3, seed_transpiler=7)
        except Exception as exc:                      # does not fit / cannot route
            rows.append(dict(k=k, ok=False, error=type(exc).__name__))
            print(f"   k={k:2d}: transpile failed ({type(exc).__name__})", flush=True)
            continue
        dur = circuit_duration(tq, backend.target)
        cz = two_qubit_stats(tq)[0]
        used = len({tq.find_bit(q).index for inst in tq.data for q in inst.qubits
                    if inst.operation.name not in ("barrier", "delay")})
        rows.append(dict(k=k, ok=True, duration_us=dur * 1e6, cz_per_copy=cz / k,
                         physical_qubits=used, transpile_s=time.time() - t0))
        print(f"   k={k:2d}: {used:3d} physical qubits, {cz / k:6.0f} CZ per copy, "
              f"{dur * 1e6:6.1f} us per shot for all {k} "
              f"-> {(dur + REP_DELAY) * 1e6 / k:6.1f} us per circuit-shot at default "
              f"delay, {dur * 1e6 / k:5.1f} at zero delay  ({time.time() - t0:.0f}s)",
              flush=True)
    return rows


def main():
    X = eeg_windows()
    backend = FakeTorino()
    base = json.load(open("results/qact_vs_act_speed.json"))
    ref_counts = {int(m): c for m, c in base["circuits_by_register"].items()}
    ref_err = next(r["recon_err"] for r in base["rows"] if "Aer" in r["engine"])
    print(f"reference (qact_vs_act_speed.py): {base['circuits_per_window']:,.0f} circuits x "
          f"{base['shots']} shots per window, recon err {ref_err:.3f}\n", flush=True)

    # ---- 3. adaptive shots, run for real in Aer ----
    q = QACT(length=N, fs=FS, device="cuda", seed=0, refine_steps=STEPS,
             omp=True, backfit_passes=0, exact_f=False, refine_mode="hw")
    aq = AdaptiveCircuitQACT(q)
    t0, errs = time.perf_counter(), []
    for i, x in enumerate(X):
        errs.append(aq.decompose(x))
        print(f"   adaptive window {i + 1}/{K}: recon err {errs[-1]:.3f}, "
              f"{sum(s for _, s in aq.log):,} shots so far "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    shots_ad = sum(s for _, s in aq.log) / K
    shots_ref = base["circuits_per_window"] * base["shots"]
    print(f"adaptive shots: {shots_ad:,.0f} shots/window vs {shots_ref:,.0f} "
          f"({shots_ref / shots_ad:.1f}x fewer); recon err {np.mean(errs):.3f} vs "
          f"{ref_err:.3f}", flush=True)

    # ---- 2. multi-programming, measured by packing k copies on the chip ----
    print("\nmulti-programming on FakeTorino (133 qubits), 9-qubit QACT circuits:",
          flush=True)
    packing = pack(backend, X, (1, 2, 4, 8, 12, 14))
    fits = [p for p in packing if p["ok"]]
    thr = lambda p, rep: (p["duration_us"] * 1e-6 + rep) / p["k"]   # s per circuit-shot
    best0 = min(fits, key=lambda p: thr(p, 0.0))
    best250 = min(fits, key=lambda p: thr(p, REP_DELAY))

    # ---- price every configuration from the same device durations ----
    ms = sorted({m for m, _ in aq.log} | set(ref_counts))
    dur = {}
    for m in ms:
        tq = transpile(build("windowed", X[0], env_of(256.0, width_for(m)), 0.03125),
                       backend=backend, optimization_level=3, seed_transpiler=7)
        dur[m] = circuit_duration(tq, backend.target)
    shots_by_m_ref = {m: ref_counts[m] * base["shots"] for m in ref_counts}
    shots_by_m_ad = {}
    for m, s in aq.log:
        shots_by_m_ad[m] = shots_by_m_ad.get(m, 0) + s / K

    def price(shots_by_m, rep, packed=None):
        if packed is None:
            return sum(s * (dur[m] + rep) for m, s in shots_by_m.items())
        # every circuit rides in a k-way packed job; the packed 9-qubit duration
        # is used for all register sizes (conservative for the smaller ones)
        return sum(shots_by_m.values()) * thr(packed, rep)

    rows = [
        ("optimised circuits (last benchmark)", price(shots_by_m_ref, REP_DELAY)),
        ("+ zero repetition delay", price(shots_by_m_ref, 0.0)),
        (f"+ multi-programming (k={best0['k']})", price(shots_by_m_ref, 0.0, best0)),
        ("+ adaptive shots (all three)", price(shots_by_m_ad, 0.0, best0)),
        (f"all three, default 250 us delay (k={best250['k']})",
         price(shots_by_m_ad, REP_DELAY, best250)),
    ]
    cls = {r["engine"]: r["seconds_per_window"] for r in base["rows"]}
    cpu, gpu = cls["classical ACT, CPU"], next(r["batched_seconds_per_window"]
                                               for r in base["rows"]
                                               if r["engine"] == "classical ACT, GPU")
    fmt = lambda x: f"{x * 1e3:,.1f} ms" if x < 1 else (
        f"{x:,.1f} s" if x < 60 else f"{x / 60:,.1f} min")
    print(f"\n{'QACT on IBM Heron (estimate)':<44}{'per window':>12}{'vs CPU ACT':>14}"
          f"{'vs GPU ACT':>14}")
    for name, s in rows:
        print(f"{name:<44}{fmt(s):>12}{s / cpu:>13,.0f}x{s / gpu:>13,.0f}x")
    print(f"{'classical ACT, CPU (measured)':<44}{fmt(cpu):>12}")
    print(f"{'classical ACT, GPU batched (measured)':<44}{fmt(gpu):>12}")

    json.dump(dict(adaptive=dict(shots_per_window=shots_ad, recon_err=float(np.mean(errs)),
                                 per_window_err=[float(e) for e in errs],
                                 reference_shots_per_window=shots_ref,
                                 reference_recon_err=ref_err,
                                 aer_seconds=aq.aer_time, S0=S0, S_MAX=S_MAX, Z=Z),
                   packing=packing,
                   duration_us={str(m): dur[m] * 1e6 for m in ms},
                   estimates={name: s for name, s in rows},
                   classical=dict(cpu=cpu, gpu_batched=gpu)),
              open("results/qact_hw_speedups.json", "w"), indent=1)
    print("\nwrote results/qact_hw_speedups.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Data that benefits quantum: chirplet lock-in on real brain fields from quantum sensors.

Recorded data -- even from quantum sensors -- reaches a computer as classical
numbers, so it gives quantum processing no advantage. Quantum advantage for
signals lives in the SENSING: probe spins whose quantum state the field changes,
processed before measurement. This asks whether that advantage would help detect
the chirplet atoms of a real evoked brain response.

Data
  MNE OPM sample (mne.datasets.opm): somatosensory evoked fields, 9 QuSpin
  optically pumped magnetometers, 201 median-nerve stimulations, 1 kHz, plus an
  OPM empty-room recording (sensor + environment, no brain).
  Zhang et al., Nat. Commun. (2025), doi:10.1038/s41467-025-66828-z, raw data
  (Zenodo 17614133): entanglement-enhanced quantum lock-in detection with two
  trapped ions, GHZ vs product states.

PRE-REGISTERED (written before running):
  1 anchor   from the lock-in data, the achieved precision ratio product/entangled
             at N = 2 (ideal: sqrt 2 = 1.414), for n = 20 and n = 30 pulses
  2 atoms    evoked field (mean of 201 trials) on the channel with the largest
             response, 256 ms after the stimulus, decomposed into 6 chirplet atoms
             by the project's ACT engine (OMP, refinement); each atom's unit-norm
             waveform is its matched lock-in reference m_k
  3 noise    per trial, projected on m_k:  total (single-trial residuals),
             sensor+environment (empty-room segments), brain background = the
             difference. PRIMARY scenario, generous to quantum: all empty-room
             noise is sensor noise that better sensors could remove.
  4 sensing  N probe spins, Rb-87 gamma = 2 pi x 7 Hz/nT, sensing time T_k = the
             atom's window, dephasing time T2 in {1, 10, 100, 1000 ms, infinite}.
             Phase noise per trial:  product states (SQL)  e^(T/T2)/sqrt(N);
             GHZ in blocks of m spins (the known optimum under uncorrelated
             dephasing, m chosen per point)  min_m e^(m T/T2)/sqrt(m N).
             Trials to detect atom k at 5 sigma:
                 R = 25 (sigma_brain^2 + (dPhi/gamma)^2) / signal_k^2
  5 check    Qiskit Aer density-matrix simulation of the lock-in circuit (N = 1..5,
             GHZ vs product, with and without dephasing) must reproduce the phase-
             noise formulas of step 4.
  claim      quantum sensing BENEFITS a regime if GHZ blocks need >= 10% fewer
             trials than product states with the same N, T and T2.
"""
import glob
import json
import time

import mne
import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error

from qbe.qact import QACT

mne.set_log_level("ERROR")
OPM = "/home/kc/datasets/opm/MNE-OPM-data/MEG/OPM"
QLID = glob.glob("/home/kc/datasets/qlid/Exp*/maintext/Fig2")[0]
FS, N_WIN, K = 1000.0, 256, 6
GAMMA = 2 * np.pi * 7e9            # rad s^-1 T^-1, Rb-87 ground state (7 Hz/nT)
T2S = (1e-3, 1e-2, 1e-1, 1.0, np.inf)
NS = np.logspace(0, 14, 29)


# ------------------------------------------------------------------ 1. anchor
def _precision(td, obs):
    """Precision at the working point = shot std / |slope| on the steepest flanks.

    The scans are resonances (the GHZ one a narrow dip, the product one a broad
    peak), so no single smooth fit suits both. The slope is the central difference
    of neighbouring means; the working points are the 4 with the steepest slope --
    chosen on the slope alone, never on the noisy std/slope ratio (picking the best
    ratios, as the first version did, selects noise minima and inflates the gain).
    """
    mu = np.array([o.mean() for o in obs])
    sd = np.array([o.std(ddof=1) for o in obs])
    slope = np.abs(mu[2:] - mu[:-2]) / (td[2:] - td[:-2])
    w = np.argsort(-slope)[:4]
    return float(np.sqrt((sd[1:-1][w] ** 2).mean()) / slope[w].mean())


def lockin_anchor(boot=300, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for n in (20, 30):
        data = {}
        for state in ("Entangled state", "Product state"):
            files = sorted(glob.glob(f"{QLID}/n={n}/{state}/*.txt"),
                           key=lambda f: float(f.rsplit("/", 1)[1][:-4]))
            td = np.array([float(f.rsplit("/", 1)[1][:-4]) for f in files])
            S = [np.loadtxt(f) for f in files]
            data[state] = (td, [d[:, 1] + d[:, 2] - d[:, 3] for d in S])  # authors' observable
        prec = {st: _precision(*data[st]) for st in data}
        gains = []
        for _ in range(boot):                   # resample repetitions within each point
            b = {st: _precision(td, [o[rng.integers(0, len(o), len(o))] for o in obs])
                 for st, (td, obs) in data.items()}
            gains.append(b["Product state"] / b["Entangled state"])
        out[n] = dict(entangled_us=prec["Entangled state"], product_us=prec["Product state"],
                      gain=prec["Product state"] / prec["Entangled state"],
                      gain_ci95=[float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
                      points=len(data["Product state"][0]),
                      reps=int(np.median([len(o) for o in data["Product state"][1]])))
    return out


# ------------------------------------------------------------------ 2-3. data
def load_epochs(fname, events=True):
    raw = mne.io.read_raw_fif(f"{OPM}/{fname}", preload=True)
    raw.filter(3.0, 150.0).notch_filter([50, 100, 150])
    picks = mne.pick_types(raw.info, meg=True)
    if not events:
        return raw.get_data(picks=picks) * 1e15                     # fT
    ev = mne.find_events(raw, stim_channel="STI101", min_duration=0.002)
    ev = ev[ev[:, 2] == 257]
    ep = mne.Epochs(raw, ev, tmin=-0.05, tmax=(N_WIN - 1) / FS, baseline=(-0.05, 0.0),
                    picks=picks, preload=True, reject=None)
    X = ep.get_data() * 1e15                                        # (trials, ch, t) in fT
    return X[:, :, ep.times >= 0][:, :, :N_WIN], [raw.ch_names[p] for p in picks]


def atoms_of(evoked):
    q = QACT(length=N_WIN, fs=FS, band_hz=(3.0, 150.0), device="cuda", refine_steps=4,
             omp=True, backfit_passes=1, exact_f=True)
    raw, err = q.transform(evoked[None].astype(np.float32), order=K, select="argmax",
                           return_raw=True)
    W = q.rebuild(raw.reshape(-1, raw.shape[-1])).cpu().numpy()     # (K, N_WIN) components
    info = []
    for k, (row, w) in enumerate(zip(raw[0], W)):
        tc, ld = row[0], row[1]
        width = np.exp(ld)                                          # samples
        fc = (row[2] + 2 * row[3] * tc) * FS / N_WIN
        info.append(dict(t_ms=float(tc), width_ms=float(width), f_hz=float(fc),
                         T_s=float(min(N_WIN, 4 * width) / FS)))
    return W, info, float(err[0])


# ------------------------------------------------------------------ 4. sensing model
def phase_noise(N, T, T2, entangled):
    """Phase-noise std accumulated over a sensing window T, for N probe spins.

    A real sensor does not hold one coherent measurement across a window much
    longer than its coherence time: it repeats interrogations of length tau <= T
    and sums them (std grows as sqrt(T / tau)). Each interrogation costs
    e^(m tau / T2) / sqrt(m N) for GHZ blocks of m spins (m = 1: independent
    spins). tau and m are both optimised. Under this uncorrelated dephasing the
    optimum for any m with tau < T is sqrt(2 e T / (N T2)) -- entanglement gives no
    gain (Huelga et al. 1997) -- unless the window is shorter than the coherence
    time, where tau = T is forced and GHZ blocks can win.
    """
    if np.isinf(T2):
        return 1.0 / N if entangled else 1.0 / np.sqrt(N)          # Heisenberg vs SQL
    taus = np.logspace(np.log10(T) - 6, np.log10(T), 400)
    if not entangled:
        ms = np.array([1.0])
    else:
        ms = np.unique(np.clip(np.round(np.logspace(0, np.log10(max(N, 1)), 120)), 1, N))
    best = np.inf
    for m in ms:
        f = np.exp(np.minimum(m * taus / T2, 700.0)) / np.sqrt(m * N) * np.sqrt(T / taus)
        best = min(best, float(f.min()))
    return best


# ------------------------------------------------------------------ 5. circuit check
def circuit_phase_noise(N, T, T2, entangled, phi=1e-3, steps=16):
    """Aer density-matrix simulation of a lock-in measurement: spins accumulate phase
    phi*(per-step modulation), dephase each step, then a parity (GHZ) or single-spin
    (product) readout. Returns the single-shot phase std = sqrt(Var)/|d<obs>/dphi|."""
    def expval(ph):
        qc = QuantumCircuit(N)
        qc.h(0)
        if entangled:
            for i in range(1, N):
                qc.cx(0, i)
        else:
            for i in range(1, N):
                qc.h(i)
        for _ in range(steps):
            for i in range(N):
                qc.rz(ph / steps, i)
                qc.id(i)
        if entangled:                                               # parity readout
            for i in range(N - 1, 0, -1):
                qc.cx(0, i)
        qc.h(0)
        qc.save_density_matrix(qubits=[0])
        nm = NoiseModel()
        if np.isfinite(T2):
            lam = 1 - np.exp(-2 * (T / steps) / T2)                  # coherence e^{-dt/T2}
            nm.add_all_qubit_quantum_error(phase_damping_error(lam), ["id"])
        sim = AerSimulator(method="density_matrix", noise_model=nm)
        rho = sim.run(transpile(qc, sim, optimization_level=0)).result().data()["density_matrix"]
        p0 = float(np.real(np.asarray(rho)[0, 0]))
        return 2 * p0 - 1                                           # <Z> of the readout qubit
    # operate at the steepest point of the fringe: a GHZ state accumulates N times
    # the single-spin phase, so its fringe is N times faster
    ph0 = np.pi / (2 * N) if entangled else np.pi / 2
    e0, e1 = expval(ph0 - phi), expval(ph0 + phi)
    slope = abs(e1 - e0) / (2 * phi)
    e = expval(ph0)
    var = 1 - e ** 2
    if entangled:
        return np.sqrt(var) / slope                                 # one parity shot
    return np.sqrt(var) / slope / np.sqrt(N)                        # N independent spins


def main():
    t0 = time.time()
    res = {}
    # 1
    anchor = lockin_anchor()
    res["anchor"] = anchor
    for n, a in anchor.items():
        print(f"[anchor] lock-in data n={n}: precision product {a['product_us']:.4f} us, "
              f"entangled {a['entangled_us']:.4f} us -> gain {a['gain']:.3f} "
              f"[95% CI {a['gain_ci95'][0]:.3f}-{a['gain_ci95'][1]:.3f}] "
              f"(ideal sqrt2 = 1.414; {a['points']} points x {a['reps']} reps)", flush=True)
    # 2-3
    X, chans = load_epochs("OPM_SEF_raw.fif")
    evoked = X.mean(0)
    ch = int(np.argmax(np.sqrt((evoked[:, 10:150] ** 2).mean(1))))
    ev = evoked[ch]
    W, info, err = atoms_of(ev)
    er_all = load_epochs("OPM_empty_room_raw.fif", events=False)
    assert er_all.shape[0] == X.shape[1], "empty-room sensor set differs"
    er = er_all[ch]
    segs = np.stack([er[i:i + N_WIN] for i in range(0, len(er) - N_WIN, N_WIN)])
    resid = X[:, ch] - ev
    print(f"[data] {X.shape[0]} trials, channel {chans[ch]}; evoked peak "
          f"{np.abs(ev).max():.0f} fT; 6-atom ACT residual {err:.3f}; "
          f"{len(segs)} empty-room segments", flush=True)
    atoms = []
    for k in range(K):
        w = W[k] / max(np.linalg.norm(W[k]), 1e-12)
        dt = 1 / FS
        sig = float(abs(w @ ev) * dt)                               # fT*s
        tot = float(np.std(resid @ w) * dt)
        env = float(np.std(segs @ w) * dt)
        # if the empty-room projection exceeds the total, the two recordings do not
        # match for this atom (e.g. slow drift); flag it and keep a floor instead of
        # declaring the brain noiseless
        mismatch = env >= tot
        brain = float(np.sqrt(max(tot ** 2 - env ** 2, (0.25 * tot) ** 2)))
        a = dict(info[k], signal_fTs=sig, noise_total_fTs=tot, noise_emptyroom_fTs=env,
                 recordings_mismatch=bool(mismatch),
                 noise_brain_fTs=brain, trials_real_OPM=float(max(1, 25 * tot ** 2 / sig ** 2)),
                 trials_perfect_sensor=float(max(1, 25 * brain ** 2 / sig ** 2)))
        atoms.append(a)
        print(f"  atom {k}: t {a['t_ms']:5.1f} ms, width {a['width_ms']:5.1f} ms, "
              f"f {a['f_hz']:5.1f} Hz | signal {sig:7.3f} fT*s  noise: total {tot:6.3f}, "
              f"empty-room {env:6.3f}, brain {brain:6.3f} | trials at 5 sigma: real OPM "
              f"{a['trials_real_OPM']:7.1f}, perfect sensor {a['trials_perfect_sensor']:7.1f}"
              + ("   [empty-room > total: noise split unreliable]" if mismatch else ""),
              flush=True)
    res["atoms"] = atoms
    # 4
    grid = []
    for a in atoms:
        for T2 in T2S:
            for N in NS:
                row = {}
                for ent in (False, True):
                    dphi = phase_noise(N, a["T_s"], T2, ent)
                    sens = dphi / GAMMA * 1e15                       # fT*s
                    row["ghz" if ent else "product"] = max(
                        1.0, 25 * (a["noise_brain_fTs"] ** 2 + sens ** 2) / a["signal_fTs"] ** 2)
                grid.append(dict(t_ms=a["t_ms"], T_s=a["T_s"], T2=None if np.isinf(T2) else T2,
                                 N=float(N),
                                 trials_product=row["product"], trials_ghz=row["ghz"],
                                 gain=row["product"] / row["ghz"]))
    res["grid"] = grid
    print("\n[sensing] entanglement gain in trials (independent / GHZ-block), best over N,"
          " per atom (T = the atom's sensing window):")
    print(f"  {'atom':<6}{'window':>8}" + "".join(
        f"{('no deph.' if np.isinf(T2) else f'T2={T2 * 1e3:g}ms'):>13}" for T2 in T2S))
    for k, a in enumerate(atoms):
        if a["recordings_mismatch"]:
            continue
        cells = []
        for T2 in T2S:
            rows = [r for r in grid if r["t_ms"] == a["t_ms"] and r["T_s"] == a["T_s"] and
                    (r["T2"] is None if np.isinf(T2) else r["T2"] == T2)]
            cells.append(max(r["gain"] for r in rows))
        print(f"  {k:<6}{a['T_s'] * 1e3:>6.0f}ms" + "".join(f"{c:>12.2f}x" for c in cells))
    print("  (a gain is capped by the brain-noise floor: once sensor noise is negligible,"
          " every sensor needs the same number of trials)")
    # 5
    print("\n[check] Aer lock-in circuit vs formula (phase std per trial, T = 50 ms):")
    chk = []
    for T2 in (np.inf, 0.2):
        for N in (1, 2, 3, 4, 5):
            for ent in (False, True):
                sim = circuit_phase_noise(N, 0.05, T2, ent)
                g = 0.0 if np.isinf(T2) else 0.05 / T2
                formula = (np.exp(N * g) / N) if ent else (np.exp(g) / np.sqrt(N))
                chk.append(dict(N=N, T2=None if np.isinf(T2) else T2, entangled=ent,
                                aer=float(sim), formula=float(formula)))
        rows = [c for c in chk if (c["T2"] is None) == np.isinf(T2)]
        print(f"  T2 = {'inf' if np.isinf(T2) else T2}: " + "  ".join(
            f"N={c['N']}{'G' if c['entangled'] else 'P'} {c['aer']:.3f}/{c['formula']:.3f}"
            for c in rows), flush=True)
    res["circuit_check"] = chk
    worst = max(abs(c["aer"] - c["formula"]) / c["formula"] for c in chk)
    print(f"  max relative deviation Aer vs formula: {worst:.2e}")
    json.dump(res, open("results/quantum_sensing_opm.json", "w"), indent=1)
    print(f"\nwrote results/quantum_sensing_opm.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

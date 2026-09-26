#!/usr/bin/env python
"""Design, preregister and dry-run ONE out-of-sample hardware batch for the fidelity-gap law.

The law (notes/fidelity_gap_law.md): a device histogram of the windowed QACT selection
circuit is modelled as Multinomial(S, q), q = F p + (1 - F)/D over the D = 2^m register
outcomes, with p the exact distribution and mirror/out-of-band bins masked after sampling.
On the Sep 25 ibm_marrakesh run (26 circuits) the law was checked in sample. This script
builds a NEW batch on EEG windows the Sep 25 run never used, predicts every circuit's
outcome BEFORE any device data exist, and writes the preregistration.

Stages (all CPU; nothing is submitted):
  1. Sep 25 rows: rebuild exact p, device F_xeb and FakeMarrakesh F_xeb per row; choose the
     a-priori F predictor by leave-one-out Brier score on those 26 rows and freeze it.
  2. Candidate pool: PhysioNet EEGMMIDB subjects 4-20, baseline runs R01 (eyes open) and
     R02 (eyes closed), 8 channels, non-overlapping 512-sample windows. Exact distribution
     and signal statistics (gap, participation ratio) at logDt 1.5 / 2.7 / 3.3 (6 / 7 / 8 q).
  3. Transpile representatives for the target backend (open-plan instance, selected by CRN)
     to get CZ counts, predict F per (qubits, variant), predict P(top-1) for every pool
     window, and pick windows that span predicted success from low to high.
  4. Build every selected circuit, transpile it for the target backend (seed 7, opt 3),
     check it against the exact target on a noiseless simulator, simulate it under the
     FakeMarrakesh noise model, and compute the law's predictions (Monte Carlo with the
     scoring rule's tie convention, plus the closed form of the note as a check).
  5. Dry run: estimate QPU seconds from the Sep 25 run (26 x 4000 shots = 36 s), run the
     frozen evaluation on the FakeMarrakesh histograms as a mock device, and estimate the
     power of the pass rules by simulating whole batches from the law.
  6. Write results/qpu_law_batch_design.json, results/qpu_law_batch_circuits.qpy and
     PREREG_QPU_LAW.md.

After the device run, the frozen analysis is

    python paper/iclr2027/exp/qpu_law_design.py --evaluate <device_results.json>

where the results file holds {"pubs": [{"id": ..., "counts": {bitstring: n}, "shots": n,
"job_id": ...}, ...]} with ids from the design json.

    python paper/iclr2027/exp/qpu_law_design.py            # design + dry run, no submission
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import qpu_law_check as law  # noqa: E402  (f_xeb, model_q, closed_form, qs: exact targets)
from qpu_law_check import qs  # noqa: E402

RES = ROOT / "paper" / "iclr2027" / "results"
PREREG = ROOT / "paper" / "iclr2027" / "PREREG_QPU_LAW.md"
OUT_JSON = RES / "qpu_law_batch_design.json"
OUT_QPY = RES / "qpu_law_batch_circuits.qpy"
SEP25 = RES / "qpu_selection_marrakesh1.json"
FAKE25 = RES / "qpu_law_fake_counts.json"
CACHE = Path(tempfile.gettempdir()) / "qpu_law_design_cache"

FS, N, NQ = qs.FS, qs.N, qs.n
C = 8.0 * N / (2 * FS ** 2)          # chirp rate 8 Hz/s, as in qpu_selection.py
LOGDT = {6: 1.5, 7: 2.7, 8: 3.3, 9: 3.9}
SHOTS = 4000
SHOTS_SCALE = 16000
SEP25_QPU_S, SEP25_SHOTS = 36.0, 26 * 4000
QPU_CAP_S = 120.0
OPEN_CRN_SUFFIX = "6223ffe6126d"
SUBJECTS = tuple(range(4, 21))
RUNS = (1, 2)
CHANNELS = ("O1", "Oz", "O2", "Pz", "Cz", "C3", "C4", "Fz")
N_PRIMARY = {6: 12, 7: 12, 8: 4}
F_MIN_8Q = 0.05
MC_REPS = 20000
SEED = 20260926
CONST_TOP1 = 13 / 26                  # Sep 25 base rates: the preregistered constant predictor
CONST_TOP3 = 16 / 26
RHO_MIN, PERM_ALPHA, TOP3_TOL = 0.40, 0.05, 0.15
N_PERM = 10000
ANCHORS = ("1|Cz|1.5|uni", "2|Cz|2.7|uni")


# ============================================================================ helpers
def counts_vec(counts, m):
    v = np.zeros(2 ** m)
    for key, n in counts.items():
        v[int(key.split()[-1], 2)] += n
    return v


def stable_rank(v, idx, mask):
    """Rank of idx among masked entries of v: larger counts, then equal counts at a lower
    index, come first (np.argmax's tie rule, the rule qpu_selection.score uses)."""
    w = np.where(mask, v, -1)
    return int(np.sum(w > w[idx]) + np.sum(w[:idx] == w[idx]) + 1)


def inband_stats(p, mask, peak):
    pin = np.sort(p[mask])[::-1]
    pm = p[mask] / p[mask].sum()
    return dict(p1=float(pin[0]), p2=float(pin[1]), gap=float(pin[0] - pin[1]),
                gap_rel=float((pin[0] - pin[1]) / pin[0]), pr_inband=float(1 / np.sum(pm ** 2)),
                p_band=float(p[mask].sum()), M_inband=int(mask.sum()), p1_full=float(p[peak]))


def p_top1_fast(q, mask, peak, S):
    """Closed form (3) of the note, vectorised over the Gauss-Hermite nodes (top-1 only)."""
    others = np.flatnonzero(mask & (np.arange(len(q)) != peak))
    q1 = q[peak]
    qj = q[others] / (1 - q1)
    n1 = S * q1 + np.sqrt(S * q1 * (1 - q1)) * law.GH_X
    rest = np.maximum(S - n1, 0.0)
    mu = rest[:, None] * qj[None, :]
    sd = np.sqrt(np.maximum(mu * (1 - qj[None, :]), 1e-12))
    pj = norm.cdf((mu - n1[:, None]) / sd)
    return float(np.sum(law.GH_W * np.prod(1 - pj, axis=1)))


def mc_rank(q, mask, peak, S, reps=MC_REPS, seed=0, ks=(1, 3, 5)):
    """Multinomial MC of the law with the scoring rule's tie convention: P(rank <= k),
    E[rank], median rank."""
    rng = np.random.default_rng(seed)
    qn = q / q.sum()
    ranks = []
    for start in range(0, reps, 2000):
        X = rng.multinomial(S, qn, size=min(2000, reps - start)).astype(np.int64)
        X[:, ~mask] = -1
        xp = X[:, peak:peak + 1]
        ranks.append((X > xp).sum(1) + (X[:, :peak] == xp).sum(1) + 1)
    r = np.concatenate(ranks)
    out = {f"p_top{k}": float(np.mean(r <= k)) for k in ks}
    out.update(exp_rank=float(r.mean()), median_rank=float(np.median(r)))
    return out


def s_star(q, p, mask, peak, F):
    i2 = int(np.flatnonzero(mask)[np.argsort(p[mask])[::-1][1]])
    q1, q2 = q[peak], q[i2]
    g = F * (p[peak] - p[i2])
    return float(law.Z95 ** 2 * (q1 + q2 - (q1 - q2) ** 2) / g ** 2) if g > 0 else float("inf")


def spearman_perm(pred, obs, n_perm=N_PERM, seed=SEED):
    """Spearman rho and a one-sided permutation p-value (rho >= observed)."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    if np.std(pred) == 0 or np.std(obs) == 0:
        return 0.0, 1.0
    rho = float(spearmanr(pred, obs).statistic)
    rng = np.random.default_rng(seed)
    from scipy.stats import rankdata
    a = rankdata(pred)
    b = rankdata(obs)
    a = (a - a.mean()) / np.linalg.norm(a - a.mean())
    b = (b - b.mean()) / np.linalg.norm(b - b.mean())
    perms = np.array([rng.permutation(b) for _ in range(n_perm)])
    null = perms @ a
    return rho, float((1 + np.sum(null >= rho - 1e-12)) / (n_perm + 1))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ============================================================================ stage 1
def sep25_rows():
    src = json.load(open(SEP25))
    fake = json.load(open(FAKE25))
    wins = {(w["subject"], w["channel"]): w for w in qs.eeg_windows()}
    rows = []
    for r in src["rows"]:
        cs = qs.make_case(wins[(r["subject"], r["channel"])], r["logdt"], C, r["variant"])
        assert cs["peak"] == r["want_peak_index"], "Sep 25 rebuild mismatch"
        m = cs["qubits"]
        key = f"{r['subject']}|{r['channel']}|{r['logdt']}|{r['variant']}"
        dv = counts_vec(r["device"]["counts"], m)
        fv = counts_vec(fake[key], m)
        p, mask, peak = cs["want"], cs["mask"], cs["peak"]
        rows.append(dict(key=key, m=m, variant=r["variant"], cz=int(r["cz_device"]),
                         p=p, mask=mask, peak=peak, shots=int(dv.sum()),
                         F_dev=law.f_xeb(dv / dv.sum(), p), F_fake=law.f_xeb(fv / fv.sum(), p),
                         top1=int(stable_rank(dv, peak, mask) == 1),
                         top3=int(stable_rank(dv, peak, mask) <= 3), x=wins[(r["subject"], r["channel"])]["x"],
                         logdt=r["logdt"], subject=r["subject"], channel=r["channel"]))
    return rows, src


class Predictor:
    """F predictors that need nothing from the device run they predict."""
    NAMES = ("cz_fit_variant", "cz_fit_pooled", "fake_raw", "fake_ratio", "fake_linear")

    def __init__(self, name, train):
        self.name = name
        Fd = np.array([t["F_dev"] for t in train])
        Ff = np.array([t["F_fake"] for t in train])
        cz = np.array([t["cz"] for t in train], float)
        var = np.array([t["variant"] for t in train])
        self.par = {}
        if name == "cz_fit_variant":
            for v in ("dyn", "uni"):
                ok = (var == v) & (Fd > 0)
                b, a = np.polyfit(cz[ok], np.log(Fd[ok]), 1)
                self.par[v] = (float(math.exp(a)), float(-b))
        elif name == "cz_fit_pooled":
            ok = Fd > 0
            b, a = np.polyfit(cz[ok], np.log(Fd[ok]), 1)
            self.par["all"] = (float(math.exp(a)), float(-b))
        elif name == "fake_ratio":
            self.par["ratio"] = float(Fd.sum() / Ff.sum())
        elif name == "fake_linear":
            b, a = np.polyfit(Ff, Fd, 1)
            self.par["a"], self.par["b"] = float(a), float(b)

    def __call__(self, cz, variant, F_fake=None):
        n = self.name
        if n == "cz_fit_variant":
            A, e = self.par[variant]
            F = A * math.exp(-e * cz)
        elif n == "cz_fit_pooled":
            A, e = self.par["all"]
            F = A * math.exp(-e * cz)
        elif n == "fake_raw":
            F = F_fake
        elif n == "fake_ratio":
            F = self.par["ratio"] * F_fake
        else:
            F = self.par["a"] + self.par["b"] * F_fake
        return float(np.clip(F, 1e-3, 1.0))


def choose_predictor(rows):
    """Leave-one-out on the 26 Sep 25 rows. Criterion (fixed before looking): lowest LOO
    Brier score of the law's top-1 prediction; ties broken by LOO mean |F error|."""
    out = {}
    for name in Predictor.NAMES:
        b1, b3, ferr, logres, logres67 = [], [], [], [], []
        for i, r in enumerate(rows):
            P = Predictor(name, rows[:i] + rows[i + 1:])
            F = P(r["cz"], r["variant"], r["F_fake"])
            cf = law.closed_form(law.model_q(r["p"], F), r["mask"], r["peak"], r["shots"], ks=(1, 3))
            b1.append((cf["p_top1_cond"] - r["top1"]) ** 2)
            b3.append((cf["p_top3_cond"] - r["top3"]) ** 2)
            ferr.append(abs(F - r["F_dev"]))
            if r["F_dev"] > 0:
                logres.append(math.log(r["F_dev"] / F))
                if r["m"] <= 7:
                    logres67.append(math.log(r["F_dev"] / F))
        out[name] = dict(loo_brier_top1=float(np.mean(b1)), loo_brier_top3=float(np.mean(b3)),
                         loo_mae_F=float(np.mean(ferr)), loo_log_residuals=[float(v) for v in logres],
                         loo_log_residual_sd=float(np.std(logres)),
                         loo_log_residual_sd_6_7q=float(np.std(logres67)))
    best = min(Predictor.NAMES, key=lambda n: (round(out[n]["loo_brier_top1"], 6), out[n]["loo_mae_F"]))
    return best, out


# ============================================================================ stage 2
def eeg_pool():
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "pool_windows.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["meta"]), z["X"]
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    data_dir = str(Path.home() / "mne_data")
    meta, X = [], []
    for s in SUBJECTS:
        for run in RUNS:
            f = eegbci.load_data(s, [run], path=data_dir, update_path=False, verbose=False)[0]
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            eegbci.standardize(raw)
            if abs(raw.info["sfreq"] - FS) > 1e-6:
                continue
            chans = [ch for ch in CHANNELS if ch in raw.ch_names]
            data = raw.get_data(picks=chans)
            for st in range(int(FS), raw.n_times - N - int(FS) + 1, N):
                for ci, ch in enumerate(chans):
                    x = data[ci, st:st + N].astype(np.float64)
                    x = x - x.mean()
                    if np.abs(x).max() == 0:
                        continue
                    meta.append((s, run, ch, st))
                    X.append(x / np.abs(x).max())
    X = np.array(X)
    np.savez(cache, meta=np.array(meta, dtype=object), X=X)
    return meta, X


def pool_stats(meta, X):
    out = {m: [] for m in (6, 7, 8)}
    for m, ld in ((6, 1.5), (7, 2.7), (8, 3.3)):
        for i, x in enumerate(X):
            cs = qs.make_case(dict(x=x), ld, C, "uni")
            assert cs["qubits"] == m
            st = inband_stats(cs["want"], cs["mask"], cs["peak"])
            out[m].append(dict(idx=i, peak=cs["peak"], **st))
    return out


# ============================================================================ backend
def open_backend(name, offline=False):
    """The target backend through the free open-plan instance, selected explicitly by CRN
    (IBM_QUANTUM_INSTANCE points at an exhausted paid instance and is never used)."""
    if offline:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        return FakeMarrakesh(), None, dict(offline=True)
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    if not tok:
        sys.exit("IBM_QUANTUM_TOKEN not set (use --offline for a FakeMarrakesh-only design)")
    svc0 = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
    crn = [dict(i)["crn"] for i in svc0.instances()
           if str(dict(i)["crn"]).rstrip(":").endswith(OPEN_CRN_SUFFIX)]
    if not crn:
        sys.exit("open-plan instance not found")
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=crn[0])
    be = svc.backend(name)
    info = dict(offline=False, instance_crn_suffix=OPEN_CRN_SUFFIX)
    try:
        info["calibration_last_update"] = str(be.properties().last_update_date)
    except Exception:
        info["calibration_last_update"] = None
    try:
        u = dict(svc.usage())
        info["usage_remaining_seconds_at_design"] = u.get("usage_remaining_seconds")
        info["usage_period"] = u.get("usage_period")
    except Exception:
        pass
    return be, svc, info


def isa_stats(isa, target, variant):
    from qbe.qact_hw import two_qubit_stats
    cz, czd, d = two_qubit_stats(isa)
    dur = None
    if variant == "uni":
        try:
            dur = float(isa.estimate_duration(target, unit="s"))
        except Exception:
            dur = None
    layout = None
    try:
        layout = [int(q) for q in isa.layout.final_index_layout()]
    except Exception:
        pass
    return dict(cz=int(cz), cz_depth=int(czd), depth=int(d), duration_s=dur, layout=layout)


def qpu_est(shots, variant, dur):
    """qpu_selection.py's per-shot model: circuit duration (1.2e-4 s placeholder for the
    dynamic variant, whose feed-forward has no calibrated duration) + 250 us delay."""
    return shots * ((dur if (variant == "uni" and dur) else 1.2e-4) + 2.5e-4)


# ============================================================================ design
def select_primary(pool, meta, F_by_m, shots):
    """For each qubit count, pick windows whose predicted P(top-1) is closest to evenly
    spaced targets between the pool's 1st percentile and its maximum. One window per
    subject within a stratum; no (subject, run, start) window used twice in the batch."""
    used_windows = set()
    chosen = {}
    for m in (8, 7, 6):
        F = F_by_m[m]
        D = 2 ** m
        cand = []
        for e in pool[m]:
            if m == 8 and not (F >= F_MIN_8Q and F * e["p1_full"] >= (1 - F) / D):
                continue
            cand.append(e)
        if not cand:
            chosen[m] = []
            continue
        P = np.array([e["P1_pred"] for e in cand])
        lo, hi = float(np.quantile(P, 0.01)), float(P.max())
        targets = np.linspace(lo, hi, N_PRIMARY[m])
        order = np.argsort(np.abs(targets - 0.5))[::-1]      # extremes first (scarcest)
        subj_used, picks = set(), {}
        for ti in order:
            best, bd = None, 1e9
            for j in np.argsort(np.abs(P - targets[ti])):
                s, run, ch, st = meta[cand[j]["idx"]]
                if s in subj_used or (s, run, st) in used_windows:
                    continue
                best, bd = cand[j], abs(P[j] - targets[ti])
                break
            if best is None:
                continue
            s, run, ch, st = meta[best["idx"]]
            subj_used.add(s)
            used_windows.add((s, run, st))
            picks[ti] = dict(best, target=float(targets[ti]))
        chosen[m] = [picks[k] for k in sorted(picks)]
        n_elig = len(cand)
        print(f"  {m} q: pool {len(pool[m])}, eligible {n_elig}, P(top-1) range "
              f"{P.min():.3f}..{P.max():.3f}, picked {len(chosen[m])}", flush=True)
    return chosen, used_windows


def synthetic_signals():
    t = np.arange(N)
    tone = np.cos(2 * np.pi * (64 * t + C * t ** 2) / N)
    noise = np.random.default_rng(7).standard_normal(N)
    snr1 = tone + noise * math.sqrt(np.sum(tone ** 2) / (1.0 * np.sum(noise ** 2)))
    white = np.random.default_rng(11).standard_normal(N)
    out = {}
    for name, x in (("chirp_pure", tone), ("chirp_snr1", snr1), ("white_noise", white)):
        x = x - x.mean()
        out[name] = x / np.abs(x).max()
    return out


def evaluate(design, device, label="device"):
    """The frozen analysis. `device` maps pub id -> {"counts": {...}, "shots": n}."""
    pubs = {p["id"]: p for p in design["pubs"]}

    def obs(pid):
        P = pubs[pid]
        m = P["qubits"]
        p = np.array(P["p_exact"])
        mask = np.array(P["mask"], bool)
        v = counts_vec(device[pid]["counts"], m)
        r = stable_rank(v, P["peak"], mask)
        q = v / v.sum()
        F = law.f_xeb(q, p)
        Fc = float(np.clip(F, 1e-6, 1))
        model = law.model_q(p, Fc)
        tvd = 0.5 * float(np.abs(q - model).sum())
        tvd_shot = law.sampling_tvd(model, int(v.sum()), reps=400)
        order = np.lexsort((np.arange(len(v)), -np.where(mask, v, -1)))
        ver = {f"k{k}": float(p[order[:k]].max() / p[P["peak"]]) for k in (1, 3, 5)}
        return dict(rank=r, top1=int(r == 1), top3=int(r <= 3), top5=int(r <= 5),
                    F_xeb=F, excess_ratio=tvd / tvd_shot, verified_energy=ver,
                    shots=int(v.sum()))

    res = {pid: obs(pid) for pid in device if pid in pubs}
    prim = [p for p in design["pubs"] if p["arm"] == "primary" and p["id"] in res]
    pr1 = np.array([p["pred"]["p_top1"] for p in prim])
    pr3 = np.array([p["pred"]["p_top3"] for p in prim])
    y1 = np.array([res[p["id"]]["top1"] for p in prim])
    y3 = np.array([res[p["id"]]["top3"] for p in prim])
    rr = np.array([1.0 / res[p["id"]]["rank"] for p in prim])
    rho, pperm = spearman_perm(pr1, rr)
    b1 = float(np.mean((pr1 - y1) ** 2))
    b1c = float(np.mean((CONST_TOP1 - y1) ** 2))
    b3 = float(np.mean((pr3 - y3) ** 2))
    b3c = float(np.mean((CONST_TOP3 - y3) ** 2))
    d3 = float(y3.mean() - pr3.mean())
    R1 = bool(rho >= RHO_MIN and pperm < PERM_ALPHA)
    R2 = bool(b1 < b1c)
    R3 = bool(abs(d3) <= TOP3_TOL)
    i1 = np.array([p["pred"]["p_top1_ideal"] for p in prim])
    i3 = np.array([p["pred"]["p_top3_ideal"] for p in prim])
    b_law = float(np.mean((pr1 - y1) ** 2) + np.mean((pr3 - y3) ** 2))
    b_ideal = float(np.mean((i1 - y1) ** 2) + np.mean((i3 - y3) ** 2))
    R4 = bool(b_law < b_ideal)
    verdict = "CONFIRMED" if (R1 and R2 and R3 and R4) else (
        "FALSIFIED" if (not R1 or not R4 or (not R2 and not R3)) else "PARTIAL")
    out = dict(label=label, n_primary=len(prim),
               R1_rank=dict(spearman_rho=rho, perm_p_one_sided=pperm, threshold=RHO_MIN, passed=R1),
               R2_brier_top1=dict(law=b1, constant=b1c, constant_value=CONST_TOP1, passed=R2),
               R3_top3_calibration=dict(observed=float(y3.mean()), predicted=float(pr3.mean()),
                                        diff=d3, tolerance=TOP3_TOL, passed=R3),
               R4_noise_term=dict(brier_sum_law=b_law, brier_sum_noiseless=b_ideal, passed=R4),
               verdict=verdict)
    # ---- secondary
    gap = np.array([p["signal"]["gap"] for p in prim])
    out["S_gap_only_spearman"] = spearman_perm(gap, rr)[0]
    out["S_brier_top3"] = dict(law=b3, constant=b3c, constant_value=CONST_TOP3)
    out["S_top1_count"] = dict(observed=int(y1.sum()), predicted=float(pr1.sum()))
    out["S_top3_count"] = dict(observed=int(y3.sum()), predicted=float(pr3.sum()))
    y5 = np.array([res[p["id"]]["top5"] for p in prim])
    out["S_top5_count"] = dict(observed=int(y5.sum()),
                               predicted=float(sum(p["pred"]["p_top5"] for p in prim)),
                               uniform=float(sum(min(5, p["signal"]["M_inband"]) / p["signal"]["M_inband"]
                                                 for p in prim)))
    Fp = [p["pred"]["F"] for p in design["pubs"] if p["id"] in res and p["variant"] == "uni"]
    Fo = [res[p["id"]]["F_xeb"] for p in design["pubs"] if p["id"] in res and p["variant"] == "uni"]
    out["S_F_calibration_uni"] = dict(spearman=float(spearmanr(Fp, Fo).statistic),
                                      median_obs_over_pred=float(np.median(np.array(Fo) / np.array(Fp))))
    for m in (6, 7, 8):
        sel = [p for p in prim if p["qubits"] == m]
        if sel:
            out[f"S_by_qubits_{m}"] = dict(
                n=len(sel), top1_obs=int(sum(res[p["id"]]["top1"] for p in sel)),
                top1_pred=float(sum(p["pred"]["p_top1"] for p in sel)),
                top3_obs=int(sum(res[p["id"]]["top3"] for p in sel)),
                top3_pred=float(sum(p["pred"]["p_top3"] for p in sel)),
                median_F_obs=float(np.median([res[p["id"]]["F_xeb"] for p in sel])),
                F_pred=float(np.median([p["pred"]["F"] for p in sel])))
    dyn = [p for p in design["pubs"] if p["arm"] == "dyn_twin" and p["id"] in res]
    if dyn:
        twins = [p["twin_of"] for p in dyn]
        out["S_dyn_twins"] = dict(
            n=len(dyn), top3_obs=int(sum(res[p["id"]]["top3"] for p in dyn)),
            top3_pred=float(sum(p["pred"]["p_top3"] for p in dyn)),
            median_excess_dyn=float(np.median([res[p["id"]]["excess_ratio"] for p in dyn])),
            median_excess_uni_twins=float(np.median([res[t]["excess_ratio"] for t in twins if t in res])),
            prediction_holds=bool(
                sum(res[p["id"]]["top3"] for p in dyn) <= sum(p["pred"]["p_top3"] for p in dyn)
                and np.median([res[p["id"]]["excess_ratio"] for p in dyn])
                > np.median([res[t]["excess_ratio"] for t in twins if t in res])))
    out["S_other_arms"] = {p["id"]: dict(arm=p["arm"], rank=res[p["id"]]["rank"],
                                         p_top1_pred=p["pred"]["p_top1"], F_pred=p["pred"]["F"],
                                         F_obs=res[p["id"]]["F_xeb"])
                           for p in design["pubs"] if p["arm"] not in ("primary",) and p["id"] in res}
    out["S_uni_primary_median_excess"] = float(np.median([res[p["id"]]["excess_ratio"] for p in prim]))
    out["per_pub"] = res
    return out


POWER_WORLDS = ("law", "F_x2", "F_x0.5", "ideal_F1", "null_uniform")


def power_sim(design, n_batches, F_logres, seed=SEED, world="law"):
    """Pass probability of R1-R4 when whole primary batches are drawn from a stated world:
    law       the law with F_true = F_pred * exp(r), r resampled from the chosen predictor's
              LOO log residuals on Sep 25;
    F_x2 / F_x0.5   the law, but the true F is twice / half the prediction (no scatter);
    ideal_F1  a noiseless sampler (shot noise only);
    null_uniform    a device that ignores the signal (q uniform).
    The rules are always scored against the preregistered plug-in predictions."""
    null = world == "null_uniform"
    rng = np.random.default_rng(seed + POWER_WORLDS.index(world))
    prim = [p for p in design["pubs"] if p["arm"] == "primary"]
    pr1 = np.array([p["pred"]["p_top1"] for p in prim])
    pr3 = np.array([p["pred"]["p_top3"] for p in prim])
    i1 = np.array([p["pred"]["p_top1_ideal"] for p in prim])
    i3 = np.array([p["pred"]["p_top3_ideal"] for p in prim])
    arrs = [(np.array(p["p_exact"]), np.array(p["mask"], bool), p["peak"], p["pred"]["F"]) for p in prim]
    passes = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "all": 0, "falsified": 0}
    for _ in range(n_batches):
        rr, y1, y3 = [], [], []
        for p, mask, peak, F in arrs:
            if null:
                q = np.full(len(p), 1.0 / len(p))
            else:
                if world == "law":
                    Ft = F * math.exp(rng.choice(F_logres))
                elif world == "F_x2":
                    Ft = 2 * F
                elif world == "F_x0.5":
                    Ft = 0.5 * F
                else:
                    Ft = 1.0
                q = law.model_q(p, float(np.clip(Ft, 1e-4, 1.0)))
            v = rng.multinomial(SHOTS, q / q.sum())
            r = stable_rank(v, peak, mask)
            rr.append(1.0 / r)
            y1.append(int(r == 1))
            y3.append(int(r <= 3))
        y1, y3 = np.array(y1), np.array(y3)
        rho, pp = spearman_perm(pr1, rr, n_perm=1000, seed=int(rng.integers(1 << 30)))
        R1 = rho >= RHO_MIN and pp < PERM_ALPHA
        R2 = np.mean((pr1 - y1) ** 2) < np.mean((CONST_TOP1 - y1) ** 2)
        R3 = abs(y3.mean() - pr3.mean()) <= TOP3_TOL
        passes["R1"] += R1
        passes["R2"] += R2
        R4 = (np.mean((pr1 - y1) ** 2) + np.mean((pr3 - y3) ** 2)
              < np.mean((i1 - y1) ** 2) + np.mean((i3 - y3) ** 2))
        passes["R3"] += R3
        passes["R4"] += R4
        passes["all"] += R1 and R2 and R3 and R4
        passes["falsified"] += (not R1) or (not R4) or ((not R2) and (not R3))
    return {k: v / n_batches for k, v in passes.items()}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--offline", action="store_true", help="FakeMarrakesh as the target (no IBM access)")
    ap.add_argument("--power-batches", type=int, default=400)
    ap.add_argument("--evaluate", default="", help="device results json (after the run)")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    t_start = time.time()

    if args.evaluate:
        design = json.load(open(OUT_JSON))
        dev = json.load(open(args.evaluate))
        device = {p["id"]: p for p in dev["pubs"]}
        res = evaluate(design, device, label=str(args.evaluate))
        out = RES / ("qpu_law_batch_evaluation_" + Path(args.evaluate).stem + ".json")
        json.dump(res, open(out, "w"), indent=1, default=float)
        print(json.dumps({k: v for k, v in res.items() if k != "per_pub"}, indent=1, default=float))
        print(f"wrote {out}")
        return

    # ------------------------------------------------------------------ stage 1
    print("stage 1: Sep 25 rows and F-predictor choice (leave-one-out)", flush=True)
    rows25, src25 = sep25_rows()
    best, loo = choose_predictor(rows25)
    for n in Predictor.NAMES:
        print(f"  {n:<15} LOO Brier top1 {loo[n]['loo_brier_top1']:.4f} top3 "
              f"{loo[n]['loo_brier_top3']:.4f}  MAE F {loo[n]['loo_mae_F']:.4f}  "
              f"sd log-res {loo[n]['loo_log_residual_sd']:.3f}", flush=True)
    print(f"  chosen: {best}", flush=True)
    PRED = Predictor(best, rows25)
    ALL_PRED = {n: Predictor(n, rows25) for n in Predictor.NAMES}
    logres = loo[best]["loo_log_residuals"]

    # ------------------------------------------------------------------ stage 2
    print("stage 2: candidate pool", flush=True)
    meta, X = eeg_pool()
    meta = [tuple(int(v) if i != 2 else str(v) for i, v in enumerate(mm)) for mm in meta]
    pool = pool_stats(meta, X)
    print(f"  {len(meta)} windows ({len(SUBJECTS)} subjects x runs {RUNS} x {len(CHANNELS)} channels)",
          flush=True)

    # ------------------------------------------------------------------ stage 3
    print("stage 3: target backend, representative CZ counts, selection", flush=True)
    backend, svc, binfo = open_backend(args.backend, args.offline)
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    import qpu_selection as qsel
    pm_dev = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    fake = FakeMarrakesh()
    pm_fake = generate_preset_pass_manager(optimization_level=3, backend=fake, seed_transpiler=7)
    rep_cz = {}
    for m in (6, 7, 8, 9):
        for var in ("uni", "dyn"):
            if m in (8, 9) and var == "dyn":
                continue
            x = X[pool[6][0]["idx"]]
            cs = qsel.make_case(dict(subject=0, channel="rep", x=x), LOGDT[m], C, var)
            assert cs["qubits"] == m
            rep_cz[(m, var)] = isa_stats(pm_dev.run(cs["qc"]), backend.target, var)["cz"]
            print(f"  representative {m} q {var}: CZ {rep_cz[(m, var)]}", flush=True)
    # a-priori F for the pool screen. Fake-based predictors need a per-circuit fake F, so
    # the screen uses the cz_fit_variant law; final predictions use the chosen predictor.
    screen = ALL_PRED["cz_fit_variant"]
    F_by_m = {m: screen(rep_cz[(m, "uni")], "uni") for m in (6, 7, 8)}
    print("  screening F (cz_fit_variant): " + ", ".join(f"{m} q {F:.3f}" for m, F in F_by_m.items()),
          flush=True)
    for m in (6, 7, 8):
        for e in pool[m]:
            cs_p = None
            x = X[e["idx"]]
            cs_p = qs.make_case(dict(x=x), LOGDT[m], C, "uni")
            e["P1_pred"] = p_top1_fast(law.model_q(cs_p["want"], F_by_m[m]), cs_p["mask"], cs_p["peak"], SHOTS)
    chosen, used = select_primary(pool, meta, F_by_m, SHOTS)

    # ------------------------------------------------------------------ pubs
    specs = []
    for m in (6, 7, 8):
        for e in chosen[m]:
            s, run, ch, st = meta[e["idx"]]
            specs.append(dict(arm="primary", variant="uni", qubits=m, logdt=LOGDT[m], shots=SHOTS,
                              source=dict(kind="eeg", subject=s, run=run, channel=ch, start=st),
                              x=X[e["idx"]]))
    wins25 = {(w["subject"], w["channel"]): w for w in qs.eeg_windows()}
    for key in ANCHORS:
        s, ch, ld, var = key.split("|")
        specs.append(dict(arm="anchor_sep25", variant=var, qubits=[k for k, v in LOGDT.items() if v == float(ld)][0],
                          logdt=float(ld), shots=SHOTS,
                          source=dict(kind="eeg", subject=int(s), run=1, channel=ch, start="center",
                                      sep25_key=key),
                          x=wins25[(int(s), ch)]["x"]))
    syn = synthetic_signals()
    for name, m in (("chirp_pure", 8), ("chirp_snr1", 8), ("white_noise", 7), ("chirp_pure", 9)):
        specs.append(dict(arm="synthetic", variant="uni", qubits=m, logdt=LOGDT[m], shots=SHOTS,
                          source=dict(kind="synthetic", name=name), x=syn[name]))
    for i, sp in enumerate(specs):
        src = sp["source"]
        if src["kind"] == "eeg":
            sp["id"] = (f"{sp['arm'][:4]}_{sp['qubits']}q_S{src['subject']:03d}R{src['run']:02d}_"
                        f"{src['channel']}_{src['start']}_{sp['variant']}")
        else:
            sp["id"] = f"syn_{sp['qubits']}q_{src['name']}_{sp['variant']}"

    # exact targets + predictions need F, F needs CZ (device) and fake F (fake sim):
    # build and transpile everything first
    CACHE.mkdir(parents=True, exist_ok=True)
    fake_cache_path = CACHE / "fake_counts.json"
    fake_cache = json.load(open(fake_cache_path)) if fake_cache_path.exists() else {}
    noisy = AerSimulator.from_backend(fake)
    noisy.set_options(max_parallel_threads=4)
    ideal = AerSimulator()

    def realise(sp):
        w = dict(subject=sp["source"].get("subject", 0), channel=sp["source"].get("channel", "syn"), x=sp["x"])
        cs = qsel.make_case(w, sp["logdt"], C, sp["variant"])
        assert cs["qubits"] == sp["qubits"], (sp["id"], cs["qubits"])
        sp["p"], sp["mask"], sp["peak"] = cs["want"], cs["mask"], cs["peak"]
        sp["qc"] = cs["qc"]
        t0 = time.time()
        sp["isa"] = pm_dev.run(cs["qc"])
        sp["isa_stats"] = isa_stats(sp["isa"], backend.target, sp["variant"])
        # fake noise prediction (qpu_selection.py protocol: fake-transpiled, seed 7 / 2)
        fk = sp["id"] + f"|{SHOTS}"
        if fk not in fake_cache:
            isa_f = pm_fake.run(cs["qc"])
            for attempt in range(4):
                try:
                    cnt = noisy.run(isa_f, shots=SHOTS, seed_simulator=2).result().get_counts()
                    break
                except MemoryError:
                    print("  MemoryError; retrying with fewer threads", flush=True)
                    noisy.set_options(max_parallel_threads=max(1, 2 >> attempt))
                    time.sleep(20)
            else:
                raise MemoryError("fake simulation failed")
            fake_cache[fk] = dict(counts={k: int(v) for k, v in cnt.items()},
                                  cz_fake=isa_stats(isa_f, fake.target, sp["variant"])["cz"])
            json.dump(fake_cache, open(fake_cache_path, "w"))
        sp["fake"] = fake_cache[fk]
        # noiseless check of the logical circuit against the exact target
        if sp["variant"] == "uni" and sp["qubits"] <= 8:
            tq = generate_preset_pass_manager(optimization_level=0, backend=ideal).run(cs["qc"])
            cnt = ideal.run(tq, shots=100_000, seed_simulator=1).result().get_counts()
            pv = counts_vec(cnt, sp["qubits"])
            sp["ideal_tvd"] = 0.5 * float(np.abs(pv / pv.sum() - sp["p"]).sum())
        print(f"  {sp['id']:<44} CZ dev {sp['isa_stats']['cz']:5d} fake {sp['fake']['cz_fake']:5d} "
              f"({time.time() - t0:.0f} s)", flush=True)

    for sp in specs:
        realise(sp)

    def predict(sp, shots=None):
        shots = shots or sp["shots"]
        m = sp["qubits"]
        p, mask, peak = sp["p"], sp["mask"], sp["peak"]
        fv = counts_vec(sp["fake"]["counts"], m)
        Ff = law.f_xeb(fv / fv.sum(), p)
        F = PRED(sp["isa_stats"]["cz"], sp["variant"], Ff)
        q = law.model_q(p, F)
        mc = mc_rank(q, mask, peak, shots, seed=SEED)
        mci = mc_rank(p.copy(), mask, peak, shots, seed=SEED)          # F = 1: shot noise only
        cf = law.closed_form(q, mask, peak, shots, ks=(1, 3, 5))
        # F-uncertainty mixture (secondary): average the closed form over F * exp(r)
        mix = np.mean([[law.closed_form(law.model_q(p, float(np.clip(F * math.exp(r), 1e-4, 1))),
                                        mask, peak, shots, ks=(1, 3))[k] for k in ("p_top1_cond", "p_top3_cond")]
                       for r in logres], axis=0)
        others = {}
        for n, Pn in ALL_PRED.items():
            Fn = Pn(sp["isa_stats"]["cz"], sp["variant"], Ff)
            cfn = law.closed_form(law.model_q(p, Fn), mask, peak, shots, ks=(1, 3))
            others[n] = dict(F=Fn, p_top1=cfn["p_top1_cond"], p_top3=cfn["p_top3_cond"])
        fr = stable_rank(fv, peak, mask)
        return dict(F=F, F_fake_xeb=Ff, p_top1=mc["p_top1"], p_top3=mc["p_top3"], p_top5=mc["p_top5"],
                    exp_rank=mc["exp_rank"], median_rank=mc["median_rank"],
                    p_top1_ideal=mci["p_top1"], p_top3_ideal=mci["p_top3"],
                    p_top1_closed=cf["p_top1_cond"], p_top3_closed=cf["p_top3_cond"],
                    p_top1_F_mixture=float(mix[0]), p_top3_F_mixture=float(mix[1]),
                    S_star_95=s_star(q, p, mask, peak, F),
                    regime="floor" if 2 * (1 - F) / 2 ** m > F * (p[peak] + np.sort(p[mask])[::-1][1]) else "signal",
                    uniform_p_top1=1 / int(mask.sum()), uniform_p_top3=min(3, int(mask.sum())) / int(mask.sum()),
                    fake_rank=fr, other_predictors=others)

    print("stage 4: predictions", flush=True)
    for sp in specs:
        sp["pred"] = predict(sp)
    # 8q/9q floor rule (applies to every circuit above 7 qubits, synthetic included)
    for sp in specs:
        if sp["qubits"] >= 8:
            F = sp["pred"]["F"]
            sp["floor_rule"] = dict(F=F, F_p1=F * float(sp["p"][sp["peak"]]),
                                    floor=(1 - F) / 2 ** sp["qubits"],
                                    passed=bool(F >= F_MIN_8Q and F * sp["p"][sp["peak"]] >= (1 - F) / 2 ** sp["qubits"])
                                    if sp["qubits"] == 8 else bool(F * sp["p"][sp["peak"]] >= (1 - F) / 2 ** sp["qubits"]))
    dropped = [sp["id"] for sp in specs if sp.get("floor_rule") and not sp["floor_rule"]["passed"]]
    specs = [sp for sp in specs if not (sp.get("floor_rule") and not sp["floor_rule"]["passed"])]
    if dropped:
        print("  dropped by the floor rule:", dropped, flush=True)

    # dyn twins: 3 per qubit count (6, 7) at the 2nd, 6th and 11th of 12 by predicted P(top-1)
    twins = []
    for m in (6, 7):
        prim = sorted([sp for sp in specs if sp["arm"] == "primary" and sp["qubits"] == m],
                      key=lambda s: s["pred"]["p_top1"])
        for pos in (1, 5, 10):
            if pos < len(prim):
                base = prim[pos]
                tw = dict(base, arm="dyn_twin", variant="dyn", twin_of=base["id"],
                          id=base["id"].replace("prim_", "dyn_").replace("_uni", "_dyn"))
                for k in ("isa", "isa_stats", "fake", "pred", "qc", "ideal_tvd"):
                    tw.pop(k, None)
                twins.append(tw)
    for tw in twins:
        realise(tw)
        tw["pred"] = predict(tw)
    specs += twins

    # shot scaling: 4 primary 6/7 q circuits re-run at 16000 shots (same ISA circuit)
    cands = [sp for sp in specs if sp["arm"] == "primary" and sp["qubits"] in (6, 7)]
    for sp in cands:
        sp["_p16"] = mc_rank(law.model_q(sp["p"], sp["pred"]["F"]), sp["mask"], sp["peak"],
                             SHOTS_SCALE, reps=5000, seed=SEED)["p_top1"]
    sig = sorted([sp for sp in cands if 0.3 <= sp["pred"]["p_top1"] <= 0.85],
                 key=lambda s: -(s["_p16"] - s["pred"]["p_top1"]))[:2]
    gapl = sorted([sp for sp in cands if sp["pred"]["p_top1"] < 0.5 and sp not in sig],
                  key=lambda s: (s["_p16"] - s["pred"]["p_top1"]))[:2]
    scale = []
    for kind, lst in (("signal_limited", sig), ("gap_limited", gapl)):
        for base in lst:
            sc = dict(base, arm="shot_scaling", shots=SHOTS_SCALE, scaling_kind=kind, twin_of=base["id"],
                      id=base["id"].replace("prim_", "shot16k_"))
            sc["pred"] = predict(sc, shots=SHOTS_SCALE)
            scale.append(sc)
    specs += scale
    for sp in specs:
        sp.pop("_p16", None)

    # ------------------------------------------------------------------ QPU estimate
    print("stage 5: QPU estimate, dry-run evaluation, power", flush=True)
    est25 = 0.0
    cz25 = {}
    for r in rows25:
        k = (r["m"], r["variant"])
        if k not in cz25:
            cs = qsel.make_case(dict(subject=r["subject"], channel=r["channel"], x=r["x"]), r["logdt"], C, r["variant"])
            cz25[k] = isa_stats(pm_dev.run(cs["qc"]), backend.target, r["variant"])["duration_s"]
        est25 += qpu_est(SHOTS, r["variant"], cz25[k])
    factor = SEP25_QPU_S / est25
    est_raw = sum(qpu_est(sp["shots"], sp["variant"], sp["isa_stats"]["duration_s"]) for sp in specs)
    est_model = factor * est_raw
    est_pershot = SEP25_QPU_S / SEP25_SHOTS * sum(sp["shots"] for sp in specs)
    est_final = max(est_model, est_pershot)
    print(f"  Sep 25 model estimate {est25:.1f} s vs 36 s used -> factor {factor:.3f}; "
          f"batch: model {est_model:.1f} s, per-shot {est_pershot:.1f} s", flush=True)
    if est_final > QPU_CAP_S:
        sys.exit(f"estimated {est_final:.0f} s exceeds the {QPU_CAP_S:.0f} s cap; redesign")

    # jobs and randomised order (seeded)
    rng = np.random.default_rng(SEED)
    jobs = dict(
        J1_uni_twirl_4k=[sp["id"] for sp in specs if sp["variant"] == "uni" and sp["shots"] == SHOTS],
        J2_dyn_default_4k=[sp["id"] for sp in specs if sp["variant"] == "dyn"],
        J3_uni_twirl_16k=[sp["id"] for sp in specs if sp["shots"] == SHOTS_SCALE])
    jobs = {k: [v[i] for i in rng.permutation(len(v))] for k, v in jobs.items()}
    job_options = dict(
        J1_uni_twirl_4k=dict(default_shots=SHOTS, dynamical_decoupling="XY4", twirling_gates=True,
                             twirling_measure=True, num_randomizations=16, shots_per_randomization=SHOTS // 16),
        J2_dyn_default_4k=dict(default_shots=SHOTS, note="default options: DD and twirling off, "
                               "as the Sep 25 dynamic job ran after DD was refused"),
        J3_uni_twirl_16k=dict(default_shots=SHOTS_SCALE, dynamical_decoupling="XY4", twirling_gates=True,
                              twirling_measure=True, num_randomizations=16,
                              shots_per_randomization=SHOTS_SCALE // 16))

    # qpy: one ISA circuit per unique circuit (the 16k pubs reuse their primary's circuit)
    from qiskit import qpy
    uniq = [sp for sp in specs if sp["arm"] != "shot_scaling"]
    for i, sp in enumerate(uniq):
        sp["isa"].name = sp["id"]
    buf = io.BytesIO()
    qpy.dump([sp["isa"] for sp in uniq], buf)
    OUT_QPY.write_bytes(buf.getvalue())
    qpy_sha = sha256_bytes(buf.getvalue())

    pubs = []
    for sp in specs:
        pubs.append(dict(
            id=sp["id"], arm=sp["arm"], variant=sp["variant"], qubits=sp["qubits"], logdt=sp["logdt"],
            shots=sp["shots"], source=sp["source"], twin_of=sp.get("twin_of"),
            scaling_kind=sp.get("scaling_kind"),
            circuit_in_qpy=sp.get("twin_of") if sp["arm"] == "shot_scaling" else sp["id"],
            x_sha256=sha256_bytes(np.asarray(sp["x"], np.float64).tobytes()),
            isa=sp["isa_stats"], cz_fake=sp["fake"]["cz_fake"], ideal_sim_tvd=sp.get("ideal_tvd"),
            peak=int(sp["peak"]), mask=[int(v) for v in sp["mask"]],
            p_exact=[float(v) for v in sp["p"]],
            signal=inband_stats(sp["p"], sp["mask"], sp["peak"]),
            floor_rule=sp.get("floor_rule"), pred=sp["pred"]))
    prim = sorted([p for p in pubs if p["arm"] == "primary"], key=lambda p: -p["pred"]["p_top1"])
    for i, p in enumerate(prim):
        p["pred"]["rank_in_batch_by_p_top1"] = i + 1

    design = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S %z"), backend=args.backend, backend_info=binfo,
        law="q = F p + (1-F)/D on the 2^m register outcomes; mirror and out-of-band bins masked "
            "after sampling; Multinomial(S, q)",
        fs=FS, N=N, rate_hz_s=8.0, tc=qs.TC, band=list(qs.BAND),
        f_predictor=dict(chosen=best, rule="lowest leave-one-out Brier (top-1) on the 26 Sep 25 rows; "
                                           "ties by LOO mean |F error|",
                         parameters=PRED.par, loo=loo, screening_predictor="cz_fit_variant",
                         representative_cz={f"{m}q_{v}": c for (m, v), c in rep_cz.items()},
                         screening_F=F_by_m),
        selection=dict(subjects=list(SUBJECTS), runs=list(RUNS), channels=list(CHANNELS),
                       window="non-overlapping 512-sample windows from 1 s into the run, zero-mean, "
                              "max-abs normalised",
                       n_windows=len(meta), n_primary_target=N_PRIMARY,
                       rule_8q=f"predicted F >= {F_MIN_8Q} and F*p1 >= (1-F)/D",
                       dropped_by_floor_rule=dropped),
        shots=SHOTS, shots_scaling=SHOTS_SCALE, jobs=jobs, job_options=job_options,
        qpu_estimate=dict(sep25_model_s=est25, sep25_used_s=SEP25_QPU_S, factor=factor,
                          batch_model_s=est_model, batch_per_shot_s=est_pershot, batch_estimate_s=est_final,
                          cap_s=QPU_CAP_S, total_shots=int(sum(sp["shots"] for sp in specs))),
        qpy_file=str(OUT_QPY.relative_to(ROOT)).replace("\\", "/"), qpy_sha256=qpy_sha,
        qpy_order=[sp["id"] for sp in uniq],
        rules=dict(primary_set="arm == primary (EEG, unitary variant, twirled)",
                   R1=f"Spearman(pred P(top-1), 1/observed rank) >= {RHO_MIN} and one-sided permutation "
                      f"p < {PERM_ALPHA} ({N_PERM} permutations, seed {SEED})",
                   R2=f"Brier(top-1) of the law < Brier of the constant {CONST_TOP1:.4f} (Sep 25 rate)",
                   R3=f"|observed top-3 fraction - mean predicted P(top-3)| <= {TOP3_TOL}",
                   R4="Brier(top-1) + Brier(top-3) of the law < the same score of the noiseless "
                      "limit of the law (F = 1, shot noise only), both from 20,000-draw Monte Carlo",
                   verdict="CONFIRMED if R1-R4 all pass; FALSIFIED if R1 fails, or R4 fails, or both "
                           "R2 and R3 fail; PARTIAL otherwise"),
        pubs=pubs)

    # ------------------------------------------------------------------ dry run
    mock = {p["id"]: dict(counts=next(sp for sp in specs if sp["id"] == p["id"])["fake"]["counts"],
                          shots=SHOTS) for p in pubs if p["shots"] == SHOTS}
    dry = evaluate(design, mock, label="FakeMarrakesh histograms as mock device")
    design["dry_run_fake_as_device"] = {k: v for k, v in dry.items() if k != "per_pub"}
    print("  dry-run verdict on FakeMarrakesh histograms:", dry["verdict"], flush=True)
    design["power"] = dict(batches=args.power_batches, worlds={})
    for wname in POWER_WORLDS:
        pw = {k: float(v) for k, v in power_sim(design, args.power_batches, logres, world=wname).items()}
        design["power"]["worlds"][wname] = pw
        print(f"  power [{wname}]: {pw}", flush=True)
    design["runtime_s"] = time.time() - t_start
    json.dump(design, open(OUT_JSON, "w"), indent=1, default=float)
    print(f"wrote {OUT_JSON} and {OUT_QPY} (sha256 {qpy_sha[:16]}...)")
    write_prereg(design)
    print(f"wrote {PREREG}")


# ============================================================================ prereg md
def f3(v):
    return "-" if v is None else (f"{v:.3f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}")


def write_prereg(d):
    P = d["pubs"]
    L = []
    a = L.append
    a("# Preregistration: out-of-sample hardware test of the fidelity-gap law\n")
    a(f"Generated by `paper/iclr2027/exp/qpu_law_design.py` on {d['created']}. Every number in this "
      "file comes from that script's output, `results/qpu_law_batch_design.json`. **The commit that "
      "adds this file is the preregistration. Nothing has been submitted to a device.** The law, its "
      "derivation and its in-sample check are in `notes/fidelity_gap_law.md`.\n")
    a("## 1. Hypothesis\n")
    a("A noisy device histogram of the windowed QACT selection circuit behaves like "
      "`Multinomial(S, F p + (1-F)/D)`: one scalar fidelity `F` per circuit, predicted before the "
      "run, plus the exact distribution `p` and the shot count `S`, predicts which circuits rank the "
      "exact best in-band atom first (top-1) or in the top 3. On Sep 25 this held in sample (26 "
      "circuits). Here it is tested on 28 new EEG circuits whose predictions are fixed below.\n")
    fp = d["f_predictor"]
    a("## 2. The a-priori F\n")
    a("F cannot come from the new histograms (that would be in-sample). Five predictors were scored "
      "leave-one-out on the 26 Sep 25 rows with a rule fixed before scoring: lowest LOO Brier score "
      "for top-1, ties broken by mean |F error|.\n")
    a("| predictor | LOO Brier top-1 | LOO Brier top-3 | LOO mean abs F error | sd of log(F_obs/F_pred) |")
    a("|---|---|---|---|---|")
    for n, v in fp["loo"].items():
        a(f"| {n}{' (chosen)' if n == fp['chosen'] else ''} | {v['loo_brier_top1']:.4f} | "
          f"{v['loo_brier_top3']:.4f} | {v['loo_mae_F']:.4f} | {v['loo_log_residual_sd']:.3f} |")
    a(f"\nChosen: **{fp['chosen']}**, parameters fitted on all 26 rows: `{json.dumps(fp['parameters'])}`. "
      "`cz_fit_*` is `F = A exp(-eps n_CZ)` with `n_CZ` the CZ count after transpiling for the "
      "device; `fake_*` start from the linear-XEB F of the circuit's FakeMarrakesh histogram "
      "(qpu_selection.py protocol: transpiler seed 7, simulator seed 2, 4000 shots). The "
      "other four predictors' numbers are stored per circuit as secondary predictions.\n")
    a(f"Pool screening used cz_fit_variant with the representative device CZ counts "
      f"{json.dumps(fp['representative_cz'])}, giving screening F "
      + ", ".join(f"{m} q {v:.3f}" for m, v in fp["screening_F"].items()) + ".\n")
    s = d["selection"]
    a("## 3. Circuits\n")
    a(f"Pool: {s['n_windows']} EEG windows from PhysioNet EEGMMIDB subjects {s['subjects'][0]}-"
      f"{s['subjects'][-1]} (Sep 25 used subjects 1-3), runs R01 (eyes open) and R02 (eyes closed), "
      f"channels {', '.join(s['channels'])}; {s['window']}. The pipeline is qpu_selection.py's: "
      "centre tc = 256, chirp rate 8 Hz/s, band 0.5-45 Hz, mirror bins masked. For each qubit count "
      "the script picked windows whose predicted P(top-1) is closest to evenly spaced targets from "
      "the pool's 1st percentile to its maximum, one window per subject within a qubit count and no "
      "window used twice. logDt 1.5 / 2.7 / 3.3 gives 6 / 7 / 8 qubits. "
      f"8-qubit rule: {s['rule_8q']}. Dropped by the floor rule: {s['dropped_by_floor_rule'] or 'none'}.\n")
    counts = {}
    for p in P:
        counts.setdefault(p["arm"], {}).setdefault(p["qubits"], 0)
        counts[p["arm"]][p["qubits"]] += 1
    a("| arm | purpose | circuits by qubit count |")
    a("|---|---|---|")
    purpose = dict(primary="the test (EEG, unitary variant, twirled)",
                   dyn_twin="same windows, dynamic variant, untwirled (prediction 1 of the note)",
                   anchor_sep25="two Sep 25 circuits re-run (day-to-day drift of F)",
                   synthetic="pure chirp / chirp at SNR 1 / white noise (prediction 3: sparsity)",
                   shot_scaling="primary circuits re-run at 16000 shots (prediction 2)")
    for arm, cc in counts.items():
        a(f"| {arm} | {purpose.get(arm, '')} | " + ", ".join(f"{m} q: {n}" for m, n in sorted(cc.items())) + " |")
    uq = [p for p in P if p["arm"] != "shot_scaling"]
    byq = {}
    for p in uq:
        byq[p["qubits"]] = byq.get(p["qubits"], 0) + 1
    a(f"\nUnique circuits: {len(uq)} (" + ", ".join(f"{m} q: {n}" for m, n in sorted(byq.items()))
      + f"); PUBs including the 16k-shot re-runs: {len(P)}.\n")
    a("### 3.1 Primary circuits (the test)\n")
    a("P(top-k) is the law's probability that the exact best in-band atom has rank <= k among in-band "
      "counts (ties: lower index first, the scoring rule), by multinomial Monte Carlo with 20,000 "
      "draws; `closed` is the note's closed form (3) as a check. E[rank] is the expected rank of the "
      "best atom. `batch rank` orders circuits by predicted P(top-1). The `F=1` columns are the "
      "law's noiseless limit (shot noise only), the comparator of R4.\n")
    a("| batch rank | id | q | CZ | gap p1-p2 | PR in-band | M' | F pred | F fake | P(top-1) | closed | P(top-3) | P(top-5) | E[rank] | P(top-1) F=1 | P(top-3) F=1 | S* 95% | regime |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted([p for p in P if p["arm"] == "primary"], key=lambda p: p["pred"]["rank_in_batch_by_p_top1"]):
        e = p["pred"]
        g = p["signal"]
        a(f"| {e['rank_in_batch_by_p_top1']} | `{p['id']}` | {p['qubits']} | {p['isa']['cz']} | {g['gap']:.4f} | "
          f"{g['pr_inband']:.2f} | {g['M_inband']} | {e['F']:.3f} | {e['F_fake_xeb']:.3f} | "
          f"{e['p_top1']:.3f} | {e['p_top1_closed']:.3f} | {e['p_top3']:.3f} | {e['p_top5']:.3f} | "
          f"{e['exp_rank']:.2f} | {e['p_top1_ideal']:.3f} | {e['p_top3_ideal']:.3f} | "
          f"{e['S_star_95']:.3g} | {e['regime']} |")
    prim = [p for p in P if p["arm"] == "primary"]
    a(f"\nPredicted totals over the {len(prim)} primary circuits: top-1 "
      f"{sum(p['pred']['p_top1'] for p in prim):.2f}, top-3 {sum(p['pred']['p_top3'] for p in prim):.2f}, "
      f"top-5 {sum(p['pred']['p_top5'] for p in prim):.2f}; uniform random proposal: top-1 "
      f"{sum(p['pred']['uniform_p_top1'] for p in prim):.2f}, top-3 "
      f"{sum(p['pred']['uniform_p_top3'] for p in prim):.2f}.\n")
    a("### 3.2 Secondary circuits\n")
    a("| id | arm | q | shots | CZ | gap | PR | F pred | P(top-1) | P(top-3) | E[rank] | note |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in P:
        if p["arm"] == "primary":
            continue
        e, g = p["pred"], p["signal"]
        note = p.get("twin_of") or p["source"].get("sep25_key") or p["source"].get("name", "")
        if p.get("scaling_kind"):
            note = f"{p['scaling_kind']}; 4k-shot twin {p['twin_of']}"
        a(f"| `{p['id']}` | {p['arm']} | {p['qubits']} | {p['shots']} | {p['isa']['cz']} | {g['gap']:.4f} | "
          f"{g['pr_inband']:.2f} | {e['F']:.3f} | {e['p_top1']:.3f} | {e['p_top3']:.3f} | {e['exp_rank']:.2f} | {note} |")
    a("\n## 4. Pass rules (primary set only)\n")
    r = d["rules"]
    a(f"- **R1 (ranking).** {r['R1']}. Observed success is scored as the reciprocal rank `1/r` of the "
      "exact best in-band atom in the device histogram (1 for a top-1 hit).")
    a(f"- **R2 (calibration, top-1).** {r['R2']}.")
    a(f"- **R3 (calibration, top-3).** {r['R3']}.")
    a(f"- **R4 (the noise term earns its place).** {r['R4']}. R1-R3 alone cannot tell the law from a "
      "noiseless sampler (section 6); R4 asks whether F improves the forecast over ignoring noise.")
    a(f"- **Verdict.** {r['verdict']}.\n")
    a("**What counts as failure.** R1 failing means the law does not rank circuits better than chance "
      "out of sample: falsified. R2 and R3 both failing means the law ranks but its probabilities are "
      "wrong (for example because the a-priori F is off): also reported as falsified for the law as a "
      "predictive tool. R4 failing means the fidelity term adds nothing over shot noise alone: the "
      "fidelity part of the law is falsified even if the signal statistics rank circuits well. A shortfall explained afterwards by drift (anchors), by a different F, or by "
      "dropping circuits does not change the verdict; such analyses are labelled exploratory.\n")
    a("**Void (not failure).** The batch is void, and nothing is claimed, if: a job fails or returns "
      "fewer than 90% of its shots; the submitted circuits are not the ones in the qpy file below "
      "(sha256 check); or the backend is not ibm_marrakesh. If the device's calibration at submission "
      "marks a used physical qubit as faulty, the circuits may be re-transpiled with the same seed; "
      "that deviation is reported and the predictions are recomputed only for F (the CZ count), with "
      "the old and new predictions both published.\n")
    a("**Secondary, reported without a verdict.** (a) Brier top-3 against the constant "
      f"{CONST_TOP3:.4f}; (b) the gap alone as a predictor (Spearman with 1/rank), since on Sep 25 it "
      "ranked success as well as the law; (c) predicted vs observed F_xeb (Spearman, median ratio); "
      "(d) note prediction 1: the dynamic twins' top-3 count at or below the law's and their median "
      "excess ratio above that of their unitary twins; (e) note prediction 2: the signal-limited "
      "16k-shot circuits reach top-1, the gap-limited ones stay below; (f) note prediction 3: the 8- "
      "and 9-qubit pure chirps succeed at the rates below while broadband windows at the same depth "
      "fail; (g) anchors: F_xeb today over Sep 25; (h) proposer-verifier top-5 recovery vs the law "
      "and vs the uniform proposal; (i) the F-uncertainty mixture and the four other F predictors, "
      "scored with the same rules.\n")
    a("## 5. Execution (to be done in a later session)\n")
    q = d["qpu_estimate"]
    a(f"Backend {d['backend']} via the open-plan instance (CRN ending {OPEN_CRN_SUFFIX}); calibration at "
      f"design time {d['backend_info'].get('calibration_last_update')}. Three jobs, PUB order "
      "randomised with seed " + str(SEED) + " (stored in the json):\n")
    for k, v in d["job_options"].items():
        a(f"- `{k}` ({len(d['jobs'][k])} PUBs): {json.dumps(v)}")
    a(f"\nISA circuits: `{d['qpy_file']}`, sha256 `{d['qpy_sha256']}`. Submit these exact circuits; "
      "the 16k-shot PUBs reuse their primary circuit.\n")
    a(f"**QPU time.** The Sep 25 run used {q['sep25_used_s']:.0f} s for 26 x 4000 shots. qpu_selection.py's "
      f"per-shot model gives {q['sep25_model_s']:.1f} s for that run (factor {q['factor']:.3f}). Scaled, "
      f"this batch ({q['total_shots']} shots) needs {q['batch_model_s']:.1f} s; the plain per-shot rate "
      f"gives {q['batch_per_shot_s']:.1f} s. Estimate: **{q['batch_estimate_s']:.0f} s** (cap "
      f"{q['cap_s']:.0f} s).\n")
    a("## 6. Dry run and power\n")
    dr = d["dry_run_fake_as_device"]
    a(f"The frozen evaluation was run with the FakeMarrakesh histograms standing in for the device: "
      f"R1 rho {dr['R1_rank']['spearman_rho']:.3f} (p {dr['R1_rank']['perm_p_one_sided']:.4f}), "
      f"R2 Brier {dr['R2_brier_top1']['law']:.3f} vs {dr['R2_brier_top1']['constant']:.3f}, "
      f"R3 top-3 {dr['R3_top3_calibration']['observed']:.3f} vs {dr['R3_top3_calibration']['predicted']:.3f}, "
      f"R4 Brier sum {dr['R4_noise_term']['brier_sum_law']:.3f} vs noiseless {dr['R4_noise_term']['brier_sum_noiseless']:.3f}; "
      f"verdict {dr['verdict']}. FakeMarrakesh is optimistic about F (by 0.30 on Sep 25), so this is a "
      "plumbing check, not a forecast.\n")
    pw = d["power"]
    a(f"Pass probabilities of the rules over {pw['batches']} simulated primary batches per world, "
      "always scored against the plug-in predictions above. `law`: the law holds and the true F "
      "scatters around the prediction as the Sep 25 LOO residuals did. `F_x2` / `F_x0.5`: the law "
      "holds but the true F is twice / half the prediction. `ideal_F1`: a noiseless sampler. "
      "`null_uniform`: a device that ignores the signal.\n")
    a("| world | R1 | R2 | R3 | R4 | all pass (CONFIRMED) | FALSIFIED |")
    a("|---|---|---|---|---|---|---|")
    for wn, v in pw["worlds"].items():
        a(f"| {wn} | {v['R1']:.3f} | {v['R2']:.3f} | {v['R3']:.3f} | {v['R4']:.3f} | {v['all']:.3f} | {v['falsified']:.3f} |")
    a("\nRead this table before the result. It shows what the four rules can and cannot tell apart. In particular, a true F at half "
      "the prediction is confirmed about as often as it is not, because R4 then still favours the "
      "law over the noiseless limit; "
      "an F error the rules cannot see is reported through the secondary F calibration (c) and the "
      "per-predictor scores (i), not through the verdict.\n")
    a("## 7. Known limits\n")
    ch = fp["loo"][fp["chosen"]]
    a("- One backend, one calibration day; F drifts between days (the anchors measure how much).")
    a(f"- The chosen predictor's LOO sd of log(F_obs/F_pred) is {ch['loo_log_residual_sd']:.3f} over all "
      f"Sep 25 rows with F > 0 and {ch['loo_log_residual_sd_6_7q']:.3f} over the 6- and 7-qubit rows "
      "only; the power simulation resamples the full set, so its F scatter is the wider one.")
    a("- 8 qubits is new: no Sep 25 circuit had 8 qubits, so its F comes from the CZ fit between the "
      "7- and 9-qubit rows. Only 4 primary circuits are at 8 qubits.")
    bi = d["backend_info"]
    a(f"- The open-plan usage period at design time ran to {(bi.get('usage_period') or {}).get('end_time')} "
      f"with {bi.get('usage_remaining_seconds_at_design')} s remaining; check the quota before submitting.")
    a("- The primary set tests the twirled unitary variant, where the law held on Sep 25. The "
      "untwirled dynamic variant is expected to deviate (secondary arm).")
    a("- F is predicted per (qubit count, variant) from CZ counts when a cz-fit predictor is chosen, "
      "so within a qubit count the ranking is driven by the signal statistics (gap, participation "
      "ratio). The Sep 25 F varied by about 0.1 within a qubit count; that scatter is in the power "
      "simulation.")
    a("- There is no quantum speedup here: the exact distribution is a periodogram the FFT computes "
      "classically. The test is of a noise law and of a certified proposer-verifier, not of advantage.")
    PREREG.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

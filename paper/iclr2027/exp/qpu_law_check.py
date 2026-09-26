#!/usr/bin/env python
"""Confront the fidelity-gap law (notes/fidelity_gap_law.md) with the Sep 25 device data.

The law models the device histogram as q = F p + (1 - F) u (global depolarizing,
u uniform over the D = 2^m register outcomes), optionally followed by a readout
channel R, and asks when the best in-band atom (exact probability p1) is still the
most frequent in-band outcome after S shots. This script:

  1. rebuilds, for every row of results/qpu_selection_marrakesh1.json, the exact
     distribution p and the in-band mask (mirror bins and out-of-band bins removed)
     with the same code that produced the device run (exp/qpu_selection.py);
  2. estimates F per row with the linear cross-entropy estimator that is unbiased
     under the model, F_xeb = (D sum_k q_k p_k - 1) / (D sum_k p_k^2 - 1), and with
     the one-point estimator F_1 = (q_peak - 1/D) / (p_peak - 1/D) (the only one
     available for the fake-noise rows, whose counts were not stored);
  3. evaluates the closed-form success probabilities (pairwise normal approximation,
     top-1 and top-k) and checks them against multinomial Monte Carlo from the model;
  4. tests whether the device residual is consistent with the model (TVD of the
     observed histogram to the model vs the TVD expected from shot noise alone), and
     refits with a per-qubit asymmetric readout channel (F, e01, e10) by maximum
     likelihood to see whether readout explains the excess;
  5. fits F ~ A exp(-eps n_CZ) across rows and reports the shot budget S* the law says
     each row needed.

CPU only, no device access. Every number in the note comes from this script's output
(results/qpu_law_check.json).

    python paper/iclr2027/exp/qpu_law_check.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))



class qs:  # noqa: N801
    """The exact-distribution half of exp/qpu_selection.py and qbe/qact_hw.py, copied
    verbatim in behaviour so this script needs neither Qiskit nor qbe.qact (which
    imports torch). make_case asserts the rebuilt peak equals the stored one."""
    FS, N, n = 160.0, 512, 9
    BAND = (0.5, 45.0)
    TC = 256.0

    @staticmethod
    def eeg_windows(subjects=(1, 2, 3), channels=("O1", "Cz")):
        import mne
        from mne.datasets import eegbci
        mne.set_log_level("ERROR")
        data_dir = str(Path.home() / "mne_data")
        out = []
        for s in subjects:
            f = eegbci.load_data(s, [1], path=data_dir, update_path=False, verbose=False)[0]
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            eegbci.standardize(raw)
            st = (raw.n_times - qs.N) // 2
            for ch in channels:
                x = raw.get_data(picks=[ch])[0, st:st + qs.N].astype(np.float64)
                x = x - x.mean()
                out.append(dict(subject=s, channel=ch, x=x / np.abs(x).max()))
        return out

    @staticmethod
    def envelope(tc, logdt):
        t = np.arange(qs.N)
        return np.exp(-((t - tc) ** 2) / (2 * np.exp(logdt) ** 2))

    @staticmethod
    def in_band(bins, c, tc):
        fc = (bins + 2 * c * tc) * qs.FS / qs.N
        return (bins < qs.N // 2) & (fc >= qs.BAND[0]) & (fc <= qs.BAND[1])

    @staticmethod
    def window_register(env, n, sigmas=4.0):
        N = 2 ** n
        keep = np.flatnonzero(env >= math.exp(-0.5 * sigmas ** 2) * env.max())
        lo, hi = int(keep[0]), int(keep[-1])
        m = min(n, max(1, math.ceil(math.log2(hi - lo + 1))))
        return m, max(0, min(lo, N - 2 ** m))

    @staticmethod
    def target_distribution(r, env, c):
        N = len(r)
        s = env * r
        s = s / np.linalg.norm(s)
        t = np.arange(N)
        z = s * np.exp(-1j * 2 * np.pi * c * t ** 2 / N)
        return np.abs(np.fft.fft(z)) ** 2 / N

    @staticmethod
    def make_case(w, logdt, c, variant):
        env = qs.envelope(qs.TC, logdt)
        m, _ = qs.window_register(env, qs.n)
        full = qs.target_distribution(w["x"], env, c)
        bins = np.arange(2 ** m) * 2 ** (qs.n - m)
        want = full[bins] / full[bins].sum()
        mask = qs.in_band(bins, c, qs.TC)
        peak = int(np.argmax(np.where(mask, want, -1)))
        return dict(want=want, bins=bins, mask=mask, peak=peak, qubits=m)

RES = ROOT / "paper" / "iclr2027" / "results"
SRC = RES / "qpu_selection_marrakesh1.json"
OUT = RES / "qpu_law_check.json"
MC_REPS = 20000
Z95 = norm.ppf(0.95)
rng = np.random.default_rng(20260926)


# ----------------------------------------------------------------------------- model
def f_xeb(q, p):
    D = len(p)
    den = D * np.sum(p * p) - 1.0
    return float((D * np.sum(q * p) - 1.0) / den)


def f_onepoint(qk, pk, D):
    return float((qk - 1.0 / D) / (pk - 1.0 / D))


def readout(p, m, e01, e10):
    """Apply an independent per-qubit readout channel: 0->1 w.p. e01, 1->0 w.p. e10."""
    R = np.array([[1 - e01, e10], [e01, 1 - e10]])  # column = true bit, row = read bit
    t = p.reshape([2] * m)
    for ax in range(m):
        t = np.moveaxis(np.tensordot(R, np.moveaxis(t, ax, 0), axes=(1, 0)), 0, ax)
    return t.reshape(-1)


def model_q(p, F, m=None, e01=0.0, e10=0.0):
    D = len(p)
    q = F * p + (1 - F) / D
    if m is not None and (e01 > 0 or e10 > 0):
        q = readout(q, m, e01, e10)
    return q


def pairwise_z(q, idx1, others, S):
    """z_j = (q1 - qj) sqrt(S) / sqrt(q1 + qj - (q1 - qj)^2): the standardized mean of
    N1 - Nj under the multinomial (exact variance, normal approximation)."""
    q1 = q[idx1]
    qj = q[others]
    var = q1 + qj - (q1 - qj) ** 2
    return (q1 - qj) * math.sqrt(S) / np.sqrt(np.maximum(var, 1e-300))


def poisson_binomial_cdf(pis, kmax):
    """P(#successes <= kmax) for independent Bernoulli(pi)."""
    dist = np.zeros(len(pis) + 1)
    dist[0] = 1.0
    for pi in pis:
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= (1 - pi)
    return float(dist[:kmax + 1].sum())


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)   # probabilists' Gauss-Hermite
GH_W = GH_W / GH_W.sum()


def closed_form(q, mask, peak, S, ks=(1, 3, 5)):
    """Two closed forms for P(best atom has rank <= k among in-band counts).

    pairwise ('pb'): events {N_j > N_1} treated as independent with probabilities
        Phi(-z_j); rank-1 = product, rank-k = Poisson-binomial CDF. Simple, but the events
        share N_1 and are positively correlated, so it is biased low when many z_j ~ 0.
    conditional ('cond'): condition on N_1 = S q1 + sqrt(S q1 (1-q1)) x, x ~ N(0,1); given
        N_1 the other counts are Binomial(S - N_1, q_j / (1 - q1)), approximately
        independent and normal, so P(rank <= k) = E_x[ PB_{k-1}( Phi((mu_j(x) - N_1(x)) /
        sigma_j(x)) ) ], a one-dimensional Gauss-Hermite integral.
    """
    others = np.flatnonzero(mask & (np.arange(len(q)) != peak))
    z = pairwise_z(q, peak, others, S)
    beat = norm.cdf(-z)                      # P(N_j > N_1), pairwise
    out = dict(z_runnerup=float(np.min(z)) if len(z) else float("inf"),
               p_top1_union_lb=float(max(0.0, 1 - beat.sum())),
               p_top1_indep=float(np.prod(1 - beat)))
    for k in ks:
        out[f"p_top{k}_pb"] = poisson_binomial_cdf(beat, k - 1)
    q1 = q[peak]
    qj = q[others] / (1 - q1)
    acc = {k: 0.0 for k in ks}
    for x, wgt in zip(GH_X, GH_W):
        n1 = S * q1 + math.sqrt(S * q1 * (1 - q1)) * x
        rest = max(S - n1, 0.0)
        mu, sd = rest * qj, np.sqrt(np.maximum(rest * qj * (1 - qj), 1e-12))
        pj = norm.cdf((mu - n1) / sd)       # P(N_j > N_1 | N_1)
        for k in ks:
            acc[k] += wgt * poisson_binomial_cdf(pj, k - 1)
    for k in ks:
        out[f"p_top{k}_cond"] = float(acc[k])
    return out


def monte_carlo(q, mask, peak, S, ks=(1, 3, 5), reps=MC_REPS, chunk=500, p=None, eta=0.9):
    qn = q / q.sum()
    tally = {k: 0 for k in ks}
    tie = am = near = 0
    for start in range(0, reps, chunk):
        nrep = min(chunk, reps - start)
        N = rng.multinomial(S, qn, size=nrep).astype(np.int32)
        N[:, ~mask] = -1
        npk = N[:, peak][:, None]
        greater = (N > npk).sum(1)
        ties = (N == npk).sum(1) - 1
        for k in ks:
            tally[k] += int(np.sum(greater <= k - 1))
        tie += int(np.sum((greater == 0) & (ties > 0)))
        amx = np.argmax(N, axis=1)
        am += int(np.sum(amx == peak))   # score(): first index on ties
        if p is not None:
            near += int(np.sum(p[amx] >= eta * p[peak]))
        del N
    out = {f"p_top{k}_mc": tally[k] / reps for k in ks}
    out["p_tie_mc"] = tie / reps
    out["p_argmax_mc"] = am / reps
    if p is not None:
        out[f"p_energy_ge_{eta}_mc"] = near / reps
    return out


def sampling_tvd(q, S, reps=1000, chunk=250):
    tv = []
    for start in range(0, reps, chunk):
        N = rng.multinomial(S, q / q.sum(), size=min(chunk, reps - start)).astype(np.float32) / S
        tv.extend(0.5 * np.abs(N - q[None, :]).sum(1))
    return float(np.mean(tv))


def loglik(counts_vec, q):
    return float(np.sum(counts_vec * np.log(np.maximum(q, 1e-12))))


def fit_readout(counts_vec, p, m):
    """Grid-then-refine ML fit of (F, e01, e10)."""
    best = (-np.inf, None)
    Fs = np.linspace(0.0, 1.0, 41)
    es = np.linspace(0.0, 0.2, 21)
    u = np.full(len(p), 1.0 / len(p))
    for e01 in es:
        for e10 in es:
            rp, ru = readout(p, m, e01, e10), readout(u, m, e01, e10)  # R is linear
            for F in Fs:
                ll = loglik(counts_vec, F * rp + (1 - F) * ru)
                if ll > best[0]:
                    best = (ll, (F, e01, e10))
    F0, a0, b0 = best[1]
    for e01 in np.clip(np.linspace(a0 - 0.01, a0 + 0.01, 11), 0, 0.5):
        for e10 in np.clip(np.linspace(b0 - 0.01, b0 + 0.01, 11), 0, 0.5):
            rp, ru = readout(p, m, e01, e10), readout(u, m, e01, e10)
            for F in np.clip(np.linspace(F0 - 0.025, F0 + 0.025, 11), 0, 1):
                ll = loglik(counts_vec, F * rp + (1 - F) * ru)
                if ll > best[0]:
                    best = (ll, (float(F), float(e01), float(e10)))
    return best


def fit_depol_only(counts_vec, p):
    best = (-np.inf, None)
    for F in np.linspace(0.0, 1.0, 1001):
        ll = loglik(counts_vec, F * p + (1 - F) / len(p))
        if ll > best[0]:
            best = (ll, float(F))
    return best


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    tot = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return float(tot / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------- main
def main():
    logging.disable(logging.WARNING)
    src = json.load(open(SRC))
    S_default = int(src["shots"])
    c = src["rate_hz_s"] * qs.N / (2 * qs.FS ** 2)
    wins = {(w["subject"], w["channel"]): w for w in qs.eeg_windows()}

    rows_out = []
    for r in src["rows"]:
        w = wins[(r["subject"], r["channel"])]
        cs = qs.make_case(w, r["logdt"], c, r["variant"])
        assert cs["peak"] == r["want_peak_index"], "exact peak mismatch: rebuild differs"
        p, mask, peak, m = cs["want"], cs["mask"], cs["peak"], cs["qubits"]
        D = 2 ** m
        dev = r["device"]
        S = int(dev["shots"])
        cnt = np.zeros(D)
        for key, v in dev["counts"].items():
            cnt[int(key.split()[-1], 2)] += v
        q_obs = cnt / cnt.sum()

        pin = np.sort(p[mask])[::-1]
        p1, p2 = float(pin[0]), float(pin[1])
        pr_full = float(1.0 / np.sum(p ** 2))
        pm = p[mask] / p[mask].sum()
        pr_band = float(1.0 / np.sum(pm ** 2))
        Mp = int(mask.sum())

        Fx = f_xeb(q_obs, p)
        F1_dev = f_onepoint(dev["p_peak_device"], dev["p_peak_exact"], D)
        F1_fake = f_onepoint(r["fake_noisy"]["p_peak_device"], r["fake_noisy"]["p_peak_exact"], D)
        Fuse = float(np.clip(Fx, 1e-6, 1.0))

        qd = model_q(p, Fuse)
        cf = closed_form(qd, mask, peak, S)
        mc = monte_carlo(qd, mask, peak, S, p=p)
        # same law under the fake-noise one-point F (no counts stored for fake rows)
        qf = model_q(p, float(np.clip(F1_fake, 1e-6, 1.0)))
        cf_fake = closed_form(qf, mask, peak, int(src["shots"]))

        # observed rank of the exact peak among in-band device counts
        cm = np.where(mask, cnt, -1)
        obs_rank = int((cm > cnt[peak]).sum()) + 1
        sel = int(np.argmax(cm))
        # proposer-verifier: device proposes its k most frequent in-band outcomes, an
        # exact classical overlap (here: the exact p, i.e. |<atom, x>|^2) picks the best
        order = np.argsort(-cm, kind="stable")
        verified = {f"k{k}": float(p[order[:k]].max() / p1) for k in (1, 2, 3, 5, 8)}

        # model adequacy
        tvd_obs_model = float(0.5 * np.abs(q_obs - qd).sum())
        tvd_shot = sampling_tvd(qd, S)
        ll_dep, F_ml = fit_depol_only(cnt, p)
        ll_ro, (F_ro, e01, e10) = fit_readout(cnt, p, m)
        q_ro = readout(F_ro * p + (1 - F_ro) / D, m, e01, e10)
        tvd_obs_ro = float(0.5 * np.abs(q_obs - q_ro).sum())
        tvd_shot_ro = sampling_tvd(q_ro, S)
        cf_ro = closed_form(q_ro, mask, peak, S)
        # where did the device's pick come from?
        z_sel_excess = float((cnt[sel] - S * qd[sel]) / math.sqrt(max(S * qd[sel], 1e-12)))
        ham = bin(sel ^ peak).count("1")

        # law's shot budget for the runner-up pairwise test at 95% one-sided
        i2 = int(np.flatnonzero(mask)[np.argsort(p[mask])[::-1][1]])
        q1, q2 = qd[peak], qd[i2]
        gap = Fuse * (p1 - p2)
        S_star = float(Z95 ** 2 * (q1 + q2 - (q1 - q2) ** 2) / gap ** 2) if gap > 0 else float("inf")
        floor = (1 - Fuse) / D
        regime = "floor" if 2 * floor > Fuse * (p1 + p2) else "signal"

        rows_out.append(dict(
            subject=r["subject"], channel=r["channel"], logdt=r["logdt"], variant=r["variant"],
            qubits=m, D=D, M_inband=Mp, cz_device=r.get("cz_device", r["cz"]), shots=S,
            p1=p1, p2=p2, gap_rel=(p1 - p2) / p1, pr_full=pr_full, pr_inband=pr_band,
            F_xeb=Fx, F_onepoint_device=F1_dev, F_onepoint_fake=F1_fake,
            F_ml_depol=F_ml, hellinger_fidelity_device=dev["fidelity"],
            floor_per_outcome=floor, variance_regime=regime,
            law=cf, law_mc=mc, law_fake_F=cf_fake,
            S_star_95=S_star, shots_over_S_star=S / S_star if S_star > 0 else None,
            observed=dict(peak_ok=int(dev["peak_ok"]), rank=obs_rank,
                          selected=sel, selected_p_exact_over_p1=float(p[sel] / p1),
                          hamming_sel_to_peak=ham, z_selected_excess_over_model=z_sel_excess,
                          fake_peak_ok=int(r["fake_noisy"]["peak_ok"]),
                          verified_energy_ratio=verified),
            adequacy=dict(tvd_obs_to_model=tvd_obs_model, tvd_shot_noise_expected=tvd_shot,
                          excess_ratio=tvd_obs_model / tvd_shot,
                          loglik_depol=ll_dep, readout_fit=dict(F=F_ro, e01=e01, e10=e10,
                                                                loglik=ll_ro,
                                                                delta_loglik=ll_ro - ll_dep),
                          tvd_obs_to_readout_model=tvd_obs_ro,
                          tvd_shot_noise_readout_model=tvd_shot_ro,
                          excess_ratio_readout=tvd_obs_ro / tvd_shot_ro,
                          law_under_readout_model=cf_ro),
        ))
        R = rows_out[-1]
        print(f"S{R['subject']:03d} {R['channel']:>3} ld{R['logdt']:<4}{R['variant']:<4} m={m} "
              f"CZ={R['cz_device']:5d} F_xeb={Fx:6.3f} gap={p1 - p2:.4f} PR={pr_band:5.1f} "
              f"z12={cf['z_runnerup']:6.2f} P1 pb/cond/mc={cf['p_top1_indep']:.3f}/{cf['p_top1_cond']:.3f}/"
              f"{mc['p_top1_mc']:.3f} P3 cond/mc={cf['p_top3_cond']:.3f}/{mc['p_top3_mc']:.3f} "
              f"obs rank={obs_rank:3d} "
              f"excess={R['adequacy']['excess_ratio']:.2f} ro_excess="
              f"{R['adequacy']['excess_ratio_readout']:.2f} S*={S_star:9.0f}")

    # ------------------------------------------------------------------ aggregates
    y = [R["observed"]["peak_ok"] for R in rows_out]
    P = [R["law"]["p_top1_cond"] for R in rows_out]
    Pmc = [R["law_mc"]["p_top1_mc"] for R in rows_out]
    Pro = [R["adequacy"]["law_under_readout_model"]["p_top1_cond"] for R in rows_out]
    Z = [R["law"]["z_runnerup"] for R in rows_out]
    top3_obs = [int(R["observed"]["rank"] <= 3) for R in rows_out]
    top5_obs = [int(R["observed"]["rank"] <= 5) for R in rows_out]
    P3 = [R["law"]["p_top3_cond"] for R in rows_out]
    P5 = [R["law"]["p_top5_cond"] for R in rows_out]
    yf = [R["observed"]["fake_peak_ok"] for R in rows_out]
    Pf = [R["law_fake_F"]["p_top1_cond"] for R in rows_out]
    mc_err = [abs(R["law"]["p_top1_cond"] - R["law_mc"]["p_top1_mc"]) for R in rows_out]
    mc_err3 = [abs(R["law"]["p_top3_cond"] - R["law_mc"]["p_top3_mc"]) for R in rows_out]
    mc_err_pb = [abs(R["law"]["p_top1_indep"] - R["law_mc"]["p_top1_mc"]) for R in rows_out]
    mc_err3_pb = [abs(R["law"]["p_top3_pb"] - R["law_mc"]["p_top3_mc"]) for R in rows_out]

    cz = np.array([R["cz_device"] for R in rows_out], float)
    Fx = np.array([R["F_xeb"] for R in rows_out])
    fits = {}
    for name, sel in (("all", np.ones(len(cz), bool)),
                      ("dyn", np.array([R["variant"] == "dyn" for R in rows_out])),
                      ("uni", np.array([R["variant"] == "uni" for R in rows_out]))):
        ok = sel & (Fx > 0)
        b, a = np.polyfit(cz[ok], np.log(Fx[ok]), 1)
        pred = a + b * cz[ok]
        ss = np.sum((np.log(Fx[ok]) - pred) ** 2)
        st = np.sum((np.log(Fx[ok]) - np.log(Fx[ok]).mean()) ** 2)
        fits[name] = dict(eps_per_cz=float(-b), A=float(math.exp(a)), n=int(ok.sum()),
                          r2_log=float(1 - ss / st))

    # out-of-sample: fit eps on the 6- and 7-qubit rows only, predict the 9-qubit rows
    small = np.array([R["qubits"] <= 7 for R in rows_out]) & (Fx > 0)
    b7, a7 = np.polyfit(cz[small], np.log(Fx[small]), 1)
    oos = []
    for R in rows_out:
        if R["qubits"] > 7:
            Fp = float(math.exp(a7 + b7 * R["cz_device"]))
            w = wins[(R["subject"], R["channel"])]
            cs = qs.make_case(w, R["logdt"], c, R["variant"])
            cfp = closed_form(model_q(cs["want"], Fp), cs["mask"], cs["peak"], R["shots"])
            oos.append(dict(case=f"S{R['subject']:03d} {R['channel']} ld{R['logdt']} {R['variant']}",
                            cz=R["cz_device"], F_pred=Fp, F_xeb_obs=R["F_xeb"],
                            p_top1_pred=cfp["p_top1_cond"], p_top3_pred=cfp["p_top3_cond"],
                            obs_rank=R["observed"]["rank"]))
    by_var = {v: dict(
        median_excess_depol=float(np.median([R["adequacy"]["excess_ratio"] for R in rows_out
                                             if R["variant"] == v])),
        median_excess_readout=float(np.median([R["adequacy"]["excess_ratio_readout"]
                                               for R in rows_out if R["variant"] == v])),
        peak_ok=int(sum(R["observed"]["peak_ok"] for R in rows_out if R["variant"] == v)),
        predicted=float(sum(R["law"]["p_top1_cond"] for R in rows_out if R["variant"] == v)),
        top3_obs=int(sum(R["observed"]["rank"] <= 3 for R in rows_out if R["variant"] == v)),
        top3_pred=float(sum(R["law"]["p_top3_cond"] for R in rows_out if R["variant"] == v)))
        for v in ("dyn", "uni")}
    gap_abs = [R["p1"] - R["p2"] for R in rows_out]
    prs = [R["pr_inband"] for R in rows_out]
    # ---- preregistered predictions (evaluated now, before any new device run)
    prereg_shots = {}
    for R in rows_out:
        if R["qubits"] > 7:
            continue
        w = wins[(R["subject"], R["channel"])]
        cs = qs.make_case(w, R["logdt"], c, R["variant"])
        qd = model_q(cs["want"], float(np.clip(R["F_xeb"], 1e-6, 1)))
        key = f"S{R['subject']:03d} {R['channel']} ld{R['logdt']} {R['variant']}"
        prereg_shots[key] = {str(S2): round(closed_form(qd, cs["mask"], cs["peak"], S2)["p_top1_cond"], 4)
                             for S2 in (4000, 16000, 64000)}
    # sparse synthetic chirplet at the 9-qubit geometry: does sparsity rescue low F?
    env9 = qs.envelope(qs.TC, 3.9)
    m9, _ = qs.window_register(env9, qs.n)
    t = np.arange(qs.N)
    k0 = 64
    tone = np.cos(2 * np.pi * (k0 * t + c * t ** 2) / qs.N)
    F9_fit = float(fits["all"]["A"] * math.exp(-fits["all"]["eps_per_cz"] * 1136))
    synth = []
    nrng = np.random.default_rng(7)
    noise = nrng.standard_normal(qs.N)
    for snr in (None, 1.0, 0.1):
        x = tone.copy()
        if snr is not None:
            x = x + noise * math.sqrt(np.sum(tone ** 2) / (snr * np.sum(noise ** 2)))
        x = x - x.mean()
        x = x / np.abs(x).max()
        cs = qs.make_case(dict(x=x), 3.9, c, "uni")
        pw, mk, pk = cs["want"], cs["mask"], cs["peak"]
        pin = np.sort(pw[mk])[::-1]
        entry = dict(snr_energy=snr if snr is not None else "inf", qubits=cs["qubits"],
                     peak_bin=pk, p1=float(pin[0]), p2=float(pin[1]),
                     pr_inband=float(1 / np.sum((pw[mk] / pw[mk].sum()) ** 2)))
        for lab, Fv in (("F_fit_1136cz", F9_fit), ("F_obs_9q", 0.018)):
            qd = model_q(pw, Fv)
            i2 = int(np.flatnonzero(mk)[np.argsort(pw[mk])[::-1][1]])
            q1, q2 = qd[pk], qd[i2]
            Sst = Z95 ** 2 * (q1 + q2 - (q1 - q2) ** 2) / (Fv * (pin[0] - pin[1])) ** 2
            entry[lab] = dict(F=Fv, S_star_95=float(Sst),
                              p_top1_at_4000=closed_form(qd, mk, pk, 4000)["p_top1_cond"])
        synth.append(entry)
    # less circular variant: F from the CZ-count fit (one law for all rows), not per-row F_xeb
    P_fit, P3_fit = [], []
    for R in rows_out:
        w = wins[(R["subject"], R["channel"])] if R["qubits"] else None
        cs = qs.make_case(w, R["logdt"], c, R["variant"])
        Fv = float(fits[R["variant"]]["A"] * math.exp(-fits[R["variant"]]["eps_per_cz"] * R["cz_device"]))
        cfv = closed_form(model_q(cs["want"], Fv), cs["mask"], cs["peak"], R["shots"])
        R["law_F_from_cz_fit"] = dict(F=Fv, **cfv)
        P_fit.append(cfv["p_top1_cond"])
        P3_fit.append(cfv["p_top3_cond"])
    # intrinsic neighbour ratio: a pure complex chirplet atom measured on its own grid
    intrinsic = {}
    for ld in (1.5, 2.7, 3.9):
        env = qs.envelope(qs.TC, ld)
        mm, _ = qs.window_register(env, qs.n)
        step = 2 ** (qs.n - mm)
        kk = 8 * step
        x = np.exp(1j * 2 * np.pi * (kk * t + c * t ** 2) / qs.N)
        full = qs.target_distribution(x, env, c)
        bins = np.arange(2 ** mm) * step
        pw = full[bins] / full[bins].sum()
        srt = np.sort(pw)[::-1]
        intrinsic[str(ld)] = dict(qubits=mm, bin_step=step, p1=float(srt[0]), p2=float(srt[1]),
                                  p2_over_p1=float(srt[1] / srt[0]),
                                  pr=float(1 / np.sum(pw ** 2)),
                                  gaussian_prediction_p2_over_p1=float(math.exp(
                                      -(2 * np.pi * step * math.exp(ld) / qs.N) ** 2)))
    agg = dict(
        n_rows=len(rows_out),
        device_peak_ok=int(sum(y)), predicted_expected_successes=float(sum(P)),
        predicted_expected_successes_mc=float(sum(Pmc)),
        predicted_expected_successes_readout_model=float(sum(Pro)),
        brier_law=float(np.mean((np.array(P) - np.array(y)) ** 2)),
        brier_constant=float(np.mean((np.mean(y) - np.array(y)) ** 2)),
        auc_law_top1=auc(P, y), auc_z_top1=auc(Z, y),
        auc_Fxeb_top1=auc(list(Fx), y),
        device_top3=int(sum(top3_obs)), predicted_top3=float(sum(P3)), auc_law_top3=auc(P3, top3_obs),
        device_top5=int(sum(top5_obs)), predicted_top5=float(sum(P5)),
        fake_peak_ok=int(sum(yf)), predicted_fake_successes=float(sum(Pf)), auc_law_fake=auc(Pf, yf),
        closed_form_vs_mc_max_abs_err_top1=float(max(mc_err)),
        closed_form_vs_mc_mean_abs_err_top1=float(np.mean(mc_err)),
        closed_form_vs_mc_max_abs_err_top3=float(max(mc_err3)),
        pairwise_indep_vs_mc_max_abs_err_top1=float(max(mc_err_pb)),
        pairwise_indep_vs_mc_mean_abs_err_top1=float(np.mean(mc_err_pb)),
        pairwise_indep_vs_mc_max_abs_err_top3=float(max(mc_err3_pb)),
        median_excess_ratio_depol=float(np.median([R["adequacy"]["excess_ratio"] for R in rows_out])),
        median_excess_ratio_readout=float(np.median([R["adequacy"]["excess_ratio_readout"]
                                                     for R in rows_out])),
        median_readout_e01=float(np.median([R["adequacy"]["readout_fit"]["e01"] for R in rows_out])),
        median_readout_e10=float(np.median([R["adequacy"]["readout_fit"]["e10"] for R in rows_out])),
        rows_with_shots_ge_Sstar=int(sum(1 for R in rows_out if R["S_star_95"] <= R["shots"])),
        peak_ok_when_shots_ge_Sstar=int(sum(R["observed"]["peak_ok"] for R in rows_out
                                            if R["S_star_95"] <= R["shots"])),
        peak_ok_when_shots_lt_Sstar=int(sum(R["observed"]["peak_ok"] for R in rows_out
                                            if R["S_star_95"] > R["shots"])),
        failures_predicted_safe=[f"S{R['subject']:03d} {R['channel']} ld{R['logdt']} {R['variant']}"
                                 for R in rows_out
                                 if R["observed"]["peak_ok"] == 0 and R["law"]["p_top1_cond"] > 0.9],
        fit_logF_vs_cz=fits,
        fit_small_only=dict(eps_per_cz=float(-b7), A=float(math.exp(a7))),
        out_of_sample_9q=oos,
        by_variant=by_var,
        F_from_cz_fit=dict(predicted_top1=float(sum(P_fit)), auc_top1=auc(P_fit, y),
                           brier=float(np.mean((np.array(P_fit) - np.array(y)) ** 2)),
                           predicted_top3=float(sum(P3_fit)), auc_top3=auc(P3_fit, top3_obs)),
        intrinsic_neighbour_ratio=intrinsic,
        near_optimal_eta=0.9,
        device_energy_ge_0p9=int(sum(R["observed"]["verified_energy_ratio"]["k1"] >= 0.9 for R in rows_out)),
        predicted_energy_ge_0p9=float(sum(R["law_mc"]["p_energy_ge_0.9_mc"] for R in rows_out)),
        prereg_p_top1_vs_shots=prereg_shots,
        prereg_synthetic_9q=synth,
        proposer_verifier={f"k{k}": dict(
            exact_best_recovered=int(sum(R["observed"]["verified_energy_ratio"][f"k{k}"] >= 1 - 1e-12
                                         for R in rows_out)),
            mean_energy_ratio=float(np.mean([R["observed"]["verified_energy_ratio"][f"k{k}"]
                                             for R in rows_out])),
            min_energy_ratio_6_7q=float(min(R["observed"]["verified_energy_ratio"][f"k{k}"]
                                            for R in rows_out if R["qubits"] <= 7)))
            for k in (1, 2, 3, 5, 8)},
        S_star_range_6_7q=[float(min(R["S_star_95"] for R in rows_out if R["qubits"] <= 7)),
                           float(max(R["S_star_95"] for R in rows_out if R["qubits"] <= 7))],
        S_star_9q=[R["S_star_95"] for R in rows_out if R["qubits"] > 7],
        F_xeb_range_by_m={str(mm): [float(min(R["F_xeb"] for R in rows_out if R["qubits"] == mm)),
                                    float(max(R["F_xeb"] for R in rows_out if R["qubits"] == mm))]
                          for mm in sorted({R["qubits"] for R in rows_out})},
        PR_inband_range_by_m={str(mm): [float(min(R["pr_inband"] for R in rows_out if R["qubits"] == mm)),
                                        float(max(R["pr_inband"] for R in rows_out if R["qubits"] == mm))]
                              for mm in sorted({R["qubits"] for R in rows_out})},
        M_inband_by_m={str(R["qubits"]): R["M_inband"] for R in rows_out},
        auc_gap_top1=auc(gap_abs, y), auc_negPR_top1=auc([-v for v in prs], y),
        variance_regimes={k: sum(1 for R in rows_out if R["variance_regime"] == k)
                          for k in ("signal", "floor")},
    )
    print("\n" + json.dumps(agg, indent=1))
    json.dump(dict(source=str(SRC.relative_to(ROOT)).replace("\\", "/"),
                   backend=src["backend"], calibration_last_update=src["calibration_last_update"],
                   mc_reps=MC_REPS, z95=Z95, aggregates=agg, rows=rows_out),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

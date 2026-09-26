#!/usr/bin/env python
"""A noisy-sampler law for chirplet atom selection, tested on the Sep 25 ibm_marrakesh data.

The QACT selection circuit is a sampler for the exact distribution p over window-local
frequency bins (Proposition "dequant": p is a periodogram of the enveloped, de-chirped
signal). On a noisy device the histogram q is modelled by a global depolarizing law

    q = F p + (1 - F) u,        u uniform over the M unmasked (in-band) outcomes,

and the question is how far one scalar F, plus the shot count, predicts whether the
device histogram ranks the exact best atom first (top-1) or among its top 3 / top 5.
For every one of the 26 device circuits this script

  1. rebuilds the exact target with qpu_selection.make_case (no hand-derived circuits),
     checks it against the stored want_peak_index / p_peak_exact, and re-scores the
     stored device counts with qpu_selection.score to make sure the file is read right;
  2. reports the rank of the exact best in-band atom in the device histogram (stable
     order: count descending, bin index ascending, which is np.argmax's tie rule), the
     top-1/3/5 indicators, and the analytic uniform-proposal baseline k/M;
  3. estimates F from the histogram with the linear cross-entropy estimator
     F_hat = (M sum p q - 1) / (M sum p^2 - 1) (unbiased under the law), on the masked
     space (the selection-relevant one) and on the full 2^m space, next to the
     Hellinger fidelities of the FakeMarrakesh prediction and the device;
  4. tests the law itself: TVD between the device histogram and F_hat p + (1-F_hat) u,
     against the multinomial sampling distribution of that TVD at the same shot count;
  5. predicts P(top-1), P(top-3), P(top-5) and the expected selected-atom energy by
     multinomial Monte Carlo at three values of F: the device's own F_hat (in-sample),
     the FakeMarrakesh histogram's F_hat (a priori: known before the device ran), and
     F = 1 (shot noise only); calibration = Brier score against a constant baseline and
     Spearman correlation of predicted success with observed selected-atom energy;
  6. runs the proposer-verifier pipeline: the device's top-k in-band bins (k = 1, 3, 5)
     are candidate atoms; each candidate's exact energy is evaluated classically by a
     direct O(N) inner product with the signal (no FFT); the best is kept and optionally
     refined with the paper's classical hardware refiner (act_gpu.refine_coord_batch,
     3-point coordinate search, 4 sweeps, float64 on CPU).

CPU only. The FakeMarrakesh histograms are regenerated exactly as qpu_collect.py does
(seed_transpiler 7, seed_simulator 2, 4000 shots) and checked against the stored
fake_noisy scores; they are cached in results/qpu_law_fake_counts.json.

    python paper/iclr2027/exp/qpu_law_analysis.py [--mc 20000] [--no-fake]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from qpu_selection import (FS, N, OUT, TC, counts_to_p, eeg_windows, envelope,  # noqa: E402
                           make_case, score)

SRC = OUT / "qpu_selection_marrakesh1.json"
KS = (1, 3, 5)


# ----------------------------------------------------------------------------- helpers
def stable_rank(v, idx):
    """1-based rank of entry idx in v under (value desc, index asc) - np.argmax's rule."""
    return int(np.sum(v > v[idx]) + np.sum(v[:idx] == v[idx]) + 1)


def topk_indices(v, k):
    order = np.lexsort((np.arange(len(v)), -v))   # value desc, index asc
    return order[:k]


def xeb_F(p, q):
    """Linear-XEB fidelity under q = F p + (1-F)/M; returns (F_hat, analytic s.e. factor).
    The s.e. needs the shot count; returned as sqrt(var per shot) * M / denom."""
    M = len(p)
    den = M * np.sum(p * p) - 1.0
    F = (M * np.sum(p * q) - 1.0) / den
    var1 = max(np.sum(p * p * q) - np.sum(p * q) ** 2, 0.0)   # per-shot variance of p(X)
    return float(F), float(M * math.sqrt(var1) / den)


def hellinger_fid(a, b):
    return float(np.sum(np.sqrt(a * b)) ** 2)


def mc_law(p, F, shots, peak, energy, R, rng):
    """Monte Carlo of the law on the masked space: shots ~ Multinomial(shots, F p + (1-F) u).
    Returns P(rank<=k) for k in KS, E[selected-atom energy], and the TVD sampling law."""
    M = len(p)
    q = np.clip(F, 0.0, 1.0) * p + (1 - np.clip(F, 0.0, 1.0)) / M
    q = q / q.sum()
    X = rng.multinomial(shots, q, size=R)
    # stable rank of `peak`: strictly larger counts + ties at lower index
    xp = X[:, peak:peak + 1]
    rank = (X > xp).sum(1) + (X[:, :peak] == xp).sum(1) + 1
    sel = np.argmax(X, axis=1)
    out = {f"p_top{k}": float(np.mean(rank <= k)) for k in KS}
    out["exp_sel_energy"] = float(np.mean(energy[sel]))
    return out, q, X


def tvd_law_test(p, F, qhat, shots, R, rng):
    """Observed TVD(qhat, model) and its sampling distribution when data follow the model
    (F re-estimated on every synthetic histogram, as it was on the real one)."""
    M = len(p)
    Fc = float(np.clip(F, 0.0, 1.0))
    model = Fc * p + (1 - Fc) / M
    obs = 0.5 * float(np.abs(qhat - model).sum())
    X = rng.multinomial(shots, model, size=R) / shots
    den = M * np.sum(p * p) - 1.0
    Fs = np.clip((M * (X @ p) - 1.0) / den, 0.0, 1.0)
    mods = Fs[:, None] * p[None, :] + (1 - Fs[:, None]) / M
    sim = 0.5 * np.abs(X - mods).sum(1)
    return obs, float(np.mean(sim)), float(np.mean(sim >= obs))


def atom_energy_direct(x, env, c, freq_bin):
    """Exact energy of the complex chirplet atom at a full-grid frequency bin, by a direct
    O(N) inner product: |sum_t x(t) env(t) exp(-i 2pi (c t^2 + k t)/N)|^2 / ||env x||^2.
    This is the quantity the circuit's distribution is proportional to."""
    t = np.arange(N)
    s = env * x
    s = s / np.linalg.norm(s)
    return float(np.abs(np.sum(s * np.exp(-2j * np.pi * (c * t ** 2 + freq_bin * t) / N))) ** 2 / N)


def atom_params(freq_bin, c, logdt):
    """Physical (tc [s], fc [Hz], dt [s], chirp [Hz/s]) of the circuit's atom at a bin, in
    act_gpu's convention (phase 2pi(fc tau + c tau^2 / 2))."""
    return (TC / FS, (freq_bin + 2 * c * TC) * FS / N, math.exp(logdt) / FS, 2 * c * FS ** 2 / N)


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None, None
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def brier(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    base = obs.mean()
    return float(np.mean((pred - obs) ** 2)), float(np.mean((base - obs) ** 2)), float(base)


# ----------------------------------------------------------------------------- fake backend
def fake_counts(cases, shots, cache):
    """FakeMarrakesh histograms, cached incrementally (keyed by case) so a run that hits the
    machine's commit limit resumes where it stopped."""
    tag = lambda cs: f"{cs['subject']}|{cs['channel']}|{cs['logdt']}|{cs['variant']}"  # noqa: E731
    d = json.load(open(cache)) if cache.exists() else {}
    if isinstance(d, dict) and all(tag(cs) in d for cs in cases):
        return [d[tag(cs)] for cs in cases]
    d = d if isinstance(d, dict) else {}
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fake = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=3, backend=fake, seed_transpiler=7)
    noisy = AerSimulator.from_backend(fake)
    for cs in cases:
        if tag(cs) in d:
            continue
        isa = pm.run(cs["qc"])
        for attempt in range(4):
            try:
                cnt = noisy.run(isa, shots=shots, seed_simulator=2).result().get_counts()
                break
            except MemoryError:
                print("  MemoryError (commit limit); retrying with fewer threads", flush=True)
                noisy.set_options(max_parallel_threads=max(1, 4 >> attempt))
                time.sleep(20)
        else:
            raise MemoryError("FakeMarrakesh simulation failed 4 times")
        d[tag(cs)] = {k: int(v) for k, v in cnt.items()}
        json.dump(d, open(cache, "w"))
        print(f"  fake {tag(cs)} done", flush=True)
    return [d[tag(cs)] for cs in cases]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mc", type=int, default=20000, help="Monte Carlo repetitions per row")
    ap.add_argument("--no-fake", action="store_true", help="skip FakeMarrakesh histograms")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    src = json.load(open(SRC))
    shots_cfg = int(src["shots"])
    c = src["rate_hz_s"] * N / (2 * FS ** 2)

    # rebuild cases in the order qpu_collect.py used, then match rows by key
    wins = eeg_windows()
    cases = []
    for w in wins:
        for ld in (1.5, 2.7):
            for variant in ("dyn", "uni"):
                cases.append(make_case(w, ld, c, variant))
    for variant in ("dyn", "uni"):
        cases.append(make_case(wins[0], 3.9, c, variant))
    xs = {(w["subject"], w["channel"]): w["x"] for w in wins}
    key = lambda r: (r["subject"], r["channel"], float(r["logdt"]), r["variant"])  # noqa: E731
    rows = {key(r): r for r in src["rows"]}
    assert len(rows) == len(cases) == 26

    fcounts = None if args.no_fake else fake_counts(cases, shots_cfg, OUT / "qpu_law_fake_counts.json")

    torch = None
    if not args.no_refine:
        import torch as _t
        torch = _t
        from qbe import act_gpu

    per, work = [], []
    checks = dict(max_abs_p_peak_exact=0.0, peak_index_mismatch=0, device_rescore_max_diff=0.0,
                  fake_rescore_max_fid_diff=0.0, direct_energy_max_rel_err=0.0)
    for ci, cs in enumerate(cases):
        r = rows[key(cs)]
        m, want, mask, peak = cs["qubits"], cs["want"], cs["mask"], cs["peak"]
        # ---- consistency with the stored file
        checks["peak_index_mismatch"] += int(peak != r["want_peak_index"])
        checks["max_abs_p_peak_exact"] = max(checks["max_abs_p_peak_exact"],
                                             abs(want[peak] - r["device"]["p_peak_exact"]))
        qfull = counts_to_p(r["device"]["counts"], m)
        rs = score(qfull, cs)
        for f in ("tvd", "fidelity", "selected_atom_energy", "peak_ok"):
            checks["device_rescore_max_diff"] = max(checks["device_rescore_max_diff"],
                                                    abs(rs[f] - r["device"][f]))
        shots = int(r["device"]["shots"])

        # ---- masked (selection) space
        idx = np.flatnonzero(mask)
        M = len(idx)
        p = want[idx] / want[idx].sum()
        cnt_full = np.rint(qfull * shots).astype(int)
        cnt = cnt_full[idx]
        n_in = int(cnt.sum())
        q = cnt / max(n_in, 1)
        pk = int(np.flatnonzero(idx == peak)[0])
        energy = want[idx] / want[idx].max()          # selected-atom energy scale (score())
        rank = stable_rank(cnt.astype(float), pk)
        F_m, se1 = xeb_F(p, q)
        F_f, se1f = xeb_F(want, qfull)
        fid_mask_dev = hellinger_fid(p, q)

        # ---- a priori F from the FakeMarrakesh histogram
        F_fake = F_fake_full = None
        if fcounts is not None:
            fq = counts_to_p(fcounts[ci], m)
            fs_ = score(fq, cs)
            checks["fake_rescore_max_fid_diff"] = max(checks["fake_rescore_max_fid_diff"],
                                                      abs(fs_["fidelity"] - r["fake_noisy"]["fidelity"]))
            fqm = fq[idx] / max(fq[idx].sum(), 1e-12)
            F_fake, _ = xeb_F(p, fqm)
            F_fake_full, _ = xeb_F(want, fq)

        # ---- law test and predictions
        tv_obs, tv_sim_mean, tv_pval = tvd_law_test(p, F_m, q, n_in, max(2000, args.mc // 5), rng)
        # leave-peak-out F: regress q_j - 1/M on p_j - 1/M over the bins j != best, so the
        # best atom's own count never enters the F used to predict its rank
        oth = np.arange(M) != pk
        F_lpo = float(np.sum((q[oth] - 1 / M) * (p[oth] - 1 / M)) / np.sum((p[oth] - 1 / M) ** 2))
        preds = {}
        work.append((p, pk, energy, n_in))

        # ---- proposer-verifier: device top-k -> exact O(N) energy -> keep best
        x = xs[(cs["subject"], cs["channel"])]
        env = envelope(TC, cs["logdt"])
        e_direct_all = np.array([atom_energy_direct(x, env, c, cs["bins"][j]) for j in idx])
        rel = e_direct_all / e_direct_all.max()
        checks["direct_energy_max_rel_err"] = max(checks["direct_energy_max_rel_err"],
                                                  float(np.max(np.abs(rel - energy))))
        assert int(np.argmax(e_direct_all)) == pk
        pv = {}
        order_E = np.sort(rel)
        for k in KS:
            top = topk_indices(cnt.astype(float), k)
            best = int(top[np.argmax(rel[top])])
            # uniform proposal: E[max energy of k distinct uniform bins], exact
            ks = min(k, M)
            wts = np.array([math.comb(j, ks - 1) for j in range(M)], float) / math.comb(M, ks)
            pv[f"k{k}"] = dict(hit=int(best == pk), energy_ratio=float(rel[best]),
                               best_local=best, uniform_hit_prob=ks / M,
                               uniform_exp_energy=float(np.sum(wts * order_E)))

        # ---- classical refinement of the verified candidate (paper's hw refiner)
        if torch is not None:
            X = torch.tensor(x / np.linalg.norm(x), dtype=torch.float64)[None, :]
            t = torch.arange(N, dtype=torch.float64) / FS
            def quad_E(P):
                P = torch.tensor(np.asarray(P, float), dtype=torch.float64)
                gc, gs = act_gpu._quad(t, P[:, 0], P[:, 1], P[:, 2], P[:, 3])
                return act_gpu._energy(X.expand(P.shape[0], -1), gc, gs)[0].numpy()
            starts = [atom_params(cs["bins"][idx[pk]], c, cs["logdt"])]
            starts += [atom_params(cs["bins"][idx[pv[f"k{k}"]["best_local"]]], c, cs["logdt"])
                       for k in KS]
            P0 = torch.tensor(starts, dtype=torch.float64)
            Pr = act_gpu.refine_coord_batch(X.expand(len(starts), -1), P0, t, FS,
                                            sweeps=args.sweeps)
            E0, Er = quad_E(P0.numpy()), quad_E(Pr.numpy())
            ref = dict(exact_best_quad_E=float(E0[0]), exact_best_refined_E=float(Er[0]))
            for j, k in enumerate(KS):
                ref[f"k{k}"] = dict(quad_E=float(E0[j + 1]), refined_E=float(Er[j + 1]),
                                    ratio_unrefined_vs_exact=float(E0[j + 1] / E0[0]),
                                    ratio_refined_vs_exact=float(Er[j + 1] / E0[0]),
                                    ratio_refined_vs_refined_exact=float(Er[j + 1] / Er[0]),
                                    refined_params=[float(v) for v in Pr[j + 1].numpy()])
        else:
            ref = None

        per.append(dict(
            subject=cs["subject"], channel=cs["channel"], logdt=cs["logdt"], variant=cs["variant"],
            qubits=m, cz=r["cz"], cz_device=r.get("cz_device"), M_unmasked=M, shots=shots,
            shots_in_band=n_in, rank_best_atom=rank,
            **{f"top{k}": int(rank <= k) for k in KS},
            uniform_top_prob={f"top{k}": min(k, M) / M for k in KS},
            selected_atom_energy_device=float(r["device"]["selected_atom_energy"]),
            selected_atom_energy_fake=float(r["fake_noisy"]["selected_atom_energy"]),
            F_xeb_masked=F_m, F_xeb_masked_se=se1 / math.sqrt(max(n_in, 1)),
            F_xeb_full=F_f, F_xeb_full_se=se1f / math.sqrt(shots),
            F_xeb_fake_masked=F_fake, F_xeb_fake_full=F_fake_full, F_leave_peak_out=F_lpo,
            hellinger_fake=float(r["fake_noisy"]["fidelity"]),
            hellinger_device=float(r["device"]["fidelity"]),
            hellinger_device_masked=fid_mask_dev,
            p_best_exact_masked=float(p[pk]), p_second_exact_masked=float(np.sort(p)[-2]),
            law_tvd_obs=tv_obs, law_tvd_sampling_mean=tv_sim_mean, law_tvd_pvalue=tv_pval,
            pred=preds, proposer_verifier=pv, refine=ref))

    # ---- predictions (second pass: the rescaled fake F needs every other row)
    #   dev        the device histogram's own F_xeb (in-sample; the best atom's count enters F)
    #   dev_lpo    leave-peak-out F (the best atom's count never enters F)
    #   fake       F_xeb of the FakeMarrakesh histogram (a priori, known before the device ran)
    #   fake_loo   fake F times one device-correction factor sum F_dev / sum F_fake fitted on
    #              the other 25 circuits (a priori up to one constant, leave-one-out)
    #   ideal      F = 1: a perfect sampler, shot noise only
    have_fake = per[0]["F_xeb_fake_masked"] is not None
    for i, (r, (p, pk, energy, n_in)) in enumerate(zip(per, work)):
        srcs = [("dev", r["F_xeb_masked"]), ("dev_lpo", r["F_leave_peak_out"])]
        if have_fake:
            fd = sum(o["F_xeb_masked"] for j, o in enumerate(per) if j != i)
            ff = sum(o["F_xeb_fake_masked"] for j, o in enumerate(per) if j != i)
            srcs += [("fake", r["F_xeb_fake_masked"]), ("fake_loo", r["F_xeb_fake_masked"] * fd / ff)]
        srcs.append(("ideal", 1.0))
        for lab, F in srcs:
            r["pred"][lab], _, _ = mc_law(p, F, n_in, pk, energy, args.mc, rng)
            r["pred"][lab]["F"] = float(F)
        pr = r["pred"]
        print(f"S{r['subject']:03d} {r['channel']:>3} ld{r['logdt']} {r['variant']} m={r['qubits']} "
              f"M={r['M_unmasked']} rank={r['rank_best_atom']} F_xeb={r['F_xeb_masked']:.3f} "
              f"F_lpo={r['F_leave_peak_out']:.3f} P1 " + " ".join(f"{k}={v['p_top1']:.2f}" for k, v in pr.items())
              + f" lawTVD p={r['law_tvd_pvalue']:.3f}", flush=True)

    # ----------------------------------------------------------------------------- aggregates
    def group(sel):
        g = [r for r in per if sel(r)]
        if not g:
            return None
        out = dict(n=len(g))
        for k in KS:
            out[f"obs_top{k}"] = int(sum(r[f"top{k}"] for r in g))
            out[f"uniform_top{k}_expected"] = float(sum(r["uniform_top_prob"][f"top{k}"] for r in g))
            for lab in ("dev", "dev_lpo", "fake", "fake_loo", "ideal"):
                if all(lab in r["pred"] for r in g):
                    out[f"pred_{lab}_top{k}_expected"] = float(sum(r["pred"][lab][f"p_top{k}"] for r in g))
            out[f"pv_k{k}_hits"] = int(sum(r["proposer_verifier"][f"k{k}"]["hit"] for r in g))
            out[f"pv_k{k}_mean_energy"] = float(np.mean([r["proposer_verifier"][f"k{k}"]["energy_ratio"] for r in g]))
            out[f"pv_k{k}_min_energy"] = float(np.min([r["proposer_verifier"][f"k{k}"]["energy_ratio"] for r in g]))
            out[f"uniform_k{k}_mean_exp_energy"] = float(np.mean([r["proposer_verifier"][f"k{k}"]["uniform_exp_energy"] for r in g]))
            if g[0]["refine"] is not None:
                out[f"refined_k{k}_mean_ratio_vs_exact"] = float(np.mean([r["refine"][f"k{k}"]["ratio_refined_vs_exact"] for r in g]))
                out[f"refined_k{k}_min_ratio_vs_exact"] = float(np.min([r["refine"][f"k{k}"]["ratio_refined_vs_exact"] for r in g]))
                out[f"refined_k{k}_mean_ratio_vs_refined_exact"] = float(np.mean([r["refine"][f"k{k}"]["ratio_refined_vs_refined_exact"] for r in g]))
                out[f"refined_k{k}_min_ratio_vs_refined_exact"] = float(np.min([r["refine"][f"k{k}"]["ratio_refined_vs_refined_exact"] for r in g]))
                out[f"refined_k{k}_reach99_refined_exact"] = int(sum(r["refine"][f"k{k}"]["ratio_refined_vs_refined_exact"] >= 0.99 for r in g))
                out[f"refined_k{k}_reach_exact"] = int(sum(r["refine"][f"k{k}"]["ratio_refined_vs_exact"] >= 1.0 for r in g))
        out["mean_F_xeb_masked"] = float(np.mean([r["F_xeb_masked"] for r in g]))
        out["mean_F_xeb_full"] = float(np.mean([r["F_xeb_full"] for r in g]))
        if g[0]["F_xeb_fake_masked"] is not None:
            out["mean_F_xeb_fake_masked"] = float(np.mean([r["F_xeb_fake_masked"] for r in g]))
        out["mean_hellinger_device"] = float(np.mean([r["hellinger_device"] for r in g]))
        out["mean_hellinger_fake"] = float(np.mean([r["hellinger_fake"] for r in g]))
        out["law_rejected_at_0.05"] = int(sum(r["law_tvd_pvalue"] < 0.05 for r in g))
        return out

    groups = {"all": group(lambda r: True)}
    for m in sorted({r["qubits"] for r in per}):
        groups[f"q{m}"] = group(lambda r, m=m: r["qubits"] == m)
    for v in ("dyn", "uni"):
        groups[v] = group(lambda r, v=v: r["variant"] == v)

    calib = {}
    obs_E = [r["selected_atom_energy_device"] for r in per]
    for lab in ("dev", "dev_lpo", "fake", "fake_loo", "ideal"):
        if not all(lab in r["pred"] for r in per):
            continue
        cl = {}
        for k in KS:
            pred = [r["pred"][lab][f"p_top{k}"] for r in per]
            obs = [r[f"top{k}"] for r in per]
            b, b0, base = brier(pred, obs)
            rho, pval = spearman(pred, obs_E)
            cl[f"top{k}"] = dict(brier=b, brier_constant=b0, base_rate=base,
                                 predicted_total=float(sum(pred)), observed_total=int(sum(obs)),
                                 spearman_pred_vs_sel_energy=rho, spearman_p=pval)
        pe = [r["pred"][lab]["exp_sel_energy"] for r in per]
        rho, pval = spearman(pe, obs_E)
        cl["exp_sel_energy"] = dict(mae=float(np.mean(np.abs(np.array(pe) - np.array(obs_E)))),
                                    mae_constant=float(np.mean(np.abs(np.mean(obs_E) - np.array(obs_E)))),
                                    spearman=rho, spearman_p=pval)
        calib[lab] = cl
    rho_F, p_F = spearman([r["F_xeb_masked"] for r in per], [r["hellinger_device"] for r in per])
    rho_Ff, p_Ff = (spearman([r["F_xeb_fake_masked"] for r in per], [r["F_xeb_masked"] for r in per])
                    if per[0]["F_xeb_fake_masked"] is not None else (None, None))
    fid_rel = dict(spearman_Fxeb_vs_hellinger_device=rho_F, p=p_F,
                   spearman_Fxeb_fake_vs_Fxeb_device=rho_Ff, p_fake=p_Ff,
                   mean_gap_fake_minus_device_Fxeb=(float(np.mean([r["F_xeb_fake_masked"] - r["F_xeb_masked"] for r in per]))
                                                    if per[0]["F_xeb_fake_masked"] is not None else None))

    res = dict(source=str(SRC.relative_to(HERE.parents[2])), backend=src["backend"], jobs=src["jobs"],
               calibration_last_update=src["calibration_last_update"], mc=args.mc, seed=args.seed,
               refine_sweeps=None if args.no_refine else args.sweeps, checks=checks,
               groups=groups, calibration=calib, fidelity_relations=fid_rel, rows=per,
               runtime_s=time.time() - t_start)
    json.dump(res, open(OUT / "qpu_law_existing.json", "w"), indent=1)
    write_md(res)
    print(json.dumps(dict(checks=checks, groups=groups, calibration=calib,
                          fidelity_relations=fid_rel), indent=1))


def f3(v, d=3):
    return "--" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def write_md(res):
    L = ["# Noisy-sampler law on the ibm_marrakesh selection data",
         "",
         f"Source: `{res['source']}` (backend {res['backend']}, calibration {res['calibration_last_update']}). "
         f"Generated by `paper/iclr2027/exp/qpu_law_analysis.py` (MC {res['mc']} per row, seed {res['seed']}).",
         "",
         "Law: q = F p + (1-F) u on the M unmasked (in-band) bins. F_xeb = (M sum p q - 1)/(M sum p^2 - 1).",
         "Rank = stable rank of the exact best in-band atom in the device histogram (count desc, index asc).",
         "Prediction sources: dev = device F_xeb (in-sample); dev_lpo = leave-peak-out F (best atom's count excluded);",
         "fake = FakeMarrakesh F_xeb (a priori); fake_loo = fake F x one device factor fitted on the other 25 circuits;",
         "ideal = F 1 (shot noise only). Energies: sel.E and pv use the circuit's objective (complex overlap);",
         "ref k1 = best-of-top-1 refined by act_gpu.refine_coord_batch, quadrature energy / exact best atom's (unrefined).",
         "",
         "## Consistency checks", ""]
    for k, v in res["checks"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Per circuit", "",
          "| subj | ch | logdt | var | q | CZ | M | rank | top1 | top3 | top5 | unif P(top1) | F_xeb mask | F lpo | F_xeb fake | Hell fake | Hell dev | law TVD p | P1 dev | P1 lpo | P1 fake | P1 fake_loo | P1 ideal | P3 lpo | sel.E dev | pv k3 E | ref k1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        pr = r["pred"]
        L.append("| " + " | ".join([
            str(r["subject"]), r["channel"], f"{r['logdt']}", r["variant"], str(r["qubits"]),
            str(r["cz"]), str(r["M_unmasked"]), str(r["rank_best_atom"]), str(r["top1"]), str(r["top3"]),
            str(r["top5"]), f3(r["uniform_top_prob"]["top1"]), f3(r["F_xeb_masked"]),
            f3(r["F_leave_peak_out"]), f3(r["F_xeb_fake_masked"]), f3(r["hellinger_fake"]),
            f3(r["hellinger_device"]), f3(r["law_tvd_pvalue"]), f3(pr["dev"]["p_top1"], 2),
            f3(pr["dev_lpo"]["p_top1"], 2), f3(pr.get("fake", {}).get("p_top1"), 2),
            f3(pr.get("fake_loo", {}).get("p_top1"), 2), f3(pr["ideal"]["p_top1"], 2),
            f3(pr["dev_lpo"]["p_top3"], 2), f3(r["selected_atom_energy_device"], 2),
            f3(r["proposer_verifier"]["k3"]["energy_ratio"], 2),
            f3(r["refine"]["k1"]["ratio_refined_vs_exact"], 2) if r["refine"] else "--"]) + " |")
    L += ["", "## By group", ""]
    keys = ["n"]
    for k in KS:
        keys += [f"obs_top{k}", f"uniform_top{k}_expected"] +                 [f"pred_{lab}_top{k}_expected" for lab in ("dev", "dev_lpo", "fake", "fake_loo", "ideal")]
    keys += [
            "pv_k1_hits", "pv_k3_hits", "pv_k5_hits", "pv_k1_mean_energy", "pv_k3_mean_energy",
            "pv_k5_mean_energy", "pv_k1_min_energy", "pv_k3_min_energy", "pv_k5_min_energy",
            "uniform_k1_mean_exp_energy", "uniform_k3_mean_exp_energy", "uniform_k5_mean_exp_energy",
            "refined_k1_mean_ratio_vs_exact", "refined_k3_mean_ratio_vs_exact", "refined_k5_mean_ratio_vs_exact",
            "refined_k1_min_ratio_vs_exact", "refined_k3_min_ratio_vs_exact",
            "refined_k1_mean_ratio_vs_refined_exact", "refined_k3_mean_ratio_vs_refined_exact",
            "refined_k1_min_ratio_vs_refined_exact", "refined_k3_min_ratio_vs_refined_exact",
            "refined_k1_reach_exact", "refined_k3_reach_exact", "refined_k5_reach_exact",
            "refined_k1_reach99_refined_exact", "refined_k3_reach99_refined_exact", "refined_k5_reach99_refined_exact",
            "refined_k5_mean_ratio_vs_refined_exact", "refined_k5_min_ratio_vs_refined_exact",
            "mean_F_xeb_masked", "mean_F_xeb_full", "mean_F_xeb_fake_masked", "mean_hellinger_device",
            "mean_hellinger_fake", "law_rejected_at_0.05"]
    gs = [g for g in res["groups"] if res["groups"][g]]
    L.append("| metric | " + " | ".join(gs) + " |")
    L.append("|---|" + "---|" * len(gs))
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f3(res["groups"][g].get(k)) for g in gs) + " |")
    L += ["", "## Calibration across the 26 circuits", "",
          "| F source | event | Brier | Brier const | base rate | pred total | obs total | Spearman(pred, sel.E) | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, cl in res["calibration"].items():
        for k in KS:
            c = cl[f"top{k}"]
            L.append(f"| {lab} | top{k} | {f3(c['brier'])} | {f3(c['brier_constant'])} | {f3(c['base_rate'])} | "
                     f"{f3(c['predicted_total'], 2)} | {c['observed_total']} | "
                     f"{f3(c['spearman_pred_vs_sel_energy'])} | {f3(c['spearman_p'], 4)} |")
        e = cl["exp_sel_energy"]
        L.append(f"| {lab} | E[sel.E] | MAE {f3(e['mae'])} | MAE const {f3(e['mae_constant'])} | | | | "
                 f"{f3(e['spearman'])} | {f3(e['spearman_p'], 4)} |")
    L += ["", "## Fidelity estimators", ""]
    for k, v in res["fidelity_relations"].items():
        L.append(f"- {k}: {f3(v, 4) if isinstance(v, float) else v}")
    L.append("")
    (OUT / "qpu_law_existing.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

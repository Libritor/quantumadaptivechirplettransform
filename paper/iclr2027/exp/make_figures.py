#!/usr/bin/env python
"""Figures and number macros for the ICLR paper, from the committed result files.

Reads
  results/qact_hardware.json           (repo root; the fake-backend hardware study)
  results/qact_vs_act_speed.json       (repo root; end-to-end speed)
  paper/iclr2027/results/qpu_selection_*.json      (real device)
  paper/iclr2027/results/fair_control_refine.json  (refinement control)
  paper/iclr2027/results/fair_control_backfit.json (backfit / low-frequency grid control)
  paper/iclr2027/results/coherent_residual.json
Writes paper/iclr2027/figures/*.pdf and paper/iclr2027/numbers.tex.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
FIG = PAPER / "figures"
FIG.mkdir(exist_ok=True)
RES = PAPER / "results"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8,
                     "pdf.fonttype": 42})
BLUE, RED, GREY, GREEN = "#2b7bba", "#d62728", "#555555", "#2ca02c"
M = {}


def macro(name, value):
    M[name] = value


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(math.floor(math.log10(p)))
    m = int(round(p / 10 ** e))
    if m == 10:
        m, e = 1, e + 1
    return f"{m}\\times10^{{{e}}}"


# ----------------------------------------------------------------------------- hardware study
def fig_hardware():
    rows = json.load(open(ROOT / "results" / "qact_hardware.json"))
    kinds = ["baseline", "folded", "aqft", "semiclassical", "windowed"]
    labels = ["baseline", "folded", "aQFT", "scQFT", "windowed"]
    widths = [1.5, 2.7, 3.9, 5.1]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    cz = [np.mean([r["cz"] for r in rows if r["kind"] == k]) for k in kinds]
    fid = [np.mean([r["fidelity"] for r in rows if r["kind"] == k]) for k in kinds]
    sel = [np.mean([r["sel_quality"] for r in rows if r["kind"] == k]) for k in kinds]
    x = np.arange(len(kinds))
    axes[0].bar(x, cz, color=GREY)
    axes[0].set_ylabel("two-qubit gates (CZ)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=6.5, rotation=35, ha="right")
    axes[1].bar(x - 0.2, fid, 0.4, color=BLUE, label="Hellinger fidelity")
    axes[1].bar(x + 0.2, sel, 0.4, color=RED, label="selected-atom energy")
    axes[1].set_ylim(0, 1.15); axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=6.5, rotation=35, ha="right")
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].set_title("Heron noise model, mean of 12 cases", fontsize=8)
    for k, col, mk in (("folded", GREY, "s"), ("windowed", BLUE, "D")):
        y = [np.mean([r["sel_quality"] for r in rows if r["kind"] == k and r["logdt"] == w]) for w in widths]
        axes[2].plot(widths, y, marker=mk, color=col, label=k)
    axes[2].set_xlabel("envelope width $\\log\\Delta t$")
    axes[2].set_ylabel("selected-atom energy"); axes[2].set_ylim(0, 1.05)
    axes[2].legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_hardware.pdf", bbox_inches="tight")
    plt.close(fig)
    for k, c, f, s in zip(kinds, cz, fid, sel):
        macro(f"hw{k.capitalize()}CZ", f"{c:,.0f}")
        macro(f"hw{k.capitalize()}Fid", f"{f:.2f}")
        macro(f"hw{k.capitalize()}Sel", f"{s:.2f}")


# ----------------------------------------------------------------------------- real device
def fig_device():
    files = sorted(glob.glob(str(RES / "qpu_selection_2*.json")) + glob.glob(str(RES / "qpu_selection_marrakesh*.json")))
    standin = False
    if not files:
        files = sorted(glob.glob(str(RES / "qpu_selection_dryrun_*.json")))
        standin = True
        if not files:
            print("no device results yet")
            macro("qpuN", "--"); return None
    d = json.load(open(files[-1]))
    rows = d["rows"]
    if standin:
        # layout stand-in only: device := prediction, marked as such in the figure
        for r in rows:
            r.setdefault("device", dict(r["fake_noisy"], counts={}, job_id="none"))
            r.setdefault("cz_device", r["cz"])
        d.setdefault("backend", "prediction only")
        d.setdefault("jobs", {})
        d.setdefault("quota_before_s", 0.0); d.setdefault("quota_after_s", 0.0)
    short = [r for r in rows if r["qubits"] <= 7]
    long_ = [r for r in rows if r["qubits"] > 7]
    macro("qpuBackend", d["backend"].replace("_", "\\_"))
    macro("qpuN", str(len(rows)))
    macro("qpuShots", f"{d['shots']:,}")
    macro("qpuJobs", ", ".join(v[0] if isinstance(v, (list, tuple)) else str(v) for v in d["jobs"].values()) or "--")
    macro("qpuShortN", str(len(short)))
    macro("qpuShortPeakOK", str(sum(r["device"]["peak_ok"] for r in short)))
    macro("qpuShortSel", f"{np.mean([r['device']['selected_atom_energy'] for r in short]):.2f}")
    macro("qpuShortSelMin", f"{np.min([r['device']['selected_atom_energy'] for r in short]):.2f}")
    macro("qpuShortFid", f"{np.mean([r['device']['fidelity'] for r in short]):.2f}")
    macro("qpuShortFidSim", f"{np.mean([r['fake_noisy']['fidelity'] for r in short]):.2f}")
    macro("qpuShortTVD", f"{np.mean([r['device']['tvd'] for r in short]):.2f}")
    for v in ("dyn", "uni"):
        sub = [r for r in short if r["variant"] == v]
        macro(f"qpu{v.capitalize()}Fid", f"{np.mean([r['device']['fidelity'] for r in sub]):.2f}")
        macro(f"qpu{v.capitalize()}Sel", f"{np.mean([r['device']['selected_atom_energy'] for r in sub]):.2f}")
        macro(f"qpu{v.capitalize()}PeakOK", str(sum(r["device"]["peak_ok"] for r in sub)))
        macro(f"qpu{v.capitalize()}N", str(len(sub)))
        macro(f"qpu{v.capitalize()}CZ", f"{np.mean([r['cz_device'] for r in sub]):.0f}")
    if long_:
        macro("qpuLongFid", f"{np.mean([r['device']['fidelity'] for r in long_]):.2f}")
        macro("qpuLongSel", f"{np.mean([r['device']['selected_atom_energy'] for r in long_]):.2f}")
        macro("qpuLongPeakOK", str(sum(r["device"]["peak_ok"] for r in long_)))
        macro("qpuLongCZ", f"{np.mean([r['cz_device'] for r in long_]):.0f}")
    macro("qpuSeconds", f"{d['quota_before_s'] - d['quota_after_s']:.0f}")
    # figure: device histogram vs exact for the first 6-qubit dynamic case, plus a
    # scatter of predicted (fake noise) vs measured selected-atom energy
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.6, 1]})
    case = next(r for r in rows if r["qubits"] == 6 and r["variant"] == "dyn")
    m = case["qubits"]
    cnt = case["device"]["counts"]
    p = np.zeros(2 ** m)
    for k, v in cnt.items():
        p[int(k.split()[-1], 2)] += v
    p = p / p.sum() if p.sum() > 0 else p
    # exact target: recompute from the saved case description is not possible without
    # the window, so the exact peak probability and index are read from the record
    ax = axes[0]
    ax.bar(np.arange(2 ** m), p, color=BLUE, width=0.8, label=f"{d['backend']} ({d['shots']:,} shots)")
    ax.axvline(case["want_peak_index"], color=RED, ls="--", lw=1, label="exact in-band peak")
    ax.set_xlabel("measured frequency bin $k'$ (window register)"); ax.set_ylabel("probability")
    ax.set_title(f"6-qubit windowed circuit, {case['cz_device']} CZ: device histogram" + (" (STAND-IN)" if standin else ""), fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for v, col, mk in (("dyn", BLUE, "o"), ("uni", RED, "s")):
        sub = [r for r in rows if r["variant"] == v]
        ax.scatter([r["fake_noisy"]["selected_atom_energy"] for r in sub],
                   [r["device"]["selected_atom_energy"] for r in sub], s=18, color=col, marker=mk,
                   label={"dyn": "semiclassical QFT", "uni": "unitary QFT"}[v])
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
    ax.set_xlabel("predicted (noise model)"); ax.set_ylabel("measured (device)")
    ax.set_title("selected-atom energy, all circuits", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_device.pdf", bbox_inches="tight")
    plt.close(fig)
    return d


# ----------------------------------------------------------------------------- fair control
def fig_fair():
    d = json.load(open(RES / "fair_control_refine.json"))
    E = d["engines"]
    names = list(E)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, P in zip(axes, ("6", "12")):
        for name in names:
            r = E[name][P]
            cls = name.startswith("classical")
            ax.scatter(r["ms_per_window"], r["mean_err"], s=26,
                       color=(GREY if cls else BLUE), marker=("o" if cls else "D"))
            ax.annotate(name.replace("classical ", "cl. ").replace(" (hw-4, exact-f, backfit 1)", ""),
                        (r["ms_per_window"], r["mean_err"]), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("ms per window (GPU, batch 320)")
        ax.set_title(f"order {P}: mean relative residual, 320 public EEG windows", fontsize=8)
    axes[0].set_ylabel("reconstruction error")
    fig.tight_layout()
    fig.savefig(FIG / "fig_fair.pdf", bbox_inches="tight")
    plt.close(fig)
    key = {"classical Adam-60": "ClAdam", "classical Adam-4": "ClAdamFour", "classical coord-4": "ClCoord",
           "classical fine-f Adam-60": "ClFineAdam", "classical fine-f coord-4": "ClFineCoord",
           "QACT shift-4": "QShift", "QACT hw-4": "QHw", "QACT parity (hw-4, exact-f, backfit 1)": "QParity"}
    for name, k in key.items():
        for P, O in (("6", "Six"), ("12", "Twelve")):
            r = E[name][P]
            macro(f"fair{k}Err{O}", f"{r['mean_err']:.3f}")
            macro(f"fair{k}Rel{O}", f"{100 * r['rel_vs_ref']:+.1f}")
            macro(f"fair{k}Better{O}", str(r["better"]))
            macro(f"fair{k}P{O}", fmt_p(max(r["wilcoxon_p"], 1e-300)))
            macro(f"fair{k}Ms{O}", f"{r['ms_per_window']:.1f}")
    macro("fairN", str(d["n_windows"]))
    # paired coord-4 vs hw-4
    for P, O in (("6", "Six"), ("12", "Twelve")):
        c = np.array(E["classical coord-4"][P]["err"]); h = np.array(E["QACT hw-4"][P]["err"])
        from scipy.stats import wilcoxon
        macro(f"fairCoordVsHwBetter{O}", str(int((c < h).sum())))
        macro(f"fairCoordVsHwP{O}", fmt_p(float(wilcoxon(c, h).pvalue)))
        macro(f"fairCoordVsHwRel{O}", f"{100 * (c.mean() / h.mean() - 1):+.1f}")
        ref = np.array(E["classical Adam-60"][P]["err"]); q = np.array(E["QACT parity (hw-4, exact-f, backfit 1)"][P]["err"])
        macro(f"fairParityMedianDiff{O}", f"{np.median(q - ref):+.3f}")
        macro(f"fairRefFail{O}", str(int((ref > 0.9).sum())))
    bf = RES / "fair_control_backfit.json"
    if bf.exists():
        b = json.load(open(bf))
        EB = b["engines"]
        key2 = {"classical Adam-60": "BfRef", "classical Adam-60 + backfit 1": "BfOne",
                "classical Adam-60, fc to 78 Hz": "BfLow", "classical Adam-60, fc to 78 Hz + backfit 1": "BfLowOne",
                "QACT parity (hw-4, exact-f, backfit 1)": "BfParity"}
        for name, k in key2.items():
            if name not in EB:
                continue
            for P, O in (("6", "Six"), ("12", "Twelve")):
                r = EB[name][P]
                macro(f"fair{k}Err{O}", f"{r['mean_err']:.3f}")
                macro(f"fair{k}Med{O}", f"{r['median_err']:.3f}")
                macro(f"fair{k}Rel{O}", f"{100 * r['rel_vs_ref']:+.1f}")
                macro(f"fair{k}Better{O}", str(r["better"]))
                macro(f"fair{k}P{O}", fmt_p(max(r["wilcoxon_p"], 1e-300)))
        drift = np.array(b.get("energy_above_45hz", b.get("energy_below_1hz", [0.0] * b["n_windows"])))
        for P, O in (("6", "Six"), ("12", "Twelve")):
            ref = np.array(EB["classical Adam-60"][P]["err"])
            fail = ref > 0.9
            macro(f"fairDriftFail{O}", f"{100 * drift[fail].mean():.0f}" if fail.any() else "--")
            macro(f"fairDriftOK{O}", f"{100 * drift[~fail].mean():.0f}")
            if "classical Adam-60, fc to 78 Hz" in EB:
                low = np.array(EB["classical Adam-60, fc to 78 Hz"][P]["err"])
                q = np.array(EB["QACT parity (hw-4, exact-f, backfit 1)"][P]["err"])
                from scipy.stats import wilcoxon
                macro(f"fairLowVsParityRel{O}", f"{100 * (low.mean() / q.mean() - 1):+.1f}")
                macro(f"fairLowVsParityBetter{O}", str(int((low < q).sum())))
                macro(f"fairLowVsParityP{O}", fmt_p(float(wilcoxon(low, q).pvalue)))
                lowbf = np.array(EB["classical Adam-60, fc to 78 Hz + backfit 1"][P]["err"])
                macro(f"fairLowBfVsParityRel{O}", f"{100 * (lowbf.mean() / q.mean() - 1):+.1f}")
                macro(f"fairLowBfVsParityBetter{O}", str(int((lowbf < q).sum())))
                macro(f"fairLowBfVsParityP{O}", fmt_p(float(wilcoxon(lowbf, q).pvalue)))
                macro(f"fairLowFail{O}", str(int((low > 0.9).sum())))


# ----------------------------------------------------------------------------- speed
def speed_macros():
    d = json.load(open(ROOT / "results" / "qact_vs_act_speed.json"))
    for r in d["rows"]:
        k = {"classical ACT, GPU": "speedClGPU", "classical ACT, CPU": "speedClCPU",
             "QACT exact simulation, GPU": "speedQSim", "QACT circuits in Aer (simulator)": "speedQAer",
             "QACT on IBM Heron, optimised": "speedHeron", "   same, repetition delay 0": "speedHeronZero",
             "QACT on IBM Heron, original design": "speedHeronOrig"}[r["engine"]]
        s = r["seconds_per_window"]
        macro(k, f"{s * 1e3:.2f} ms" if s < 1 else (f"{s:.0f} s" if s < 3600 else f"{s / 3600:.1f} h"))
        if r["batched_seconds_per_window"]:
            macro(k + "Batched", f"{r['batched_seconds_per_window'] * 1e3:.3f} ms")
        if r["recon_err"] is not None:
            macro(k + "Err", f"{r['recon_err']:.2f}")
    macro("speedCircuitsPerWindow", f"{d['circuits_per_window']:,.0f}")
    s = json.load(open(ROOT / "results" / "qact_hw_speedups.json"))
    macro("speedAdaptiveShots", f"{s['adaptive']['reference_shots_per_window'] / s['adaptive']['shots_per_window']:.1f}")
    macro("speedAllThree", f"{s['estimates']['+ adaptive shots (all three)']:.1f} s")
    macro("speedPack", str(max(p["k"] for p in s["packing"] if p["ok"])))


# ----------------------------------------------------------------------------- coherent residual
def fig_coherent():
    d = json.load(open(RES / "coherent_residual.json"))
    macro("cohNumpyDiff", f"{d['numpy_max_diff']:.0e}")
    macro("cohQiskitDiff", f"{d['qiskit_max_diff']:.0e}")
    macro("cohSuccess", f"{d['numpy_success']:.3f}")
    macro("cohReflCZ", str(d["reflection_two_qubit_gates"]))
    macro("cohK", str(d["K"])); macro("cohN", str(d["N"]))
    tab = d.get("set_c_success_probability_err2")
    if tab:
        fig, ax = plt.subplots(figsize=(3.2, 2.0))
        orders = sorted(int(o) for o in tab)
        for v, lab, col in (("F0", "symmetric atom", GREY), ("v14full", "skew atom, full grid", BLUE)):
            y = [tab[str(o)][v] for o in orders]
            ax.plot(orders, y, marker="o", color=col, label=lab)
            ax.plot(orders, [1 / math.sqrt(t) for t in y], ls="--", color=col, lw=0.8)
        ax.set_xlabel("atoms $K$"); ax.set_ylabel("success prob. (solid)\nrepetitions with AA (dashed)")
        ax.set_ylim(0, 3.2); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "fig_coherent.pdf", bbox_inches="tight"); plt.close(fig)
        for o in orders:
            macro("cohSetC" + {8: "Eight", 16: "Sixteen", 24: "TwentyFour", 32: "ThirtyTwo"}[o], f"{tab[str(o)]['F0']:.2f}")


def main():
    fig_hardware()
    speed_macros()
    fig_fair()
    fig_coherent()
    fig_device()
    with open(PAPER / "numbers.tex", "w", encoding="utf-8") as fh:
        fh.write("% generated by exp/make_figures.py -- do not edit\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(M)} macros and figures to {FIG}")


if __name__ == "__main__":
    main()

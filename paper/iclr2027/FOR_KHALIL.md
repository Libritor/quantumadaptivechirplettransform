# For Khalil: what is here, what to run on the RTX 5090, what to push

Branch `iclr2027` of this repository carries an ICLR 2027 manuscript built on your study
(`paper/iclr2027/main.tex`, PDF at `paper/iclr2027/main.pdf`) plus the experiments that were
missing from it. Everything below regenerates from result files; nothing in the paper is typed
in by hand except the tables copied from your README.

## 1. Get the branch and put it on your GitLab

```bash
cd quantumadaptivechirplettransform
git remote add vicol https://github.com/Libritor/quantumadaptivechirplettransform.git
git fetch vicol
git checkout -b iclr2027 vicol/iclr2027
git push origin iclr2027        # -> https://gitlab.com/Kchagh/quantumadaptivechirplettransform/-/tree/iclr2027
```

## 2. What was added (all under `paper/iclr2027/exp/`)

| script | what it does | result file |
|---|---|---|
| `fair_control_refine.py` | the control your denoising section named but did not run: the classical `act_gpu` engine given QACT's 3-point coordinate refiner (and its 0.5 Hz grid), on 320 public EEGMMIDB windows, paired Wilcoxon, orders 6 and 12 | `results/fair_control_refine.json` |
| `fair_control_backfit.py` | backfit given to the classical engine, and the seed grid extended to 78 Hz: QACT parity's -6.3% mean edge at order 12 is 14 windows dominated by 60 Hz mains that the 1-40 Hz seed grid cannot seed; with the extended grid the classical engine is -12.6% relative to QACT parity (289/320 better) | `results/fair_control_backfit.json` |
| `coherent_residual.py` | the "coherent residual" you flagged as the open piece: a one-ancilla reflection (I+R)/2 applied K times equals the classical MP residual exactly, success probability = residual energy fraction; verified numpy 1e-16, qiskit 8e-10 | `results/coherent_residual.json` |
| `qpu_selection.py`, `qpu_collect.py` | the windowed selection circuit on a REAL IBM Heron (`ibm_marrakesh`), 26 circuits, 4,000 shots, certified against the exact FFT distribution; DD is refused for dynamic circuits, so the dynamic job runs with default options and the unitary-QFT job with DD + twirling | `results/qpu_selection_marrakesh1.json` (lands when the queue releases the jobs) |
| `make_figures.py`, `build_paper.py` | figures + `numbers.tex` from the JSONs; pdflatex/bibtex build | `figures/`, `numbers.tex` |

Findings so far, in one line each: at equal refinement rule the classical engine beats QACT
(coord-4: -3.6% vs QACT hw-4 at order 12, 264/320); the 0.5 Hz grid changes nothing on
reconstruction; QACT parity's mean edge is the seed band (mains), not the transform.

## 3. What to run on the 5090 (only you can: it needs EEG-Memristor-ACT and the EEGMAT data)

The one remaining unexplained number in the paper is yours: `QACT parity hw` 1.213 beating
`classical ACT fine-f` 1.257 on denoising (p = 0.0007). Two classical arms decide it, and both
are ten-line changes to `compare_denoise.py` / `qbe/denoise.py`:

1. **`classical fine-f + hw refiner`**: the classical engine with the 0.5 Hz grid and the
   refinement replaced by the 3-point coordinate search, 4 sweeps. The function is
   `refine_coord_batch` in `paper/iclr2027/exp/fair_control_refine.py`; drop it into
   `act_gpu.decompose_batch` in place of `refine_batch` (a `refine=` argument is the cleanest).
2. **`classical fine-f, seed grid to 78 Hz`**: EEGMAT is 50 Hz mains and the injected artifacts
   include mains, but the classical seed grid stops at 40 Hz with an in-band refiner, so it
   cannot seed a mains atom; QACT's exact-f update can. Extend the `fc` axis of
   `gpu_dictionary_grid` to 78 Hz (1 Hz steps above 40 are enough) and rerun. Your artifact
   rule already flags mains-like atoms, so this is exactly the mechanism that would move the
   delta-band and time-fidelity terms.

Protocol unchanged: thresholds tuned on the first 18 subjects and frozen, scored on the 18
held out, paired Wilcoxon against `QACT parity hw` (1.213) and against `classical fine-f`
(1.257), alpha = 0.05/4. Please write `results/denoise_controls_iclr.json` with the per-subject
scores and the paired tests, and push it on `iclr2027`. It becomes two rows of Table 2 and one
sentence in Section 5.2; folding it in takes fifteen minutes on this end.

Optional sanity pass (fast on your GPU, needs only the MNE download of about 50 MB):

```bash
python paper/iclr2027/exp/fair_control_refine.py --subjects 20 --orders 6,12
python paper/iclr2027/exp/fair_control_backfit.py --subjects 20 --orders 6,12
python paper/iclr2027/exp/make_figures.py && python paper/iclr2027/exp/build_paper.py
```

Expected wall times: the two fair-control scripts took about 8 and 5 minutes here on an RTX
4060 Laptop (8 GB); on the 5090 expect about 2 minutes each. The denoising arms above should be
in the 10-20 minute range on the 5090 for both arms together (your parity QACT arm is the slow
one; the classical arms are a few ms per window batched).

## 4. Things only you can answer

- Author order and affiliations for the camera-ready (the submission is anonymous).
- Whether you are comfortable with how the abstract and introduction describe the study; the
  paper leans on your pre-registered controls as its main argument.
- The ICLR 2027 full-paper deadline is September 25, 2026, 11:59 PM AoE, and the abstract had
  to be registered by September 18; if that did not happen, the same paper goes to ICML 2027
  (January 22, 2027) with your denoising arms folded in and, ideally, the CHB-MIT
  quantum-vs-classical comparison in the all-features regime, which you noted has not been run.

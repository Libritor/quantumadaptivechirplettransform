# Search for a quantum ACT formulation that beats classical

A loop: run an iteration, learn from it (and from the literature), design the
next, repeat — until a formulation beats classical in speed or accuracy, or the
credible hypotheses run out. This file is the loop's memory.

## Success criterion (fixed before iteration 1; not to be relaxed later)

A formulation **beats classical** only if, on a pre-registered benchmark, it beats
the strongest classical baseline — *including a classical implementation of the
same idea* — in one of:

- **speed**: less wall-clock time per EEG window at equal or better accuracy,
  with quantum time estimated from real device calibrations (IBM Heron) or a
  fault-tolerant resource estimate (Microsoft qdk), classical time measured on
  the RTX 5090 machine;
- **accuracy**: better accuracy at equal or lower wall-clock time.

Why the classical port has to be in the comparison: every quantum circuit in
this project is small enough to simulate, so whatever it computes, a classical
program computes too. An accuracy gain that its own classical port matches is an
algorithmic finding, not a quantum one (logged as such, not counted as success).

Stop rule: stop when every queued hypothesis is tested and a fresh literature
search yields no new credible candidate.

## Already ruled out (iterations before this loop, all in README.md)

| formulation | result |
|---|---|
| QSVC / VQC / projected kernels / re-uploading / hybrid classifiers on ACT features | parity or worse |
| QLSTM, quantum reservoirs | reservoir entanglement +0.024 vs its own control, but classical ESN wins |
| QACT (QFT-based dictionary search), sampled or argmax | parity after full functional parity; apparent win was the QFT's finer frequency grid, matched by a classical 0.5 Hz grid |
| cubic / skew atom families, proposal selection | no feature gain |
| 3-point refinement | better than parameter shift, but a classical optimiser |
| QACT circuits on IBM Heron (optimised + 3 published speed-ups) | 8.0 s/window vs 55 ms CPU, 0.7 ms GPU |
| Hybrid ACT q6-q9 | q6-q8 parity, q9 broken by noise; all slower |
| error-corrected q8/q9 | quality restored; 12 min - 111 h per window |
| q8 vs classical, powered (60 windows) | no difference; noisy selection = classical random mix |

## Hypothesis queue

Each entry: the idea, why it might win, and the classical control it must beat.

1. **Fault-tolerant coherent dictionary search** (Dürr–Høyer over the chirp x
   frequency register, amplitude estimation for scores). Asymptotically √M
   instead of M dictionary evaluations. Control: FFT-based classical matching
   pursuit, measured. Question: at what dictionary size, if any, does the
   crossover happen, with and without QRAM?
2. **Joint support selection as optimisation** (QUBO over candidate atoms,
   solved by QAOA; cf. compressed sensing with QAOA, Phys. Rev. A 110:062410,
   2024). Greedy matching pursuit is myopic; optimising the support jointly can
   fit better. Controls: greedy OMP, and classical solvers of the same QUBO
   (exhaustive at small size, simulated annealing).
3. **Research round**: literature since 2024 on quantum speed-ups for
   time-frequency analysis, sparse coding or matching pursuit on classical data.

## Iterations

(newest last)

### Iteration 1 — fault-tolerant coherent dictionary search (`qsearch_01_ft_crossover.py`)

Hypothesis 1. Quantum cost per selection taken as sqrt(M) x (1/eps) x t_U with
every constant tilted toward quantum (Durr–Høyer constant 1 instead of the proven
22.5; no U-dagger or reflections; per-call error budget fixed at 1e-4 however many
calls). t_U from Microsoft's resource estimator for one oracle call; classical cost
per (envelope, chirp) pair measured on this machine.

| oracle | qubits | t_U | crossover M* vs 1-window CPU (1.9 us/pair) | vs batched GPU (28 ns/pair) |
|---|---|---|---|---|
| explicit loading | GATE_NS_E4 | 42.9 ms | 3.2e15 (eps 0.1) | 1.5e19 |
| QRAM for free | GATE_NS_E4 | 3.0 ms | **1.5e13** (eps 0.1) | 7.3e16 |
| QRAM for free | GATE_NS_E3 | 6.3 ms | 6.9e13 (eps 0.1) | 3.3e17 |

Current dictionary: 22,784 atoms. Every chirplet a 512-sample window can resolve
(512 positions x 512 frequencies x 100 widths x 1,000 chirp rates): 2.6e10. The
most favourable crossover is 600x beyond that.

The only escape is longer signals: with free QRAM, t_U grows polylogarithmically
in N while the resolvable dictionary grows ~N^2, so a crossover exists in principle
for signals of roughly 10^5-10^7+ samples (tens of minutes to days of EEG treated
as ONE signal) — before restoring the proven constants (x ~2,000 on M*), QRAM's
real cost (Jaques & Rattew 2025), or MPTK-style local updates, which let
classical matching pursuit avoid rescoring the whole dictionary each iteration.
Windowed EEG has no reason to be decomposed as one signal of that length.

**Verdict: no. Speed via fault-tolerant search is closed for this problem.** This
also closes speed for every variant that still loads recorded EEG into circuits,
since state loading and per-shot overheads were already the binding costs.

Remaining route to a win must be accuracy at equal time, and any simulable circuit
is matched by its own classical port, so an accuracy win needs a formulation whose
classical port is too costly to run in the same time. Next: research round
(hypothesis 3) before hypothesis 2, because a QAOA support-selection instance
small enough to simulate is also small enough to solve exactly by classical
exhaustive search.

### Iteration 2 — research round (hypothesis 3), and what it means for hypothesis 2

Searched for 2024–2026 claims of quantum advantage in time-frequency analysis,
sparse coding, matching pursuit or compressed sensing on classical data.

- **Time-frequency analysis:** only hybrid quantum-classical *classifiers* of
  time-frequency images (e.g. radar signal recognition, 2025), with no advantage
  claim over classical methods. Nothing on the transform itself.
- **Chevalier, Roga, Takeoka, "Compressed sensing enhanced by a quantum
  approximate optimization algorithm", Phys. Rev. A 110:062410 (2024),
  arXiv:2403.17399.** QAOA for matching pursuit's support-detection step,
  simulated only, 6-qubit problems (64-dim spectra, sparsity 2-3). Its headline
  (53% vs 36% reconstruction success) compares QAOA given *richer* measurements
  (random quadruplet patterns) against a classical solver given simpler
  (nearest-neighbour) ones. With identical measurements the authors state QAOA
  "can only perform as well or worse than the classical one"; brute force solves
  their example. It also presupposes *designed* measurement patterns that turn
  support detection into an Ising problem, whereas in ACT that step is solved
  exactly by an FFT.
- **Hahn & Romero, "Computational phase transitions in binary compressed sensing:
  quantum annealing inside the relaxation gap", arXiv:2606.00806 (2026).** D-Wave
  recovers some sparse *binary* signals that nine classical solvers miss (7% vs
  0% at n=32, k=5), but the authors call it numerically fragile (2 of 30 trials),
  report that it "collapses at n=64" where AMP dominates everywhere, and did not
  try Gurobi, SCIP or CPLEX. ACT coefficients are continuous, not binary.
- Re-confirmed: the published exponential advantages for signals (Huang et al.,
  Science 2022; Kannan et al., arXiv:2608.13521, 2026) need the signal to reach a
  qubit directly (quantum sensing), not recorded arrays.

**Hypothesis 2 (QAOA joint support selection) is refuted by the criterion
itself, so it was not run.** Any QUBO small enough to simulate QAOA on is small
enough to solve *exactly* by classical exhaustive search or branch-and-bound. The
classical port therefore matches or beats QAOA's accuracy, in less time; and the
literature above shows the claimed QAOA/annealing gains either come from giving
the quantum solver more information, or vanish as problems grow. Whether *joint*
support selection beats greedy OMP for ACT is still an interesting question, but
a classical one.

## Outcome

The queue is empty, and the research round produced no new candidate that could
meet the criterion. Per the stop rule, the search ends here with a negative
result: **no quantum ACT formulation found that beats classical in speed or
accuracy.** Why, in one paragraph:

- *Speed* is bounded by loading recorded EEG into circuits and by per-shot
  overheads. Even fault-tolerant √M search with free QRAM only wins for
  dictionaries ~600x larger than every chirplet a window can resolve
  (iteration 1).
- *Accuracy* has no quantum-exclusive route at simulable sizes: every circuit
  that can be tested is matched by its classical port (the powered q8 test, the
  classical noise-mix controls), and published "wins" reduce to extra information,
  fragile small-n effects, or quantum data.

What would reopen the search: signals that reach a qubit directly (quantum
sensors), fault-tolerant hardware whose logical operations are orders of
magnitude faster than today's estimates (iteration 1's t_U), or a new paper
demonstrating a robust quantum advantage for sparse approximation on classical
data.

---

## Search reopened (user request): broader space, revised criterion

The first search only looked *inside* the transform (how atoms are found and
fitted) and used a criterion under which no simulable circuit could ever win on
accuracy (it had to beat a classical simulation of itself). Reopened with:

**Scope:** any quantum formulation built on ACT — including quantum models that
*consume* chirplet atoms (the seizure-detection task, where chirplet features are
worth +9 to +13 points over amplitude features), new architectures, and niche
literature.

**Criterion (revised; applies from here on, not retroactively):**

- **Quantum win:** on a pre-registered benchmark with patient-wise or
  subject-wise folds, beats the *best tuned classical models* on the same inputs
  (not just the ones previously used), and the quantum model's own classical
  simulation cost grows exponentially with scale. At the sizes that can be tested
  here this is evidence, not proof.
- **Quantum-inspired win:** beats the best tuned classical models but is
  efficiently simulable classically; logged as a new classical method.
- **Speed:** unchanged.
- Classical baselines must be strong (tuned gradient boosting, RBF-SVM,
  regularised linear, small neural nets), because most published QML "wins"
  vanish against tuned classical baselines.

### Iteration 3 — research round in the spaces skipped so far

- **Quantum reservoir computing (2026):** Hou et al., *Phys. Rev. Lett.* 136:120602
  (9 NMR spins beat echo-state networks with thousands of nodes on weather
  forecasting; arXiv:2508.12383); Li, Mukhopadhyay, Bayat, Habibnia, *Phys. Rev.
  Research* 8:023028 (10-qubit fully connected transverse-field Ising reservoir
  with separate input and memory qubits beats econometric and ML models on
  volatility). But this project already tested a closely matching design (4 x 8
  qubits, transverse-field Ising, input qubits replaced each step) on EEG: it beat
  the ESN on NARMA-10 and memory capacity, then lost to it on EEG for 22/23 people
  — the same benchmark-yes, EEG-no pattern Wolff et al. (arXiv:2608.00139) report.
  Deprioritised.
- **Gaussian boson sampling graph kernels** (Schuld et al., *Phys. Rev. A*
  101:032314, 2020) beat classical graph kernels on standard benchmarks, and GBS is
  #P-hard to simulate — but the kernels use coarse-grained sample statistics, and
  sufficient conditions exist for estimating those efficiently classically. Likely
  quantum-inspired at best; queued, with that to be checked first.
- **Quantum convolutional networks:** shown to be effectively classically
  simulable (arXiv:2408.12739) — quantum-inspired at best. Headline image wins use
  weak baselines.
- **QML for EEG/seizures (2025-2026):** hybrid networks and "quantum-inspired"
  feature fusion, none compared against strong tuned classical baselines.
- **Benchmark literature:** Bowles, Ahmed, Schuld, arXiv:2403.07059 (2024) — across
  160 datasets, out-of-the-box classical models beat 12 QML models, and removing
  entanglement often did not hurt. Sets the bar: strong classical baselines, and an
  entanglement-off control for every quantum model.
- **QSVT signal denoising** (arXiv:2312.15411): speed limited by data loading, and
  low-rank QSVT is dequantised (arXiv:2303.01492). Deprioritised.

New queue (all on CHB-MIT patient-wise seizure detection, the task where chirplet
features already help):

4. **Quantum atom-set kernel (new, ACT-specific):** each channel-epoch is a set of
   chirplet atoms; embed atoms with an IQP feature map, form the energy-weighted
   mixture rho, kernel Tr(rho rho'). Controls: the same set kernel with an RBF
   instead of the quantum map; the quantum map with entanglement off; tuned
   HistGradientBoosting / RBF-SVM / logistic on the flat features.
5. GBS graph kernel on channel graphs (after checking classical estimability).
6. Trainable quantum kernel alignment (not tried before; fixed feature maps were).

### Iteration 4 — quantum atom-set kernel (`qsearch_04_atomset_kernel.py`, `qsearch_04b_bandwidth.py`)

CHB-MIT seizure detection, 10,856 epochs, 23 patients, ACT information only, every
model tuned identically on its outer training fold. Pre-registered rule: a quantum
win needs Q significantly better than all four classical models.

| model | per-patient accuracy |
|---|---|
| Q  quantum atom-set kernel (IQP/ZZ map, 8 qubits, full entanglement) | 62.57% |
| Q with bandwidth grid extended down to 0.03 (4b) | 62.31% |
| Qp same map, entanglement off | 64.49% |
| R  classical RBF atom-set kernel | 62.68% |
| F1 logistic regression, flat ACT features | 62.66% |
| **F2 RBF-SVM, flat ACT features** | **67.36%** |
| F3 gradient boosting, flat ACT features | 65.88% |

Q vs F2: −4.79 points, better for 4/23 patients, p = 0.0015 (worse). Q vs R, F1: no
difference. Q vs Qp: −1.92 (entanglement hurts; p = 0.023). The bandwidth fairness
check (the first run's tuner sat on the grid's lower edge) moved the choice inside
the grid and changed nothing.

**Verdict: no.** Two lessons for the next design:
1. The set representation itself is the weak link — even the *classical* set kernel
   loses to flat features with an RBF-SVM. A quantum model on a worse representation
   cannot win.
2. Entanglement hurts here, as in Bowles, Ahmed & Schuld (2024): the entangled map
   needed a gentler encoding (bandwidth 0.25 vs 1-2 without entanglement) and still
   lost to its own product-state version — the concentration problem.

Next: a targeted research round — *where* have quantum models beaten **tuned**
classical baselines on real data, and what conditions did those wins share?

### Iteration 5 — research round 2 and its two leads (`qsearch_05_allfeature_lowdata.py`, `qsearch_05c_z8_bandwidth.py`)

Research round 2 asked where quantum models have beaten *tuned* classical baselines on
real data. The most rigorous answer: arXiv:2604.18837 (2026), 970 experiments on 9
tabular datasets including IBM hardware — none of 29 quantum-vs-classical
comparisons significant. Two leads worth testing:

**5a — all-feature quantum kernel** (after Delilbasic et al., arXiv:2605.17587,
2026, hyperspectral data: 78.0 +/- 6.2% vs RBF 72.0 +/- 5.0%). One qubit per
feature, no feature selection; with product states the kernel is exactly
prod cos^2(s dz), which tends to an RBF kernel as s -> 0. (That paper simulates its
entangled versions with tensor networks at O(n^2) cost — efficiently classical.)
CHB-MIT, 270 features: **74.81%** vs tuned RBF-SVM 76.82% (-2.02, p = 0.091) and
gradient boosting 75.46% (-0.65, p = 0.32). No gain; it behaves like the RBF kernel
it approximates.

**5b — small-data regime** (Caro et al., Nat. Commun. 2022). Learning curves at 20-200
training epochs, every kernel on the same label-free bandwidth rule. The first run
showed the entangled 8-qubit kernel losing by 6-10 points everywhere — **but that
was my grid, not the model**: its kernel had concentrated (median off-diagonal value
0.05-0.09). With gentler bandwidths (5c, same training draws):

| n | entangled Z8 | vs no-entanglement control | vs RBF-SVM | vs logistic |
|---|---|---|---|---|
| 20 | 65.39% | +1.16 (15/23, p = 0.035) | -0.12 | -1.08 |
| 50 | 69.41% | +0.58 (16/23, p = 0.13) | -0.57 | +0.17 |
| 100 | 69.94% | +0.38 | -0.96 | +0.29 |
| 200 | 70.59% | -0.47 | -2.76 | -0.93 |

**Verdict: no win, parity.** A hint that entanglement helps at the smallest n (+1.16
over its own control for 15/23 patients) is not significant after correction and
never turns into beating RBF-SVM or logistic regression.

Lessons so far (iterations 4-5): on this task's features every quantum kernel lands
at parity or below once properly tuned, exactly as the tabular-data literature
predicts; concentration must be checked for every entangled map (it silently
produced a false 8-point loss here); and the quantum-hard kernels (IQP) never beat
their own product-state versions by a significant margin.

Next: Gaussian boson sampling graph kernels on EEG channel graphs — the one
published quantum-hard feature map with a benchmark win — after checking whether
its features are classically estimable.

### Iteration 6 — research round 3: theory, graph kernels, covariant kernels

- **Gaussian boson sampling graph kernels are not quantum-hard for EEG graphs.**
  Oh, Fefferman, Jiang, Quesada, *PRX Quantum* 5:020341 (2024): with a
  non-negative adjacency matrix (any channel-coupling graph) GBS needs no quantum
  interference; a quantum-inspired classical sampler reproduces its graph
  applications and the GBS advantage "is not significant in general". Dropped (and
  classical cross-channel features already showed no benefit here).
- **Provable quantum learning advantages** on classical data need labels built on a
  quantum-hard function (Liu, Arunachalam, Temme 2021: discrete logarithm; later
  general quantum-computational constructions) or quantum data/observables; with
  equivalent classical data access they evaporate. Nothing known gives seizure
  labels on EEG either structure.
- **Covariant quantum kernels** (Glick et al., *Nature Physics* 20:479, 2024) are the
  most ACT-native architecture left — chirplets are an orbit of the time-shift x
  frequency-shift x chirp group acting on a Gaussian, which QACT's gates already
  represent. But the largest real-world test, Agliardi et al., *npj Quantum
  Information* (2025), arXiv:2412.07915 (up to 156 qubits on IBM hardware), reaches
  accuracy "comparable to classical SVCs" / "in line with classical" (80%) — parity.
  With a Gaussian fiducial the chirplet-group kernel also has a classical closed
  form.

## Outcome of the reopened search

Stopped again under the stop rule, with a much broader evidence base: five
iterations run here (fault-tolerant search; quantum atom-set kernel; all-feature
kernel; small-data learning curves; bandwidth fairness checks) and three research
rounds. **No formulation beat classical.** Every properly tuned quantum model landed
at parity or below; the only positive signal (entanglement +1.16 points over its
own control at n = 20, 15/23 patients) is not significant after correction and never
beats classical models.

Why stop instead of continuing indefinitely: every remaining candidate has published
evidence of parity on real classical data, and the theory explains it. An
open-ended search that keeps testing near-parity variants at p < 0.05 will
eventually produce a false "win" by chance; a real finding needs a candidate with a
reason to win, tested with family-wise correction across the whole search.

What would genuinely reopen it:
- data with quantum structure — EEG/MEG measured by quantum sensors coupled to a
  quantum processor (the setting of Huang et al. 2022 and Kannan et al. 2026);
- a learning target on EEG with known quantum-hard structure (none known);
- fault-tolerant hardware orders of magnitude faster than iteration 1's estimates.

## Iteration 7 — data with quantum structure: quantum sensing of real brain fields (`quantum_sensing_opm.py`)

The first item on the "what would reopen it" list. Recorded data, even from a
quantum sensor, reaches the computer as classical numbers, so the only place a
quantum advantage can live is *in the sensing*: probe spins, possibly entangled,
that pick up the field before they are measured. This asks whether that would help
detect the chirplet atoms of a real evoked brain response.

**Data (downloaded with permission):** the MNE OPM sample (somatosensory evoked
fields from 9 QuSpin optically pumped magnetometers, 201 median-nerve stimuli, plus an
empty-room recording), and the raw data of Zhang et al., *Nat. Commun.* (2025),
doi:10.1038/s41467-025-66828-z (Zenodo 17614133): entanglement-enhanced quantum
lock-in detection with two trapped ions, GHZ vs product states.

**Method (pre-registered in the script's docstring):** decompose the evoked field
(channel MEG1231, peak 1269 fT) into 6 chirplet atoms with QACT (residual 0.043).
Each atom's waveform becomes a matched lock-in reference. Project single-trial
residuals and empty-room segments onto it to split per-trial noise into
sensor+environment and brain background. Then model N probe spins (Rb-87) with
dephasing time T2 and count the trials needed to detect each atom at 5 sigma, using
product states (standard quantum limit) and optimal GHZ blocks. An Aer
density-matrix simulation of the lock-in circuit must reproduce the phase-noise
formulas; it does, with a maximum relative deviation of 4e-6.

**Two corrections made before the result was read.** Neither changed the direction
of the result:
- The first sensing model held one coherent interrogation across the whole atom
  window (≈0.2 s). That is unphysical when T2 ≪ 0.2 s. A real sensor repeats shorter
  interrogations, so the model now optimises the interrogation length and the GHZ
  block size. This is Huelga et al. (1997): under uncorrelated dephasing,
  entanglement gains only a constant factor, and only when the window is shorter than
  T2.
- The first anchor estimator averaged the 3 *best* noise/slope ratios. That
  selects noise minima, and it reported gains of 1.85/1.71, above the ideal
  √2 = 1.414. It now takes the 4 steepest points, chosen by slope alone, with a
  bootstrap over repetitions.

**Results**

| check | result |
|---|---|
| anchor (real ion-trap data), n = 20 pulses | entangled/product gain 1.51 (95% CI 1.40–1.81), consistent with √2 |
| anchor, n = 30 pulses | 2.31 (CI 2.06–2.53), above √2. The authors' observable is a GHZ-parity readout that is not optimal for product states, so read this as "the advantage is real in hardware", not as its size |
| OPM: trials to detect each atom, real sensor | 4.7 / 24 / 94 / 154 / 103 (atoms 0, 2–5) |
| same with a *perfect, noiseless* sensor | 3.8 / 19 / 82 / 115 / 45. Removing **all** sensor noise saves 1.1–2.3x, which is the ceiling for any better sensor, quantum or not |
| atom 1 (near-DC, 241 ms) | empty-room noise exceeds in-trial noise, so the noise split is unreliable. Excluded |

Entanglement gain in trials (product / GHZ, same N, T and T2):

| probe spins N | T2 = 1–100 ms | T2 = 1 s | no dephasing (ideal) |
|---|---|---|---|
| 1e2 – 1e6 | 1.00x | 1.2–1.6x | 50–5,500x |
| 1e9 | 1.00x | 1.0–1.2x | 1.1–6.5x |
| 1e11 – 1e13 (a real vapour cell) | 1.00x | 1.00x | ≤ 1.06x |

**Verdict.** The pre-registered claim (GHZ needs ≥ 10% fewer trials than product
states at the same N, T and T2) holds only for N ≤ 3e9 spins with T2 ≥ 1 s.
That regime is useless in practice. The small entangled sensor that "wins" there
still needs hundreds to millions of trials, while an ordinary unentangled vapour cell
(≈1e12 atoms, ms coherence) already needs 4–115, within 1.1–2.3x of a perfect
sensor. At realistic N the limit is the brain's own background activity, not the
probe's quantum noise, and no sensing strategy can remove that. So quantum sensing
does not help chirplet detection of brain fields either. This matches the
literature: OPM-MEG is limited by environmental and physiological noise, not by the
standard quantum limit.

What the data do show: entanglement advantage is real in hardware (the
ion-trap anchor). It pays off when the sensor is **small and the target is not
buried in biological noise**, for example single-spin or NV-centre sensing of
microscopic sources such as single neurons or axons at micrometre range. That is a
different measurement from EEG/MEG, and no public dataset of it exists that this
project could use.

# A quantum Adaptive Chirplet Transform (QACT) — draft

## Why ACT is unusually well suited to a quantum circuit

A chirplet atom is a Gaussian-windowed linear chirp. In our sample-domain
convention, with `t ∈ {0..N-1}`, `N = 2^n`:

```
ψ_θ(t) = A · exp( −(t−tc)² / (2Δt²) ) · exp( i·2π/N · [ c·(t−tc)² + fc·(t−tc) ] )
         └──── real envelope ────┘   └──────── pure phase ────────┘
θ = (tc, fc, Δt, c)
```

Encode the time index in `n = log₂N` qubits, `|t⟩ = |b_{n−1}…b_0⟩`, `t = Σ_k b_k 2^k`.
Then each factor of the atom is a natural circuit primitive:

**1. Linear phase (centre frequency) — n gates, exact.**
`exp(i·2π f t / N)` is diagonal and separable:
`= ⊗_k exp(i·2π f 2^k b_k / N)` → one phase gate `P(2π f 2^k / N)` per qubit.

**2. Quadratic phase (chirp) — n + n(n−1)/2 gates, exact.**
Because `b_k² = b_k`,
```
t² = (Σ_k b_k 2^k)² = Σ_k b_k 2^{2k} + 2 Σ_{j<k} b_j b_k 2^{j+k}
```
so `exp(i·2π c t²/N)` is a product of single-qubit phases `P(2π c 2^{2k}/N)` and
**two-qubit** controlled phases `CP(2π c 2^{j+k+1}/N)`. A chirp is therefore
2-local in the binary encoding — no approximation, no Trotterisation.

**3. Time centre costs nothing extra in the phase.** Expanding
`c(t−tc)² + fc(t−tc)` gives `c·t² + (fc − 2c·tc)·t + const`, so a shifted chirp
is the *same* quadratic phase plus a linear phase at an effective frequency
`f_eff = fc − 2c·tc` and a global phase. `tc` only enters through the envelope.

**4. Envelope = diagonal filter.** The Gaussian is not unitary. Standard
block-encoding: one ancilla, controlled `R_y(2·arcsin(env(t)))`, post-select.
Success probability `Σ_t |env(t)·x(t)|²` is the measurement overhead. (On
hardware one could instead prepare the atom directly: a Gaussian is log-concave,
so Grover–Rudolph state preparation applies.)

**5. The whole frequency axis of the dictionary comes free from one QFT.**
Let `|x⟩` be the amplitude-encoded residual. Apply the envelope filter for
`(tc, Δt)`, then the conjugate quadratic phase for `c`, then the QFT. The
amplitude on frequency `f` is exactly the correlation
`⟨ψ_(tc,f,Δt,c) | x⟩` (up to a known phase). **Measuring the register samples
atoms with probability ∝ |⟨ψ|x⟩|²** — precisely the quantity matching pursuit
maximises. The selection step is a native quantum sampler.

## The algorithm

Per matching-pursuit iteration, on residual `r`:

```
for each (tc, Δt, c) in the dictionary grid:          # M' triples
    |r⟩            ← amplitude-encode residual            (n qubits)
    envelope filter  for (tc, Δt)                         (1 ancilla, post-select)
    conjugate chirp  for c                                (n + n(n−1)/2 gates)
    QFT                                                   (O(n²) gates)
    measure                                               → f ~ |⟨ψ_θ|r⟩|²
pick the most-sampled θ           (stochastic argmax; → classical argmax as shots→∞)
estimate its coefficient          (amplitude estimation, or classically)
subtract, repeat                                          # P atoms
```

Two ways to handle the remaining `M'` axes:

* **Enumerate** them (one circuit each) — what the reference implementation does.
* **Superpose** them: put `(tc, Δt, c)` in index registers and make the envelope
  and chirp *controlled* on those registers. Then Dürr–Høyer quantum maximum
  finding over the index register with an amplitude-estimation oracle finds the
  best atom in `O(√M')` evaluations instead of `O(M')`.

## Complexity, stated honestly

| step | classical | QACT (enumerated) | QACT (superposed + max-finding) |
|---|---|---|---|
| correlations over all `fc` for one `(tc,Δt,c)` | FFT, `O(N log N)` | 1 circuit, `O(n²)` gates | — |
| search over `(tc, Δt, c)` | `O(M')` FFTs | `O(M')` circuits | **`O(√M')`** (Dürr–Høyer) |
| coefficient of chosen atom | `O(N)` | `O(1/ε)` shots | `O(1/ε)` (amplitude estimation) |
| per iteration, ideal | `O(M' N log N)` | `O(M' · shots · n²)` | `O(√M' · ε⁻¹ · polylog N)` |

**The catch, and it is the same catch as everywhere else in QML.** The
`polylog N` column assumes the residual can be *prepared* in polylog time,
i.e. QRAM. Preparing an arbitrary N-sample residual costs `O(N)` without it —
and matching pursuit re-prepares a **new** residual every iteration, `P` times.
So the honest claim is:

* the per-atom dictionary search has a genuine quadratic quantum speedup
  (`√M'`), from Dürr–Høyer, which is a proven result;
* it is destroyed by state preparation unless residuals are available in QRAM,
  or unless the residual is itself produced coherently (subtracting a chirplet
  is a diagonal-plus-rank-1 update, so keeping it coherent is conceivable —
  this is the interesting open piece of this design);
* on a simulator, QACT costs about the same as classical ACT, because the
  simulator does the same FFT-scale arithmetic.

## What is genuinely different, and testable now

Even with no speedup, QACT is not merely classical ACT rewritten:

1. **Selection is stochastic** — atoms are *sampled* ∝ `|⟨ψ|x⟩|²` instead of
   argmax'd. Finite shots make this a randomised matching pursuit, which can
   escape the greedy trap that makes classical MP pick a "needle" atom (the
   degenerate atoms documented in `qbe/features.py`).
2. **Finite shot noise and post-selection loss are intrinsic**, and `shots`
   interpolates between quantum-sampled and classical-argmax behaviour.
3. Because the features come out in the same 13-per-channel format, QACT
   features can be dropped into every existing experiment and compared against
   classical ACT features on identical folds.

## Functional parity with the classical engine

QACT originally implemented only the *selection* half of ACT. Everything the
classical engine (`qbe/act_gpu.py`) does is now present, each expressed in the
form the quantum encoding makes natural rather than ported as classical code:

| classical function | QACT implementation |
|---|---|
| dictionary search `D.best` | one simulated QFT per (envelope, chirp) pair returns the whole frequency axis; the outcome is sampled (`select="sample"`) or argmax'd (the control) |
| least-squares energy criterion, normalised by atom norm | the measured histogram divided by the envelope filter's success probability, i.e. the distribution **conditioned on post-selection** — see below |
| off-grid refinement (Adam on tc, fc, log dt, c[, log dt_r]) | parameter-shift rule for the phase parameters (exact: E is degree-1 trigonometric in every gate angle), central differences for the envelope parameters |
| OMP joint refit (`_joint_refit`) | same least-squares refit, whose normal matrix is the Gram of atom overlaps ⟨ψᵢ\|ψⱼ⟩ — the swap-test observable, so no primitive beyond the ones the circuit already provides |
| two-width asymmetric envelope (ACTv9Asym) | `asym_ratios`, a fourth envelope grid axis; refinable via `refine_extra` |
| per-window stopping rule (`min_amp`, active mask, `E > 0`) | `min_amp`, same semantics |
| — | **exact frequency update**: E(f) is the periodogram, so one QFT gives the optimum directly instead of gradient-stepping toward it (`exact_f`) |
| — | **backfit**: cyclic re-refinement of each atom against the residual with its own contribution added back (`backfit_passes`) |
| — | **cubic chirp and skew envelope**, refined by the same shift rule (c₃'s gates are 3-local, so singles, pairs *and* triples enter its chain rule) |

**Measured outcome of parity.** On EEGMAT denoising against ICA-cleaned ground
truth, QACT at parity scores 1.272 vs the classical engine's 1.439 — but that
margin is entirely the frequency grid. Giving the classical seed grid the same
0.5 Hz spacing the QFT provides for free brings it to 1.257, and the paired test
against QACT then shows no difference (*p* = 0.15; *p* = 1.00 for the asymmetric
pair). The honest claim is **parity at a 2.25x smaller dictionary**, not an
advantage.

### The selection criterion was wrong, and the quantum reading fixes it

`_power` returned the raw `|⟨ψ|r⟩|²`. That is the **joint** probability that the
envelope filter succeeds *and* frequency k is read, so it systematically
over-ranks wide, high-norm envelopes. The classical engine never had this
problem because its criterion divides by the atom's norm.

The fix is not a normalisation bolted on for parity — it is the quantity an
experiment actually reports. A run keeps only post-selected shots, so the
observed distribution is **conditional** on the filter succeeding: divide by
`‖ψ‖²`, which `_power` already computes per envelope as the post-selection
success probability. That conditional distribution is simultaneously (a) what
the hardware measures, (b) what the refiner maximises, and (c) what the
classical least-squares criterion normalises by. All three now agree.

On planted chirplets this single change cut reconstruction error from 0.307 to
0.169, and on real EEG denoising it moved QACT from 70% worse than the classical
engine to statistically indistinguishable from it (README has the full table and
the control that shows the remaining difference is dictionary frequency
resolution, not the quantum formulation). `norm_select=False` restores the old behaviour, and with
`omp=False, backfit_passes=0, exact_f=False, norm_select=False` the engine
reproduces the pre-parity reference feature matrix **bitwise** (verified against
`results/X_chbmitqactonly_ref.npy`: 100% of 2.5M entries identical).

References: Dürr & Høyer, "A quantum algorithm for finding the minimum"
(quant-ph/9607014); Brassard, Høyer, Mosca & Tapp, "Quantum amplitude
amplification and estimation" (2002); Grover & Rudolph, "Creating superpositions
that correspond to efficiently integrable probability distributions"
(quant-ph/0208112); Mann & Haykin (1995) for the chirplet transform itself.
See also REFERENCES.md.

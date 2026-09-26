# The fidelity-gap law: when does a noisy quantum sampler still pick the right chirplet?

Theory note for the QACT paper (branch `iclr2027`, now aimed at ICML 2027). Scope: a
phenomenological law for hardware atom selection, its physics reading, and the
proposer-verifier design it motivates. Every number below comes from
`exp/qpu_law_check.py` run on the Sep 25 ibm_marrakesh data
(`results/qpu_selection_marrakesh1.json`, 26 circuits, 4000 shots each). Its output is
`results/qpu_law_check.json` with a console log in `results/qpu_law_check.log`. Nothing
here edits `main.tex`.

---

## 0. Setting and notation

One QACT slice fixes the atom centre `tc`, width `Δt = exp(logdt)` and chirp rate `c`,
and asks which frequency `k` gives the best atom. The windowed circuit loads
`s = env·x / ||env·x||` on an `m`-qubit window register (`D = 2^m` outcomes), multiplies
by the chirp `exp(-i 2π c t²/N)`, applies an inverse QFT and measures. The exact outcome
distribution is

    p_k = |<g_k, x>|² / Σ_k' |<g_k', x>|²,   g_k(t) = env(t) · exp(i 2π (k t + c t²)/N),

sampled on the window grid `k = k'·2^(n-m)`. Mirror bins (`k ≥ N/2`) and out-of-band
bins are **masked after sampling**, leaving `M'` admissible outcomes (13 of 64 at `m = 6`,
26 of 128 at `m = 7`, 104 of 512 at `m = 9`). Let `1` be the best admissible outcome
(`p1`) and `2` the admissible runner-up (`p2`). The gap is `Δ = p1 - p2`.

## 1. The law

### 1.1 Device model (phenomenological)

    q = R [ F p + (1 - F) u ],   u_k = 1/D,

- `F ∈ [0, 1]` is a global depolarizing fidelity. The register outcomes see the white
  floor `(1-F)/D` whether or not they are later masked.
- `R` is a readout channel. The plain law uses `R = I`; §1.5 adds per-qubit flips.
- `S` shots give counts `N ~ Multinomial(S, q)`. The device picks the admissible argmax.

If we condition on admissible shots only, the floor per admissible outcome becomes
`(1-F)/D` divided by the admissible mass `F P_band + (1-F) M'/D`. So "a uniform floor over
the `M'` unmasked outcomes" is the same model. The ordering is unchanged and the effective
shot count shrinks to `S·(F P_band + (1-F) M'/D)`. Below we keep the unconditioned `q`
with `S` register shots.

**Estimating F.** Under this model, `Σ_k q_k p_k = F Σ p² + (1-F)/D`. So the linear
cross-entropy estimator

    F_xeb = (D Σ_k q_k p_k - 1) / (D Σ_k p_k² - 1)

is unbiased for any `p`, not only Porter-Thomas ones. This is the linear XEB of Boixo et
al. (2018) and Arute et al. (2019) with the normalisation made exact for a known,
non-random target. The target is classically cheap here, so it can be evaluated on every
circuit.

### 1.2 Pairwise condition (best atom vs runner-up)

The uniform floor cancels exactly in the mean of the count difference:

    E[N1 - N2] = S (q1 - q2) = S F (p1 - p2),
    Var[N1 - N2] = S [ q1 + q2 - (q1 - q2)² ].

Under the normal approximation:

    P(N1 > N2) ≈ Φ(z12),   z12 = F (p1 - p2) √S / sqrt(q1 + q2 - (q1 - q2)²),      (1)

and the best atom beats the runner-up with one-sided confidence `1-α` when `z12 ≥ z_α`:

    F Δ √S  ≥  z_α · sqrt(q1 + q2 - (q1 - q2)²)        (the fidelity-gap law)

    S*  =  z_α² (q1 + q2 - (q1 - q2)²) / (F² Δ²).                                  (2)

**When the floor matters.** Since `q_i = F p_i + (1-F)/D`, there are two regimes:

- *signal-limited*, `F (p1 + p2) ≫ 2(1-F)/D`: `S* ≈ z² (p1 + p2) / (F Δ²)`, so `S* ∝ 1/F`.
- *floor-limited*, `F (p1 + p2) ≪ 2(1-F)/D`: `S* ≈ 2 z² (1-F) / (D F² Δ²)`, so `S* ∝ 1/(D F²)`.

The floor never biases the ranking. It only adds variance, and it matters once
`F ≲ 1/(1 + D p1)`. On the device data, 18 of 26 rows are signal-limited and 8 are
floor-limited (criterion `2(1-F)/D` vs `F(p1+p2)`, with `F = F_xeb`).

### 1.3 All competitors: closed form for top-1 and top-k

The events `{N_j > N_1}` share `N_1`, so they are positively correlated. The naive
product `Π_j Φ(z_1j)` is therefore biased low. We condition on `N_1` instead. Write
`N_1 = S q1 + x sqrt(S q1 (1 - q1))` with `x ~ N(0,1)`. Given `N_1`, the other admissible
counts are `Binomial(S - N_1, q̃_j)` with `q̃_j = q_j / (1 - q1)`: nearly independent and
approximately normal. Then

    π_j(x) = Φ( (μ_j(x) - N_1(x)) / σ_j(x) ),
    μ_j = (S - N_1) q̃_j,   σ_j² = (S - N_1) q̃_j (1 - q̃_j),

    P(best atom is top-1) = ∫ φ(x) Π_{j≠1, admissible} (1 - π_j(x)) dx,                      (3)
    P(best atom in top-k) = ∫ φ(x) PB_{k-1}( {π_j(x)} ) dx,                                   (4)

where `PB_{k-1}` is the Poisson-binomial CDF at `k-1`. This is a one-dimensional integral:
80-node Gauss-Hermite quadrature plus an O(M' k) recursion. The same formula with
"best atom" replaced by the near-optimal set `G_η = {j : p_j ≥ η p1}` gives the
selected-energy guarantee.

**Check against multinomial Monte Carlo** (20,000 draws per row from the same `q`):
(3) is within **0.025** of Monte Carlo on every row (mean absolute error **0.008**), and
(4) with `k = 3` is within **0.021**. The naive pairwise-independent product is off by up
to **0.277** at top-1 (mean 0.097) and **0.378** at top-3. The conditioning is not
optional.

### 1.4 What the law says about the Sep 25 device run

`F = F_xeb` per row (in-sample; see caveat), `S = 4000`:

| quantity | device | law |
|---|---|---|
| exact best atom selected (top-1) | **13 / 26** | 14.34 expected (MC 14.55) |
| Brier score, top-1 | law 0.202 | constant predictor 0.250 |
| AUC, P_top1 vs success | 0.769 | (z12 alone 0.787; F_xeb alone 0.669) |
| best atom within top-3 | 16 / 26 | 20.46 expected (AUC 0.856) |
| best atom within top-5 | 20 / 26 | 21.83 expected |
| selected energy ≥ 0.9·best (η = 0.9) | 17 / 26 | 19.36 expected (MC) |
| FakeMarrakesh top-1 (one-point F) | 19 / 26 | 17.46 expected (AUC 0.872) |

Less circular variant: take `F` from the per-variant CZ fit of §1.6 instead of each row's
own histogram. It predicts 14.46 top-1 (AUC 0.775, Brier 0.197) and 20.75 top-3 (AUC
0.881).

The law gets the top-1 rate right, and it ranks which circuits will fail about as well as
the signal-side quantities alone (AUC of the gap `Δ` alone 0.796, of `-PR` alone 0.784).
At a single circuit geometry, what decides success is mostly the signal's structure in
the family (§2.4), not the hardware.

Only 2 of 26 rows had `S ≥ S*` (95%, runner-up). Both succeeded. Of the 24 rows with
`S < S*`, 11 still succeeded; those are near coin flips, as (3) says. `S*` spans 1.27e3 to
1.27e7 shots over the 6- and 7-qubit rows, and 6.6e9 to 8.6e9 for the two 9-qubit rows.
One failure was predicted safe (P_top1 > 0.9): S002 O1 logDt 1.5 uni. Its best atom came
second.

### 1.5 Where the model fails, and what the data say about it

The model is **global white noise plus shot noise**. It cannot represent:

1. **Coherent errors.** An over-rotation of the linear chirp phases is a deterministic
   frequency translation, and a miscalibrated quadratic phase is a wrong shear. Either one
   moves the peak instead of flattening it, and no number of shots averages it away.
2. **Readout bias.** With independent symmetric flips `e`, `R` is doubly stochastic and
   `R u = u`, but `(Rp)_k = Σ_j e^{d(j,k)} (1-e)^{m-d(j,k)} p_j` mixes Hamming neighbours.
   In frequency, those are `k ± 2^b`, which can be far apart. So readout couples distant
   atoms and can shrink or invert the gap. With asymmetric flips (`e10 ≠ e01`), `R u ≠ u`:
   the floor itself tilts toward low-weight strings and no longer cancels.
3. **Non-uniform (T1-like) noise.** Amplitude damping pulls toward `|0…0>`, which is
   frequency bin 0. For this geometry, bin 0 is *in band*: it sits at the chirp's
   instantaneous frequency at the atom centre, `rate × tc / FS = 8 Hz/s × 1.6 s = 12.8 Hz`.
   This bias can select a wrong atom that stays admissible. It is a hypothesis; this
   note does not test it.

**Adequacy test.** We compare the observed TVD between the device histogram and the
model `q` with the TVD expected from shot noise alone under `q` (the "excess ratio"; 1
means consistent):

| | median excess ratio, depolarizing | + fitted readout (F, e01, e10) by ML |
|---|---|---|
| all 26 rows | 1.68 | 1.50 |
| dynamic (semiclassical QFT, no twirling; DD refused) | **2.32** | 2.13 |
| unitary (DD XY4 + gate/measure twirling ×16) | **1.47** | 1.29 |

The ML readout fit returns median `e01 = 0.13` and `e10 = 0.113`. Rates that large do not
look like physical readout error. The fitted channel is absorbing structured error of
another kind, and it only removes part of the excess. The physics is clean, though:

- **Twirled circuits obey the law.** In the unitary variant, the top-3 count is 10
  observed vs 10.42 predicted, and top-1 is 7 vs 7.37.
- **Untwirled dynamic circuits break it.** In the dynamic variant, top-3 is 6 observed vs
  10.05 predicted, and top-1 is 6 vs 6.97.

This is what Pauli twirling is supposed to do: it turns coherent errors into stochastic
Pauli noise. For random-like circuits, Dalzell, Hunter-Jones and Brandão (2024) show that
local noise becomes global white noise, which is exactly the regime where (1)–(4) are
the right description.

### 1.6 Scaling with qubit count

Fitting `F_xeb = A exp(-ε n_CZ)` (log-linear, device CZ counts after transpilation):

| rows | ε per CZ | A | R² (log F) |
|---|---|---|---|
| all 26 | 0.00291 | 0.540 | 0.889 |
| dynamic 13 | 0.00309 | 0.482 | 0.893 |
| unitary 13 | 0.00287 | 0.628 | 0.979 |

`A < 1` absorbs state preparation and measurement (SPAM) plus the non-CZ layers.
**Out-of-sample:** fitting only the 6- and 7-qubit rows (`ε = 0.00349`, `A = 0.619`)
predicts `F = 0.0117` and `0.0095` for the two 9-qubit circuits (1136 and 1197 CZ). The
observed `F_xeb` is 0.0179 and 0.0205. The fit gets the order of magnitude right, but it
is pessimistic by a factor of 1.5 to 2.2, so the decay is not a single exponential in CZ
count. At those predicted `F`, (3) gives P_top1 of 0.011, and the device ranked the best
atom 65th and 94th.

`F_xeb` ranges from 0.30 to 0.49 at `m = 6`, from 0.12 to 0.26 at `m = 7`, and from 0.018
to 0.020 at `m = 9`. Generic amplitude loading costs `n_CZ = Θ(2^m)` (here 138, 294 and
1136 CZ for the dynamic variant at m = 6, 7, 9), and `M' ∝ 2^m`. Two limits of (2):

- **Noise-like signal in the family** (a heuristic, not derived). The top admissible
  probabilities behave like order statistics of `M'` exponential variates, whose top
  spacing is O(1/M'), so `Δ ∝ 2^-m` and `p1 ∝ m 2^-m`. Then
  `S* ∝ m 2^m e^{ε n_CZ(m)}` (signal-limited) or `2^m e^{2ε n_CZ(m)}` (floor-limited).
  With `n_CZ ∝ 2^m`, this is doubly exponential in `m`: hopeless beyond a few qubits.
- **Sparse signal** (a few chirplets carry the slice). `Δ = O(1)` up to the intrinsic
  grid limit of §2.3, and `S* ∝ 1/F`. Qubit count enters only through the circuit
  fidelity.

The law therefore says that **hardware QACT is viable exactly when the signal is sparse
in the chirplet family**, and that qubit count is not the axis to optimise. §3's
preregistered synthetic test makes this quantitative.

---

## 2. Physics: chirplets are sheared squeezed states, selection is a Husimi projection

### 2.1 The atoms

A Gaussian chirplet `g(t) = exp(-(t - tc)²/(2Δt²)) exp(i(ω0 (t - tc) + c'(t - tc)²))` is,
in the language of continuous-variable quantum optics, a **displaced, squeezed and
sheared vacuum**:

- displacement to `(tc, ω0)` in the time-frequency plane (`t ↔ x`, `ω ↔ p`);
- squeezing set by `Δt` (a minimum-uncertainty state, with `Δt·Δω = 1/2` in amplitude
  units);
- shear set by the chirp rate `c'`.

Its Wigner function is a Gaussian ellipse of minimum area, sheared by the chirp. These
are the Gaussian states of Weedbrook et al. (2012). Mann and Haykin's chirplet
transform (1995) is the expansion onto this family. Mihovilovic and Bracewell's adaptive
chirplet (1991) fits its members to the signal. Matching pursuit (Mallat and Zhang, 1993)
selects greedily from it. The QACT dictionary is the discretization of this family on
`Z_N`.

### 2.2 The circuit is a discretized Gaussian (metaplectic) circuit

In continuous variables, the Gaussian unitaries represent the symplectic group on phase
space (the metaplectic representation; Folland, 1989; Weedbrook et al., 2012):

- multiplication by `exp(i s x²/2)` is the **shear** `(x, p) → (x, p + s x)`, the CV
  quadratic phase gate;
- the Fourier transform is the **quarter rotation** `(x, p) → (p, -x)`;
- fractional Fourier transforms are rotations by any angle (Almeida, 1994).

The QACT circuit is the discrete analogue on the window register. The O(m²)
controlled-phase chirp gates implement `exp(-i 2π c t²/N)`, a shear that un-tilts the
atom's ellipse to horizontal. The inverse QFT is the quarter rotation, and a
computational-basis measurement then reads the frequency coordinate. The circuit measures
along a sheared-and-rotated axis of the time-frequency plane. We use the continuous
correspondence as a guide only. The discrete facts we rely on are Parseval
(`Σ_k p_k = 1`) and the overlap identity in §0, and qpu_selection.py's noiseless check
verifies both (200k-shot Aer runs against the numpy target).

### 2.3 Selection distribution = Husimi-type projection; the Heisenberg-cell gap limit

Because the envelope is folded into the loaded state, `p_k = |<g_k, x>|² / ||env·x||²`.
This is the squared overlap of the signal with a family of squeezed, sheared coherent
states. It generalises the Husimi Q function (Husimi, 1940), where ordinary coherent
states are used. Equivalently, `p_k` is a chirped-Gaussian-window spectrogram, i.e. the
signal's Wigner function smoothed by the atom's Wigner ellipse (continuous Moyal identity;
Folland, 1989). Two consequences:

- **Why a quantum measurement can sample it.** The Wigner function takes negative values
  and is not a probability distribution. Its Gaussian smoothing is nonnegative and
  normalized, so a projective measurement can sample it directly. As `Δt → ∞`, the slice
  tends to a line integral of the Wigner function along `ω = ω0 + 2c't`. That is Wood and
  Barry's Radon-Wigner transform for chirp detection (1994), or equivalently
  `|fractional FT|²` (Almeida, 1994). The finite-`Δt` QACT slice is its localized,
  positive version.
- **The grid limits the gap, not only noise.** Neighbouring outcomes are near-duplicates
  of the same Heisenberg cell. For a pure atom on the window grid (step `2^(n-m)` bins),

      p2 / p1 = exp( -(2π · step · Δt / N)² ),

  and the script reproduces this exactly (to ≤ 7e-7):

  | logdt | qubits | step | p2/p1 (numpy) | formula | PR of the pure atom |
  |---|---|---|---|---|---|
  | 1.5 | 6 | 8 | 0.8240 | 0.8240 | 5.70 |
  | 2.7 | 7 | 4 | 0.5866 | 0.5866 | 3.43 |
  | 3.9 | 9 | 1 | 0.6924 | 0.6924 | 4.13 |

  So even a perfectly matched chirplet has an intrinsic relative gap of only 0.18 to 0.41,
  and a participation ratio of 3 to 6. Exact-argmax success is therefore stricter than
  what matching pursuit needs. The natural success event is `G_η` (selected energy
  ≥ η·best), and the device meets η = 0.9 in 17 of 26 rows vs 13 for the exact argmax.

### 2.4 The two numbers that decide hardware success

- **Sparsity in the family:** the participation ratio `PR = 1/Σ p_k²` (the effective
  number of occupied atoms in the slice, computed on the admissible, renormalized
  distribution). It also sets the XEB normalisation (`Σ p² = 1/PR`). On the EEG windows,
  PR ranges from 4.9 to 9.1 at `m = 6`, from 7.3 to 16.6 at `m = 7`, and is 42.0 at
  `m = 9`.
- **The gap `Δ = p1 - p2`**, which enters (2) squared.

`F` sets the price per shot, and `PR` and `Δ` set how many shots are needed. On the Sep 25
data, `Δ` alone ranks success (AUC 0.796) as well as the full law does. The hardware
question and the sparsity question are the same question.

---

## 3. Proposer-verifier design

**Protocol.** The device draws `S` shots and proposes its `k` most frequent admissible
outcomes. A classical verifier computes the exact overlaps `|<g_j, x>|²` for those `k`.
Each overlap costs O(W) multiply-adds over the truncated window (`W = 2^m` samples; O(N)
in general). The verifier reports the best verified atom and its exact energy. The
device's error mode changes from "wrong atom, silently" to "a lower verified energy,
reported". There are no false accepts, by construction.

**Choosing k.** Use (4): take the smallest `k` with `P(best in top-k) ≥ 1 - δ` at the
estimated `F`. Measured on the device data (exact energy of the best verified candidate,
as a fraction of the best admissible atom):

| k | exact best recovered | mean energy ratio | worst energy ratio (6–7 q rows) |
|---|---|---|---|
| 1 | 13 / 26 | 0.784 | 0.0002 |
| 2 | 16 / 26 | 0.850 | 0.0017 |
| 3 | 16 / 26 | 0.858 | 0.053 |
| 5 | 20 / 26 | 0.904 | 0.297 |
| 8 | 22 / 26 | 0.945 | 0.491 |

**Cost and when it pays.** The total cost is `S·τ_shot + k·W`. Against the device alone,
verification is always worth it: `k·W` is negligible next to thousands of shots, and it
turns 13/26 exact recoveries into 20/26 at `k = 5`. **Against classical computation it
does not pay in this construction.** The exact verifier needs classical access to `x`.
With `x` in hand, the full classical slice is one FFT of O(W log W), which is cheaper
than loading `x` into amplitudes (Θ(2^m) CZ) and far cheaper than `S*` shots. An
advantage would need both of the following:

- the state arrives coherently, as quantum data from a sensor, so no loading cost is paid
  (the verifier then has to become a quantum overlap estimate at O(1/ε²) shots per
  candidate, not O(W));
- the device searches a space far larger than a single slice (superposed `tc`, `Δt` and
  `c`), which the present construction does not do.

The honest contribution is the **audit**. Every hardware atom comes with an exact
certificate, the law predicts in advance how many shots and candidates are needed, and
the XEB fidelity is computable on every circuit because the target is classically cheap.

---

## 4. Testable predictions (preregistered in `results/qpu_law_check.json`)

1. **Twirling restores the law.** Unitary (twirled) circuits already match it (top-3:
   10.42 predicted vs 10 observed). Untwirled dynamic circuits underperform it (10.05 vs
   6), with a higher excess ratio (2.32 vs 1.47). Prediction: running the dynamic
   variant with measurement twirling (or randomized compiling of the feed-forward
   phases), if the runtime allows it, brings its excess ratio toward ~1.5 and its top-3
   count toward the law's 10.
2. **Shot scaling follows (3), row by row.** At the Sep 25 `F`, P_top1 goes from 4000 to
   16,000 to 64,000 shots as follows (`prereg_p_top1_vs_shots`):
   - S001 Cz logDt 1.5 dyn: 0.801 → 0.955 → 0.9997
   - S002 Cz logDt 2.7 dyn: 0.753 → 0.949 → 0.9995
   - S003 O1 logDt 1.5 uni: 0.544 → 0.741 → 0.951
   - S003 Cz logDt 2.7 dyn (gap-limited): 0.113 → 0.172 → 0.250

   So extra shots rescue signal-limited rows and barely move gap-limited ones.
3. **Sparsity, not qubit count, decides success.** At the 9-qubit geometry (logDt 3.9)
   with `F` from the CZ fit (0.0198) and 4000 shots, (3) predicts P_top1:
   - pure synthetic chirp (PR 4.13): **0.620**, `S* = 2.4e4`;
   - chirp in white noise at energy SNR 1 (PR 8.45): 0.376, `S* = 6.5e4`;
   - SNR 0.1 (PR 49.1): 0.030, `S* = 1.2e6`;
   - the real EEG window at the same `F` class (PR 42.0): 0.012, `S* ≈ 7e9`.

   A 9-qubit device run on the synthetic chirp should succeed at a rate near 0.6 while the
   EEG window fails, on the same circuit depth. With the observed `F = 0.018` instead, the
   prediction for the pure chirp is 0.596.

## 5. Caveats

- `F_xeb` is estimated from the same histogram used to score success, so the in-sample
  law is mildly circular. The CZ-fit variant (one `F(n_CZ)` law per variant) avoids
  per-row fitting and gives the same picture (14.46 predicted vs 13 observed top-1).
- 26 circuits on one device and one calibration day (2026-09-25). The AUCs carry wide
  uncertainty, and no confidence intervals are claimed.
- FakeMarrakesh rows use the one-point estimator `F_1 = (q_peak - 1/D)/(p_peak - 1/D)`,
  because their counts were not stored.
- The O(1/M') gap scaling for noise-like spectra in §1.6 is a heuristic, not a
  derivation.

## References (all verified by DOI lookup on 2026-09-26)

- S. Mann and S. Haykin, "The chirplet transform: physical considerations," *IEEE Trans. Signal Processing* 43(11):2745–2761, 1995. doi:10.1109/78.482123
- A. Mihovilovic and R. N. Bracewell, "Adaptive chirplet representation of signals on time-frequency plane," *Electronics Letters* 27(13):1159–1161, 1991. doi:10.1049/el:19910723
- J. C. Wood and D. T. Barry, "Radon transformation of time-frequency distributions for analysis of multicomponent signals," *IEEE Trans. Signal Processing* 42(11):3166–3177, 1994. doi:10.1109/78.330375
- J. C. Wood and D. T. Barry, "Linear signal synthesis using the Radon-Wigner transform," *IEEE Trans. Signal Processing* 42(8):2105–2111, 1994. doi:10.1109/78.301845
- L. B. Almeida, "The fractional Fourier transform and time-frequency representations," *IEEE Trans. Signal Processing* 42(11):3084–3091, 1994. doi:10.1109/78.330368
- S. Mallat and Z. Zhang, "Matching pursuits with time-frequency dictionaries," *IEEE Trans. Signal Processing* 41(12):3397–3415, 1993. doi:10.1109/78.258082
- S. Boixo et al., "Characterizing quantum supremacy in near-term devices," *Nature Physics* 14:595–600, 2018. doi:10.1038/s41567-018-0124-x (arXiv:1608.00263)
- F. Arute et al., "Quantum supremacy using a programmable superconducting processor," *Nature* 574:505–510, 2019. doi:10.1038/s41586-019-1666-5
- A. M. Dalzell, N. Hunter-Jones and F. G. S. L. Brandão, "Random quantum circuits transform local noise into global white noise," *Commun. Math. Phys.* 405, 2024. doi:10.1007/s00220-024-04958-z
- S. Bravyi, S. Sheldon, A. Kandala, D. C. McKay and J. M. Gambetta, "Mitigating measurement errors in multiqubit experiments," *Phys. Rev. A* 103:042605, 2021. doi:10.1103/PhysRevA.103.042605
- C. Weedbrook et al., "Gaussian quantum information," *Rev. Mod. Phys.* 84:621–669, 2012. doi:10.1103/RevModPhys.84.621
- G. B. Folland, *Harmonic Analysis in Phase Space* (Annals of Mathematics Studies 122), Princeton University Press, 1989. doi:10.1515/9781400882427
- K. Husimi, "Some formal properties of the density matrix," *Proc. Phys.-Math. Soc. Japan* 22(4):264–314, 1940. doi:10.11429/ppmsj1919.22.4_264 (verified on J-STAGE; Crossref does not index it)
- R. B. Griffiths and C.-S. Niu, "Semiclassical Fourier transform for quantum computation," *Phys. Rev. Lett.* 76:3228–3231, 1996. doi:10.1103/PhysRevLett.76.3228 (the dynamic-variant inverse QFT)

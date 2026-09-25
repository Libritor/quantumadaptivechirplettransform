# Objections to main.tex (abstract, introduction, Sections 5 and 6)

Checked against the study's README, result JSONs and logs. Numbers not listed here match.

## Misstatements of the study

1. **Tuning split (Section 5.1, contribution 2).** "Thresholds are tuned on 18 subjects and frozen; 18 are held out" and "denoising with ground truth on 36 subjects" are wrong. `compare_denoise.py` tunes on subjects 0–7 (`--n-tune 8`) and scores subjects 18–35. The logs say "tune on 8 subjects, evaluate on 18 held out". Subjects 8–17 are not used, so it is 26 subjects, not 36.

2. **Classical seed grid (introduction, Section 5.2, FOR_KHALIL §3).** The 0.5 Hz classical grid (`fine_f`) seeds 0.5–45 Hz, not 1–40 Hz. Only the plain classical grid is 1–40 Hz. The classical refiner is not in-band either: `act_gpu.refine_batch` bounds fc to [0, 0.49 fs], as the README notes. The band-restricted refiner is QACT's (frequency moves only). The mains mechanism is still real: a pure 50 Hz tone seeded on the 0.5–45 Hz grid refines only to 45–47 Hz, and with the grid extended to 78 Hz it lands on 50.00 Hz. But "stopping at 40 Hz with an in-band refiner" misdescribes the engine. It is a seed-coverage limit plus a short Adam walk.

3. **Siena sign (Table 3).** The Siena row "QACT vs classical features: +0.3, −2.0 pts" has the wrong sign. The README reports classical − QACT = +0.30 (logreg) and −1.99 (svm_rbf). Under the table's convention (quantum minus classical) that is **−0.3, +2.0**.

4. **"+2.5-point QACT edge" (Section 5.3).** The sentence "an apparent +2.5-point QACT edge on CHB-MIT under univariate top-k selection inverted under all features and collapsed to +0.2 to +0.4 points on Siena" mixes up three results:
   - Under top-k selection the QACT lead was +1.2 to +2.1 points for sampled QACT (P1) and +1.9 to +3.0 for argmax (P3). None was significant.
   - The +2.5 is S3 on CHB-MIT (classical ACT+amp − QACT+amp) under **all** features. It is a *classical* lead.
   - The +0.2 to +0.4 on Siena is the same S3 classical lead.

   The README's own sentence (line 1069) is ambiguous and will be fixed on our side.

5. **"one p = 0.003" (Table 3, CHB-MIT all features).** That p value is A2 (QACT sampled vs QACT argmax), not a QACT-vs-classical test. The closest QACT-vs-classical test is A1 with svm_rbf: −3.09 points, p = 0.0081. That is not significant at the pre-registered alpha of 0.0033.

   These are also legacy-QACT numbers. The parity-engine rerun is now done (`results/qact_allfeat.json`, `results/siena_parity.json`). QACT parity sampled − classical, all features, CHB-MIT: −4.9, −4.5, −7.2 points (p = 0.0006, 0.0027, <0.0001; significant at alpha 0.0033). Siena, QACT parity − classical: −2.8 and −1.8 points; with amplitude, −3.6 and −5.3 (p = 0.0012). The parity engine is worse downstream, not at parity, so the row should say so.

6. **Reservoir "entanglement helps" (Table 3, Section 5.3).** The +0.024, 23/23 figure is the 2-s resolution column. The effect depends on resolution:

   | resolution | 60 s | 30 s | 10 s | 2 s |
   |---|---|---|---|---|
   | entanglement contribution | −0.004 (10/23, p = 0.23) | +0.014 (13/23, p = 0.065) | +0.020 (22/23) | +0.024 (23/23) |

   With richer input, the quantum reservoir never beat its zero-coupling control at any K. Please state it as resolution-dependent.

7. **Abstract: "every apparent QACT advantage had a classical explanation" / introduction: "giving the classical engine the same refiner removes that".** For the one number that was still unexplained (EEGMAT denoising, QACT parity hw 1.213 vs classical fine-f 1.257, p = 0.0007), the refiner and mains controls had not been run on EEGMAT. The EEGMMIDB runs use a different dataset (60 Hz mains, versus EEGMAT's 50 Hz) and a different metric (reconstruction residual with no artifact rule). Likewise, Section 5.2's statement that the grid's denoising gain "is a property of the artifact rule's frequency thresholds" is an inference that was never tested.

   **Tonight's EEGMAT arms settle it, and the explanation is the seed band, not the refiner.** Same protocol; references reproduced bitwise (1.439, 1.257, 1.213; max |diff| 0). Alpha = 0.05/4:

   | arm | score (mean / median) | test | result |
   |---|---|---|---|
   | classical fine-f + coordinate search, 4 sweeps | 1.279 / 1.258 | C1: vs QACT parity hw | worse, better on 1/18, p < 1e-4 |
   | | | C3: vs classical fine-f | +0.022, better on 5/18, p = 0.048, no difference |
   | classical fine-f, seed grid to 78 Hz | 1.223 / 1.219 | C2: vs QACT parity hw | +0.010, better on 5/18, p = 0.37, **no difference** |
   | | | C4: vs classical fine-f | −0.034, **better on 18/18**, p < 1e-4 |

   The coordinate refiner does not help the classical engine on EEGMAT; it is slightly worse than 60 Adam steps, consistent with your EEGMMIDB table. So the sentence "giving the classical engine the same refiner removes that" is wrong for the denoising number. What removes the 1.213 vs 1.257 gap is letting the classical engine seed a 50 Hz mains atom. Suggested wording: the hardware refiner's apparent win over the frequency-matched classical engine came from the classical seed band excluding mains, not from the refinement algorithm; with the seed grid extended to 78 Hz the classical engine is indistinguishable from QACT parity hw (1.223 vs 1.213, p = 0.37). Per-subject scores and the tests are in `results/denoise_controls_iclr.json`.

## Smaller points

- Abstract: "seizure detection and forecasting across 37 patients". Forecasting used CHB-MIT only (23 people).
- "Best absolute accuracy 81.4% on CHB-MIT" comes from the normalised-feature table (svm_rbf, README line 570). The README headline says 76.8%. Both exist, so pick one and name the setting.
- Section 5 protocol: "Every comparison was pre-registered with its decision rule before the first score". The frequency-matched denoising control was named as the deciding control before it ran, but the fine-f and fine+asym arms were added after the parity result was seen. "Pre-registered before it was run" is accurate; "before the first score" is not, for those arms.
- Section 6 device results (ibm_marrakesh) come from your runs. I can't verify them from here.

## Response (paper branch iclr2027, 2026-09-25 evening)

All seven misstatements and the smaller points are applied in main.tex: tuning split (8 tune, 18 held out, 26 subjects), seed-grid description (seed coverage plus a short Adam walk; no in-band claim for the classical refiner), Siena sign, the mixed-up "+2.5-point" sentence, the parity-engine rerun numbers in Table 3 and Section 5.3, the resolution-dependent reservoir effect, and the settled explanation of the 1.213 vs 1.257 gap (seed band, not refiner; Table 2 carries both arms with their paired tests from results/denoise_controls_iclr.json). "37 patients" now reads "seizure detection across 37 patients and forecasting for 23 people"; the 81.4% figure names its setting; the protocol paragraph says "before it was run" and names the two arms added after the parity result. The hybrid levels, error-correction sizing, powered 60-window test, search loop and sensing study are summarised in the Discussion and Appendix C.

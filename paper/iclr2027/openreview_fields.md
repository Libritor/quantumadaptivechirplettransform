# OpenReview fields (target venue to be decided: ICML 2027 or a later ML venue)

**Title.** A Quantum Chirplet Transform, Audited: Exact Circuits, Fair Controls, and a Real-Device Test

**TL;DR.** A quantum adaptive chirplet transform with exact circuits and exact gradients, brought to full parity with a strong classical engine, is its classical peer's equal on real EEG under pre-registered controls; we dequantise it exactly, keep the residual coherent, and run it on an IBM Heron processor.

**Keywords.** quantum machine learning; adaptive chirplet transform; matching pursuit; sparse representations; EEG; time-frequency analysis; dequantization; benchmarking; quantum hardware; quantum Fourier transform; parameter-shift rule; seizure detection; quantum reservoir computing; reproducibility

**Primary area.** applications to physical sciences (physics, chemistry, biology, etc.) — alternative: other topics in machine learning (quantum machine learning); a reviewer-friendly second choice is "learning on time series and dynamical systems" or "applications to neuroscience & cognitive science".

**Abstract.** (generated from main.tex by exp/refresh_openreview_abstract.py)

Does a quantum-native signal representation, given every function of its classical peer, learn or reconstruct anything the classical one does not? We answer this for one representation with an audit, and the audit is the contribution. The vehicle is a quantum adaptive chirplet transform (QACT), whose atoms are exact shallow circuits. The audit has three parts: functional parity with a strong classical engine, pre-registered paired comparisons on real EEG, and a control that isolates each apparent win. Every apparent QACT advantage had a classical explanation: the transform's free 0.5 Hz frequency axis, and a classical seed band that excludes mains interference. At equal function set the classical engine is at least as good. The quantum sampler dequantises exactly to a periodogram, computable in O(N log N). Two things survive: the dictionary is gate-native and needs no QRAM, and the matching-pursuit residual can be kept coherent by a one-ancilla reflection whose success probability equals the residual energy fraction, which we verify on circuits. On an IBM Heron processor, six-qubit selection circuits pick the exact atom in 8 of 12 cases and never fall below 0.71 of the best atom's energy; seven-qubit circuits are marginal and long atoms fail. Downstream, the chirplet features themselves add 9-13 accuracy points over amplitude features for cross-patient seizure detection on two databases, while quantum kernels, classifiers, recurrent models and reservoirs land at parity or below. A fully controlled parity result is the baseline any quantum advantage for recorded signals has to clear.

**Authors (camera-ready only).** Khalil Chaghouri, Alexander Vicol, Steve Mann (University of Toronto). Confirm order and affiliations with Khalil before submission.

**Submission checklist.** (Written for ICLR 2027, whose registration was missed; the 9-page discipline still applies. Re-check dates, page limit and style file against the venue once chosen.)
- Abstract must have been registered on OpenReview by September 18, 2026 AoE; the full paper is due September 25, 2026, 11:59 pm AoE (7:59 am Toronto time on September 26).
- Main text 9 pages maximum (references, appendix, AI-use / reproducibility / ethics statements excluded); the AI-use statement is required and is in the manuscript.
- Upload `main.pdf`; keep `\iclrfinalcopy` commented (anonymous).
- Supplementary: the repository snapshot (this branch) as a zip if desired.

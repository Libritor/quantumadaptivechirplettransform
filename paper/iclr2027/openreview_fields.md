# OpenReview fields (ICLR 2027)

**Title.** A Quantum Chirplet Transform, Audited: Exact Circuits, Fair Controls, and a Real-Device Test

**TL;DR.** A quantum adaptive chirplet transform with exact circuits and exact gradients, brought to full parity with a strong classical engine, is its classical peer's equal on real EEG under pre-registered controls; we dequantise it exactly, keep the residual coherent, and run it on an IBM Heron processor.

**Keywords.** quantum machine learning; adaptive chirplet transform; matching pursuit; sparse representations; EEG; time-frequency analysis; dequantization; benchmarking; quantum hardware; quantum Fourier transform; parameter-shift rule; seizure detection; quantum reservoir computing; reproducibility

**Primary area.** applications to physical sciences (physics, chemistry, biology, etc.) — alternative: other topics in machine learning (quantum machine learning); a reviewer-friendly second choice is "learning on time series and dynamical systems" or "applications to neuroscience & cognitive science".

**Abstract.** (as in main.tex; the device numbers fill in from the result file)

Quantum machine learning for time series usually places the quantum component downstream of a classical representation. We put it inside the representation. The adaptive chirplet transform (ACT) writes a signal as a few Gaussian-windowed linear chirps chosen by matching pursuit; because a chirp's quadratic phase is exactly 2-local in the binary encoding of time, a chirplet atom is an O(n^2)-gate circuit on n = log2 N qubits, one quantum Fourier transform returns the atom's whole frequency axis at once, measurement samples atoms with probability proportional to the correlation matching pursuit maximises, and the parameter-shift rule refines every phase parameter exactly. We call the result the quantum ACT (QACT) and audit it as a classical method would be audited. Every function of a strong classical engine is reproduced in quantum-native form, and the distribution a device actually reports, conditioned on post-selection, turns out to be the classical normalised selection criterion. On real EEG, in pre-registered comparisons with paired tests, every apparent QACT advantage had a classical explanation: the transform's free 0.5 Hz frequency axis, a coordinate-search refiner, and a seed band that excludes mains interference. We run the controls that isolate each and find the classical engine at least as good at equal function set; downstream, quantum kernels, variational classifiers, quantum recurrent models and quantum reservoirs land at parity or below on seizure detection and forecasting across 37 patients, while the chirplet features themselves add 9-13 accuracy points. We give the exact dequantisation of the sampler (a periodogram, O(N log N)) and identify what survives it: the dictionary is gate-native and needs no QRAM, and the matching-pursuit residual can be kept coherent by a one-ancilla reflection whose success probability is exactly the residual energy fraction, which we verify on circuits. On an IBM Heron processor, folding the envelope into state preparation, a semiclassical QFT and window-local registers reduce a short atom's circuit from 2,027 to about 140 two-qubit gates, and the device then selects the atom the exact distribution selects for short atoms, while long atoms do not survive. A fully controlled parity result is, we argue, the baseline any quantum advantage for recorded signals has to clear.

**Authors (camera-ready only).** Khalil Chaghouri, Alexander Vicol, Steve Mann (University of Toronto). Confirm order and affiliations with Khalil before submission.

**Submission checklist.**
- Abstract must have been registered on OpenReview by September 18, 2026 AoE; the full paper is due September 25, 2026, 11:59 pm AoE (7:59 am Toronto time on September 26).
- Main text 9 pages maximum (references, appendix, AI-use / reproducibility / ethics statements excluded); the AI-use statement is required and is in the manuscript.
- Upload `main.pdf`; keep `\iclrfinalcopy` commented (anonymous).
- Supplementary: the repository snapshot (this branch) as a zip if desired.

# References

Works this project builds on, compares against, or uses to set expectations.
Grouped by where they matter in the code.

## Data

- **Zyma I., et al.** "Electroencephalograms during Mental Arithmetic Task Performance." *Data* 4(1):14 (2019). — EEGMAT (`acquire.load_eegmat`).
- **Shoeb A.** "Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment." PhD thesis, MIT (2009). — CHB-MIT Scalp EEG Database (`acquire.load_chbmit`).
- **Detti P.** "Siena Scalp EEG Database" v1.0.0, PhysioNet (2020). — 14 patients, 47 seizures, used as the independent confirmation set (`acquire.load_siena`).
- **Goldberger A.L., et al.** "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation* 101(23):e215–e220 (2000). — hosting of both datasets.

## Signal processing and features

- **Mann S., Haykin S.** "The chirplet transform: physical considerations." *IEEE Trans. Signal Processing* 43(11):2745–2761 (1995). — the chirplet / adaptive chirplet transform.
- **Vicol A., Sher S.A., Chen X., Mann S.** "Chirplet-Based Analysis of Gravitational-Waves: From Classical Chirplets to Best Chirplet Chains" (2026), and the [adaptive-chirplet-transform](https://gitlab.com/adaptive-chirplet-transform/adaptive-chirplet-transform) repository. — CPU ACT engine and the validated Muse protocol (`features.py`).
- **Esteller R., et al.** "Line length: an efficient feature for seizure onset detection." *Proc. IEEE EMBS* (2001). — log line-length amplitude feature (CHB-MIT).

## Quantum kernels and classifiers (`quantum.py`, `quantum_fast.py`, `hybrid.py`)

- **Havlíček V., et al.** "Supervised learning with quantum-enhanced feature spaces." *Nature* 567:209–212 (2019). — ZZ feature map, fidelity kernel, QSVC.
- **Shaydulin R., Wild S.M.** "Importance of kernel bandwidth in quantum machine learning." *Phys. Rev. A* 106:042407 (2022). — bandwidth (`angle_scale`).
- **Huang H.-Y., et al.** "Power of data in quantum machine learning." *Nature Communications* 12:2631 (2021). — projected quantum kernel (`pqk`).
- **Pérez-Salinas A., et al.** "Data re-uploading for a universal quantum classifier." *Quantum* 4:226 (2020). — re-uploading.
- **Glick J.R., et al.** "Covariant quantum kernels for data with group structure." *Nature Physics* (2024); arXiv:2105.03406.
- **Liu Y., Arunachalam S., Temme K.** "A rigorous and robust quantum speed-up in supervised machine learning." *Nature Physics* 17:1013–1017 (2021).
- **Huang H.-Y., et al.** "Quantum advantage in learning from experiments." *Science* 376:1182–1186 (2022).
- **Bowles J., Ahmed S., Schuld M.** "Better than classical? The subtle art of benchmarking quantum machine learning models." arXiv:2403.07059 (2024).

## Quantum sequence models (`quantum_torch.py`, `sequence.py`)

- **Chen S.Y.-C., Yoo S., Fang Y.-L.L.** "Quantum Long Short-Term Memory." arXiv:2009.01783 (2020); ICASSP 2022. — QLSTM.

## Quantum reservoir computing (`reservoir.py`)

- **Fujii K., Nakajima K.** "Harnessing Disordered-Ensemble Quantum Dynamics for Machine Learning." *Phys. Rev. Applied* 8:024030 (2017). — QRC; 5–7 qubits comparable to 100–500-node RNNs; input-qubit replacement.
- **Nakajima K., Fujii K., Negoro M., Mitarai K., Kitagawa M.** "Boosting Computational Power through Spatial Multiplexing in Quantum Reservoir Computing." *Phys. Rev. Applied* 11:034021 (2019). — parallel reservoirs.
- **Martínez-Peña R., Giorgi G.L., Nokkala J., Soriano M.C., Zambrini R.** "Dynamical Phase Transitions in Quantum Reservoir Computing." *Phys. Rev. Lett.* 127:100502 (2021). — disordered transverse-field Ising reservoir; best performance near the thermalisation transition.
- **Mujal P., Martínez-Peña R., Giorgi G.L., Soriano M.C., Zambrini R.** "Time-series quantum reservoir computing with weak and projective measurements." *npj Quantum Information* 9:16 (2023). — restart protocol for realistic readout.
- **Kobayashi K., Fujii K., Yamamoto N.** "Feedback-Driven Quantum Reservoir Computing for Time-Series Analysis." *PRX Quantum* 5:040325 (2024).
- **Götting N., Lohof F., Gies C.** "Exploring quantum mechanical advantage for reservoir computing." *Phys. Rev. A* 108:052427 (2023). — entanglement vs memory capacity; motivates the zero-coupling control.
- **Xiong W., Holmes Z., Angrisani A., Suzuki Y., Chotibut T., Thanasilp S.** "Role of scrambling and noise in temporal information processing with quantum systems." arXiv:2505.10080 (2025). — exponential concentration and memory decay with size; motivates small parallel reservoirs.
- "Exponential concentration and symmetries in Quantum Reservoir Computing." arXiv:2505.10062 (2025). — symmetry prevents concentration (our Hamiltonian conserves Z-parity).
- **Zhu C., Ehlers P.J., Nurdin H.I., Soh D.** "Minimalistic and scalable quantum reservoir computing enhanced with feedback." *npj Quantum Information* 11 (2025); arXiv:2412.17817.
- **Hamhoum W., Cherkaoui S., Laprade J.-F., Ahmad O., Wang S.** "Multivariate Time Series Forecasting with Gate-Based Quantum Reservoir Computing on NISQ Hardware." arXiv:2510.13634 (2025). — QRC ties classical reservoirs; NVAR sometimes wins.
- **Kornjača M., et al.** "Large-scale quantum reservoir learning with an analog quantum computer." arXiv:2407.02553 (2024). — 108-qubit hardware QRC.
- **Wolff A., Hamilton K., Rhrissorrakrai K., Parida L., Utro F., Dumas G.** "A Quantum Reservoir for Neurodynamical Forecasting." arXiv:2608.00139 (2026). — QRC on EEG forecasting; did not match classical on EEG.

## Quantum ACT (`qact.py`, docs/QUANTUM_ACT.md)

- **Dürr C., Høyer P.** "A quantum algorithm for finding the minimum." quant-ph/9607014 (1996). — O(sqrt(M)) dictionary search.
- **Brassard G., Høyer P., Mosca M., Tapp A.** "Quantum amplitude amplification and estimation." *Contemporary Mathematics* 305 (2002). — coefficient estimation.
- **Grover L., Rudolph T.** "Creating superpositions that correspond to efficiently integrable probability distributions." quant-ph/0208112 (2002). — Gaussian envelope state preparation.
- **Mitarai K., Negoro M., Kitagawa M., Fujii K.** "Quantum circuit learning." *Phys. Rev. A* 98:032309 (2018). — parameter-shift rule used for off-grid refinement.

## Classical reservoir baselines and benchmarks

- **Jaeger H.** "The 'echo state' approach to analysing and training recurrent neural networks." GMD Report 148 (2001). — echo state network.
- **Gauthier D.J., Bollt E., Griffith A., Barbosa W.A.S.** "Next generation reservoir computing." *Nature Communications* 12:5564 (2021). — NG-RC / NVAR baseline.
- **Buteneers P., et al.** "Real-time detection of epileptic seizures in animal models using reservoir computing." *Epilepsy Research* (2013). — classical reservoirs for seizure detection.
- **Dambre J., Verstraeten D., Schrauwen B., Massar S.** "Information processing capacity of dynamical systems." *Scientific Reports* 2:514 (2012). — memory capacity.
- **Atiya A.F., Parlos A.G.** "New results on recurrent network training." *IEEE Trans. Neural Networks* 11(3):697–709 (2000). — NARMA benchmark.

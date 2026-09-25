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

## Running QACT on real hardware, and how fast it can go (`qact_hw.py`, `qact_hardware.py`, `qact_vs_act_speed.py`, `qact_hw_speedups.py`)

Implemented here:

- **Griffiths R. B., Niu C.-S.** "Semiclassical Fourier transform for quantum computation." *Phys. Rev. Lett.* 76:3228 (1996). — the QFT as mid-circuit measurement plus classically controlled single-qubit phases: no two-qubit gates (`qact_hw.py`, `semiclassical` / `windowed`).
- **Huang R., Izmaylov A. F.** "Quantum gambling: best-arm strategies for generator selection in adaptive variational algorithms." arXiv:2509.14917 (2025). — successive elimination over candidate measurements; 69–93% fewer measurements in their setting (Table 1). The model for adaptive shot allocation in QACT's atom selection.
- **Niu S., Todri-Sanial A.** "Enabling multi-programming mechanism for quantum computing in the NISQ era." *Quantum* 7:925 (2023). — running several small circuits on one chip at once, with crosstalk-aware qubit partitioning.
- **Ohkura Y., Satoh T., Van Meter R.** "Simultaneous execution of quantum circuits on current and near-future NISQ systems." *IEEE Trans. Quantum Eng.* 3 (2022). — multi-programming on IBM hardware; reports a trade-off between success rate and execution time.

- **Fowler A. G., Mariantoni M., Martinis J. M., Cleland A. N.** "Surface codes: towards practical large-scale quantum computation." *Phys. Rev. A* 86:032324 (2012). — the surface code sized in `qact_qec.py`.
- **Beverland M. E., Murali P., Troyer M., et al.** "Assessing requirements to scale to practical quantum advantage." arXiv:2211.07629 (2022). — the methodology behind Microsoft's resource estimator (`qdk`), used to size codes, T-state factories, physical qubits and runtime for the QACT circuits.

Relevant, not implemented (need hardware or data access this project does not have):

- **Bellante A., Zanero S.** "Quantum matching pursuit: a quantum algorithm for sparse representations." *Phys. Rev. A* 105:022414 (2022); arXiv:2208.04145. — Theorem 8: Õ(kn log n + k·sqrt(m)/ξ·log(km/δ)) vs classical O(knm). Quadratic in dictionary size, but assumes fault-tolerant QRAM, and the kn log n term matches FFT-based classical MP.
- **Bellante A., Vanerio S., Zanero S.** "Quantum sparse recovery and quantum orthogonal matching pursuit." arXiv:2510.06925 (2025). — polynomial speed-ups over classical OMP in the QRAM model; input is coherent access to a quantum state.
- **Krstulovic S., Gribonval R.** "MPTK: Matching pursuit made tractable." *Proc. ICASSP* (2006). — the classical baseline any quantum MP must beat: O(N log N) per iteration for structured (Gabor, chirp) dictionaries.
- **Holmes A., Matsuura A. Y.** "Efficient quantum circuits for accurate state preparation of smooth, differentiable functions." arXiv:2005.04351 (2020). — linear-depth loading via matrix product states; candidate for the state preparation that is ~87% of QACT's remaining two-qubit gates.
- **Moosa M., Watts T. W., Chen Y., Sarma A., McMahon P. L.** "Linear-depth quantum circuits for loading Fourier approximations of arbitrary functions." *Quantum Sci. Technol.* (2023), doi:10.1088/2058-9565/acfc62; arXiv:2302.03888. — depth linear in the number of Fourier coefficients; relevant because EEG is band-limited.
- **Smith K. C., Khan A., Clark B. K., Girvin S. M., Wei T.-C.** "Constant-depth preparation of matrix product states with adaptive quantum circuits." arXiv:2404.16083 (2024). — constant depth for symmetric MPS; checked and does not cover generic EEG windows.
- **Suzuki Y., Uno S., et al.** "Amplitude estimation without phase estimation." *Quantum Inf. Process.* 19 (2020); arXiv:1904.10246. — near-quadratic shot reduction with short circuits.
- **Giurgica-Tiron T., Kerenidis I., Labib F., Prakash A., Zeng W.** "Low depth algorithms for quantum amplitude estimation." *Quantum* 6:745 (2022). — optimal depth/query trade-off N·D = O(1/ε²), analysed under depolarising noise.
- **Erle J., Koczor B.** "Nearly optimal amplitude estimation at any depth." arXiv:2608.24434 (2026). — no ancillas or controlled Grover operators; aimed at the early fault-tolerant regime.

- **Chevalier B., Roga W., Takeoka M.** "Compressed sensing enhanced by a quantum approximate optimization algorithm." *Phys. Rev. A* 110:062410 (2024); arXiv:2403.17399. — QAOA for matching-pursuit support detection; simulated, 6 qubits; its gain comes from richer measurements, not the solver (docs/QACT_SEARCH_LOG.md, iteration 2).
- **Hahn W., Romero N.** "Computational phase transitions in binary compressed sensing: quantum annealing inside the relaxation gap." arXiv:2606.00806 (2026). — D-Wave recovers some binary sparse signals classical solvers miss at n=32; fragile (2/30 trials) and gone at n=64.

Why no speed-up over classical is expected for recorded signals:

- **Aaronson S.** "Read the fine print." *Nature Physics* 11:291–293 (2015). — data loading and output caveats that erase many claimed quantum-ML speed-ups.
- **Tang E.** "A quantum-inspired classical algorithm for recommendation systems." *Proc. STOC* (2019). — the first dequantization.
- **Chia N.-H., Gilyén A., Li T., Lin H.-H., Tang E., Wang C.** "Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning." arXiv:1910.06151 (2019). — with matching classical data access, the advantage becomes polynomial at best.
- **Tang E.** "Dequantizing algorithms to understand quantum advantage in machine learning." *Nature Reviews Physics* 4:692–693 (2022).
- **Jaques S., Rattew A. G.** "QRAM: a survey and critique." *Quantum* 9:1922 (2025). — cheap, scalable QRAM (assumed by quantum MP) is unlikely.

Quantum learning models on classical data — the reopened search (docs/QACT_SEARCH_LOG.md, iterations 3-6):

- **Bowles J., Ahmed S., Schuld M.** "Better than classical? The subtle art of benchmarking quantum machine learning models." arXiv:2403.07059 (2024). — 12 QML models vs out-of-the-box classical on 160 datasets: classical wins; entanglement often unnecessary.
- **"Benchmarking quantum kernel support vector machines against classical baselines on tabular data."** arXiv:2604.18837 (2026). — 970 experiments incl. IBM hardware; none of 29 quantum-vs-classical comparisons significant.
- **Delilbasic A., Miroszewski A., Wijata A., Nalepa J., Mielczarek J., Riedel M., Cavallaro G.** "Large-scale quantum kernels for hyperspectral data classification." arXiv:2605.17587 (2026). — all-band fidelity kernels, tensor-network simulated; tested here as iteration 5a (no gain).
- **Caro M. C., Huang H.-Y., Cerezo M., Sharma K., Sornborger A., Cincio L., Coles P. J.** "Generalization in quantum machine learning from few training data." *Nature Communications* 13:4919 (2022). — motivated iteration 5b.
- **Hou et al.** "High-accuracy temporal prediction via experimental quantum reservoir computing in correlated spins." *Phys. Rev. Lett.* 136:120602 (2026); arXiv:2508.12383.
- **Li Q., Mukhopadhyay C., Bayat A., Habibnia A.** "Quantum reservoir computing for realized volatility forecasting." *Phys. Rev. Research* 8:023028 (2026); arXiv:2505.13933.
- **Schuld M., Brádler K., Israel R., Su D., Gupt B.** "Measuring the similarity of graphs with a Gaussian boson sampler." *Phys. Rev. A* 101:032314 (2020).
- **Oh C., Fefferman B., Jiang L., Quesada N.** "Quantum-inspired classical algorithm for graph problems by Gaussian boson sampling." *PRX Quantum* 5:020341 (2024). — why GBS graph kernels are not quantum-hard for non-negative graphs.
- **Liu Y., Arunachalam S., Temme K.** "A rigorous and robust quantum speed-up in supervised machine learning." *Nature Physics* 17:1013 (2021). — the discrete-log construction: what provable advantage requires.
- **Glick J. R., Gujarati T. P., Córcoles A. D., et al.** "Covariant quantum kernels for data with group structure." *Nature Physics* 20:479 (2024).
- **Agliardi G., Cortiana G., Dekusar A., et al.** "Mitigating exponential concentration in covariant quantum kernels for subspace and real-world data." *npj Quantum Information* (2025); arXiv:2412.07915. — up to 156 qubits on IBM hardware; accuracy in line with classical.

Where published speed-ups for signals do exist (quantum data, not recorded arrays):

- **Huang H.-Y., Broughton M., Cotler J., Chen S., Li J., Mohseni M., Neven H., Babbush R., Kueng R., Preskill J., McClean J. R.** "Quantum advantage in learning from experiments." *Science* 376:1182–1186 (2022).
- **Kannan et al.** "Exponential quantum advantage for learning signals with a single qubit." arXiv:2608.13521 (2026). — up to 10^7-fold fewer measurements, for a qubit coupled directly to the sensor.

## Classical reservoir baselines and benchmarks

- **Jaeger H.** "The 'echo state' approach to analysing and training recurrent neural networks." GMD Report 148 (2001). — echo state network.
- **Gauthier D.J., Bollt E., Griffith A., Barbosa W.A.S.** "Next generation reservoir computing." *Nature Communications* 12:5564 (2021). — NG-RC / NVAR baseline.
- **Buteneers P., et al.** "Real-time detection of epileptic seizures in animal models using reservoir computing." *Epilepsy Research* (2013). — classical reservoirs for seizure detection.
- **Dambre J., Verstraeten D., Schrauwen B., Massar S.** "Information processing capacity of dynamical systems." *Scientific Reports* 2:514 (2012). — memory capacity.
- **Atiya A.F., Parlos A.G.** "New results on recurrent network training." *IEEE Trans. Neural Networks* 11(3):697–709 (2000). — NARMA benchmark.

## Quantum sensing of brain fields (`quantum_sensing_opm.py`)

- **Zhang et al.** "Entanglement-enhanced quantum lock-in detection" (two trapped ions, GHZ vs product states). *Nature Communications* (2025), doi:10.1038/s41467-025-66828-z; raw data Zenodo 17614133. The real-hardware anchor for the entanglement gain.
- **MNE-Python OPM sample dataset** (`mne.datasets.opm`): somatosensory evoked fields from QuSpin optically pumped magnetometers, plus an empty-room recording. Gramfort A. et al., *Frontiers in Neuroscience* 7:267 (2013) for MNE.
- **Huelga S.F., Macchiavello C., Pellizzari T., Ekert A.K., Plenio M.B., Cirac J.I.** "Improvement of frequency standards with quantum entanglement." *Physical Review Letters* 79:3865 (1997). Under uncorrelated dephasing, entanglement gives only a constant-factor gain.
- **Giovannetti V., Lloyd S., Maccone L.** "Quantum-enhanced measurements: beating the standard quantum limit." *Science* 306:1330 (2004). The standard quantum limit vs the Heisenberg limit.
- **Boto E. et al.** "Moving magnetoencephalography towards real-world applications with a wearable system." *Nature* 555:657 (2018). OPM-MEG; its sensitivity is limited by environmental and physiological noise.

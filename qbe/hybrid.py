"""Hybrid classical-quantum classifier.

WHY A HYBRID
------------
Three fair comparisons on EEGMAT (see README) found the same thing: on this
weak, near-linear signal the strongest component is a heavily regularised
linear readout of the chirplet features, and every purely quantum model landed
1-9 points behind it. So the hybrid keeps that component and asks the quantum
circuit only for what a linear model cannot express on its own:

    chirplet features --+-- classical branch: top-k features, as-is ----------+
                        |                                                      +--> L2 logistic regression
                        +-- quantum branch: top-n features -> angles ->        |
                            ZZ feature map (optionally re-uploaded) ->         |
                            single-qubit Pauli expectations <X>,<Y>,<Z> -------+

The quantum branch contributes nonlinear terms (<X_k> ~ cos 2x) and, through
the ZZ entangling layers, pairwise feature interactions. Because the classical
branch passes straight through, the readout can always fall back to plain
logreg, so the hybrid should not lose badly.

THE CONTROL THAT MATTERS
------------------------
That same fallback means a higher score does not by itself show the *quantum*
part helped -- any extra nonlinear features might. `feature_map="z"` is the
control: without entanglement the circuit is a product state, and its Pauli
expectations are exactly cos/sin of the input angles, which a classical
computer produces trivially. Entanglement earns credit only if the `zz` hybrid
beats the `z` hybrid.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .quantum import SafeSelectKBest
from .quantum_fast import PQKFeatures


def make_hybrid_pipeline(
    *,
    k_classical: int = 20,
    n_qubits: int = 8,
    layers: int = 2,
    reupload: bool = False,
    entanglement: str = "linear",
    feature_map: str = "zz",
    angle_scale: float = 0.3,
    C: float = 0.1,
) -> Pipeline:
    """Build the hybrid. Every selector and scaler is fitted inside the
    pipeline, so cross-validation never lets them see a held-out fold."""
    n_angles = n_qubits * layers if reupload else n_qubits
    classical = Pipeline([
        ("select", SafeSelectKBest(f_classif, k=k_classical)),
    ])
    quantum = Pipeline([
        ("select", SafeSelectKBest(f_classif, k=n_angles)),
        ("angles", MinMaxScaler(feature_range=(0.0, angle_scale * np.pi))),
        ("circuit", PQKFeatures(n_qubits=n_qubits, layers=layers,
                                entanglement=entanglement,
                                feature_map=feature_map, reupload=reupload)),
        ("std", StandardScaler()),
    ])
    return Pipeline([
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("branches", FeatureUnion([("classical", classical),
                                   ("quantum", quantum)])),
        ("clf", LogisticRegression(max_iter=3000, C=C)),
    ])

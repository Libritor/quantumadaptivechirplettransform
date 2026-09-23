"""Qiskit classifiers over chirplet features.

THE ALGORITHM
-------------
`qsvc` implements the quantum kernel method of Havlicek et al., "Supervised
learning with quantum-enhanced feature spaces" (Nature 567, 2019). Each feature
vector x is encoded into a circuit U(x) -- here a `ZZFeatureMap`, which applies
single-qubit rotations proportional to each feature and two-qubit ZZ rotations
proportional to feature *products*, so the entangling layer is what carries the
feature interactions. The kernel is the state overlap

    K(x, x') = |<0| U'(x')^dagger U(x) |0>|^2

and that Gram matrix is handed to an ordinary SVM. Nothing quantum is trained:
the kernel is deterministic and the remaining optimisation is the SVM's convex
dual. That is the practical reason to prefer it over VQC -- no barren plateaus
and no optimiser tuning.

`vqc` is the variational alternative: the same feature map followed by a
trainable `RealAmplitudes` ansatz, optimised classically. It is included for
comparison and is genuinely harder to train.

Classical baselines are not optional garnish. A quantum classifier that does
not beat an RBF SVM on the same features has demonstrated nothing, so
`run_comparison` always reports them side by side.

QUBIT BUDGET, AND WHY NOT PCA
-----------------------------
One qubit per input feature, so the 52 raw chirplet features must be reduced.
PCA is the reduction usually reached for in this literature, and on these
features it is actively harmful: PCA maximises *variance*, not class
separation. Measured on synthetic eyes-open/closed data, the best single
chirplet feature reaches Cohen's d = 0.80 (`TP10_frac_alpha` -- posterior alpha,
exactly the expected physiology), but the best PCA(4) component only reaches
d = 0.47. The discriminative directions are simply not the high-variance ones.

`reduce="selectk"` is therefore the default: a univariate ANOVA F-test keeps
the `n_qubits` features that actually separate the classes. On the same data it
selects TP9/TP10 alpha fraction and oscillatory fraction -- the physiologically
right answer. `reduce="pca"` remains available for comparison.

Either way the reducer is fitted *inside* the cross-validation pipeline, so it
never sees the held-out fold. Fitting it on the full dataset first would leak
test information and inflate the reported accuracy.

Features are finally scaled to [0, pi]: angle encoding is periodic, so
unbounded inputs would alias distinct feature values onto the same state.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

CLASSIFIERS = ("qsvc", "vqc", "hybrid", "svm_rbf", "logreg")
REDUCERS = ("selectk", "pca", "none")


class SafeSelectKBest(SelectKBest):
    """SelectKBest that clamps k to the number of features available.

    Needed because the variance filter ahead of it drops constant features
    (chirplet band fractions are genuinely all-zero for bands with no atoms),
    so the surviving count is not known until fit time and a fixed k can
    exceed it inside a CV fold.
    """

    def fit(self, X, y):
        self.k = min(self.k, X.shape[1]) if isinstance(self.k, int) else self.k
        return super().fit(X, y)


class VQCClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn wrapper around qiskit-machine-learning's VQC.

    qiskit-machine-learning 0.9.x predates scikit-learn 1.7's `__sklearn_tags__`
    protocol, so a bare VQC raises AttributeError the moment it is placed in a
    Pipeline or passed to cross_val_score. Delegating through a real
    BaseEstimator subclass restores tag support, cloning and CV.
    """

    def __init__(self, n_qubits=4, reps=2, entanglement="linear", maxiter=60, seed=0):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.maxiter = maxiter
        self.seed = seed

    def fit(self, X, y):
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.optimizers import COBYLA

        algorithm_globals_seed(self.seed)
        self.classes_ = np.unique(y)
        self._vqc = VQC(
            feature_map=build_feature_map(self.n_qubits, self.reps, self.entanglement),
            ansatz=RealAmplitudes(self.n_qubits, reps=self.reps),
            optimizer=COBYLA(maxiter=self.maxiter),
        )
        self._vqc.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        pred = np.asarray(self._vqc.predict(np.asarray(X)))
        # VQC returns one-hot rows; collapse to class labels.
        if pred.ndim == 2 and pred.shape[1] == len(self.classes_):
            return self.classes_[pred.argmax(axis=1)]
        return pred.ravel()


FEATURE_MAPS = ("zz", "z", "pauli")


def build_feature_map(
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    kind: str = "zz",
):
    """Build the data-encoding circuit.

    `zz`    ZZFeatureMap -- the second-order encoding of Havlicek et al. Its
            entangling layer applies rotations proportional to feature
            *products*, which is what makes the induced kernel non-classical.
    `z`     ZFeatureMap -- first order, no entanglement. The resulting kernel
            factorises over qubits, so it is closer to a classical product
            kernel. A useful control: if `z` matches `zz`, the entanglement is
            not contributing anything.
    `pauli` PauliFeatureMap with Z/Y/ZZ terms -- a richer alternative basis.

    `reps` repeats the encoding block; `linear` entanglement keeps the circuit
    shallow, which matters on real hardware.
    """
    from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap

    if kind == "zz":
        return ZZFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )
    if kind == "z":
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    if kind == "pauli":
        return PauliFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement,
            paulis=["Z", "Y", "ZZ"],
        )
    raise ValueError(f"unknown feature map {kind!r}; choose from {FEATURE_MAPS}")


def build_quantum_kernel(
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    kind: str = "zz",
    fast: bool = True,
):
    """Fidelity quantum kernel K(x,x') = |<0|U'(x')^dag U(x)|0>|^2.

    `fast` selects `FidelityStatevectorKernel`, which contracts statevectors
    directly instead of running the ComputeUncompute circuit through a sampler.
    It is ~140x faster here and numerically identical on a simulator (verified:
    off-diagonal means agree to 4 decimal places). Set `fast=False` for the
    sampler path, which is what a real backend would use.
    """
    fm = build_feature_map(n_qubits, reps, entanglement, kind)
    if fast:
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel

        return FidelityStatevectorKernel(feature_map=fm)
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    return FidelityQuantumKernel(feature_map=fm)


def make_classifier(
    kind: str,
    n_qubits: int,
    *,
    reps: int = 2,
    entanglement: str = "linear",
    maxiter: int = 60,
    C: float = 1.0,
    seed: int = 0,
    feature_map: str = "zz",
    fast: bool = True,
    gamma: str | float = "scale",
):
    """Build one bare estimator (no preprocessing -- see `make_pipeline`)."""
    if kind == "qsvc":
        from qiskit_machine_learning.algorithms import QSVC

        return QSVC(
            quantum_kernel=build_quantum_kernel(
                n_qubits, reps, entanglement, feature_map, fast
            ),
            C=C,
        )
    if kind == "vqc":
        return VQCClassifier(
            n_qubits=n_qubits, reps=reps, entanglement=entanglement,
            maxiter=maxiter, seed=seed,
        )
    if kind == "svm_rbf":
        return SVC(kernel="rbf", C=C, gamma=gamma)
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=C)
    raise ValueError(f"unknown classifier {kind!r}; choose from {CLASSIFIERS}")


def algorithm_globals_seed(seed: int) -> None:
    try:
        from qiskit_machine_learning.utils import algorithm_globals

        algorithm_globals.random_seed = seed
    except Exception:
        pass


def make_reducer(reduce: str, n_qubits: int, seed: int):
    if reduce == "selectk":
        return ("reduce", SafeSelectKBest(f_classif, k=n_qubits))
    if reduce == "pca":
        return ("reduce", PCA(n_components=n_qubits, random_state=seed))
    if reduce == "none":
        return ("reduce", "passthrough")
    raise ValueError(f"unknown reducer {reduce!r}; choose from {REDUCERS}")


def make_pipeline(
    kind: str,
    n_qubits: int,
    *,
    reps: int = 2,
    entanglement: str = "linear",
    maxiter: int = 60,
    C: float = 1.0,
    seed: int = 0,
    reduce: str = "selectk",
    angle_scale: float = 1.0,
    feature_map: str = "zz",
    fast: bool = True,
    gamma: str | float = "scale",
) -> Pipeline:
    """Full pipeline: drop constants -> standardise -> reduce -> angles -> classify.

    `kind="hybrid"` delegates to `hybrid.make_hybrid_pipeline`, keeping the top
    `max(20, n_qubits)` chirplet features as a classical branch alongside the
    circuit.

    All preprocessing lives inside the Pipeline so cross-validation refits it
    per fold. Fitting the reducer or scaler on the full dataset before
    splitting would leak test information and inflate the reported accuracy --
    which matters most for `selectk`, where leakage would be severe.
    """
    if kind == "hybrid":
        from .hybrid import make_hybrid_pipeline

        return make_hybrid_pipeline(
            k_classical=max(20, n_qubits), n_qubits=n_qubits, layers=reps,
            entanglement=entanglement, feature_map=feature_map,
            angle_scale=angle_scale if angle_scale != 1.0 else 0.3,
            C=C if C != 1.0 else 0.1,
        )
    steps = [
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        make_reducer(reduce, n_qubits, seed),
    ]
    if kind in ("qsvc", "vqc"):
        # Angle encoding is periodic, so inputs must be bounded. `angle_scale`
        # is the kernel BANDWIDTH, and it is the dominant hyperparameter for
        # quantum kernels (Shaydulin & Wild, 2022): encoding into the full
        # [0, pi] spreads states so far apart that the Gram matrix concentrates
        # toward the identity and the model cannot generalise. Shrinking the
        # range is the quantum analogue of lowering gamma in an RBF kernel.
        steps.append(
            ("angles", MinMaxScaler(feature_range=(0.0, angle_scale * np.pi)))
        )
    steps.append(
        (
            "clf",
            make_classifier(
                kind, n_qubits, reps=reps, entanglement=entanglement,
                maxiter=maxiter, C=C, seed=seed, feature_map=feature_map,
                fast=fast, gamma=gamma,
            ),
        )
    )
    return Pipeline(steps)


@dataclass
class Result:
    name: str
    mean: float
    std: float
    scores: np.ndarray
    seconds: float
    extra: dict = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"{self.name:<10} {self.mean * 100:6.1f}% +/- {self.std * 100:4.1f}"
            f"   [{', '.join(f'{s:.2f}' for s in self.scores)}]"
            f"   {self.seconds:6.1f}s"
        )


def make_cv(y: np.ndarray, groups: np.ndarray | None, folds: int, seed: int):
    """Grouped CV whenever groups are supplied.

    On a multi-subject corpus this is not a refinement, it is a correctness
    requirement. Epochs from one subject are highly self-similar, so an
    ungrouped split puts near-duplicates of the test data in the training set
    and the model can score well by recognising the *subject* rather than the
    cognitive state. Grouping by subject forces generalisation to people the
    model has never seen.
    """
    if groups is None:
        n_min = int(np.bincount(y).min())
        folds = max(2, min(folds, n_min))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), folds
    n_groups = len(np.unique(groups))
    folds = max(2, min(folds, n_groups))
    return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed), folds


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    n_qubits: int,
    *,
    folds: int = 5,
    seed: int = 0,
    groups: np.ndarray | None = None,
    **kwargs,
) -> Result:
    """Cross-validate one classifier, grouped by subject when `groups` is given."""
    cv, folds = make_cv(y, groups, folds, seed)
    pipe = make_pipeline(kind, n_qubits, seed=seed, **kwargs)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipe, X, y, cv=cv, groups=groups, scoring="accuracy", n_jobs=1
        )
    return Result(
        name=kind, mean=float(scores.mean()), std=float(scores.std()),
        scores=scores, seconds=time.time() - t0,
        extra={"folds": folds, "n_qubits": n_qubits,
               "grouped": groups is not None},
    )


def run_comparison(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    *,
    kinds: tuple[str, ...] = CLASSIFIERS,
    folds: int = 5,
    seed: int = 0,
    verbose: bool = True,
    groups: np.ndarray | None = None,
    **kwargs,
) -> list[Result]:
    """Cross-validate several classifiers on identical folds and report."""
    if verbose and groups is not None:
        print(f"[quantum] subject-wise CV over {len(np.unique(groups))} groups")
    if verbose:
        chance = float(np.bincount(y).max()) / len(y)
        how = kwargs.get("reduce", "selectk")
        print(
            f"\n[quantum] {X.shape[0]} epochs, {X.shape[1]} features "
            f"-> {how}({n_qubits}) -> {n_qubits} qubits"
        )
        print(f"[quantum] majority-class baseline = {chance * 100:.1f}%\n")
        print(f"{'model':<10} {'accuracy':>14}   {'per-fold':<34} {'time':>7}")
        print("-" * 72)
    results = []
    for kind in kinds:
        r = evaluate(X, y, kind, n_qubits, folds=folds, seed=seed,
                     groups=groups, **kwargs)
        results.append(r)
        if verbose:
            print(r.line())
    return results


def kernel_matrix(
    X: np.ndarray,
    n_qubits: int,
    y: np.ndarray | None = None,
    *,
    reduce: str = "selectk",
    **kwargs,
) -> np.ndarray:
    """Quantum Gram matrix for the given features -- useful for inspection.

    Applies the same reduction/angle-scaling chain the classifier uses, so the
    matrix reflects what QSVC actually sees. `y` is required for the supervised
    `selectk` reduction. This fits on all of X by design: it is for looking at
    the kernel, not for estimating accuracy.
    """
    if reduce == "selectk" and y is None:
        raise ValueError("reduce='selectk' needs y; pass y or use reduce='pca'")
    pre = Pipeline(
        [
            ("var", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            make_reducer(reduce, n_qubits, 0),
            ("angles", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]
    )
    Z = pre.fit_transform(X, y)
    return build_quantum_kernel(n_qubits, **kwargs).evaluate(Z)

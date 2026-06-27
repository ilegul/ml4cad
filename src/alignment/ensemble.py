"""
src/alignment/ensemble.py
─────────────────────────
Model factory, the paper's ENSEMBLE, and the leakage-free pipeline.

The paper's classifier is an **ensemble = mean of the predicted probabilities of
LogisticRegression + RandomForest + AdaBoost**. We reproduce it with a soft
``VotingClassifier`` (soft voting averages class probabilities), and also expose
the individual models plus the extra models requested for completeness
(SVC, k-NN, MLP, GradientBoosting, XGBoost).

Every estimator is wrapped in a **leakage-free** imblearn pipeline:

    SimpleImputer(median)  ->  StandardScaler  ->  oversampler  ->  estimator

Imputation, scaling and oversampling are all fitted *inside* the pipeline, so
when the pipeline is fitted on a training fold nothing leaks from validation /
test. Default oversampler is ``RandomOverSampler`` (it duplicates minority rows,
so it never creates fractional values on the 0/1 dummies); ``SMOTENC`` is
available as an option with the categorical columns flagged.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE

# Individual models that make up the paper ensemble, in order.
ENSEMBLE_MEMBERS = ["LogisticRegression", "RandomForest", "AdaBoost"]

# All models reported in the alignment notebook.
ALIGN_MODELS = [
    "LogisticRegression", "RandomForest", "AdaBoost",   # ensemble members
    "SVC", "KNeighbors", "MLP", "GradientBoosting", "XGBoost",
    "ENSEMBLE",
]


def build_estimator(name: str, random_state: int = RANDOM_STATE):
    """Create a fresh estimator instance by name (no pipeline wrapping)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, AdaBoostClassifier,
        GradientBoostingClassifier, VotingClassifier,
    )
    from sklearn.neural_network import MLPClassifier

    def _make(n):
        if n == "LogisticRegression":
            return LogisticRegression(max_iter=1000, solver="lbfgs",
                                      random_state=random_state)
        if n == "RandomForest":
            return RandomForestClassifier(n_estimators=100, n_jobs=1,
                                          random_state=random_state)
        if n == "AdaBoost":
            return AdaBoostClassifier(n_estimators=50,
                                      random_state=random_state)
        if n == "SVC":
            return SVC(kernel="rbf", probability=True, max_iter=5000,
                       random_state=random_state)
        if n == "KNeighbors":
            return KNeighborsClassifier(n_neighbors=5)
        if n == "MLP":
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                 early_stopping=True,
                                 random_state=random_state)
        if n == "GradientBoosting":
            return GradientBoostingClassifier(random_state=random_state)
        if n == "XGBoost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", n_jobs=1, verbosity=0,
                random_state=random_state)
        raise ValueError(f"Unknown model: {n}")

    if name == "ENSEMBLE":
        estimators = [(m, _make(m)) for m in ENSEMBLE_MEMBERS]
        # soft voting == mean of predicted probabilities (the paper's recipe)
        return VotingClassifier(estimators=estimators, voting="soft",
                                n_jobs=1)
    return _make(name)


def _make_sampler(sampler: str, categorical_features, random_state):
    """Create the oversampler. ``categorical_features`` only used by SMOTENC."""
    if sampler in (None, "none"):
        return None
    if sampler == "RandomOverSampler":
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(random_state=random_state)
    if sampler == "SMOTENC":
        from imblearn.over_sampling import SMOTENC
        if not categorical_features:
            raise ValueError("SMOTENC requires categorical_features indices.")
        return SMOTENC(categorical_features=categorical_features,
                       random_state=random_state)
    raise ValueError(f"Unknown sampler: {sampler}")


def build_pipeline(model_name: str,
                   sampler: str = "RandomOverSampler",
                   categorical_features=None,
                   random_state: int = RANDOM_STATE):
    """
    Build the leakage-free imblearn pipeline for a model.

    steps: SimpleImputer(median) -> StandardScaler -> [sampler] -> estimator
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    samp = _make_sampler(sampler, categorical_features, random_state)
    if samp is not None:
        steps.append(("sampler", samp))
    steps.append(("clf", build_estimator(model_name, random_state)))
    return ImbPipeline(steps)

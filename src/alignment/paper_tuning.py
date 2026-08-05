"""Paper-like extended random-search tuning for the alignment analysis.

This module implements the *scheme B* flow described by Pingitore et al.:

* a stratified 60/20/20 train/validation/test split;
* a 5,000-draw random search with two-fold CV on the training partition only;
* model/sampling/ensemble choices made without looking at the test partition;
* a final, one-shot validation and test evaluation at threshold 0.5.

The implementation is deliberately separate from :mod:`run_alignment`.  It is
safe to import from a notebook because importing or loading cached results never
starts a search.  ``run_paper_tuning`` is the only high-level entry point that
fits models.

The public paper repository contains a few parameter combinations that are no
longer valid with current scikit-learn/XGBoost releases.  Search ranges are kept
as published, with three documented compatibility corrections:

* LogisticRegression uses conditional solver/penalty dictionaries;
* RandomForest ``min_samples_split`` starts at 2 rather than the invalid 1;
* XGBoost uses the canonical ``learning_rate``, ``reg_lambda`` and
  ``reg_alpha`` parameter names and booster-conditional dictionaries.

Missing values remain a declared dataset-level difference from the paper:
median imputation is fitted inside every training fold.  No sampling is used in
the 5,000-draw search.  The optional sampling refinement reproduces the paper's
later under/over-sampling experiment for the final LR/RF/AdaBoost members, with
all samplers kept inside an imbalanced-learn pipeline.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from configs.config import (
    RANDOM_STATE,
    REPORTS_DIR,
    canonical_feature_set_name,
    feature_set_cache_id,
)
from src.alignment.features_align import (
    extract_Xy_align,
    get_all_align_feature_set_names,
)
from src.alignment.holdout_split import split_60_20_20


PAPER_TUNING_VERSION = "paper-tuning-scheme-b-v1"
PAPER_SEARCH_SPACE_VERSION = "pingitore-public-ranges-compatible-v1"
PAPER_SAMPLING_VERSION = "pingitore-under-over-validation-v1"

PAPER_MODELS = (
    "LogisticRegression",
    "SVC",
    "KNeighbors",
    "RandomForest",
    "AdaBoost",
    "MLP",
    "GradientBoosting",
    "XGBoost",
)
PAPER_ENSEMBLE_MEMBERS = (
    "LogisticRegression",
    "RandomForest",
    "AdaBoost",
)

PAPER_PROFILES: dict[str, dict[str, tuple[Any, ...]]] = {
    "primary": {
        "feature_sets": ("CV17", "CV17_THY_CONT_STATES"),
        "horizons": (7, 10),
        "models": PAPER_MODELS,
    },
    "all_features": {
        "feature_sets": tuple(get_all_align_feature_set_names()),
        "horizons": (7, 10),
        "models": PAPER_MODELS,
    },
    # Reduced scope only.  Callers should also explicitly set n_iter=2 (or a
    # similarly small value) when they want a true smoke run.
    "smoke": {
        "feature_sets": ("CV17", "CV17_THY_CONT_STATES"),
        "horizons": (7,),
        "models": PAPER_ENSEMBLE_MEMBERS,
    },
}

PAPER_UNDER_RATIOS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
PAPER_OVER_SAMPLERS = ("SMOTE", "BorderlineSMOTE", "SVMSMOTE")
PAPER_K_NEIGHBORS = (2, 3, 4)


def _default_cache_dir() -> Path:
    return REPORTS_DIR / "alignment" / "paper_tuning"


@dataclass(frozen=True)
class PaperTuningConfig:
    """Configuration for paper-like scheme-B tuning.

    Defaults reproduce the random-search budget reported by Pingitore et al.
    The optional sampling refinement is disabled by default because it adds up
    to hundreds of fits per ensemble member.
    """

    n_iter: int = 5000
    cv: int = 2
    scoring: str = "f1_macro"
    n_jobs: int = -1
    seed: int = RANDOM_STATE
    profile: str = "primary"
    threshold: float = 0.5
    sampling_refinement: bool = False
    sampling_repeats: int = 5
    bootstrap_repeats: int = 1000
    verbose: int = 0
    pre_dispatch: str = "2*n_jobs"
    cache_dir: Path = field(default_factory=_default_cache_dir)

    def __post_init__(self):
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self.cv < 2:
            raise ValueError("cv must be >= 2")
        if self.profile not in PAPER_PROFILES:
            raise ValueError(
                f"Unknown profile {self.profile!r}; choose from "
                f"{tuple(PAPER_PROFILES)}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if self.sampling_repeats < 1:
            raise ValueError("sampling_repeats must be >= 1")
        if self.bootstrap_repeats < 1:
            raise ValueError("bootstrap_repeats must be >= 1")


@dataclass
class PaperTuningRun:
    """Result bundle returned by the run and cache-only loader."""

    search_results: pd.DataFrame
    sampling_results: pd.DataFrame
    ensemble_results: pd.DataFrame
    incremental_results: pd.DataFrame
    artifacts: dict[str, dict[str, Any]]
    status: str
    message: str = ""
    run_signature: str | None = None

    def get_artifact(self, horizon: int, feature_set: str, model: str,
                     stage: str = "search") -> dict[str, Any]:
        """Load a saved joblib artifact by its semantic coordinates."""
        canonical = canonical_feature_set_name(feature_set)
        key = artifact_key(stage, horizon, canonical, model)
        if key not in self.artifacts:
            raise KeyError(f"Artifact not available: {key}")
        path = Path(self.artifacts[key]["joblib"])
        if not path.exists():
            raise FileNotFoundError(path)
        artifact = joblib.load(path)
        artifact_feature_set = canonical_feature_set_name(
            artifact.get("feature_set", canonical)
        )
        if artifact_feature_set != canonical:
            raise ValueError(
                f"Artifact feature set mismatch: {artifact_feature_set} != {canonical}"
            )
        artifact["feature_set"] = canonical
        artifact["cache_joblib"] = str(path)
        artifact["cache_json"] = self.artifacts[key].get("json")
        if "predictions_csv" in self.artifacts[key]:
            artifact["predictions_csv"] = self.artifacts[key][
                "predictions_csv"]
        return artifact


class _MLPHiddenLayerDistribution:
    """Draw one- or two-layer shapes from the ranges in the paper notebook."""

    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        if rng.randint(0, 2) == 0:
            return (int(rng.randint(100, 300)),
                    int(rng.randint(50, 150)))
        return (int(rng.randint(50, 300)),)

    def __repr__(self):
        return "MLPHiddenLayers((100..299,50..149) or (50..299,))"


def _build_classifier(model_name: str, seed: int):
    """Fresh estimator using the imbalance settings of the paper code."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=seed)
    if model_name == "SVC":
        # The paper used probability=True.  It is intentionally retained even
        # though Platt calibration makes the 5,000-draw search expensive.
        return SVC(class_weight="balanced", probability=True,
                   random_state=seed)
    if model_name == "KNeighbors":
        return KNeighborsClassifier()
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if model_name == "AdaBoost":
        return AdaBoostClassifier(random_state=seed)
    if model_name == "MLP":
        return MLPClassifier(random_state=seed)
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed)
    if model_name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_jobs=1, random_state=seed, eval_metric="logloss", verbosity=0)
    raise ValueError(f"Unknown paper model: {model_name}")


def get_paper_search_space(model_name: str,
                           seed: int = RANDOM_STATE) -> list[dict[str, Any]]:
    """Return the Pingitore parameter distributions with compatibility fixes."""
    del seed  # reserved for future stochastic distribution variants

    if model_name == "LogisticRegression":
        common = {
            "clf__C": randint(1, 10),
            "clf__max_iter": randint(50, 500),
            "clf__warm_start": [True, False],
        }
        # Conditional branches avoid the invalid combinations produced by the
        # unconditioned public notebook space.
        return [
            {**common, "clf__solver": ["liblinear"],
             "clf__penalty": ["l1"], "clf__dual": [False]},
            {**common, "clf__solver": ["liblinear"],
             "clf__penalty": ["l2"], "clf__dual": [True, False]},
            {**common, "clf__solver": ["newton-cg", "lbfgs", "sag"],
             "clf__penalty": ["l2"], "clf__dual": [False]},
            {**common, "clf__solver": ["saga"],
             "clf__penalty": ["l1", "l2"], "clf__dual": [False]},
            {**common, "clf__solver": ["saga"],
             "clf__penalty": ["elasticnet"], "clf__dual": [False],
             "clf__l1_ratio": uniform(0.0, 1.0)},
        ]

    if model_name == "SVC":
        common = {
            "clf__C": randint(100, 600),
            "clf__gamma": ["scale", "auto"],
            "clf__max_iter": [400, 800, 1200, 1600],
        }
        return [
            {**common, "clf__kernel": ["rbf"]},
            {**common, "clf__kernel": ["poly"],
             "clf__degree": randint(5, 200),
             "clf__coef0": uniform(0.0, 1.0)},
            {**common, "clf__kernel": ["sigmoid"],
             "clf__coef0": uniform(0.0, 1.0)},
        ]

    if model_name == "KNeighbors":
        return [{
            "clf__n_neighbors": randint(2, 100),
            "clf__weights": ["uniform", "distance"],
            "clf__algorithm": ["ball_tree", "kd_tree"],
            "clf__leaf_size": randint(10, 60),
        }]

    if model_name == "RandomForest":
        return [{
            "clf__n_estimators": randint(10, 200),
            "clf__criterion": ["gini", "entropy"],
            # randint(1, 8) in the public notebook can generate invalid 1.
            "clf__min_samples_split": randint(2, 8),
            "clf__min_samples_leaf": randint(1, 5),
            "clf__max_features": ["sqrt", "log2", None],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        }]

    if model_name == "AdaBoost":
        return [{
            "clf__n_estimators": randint(10, 100),
            "clf__learning_rate": uniform(0.2, 1.0),
        }]

    if model_name == "MLP":
        return [{
            "clf__hidden_layer_sizes": _MLPHiddenLayerDistribution(),
            "clf__solver": ["sgd", "adam"],
            "clf__learning_rate_init": uniform(0.0005, 0.005),
            "clf__learning_rate": ["constant", "adaptive"],
            "clf__alpha": uniform(0.0, 1.0),
            "clf__early_stopping": [True],
            "clf__max_iter": randint(300, 500),
        }]

    if model_name == "GradientBoosting":
        return [{
            "clf__learning_rate": uniform(0.03, 0.2),
            "clf__n_estimators": randint(10, 100),
            "clf__max_depth": randint(2, 6),
            "clf__max_features": ["sqrt", "log2", None],
            "clf__subsample": [0.25, 0.5, 0.75, 1.0],
        }]

    if model_name == "XGBoost":
        common = {
            "clf__learning_rate": uniform(0.05, 0.5),
            "clf__n_estimators": randint(10, 100),
            "clf__reg_lambda": uniform(0.5, 1.5),
            "clf__reg_alpha": uniform(0.0, 0.5),
            "clf__scale_pos_weight": [0.2, 0.4, 0.8, 1.0, 2.0],
        }
        tree = {
            **common,
            "clf__booster": ["gbtree", "dart"],
            "clf__gamma": uniform(0.0, 0.2),
            "clf__max_depth": [2, 3, 4, 6],
            "clf__subsample": [0.25, 0.5, 0.75, 1.0],
        }
        linear = {**common, "clf__booster": ["gblinear"]}
        return [tree, linear]

    raise ValueError(f"Unknown paper model: {model_name}")


def _distribution_label(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    dist = getattr(value, "dist", None)
    if dist is not None:
        args = getattr(value, "args", ())
        kwds = getattr(value, "kwds", {})
        return f"{dist.name}(args={args}, kwds={kwds})"
    return repr(value)


def paper_search_space_summary() -> pd.DataFrame:
    """Tidy, display-ready description of every model search branch."""
    rows = []
    notes = {
        "LogisticRegression": "conditional solver/penalty compatibility",
        "RandomForest": "min_samples_split >= 2 compatibility fix",
        "XGBoost": "canonical names; booster-conditional parameters",
    }
    for model in PAPER_MODELS:
        for branch, space in enumerate(get_paper_search_space(model), start=1):
            for parameter, distribution in sorted(space.items()):
                rows.append({
                    "model": model,
                    "branch": branch,
                    "parameter": parameter,
                    "distribution": _distribution_label(distribution),
                    "compatibility_note": notes.get(model, ""),
                })
    return pd.DataFrame(rows)


def _base_pipeline(model_name: str, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", _build_classifier(model_name, seed)),
    ])


def _dependency_versions() -> dict[str, str]:
    packages = (
        "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
        "joblib", "xgboost",
    )
    out = {"python": platform.python_version()}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not-installed"
    return out


def _config_payload(config: PaperTuningConfig) -> dict[str, Any]:
    return {
        "n_iter": config.n_iter,
        "cv": config.cv,
        "scoring": config.scoring,
        "n_jobs": config.n_jobs,
        "seed": config.seed,
        "profile": config.profile,
        "threshold": config.threshold,
        "sampling_refinement": config.sampling_refinement,
        "sampling_repeats": config.sampling_repeats,
        "bootstrap_repeats": config.bootstrap_repeats,
        "pre_dispatch": config.pre_dispatch,
        "paper_tuning_version": PAPER_TUNING_VERSION,
        "search_space_version": PAPER_SEARCH_SPACE_VERSION,
        "sampling_version": PAPER_SAMPLING_VERSION,
    }


def _data_fingerprint(X: pd.DataFrame, y: Sequence[int]) -> str:
    digest = hashlib.sha256()
    schema = [(str(c), str(X[c].dtype)) for c in X.columns]
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(
        X, index=True, categorize=True).values.tobytes())
    ys = pd.Series(np.asarray(y), index=X.index, name="target")
    digest.update(pd.util.hash_pandas_object(
        ys, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()


def paper_job_signature(X: pd.DataFrame, y: Sequence[int], horizon: int,
                        feature_set: str, model_name: str,
                        config: PaperTuningConfig) -> str:
    """Stable per-job signature used by both the runner and cache loader."""
    payload = {
        "horizon": int(horizon),
        # Stable persistence ID: a display-only rename must not invalidate a
        # completed 5,000-draw search over the identical feature matrix.
        "feature_set": feature_set_cache_id(feature_set),
        "features": list(map(str, X.columns)),
        "model": model_name,
        "config": _config_payload(config),
        "dependencies": _dependency_versions(),
        "data_fingerprint": _data_fingerprint(X, y),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _cache_paths(cache_dir: Path, stage: str, horizon: int,
                 feature_set: str, model_name: str,
                 signature: str) -> tuple[Path, Path]:
    cache_id = feature_set_cache_id(feature_set)
    stem = (f"{stage}_h{int(horizon)}_{_safe_name(cache_id)}_"
            f"{_safe_name(model_name)}_{signature[:20]}")
    return cache_dir / f"{stem}.joblib", cache_dir / f"{stem}.json"


def _json_safe(value: Any):
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _atomic_json_dump(payload: Mapping[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=False,
                              indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _atomic_joblib_dump(payload: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, tmp, compress=3)
    tmp.replace(path)


def _atomic_csv_dump(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_job_cache(joblib_path: Path, json_path: Path,
                    signature: str, expected_feature_set: str
                    ) -> dict[str, Any] | None:
    if not (joblib_path.exists() and json_path.exists()):
        return None
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return None
        artifact = joblib.load(joblib_path)
    except Exception:
        return None
    if artifact.get("signature") != signature:
        return None
    expected = canonical_feature_set_name(expected_feature_set)
    metadata_name = canonical_feature_set_name(
        metadata.get("feature_set", expected)
    )
    artifact_name = canonical_feature_set_name(
        artifact.get("feature_set", expected)
    )
    if metadata_name != expected or artifact_name != expected:
        return None
    artifact["feature_set"] = expected
    artifact["cache_hit"] = True
    artifact["cache_joblib"] = str(joblib_path)
    artifact["cache_json"] = str(json_path)
    return artifact


def _binary_metrics(y_true: Sequence[int], proba: Sequence[float],
                    threshold: float = 0.5) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(proba, dtype=float)
    pred = (p_arr >= threshold).astype(int)
    auc = (roc_auc_score(y_arr, p_arr)
           if np.unique(y_arr).size == 2 else np.nan)
    return {
        "f1_macro": float(f1_score(y_arr, pred, average="macro")),
        "roc_auc": float(auc),
        "precision_1": float(precision_score(
            y_arr, pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(
            y_arr, pred, pos_label=1, zero_division=0)),
        "precision_0": float(precision_score(
            y_arr, pred, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(
            y_arr, pred, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
    }


def _top_search_candidates(search: RandomizedSearchCV,
                           n_top: int = 10) -> list[dict[str, Any]]:
    results = search.cv_results_
    order = np.argsort(results["rank_test_score"])[:n_top]
    return [{
        "rank": int(results["rank_test_score"][i]),
        "mean_test_score": float(results["mean_test_score"][i]),
        "std_test_score": float(results["std_test_score"][i]),
        "params": _json_safe(results["params"][i]),
    } for i in order]


def run_paper_tuning_job(
    X: pd.DataFrame,
    y: Sequence[int],
    horizon: int,
    feature_set: str,
    model_name: str,
    config: PaperTuningConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Fit or load one model/feature/horizon scheme-B tuning job.

    ``RandomizedSearchCV.fit`` receives only the 60% training partition.  The
    validation and test arrays are evaluated only after ``best_estimator_`` is
    fixed; neither can influence the search ranking.
    """
    if model_name not in PAPER_MODELS:
        raise ValueError(f"Unknown paper model: {model_name}")
    X = pd.DataFrame(X).copy()
    y = pd.Series(np.asarray(y, dtype=int), index=X.index)
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")

    signature = paper_job_signature(
        X, y, horizon, feature_set, model_name, config)
    joblib_path, json_path = _cache_paths(
        config.cache_dir, "search", horizon, feature_set, model_name,
        signature)
    if not force:
        cached = _load_job_cache(
            joblib_path, json_path, signature, feature_set
        )
        if cached is not None:
            return cached

    idx_train, idx_val, idx_test = split_60_20_20(y, seed=config.seed)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    inner_cv = StratifiedKFold(
        n_splits=config.cv, shuffle=True, random_state=config.seed)
    search = RandomizedSearchCV(
        estimator=_base_pipeline(model_name, config.seed),
        param_distributions=get_paper_search_space(model_name, config.seed),
        n_iter=config.n_iter,
        scoring=config.scoring,
        cv=inner_cv,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        refit=True,
        return_train_score=False,
        error_score=np.nan,
        verbose=config.verbose,
        pre_dispatch=config.pre_dispatch,
    )
    started = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - started
    estimator = search.best_estimator_

    validation_proba = estimator.predict_proba(X_val)[:, 1]
    test_proba = estimator.predict_proba(X_test)[:, 1]
    artifact = {
        "stage": "search",
        "signature": signature,
        "horizon": int(horizon),
        "feature_set": feature_set,
        "model": model_name,
        "estimator": estimator,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "top_candidates": _top_search_candidates(search),
        "elapsed_seconds": float(elapsed),
        "idx_train": np.asarray(idx_train),
        "idx_validation": np.asarray(idx_val),
        "idx_test": np.asarray(idx_test),
        "index_train": np.asarray(X.index)[idx_train],
        "index_validation": np.asarray(X.index)[idx_val],
        "index_test": np.asarray(X.index)[idx_test],
        "y_validation": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "validation_proba": np.asarray(validation_proba),
        "test_proba": np.asarray(test_proba),
        "validation_metrics": _binary_metrics(
            y_val, validation_proba, config.threshold),
        "test_metrics": _binary_metrics(
            y_test, test_proba, config.threshold),
        "threshold": float(config.threshold),
        "cache_hit": False,
    }
    metadata = {
        "signature": signature,
        "stage": "search",
        "horizon": int(horizon),
        "feature_set": feature_set,
        "features": list(map(str, X.columns)),
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "validation_metrics": artifact["validation_metrics"],
        "test_metrics": artifact["test_metrics"],
        "n_train": len(idx_train),
        "n_validation": len(idx_val),
        "n_test": len(idx_test),
        "config": _config_payload(config),
        "dependencies": _dependency_versions(),
        "artifact_joblib": str(joblib_path),
    }
    _atomic_joblib_dump(artifact, joblib_path)
    _atomic_json_dump(metadata, json_path)
    artifact["cache_joblib"] = str(joblib_path)
    artifact["cache_json"] = str(json_path)
    return artifact


def valid_paper_under_ratios(y: Sequence[int]) -> tuple[float, ...]:
    """Paper under-ratios that are feasible for the supplied binary target.

    A float RandomUnderSampler ratio lower than the current minority/majority
    ratio would require *adding* majority samples and is therefore invalid.
    """
    counts = pd.Series(np.asarray(y, dtype=int)).value_counts()
    if len(counts) != 2:
        return tuple()
    actual = float(counts.min() / counts.max())
    return tuple(r for r in PAPER_UNDER_RATIOS if r > actual + 1e-12)


def paper_sampling_candidates(y_train: Sequence[int]) -> list[dict[str, Any]]:
    """Deterministic candidate list for the optional sampling refinement."""
    under_options: list[float | None] = [None]
    under_options.extend(valid_paper_under_ratios(y_train))
    return [
        {"under_ratio": ratio, "over_sampler": sampler,
         "k_neighbors": k}
        for ratio in under_options
        for sampler in PAPER_OVER_SAMPLERS
        for k in PAPER_K_NEIGHBORS
    ]


def _set_random_state(estimator, seed: int):
    params = estimator.get_params(deep=False)
    updates = {}
    if "random_state" in params:
        updates["random_state"] = seed
    if "n_jobs" in params:
        updates["n_jobs"] = 1
    if updates:
        estimator.set_params(**updates)
    return estimator


def _sampling_pipeline(base_estimator: Pipeline, candidate: Mapping[str, Any],
                       seed: int):
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE, SVMSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    ratio = candidate["under_ratio"]
    under = ("passthrough" if ratio is None else RandomUnderSampler(
        sampling_strategy=float(ratio), random_state=seed))
    sampler_name = candidate["over_sampler"]
    sampler_cls = {
        "SMOTE": SMOTE,
        "BorderlineSMOTE": BorderlineSMOTE,
        "SVMSMOTE": SVMSMOTE,
    }[sampler_name]
    over = sampler_cls(
        sampling_strategy=1.0,
        k_neighbors=int(candidate["k_neighbors"]),
        random_state=seed,
    )
    clf = _set_random_state(
        clone(base_estimator.named_steps["clf"]), seed)
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("under", under),
        ("over", over),
        ("clf", clf),
    ])


def _sampling_signature(base_signature: str, config: PaperTuningConfig) -> str:
    payload = {
        "base_signature": base_signature,
        "version": PAPER_SAMPLING_VERSION,
        "repeats": config.sampling_repeats,
        "seed": config.seed,
        "threshold": config.threshold,
        "dependencies": _dependency_versions(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def refine_paper_sampling(
    X: pd.DataFrame,
    y: Sequence[int],
    search_artifact: Mapping[str, Any],
    config: PaperTuningConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Select paper-like sampling on validation, then evaluate test once.

    Only LR, RF and AdaBoost are accepted because they form the final paper
    ensemble.  Candidate ranking uses mean validation F1-macro over the
    configured repeats.  Test metrics are computed *after* the winner is fixed.
    """
    model_name = str(search_artifact["model"])
    if model_name not in PAPER_ENSEMBLE_MEMBERS:
        raise ValueError(
            "Sampling refinement is limited to LR/RF/AdaBoost members")
    X = pd.DataFrame(X).copy()
    y = pd.Series(np.asarray(y, dtype=int), index=X.index)
    signature = _sampling_signature(search_artifact["signature"], config)
    horizon = int(search_artifact["horizon"])
    feature_set = str(search_artifact["feature_set"])
    joblib_path, json_path = _cache_paths(
        config.cache_dir, "sampling", horizon, feature_set, model_name,
        signature)
    if not force:
        cached = _load_job_cache(
            joblib_path, json_path, signature, feature_set
        )
        if cached is not None:
            return cached

    idx_train = np.asarray(search_artifact["idx_train"], dtype=int)
    idx_val = np.asarray(search_artifact["idx_validation"], dtype=int)
    idx_test = np.asarray(search_artifact["idx_test"], dtype=int)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    evaluations = []
    for candidate in paper_sampling_candidates(y_train):
        f1_values, auc_values, errors = [], [], []
        for repeat in range(config.sampling_repeats):
            repeat_seed = config.seed + repeat
            pipe = _sampling_pipeline(
                search_artifact["estimator"], candidate, repeat_seed)
            try:
                pipe.fit(X_train, y_train)
                proba = pipe.predict_proba(X_val)[:, 1]
                metrics = _binary_metrics(y_val, proba, config.threshold)
                f1_values.append(metrics["f1_macro"])
                auc_values.append(metrics["roc_auc"])
            except Exception as exc:  # candidate-level failure is auditable
                errors.append(f"{type(exc).__name__}: {exc}")
        if f1_values:
            evaluations.append({
                **candidate,
                "mean_validation_f1": float(np.mean(f1_values)),
                "std_validation_f1": float(np.std(f1_values, ddof=1))
                if len(f1_values) > 1 else 0.0,
                "mean_validation_auc": float(np.mean(auc_values)),
                "successful_repeats": len(f1_values),
                "errors": errors,
            })
    if not evaluations:
        raise RuntimeError("Every paper sampling candidate failed")

    def _candidate_key(row):
        auc = row["mean_validation_auc"]
        return (row["mean_validation_f1"],
                auc if np.isfinite(auc) else -np.inf)

    winner = max(evaluations, key=_candidate_key)
    chosen = {k: winner[k] for k in (
        "under_ratio", "over_sampler", "k_neighbors")}
    final_estimator = _sampling_pipeline(
        search_artifact["estimator"], chosen, config.seed)
    # Refit on train only.  Validation was used for selection and is not folded
    # into training, matching the explicit requirement for this refinement.
    final_estimator.fit(X_train, y_train)
    validation_proba = final_estimator.predict_proba(X_val)[:, 1]
    test_proba = final_estimator.predict_proba(X_test)[:, 1]
    artifact = {
        "stage": "sampling",
        "signature": signature,
        "base_signature": search_artifact["signature"],
        "horizon": horizon,
        "feature_set": feature_set,
        "model": model_name,
        "estimator": final_estimator,
        "chosen_sampling": chosen,
        "sampling_candidates": evaluations,
        "selection_validation_f1": winner["mean_validation_f1"],
        "idx_train": idx_train,
        "idx_validation": idx_val,
        "idx_test": idx_test,
        "index_train": np.asarray(X.index)[idx_train],
        "index_validation": np.asarray(X.index)[idx_val],
        "index_test": np.asarray(X.index)[idx_test],
        "y_validation": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "validation_proba": np.asarray(validation_proba),
        "test_proba": np.asarray(test_proba),
        "validation_metrics": _binary_metrics(
            y_val, validation_proba, config.threshold),
        "test_metrics": _binary_metrics(
            y_test, test_proba, config.threshold),
        "threshold": float(config.threshold),
        "cache_hit": False,
    }
    metadata = {
        "signature": signature,
        "base_signature": search_artifact["signature"],
        "stage": "sampling",
        "horizon": horizon,
        "feature_set": feature_set,
        "model": model_name,
        "chosen_sampling": chosen,
        "selection_validation_f1": winner["mean_validation_f1"],
        "validation_metrics": artifact["validation_metrics"],
        "test_metrics": artifact["test_metrics"],
        "config": _config_payload(config),
        "dependencies": _dependency_versions(),
        "artifact_joblib": str(joblib_path),
    }
    _atomic_joblib_dump(artifact, joblib_path)
    _atomic_json_dump(metadata, json_path)
    artifact["cache_joblib"] = str(joblib_path)
    artifact["cache_json"] = str(json_path)
    return artifact


def _artifact_rows(artifact: Mapping[str, Any], protocol: str
                   ) -> list[dict[str, Any]]:
    common = {
        "horizon": int(artifact["horizon"]),
        "feature_set": artifact["feature_set"],
        "model": artifact["model"],
        "protocol": protocol,
        "threshold": float(artifact["threshold"]),
        "signature": artifact["signature"],
        "cache_hit": bool(artifact.get("cache_hit", False)),
    }
    if "best_cv_score" in artifact:
        common["best_cv_score"] = float(artifact["best_cv_score"])
    if "chosen_sampling" in artifact:
        chosen = artifact["chosen_sampling"]
        common.update({
            "under_ratio": chosen["under_ratio"],
            "over_sampler": chosen["over_sampler"],
            "k_neighbors": chosen["k_neighbors"],
            "selection_validation_f1": artifact[
                "selection_validation_f1"],
        })
    rows = []
    for split, key in (("validation", "validation_metrics"),
                       ("test", "test_metrics")):
        metrics = artifact[key]
        n = len(artifact[f"y_{split}"])
        rows.append({**common, "split": split, "n": n, **metrics})
    return rows


def artifact_key(stage: str, horizon: int, feature_set: str,
                 model: str) -> str:
    canonical = canonical_feature_set_name(feature_set)
    return f"{stage}|h{int(horizon)}|{canonical}|{model}"


def _artifact_manifest_entry(artifact: Mapping[str, Any]) -> dict[str, Any]:
    entry = {
        "joblib": artifact["cache_joblib"],
        "json": artifact["cache_json"],
        "signature": artifact["signature"],
    }
    if "predictions_csv" in artifact:
        entry["predictions_csv"] = artifact["predictions_csv"]
    return entry


def _ensemble_signature(member_artifacts: Sequence[Mapping[str, Any]],
                        config: PaperTuningConfig) -> str:
    payload = {
        "members": [a["signature"] for a in member_artifacts],
        "threshold": config.threshold,
        "version": PAPER_TUNING_VERSION,
        "dependencies": _dependency_versions(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_or_load_ensemble(member_artifacts: Sequence[Mapping[str, Any]],
                            config: PaperTuningConfig,
                            force: bool = False) -> dict[str, Any]:
    if {a["model"] for a in member_artifacts} != set(PAPER_ENSEMBLE_MEMBERS):
        raise ValueError("Paper ensemble requires exactly LR, RF and AdaBoost")
    ordered = sorted(member_artifacts,
                     key=lambda a: PAPER_ENSEMBLE_MEMBERS.index(a["model"]))
    reference = ordered[0]
    for artifact in ordered[1:]:
        for key in ("idx_validation", "idx_test", "y_validation", "y_test"):
            if not np.array_equal(reference[key], artifact[key]):
                raise ValueError("Ensemble members do not share identical splits")

    horizon = int(reference["horizon"])
    feature_set = str(reference["feature_set"])
    signature = _ensemble_signature(ordered, config)
    joblib_path, json_path = _cache_paths(
        config.cache_dir, "ensemble", horizon, feature_set,
        "ENSEMBLE_PAPER", signature)
    pred_path = joblib_path.with_name(
        joblib_path.stem + "_predictions.csv")
    if not force:
        cached = _load_job_cache(
            joblib_path, json_path, signature, feature_set
        )
        if cached is not None and pred_path.exists():
            cached["predictions_csv"] = str(pred_path)
            return cached

    validation_proba = np.mean(
        [np.asarray(a["validation_proba"]) for a in ordered], axis=0)
    test_proba = np.mean(
        [np.asarray(a["test_proba"]) for a in ordered], axis=0)
    artifact = {
        "stage": "ensemble",
        "signature": signature,
        "horizon": horizon,
        "feature_set": feature_set,
        "model": "ENSEMBLE_PAPER",
        "member_models": list(PAPER_ENSEMBLE_MEMBERS),
        "member_signatures": [a["signature"] for a in ordered],
        "member_joblib": [a["cache_joblib"] for a in ordered],
        "idx_validation": np.asarray(reference["idx_validation"]),
        "idx_test": np.asarray(reference["idx_test"]),
        "index_validation": np.asarray(reference["index_validation"]),
        "index_test": np.asarray(reference["index_test"]),
        "y_validation": np.asarray(reference["y_validation"]),
        "y_test": np.asarray(reference["y_test"]),
        "validation_proba": validation_proba,
        "test_proba": test_proba,
        "validation_metrics": _binary_metrics(
            reference["y_validation"], validation_proba, config.threshold),
        "test_metrics": _binary_metrics(
            reference["y_test"], test_proba, config.threshold),
        "threshold": float(config.threshold),
        "cache_hit": False,
    }
    predictions = pd.concat([
        pd.DataFrame({
            "horizon": horizon,
            "feature_set": feature_set,
            "model": "ENSEMBLE_PAPER",
            "split": split,
            "row_position": artifact[f"idx_{key}"],
            "row_index": artifact[f"index_{key}"],
            "y": artifact[f"y_{key}"],
            "p_event": artifact[f"{key}_proba"],
            "threshold": float(config.threshold),
        })
        for split, key in (("validation", "validation"), ("test", "test"))
    ], ignore_index=True)
    metadata = {
        "signature": signature,
        "stage": "ensemble",
        "horizon": horizon,
        "feature_set": feature_set,
        "model": "ENSEMBLE_PAPER",
        "member_models": list(PAPER_ENSEMBLE_MEMBERS),
        "member_signatures": artifact["member_signatures"],
        "member_joblib": artifact["member_joblib"],
        "validation_metrics": artifact["validation_metrics"],
        "test_metrics": artifact["test_metrics"],
        "predictions_csv": str(pred_path),
        "config": _config_payload(config),
        "dependencies": _dependency_versions(),
        "artifact_joblib": str(joblib_path),
    }
    _atomic_joblib_dump(artifact, joblib_path)
    _atomic_csv_dump(predictions, pred_path)
    _atomic_json_dump(metadata, json_path)
    artifact["cache_joblib"] = str(joblib_path)
    artifact["cache_json"] = str(json_path)
    artifact["predictions_csv"] = str(pred_path)
    return artifact


def paired_incremental_bootstrap(
    y_true: Sequence[int],
    proba_base: Sequence[float],
    proba_thyroid: Sequence[float],
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Paired test-set delta F1/AUROC with percentile bootstrap CIs."""
    y_arr = np.asarray(y_true, dtype=int)
    p_base = np.asarray(proba_base, dtype=float)
    p_thy = np.asarray(proba_thyroid, dtype=float)
    if not (len(y_arr) == len(p_base) == len(p_thy)):
        raise ValueError("Paired arrays must have identical lengths")
    pred_base = (p_base >= threshold).astype(int)
    pred_thy = (p_thy >= threshold).astype(int)
    point_f1 = (f1_score(y_arr, pred_thy, average="macro")
                - f1_score(y_arr, pred_base, average="macro"))
    point_auc = (roc_auc_score(y_arr, p_thy)
                 - roc_auc_score(y_arr, p_base))
    rng = np.random.default_rng(seed)
    delta_f1, delta_auc = [], []
    for _ in range(n_boot):
        sample = rng.integers(0, len(y_arr), len(y_arr))
        y_b = y_arr[sample]
        if np.unique(y_b).size != 2:
            continue
        delta_f1.append(
            f1_score(y_b, pred_thy[sample], average="macro")
            - f1_score(y_b, pred_base[sample], average="macro"))
        delta_auc.append(
            roc_auc_score(y_b, p_thy[sample])
            - roc_auc_score(y_b, p_base[sample]))
    if not delta_f1:
        raise RuntimeError("No valid two-class bootstrap samples")
    return {
        "delta_f1_macro": float(point_f1),
        "delta_f1_ci_lo": float(np.percentile(delta_f1, 2.5)),
        "delta_f1_ci_hi": float(np.percentile(delta_f1, 97.5)),
        "delta_auroc": float(point_auc),
        "delta_auroc_ci_lo": float(np.percentile(delta_auc, 2.5)),
        "delta_auroc_ci_hi": float(np.percentile(delta_auc, 97.5)),
    }


def _profile(config: PaperTuningConfig) -> dict[str, tuple[Any, ...]]:
    return PAPER_PROFILES[config.profile]


def _cache_profile(config: PaperTuningConfig) -> dict[str, tuple[Any, ...]]:
    """Profile serialized with stable cache IDs instead of display labels."""
    profile = _profile(config)
    return {
        **profile,
        "feature_sets": tuple(
            feature_set_cache_id(name) for name in profile["feature_sets"]
        ),
    }


def _expected_job_signatures(cohorts: Mapping[int, pd.DataFrame],
                             config: PaperTuningConfig
                             ) -> tuple[list[str], dict[tuple[int, str], tuple]]:
    signatures = []
    extracted: dict[tuple[int, str], tuple] = {}
    spec = _profile(config)
    for horizon in spec["horizons"]:
        cohort = cohorts[int(horizon)]
        for feature_set in spec["feature_sets"]:
            X, y = extract_Xy_align(
                cohort, feature_set, f"y{int(horizon)}")
            extracted[(int(horizon), feature_set)] = (X, y)
            for model in spec["models"]:
                signatures.append(paper_job_signature(
                    X, y, int(horizon), feature_set, model, config))
    return signatures, extracted


def _run_signature(cohorts: Mapping[int, pd.DataFrame],
                   config: PaperTuningConfig
                   ) -> tuple[str, dict[tuple[int, str], tuple]]:
    signatures, extracted = _expected_job_signatures(cohorts, config)
    payload = {
        "job_signatures": signatures,
        "config": _config_payload(config),
        "profile": _cache_profile(config),
    }
    raw = json.dumps(_json_safe(payload), sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), extracted


def _run_paths(config: PaperTuningConfig, signature: str
               ) -> tuple[Path, dict[str, Path]]:
    stem = f"run_{config.profile}_{signature[:20]}"
    manifest = config.cache_dir / f"{stem}.json"
    tables = {
        "search_results": config.cache_dir / f"{stem}_search.csv",
        "sampling_results": config.cache_dir / f"{stem}_sampling.csv",
        "ensemble_results": config.cache_dir / f"{stem}_ensemble.csv",
        "incremental_results": config.cache_dir / f"{stem}_incremental.csv",
    }
    return manifest, tables


def _empty_run(status: str, message: str,
               run_signature: str | None = None) -> PaperTuningRun:
    return PaperTuningRun(
        search_results=pd.DataFrame(),
        sampling_results=pd.DataFrame(),
        ensemble_results=pd.DataFrame(),
        incremental_results=pd.DataFrame(),
        artifacts={}, status=status, message=message,
        run_signature=run_signature)


def _read_result_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _canonicalize_result_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose canonical labels while accepting historical cached tables."""
    frame = frame.copy()
    for column in ("feature_set", "base"):
        if column in frame.columns:
            frame[column] = frame[column].map(canonical_feature_set_name)
    return frame


def _canonicalize_artifact_manifest(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Canonicalize semantic artifact keys without touching legacy paths."""
    canonical: dict[str, dict[str, Any]] = {}
    for key, entry in artifacts.items():
        parts = str(key).split("|", 3)
        if len(parts) != 4 or not parts[1].startswith("h"):
            raise ValueError(f"Malformed artifact key: {key}")
        stage, horizon_token, feature_set, model = parts
        new_key = artifact_key(
            stage, int(horizon_token.removeprefix("h")), feature_set, model
        )
        if new_key in canonical:
            raise ValueError(f"Duplicate canonical artifact key: {new_key}")
        canonical[new_key] = dict(entry)
    return canonical


def load_cached_paper_tuning(
    cohorts: Mapping[int, pd.DataFrame],
    config: PaperTuningConfig | None = None,
) -> PaperTuningRun:
    """Read-only cache loader; this function never fits or writes anything."""
    config = config or PaperTuningConfig()
    signature, _ = _run_signature(cohorts, config)
    manifest_path, table_paths = _run_paths(config, signature)
    if not manifest_path.exists():
        return _empty_run(
            "cache_missing",
            "No complete paper-tuning cache exists for this profile/config/data.",
            signature)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _empty_run(
            "cache_invalid", f"Cannot read cache manifest: {exc}", signature)
    if manifest.get("run_signature") != signature:
        return _empty_run(
            "cache_invalid", "Run signature mismatch in cache manifest.",
            signature)
    artifacts = _canonicalize_artifact_manifest(
        manifest.get("artifacts", {})
    )
    expected_paths: list[str] = [str(path) for path in table_paths.values()]
    for entry in artifacts.values():
        expected_paths.extend(
            str(entry[name]) for name in ("joblib", "json", "predictions_csv")
            if entry.get(name))
    missing = [path for path in expected_paths if not Path(path).exists()]
    status = "cache_complete" if not missing else "cache_partial"
    message = ("Complete cached run loaded." if not missing else
               f"Loaded manifest, but {len(missing)} artifacts are missing.")
    return PaperTuningRun(
        search_results=_canonicalize_result_table(
            _read_result_table(table_paths["search_results"])
        ),
        sampling_results=_canonicalize_result_table(
            _read_result_table(table_paths["sampling_results"])
        ),
        ensemble_results=_canonicalize_result_table(
            _read_result_table(table_paths["ensemble_results"])
        ),
        incremental_results=_canonicalize_result_table(
            _read_result_table(table_paths["incremental_results"])
        ),
        artifacts=artifacts,
        status=status,
        message=message,
        run_signature=signature,
    )


def run_paper_tuning(
    cohorts: Mapping[int, pd.DataFrame],
    config: PaperTuningConfig | None = None,
    force: bool = False,
) -> PaperTuningRun:
    """Run/resume the selected paper-like profile and persist all outputs."""
    config = config or PaperTuningConfig()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    run_signature, extracted = _run_signature(cohorts, config)
    manifest_path, table_paths = _run_paths(config, run_signature)
    spec = _profile(config)

    search_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    ensemble_artifacts: dict[tuple[int, str], dict[str, Any]] = {}

    for horizon in spec["horizons"]:
        for feature_set in spec["feature_sets"]:
            X, y = extracted[(int(horizon), feature_set)]
            final_members = []
            for model in spec["models"]:
                search_artifact = run_paper_tuning_job(
                    X, y, int(horizon), feature_set, model, config,
                    force=force)
                search_rows.extend(_artifact_rows(
                    search_artifact, "paper_random_search"))
                key = artifact_key(
                    "search", int(horizon), feature_set, model)
                artifacts[key] = _artifact_manifest_entry(search_artifact)
                active_artifact = search_artifact

                if (config.sampling_refinement
                        and model in PAPER_ENSEMBLE_MEMBERS):
                    sampled = refine_paper_sampling(
                        X, y, search_artifact, config, force=force)
                    sampling_rows.extend(_artifact_rows(
                        sampled, "paper_sampling_refinement"))
                    sample_key = artifact_key(
                        "sampling", int(horizon), feature_set, model)
                    artifacts[sample_key] = _artifact_manifest_entry(sampled)
                    active_artifact = sampled
                if model in PAPER_ENSEMBLE_MEMBERS:
                    final_members.append(active_artifact)

            if len(final_members) == len(PAPER_ENSEMBLE_MEMBERS):
                ensemble = _build_or_load_ensemble(
                    final_members, config, force=force)
                ensemble_artifacts[(int(horizon), feature_set)] = ensemble
                protocol = ("paper_ensemble_sampling_refined"
                            if config.sampling_refinement
                            else "paper_ensemble_tuned")
                ensemble_rows.extend(_artifact_rows(ensemble, protocol))
                ens_key = artifact_key(
                    "ensemble", int(horizon), feature_set, "ENSEMBLE_PAPER")
                artifacts[ens_key] = _artifact_manifest_entry(ensemble)

    for horizon in spec["horizons"]:
        base = ensemble_artifacts.get((int(horizon), "CV17"))
        thyroid = ensemble_artifacts.get((int(horizon), "CV17_THY_CONT_STATES"))
        if base is None or thyroid is None:
            continue
        if not (np.array_equal(base["idx_test"], thyroid["idx_test"])
                and np.array_equal(base["y_test"], thyroid["y_test"])):
            raise ValueError(
                "CV17 and CV17_THY_CONT_STATES do not share the same paired test rows")
        delta = paired_incremental_bootstrap(
            base["y_test"], base["test_proba"], thyroid["test_proba"],
            n_boot=config.bootstrap_repeats, seed=config.seed,
            threshold=config.threshold)
        incremental_rows.append({
            "horizon": int(horizon),
            "base": "CV17",
            "feature_set": "CV17_THY_CONT_STATES",
            "model": "ENSEMBLE_PAPER",
            "split": "test",
            "threshold": config.threshold,
            **delta,
        })

    search_df = pd.DataFrame(search_rows)
    sampling_df = pd.DataFrame(sampling_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    incremental_df = pd.DataFrame(incremental_rows)
    frames = {
        "search_results": search_df,
        "sampling_results": sampling_df,
        "ensemble_results": ensemble_df,
        "incremental_results": incremental_df,
    }
    for name, frame in frames.items():
        # Give intentionally empty tables a readable schema for cache loading.
        if frame.empty:
            frame = pd.DataFrame(columns=[
                "horizon", "feature_set", "model", "protocol", "split"])
            frames[name] = frame
        _atomic_csv_dump(frame, table_paths[name])

    manifest = {
        "run_signature": run_signature,
        "status": "cache_complete",
        "profile": config.profile,
        "config": _config_payload(config),
        "dependencies": _dependency_versions(),
        "tables": {name: str(path) for name, path in table_paths.items()},
        "artifacts": artifacts,
    }
    _atomic_json_dump(manifest, manifest_path)
    return PaperTuningRun(
        search_results=frames["search_results"],
        sampling_results=frames["sampling_results"],
        ensemble_results=frames["ensemble_results"],
        incremental_results=frames["incremental_results"],
        artifacts=artifacts,
        status="computed",
        message="Paper-like tuning run completed and cached.",
        run_signature=run_signature,
    )


__all__ = [
    "PAPER_MODELS",
    "PAPER_PROFILES",
    "PAPER_ENSEMBLE_MEMBERS",
    "PaperTuningConfig",
    "PaperTuningRun",
    "artifact_key",
    "get_paper_search_space",
    "paper_search_space_summary",
    "paper_job_signature",
    "run_paper_tuning_job",
    "valid_paper_under_ratios",
    "paper_sampling_candidates",
    "refine_paper_sampling",
    "paired_incremental_bootstrap",
    "run_paper_tuning",
    "load_cached_paper_tuning",
]

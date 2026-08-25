"""Models, sampling, tuning, calibration, thresholds and frozen test predictions."""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import (
    SMOTE, SMOTENC, BorderlineSMOTE, RandomOverSampler, SVMSMOTE,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import utils

EPS = 1e-6

# Versioned per model, so changing one search space invalidates only that
# model's cached tuning and leaves every other cached job usable.
SEARCH_SPACE_VERSION = {
    "LogisticRegression": "v1",
    "SVC": "v2",  # bounded max_iter and C
    "KNeighbors": "v1",
    "RandomForest": "v1",
    "AdaBoost": "v1",
    "MLP": "v1",
    "GradientBoosting": "v1",
    "XGBoost": "v1",
}


def search_space_version(model_name: str) -> str:
    if model_name not in SEARCH_SPACE_VERSION:
        raise KeyError(f"No search-space version registered for {model_name!r}")
    return SEARCH_SPACE_VERSION[model_name]


# ---------------------------------------------------------------------------
# Model and sampler factories
# ---------------------------------------------------------------------------

def get_model(name: str, class_weight: bool = False, seed: int = config.SEED,
              probability: bool = True):
    """Estimator with a deterministic random state where supported."""
    balanced = "balanced" if class_weight else None
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=5000, class_weight=balanced,
                                  random_state=seed)
    if name == "SVC":
        # A finite iteration budget bounds candidates whose solver would not
        # terminate in reasonable time, and lets them surface as convergence
        # warnings rather than stalling the search.
        # probability=False applies to the search only: the f1_macro scorer
        # calls predict, so fitting Platt probabilities for every candidate is
        # wasted work. The winner is refitted with probabilities afterwards.
        return SVC(probability=probability, class_weight=balanced,
                   max_iter=500_000, random_state=seed)
    if name == "KNeighbors":
        return KNeighborsClassifier()
    if name == "RandomForest":
        return RandomForestClassifier(class_weight=balanced, random_state=seed,
                                      n_jobs=1)
    if name == "AdaBoost":
        return AdaBoostClassifier(random_state=seed)
    if name == "MLP":
        return MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        # tree_method belongs to the gbtree branch only; setting it globally
        # makes gblinear candidates emit an unused-parameter warning.
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss")
    raise KeyError(f"Unknown model {name!r}")


def supports_class_weight(name: str) -> bool:
    # XGBoost is excluded on purpose: it handles imbalance through
    # scale_pos_weight, which the search already covers, so a class_weight row
    # would silently duplicate the unsampled one.
    return name in {"LogisticRegression", "SVC", "RandomForest"}


def get_sampler(name: str, feature_set: str, seed: int = config.SEED):
    if name in ("none", "class_weight"):
        return None
    if name == "RandomOverSampler":
        return RandomOverSampler(random_state=seed)
    if name == "SMOTENC":
        return SMOTENC(categorical_features=utils.categorical_indices(feature_set),
                       random_state=seed)
    if name == "SMOTE":
        return SMOTE(random_state=seed)
    if name == "BorderlineSMOTE":
        return BorderlineSMOTE(random_state=seed)
    if name == "SVMSMOTE":
        return SVMSMOTE(random_state=seed)
    raise KeyError(f"Unknown sampler {name!r}")


def make_pipeline(model_name: str, sampler: str, feature_set: str,
                  seed: int = config.SEED, probability: bool = True):
    """Leakage-free pipeline whose step order suits the chosen sampler.

    SMOTENC must see unscaled columns so its categorical indices still refer to
    binary values; every other sampler runs after scaling. Scaling is kept for
    tree models too, for exact comparability with the paper protocol.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    sampler_obj = get_sampler(sampler, feature_set, seed)
    model = get_model(model_name, class_weight=(sampler == "class_weight"),
                      seed=seed, probability=probability)

    if sampler == "SMOTENC":
        steps.append(("sampler", sampler_obj))
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("scaler", StandardScaler()))
        if sampler_obj is not None:
            steps.append(("sampler", sampler_obj))
    steps.append(("model", model))

    if any(step == "sampler" for step, _ in steps):
        return ImbPipeline(steps)
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Hyperparameter spaces
# ---------------------------------------------------------------------------

def positive_weight_ratio(y_train) -> float:
    """Negative-to-positive ratio, from the training partition only."""
    y = np.asarray(y_train)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive cases in the training partition")
    return float((len(y) - n_pos) / n_pos)


def search_space(model_name: str, y_train) -> list:
    """Search space as a list of solver- or booster-compatible branches."""
    if model_name == "LogisticRegression":
        return [
            {"model__solver": ["liblinear"], "model__penalty": ["l1", "l2"],
             "model__C": loguniform(1e-3, 1e2)},
            {"model__solver": ["lbfgs"], "model__penalty": ["l2"],
             "model__C": loguniform(1e-3, 1e2)},
            {"model__solver": ["saga"], "model__penalty": ["elasticnet"],
             "model__C": loguniform(1e-3, 1e2),
             "model__l1_ratio": uniform(0, 1)},
        ]
    if model_name == "SVC":
        # C is capped at 100: above that the margin is effectively hard and the
        # solver cost explodes without improving validation performance.
        return [
            {"model__kernel": ["rbf"], "model__C": loguniform(1e-2, 1e2),
             "model__gamma": loguniform(1e-4, 1e0)},
            {"model__kernel": ["linear"], "model__C": loguniform(1e-2, 1e2)},
        ]
    if model_name == "KNeighbors":
        return [{
            "model__n_neighbors": randint(3, 60),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }]
    if model_name == "RandomForest":
        return [{
            "model__n_estimators": randint(100, 800),
            "model__max_depth": [None, 4, 6, 8, 12, 20],
            "model__min_samples_split": randint(2, 40),
            "model__min_samples_leaf": randint(1, 40),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        }]
    if model_name == "AdaBoost":
        return [{
            "model__n_estimators": randint(50, 600),
            "model__learning_rate": loguniform(1e-2, 2e0),
        }]
    if model_name == "MLP":
        return [{
            "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
            "model__alpha": loguniform(1e-6, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-1),
            "model__activation": ["relu", "tanh"],
        }]
    if model_name == "GradientBoosting":
        return [{
            "model__n_estimators": randint(50, 500),
            "model__learning_rate": loguniform(1e-3, 5e-1),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
            "model__min_samples_leaf": randint(1, 40),
        }]
    if model_name == "XGBoost":
        # scale_pos_weight is anchored on the training imbalance only; the
        # paper's fixed range assumed the opposite class orientation.
        ratio = positive_weight_ratio(y_train)
        weights = [1.0, 0.5 * ratio, 1.0 * ratio, 1.5 * ratio, 2.0 * ratio]
        shared = {
            "model__n_estimators": randint(100, 800),
            "model__learning_rate": loguniform(1e-3, 5e-1),
            "model__scale_pos_weight": weights,
            "model__reg_lambda": loguniform(1e-3, 1e2),
            "model__reg_alpha": loguniform(1e-4, 1e1),
        }
        tree = {
            "model__booster": ["gbtree"],
            "model__tree_method": ["hist"],
            "model__max_depth": randint(2, 9),
            "model__min_child_weight": randint(1, 20),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__gamma": loguniform(1e-4, 1e1),
        }
        linear = {
            "model__booster": ["gblinear"],
            "model__updater": ["coord_descent"],
        }
        return [{**shared, **tree}, {**shared, **linear}]
    raise KeyError(f"Unknown model {model_name!r}")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune(X_train, y_train, model_name: str, sampler: str, feature_set: str,
         n_iter: int = None, cv: int = None, seed: int = config.SEED) -> dict:
    """Random search on the training partition only.

    Returns the refitted best estimator plus diagnostics: failed candidates,
    NaN scores and captured convergence warnings.
    """
    n_iter = config.SEARCH_ITER if n_iter is None else n_iter
    cv = config.INNER_CV if cv is None else cv

    # The f1_macro scorer calls predict, so the search runs without probability
    # estimates and the winner is refitted with them below.
    pipe = make_pipeline(model_name, sampler, feature_set, seed, probability=False)
    spaces = search_space(model_name, y_train)
    inner = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        pipe, param_distributions=spaces, n_iter=n_iter, scoring="f1_macro",
        cv=inner, n_jobs=config.N_JOBS, refit=False, random_state=seed,
        error_score=np.nan, return_train_score=False)

    started = time.perf_counter()
    search.fit(X_train, y_train)

    best_params = {k: _jsonable(v) for k, v in search.best_params_.items()}
    estimator = make_pipeline(model_name, sampler, feature_set, seed)
    estimator.set_params(**search.best_params_)

    # Candidate-level warnings are raised inside the parallel workers and cannot
    # be captured here, so only this final in-process fit is counted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    convergence = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

    return {
        "model": model_name,
        "sampler": sampler,
        "feature_set": feature_set,
        "estimator": estimator,
        "best_params": best_params,
        "best_cv_f1_macro": float(search.best_score_),
        "n_candidates": int(scores.size),
        "n_failed_candidates": int(np.isnan(scores).sum()),
        "final_fit_convergence_warnings": int(convergence),
        "final_fit_other_warnings": int(len(caught) - convergence),
        "elapsed_seconds": round(elapsed, 2),
        "positive_weight_ratio": positive_weight_ratio(y_train),
    }


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    return value


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def training_oof_proba(estimator, X_train, y_train, cv: int = None,
                       seed: int = config.SEED) -> np.ndarray:
    """Out-of-fold probabilities inside the training partition.

    The calibrator is fitted on these, so the validation outcomes stay
    available for threshold selection alone.
    """
    cv = config.CALIBRATION_CV if cv is None else cv
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return cross_val_predict(estimator, X_train, y_train, cv=folds,
                             method="predict_proba", n_jobs=config.N_JOBS)[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(p_oof, y_train, method: str = None):
    """Fit a calibrator on training out-of-fold probabilities."""
    method = config.CALIBRATION_PRIMARY if method is None else method
    if method == "none":
        return None
    if method == "sigmoid":
        model = LogisticRegression(max_iter=1000)
        model.fit(_logit(p_oof).reshape(-1, 1), np.asarray(y_train))
        return ("sigmoid", model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(np.asarray(p_oof, dtype=float), np.asarray(y_train))
        return ("isotonic", model)
    raise KeyError(f"Unknown calibration method {method!r}")


def apply_calibrator(calibrator, p) -> np.ndarray:
    if calibrator is None:
        return np.asarray(p, dtype=float)
    kind, model = calibrator
    if kind == "sigmoid":
        return model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    return model.predict(np.asarray(p, dtype=float))


def _calibration_in_the_large(y_true, z, lo: float = -20.0, hi: float = 20.0) -> float:
    """Intercept a solving mean(sigmoid(a + logit(p))) = mean(y), slope fixed at 1."""
    target = float(np.mean(y_true))

    def gap(a):
        return float(np.mean(1.0 / (1.0 + np.exp(-(a + z))))) - target

    if gap(lo) > 0 or gap(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def calibration_report(y_true, y_proba, n_bins: int = None) -> dict:
    """Brier, calibration intercept and slope, ECE and observed-to-expected."""
    n_bins = config.CALIBRATION_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    z = _logit(y_proba)
    slope = intercept = float("nan")
    if len(np.unique(y_true)) == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = LogisticRegression(penalty=None, max_iter=1000).fit(
                z.reshape(-1, 1), y_true)
        slope = float(fit.coef_[0][0])
        intercept = _calibration_in_the_large(y_true, z)

    points = utils.reliability_points(y_true, y_proba, n_bins)
    ece = float((points["n"] / points["n"].sum()
                 * (points["observed_rate"] - points["mean_predicted"]).abs()).sum())
    observed = float(y_true.mean())
    expected = float(y_proba.mean())
    return {
        "brier": float(np.mean((y_proba - y_true) ** 2)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": ece,
        "observed_rate": observed,
        "expected_rate": expected,
        "observed_to_expected": float(observed / expected) if expected > 0 else float("nan"),
    }


def decision_curve(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Net benefit against treat-all and treat-none, with a degeneracy flag."""
    thresholds = config.CLINICAL_RISK_THRESHOLDS if thresholds is None else thresholds
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for t in thresholds:
        flagged = y_proba >= t
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        odds = t / (1 - t)
        rows.append({
            "threshold": float(t),
            "n_flagged": int(flagged.sum()),
            "flagged_fraction": float(flagged.mean()),
            "true_positives": tp,
            "false_positives": fp,
            "net_benefit_model": tp / n - (fp / n) * odds,
            "net_benefit_treat_all": prevalence - (1 - prevalence) * odds,
            "net_benefit_treat_none": 0.0,
            "degenerate": bool(flagged.all() or (~flagged).all()),
        })
    return pd.DataFrame(rows)


def clinical_impact(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    thresholds = config.CLINICAL_RISK_THRESHOLDS if thresholds is None else thresholds
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    rows = []
    for t in thresholds:
        flagged = y_proba >= t
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        tn = int((~flagged & (y_true == 0)).sum())
        rows.append({
            "threshold": float(t),
            "high_risk_per_1000": 1000.0 * flagged.mean(),
            "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "degenerate": bool(flagged.all() or (~flagged).all()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frozen final-test evaluation
# ---------------------------------------------------------------------------

def freeze_test_predictions(name: str, patient_ids, y_true, proba_raw,
                            proba_calibrated, threshold: float,
                            signature: str = None,
                            **manifest_fields) -> pd.DataFrame:
    """Persist the final test predictions; later steps only read them.

    Everything downstream of the test set (ablation, permutation importance,
    the ML indicator) consumes these immutable artifacts and must never refit
    a model, refit calibration or move a threshold. Test outcomes are used for
    none of the development decisions.
    """
    frame = pd.DataFrame({
        config.ID_COL: np.asarray(patient_ids),
        "y_true": np.asarray(y_true, dtype=int),
        "proba_raw": np.asarray(proba_raw, dtype=float),
        "proba_calibrated": np.asarray(proba_calibrated, dtype=float),
    })
    frame["prediction"] = (frame["proba_calibrated"] >= threshold).astype(int)
    frame["p_survive"] = 1.0 - frame["proba_calibrated"]

    path = config.PRED_DIR / f"{name}_test.csv"
    manifest_path = config.PRED_DIR / f"{name}_test_manifest.json"
    if config.USE_CACHE and path.exists() and manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("signature") == signature:
            return pd.read_csv(path)

    utils.write_frame(frame, path)
    utils.write_manifest(
        manifest_path, artifact=name, signature=signature,
        threshold=float(threshold), n_test=int(len(frame)),
        n_events=int(frame["y_true"].sum()), **manifest_fields)
    return frame


def load_test_predictions(name: str) -> pd.DataFrame:
    path = config.PRED_DIR / f"{name}_test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 3_sampling_ensemble_calibration.ipynb first.")
    return pd.read_csv(path)


def save_model(estimator, name: str, **manifest_fields) -> Path:
    path = config.MODELS_DIR / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(estimator, path)
    utils.write_manifest(config.MODELS_DIR / f"{name}_manifest.json",
                         artifact=name, **manifest_fields)
    return path


def load_model(name: str):
    path = config.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the tuning notebook first.")
    return load(path)


def model_manifest(name: str) -> dict:
    path = config.MODELS_DIR / f"{name}_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))

"""Acceptance tests: cohort construction, split integrity and leakage guards.

Tests that need artifacts from the notebooks skip when those artifacts are
absent, so the suite runs on a fresh clone. A legacy artifact can never satisfy
a test here: everything is read from the paths the current pipeline writes.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import ensemble as ens
import survival
import train
import utils


def _load(path):
    path = Path(path)
    if not path.exists():
        pytest.skip(f"{path.name} not produced yet; run the notebooks first")
    return utils.read_frame(path)


@pytest.fixture(scope="module")
def full_cohort():
    return _load(config.COHORT_FULL)


@pytest.fixture(scope="module")
def strict_cohort():
    return _load(str(config.COHORT_STRICT).format(horizon=config.HORIZONS[0]))


# ---------------------------------------------------------------------------
# Imports must not train anything
# ---------------------------------------------------------------------------

def test_importing_modules_trains_nothing():
    script = """
import sys
from sklearn.base import BaseEstimator

def forbidden(self, *args, **kwargs):
    raise AssertionError(f"{type(self).__name__}.fit called during import")

BaseEstimator.fit = forbidden
import config, utils, train, ensemble, survival
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def test_patient_identifiers_are_unique(full_cohort):
    utils.check_unique_ids(full_cohort)


def test_event_free_patients_were_observed_through_the_horizon(strict_cohort):
    horizon_days = config.HORIZONS[0] * config.DAYS_PER_YEAR
    event_free = strict_cohort[strict_cohort[config.TARGET] == 0]
    assert (event_free["followup_days"] >= horizon_days).all()


def test_events_occurred_within_the_horizon(strict_cohort):
    horizon_days = config.HORIZONS[0] * config.DAYS_PER_YEAR
    events = strict_cohort[strict_cohort[config.TARGET] == 1]
    assert (events["followup_days"] <= horizon_days).all()
    assert (events["event_cardiac"] == 1).all()


def test_strict_cohort_excludes_early_noncardiac_deaths(strict_cohort):
    horizon_days = config.HORIZONS[0] * config.DAYS_PER_YEAR
    early_noncardiac = ((strict_cohort["event_noncardiac"] == 1)
                        & (strict_cohort["followup_days"] < horizon_days))
    assert not early_noncardiac.any()


def test_missing_thyroid_status_is_not_imputed_as_normal(full_cohort):
    unknown = full_cohort["Euthyroid"].isna()
    if not unknown.any():
        pytest.skip("no missing thyroid status in this dataset")
    assert full_cohort.loc[unknown, "thyroid_abnormal"].isna().all()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_are_disjoint_and_cover_the_cohort(strict_cohort):
    splits = utils.load_splits(config.HORIZONS[0])
    utils.assert_disjoint(splits)
    covered = sorted(splits["train"] + splits["valid"] + splits["test"])
    assert covered == sorted(strict_cohort[config.ID_COL].astype(int))


def test_every_feature_set_uses_the_same_patients(strict_cohort):
    reference = None
    for feature_set in config.FEATURE_SETS:
        _, _, ids = utils.extract_xy(strict_cohort, feature_set)
        if reference is None:
            reference = ids
        utils.assert_same_patients(reference, ids)


def test_missing_feature_raises_instead_of_degrading(strict_cohort):
    damaged = strict_cohort.drop(columns=["TSH"])
    with pytest.raises(KeyError):
        utils.extract_xy(damaged, config.PRIMARY_THYROID)


def test_assert_same_patients_rejects_a_mismatch():
    with pytest.raises(AssertionError):
        utils.assert_same_patients(np.array([1, 2, 3]), np.array([1, 2, 4]))


# ---------------------------------------------------------------------------
# Leakage guards
# ---------------------------------------------------------------------------

def test_hyperparameter_search_never_sees_validation_or_test(strict_cohort, monkeypatch):
    splits = utils.load_splits(config.HORIZONS[0])
    X, y, ids = utils.extract_xy(strict_cohort, config.PRIMARY_BASELINE)
    masks = utils.split_masks(ids, splits)
    X_train, y_train = X[masks["train"]], y[masks["train"]]

    seen = {}
    original = train.RandomizedSearchCV.fit

    def spy(self, X_seen, y_seen=None, **kwargs):
        seen["index"] = set(np.asarray(X_seen.index))
        return original(self, X_seen, y_seen, **kwargs)

    monkeypatch.setattr(train.RandomizedSearchCV, "fit", spy)
    train.tune(X_train, y_train, "LogisticRegression", "none",
               config.PRIMARY_BASELINE, n_iter=2, cv=2)

    forbidden = set(np.flatnonzero(masks["valid"] | masks["test"]))
    assert seen["index"] == set(np.flatnonzero(masks["train"]))
    assert not (seen["index"] & forbidden)


def test_preprocessing_and_sampling_are_fitted_on_training_rows_only(strict_cohort):
    splits = utils.load_splits(config.HORIZONS[0])
    X, y, ids = utils.extract_xy(strict_cohort, config.PRIMARY_BASELINE)
    masks = utils.split_masks(ids, splits)
    pipe = train.make_pipeline("LogisticRegression", "RandomOverSampler",
                               config.PRIMARY_BASELINE)
    pipe.fit(X[masks["train"]], y[masks["train"]])

    imputer = pipe.named_steps["imputer"]
    assert imputer.statistics_.shape[0] == X.shape[1]
    scaler = pipe.named_steps["scaler"]
    # The scaler sees the resampled training rows, never validation or test.
    assert scaler.n_samples_seen_ >= int(masks["train"].sum())
    assert scaler.n_samples_seen_ <= 2 * int(masks["train"].sum())


def test_smotenc_runs_before_scaling_so_dummies_stay_categorical():
    pipe = train.make_pipeline("LogisticRegression", "SMOTENC",
                               config.PRIMARY_THYROID)
    order = [name for name, _ in pipe.steps]
    assert order.index("sampler") < order.index("scaler")

    other = train.make_pipeline("LogisticRegression", "RandomOverSampler",
                                config.PRIMARY_THYROID)
    order = [name for name, _ in other.steps]
    assert order.index("scaler") < order.index("sampler")


def test_categorical_indices_match_the_feature_matrix():
    features = utils.feature_list(config.PRIMARY_THYROID)
    indices = utils.categorical_indices(config.PRIMARY_THYROID)
    assert all(0 <= i < len(features) for i in indices)
    for i in indices:
        assert features[i] not in config.CONTINUOUS_FEATURES
    for name in config.THYROID_CONT:
        assert features.index(name) not in indices


def test_scale_pos_weight_is_derived_from_the_training_partition():
    y_train = np.array([0] * 80 + [1] * 20)
    ratio = train.positive_weight_ratio(y_train)
    assert ratio == pytest.approx(4.0)
    space = train.search_space("XGBoost", y_train)
    weights = space[0]["model__scale_pos_weight"]
    assert ratio in weights
    assert max(weights) >= ratio


def test_search_space_versions_are_per_model():
    """Changing one search space must not invalidate the other cached jobs."""
    for model in config.MODELS:
        assert isinstance(train.search_space_version(model), str)
    with pytest.raises(KeyError):
        train.search_space_version("NotAModel")

    signatures = {
        model: utils.cache_signature(stage="tune", model=model,
                                     search_space=train.search_space_version(model))
        for model in config.MODELS
    }
    assert len(set(signatures.values())) == len(config.MODELS)


def test_svc_search_space_is_bounded():
    """An unbounded SVC space made a single search run for eleven hours."""
    estimator = train.get_model("SVC")
    assert 0 < estimator.max_iter < 10_000_000
    space = train.search_space("SVC", np.array([0] * 80 + [1] * 20))
    for branch in space:
        assert branch["model__C"].support()[1] <= 100


def test_calibrator_is_fitted_on_training_out_of_fold_probabilities():
    rng = np.random.default_rng(0)
    p_oof = rng.uniform(0.01, 0.99, 400)
    y_train = (rng.uniform(size=400) < p_oof).astype(int)
    calibrator = train.fit_calibrator(p_oof, y_train, "sigmoid")
    assert calibrator[0] == "sigmoid"
    calibrated = train.apply_calibrator(calibrator, np.array([0.1, 0.5, 0.9]))
    assert calibrated.shape == (3,)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_threshold_selection_uses_only_the_supplied_observations():
    rng = np.random.default_rng(1)
    y_valid = rng.integers(0, 2, 300)
    p_valid = rng.uniform(size=300)
    first = utils.optimize_threshold(y_valid, p_valid)
    second = utils.optimize_threshold(y_valid, p_valid)
    assert first == second
    assert config.THRESHOLD_GRID.min() <= first <= config.THRESHOLD_GRID.max()


# ---------------------------------------------------------------------------
# Frozen artifacts
# ---------------------------------------------------------------------------

def _manifests():
    paths = sorted(config.PRED_DIR.glob("*_test_manifest.json"))
    if not paths:
        pytest.skip("no frozen test predictions yet")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_frozen_manifests_record_a_non_test_threshold_source():
    for manifest in _manifests():
        assert manifest["threshold_source"] in {"own validation",
                                                "baseline validation"}
        assert manifest["calibration"] == config.CALIBRATION_PRIMARY


def test_frozen_predictions_cover_exactly_the_test_patients():
    splits = utils.load_splits(config.HORIZONS[0])
    paths = sorted(config.PRED_DIR.glob(f"h{config.HORIZONS[0]}_*_test.csv"))
    if not paths:
        pytest.skip("no frozen test predictions yet")
    for path in paths:
        frame = pd.read_csv(path)
        assert sorted(frame[config.ID_COL]) == sorted(splits["test"])


def test_paired_arms_share_identical_patients():
    horizon = config.HORIZONS[0]
    base = config.PRED_DIR / f"h{horizon}_{config.PRIMARY_BASELINE}_paper_test.csv"
    other = config.PRED_DIR / f"h{horizon}_{config.PRIMARY_THYROID}_paper_test.csv"
    if not (base.exists() and other.exists()):
        pytest.skip("frozen arms not produced yet")
    utils.assert_same_patients(pd.read_csv(base)[config.ID_COL].to_numpy(),
                               pd.read_csv(other)[config.ID_COL].to_numpy())


def test_legacy_cache_is_outside_the_active_tree():
    manifest = config.RESULTS_DIR / "legacy_cache_manifest.json"
    if not manifest.exists():
        pytest.skip("no legacy cache recorded")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"].startswith("retained for traceability")
    assert not (ROOT / payload["original_path"]).exists()


# ---------------------------------------------------------------------------
# Competing-risks predictions
# ---------------------------------------------------------------------------

def test_cardiac_cif_is_bounded_and_monotone():
    rng = np.random.default_rng(5)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    cause = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    frame = survival.competing_frame(pd.DataFrame({
        "survival_time_years": rng.exponential(6.0, size=n) + 0.05,
        "event_cardiac": (cause == 1).astype(int),
        "event_noncardiac": (cause == 2).astype(int),
    }))
    models = survival.fit_cause_specific(X, frame)
    times = np.array([1.0, 3.0, 5.0, 7.0])
    cif = survival.predict_cardiac_cif(models, X, times)
    assert cif.shape == (n, len(times))
    assert np.all((cif >= 0.0) & (cif <= 1.0))
    # A cumulative incidence can never decrease in time.
    assert np.all(np.diff(cif, axis=1) >= -1e-12)
    # An extreme covariate must not push the truncated incidence past 1: the
    # fT3/fT4 ratio contains outliers orders of magnitude above its median.
    extreme = X.copy()
    extreme.loc[extreme.index[:5], "a"] = 500.0
    cif_extreme = survival.predict_cardiac_cif(models, extreme, times)
    assert np.all((cif_extreme >= 0.0) & (cif_extreme <= 1.0))


def test_cause_specific_fits_censor_the_competing_event():
    rng = np.random.default_rng(6)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n)})
    cause = rng.choice([0, 1, 2], size=n, p=[0.4, 0.3, 0.3])
    frame = survival.competing_frame(pd.DataFrame({
        "survival_time_years": rng.exponential(5.0, size=n) + 0.05,
        "event_cardiac": (cause == 1).astype(int),
        "event_noncardiac": (cause == 2).astype(int),
    }))
    models = survival.fit_cause_specific(X, frame)
    # The baseline grid carries every observed time, but the hazard may only
    # jump at the fitted cause's own event times.
    for code in (1, 2):
        baseline = models[code].steps[-1][1].cum_baseline_hazard_
        increments = np.diff(np.concatenate(
            [[0.0], np.asarray(baseline.y, dtype=float)]))
        jump_times = np.asarray(baseline.x, dtype=float)[increments > 1e-12]
        own = frame.loc[frame["cause_code"] == code, "surv_time"].to_numpy()
        assert np.isin(jump_times, own).all()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def test_paired_bootstrap_draws_are_shared_for_a_given_seed():
    first = utils.bootstrap_indices(50, 10, seed=7)
    second = utils.bootstrap_indices(50, 10, seed=7)
    assert np.array_equal(first, second)


def test_identical_predictions_give_a_zero_delta():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    delta = utils.paired_delta(y, p, p, 0.5, 0.5, n_boot=50)
    assert np.allclose(delta["delta"], 0.0)
    assert not delta["excludes_zero"].any()


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.9])
    q = utils.benjamini_hochberg(p)
    assert np.all(q >= p)
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all((q >= 0) & (q <= 1))
    assert utils.benjamini_hochberg(np.array([0.04]))[0] == pytest.approx(0.04)


def test_ensemble_probability_is_the_member_mean():
    members = [np.array([0.2, 0.4]), np.array([0.4, 0.8])]
    assert np.allclose(ens.mean_proba(members), [0.3, 0.6])


def test_ensemble_alignment_check_rejects_mismatched_members():
    with pytest.raises(AssertionError):
        ens.check_members_aligned({"a": np.array([1, 2]), "b": np.array([1, 3])})

"""Nested stacking, frozen development decisions and bootstrap equivalence."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import predictive
import stacking
import utils

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("stacking_sensitivity", ROOT / "10_stacking_sensitivity.py")
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)


def test_candidates_cover_all_subsets_once():
    models = list("abcdefgh")
    subsets = stacking.candidate_subsets(models)
    assert len(subsets) == 247
    assert len({tuple(s) for s in subsets}) == 247
    assert all(2 <= len(s) <= 8 for s in subsets)
    with pytest.raises(ValueError, match="distinct"):
        stacking.candidate_subsets(["a", "a"])


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_bootstrap_reproduces_existing_paired_inference(ties):
    rng = np.random.default_rng(91)
    y = np.tile([0, 0, 0, 1], 30)
    base, other = rng.random((2, len(y)))
    if ties:
        base, other = np.round(base, 1), np.round(other, 1)
    expected = utils.paired_delta(y, base, other, 0.4, 0.6, n_boot=50)
    counts = stacking.bootstrap_counts(y, 50)
    actual = stacking.paired_metrics(y, base, other, 0.4, 0.6, counts)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)


def test_bootstrap_skips_single_class_draws_and_rejects_invalid_alignment():
    y = np.array([0, 1])
    counts = stacking.bootstrap_counts(y, 100)
    assert np.all((counts @ y > 0) & (counts @ y < 2))
    with pytest.raises(ValueError, match="aligned"):
        stacking.bootstrap_metrics(y, np.array([0.2]), 0.5, counts)


def test_holm_adjustment_matches_step_down_examples():
    np.testing.assert_allclose(stacking.holm_adjust([0.04, 0.01, 0.2]), [0.08, 0.03, 0.2])
    with pytest.raises(ValueError, match="finite"):
        stacking.holm_adjust([np.nan])


def test_nested_oof_excludes_outer_patients_from_inner_and_meta_training(tmp_path, monkeypatch):
    X = pd.DataFrame({"patient": np.arange(100), "signal": np.linspace(-1, 1, 100)})
    y = np.tile([0, 1], 50)
    seen = []

    def inner(estimator, Xi, yi, n_jobs):
        seen.append((Xi.index.to_numpy(), yi.copy()))
        assert len(Xi) == 80
        return np.linspace(0.1, 0.9, len(Xi))

    monkeypatch.setattr(stacking.train, "training_oof_proba", inner)
    members = {"LR": {"estimator": LogisticRegression(), "signature": "base"}}
    definitions = {"own": {"members": ["LR"], "C": 1.0},
                   "same_baseline": {"members": ["LR"], "C": 0.1}}
    result = stacking.nested_stack_oof(X, y, members, definitions, {"cache": tmp_path})
    assert len(seen) == 5  # Base fits are shared between the two meta-models.
    assert all(np.isfinite(p).all() and p.shape == y.shape for p in result.values())
    held = []
    for path in tmp_path.glob("nested_*.joblib"):
        from joblib import load
        cache = load(path)
        assert not set(cache["fit_indices"]) & set(cache["held_indices"])
        held.extend(cache["held_indices"])
    assert sorted(held) == list(range(100))
    # A second pass must not fit base models again.
    again = stacking.nested_stack_oof(X, y, members, definitions, {"cache": tmp_path})
    assert len(seen) == 5
    for key in result:
        np.testing.assert_array_equal(result[key], again[key])


def test_validation_selection_is_independent_of_test_scores(tmp_path):
    rng = np.random.default_rng(23)
    train = rng.random((80, 2))
    valid = rng.random((20, 2))
    development = {"train": train, "valid": valid, "train_y": np.tile([0, 1], 40),
                   "valid_y": np.tile([0, 1], 10), "signature": "development"}
    original_models = sensitivity.config.MODELS
    try:
        sensitivity.config.MODELS = ["LR", "RF"]
        first = sensitivity.validation_search(7, "CV17", development, {"cache": tmp_path})
        changed = {**development, "test_y": np.ones(100), "test_f1": 1.0}
        second = sensitivity.validation_search(7, "CV17", changed, {"cache": tmp_path})
        pd.testing.assert_frame_equal(first, second)
    finally:
        sensitivity.config.MODELS = original_models


def test_meta_model_scaling_uses_training_inputs_only():
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    meta = stacking.meta_model(1.0).fit(X, [0, 1, 0, 1])
    before = meta.named_steps["standardscaler"].mean_.copy()
    meta.predict_proba([[100, -100]])
    np.testing.assert_array_equal(before, meta.named_steps["standardscaler"].mean_)


def test_parallel_nested_fits_reproduce_serial_predictions(tmp_path):
    from joblib import parallel_config
    rng = np.random.default_rng(62)
    X = pd.DataFrame(rng.normal(size=(60, 3)))
    y = np.tile([0, 1], 30)
    members = {"LR": {"estimator": LogisticRegression(), "signature": "LR"},
               "NB": {"estimator": GaussianNB(), "signature": "NB"}}
    definitions = {"own": {"members": ["LR", "NB"], "C": 1.0}}
    serial = stacking.nested_stack_oof(X, y, members, definitions, {"cache": tmp_path / "serial"})
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parallel = stacking.nested_stack_oof(X, y, members, definitions,
                                             {"cache": tmp_path / "parallel"}, jobs=2)
    np.testing.assert_allclose(serial["own"], parallel["own"], rtol=0, atol=1e-12)


def test_summary_rejects_incomplete_run(tmp_path):
    utils.write_manifest(tmp_path / "run_manifest.json", complete=False)
    with pytest.raises(ValueError, match="Complete the stacking"):
        sensitivity.write_summary({"results": tmp_path})


def test_every_arm_and_threshold_is_frozen_before_test_evaluation(tmp_path, monkeypatch):
    paths = {key: tmp_path / key for key in ("results", "models", "predictions", "cache", "figures")}
    events = []
    scores = pd.DataFrame([
        {"configuration": "LR+RF;C=1", "members": "LogisticRegression,RandomForest", "C": 1.0,
         "f1_macro": 0.75, "roc_auc": 0.8, "auprc": 0.6, "brier": 0.1,
         "n": 40, "n_events": 20, "threshold": 0.5},
        {"configuration": "LR+RF+AB;C=1", "members": ",".join(sensitivity.config.PAPER_ENSEMBLE), "C": 1.0,
         "f1_macro": 0.7, "roc_auc": 0.8, "auprc": 0.6, "brier": 0.1,
         "n": 40, "n_events": 20, "threshold": 0.5}])
    monkeypatch.setattr(sensitivity, "output_paths", lambda smoke: paths)
    monkeypatch.setattr(sensitivity, "reference_scores", lambda: pd.DataFrame())
    monkeypatch.setattr(sensitivity, "development", lambda h, fs, *args: ({"signature": fs}, {}))
    monkeypatch.setattr(sensitivity, "validation_search", lambda *args: scores.copy())
    monkeypatch.setattr(sensitivity, "original_artifacts", lambda: {})
    monkeypatch.setattr(sensitivity.pred, "load_partition", lambda h, fs: {"signature": fs})

    def prepare(h, fs, data, dev, own, baseline, paper, ready, *args):
        policies = ("own", "paper") if fs == sensitivity.BASE else sensitivity.POLICIES
        for policy in policies:
            name = sensitivity.arm_name(h, fs, policy)
            events.append(("prepare", name))
            ready[name] = {"name": name, "horizon": h, "feature_set": fs, "policy": policy,
                           "configuration": own["configuration"], "members": ["LR", "RF"], "C": 1.0,
                           "signature": name, "nested_signature": name,
                           "thresholds": {mode: 0.4 for mode in stacking.MODES}}

    def evaluate(arm, *args):
        assert len(events) >= 6
        assert (paths["results"] / "frozen_selection.json").exists()
        frozen = predictive.read_json(paths["results"] / "frozen_selection.json")
        assert len(frozen["arms"]) == 6
        events.append(("test", arm["name"]))
        y = np.tile([0, 1], 20)
        p = np.where(y == 1, 0.7, 0.3)
        return pd.DataFrame({"patient_id": np.arange(40), "y_true": y,
                             **{f"proba_{mode}": p for mode in stacking.MODES}})

    monkeypatch.setattr(sensitivity, "prepare_group", prepare)
    monkeypatch.setattr(sensitivity, "evaluate_arm", evaluate)
    monkeypatch.setattr(sensitivity, "evaluate_comparisons", lambda *args: None)
    monkeypatch.setattr(sensitivity, "check_raw_reproduction", lambda *args: None)
    monkeypatch.setattr(sensitivity, "write_summary", lambda *args: None)
    sensitivity.main(["--smoke"])
    assert [kind for kind, _ in events] == ["prepare"] * 6 + ["test"] * 6
    assert predictive.read_json(paths["results"] / "run_manifest.json")["complete"]


def test_test_prediction_cache_rejects_changed_patient_order(tmp_path, monkeypatch):
    frame = pd.DataFrame({"patient_id": [2, 1], "y_true": [0, 1],
                          **{f"proba_{mode}": [0.2, 0.8] for mode in stacking.MODES}})
    paths = {"predictions": tmp_path}
    signature = "test"
    monkeypatch.setattr(sensitivity.utils, "cache_signature", lambda **parts: signature)
    monkeypatch.setattr(sensitivity.pred, "partition", lambda data, split: (
        pd.DataFrame({"Age": [40, 60]}), np.array([0, 1]), np.array([1, 2])))
    utils.write_frame(frame, tmp_path / "arm_test.csv")
    utils.write_manifest(tmp_path / "arm_test_manifest.json", signature=signature)
    with pytest.raises(AssertionError, match="identical patients"):
        sensitivity.evaluate_arm({"name": "arm", "signature": "arm"}, {}, paths)


def test_summary_and_figures_rebuild_from_frozen_outputs(tmp_path, monkeypatch):
    paths = {key: tmp_path / key for key in ("results", "predictions", "figures")}
    for path in paths.values():
        path.mkdir()
    y = np.tile([0, 1], 20)
    p = np.linspace(0.1, 0.9, len(y))
    arms, choices, metrics, deltas, curves = {}, [], [], [], []
    for fs in (sensitivity.BASE, sensitivity.config.PRIMARY_THYROID):
        name = sensitivity.arm_name(7, fs, "own")
        arm = {"horizon": 7, "feature_set": fs, "policy": "own", "configuration": "LR+RF;C=1",
               "thresholds": {mode: 0.5 for mode in stacking.MODES}}
        arms[name] = arm
        choices.append({**arm, "f1_macro": 0.7})
        frame = pd.DataFrame({"patient_id": np.arange(40), "y_true": y,
                              **{f"proba_{mode}": p for mode in stacking.MODES}})
        for split in ("valid", "test"):
            utils.write_frame(frame, paths["predictions"] / f"{name}_{split}.csv")
        for mode in stacking.MODES:
            metrics.append({**arm, "mode": mode, **utils.classification_metrics(y, p)})
            curves.append(sensitivity.train.decision_curve(y, p).assign(horizon=7, feature_set=fs,
                                                                        policy="own", mode=mode))
            if fs != sensitivity.BASE:
                deltas.append({"horizon": 7, "feature_set": fs, "policy": "own", "mode": mode,
                               "metric": "f1_macro", "delta": 0.0, "ci_lo": -0.1, "ci_hi": 0.1})
    utils.write_manifest(paths["results"] / "run_manifest.json", complete=True)
    utils.write_manifest(paths["results"] / "frozen_selection.json", arms=arms)
    for name, rows in [("test_metrics", metrics), ("paired_deltas", deltas),
                       ("selected_configurations", choices)]:
        utils.write_frame(pd.DataFrame(rows), paths["results"] / f"{name}.csv")
    utils.write_frame(pd.concat(curves), paths["results"] / "decision_curves.csv")
    monkeypatch.setattr(sensitivity, "check_raw_reproduction", lambda *args: None)
    sensitivity.write_summary(paths)
    assert (paths["results"] / "validation_metrics.csv").exists()
    assert (paths["results"] / "test_reliability.csv").exists()
    assert (paths["figures"] / "h7_stacking_calibration.png").exists()
    assert (paths["figures"] / "h7_stacking_deltas.png").exists()


def test_raw_reproduction_detects_drift_and_supports_partial_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(sensitivity.config, "RESULTS_DIR", tmp_path)
    record = {"horizon": 7, "feature_set": "CV17", "policy": "own", "mode": "raw_fixed",
              "threshold": 0.5, "f1_macro": 0.7, "roc_auc": 0.8, "auprc": 0.6, "brier": 0.1}
    reference = pd.DataFrame([record, {**record, "horizon": 10}])
    utils.write_frame(reference, tmp_path / "stacking_raw_reference.csv")
    actual = pd.DataFrame([record])
    sensitivity.check_raw_reproduction(actual, {"results": tmp_path})
    assert len(pd.read_csv(tmp_path / "reproduction_check.csv")) == 1
    actual.loc[0, "f1_macro"] += 0.01
    with pytest.raises(ValueError, match="exploratory raw"):
        sensitivity.check_raw_reproduction(actual, {"results": tmp_path})

"""Selection, leakage and cache guards for supplementary predictive analyses."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("predictive_sensitivity",
                                            ROOT / "9_predictive_sensitivity.py")
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)


def scores():
    rows = []
    for feature_set in ["CV17", "CV17_THY_CONT"]:
        for name, kind, members, f1 in [
                ("RandomForest", "single", "RandomForest", 0.73),
                ("paper", "ensemble", "LogisticRegression,RandomForest,AdaBoost", 0.72),
                ("diverse", "ensemble", "LogisticRegression,XGBoost,MLP", 0.71)]:
            rows.append({"horizon": 7, "feature_set": feature_set,
                         "configuration": name, "kind": kind, "members": members,
                         "threshold": 0.5, "n": 20, "n_events": 5,
                         "f1_macro": f1, "roc_auc": 0.8, "auprc": 0.6, "brier": 0.1})
    return pd.DataFrame(rows)


def test_selection_uses_the_fixed_validation_rule():
    table = scores().query("feature_set == 'CV17'").copy()
    table["f1_macro"] = 0.7
    table.loc[table.configuration == "paper", "roc_auc"] = 0.9
    assert sensitivity.rank_scores(table).iloc[0].configuration == "paper"
    table["roc_auc"] = 0.8
    assert sensitivity.rank_scores(table).iloc[0].configuration == "RandomForest"
    table.loc[table.configuration == "paper", "threshold"] = 0.4
    with pytest.raises(ValueError, match="fixed raw threshold"):
        sensitivity.rank_scores(table)


def test_selection_rejects_different_patients_and_nonfinite_scores():
    table = scores().query("feature_set == 'CV17'").copy()
    table.iloc[0, table.columns.get_loc("n")] = 19
    with pytest.raises(ValueError, match="same validation cohort"):
        sensitivity.rank_scores(table)
    table["n"] = 20
    table.iloc[0, table.columns.get_loc("roc_auc")] = np.nan
    with pytest.raises(ValueError, match="finite scores"):
        sensitivity.rank_scores(table)


def test_plan_is_independent_of_test_scores_and_preserves_membership():
    original = scores()
    arms, pairs, selected = sensitivity.comparison_plan(original)
    contaminated = original.assign(test_f1_macro=[1, 0, 0, 0, 1, 0])
    next_arms, next_pairs, next_selected = sensitivity.comparison_plan(contaminated)
    assert arms == next_arms and pairs == next_pairs
    pd.testing.assert_frame_equal(selected, next_selected)
    assert arms["h7_CV17_THY_CONT_RandomForest_locked"]["threshold_from"] == "h7_CV17_RandomForest"
    assert arms["h7_CV17_THY_CONT_RandomForest_locked"]["params_source"] == "CV17"
    assert arms["h7_CV17_paper"]["members"] == ["LogisticRegression", "RandomForest", "AdaBoost"]


def test_cache_is_invalidated_by_changed_signature(tmp_path):
    path = tmp_path / "member.joblib"
    sensitivity.store({"value": 1}, path, "original")
    assert sensitivity.cached(path, "original")["value"] == 1
    assert sensitivity.cached(path, "changed") is None
    assert sensitivity.cached(path, "original", force=True) is None
    path.write_bytes(b"")
    assert sensitivity.cached(path, "original") is None


def test_development_signature_does_not_depend_on_test_outcomes(monkeypatch):
    frame = pd.DataFrame({"patient_id": [1, 2, 3, 4, 5, 6],
                          "y_event": [0, 1, 0, 1, 0, 1]})
    for column in sensitivity.utils.feature_list("CV17"):
        frame[column] = np.arange(6, dtype=float)
    splits = {"train": [1, 2], "valid": [3, 4], "test": [5, 6]}
    monkeypatch.setattr(sensitivity.utils, "read_frame", lambda path: frame.copy())
    monkeypatch.setattr(sensitivity.utils, "load_splits", lambda horizon: splits)
    first = sensitivity.load_partition(7, "CV17")["signature"]
    frame.loc[frame.patient_id.isin([5, 6]), "y_event"] = [1, 0]
    assert sensitivity.load_partition(7, "CV17")["signature"] == first
    frame.loc[frame.patient_id == 1, "Age"] = 100
    assert sensitivity.load_partition(7, "CV17")["signature"] != first
    splits["valid"] = [3, 4, 5]
    with pytest.raises(AssertionError, match="share"):
        sensitivity.load_partition(7, "CV17")


def test_all_arms_are_prepared_before_any_test_evaluation(tmp_path, monkeypatch):
    paths = {key: tmp_path / key for key in ["results", "models", "predictions", "cache", "figures"]}
    table = scores()
    specs, pairs, _ = sensitivity.comparison_plan(table)
    events = []
    monkeypatch.setattr(sensitivity, "output_paths", lambda smoke: paths)
    monkeypatch.setattr(sensitivity, "validation_comparison", lambda *args: (
        table, {(7, fs): {} for fs in ["CV17", "CV17_THY_CONT"]}, {}))

    def prepare(spec, *args):
        events.append(("prepare", spec["name"]))
        return {**spec, "signature": spec["name"], "threshold": 0.4}

    def evaluate(arm, *args):
        assert len([event for event in events if event[0] == "prepare"]) == len(specs)
        assert (paths["results"] / "frozen_selection.json").exists()
        events.append(("test", arm["name"]))
        return pd.DataFrame({"patient_id": [1, 2], "y_true": [0, 1],
                             "proba_raw": [0.2, 0.8], "proba_calibrated": [0.2, 0.8]}), "test"

    monkeypatch.setattr(sensitivity, "prepare_arm", prepare)
    monkeypatch.setattr(sensitivity, "evaluate_arm", evaluate)
    monkeypatch.setattr(sensitivity, "check_original_test", lambda *args: None)
    monkeypatch.setattr(sensitivity, "paired_comparison", lambda *args: pd.DataFrame({"delta": [0.0]}))
    monkeypatch.setattr(sensitivity, "write_summary", lambda paths: None)
    monkeypatch.setattr(sensitivity, "write_figures", lambda paths: None)
    monkeypatch.setattr(sensitivity, "original_artifacts", lambda: {})
    sensitivity.main(["--smoke"])
    assert len([event for event in events if event[0] == "test"]) == len(specs)
    metrics = pd.read_csv(paths["results"] / "test_metrics.csv")
    assert metrics.loc[metrics.probabilities == "uncalibrated_fixed", "threshold_source"].eq(
        "fixed 0.5").all()


def test_paired_comparison_rejects_misaligned_patients(tmp_path):
    pair = {"base_arm": "base", "other_arm": "other"}
    base = pd.DataFrame({"patient_id": [1, 2], "y_true": [0, 1]})
    other = pd.DataFrame({"patient_id": [2, 1], "y_true": [0, 1]})
    with pytest.raises(AssertionError, match="identical patients"):
        sensitivity.paired_comparison(pair, {}, {"base": (base, "a"), "other": (other, "b")},
                                      {"cache": tmp_path}, 20)


def test_summary_rejects_an_incomplete_run(tmp_path):
    sensitivity.utils.write_manifest(tmp_path / "run_manifest.json", complete=False)
    with pytest.raises(ValueError, match="Complete the sensitivity run"):
        sensitivity.write_summary({"results": tmp_path})


def test_original_prediction_reproduction_rejects_changed_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(sensitivity.config, "PRED_DIR", tmp_path)
    arm = {"name": "h7_CV17_paper", "kind": "ensemble", "threshold_from": None,
           "threshold": 0.4}
    frame = pd.DataFrame({"patient_id": [1, 2], "y_true": [0, 1],
                          "proba_raw": [0.2, 0.8], "proba_calibrated": [0.1, 0.7]})
    sensitivity.utils.write_frame(frame, tmp_path / "h7_CV17_paper_test.csv")
    sensitivity.utils.write_manifest(tmp_path / "h7_CV17_paper_test_manifest.json", threshold=0.4)
    assert sensitivity.check_original_test(arm, frame)["proba_calibrated_max_difference"] == 0
    frame.loc[0, "proba_calibrated"] = 0.2
    with pytest.raises(ValueError, match="original frozen ensemble"):
        sensitivity.check_original_test(arm, frame)

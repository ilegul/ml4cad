"""Fixed-prediction bootstrap and point-reproduction guards for thesis CI tables."""

import numpy as np
import pandas as pd
import pytest

import table_confidence_intervals as ci
import train
import utils


def test_classification_bootstrap_keeps_threshold_and_probabilities_fixed(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A bootstrap sample reselected a threshold or refitted a calibrator")
    monkeypatch.setattr(utils, "optimize_threshold", forbidden)
    monkeypatch.setattr(train, "fit_calibrator", forbidden)
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([.1, .4, .3, .8, .6, .5])
    idx = np.array([1, 1, 2, 3, 4, 5])
    before = p.copy()
    actual = ci.classification_draw(y, p, .39, idx, ["f1_macro", "roc_auc", "auprc", "brier"])
    expected = utils.classification_metrics(y[idx], p[idx], .39)
    for metric in actual:
        assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)
    np.testing.assert_array_equal(p, before)


def test_percentiles_use_only_finite_replicates_for_each_metric():
    values = np.array([1., 2., np.nan, np.inf, 3., 4.])
    lo, hi, used = ci.percentile_interval(values)
    np.testing.assert_allclose([lo, hi], np.percentile([1., 2., 3., 4.], [2.5, 97.5]))
    assert used == 4
    assert ci.percentile_interval([np.nan])[2] == 0


def test_calibration_recomputes_metrics_without_changing_the_probability_mapping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The stored probability calibrator was refitted")
    monkeypatch.setattr(train, "fit_calibrator", forbidden)
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([.15, .23, .6, .35, .55, .65, .75, .4])
    idx = np.array([0, 1, 2, 2, 3, 4, 6, 7])
    before = p.copy()
    actual = ci.calibration_draw(y, p, idx)
    expected = train.calibration_report(y[idx], p[idx])
    for metric in actual:
        assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)
    np.testing.assert_array_equal(p, before)


def test_failed_slope_keeps_other_metrics(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Non-finite evaluation slope")
    monkeypatch.setattr(train, "calibration_report", fail)
    y = np.array([0, 1, 0, 1])
    p = np.array([.2, .4, .6, .8])
    actual = ci.calibration_draw(y, p, np.arange(4))
    assert np.isnan(actual["calibration_slope"])
    assert actual["observed_to_expected"] == pytest.approx(y.mean() / p.mean())
    assert actual["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert np.isfinite(actual["ece"])


def test_single_class_draw_is_excluded_from_classification():
    actual = ci.classification_draw(np.array([0, 1, 1]), np.array([.3, .6, .8]), .5,
                                    np.array([1, 1, 2]), ["f1_macro", "roc_auc"])
    assert all(np.isnan(value) for value in actual.values())


def test_discrepancy_blocks_ci_export_and_records_the_failed_point(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "OUTPUT", tmp_path)
    verification = []
    with pytest.raises(ValueError, match="Point estimate mismatch"):
        ci.verify_points({"table": "5.3"}, {"f1_macro": .72}, {"f1_macro": .7313},
                         ["f1_macro"], verification, .00005)
    assert verification[-1]["matches"] is False
    assert (tmp_path / "point_estimate_verification.csv").exists()
    assert not list(tmp_path.glob("*_ci.csv"))


def test_thesis_display_does_not_replace_the_verified_source_estimate():
    rows = pd.DataFrame({"table": ["5.5"], "horizon": [7], "feature_set": ["CV17_THY_CONT"],
                         "model_or_configuration": ["Reference soft voting"], "metric": ["AUROC"],
                         "estimate": [.8474765258215964], "ci_lo": [.812], "ci_hi": [.880]})
    before = rows.copy(deep=True)
    text = ci.display_ci(rows, 7, "CV17_THY_CONT", "paper", "AUROC", preserve_thesis_labels=True)
    assert text == "0.848 [0.812, 0.880]"
    assert ci.display_ci(rows, 7, "CV17_THY_CONT", "paper", "AUROC") == "0.847 [0.812, 0.880]"
    pd.testing.assert_frame_equal(before, rows)

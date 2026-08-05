"""Tests for decision-analytic summaries in src.alignment.clinical_utility."""

import unittest

import numpy as np
import pandas as pd

from src.alignment.clinical_utility import (
    calibration_curve_table,
    calibration_summary,
    clinical_impact_table,
    decision_curve,
    paired_bootstrap_delta_net_benefit,
)


class CalibrationTests(unittest.TestCase):
    def test_brier_ece_and_rates_match_hand_calculation(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])

        table = calibration_curve_table(y, p, n_bins=2)
        self.assertEqual(table["n"].tolist(), [2, 2])
        np.testing.assert_allclose(table["mean_predicted"], [0.25, 0.75])
        np.testing.assert_allclose(table["observed_rate"], [0.0, 1.0])
        np.testing.assert_allclose(table["absolute_calibration_gap"], [0.25, 0.25])

        summary = calibration_summary(y, p, n_bins=2)
        # (0.1^2 + 0.4^2 + 0.4^2 + 0.1^2) / 4
        self.assertAlmostEqual(summary["brier"], 0.085)
        self.assertAlmostEqual(summary["ece"], 0.25)
        self.assertAlmostEqual(summary["observed_rate"], 0.5)
        self.assertAlmostEqual(summary["expected_rate"], 0.5)
        self.assertAlmostEqual(summary["observed_to_expected_ratio"], 1.0)

    def test_recalibration_intercept_zero_and_slope_one_for_grouped_example(self):
        # At risk 0.2, one of five events occurs; at risk 0.8, four of five
        # events occur.  The unpenalised grouped logistic solution is exactly
        # intercept=0 and slope=1 on the prediction logit scale.
        y = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
        p = np.array([0.2] * 5 + [0.8] * 5)
        summary = calibration_summary(y, p, n_bins=2)
        self.assertAlmostEqual(summary["calibration_intercept"], 0.0, places=5)
        self.assertAlmostEqual(summary["calibration_slope"], 1.0, places=5)
        self.assertEqual(summary["recalibration_status"], "ok")

    def test_zero_one_probabilities_are_clipped_for_recalibration(self):
        # Extreme probabilities are present, but class ranges overlap so the
        # unpenalised recalibration MLE remains finite.
        summary = calibration_summary([0, 1, 0, 1], [0.0, 1.0, 0.8, 0.2])
        self.assertTrue(np.isfinite(summary["calibration_intercept"]))
        self.assertTrue(np.isfinite(summary["calibration_slope"]))

    def test_separated_recalibration_returns_nan(self):
        summary = calibration_summary([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertTrue(np.isnan(summary["calibration_intercept"]))
        self.assertTrue(np.isnan(summary["calibration_slope"]))
        self.assertEqual(
            summary["recalibration_status"], "complete_or_quasi_separation"
        )

    def test_non_identifiable_recalibration_returns_nan(self):
        constant_prediction = calibration_summary([0, 1], [0.5, 0.5])
        self.assertTrue(np.isnan(constant_prediction["calibration_intercept"]))
        self.assertTrue(np.isnan(constant_prediction["calibration_slope"]))
        self.assertEqual(
            constant_prediction["recalibration_status"], "constant_predictions"
        )

        one_class = calibration_summary([0, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(one_class["calibration_intercept"]))
        self.assertTrue(np.isnan(one_class["calibration_slope"]))
        self.assertEqual(one_class["recalibration_status"], "one_class")

    def test_zero_expected_rate_has_undefined_observed_expected_ratio(self):
        summary = calibration_summary([0, 1], [0.0, 0.0])
        self.assertEqual(summary["expected_rate"], 0.0)
        self.assertTrue(np.isnan(summary["observed_to_expected_ratio"]))

    def test_quantile_curve_handles_tied_probabilities(self):
        table = calibration_curve_table(
            [0, 1, 0, 1], [0.2, 0.2, 0.8, 0.8], n_bins=4, strategy="quantile"
        )
        self.assertEqual(int(table["n"].sum()), 4)
        self.assertTrue((table["bin_upper"] > table["bin_lower"]).all())


class DecisionCurveTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0])
        self.p = np.array([0.9, 0.8, 0.4, 0.1])

    def test_net_benefit_matches_hand_calculation(self):
        curve = decision_curve(self.y, self.p, [0.5, 0.25])

        # At 0.5: TP=1, FP=1, threshold odds=1, n=4.
        self.assertAlmostEqual(curve.loc[0, "net_benefit_model"], 0.0)
        self.assertAlmostEqual(curve.loc[0, "net_benefit_treat_all"], 0.0)
        self.assertAlmostEqual(curve.loc[0, "net_benefit_treat_none"], 0.0)

        # At 0.25: TP=2, FP=1, odds=1/3.
        self.assertAlmostEqual(curve.loc[1, "net_benefit_model"], 5.0 / 12.0)
        self.assertAlmostEqual(curve.loc[1, "net_benefit_treat_all"], 1.0 / 3.0)

    def test_paired_delta_point_estimate_and_seed_are_deterministic(self):
        comparator = np.array([0.9, 0.2, 0.4, 0.1])
        first = paired_bootstrap_delta_net_benefit(
            self.y, self.p, comparator, [0.5], n_boot=250, seed=123
        )
        second = paired_bootstrap_delta_net_benefit(
            self.y, self.p, comparator, [0.5], n_boot=250, seed=123
        )
        # Model NB=0; comparator TP=1, FP=0 -> NB=0.25.
        self.assertAlmostEqual(first.loc[0, "delta_net_benefit"], -0.25)
        pd.testing.assert_frame_equal(first, second)

    def test_identical_models_have_zero_delta_and_zero_interval(self):
        paired = paired_bootstrap_delta_net_benefit(
            self.y, self.p, self.p, [0.2, 0.5, 0.8], n_boot=100, seed=7
        )
        np.testing.assert_array_equal(paired["delta_net_benefit"], 0.0)
        np.testing.assert_array_equal(paired["delta_ci_low"], 0.0)
        np.testing.assert_array_equal(paired["delta_ci_high"], 0.0)

    def test_decision_curve_can_append_paired_comparison(self):
        result = decision_curve(
            self.y,
            self.p,
            [0.5],
            y_proba_comparator=self.p,
            n_boot=20,
            seed=9,
        )
        self.assertIn("net_benefit_comparator", result.columns)
        self.assertAlmostEqual(result.loc[0, "delta_net_benefit"], 0.0)


class ClinicalImpactTests(unittest.TestCase):
    def test_clinical_counts_and_rates_match_hand_calculation(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.4, 0.1])
        row = clinical_impact_table(y, p, [0.5]).iloc[0]

        self.assertEqual((row.tp, row.fp, row.tn, row.fn), (1, 1, 1, 1))
        self.assertAlmostEqual(row.sensitivity, 0.5)
        self.assertAlmostEqual(row.specificity, 0.5)
        self.assertAlmostEqual(row.ppv, 0.5)
        self.assertAlmostEqual(row.npv, 0.5)
        self.assertEqual(row.high_risk_n, 2)
        self.assertAlmostEqual(row.high_risk_pct, 50.0)
        self.assertEqual(row.events_detected, 1)
        self.assertEqual(row.events_missed, 1)
        self.assertAlmostEqual(row.tp_per_1000, 250.0)
        self.assertAlmostEqual(row.fp_per_1000, 250.0)
        self.assertAlmostEqual(row.high_risk_per_1000, 500.0)
        self.assertAlmostEqual(row.net_benefit, 0.0)

    def test_no_high_risk_predictions_use_nan_for_undefined_ppv(self):
        row = clinical_impact_table(
            [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], [0.95]
        ).iloc[0]
        self.assertEqual((row.tp, row.fp, row.tn, row.fn), (0, 0, 2, 2))
        self.assertAlmostEqual(row.sensitivity, 0.0)
        self.assertAlmostEqual(row.specificity, 1.0)
        self.assertTrue(np.isnan(row.ppv))
        self.assertAlmostEqual(row.npv, 0.5)
        self.assertAlmostEqual(row.net_benefit, 0.0)


class ValidationTests(unittest.TestCase):
    def test_rejects_invalid_probabilities_and_outcomes(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            calibration_summary([0, 1], [0.2, 1.1])
        with self.assertRaisesRegex(ValueError, "binary"):
            calibration_summary([0, 2], [0.2, 0.8])
        with self.assertRaisesRegex(ValueError, "equal length"):
            decision_curve([0, 1], [0.2], [0.5])

    def test_rejects_threshold_endpoints(self):
        for threshold in (0.0, 1.0):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "strictly between"):
                    clinical_impact_table([0, 1], [0.2, 0.8], [threshold])

    def test_rejects_invalid_bootstrap_configuration(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            paired_bootstrap_delta_net_benefit(
                [0, 1], [0.2, 0.8], [0.3, 0.7], [0.5], n_boot=0
            )

    def test_accepts_one_pass_iterables(self):
        summary = calibration_summary(iter([0, 1]), iter([0.2, 0.8]), n_bins=2)
        self.assertEqual(summary["n"], 2)

        curve = decision_curve(iter([0, 1]), iter([0.2, 0.8]), iter([0.5]))
        self.assertEqual(len(curve), 1)

        paired = paired_bootstrap_delta_net_benefit(
            iter([0, 1]),
            iter([0.2, 0.8]),
            iter([0.3, 0.7]),
            iter([0.5]),
            n_boot=5,
        )
        self.assertEqual(len(paired), 1)


if __name__ == "__main__":
    unittest.main()

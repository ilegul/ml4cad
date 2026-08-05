"""Smoke tests for the self-contained paper-like tuning module."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from configs.config import (
    CARDIO_17,
    FEATURE_SETS,
    LEGACY_FEATURE_SET_ALIASES,
    THYROID_RAW9,
    canonical_feature_set_name,
    feature_set_cache_id,
)
from src.alignment import paper_tuning as pt
from src.alignment.features_align import get_align_feature_set
from src.alignment.holdout_split import split_60_20_20


def _synthetic(seed=7):
    X, y = make_classification(
        n_samples=180,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        weights=[0.75, 0.25],
        class_sep=1.0,
        random_state=seed,
    )
    index = pd.Index(np.arange(1000, 1000 + len(y)), name="patient_id")
    frame = pd.DataFrame(
        X, index=index, columns=[f"x{i}" for i in range(X.shape[1])])
    target = pd.Series(y, index=index)
    return frame, target


class PaperTuningTests(unittest.TestCase):
    def _config(self, cache_dir, **kwargs):
        defaults = dict(
            n_iter=2,
            cv=2,
            n_jobs=1,
            seed=19,
            profile="smoke",
            bootstrap_repeats=50,
            cache_dir=Path(cache_dir),
        )
        defaults.update(kwargs)
        return pt.PaperTuningConfig(**defaults)

    def test_defaults_profiles_and_search_space_summary(self):
        config = pt.PaperTuningConfig()
        self.assertEqual(config.n_iter, 5000)
        self.assertEqual(config.cv, 2)
        self.assertEqual(config.scoring, "f1_macro")
        self.assertEqual(config.n_jobs, -1)
        self.assertIn("primary", pt.PAPER_PROFILES)
        self.assertIn("all_features", pt.PAPER_PROFILES)
        self.assertIn("smoke", pt.PAPER_PROFILES)

        summary = pt.paper_search_space_summary()
        self.assertEqual(set(summary["model"]), set(pt.PAPER_MODELS))
        rf = pt.get_paper_search_space("RandomForest")[0]
        # scipy randint's support starts at the first positional argument.
        self.assertEqual(rf["clf__min_samples_split"].args[0], 2)
        xgb_params = set().union(
            *(branch.keys() for branch in
              pt.get_paper_search_space("XGBoost")))
        self.assertIn("clf__learning_rate", xgb_params)
        self.assertIn("clf__reg_lambda", xgb_params)
        self.assertNotIn("clf__eta", xgb_params)

    def test_feature_set_naming_composition_and_alignment_encoding(self):
        expected = {
            "CV17": CARDIO_17,
            "CV17_THY_CONT_STATES": CARDIO_17 + THYROID_RAW9,
            "CV17_THY_ABNORMAL_BIN": CARDIO_17 + ["Thyroid_abnormal"],
            "CV17_THY_STATE_ORD": CARDIO_17 + ["thyroid_ord"],
            "CV17_THY_CONT": CARDIO_17 + ["TSH", "fT3", "fT4"],
            "CV17_THY_STATES": CARDIO_17 + [
                "Euthyroid", "SCH", "SCT", "Low_T3",
                "Hypothyroid", "Hyperthyroid",
            ],
            "CV17_THY_CONT_STATES_RATIO": (
                CARDIO_17 + THYROID_RAW9 + ["fT3_fT4_ratio"]
            ),
            "CV17_THY_CONT_RATIO": (
                CARDIO_17 + ["TSH", "fT3", "fT4", "fT3_fT4_ratio"]
            ),
        }
        self.assertEqual(list(FEATURE_SETS), list(expected))
        self.assertEqual(FEATURE_SETS, expected)
        self.assertEqual(len(FEATURE_SETS), 8)
        self.assertTrue(all(len(values) == len(set(values))
                            for values in FEATURE_SETS.values()))

        state_indicator_sets = {
            "CV17_THY_CONT_STATES",
            "CV17_THY_STATES",
            "CV17_THY_CONT_STATES_RATIO",
        }
        for name, canonical_features in FEATURE_SETS.items():
            aligned = get_align_feature_set(name)
            dropped = set(canonical_features) - set(aligned)
            expected_drop = {"Euthyroid"} if name in state_indicator_sets else set()
            self.assertEqual(dropped, expected_drop, name)

        all_profile = pt.PAPER_PROFILES["all_features"]["feature_sets"]
        self.assertEqual(all_profile, tuple(expected))
        self.assertTrue(set(LEGACY_FEATURE_SET_ALIASES).isdisjoint(FEATURE_SETS))
        for legacy, canonical in LEGACY_FEATURE_SET_ALIASES.items():
            self.assertEqual(canonical_feature_set_name(legacy), canonical)
        self.assertEqual(
            feature_set_cache_id("CV17_THY_CONT_STATES"), "CV17_THY26"
        )

    def test_per_job_cache_signature_and_deterministic_outputs(self):
        X, y = _synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp)
            first = pt.run_paper_tuning_job(
                X, y, 7, "SYNTH", "LogisticRegression", config,
                force=True)
            cached = pt.run_paper_tuning_job(
                X, y, 7, "SYNTH", "LogisticRegression", config,
                force=False)
            repeated = pt.run_paper_tuning_job(
                X, y, 7, "SYNTH", "LogisticRegression", config,
                force=True)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(first["signature"], cached["signature"])
            np.testing.assert_allclose(
                first["test_proba"], cached["test_proba"], rtol=0, atol=0)
            np.testing.assert_allclose(
                first["test_proba"], repeated["test_proba"],
                rtol=0, atol=1e-12)

            changed_config = replace(config, n_iter=3)
            sig_changed_config = pt.paper_job_signature(
                X, y, 7, "SYNTH", "LogisticRegression", changed_config)
            self.assertNotEqual(first["signature"], sig_changed_config)

            X_changed = X.copy()
            X_changed.iloc[0, 0] += 0.25
            sig_changed_data = pt.paper_job_signature(
                X_changed, y, 7, "SYNTH", "LogisticRegression", config)
            self.assertNotEqual(first["signature"], sig_changed_data)
            self.assertTrue(Path(first["cache_joblib"]).exists())
            self.assertTrue(Path(first["cache_json"]).exists())

    def test_legacy_feature_set_cache_is_reused_without_refit(self):
        X, y = _synthetic(seed=23)
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, n_iter=1)
            legacy = pt.run_paper_tuning_job(
                X, y, 7, "CV17_THY26", "LogisticRegression", config,
                force=True,
            )
            with patch.object(
                pt.RandomizedSearchCV,
                "fit",
                side_effect=AssertionError("cache miss caused an unexpected refit"),
            ):
                canonical = pt.run_paper_tuning_job(
                    X, y, 7, "CV17_THY_CONT_STATES",
                    "LogisticRegression", config, force=False,
                )

            self.assertTrue(canonical["cache_hit"])
            self.assertEqual(canonical["feature_set"], "CV17_THY_CONT_STATES")
            self.assertEqual(legacy["signature"], canonical["signature"])
            self.assertEqual(legacy["cache_joblib"], canonical["cache_joblib"])
            self.assertIn("CV17_THY26", Path(canonical["cache_joblib"]).name)

    def test_random_search_fit_never_receives_validation_or_test(self):
        X, y = _synthetic(seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, n_iter=1)
            _, _, expected_test = split_60_20_20(y, seed=config.seed)
            expected_test_labels = set(X.index[expected_test])
            seen_fit_labels = []
            original_fit = pt.RandomizedSearchCV.fit

            def spy_fit(search, X_fit, y_fit=None, **fit_params):
                seen_fit_labels.append(set(X_fit.index))
                return original_fit(search, X_fit, y_fit, **fit_params)

            with patch.object(pt.RandomizedSearchCV, "fit", new=spy_fit):
                artifact = pt.run_paper_tuning_job(
                    X, y, 7, "SYNTH", "LogisticRegression", config,
                    force=True)

            self.assertEqual(len(seen_fit_labels), 1)
            self.assertTrue(seen_fit_labels[0].isdisjoint(
                expected_test_labels))
            self.assertEqual(
                seen_fit_labels[0], set(artifact["index_train"]))
            self.assertTrue(set(artifact["index_validation"]).isdisjoint(
                seen_fit_labels[0]))

    def test_cache_only_loader_never_starts_a_search(self):
        rng = np.random.default_rng(4)
        features = list(dict.fromkeys(
            get_align_feature_set("CV17")
            + get_align_feature_set("CV17_THY_CONT_STATES")))
        cohort = pd.DataFrame(
            rng.normal(size=(120, len(features))), columns=features)
        cohort["y7"] = np.array([0] * 84 + [1] * 36)
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, n_iter=1)
            with patch.object(
                    pt.RandomizedSearchCV, "fit",
                    side_effect=AssertionError("cache loader attempted fitting")):
                loaded = pt.load_cached_paper_tuning({7: cohort}, config)
        self.assertEqual(loaded.status, "cache_missing")
        self.assertTrue(loaded.search_results.empty)
        self.assertIn("No complete", loaded.message)

    def test_dynamic_sampling_ratios_and_paired_bootstrap_are_deterministic(self):
        # Existing minority/majority ratio is 0.60, therefore none of the
        # paper's 0.20..0.50 under-ratios is feasible; no-under remains.
        y_high = np.array([0] * 100 + [1] * 60)
        self.assertEqual(pt.valid_paper_under_ratios(y_high), tuple())
        candidates = pt.paper_sampling_candidates(y_high)
        self.assertEqual({c["under_ratio"] for c in candidates}, {None})
        self.assertEqual(len(candidates), 9)  # 3 samplers x 3 k values

        rng = np.random.default_rng(2)
        y = np.array([0] * 80 + [1] * 40)
        base = np.clip(0.25 + 0.35 * y + rng.normal(0, 0.15, len(y)), 0, 1)
        thy = np.clip(base + rng.normal(0, 0.03, len(y)), 0, 1)
        a = pt.paired_incremental_bootstrap(
            y, base, thy, n_boot=50, seed=3)
        b = pt.paired_incremental_bootstrap(
            y, base, thy, n_boot=50, seed=3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

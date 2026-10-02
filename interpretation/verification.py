"""Reconstruct frozen probabilities and validate stored explanatory outputs."""

import json

import numpy as np
import pandas as pd

import config
import utils
from . import classification as shared
from . import importance


def verify_group(horizon, feature_set, initialise_manifest=False):
    registry, training, train_ids, test, outcomes, ids = shared.setup(horizon, feature_set)
    probabilities = registry(test.to_numpy(dtype=float))
    source_error = max([float(np.max(np.abs(probabilities[:, j] - expected)))
                        for j, expected in registry.checks] or [0.])
    if source_error > 1e-8:
        raise ValueError('Frozen predictions differ from the original saved outputs')
    rows = []
    for stage in ("shap", "importance"):
        folder = shared.RESULTS / stage / "groups" / f"h{horizon}_{feature_set}"
        if initialise_manifest:
            shared.write_provenance(folder, registry, registry.data, registry.development,
                                    registry.source_paths)
        manifest = shared.check_provenance(folder)
        if manifest['partition_signature'] != registry.data['signature']:
            raise ValueError('Partition signature changed')
        info = json.loads((folder / "complete.json").read_text())
        if stage == "shap":
            patients = pd.read_csv(folder / "patients.csv")
            background = pd.read_csv(folder / "background_patients.csv")
            selected = pd.Index(ids).get_indexer(patients.patient_id)
            bg = pd.Index(train_ids).get_indexer(background.patient_id)
            if (selected < 0).any() or (bg < 0).any():
                raise ValueError("Explanation samples are not in their assigned partitions")
            if not np.array_equal(outcomes[selected], patients.y_true):
                raise ValueError("Explanation outcomes changed")
            rng = np.random.default_rng(shared.SHAP_SEED + horizon)
            if not np.array_equal(bg, rng.choice(len(training), info['background'], replace=False)):
                raise ValueError('Training background does not match the documented sample')
            if not np.array_equal(selected, rng.choice(len(test), info['patients'], replace=False)):
                raise ValueError('Test sample does not match the documented sample')
            outputs = pd.read_csv(folder / "outputs.csv")
            pd.testing.assert_frame_equal(outputs[['arm', 'kind', 'members', 'scale']],
                                          pd.DataFrame(registry.meta)[['arm', 'kind', 'members', 'scale']])
            with np.load(folder / "attributions.npz", allow_pickle=False) as values:
                if list(values['feature_names']) != list(test.columns):
                    raise ValueError('Explanation predictors changed')
                np.testing.assert_allclose(values['features'], test.iloc[selected], rtol=0, atol=0)
                probability_error = float(np.max(np.abs(values['predictions'] - probabilities[selected])))
                additivity_error = float(np.max(np.abs(values['base_values'] +
                                                       values['values'].sum(axis=1) - values['predictions'])))
                background_error = float(np.max(np.abs(values['base_values'] -
                    registry(training.iloc[bg].to_numpy(dtype=float)).mean(axis=0))))
                if not np.isfinite(values['values']).all():
                    raise ValueError('Non-finite SHAP attributions')
        else:
            with np.load(folder / "reference.npz", allow_pickle=False) as reference:
                utils.assert_same_patients(ids, reference['patient_ids'])
                np.testing.assert_array_equal(outcomes, reference['y_true'])
                np.testing.assert_allclose(training.mean(numeric_only=True), reference['training_means'],
                                           rtol=0, atol=1e-12)
                probability_error = float(np.max(np.abs(reference['probabilities'] - probabilities)))
                rng = np.random.default_rng(config.SEED)
                orders = [rng.permutation(len(outcomes)) for _ in range(info['permutation_repeats'])]
                np.testing.assert_array_equal(orders, reference['permutation_orders'])
            cases = importance.cases_for(horizon, feature_set, registry)
            saved = pd.read_csv(folder / 'configurations.csv')
            pd.testing.assert_frame_equal(cases[saved.columns], saved, check_dtype=False,
                                          check_exact=False, rtol=0, atol=1e-12)
            scores, _ = importance.scores(outcomes, probabilities, cases)
            baseline = pd.read_csv(folder / 'baseline_metrics.csv')
            for metric, values in scores.items():
                np.testing.assert_allclose(values, baseline[metric], rtol=0, atol=1e-12)
            additivity_error, background_error = 0., 0.
        if max(probability_error, additivity_error, background_error) > 1e-8:
            raise ValueError('Frozen explanations no longer reproduce their predictive sources')
        rows.append({'horizon': horizon, 'feature_set': feature_set, 'stage': stage,
                     'outputs': len(registry.meta), 'probability_max_error': probability_error,
                     'original_prediction_max_error': source_error,
                     'additivity_max_error': additivity_error,
                     'background_mean_max_error': background_error})
    shared.log(f"Verified h{horizon} {feature_set}: {len(registry.meta)} probability outputs")
    return rows

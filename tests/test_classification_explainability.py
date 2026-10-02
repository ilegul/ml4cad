"""Numerical and read-only guards for the classification interpretation extension."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

import config
import predictive
import utils
from interpretation import classification, importance


@pytest.mark.parametrize('probability_kind', ['continuous', 'ties', 'constant'])
def test_vectorised_metrics_match_sklearn(probability_kind):
    rng = np.random.default_rng(12)
    y = np.tile([0, 0, 1], 11)
    p = rng.random((len(y), 4))
    if probability_kind == 'ties':
        p = np.round(p, 1)
    elif probability_kind == 'constant':
        p[:] = [.0, .3, .5, 1.]
    cases = pd.DataFrame({'probability_index': [0, 1, 2, 3, 2],
                          'threshold': [.5, .2, .5, .8, .9]})
    actual, predictions = importance.scores(y, p, cases)
    for i, case in cases.iterrows():
        probabilities = p[:, int(case.probability_index)]
        assert actual['f1_macro'][i] == pytest.approx(f1_score(y, predictions[:, i], average='macro'))
        assert actual['roc_auc'][i] == pytest.approx(roc_auc_score(y, probabilities))
        assert actual['auprc'][i] == pytest.approx(average_precision_score(y, probabilities))
        assert actual['brier'][i] == pytest.approx(brier_score_loss(y, probabilities))


def test_bootstrap_reproduces_patient_draws_and_excludes_single_class(monkeypatch):
    monkeypatch.setattr(config, 'BOOTSTRAP', 50)
    y = np.array([0, 0, 1])
    predictions = np.array([[0, 1], [0, 0], [1, 0]], dtype=bool)
    weights, positive = importance.bootstrap_weights(y)
    assert np.all((positive > 0) & (positive < len(y)))
    actual = importance.bootstrap_f1(weights, positive, predictions, y)
    draws = [idx for idx in utils.bootstrap_indices(len(y), 50, config.SEED)
             if len(np.unique(y[idx])) == 2]
    expected = [[f1_score(y[idx], predictions[idx, j], average='macro')
                 for j in range(2)] for idx in draws]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_interpretation_rejects_missing_or_stale_cache_without_fitting(tmp_path):
    original_loader = predictive.cached
    with pytest.raises(ValueError, match='missing or stale'):
        classification.required_cache(tmp_path / 'missing.joblib', 'expected')
    predictive.store({'value': 1}, tmp_path / 'old.joblib', 'old')
    with pytest.raises(ValueError, match='missing or stale'):
        classification.required_cache(tmp_path / 'old.joblib', 'expected')
    assert predictive.cached is original_loader


def test_expected_predictions_require_exact_unique_patient_coverage(tmp_path):
    path = tmp_path / 'predictions.csv'
    pd.DataFrame({'patient_id': [8, 3], 'p': [.7, .2]}).to_csv(path, index=False)
    assert classification.expected_frame(path, [3, 8]).p.tolist() == [.2, .7]
    with pytest.raises(ValueError, match='cover'):
        classification.expected_frame(path, [3, 9])
    pd.DataFrame({'patient_id': [3, 3], 'p': [.7, .2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='cover'):
        classification.expected_frame(path, [3, 3])


def test_exact_mask_replay_matches_direct_shap_for_multiple_probability_outputs():
    class Predictor:
        outputs = [None, None]

        def __call__(self, values):
            z = np.asarray(values) @ np.array([.4, -.8, 1.2])
            p = 1. / (1. + np.exp(-z))
            return np.column_stack([p, p * p])

    background = np.array([[0., 1., 0.], [1., 0., 1.]])
    patients = np.array([[.3, .4, .7], [1., 0., 0.]])
    args = SimpleNamespace(background=2, cycles=2)
    predictor = Predictor()
    replay, info = classification.batched_explanation(predictor, background, patients,
                                                       ['a', 'b', 'c'], args, 42)
    direct = shap.PermutationExplainer(predictor, shap.maskers.Independent(background),
                                      feature_names=['a', 'b', 'c'], seed=42)(
        patients, max_evals=14, error_bounds=True, batch_size=4096, silent=True)
    np.testing.assert_allclose(replay.values, direct.values, rtol=0, atol=1e-14)
    np.testing.assert_allclose(replay.error_std, direct.error_std, rtol=0, atol=1e-14)
    np.testing.assert_allclose(replay.base_values + replay.values.sum(axis=1),
                               predictor(patients), rtol=0, atol=1e-14)
    assert info['unique_masked_rows'] <= info['planned_rows']


def test_scores_reject_undefined_discrimination():
    with pytest.raises(ValueError, match='both binary'):
        importance.scores(np.zeros(3), np.full((3, 1), .5),
                          pd.DataFrame({'probability_index': [0], 'threshold': [.5]}))


def test_matched_selection_keeps_members_but_reselects_C_on_validation(tmp_path, monkeypatch):
    root = tmp_path / 'results'
    extension = root / 'classification_explainability'
    extension.mkdir(parents=True)
    (root / 'stacking_sensitivity').mkdir()
    rows = []
    for fs in ('CV17', 'CV17_THY_CONT'):
        for C in (.01, .1, 1., 10., 100.):
            winner = C == (.01 if fs == 'CV17' else 1.)
            rows.append({'horizon': 7, 'feature_set': fs, 'members': 'A,B', 'C': C,
                         'configuration': f'A+B;C={C:g}', 'n': 20, 'n_events': 5,
                         'threshold': .5, 'f1_macro': .8 if winner else .6,
                         'roc_auc': .8, 'auprc': .6, 'brier': .1})
    # A freely selected different composition cannot enter the fixed-members family.
    rows.append({**rows[-1], 'members': 'C,D', 'configuration': 'C+D;C=100', 'f1_macro': .99})
    scores = pd.DataFrame(rows)
    scores.to_csv(root / 'stacking_sensitivity/validation_scores.csv', index=False)
    chosen = scores[scores.feature_set.eq('CV17_THY_CONT') & scores.C.eq(1.)].copy()
    chosen['baseline_C'] = .01
    selection = extension / 'matched_C_validation_selection.csv'
    chosen.to_csv(selection, index=False)
    monkeypatch.setattr(config, 'RESULTS_DIR', root)
    monkeypatch.setattr(classification, 'RESULTS', extension)
    actual = classification.matched_C_selection(7, 'CV17_THY_CONT')
    assert actual.members == 'A,B'
    assert actual.C == 1.
    assert actual.baseline_C == .01
    chosen['C'] = .01
    chosen.to_csv(selection, index=False)
    with pytest.raises(ValueError, match='differs'):
        classification.matched_C_selection(7, 'CV17_THY_CONT')

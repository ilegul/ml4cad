"""Full-test training-mean ablation and joint permutation importance."""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

import config
import predictive as pred
import utils
from . import classification as shared
from .classification import log

ROOT = config.ROOT
OUT = shared.RESULTS / "importance"


def cases_for(h, fs, registry):
    data = pred.load_partition(h, fs)
    Xv, yv, _ = pred.partition(data, 'valid')
    valid = registry(Xv.to_numpy(dtype=float))
    rows = []
    for j, meta in enumerate(registry.meta):
        source = str(meta['source'])
        scale = meta['scale']
        modes = ['raw_fixed', 'raw_threshold'] if scale == 'raw' else [scale]
        for mode in modes:
            threshold_source = 'fixed 0.5'
            threshold = .5
            if mode != 'raw_fixed':
                if meta['kind'] == 'individual':
                    threshold = utils.optimize_threshold(yv, valid[:, j])
                    threshold_source = 'own validation'
                elif source.startswith('cache/classification_explainability/'):
                    prepared = joblib.load(ROOT / source)
                    selection = pd.read_csv(shared.RESULTS / 'matched_C_validation_selection.csv')
                    C = float(selection[(selection.horizon == h) & selection.feature_set.eq(fs)].iloc[0].C)
                    threshold = prepared['C_specs'][str(C)]['thresholds'][mode]
                    threshold_source = 'saved reselected-C validation threshold'
                elif meta['kind'] == 'stacking':
                    manifest = json.loads((ROOT / source.replace('.joblib', '_manifest.json')).read_text())
                    threshold = manifest['thresholds'][mode]
                    threshold_source = 'saved stacking threshold'
                else:
                    # Voting raw sensitivity keeps the saved calibrated threshold,
                    # matching the original notebook's raw-probability evaluation.
                    manifest = json.loads((ROOT / source.replace('.joblib', '_manifest.json')).read_text())
                    if 'threshold' not in manifest:
                        name = Path(source).stem.removeprefix('final_')
                        manifest = json.loads((ROOT / f'predictions/{name}_test_manifest.json').read_text())
                    threshold = manifest['threshold']
                    threshold_source = 'saved voting threshold (also applied to raw)'
            rows.append({**meta, 'mode': mode, 'probability_index': j, 'threshold': threshold,
                         'threshold_source': threshold_source, 'horizon': h, 'feature_set': fs})
    return pd.DataFrame(rows)


def targets_for(Xt):
    features = list(Xt.columns)
    correlation = Xt.corr(method='spearman').fillna(0.)
    distance = 1. - correlation.abs()
    np.fill_diagonal(distance.values, 0.)
    labels = fcluster(linkage(squareform(distance.values, checks=False), method='average'),
                      t=config.N_FEATURE_CLUSTERS, criterion='maxclust')
    targets = [{'target': f, 'target_kind': 'single variable', 'columns': [f]} for f in features]
    for label in sorted(set(labels)):
        members = [f for f, group in zip(features, labels) if group == label]
        targets.append({'target': f'cluster {label}: ' + ', '.join(members),
                        'target_kind': 'cluster', 'columns': members})
    blocks = {'complete thyroid block': [f for f in features if f not in config.CARDIO],
              'continuous thyroid biomarkers': [f for f in config.THYROID_CONT if f in features],
              'thyroid state indicators': [f for f in config.THYROID_STATES if f in features],
              'thyroid ratio': [f for f in features if f == 'fT3_fT4_ratio'],
              'all cardiovascular variables': [f for f in features if f in config.CARDIO]}
    targets += [{'target': name, 'target_kind': 'block', 'columns': cols}
                for name, cols in blocks.items() if cols]
    for target in targets:
        target['n_variables'] = len(target['columns'])
        is_cardio = [c in config.CARDIO for c in target['columns']]
        target['block'] = 'cardiovascular' if all(is_cardio) else 'thyroid' if not any(is_cardio) else 'mixed'
    return targets, pd.DataFrame({'feature': features, 'cluster': labels})


def macro_f1(tp, fp, fn, tn):
    tp, fp, fn, tn = [np.asarray(value, dtype=np.float64) for value in (tp, fp, fn, tn)]
    a, b = 2 * tp + fp + fn, 2 * tn + fp + fn
    return np.divide(tp, a, out=np.zeros_like(tp, dtype=float), where=a != 0) + np.divide(
        tn, b, out=np.zeros_like(tn, dtype=float), where=b != 0)


def scores(y, probabilities, cases):
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Discrimination requires both binary outcome classes')
    if len(probabilities) != len(y) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid probability matrix')
    order = np.argsort(-probabilities, axis=0, kind='stable')
    sorted_p = np.take_along_axis(probabilities, order, axis=0)
    labels = y[order]
    n = len(y)
    index = np.arange(n)[:, None]
    ends = np.concatenate([sorted_p[:-1] != sorted_p[1:], np.ones((1, probabilities.shape[1]), bool)])
    starts = np.concatenate([np.ones((1, probabilities.shape[1]), bool), sorted_p[:-1] != sorted_p[1:]])
    last = np.minimum.accumulate(np.where(ends, index, n)[::-1], axis=0)[::-1]
    first = np.maximum.accumulate(np.where(starts, index, 0), axis=0)
    ranks = n - (first + last) / 2.
    npos, nneg = y.sum(), n - y.sum()
    auc = ((ranks * labels).sum(axis=0) - npos * (npos + 1) / 2.) / (npos * nneg)
    precision = labels.cumsum(axis=0) / (index + 1)
    ap = (np.take_along_axis(precision, last, axis=0) * labels).sum(axis=0) / npos
    brier = ((probabilities - y[:, None]) ** 2).mean(axis=0)
    predictions = probabilities[:, cases.probability_index] >= cases.threshold.to_numpy()[None, :]
    tp = (predictions & y[:, None].astype(bool)).sum(axis=0)
    fp = predictions.sum(axis=0) - tp
    fn, tn = npos - tp, nneg - fp
    return {'f1_macro': macro_f1(tp, fp, fn, tn), 'roc_auc': auc[cases.probability_index],
            'auprc': ap[cases.probability_index], 'brier': brier[cases.probability_index]}, predictions


def bootstrap_weights(y):
    draws = utils.bootstrap_indices(len(y), config.BOOTSTRAP, config.SEED)
    weights = np.asarray([np.bincount(idx, minlength=len(y)) for idx in draws], dtype=np.float32)
    positive = weights @ y.astype(np.float32)
    keep = (positive > 0) & (positive < len(y))
    return weights[keep], positive[keep, None]


def bootstrap_f1(weights, positive, predictions, y):
    pred = predictions.astype(np.float32)
    predicted_positive = weights @ pred
    tp = weights @ (pred * y[:, None])
    fp, fn = predicted_positive - tp, positive - tp
    tn = len(y) - positive - fp
    return macro_f1(tp, fp, fn, tn)


def predict_variants(registry, inputs, targets, operations, folder):
    """Deduplicate perturbed input rows across targets and permutation repeats."""
    keys, unique, positions = {}, [], []
    for target, repeat in operations:
        modified = inputs.copy()
        columns = [registry.columns.index(c) for c in target['columns']]
        if repeat < 0:
            modified[:, columns] = target['means']
        else:
            order = target['orders'][repeat]
            modified[:, columns] = modified[order][:, columns]
        indices = []
        for row in modified:
            key = row.tobytes()
            if key not in keys:
                keys[key] = len(unique)
                unique.append(row.copy())
            indices.append(keys[key])
        positions.append(indices)
    log(f'{folder.name}: {len(operations)} perturbations, {len(unique)} unique input rows')
    # Keep memory bounded while still amortising pipeline prediction overhead.
    unique = np.asarray(unique)
    probabilities = np.concatenate([registry(unique[i:i + 32768]) for i in range(0, len(unique), 32768)])
    return probabilities, positions


def run_group(h, fs, repeats, refresh=False):
    folder = OUT / 'groups' / f'h{h}_{fs}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists() and not refresh:
        shared.check_provenance(folder)
        info = json.loads((folder / 'complete.json').read_text())
        if info['permutation_repeats'] != repeats or info['bootstrap'] != config.BOOTSTRAP:
            raise ValueError('Completed importance results use different resampling settings')
        log(f'{folder.name}: completed cache retained')
        return
    started = time.time()
    registry, Xt, it, Xe, y, ids = shared.setup(h, fs)
    shared.write_provenance(folder, registry, registry.data, registry.development, registry.source_paths)
    cases = cases_for(h, fs, registry)
    cases.to_csv(folder / 'configurations.csv', index=False)
    inputs = Xe.to_numpy(dtype=float)
    base = registry(inputs)
    error = max([float(np.max(np.abs(base[:, j] - p))) for j, p in registry.checks] or [0])
    if error > 1e-8:
        raise ValueError('Frozen test probabilities not reproduced')
    baseline, base_pred = scores(y, base, cases)
    # Verify the vectorised metrics, including tied scores, against sklearn.
    for k, row in cases.iterrows():
        p = base[:, row.probability_index]
        expected = [f1_score(y, base_pred[:, k], average='macro'), roc_auc_score(y, p),
                    average_precision_score(y, p), brier_score_loss(y, p)]
        actual = [baseline[m][k] for m in ('f1_macro', 'roc_auc', 'auprc', 'brier')]
        if not np.allclose(actual, expected, atol=1e-10, rtol=0):
            raise ValueError('Vectorised metrics differ from sklearn')
    targets, clusters = targets_for(Xt)
    clusters.to_csv(folder / 'clusters.csv', index=False)
    rng = np.random.default_rng(config.SEED)
    orders = np.asarray([rng.permutation(len(y)) for _ in range(repeats)])
    means = Xt.mean(numeric_only=True)
    for target in targets:
        target['means'] = means[target['columns']].to_numpy()
        target['orders'] = orders
    np.savez_compressed(folder / 'reference.npz', probabilities=base, patient_ids=ids, y_true=y,
                        training_means=means.to_numpy(), feature_names=np.asarray(Xe.columns, dtype=str),
                        permutation_orders=orders)
    weights, positive = bootstrap_weights(y)
    base_boot = bootstrap_f1(weights, positive, base_pred, y)
    baseline_frame = cases.copy()
    for metric, values in baseline.items():
        baseline_frame[metric] = values
    baseline_frame.to_csv(folder / 'baseline_metrics.csv', index=False)
    operations = [(target, -1) for target in targets] + [(target, r) for target in targets for r in range(repeats)]
    probabilities, positions = predict_variants(registry, inputs, targets, operations, folder)
    ablation, permutations = [], []
    for (target, repeat), indices in zip(operations, positions):
        altered = probabilities[indices]
        actual, pred_altered = scores(y, altered, cases)
        metadata = {key: target[key] for key in ('target', 'target_kind', 'block', 'n_variables')}
        frame = cases.assign(**metadata)
        for metric in baseline:
            frame[f'baseline_{metric}'] = baseline[metric]
            frame[f'altered_{metric}'] = actual[metric]
            frame[f'{metric}_drop'] = ((actual[metric] - baseline[metric]) if metric == 'brier'
                                     else baseline[metric] - actual[metric])
        if repeat < 0:
            altered_boot = bootstrap_f1(weights, positive, pred_altered, y)
            diffs = base_boot - altered_boot
            ratio = np.divide(base_boot, altered_boot, out=np.full_like(base_boot, np.nan), where=altered_boot > 0)
            frame['f1_drop_ci_lo'], frame['f1_drop_ci_hi'] = np.percentile(diffs, [2.5, 97.5], axis=0)
            frame['importance_ratio'] = baseline['f1_macro'] / actual['f1_macro']
            frame['importance_ratio_ci_lo'], frame['importance_ratio_ci_hi'] = np.nanpercentile(ratio, [2.5, 97.5], axis=0)
            frame['bootstrap_valid'] = len(weights)
            ablation.append(frame)
        else:
            frame['repeat'] = repeat + 1
            permutations.append(frame)
    ablation = pd.concat(ablation, ignore_index=True)
    permutations = pd.concat(permutations, ignore_index=True)
    ablation.to_csv(folder / 'ablation.csv', index=False)
    permutations.to_csv(folder / 'permutation_repeats.csv.gz', index=False)
    columns = list(cases.columns) + list(metadata)
    grouped = permutations.groupby(columns, dropna=False)
    summary = grouped[[m + '_drop' for m in baseline]].agg(['mean', lambda x: np.std(x, ddof=0)]).reset_index()
    summary.columns = [a if not b else a.replace('_drop', '') + '_drop_' + ('std' if b != 'mean' else 'mean')
                       for a, b in summary.columns]
    summary.to_csv(folder / 'permutation.csv', index=False)
    info = {'horizon': h, 'feature_set': fs, 'n_test': len(y), 'n_events': int(y.sum()),
            'probability_outputs': len(registry.outputs), 'evaluation_cases': len(cases), 'targets': len(targets),
            'permutation_repeats': repeats, 'bootstrap': config.BOOTSTRAP,
            'probability_reproduction_max_error': error, 'seconds': time.time() - started}
    (folder / 'complete.json').write_text(json.dumps(info, indent=2))
    log(f'{folder.name}: complete, {len(cases)} cases and {len(targets)} targets, {info["seconds"]:.1f}s')
    gc.collect()

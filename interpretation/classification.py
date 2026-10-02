"""Frozen predictive functions and exact batched SHAP evaluation."""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import config
import predictive as pred
import stacking
import train
import utils

ROOT = config.ROOT
RESULTS = config.RESULTS_DIR / "classification_explainability"
CACHE = config.CACHE_DIR / "classification_explainability"
PREDICTIONS = config.PRED_DIR / "classification_explainability"
SHAP_SEED = 20261002
VERSION = "v1"


def log(message):
    pred.log(message)


def required_cache(path, signature):
    """Interpretation never falls back to fitting a missing predictive pipeline."""
    value = pred.cached(path, signature)
    if value is None:
        raise ValueError(f"Required fitted cache missing or stale: {path}. "
                         "Complete the predictive analyses before interpretation.")
    return value


def load_development(horizon, feature_set):
    data = pred.load_partition(horizon, feature_set)
    scores = pd.read_csv(config.RESULTS_DIR / "predictive_sensitivity/validation_scores.csv")
    rows = scores[(scores.horizon == horizon) & scores.feature_set.eq(feature_set)
                  & scores.kind.eq("single")]
    if len(rows) != len(config.MODELS) or set(rows.configuration) != set(config.MODELS):
        raise ValueError("Incomplete individual-classifier development scores")
    definitions = {m: pred.member_spec(horizon, feature_set, m,
                   rows[rows.configuration.eq(m)].iloc[0].samplers) for m in config.MODELS}
    signature = utils.cache_signature(stage="stacking_development", version=stacking.VERSION,
                                      development=data["signature"], definitions=definitions)
    path = config.CACHE_DIR / "stacking_sensitivity" / f"development_{signature}.joblib"
    development = required_cache(path, signature)
    for split in ("train", "valid"):
        _, y, ids = pred.partition(data, split)
        utils.assert_same_patients(ids, development[f"{split}_ids"])
        if not np.array_equal(y, development[f"{split}_y"]):
            raise ValueError("Cached development outcomes do not match")
        if not np.isfinite(development[split]).all():
            raise ValueError("Invalid cached OOF or validation probabilities")
    for m, definition in definitions.items():
        if development["members"][m]["spec"] != definition:
            raise ValueError("Cached classifier specification does not match")
    return data, development, path


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(folder, registry, data, development, paths):
    sources = sorted(set(paths + [ROOT / meta["source"] for meta in registry.meta
                                 if meta["kind"] != "individual"]))
    records = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size, "sha256": file_digest(path)} for path in sources]
    metadata = {"version": VERSION, "development_signature": development["signature"],
                "partition_signature": data["signature"], "sources": records,
                "members": {m: item["spec"] for m, item in development["members"].items()}}
    (folder / "source_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def check_provenance(folder):
    manifest = json.loads((folder / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError("Interpretation implementation version changed")
    for source in manifest["sources"]:
        path = ROOT / source["path"]
        if not path.exists() or file_digest(path) != source["sha256"]:
            raise ValueError(f"Interpretation source changed: {path}")
    return manifest


def write_inventory():
    """Inventory extension outputs without rewriting original artifact checksums."""
    roots = (RESULTS, CACHE, PREDICTIONS)
    destination = RESULTS / 'artifact_inventory.csv'
    paths = sorted(path for root in roots for path in root.rglob('*')
                   if path.is_file() and path != destination and 'pilot_batched' not in path.parts)
    records = [{'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                'size_bytes': path.stat().st_size, 'sha256': file_digest(path)} for path in paths]
    pd.DataFrame(records).to_csv(destination, index=False)
    original = config.RESULTS_DIR / 'model_cache_inventory.csv'
    if original.exists():
        inventory = pd.read_csv(original)
        added = pd.DataFrame([row for row in records if row['path'].startswith('cache/')])
        if not added.empty:
            inventory = inventory[~inventory.path.isin(added.path)]
            pd.concat([inventory, added], ignore_index=True).to_csv(original, index=False)


def matched_C_selection(horizon, feature_set):
    """Validate the additional fixed-members selection against validation only."""
    scores = pd.read_csv(config.RESULTS_DIR / 'stacking_sensitivity/validation_scores.csv')
    baseline = pred.rank_scores(scores[scores.horizon.eq(horizon) &
                                      scores.feature_set.eq(config.PRIMARY_BASELINE)]).iloc[0]
    candidates = scores[scores.horizon.eq(horizon) & scores.feature_set.eq(feature_set)
                        & scores.members.eq(baseline.members)]
    if len(candidates) != len(stacking.REGULARIZATION) or set(candidates.C) != set(stacking.REGULARIZATION):
        raise ValueError('Incomplete fixed-members regularization search')
    selected = pred.rank_scores(candidates).iloc[0]
    saved = pd.read_csv(RESULTS / 'matched_C_validation_selection.csv')
    rows = saved[saved.horizon.eq(horizon) & saved.feature_set.eq(feature_set)]
    if len(rows) != 1:
        raise ValueError('Expected one fixed-members validation selection')
    row = rows.iloc[0]
    if row.members != selected.members or row.C != selected.C or row.baseline_C != baseline.C:
        raise ValueError('Fixed-members selection differs from stored validation scores')
    pred.check_scores(row, selected, 'fixed-members C selection')
    return row


class Outputs:
    def __init__(self, columns, test):
        self.columns = columns
        self.test = test
        self.pipelines = []
        self.keys = {}
        self.outputs = []
        self.meta = []
        self.checks = []
        self.expected_paths = []

    def pipeline(self, estimator):
        # In-memory execution limits do not change the fitted predictive function.
        execution = {key: 1 for key in estimator.get_params(deep=True)
                     if key == 'n_jobs' or key.endswith('__n_jobs')}
        if execution:
            estimator.set_params(**execution)
        # Sampling steps are inactive during prediction. Hash every fitted
        # transformer and the final estimator to share identical prediction paths.
        steps = [(name, step) for name, step in estimator.steps[:-1]
                 if step not in (None, 'passthrough') and hasattr(step, 'transform')]
        key = joblib.hash(steps + [estimator.steps[-1]])
        if key not in self.keys:
            self.keys[key] = len(self.pipelines)
            self.pipelines.append(estimator)
        return self.keys[key]

    def add(self, name, kind, members, estimators, meta=None, calibrators=None,
            source=None, expected=None):
        indices = [self.pipeline(estimators[m]) for m in members]
        for scale, calibrator in {'raw': None, **(calibrators or {})}.items():
            self.outputs.append((indices, meta, calibrator))
            self.meta.append({'arm': name, 'kind': kind, 'members': ','.join(members),
                              'scale': scale, 'source': (source or '').replace('\\', '/')})
            if expected is not None:
                if expected.attrs.get('source') not in self.expected_paths:
                    self.expected_paths.append(expected.attrs['source'])
                column = 'proba_raw_fixed' if scale == 'raw' and kind == 'stacking' else 'proba_raw'
                if scale != 'raw':
                    column = 'proba_calibrated' if kind == 'voting' else f'proba_{scale}'
                if column in expected:
                    self.checks.append((len(self.outputs) - 1, expected[column].to_numpy()))

    def __call__(self, values):
        frame = pd.DataFrame(values, columns=self.columns)
        base = [pipe.predict_proba(frame)[:, 1] for pipe in self.pipelines]
        results = []
        for indices, meta, calibrator in self.outputs:
            inputs = np.column_stack([base[i] for i in indices])
            p = inputs[:, 0] if len(indices) == 1 else inputs.mean(axis=1)
            if meta is not None:
                p = meta.predict_proba(inputs)[:, 1]
            results.append(train.apply_calibrator(calibrator, p))
        return np.column_stack(results)


def expected_frame(path, ids):
    frame = pd.read_csv(path).set_index('patient_id')
    if not frame.index.is_unique or set(frame.index) != set(ids):
        raise ValueError(f'Frozen predictions do not cover the test patients: {path}')
    frame = frame.loc[ids]
    frame.attrs['source'] = path
    return frame


def batched_explanation(registry, background, patients, columns, args, seed):
    """Plan fixed permutation masks then evaluate frozen models in bulk.

    PermutationExplainer's mask schedule is independent of predicted values.
    Replay uses the same seed and exact input rows; SHAP performs all marginal
    accumulation and uncertainty calculations itself.
    """
    planned = []
    def collect(values):
        planned.append(np.asarray(values, dtype=float).copy())
        return np.zeros((len(values), len(registry.outputs)))
    masker = shap.maskers.Independent(background, max_samples=args.background)
    options = {'max_evals': args.cycles * (2 * len(columns) + 1),
               'error_bounds': True, 'batch_size': 4096, 'silent': True}
    start = time.time()
    planner = shap.PermutationExplainer(collect, masker, feature_names=columns, seed=seed)
    planner(patients, **options)
    keys, rows = {}, []
    for values in planned:
        for row in values:
            key = row.tobytes()
            if key not in keys:
                keys[key] = len(rows)
                rows.append(row.copy())
    planned_rows = sum(len(values) for values in planned)
    del planned, planner
    planning_seconds = time.time() - start
    log(f'Permutation masks: {planned_rows} rows, {len(rows)} unique; bulk prediction')
    start = time.time()
    probabilities = registry(np.asarray(rows))
    prediction_seconds = time.time() - start
    def replay(values):
        return probabilities[[keys[np.asarray(row, dtype=float).tobytes()] for row in values]]
    start = time.time()
    explainer = shap.PermutationExplainer(replay, masker, feature_names=columns, seed=seed)
    result = explainer(patients, **options)
    info = {'planned_rows': planned_rows, 'unique_masked_rows': len(rows),
            'planning_seconds': planning_seconds, 'bulk_prediction_seconds': prediction_seconds,
            'replay_seconds': time.time() - start, 'execution': 'Exact mask planning and seeded SHAP replay'}
    return result, info


def setup(h, fs):
    data, dev, development_path = load_development(h, fs)
    Xt, yt, it = pred.partition(data, 'train')
    Xe, ye, ie = pred.partition(data, 'test')
    registry = Outputs(list(Xe.columns), Xe)
    own = {m: dev['members'][m]['estimator'] for m in config.MODELS}
    for i, m in enumerate(config.MODELS):
        cal = train.fit_calibrator(dev['train'][:, i], yt, 'sigmoid')
        registry.add(m, 'individual', [m], own, calibrators={'sigmoid': cal},
                     source='Cached fitted individual; sigmoid fitted to cached training OOF')
    selected = pd.read_csv(ROOT / 'results/predictive_sensitivity/selected_configurations.csv')
    best = selected[(selected.horizon == h) & selected.feature_set.eq(fs)
                    & selected.scope.eq('ensembles')].iloc[0].configuration
    votes = sorted(set(['paper', 'paper' if h == 7 else 'diverse', best]))
    for candidate in votes:
        name = f'h{h}_{fs}_{candidate}'
        path = ROOT / f'models/predictive_sensitivity/{name}.joblib'
        if path.exists():
            bundle = joblib.load(path)
            members, estimators = bundle['members'], bundle['members_fitted']
            expected = ROOT / f'predictions/predictive_sensitivity/{name}_test.csv'
        else:
            path = ROOT / f'models/final_{name}.joblib'
            bundle = joblib.load(path)
            members, estimators = bundle['members_order'], bundle['members']
            expected = ROOT / f'predictions/{name}_test.csv'
        registry.add(candidate, 'voting', members, estimators,
                     calibrators={'sigmoid': bundle['calibrator']}, source=str(path.relative_to(ROOT)),
                     expected=expected_frame(expected, ie))
    if fs != 'CV17':
        name = f'h{h}_{fs}_paper_locked'
        path = ROOT / f'models/final_{name}.joblib'
        bundle = joblib.load(path)
        registry.add('paper_locked', 'voting', bundle['members_order'], bundle['members'],
                     calibrators={'sigmoid': bundle['calibrator']}, source=str(path.relative_to(ROOT)),
                     expected=expected_frame(ROOT / f'predictions/{name}_test.csv', ie))
    policies = ('own', 'paper') if fs == 'CV17' else ('own', 'same_baseline', 'locked_baseline', 'paper')
    for policy in policies:
        name = f'h{h}_{fs}_{policy}'
        path = ROOT / f'models/stacking_sensitivity/{name}.joblib'
        if not path.exists():
            raise ValueError(f'Required selected stacking pipeline missing: {path}')
        bundle = joblib.load(path)
        registry.add(policy, 'stacking', bundle['members'], bundle['model'].members,
                     meta=bundle['model'].meta, calibrators=bundle['calibrators'],
                     source=str(path.relative_to(ROOT)),
                     expected=expected_frame(ROOT / f'predictions/stacking_sensitivity/{name}_test.csv', ie))
    if fs != 'CV17':
        selection = matched_C_selection(h, fs)
        C = float(selection.C)
        prepared = joblib.load(CACHE / f'prepared_h{h}_{fs}.joblib')
        utils.assert_same_patients(it, prepared['train_ids'])
        if prepared['development_signature'] != data['signature']:
            raise ValueError('Reselected-C bundle does not match development inputs')
        spec = prepared['C_specs'][str(C)]
        if spec['members'] != selection.members.split(',') or spec['C'] != C:
            raise ValueError('Reselected-C fitted specification differs from validation selection')
        expected = PREDICTIONS / f'h{h}_{fs}_matched_reselected_C_test.csv'
        registry.add('same_members_reselected_C', 'stacking', spec['members'], own,
                     meta=spec['meta'], calibrators=spec['calibrators'],
                     source=f'cache/classification_explainability/prepared_h{h}_{fs}.joblib', expected=expected_frame(expected, ie))
    registry.development = dev
    registry.data = data
    registry.source_paths = [development_path, config.SPLITS_FILE,
        config.RESULTS_DIR / 'predictive_sensitivity/validation_scores.csv',
        config.RESULTS_DIR / 'predictive_sensitivity/selected_configurations.csv',
        config.RESULTS_DIR / 'classification_explainability/matched_C_validation_selection.csv',
        config.RESULTS_DIR / 'stacking_sensitivity/validation_scores.csv',
        type(config.COHORT_STRICT)(str(config.COHORT_STRICT).format(horizon=h))]
    registry.source_paths.extend(registry.expected_paths)
    for meta in registry.meta:
        if meta['kind'] == 'individual':
            continue
        manifest = (ROOT / meta['source']).with_name(
            Path(meta['source']).stem + '_manifest.json')
        if manifest.exists():
            registry.source_paths.append(manifest)
        if meta['kind'] == 'voting' and Path(meta['source']).stem.startswith('final_'):
            name = Path(meta['source']).stem.removeprefix('final_')
            registry.source_paths.append(config.PRED_DIR / f'{name}_test_manifest.json')
    return registry, Xt, it, Xe, ye, ie

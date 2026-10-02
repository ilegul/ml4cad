"""Probability-scale SHAP for frozen classifiers, voting and stacking."""

import gc
import json
import time

import numpy as np
import pandas as pd
import shap

import config
from . import classification as shared
from .classification import log

OUT = shared.RESULTS / "shap"


def run_group(h, fs, args):
    name = f'h{h}_{fs}'
    folder = OUT / ('pilot_batched' if args.pilot else 'groups') / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        info = json.loads((folder / 'complete.json').read_text())
        shared.check_provenance(folder)
        if any(info[key] != getattr(args, key) for key in ('patients', 'background', 'cycles')):
            raise ValueError(f'{name}: cached explanations use different sampling settings')
        log(f'{name}: completed cache retained')
        return
    log(f'{name}: loading frozen pipelines')
    registry, Xt, it, Xe, ye, ie = shared.setup(h, fs)
    shared.write_provenance(folder, registry, registry.data, registry.development, registry.source_paths)
    rng = np.random.default_rng(shared.SHAP_SEED + h)
    bgidx = rng.choice(len(Xt), args.background, replace=False)
    selected = rng.choice(len(Xe), args.patients, replace=False)
    # Sampling does not depend on labels or observed model outcomes.
    background = Xt.iloc[bgidx].to_numpy(dtype=float)
    patients = Xe.iloc[selected].to_numpy(dtype=float)
    full = registry(Xe.to_numpy(dtype=float))
    error = max([float(np.max(np.abs(full[:, j] - p))) for j, p in registry.checks] or [0])
    if error > 1e-8:
        raise ValueError(f'Frozen probability mismatch: {error}')
    log(f'{name}: {len(registry.outputs)} outputs, {len(registry.pipelines)} unique pipelines; '
        f'probability reproduction error {error:.2g}')
    pd.DataFrame(registry.meta).to_csv(folder / 'outputs.csv', index=False)
    pd.DataFrame({'patient_id': ie[selected], 'y_true': ye[selected]}).to_csv(folder / 'patients.csv', index=False)
    pd.DataFrame({'patient_id': it[bgidx]}).to_csv(folder / 'background_patients.csv', index=False)
    pd.DataFrame(background, columns=Xe.columns).to_csv(folder / 'background_features.csv', index=False)
    pd.DataFrame(patients, columns=Xe.columns).to_csv(folder / 'patient_features.csv', index=False)
    start = time.time()
    result, execution = shared.batched_explanation(registry, background, patients, list(Xe.columns), args, shared.SHAP_SEED + h)
    if args.pilot:
        direct = shap.PermutationExplainer(registry, shap.maskers.Independent(background, max_samples=args.background),
                                           feature_names=list(Xe.columns), seed=shared.SHAP_SEED + h)(
            patients, max_evals=args.cycles * (2 * len(Xe.columns) + 1),
            error_bounds=True, batch_size=4096, silent=True)
        replay_error = float(np.max(np.abs(direct.values - result.values)))
        if replay_error > 1e-8:
            raise ValueError(f'Batched SHAP differs from direct evaluation: {replay_error}')
        execution['direct_vs_replay_max_error'] = replay_error
    predicted = full[selected]
    residual = float(np.max(np.abs(result.base_values + result.values.sum(axis=1) - predicted)))
    baseline_error = float(np.max(np.abs(result.base_values - registry(background).mean(axis=0))))
    if residual > 1e-8 or baseline_error > 1e-8 or not np.isfinite(result.values).all():
        raise ValueError('SHAP probability reconstruction failed')
    np.savez_compressed(folder / 'attributions.npz', values=result.values, base_values=result.base_values,
                        predictions=predicted, features=patients, feature_names=np.asarray(Xe.columns, dtype=str),
                        error_std=result.error_std)
    records = []
    for j, meta in enumerate(registry.meta):
        importance = np.abs(result.values[:, :, j]).mean(axis=0)
        order = np.argsort(-importance)
        for rank, k in enumerate(order, 1):
            records.append({**meta, 'horizon': h, 'feature_set': fs, 'feature': Xe.columns[k],
                            'rank': rank, 'mean_abs_shap': importance[k],
                            'mean_shap': result.values[:, k, j].mean(),
                            'is_thyroid': Xe.columns[k] not in config.FEATURE_SETS['CV17']})
    pd.DataFrame(records).to_csv(folder / 'importance.csv', index=False)
    info = {'horizon': h, 'feature_set': fs, 'patients': args.patients, 'background': args.background,
            'cycles': args.cycles, 'outputs': len(registry.outputs), 'pipelines': len(registry.pipelines),
            'seconds': time.time() - start, 'probability_reproduction_max_error': error,
            'additivity_max_error': residual, 'background_mean_max_error': baseline_error,
            'explainer': 'PermutationExplainer', 'link': 'identity', 'shap_version': shap.__version__, 'seed': shared.SHAP_SEED + h, **execution}
    (folder / 'complete.json').write_text(json.dumps(info, indent=2))
    log(f'{name}: completed in {info["seconds"]:.1f}s; additivity error {residual:.2g}')
    del registry, result
    gc.collect()

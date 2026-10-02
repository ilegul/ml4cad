"""Tables, figures and browsable reports from executed interpretation artifacts."""

import html
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from .classification import RESULTS

ROOT = config.ROOT

OUT = RESULTS / "shap"
METHODS = """# Classification SHAP extension

The extension explains all eight individual classifier families and the selected
soft-voting and stacking options in five predictor sets at seven and ten years.
The reference, adapted and independently selected voting compositions are retained
when distinct. Locked reference voting and the available stacking policies are also
included. Policy labels are retained when their predictive functions coincide.

Each horizon uses the same randomly sampled 64 test patients and 16 training
background patients across predictor sets. Sampling is independent of outcomes,
with seed 20261002 plus the horizon. PermutationExplainer uses an identity link
and two antithetic forward/backward cycles per patient. The resulting global
summaries describe this test sample. They do not describe the complete test cohort.
The saved error_std is path variability, not a population confidence interval.
Numerical additivity verifies reconstruction, not Monte Carlo convergence.

The explained function includes preprocessing, the complete classifier or
combination and any calibration mapping. Individual pipelines use the sampler
and hyperparameters selected for their respective family and predictor set.
Sigmoid mappings for individual classifiers are fitted only to existing training
OOF probabilities. Voting uses its saved sigmoid calibrator. Stacking retains
raw, sigmoid and isotonic probabilities. Classification thresholds are not part
of the probability explained by SHAP.

The existing same_baseline stacking policy retains baseline members and C.
The separate same_members_reselected_C sensitivity fixes baseline members while
selecting C from five candidates using stored validation macro-F1 at 0.5,
AUROC and configuration name as tie breakers. Its frozen fitted meta-models,
OOF-derived calibrators and validation thresholds are preserved separately.

Masking uses an independent training background. It can break dependence between
retained and masked predictors, including relationships between biomarkers,
state indicators and the fT3/fT4 ratio. These are interventional approximations,
not conditional or causal explanations. Thyroid attribution share is the sum
of mean absolute SHAP for thyroid predictors divided by that sum for all
predictors. It is not an AUROC increment or a test of incremental value.
The joint contribution sums thyroid attributions within each patient before
taking an absolute value or a signed mean. Predictor redundancy affects attribution.

The previous Random Forest component TreeSHAP remains available in notebook 4.
It uses a different fitted component, explanation method and background. These
new individual-classifier explanations do not replace it. Fixed-horizon cardiac
death probabilities are conditional strict-cohort risks, not competing-risk
population cumulative incidence.

Each group preserves patient identifiers, predictor inputs, output metadata,
attribution arrays and source manifests. All predictors appear in the CSV
tables; figures display the twenty largest mean absolute contributions.
Exact row deduplication and seeded SHAP replay reduce repeated prediction calls.
The direct/replay check is preserved in replay_verification.json.
"""


def slug(value):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', str(value))


def overview(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    selected = table[table.kind.eq('individual') & table.scale.eq('sigmoid')
                     & ~table.feature_set.eq('CV17')]
    feature_sets = ['CV17_THY_CONT', 'CV17_THY_STATES', 'CV17_THY_CONT_STATES', 'CV17_THY_CONT_RATIO']
    names = sorted(selected.arm.unique())
    maximum = max(1., selected.thyroid_attribution_share_pct.max())
    for h, ax in zip([7, 10], axes):
        block = selected[selected.horizon.eq(h)].pivot(index='arm', columns='feature_set',
                                                      values='thyroid_attribution_share_pct').reindex(index=names, columns=feature_sets)
        im = ax.imshow(block, cmap='YlOrRd', vmin=0, vmax=maximum, aspect='auto')
        ax.set_xticks(range(4), ['Continuous', 'States', 'Continuous + states', 'Continuous + ratio'],
                      rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=10)
        ax.set_title(f'{h} years')
        for i in range(len(names)):
            for j in range(4):
                value = block.iloc[i, j]
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=10,
                        color='white' if value > maximum * .6 else '#173345')
    fig.suptitle('Thyroid share of absolute SHAP attribution (%)\nIndividual classifiers, sigmoid probability, shared test sample', fontsize=13)
    fig.subplots_adjust(left=.17, right=.86, bottom=.2, top=.85, wspace=.15)
    cax = fig.add_axes([.89, .25, .018, .52])
    fig.colorbar(im, cax=cax, label='Attribution share (%)')
    fig.savefig(OUT / 'individual_thyroid_share.png', dpi=160)
    plt.close(fig)


def figure(folder, data, j, meta):
    target = folder / 'figures' / f'{j:02d}_{slug(meta.arm)}_{meta.scale}.png'
    if target.exists():
        return str(target.relative_to(OUT)).replace('\\', '/')
    values = data['values'][:, :, j]
    inputs = data['features']
    names = data['feature_names']
    importance = np.abs(values).mean(axis=0)
    top = np.argsort(importance)[-min(20, len(names)):]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.6]})
    y = np.arange(len(top))
    thyroid = np.array([name in ['TSH', 'fT3', 'fT4', 'SCH', 'SCT', 'Low_T3',
                                'Hypothyroid', 'Hyperthyroid', 'fT3_fT4_ratio'] for name in names])
    axes[0].barh(y, importance[top], color=np.where(thyroid[top], '#c75d3a', '#3a718e'))
    axes[0].set_yticks(y, names[top], fontsize=9)
    axes[0].set_xlabel('Mean absolute SHAP value (probability units)')
    rng = np.random.default_rng(8721)
    scatter = None
    for pos, k in enumerate(top):
        x = inputs[:, k]
        finite = np.isfinite(x)
        if finite.any():
            lo, hi = np.percentile(x[finite], [5, 95])
            colours = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
            scatter = axes[1].scatter(values[finite, k], pos + rng.uniform(-.23, .23, finite.sum()),
                                      c=colours[finite], cmap='coolwarm', vmin=0, vmax=1,
                                      s=13, alpha=.75, edgecolors='none')
        if (~finite).any():
            axes[1].scatter(values[~finite, k], pos + rng.uniform(-.23, .23, (~finite).sum()),
                            color='grey', s=13, alpha=.75)
    axes[1].axvline(0, color='grey', linewidth=.7)
    axes[1].set_yticks(y, names[top], fontsize=9)
    axes[1].set_xlabel('SHAP value for cardiac-death probability')
    if scatter is not None:
        bar = fig.colorbar(scatter, ax=axes[1], fraction=.025, pad=.025)
        bar.set_ticks([0, 1], labels=['Low', 'High'])
        bar.set_label('Predictor value')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{folder.name}: {meta.kind} / {meta.arm} / {meta.scale}\n'
                 f'{len(values)} shared test patients; orange bars: thyroid predictors', fontsize=11)
    fig.tight_layout()
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return str(target.relative_to(OUT)).replace('\\', '/')


def main():
    records, importance_frames, checks = [], [], []
    sample_coverage = []
    validation = pd.read_csv(ROOT / 'results/predictive_sensitivity/validation_scores.csv')
    for folder in sorted((OUT / 'groups').glob('*')):
        complete = folder / 'complete.json'
        if not complete.exists():
            continue
        info = json.loads(complete.read_text())
        if not (folder / 'background_features.csv').exists():
            import predictive as pred
            partition = pred.load_partition(info['horizon'], info['feature_set'])
            training, _, ids = pred.partition(partition, 'train')
            selected_ids = pd.read_csv(folder / 'background_patients.csv').patient_id
            indices = pd.Index(ids).get_indexer(selected_ids)
            if (indices < 0).any():
                raise ValueError('Background contains non-training patients')
            training.iloc[indices].to_csv(folder / 'background_features.csv', index=False)
        patient_inputs = pd.read_csv(folder / 'patient_features.csv')
        reference_inputs = pd.read_csv(folder / 'background_features.csv')
        for feature in patient_inputs:
            patient = patient_inputs[feature]
            reference = reference_inputs[feature]
            sample_coverage.append({'horizon': info['horizon'], 'feature_set': info['feature_set'],
                                    'feature': feature, 'test_unique_values': patient.nunique(dropna=False),
                                    'reference_unique_values': reference.nunique(dropna=False),
                                    'combined_unique_values': pd.concat([patient, reference]).nunique(dropna=False),
                                    'test_nonzero': int(patient.fillna(0).ne(0).sum()),
                                    'reference_nonzero': int(reference.fillna(0).ne(0).sum())})
        checks.append(info)
        importance = pd.read_csv(folder / 'importance.csv')
        importance_frames.append(importance)
        outputs = pd.read_csv(folder / 'outputs.csv')
        data = np.load(folder / 'attributions.npz', allow_pickle=False)
        ids = pd.read_csv(folder / 'patients.csv').patient_id
        single_check = 0.
        for j, meta in outputs[outputs.kind.eq('individual')].iterrows():
            source = ROOT / f'predictions/predictive_sensitivity/{folder.name}_{meta.arm}_test.csv'
            if source.exists():
                expected = pd.read_csv(source).set_index('patient_id').loc[ids]
                column = 'proba_raw' if meta.scale == 'raw' else 'proba_calibrated'
                single_check = max(single_check, float(np.max(np.abs(
                    data['predictions'][:, j] - expected[column].to_numpy()))))
        if single_check > 1e-8:
            raise ValueError(f'Individual frozen probability mismatch: {single_check}')
        info['individual_saved_probability_max_error'] = single_check
        for j, meta in outputs.iterrows():
            block = importance[(importance.arm == meta.arm) & (importance.kind == meta.kind)
                               & (importance.scale == meta.scale)].sort_values('rank')
            total = block.mean_abs_shap.sum()
            thy = block[block.is_thyroid]
            indices = np.isin(data['feature_names'], thy.feature.to_numpy())
            phi = data['values'][:, :, j]
            image = figure(folder, data, j, meta)
            meta_C = np.nan
            sampler = ''
            if meta.kind == 'individual':
                selected = validation[(validation.horizon == info['horizon'])
                                      & validation.feature_set.eq(info['feature_set'])
                                      & validation.configuration.eq(meta.arm)]
                sampler = selected.iloc[0].samplers
            if meta.kind == 'stacking':
                source = str(meta.source)
                if source.startswith('cache/classification_explainability/'):
                    selection = pd.read_csv(RESULTS / 'matched_C_validation_selection.csv')
                    meta_C = float(selection[(selection.horizon == info['horizon']) &
                                             selection.feature_set.eq(info['feature_set'])].iloc[0].C)
                else:
                    manifest = ROOT / source.replace('.joblib', '_manifest.json')
                    if manifest.exists():
                        meta_C = json.loads(manifest.read_text()).get('C', np.nan)
            records.append({**meta.to_dict(), 'horizon': info['horizon'], 'feature_set': info['feature_set'],
                            'meta_C': meta_C, 'sampler': sampler,
                            'patients': info['patients'], 'thyroid_attribution_share_pct': 100 * thy.mean_abs_shap.sum() / total,
                            'mean_abs_joint_thyroid_attribution': np.abs(phi[:, indices].sum(axis=1)).mean(),
                            'mean_joint_thyroid_attribution': phi[:, indices].sum(axis=1).mean(),
                            'top_three': ', '.join(block.head(3).feature),
                            'best_thyroid_predictor': thy.iloc[0].feature if len(thy) else '',
                            'best_thyroid_rank': int(thy.iloc[0]['rank']) if len(thy) else np.nan,
                            'baseline_probability': data['base_values'][:, j].mean(),
                            'mean_test_probability': data['predictions'][:, j].mean(), 'figure': image})
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError('No completed groups')
    for h, groups in table.groupby('horizon'):
        reference_ids, reference_bg = None, None
        for fs in groups.feature_set.unique():
            folder = OUT / 'groups' / f'h{h}_{fs}'
            ids = pd.read_csv(folder / 'patients.csv').patient_id.tolist()
            bg = pd.read_csv(folder / 'background_patients.csv').patient_id.tolist()
            if reference_ids is None:
                reference_ids, reference_bg = ids, bg
            elif ids != reference_ids or bg != reference_bg:
                raise ValueError('Patient samples are not shared across representations')
            families = groups[(groups.feature_set == fs) & groups.kind.eq('individual')
                              & groups.scale.eq('raw')].arm.nunique()
            if families != 8:
                raise ValueError('Individual classifier coverage is incomplete')
    if len(checks) != 10:
        raise ValueError(f'Expected ten complete horizon/predictor groups, found {len(checks)}')
    table.to_csv(OUT / 'all_explanations.csv', index=False)
    overview(table)
    pd.DataFrame(sample_coverage).to_csv(OUT / 'predictor_sample_coverage.csv', index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(OUT / 'all_feature_importance.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'numerical_checks.csv', index=False)
    coverage = table.groupby(['horizon', 'feature_set', 'kind']).size().unstack(fill_value=0)
    coverage.to_csv(OUT / 'coverage.csv')
    continuous = table[table.feature_set.eq('CV17_THY_CONT') & table.scale.eq('sigmoid')
                       & table.kind.eq('individual')]
    thyroid_table = continuous.pivot(index='arm', columns='horizon', values='thyroid_attribution_share_pct')
    thyroid_table.to_csv(OUT / 'continuous_individual_thyroid_share.csv')
    notes = METHODS
    notes += f"\nCompleted groups: {len(checks)}; labelled options: {table[table.scale.eq('raw')].shape[0]}; explanations: {len(table)}.\n"
    notes += "\nContinuous thyroid predictors, sigmoid attribution share (%):\n\n"
    notes += "| Classifier | Seven years | Ten years |\n|:---|---:|---:|\n"
    for name, row in thyroid_table.iterrows():
        notes += f"| {name} | {row[7]:.2f} | {row[10]:.2f} |\n"
    (OUT / 'summary.md').write_text(notes, encoding='utf-8')
    selectors = ''.join(f'<label>{label} <select id="{key}"><option value="">All</option>' +
                        ''.join(f'<option>{html.escape(str(x))}</option>' for x in sorted(table[key].unique())) +
                        '</select></label> ' for key, label in [('horizon', 'Horizon'), ('feature_set', 'Set'),
                                                              ('kind', 'Kind'), ('scale', 'Scale')])
    cards = []
    for _, row in table.iterrows():
        attrs = ' '.join(f'data-{key}="{html.escape(str(row[key]))}"' for key in ['horizon', 'feature_set', 'kind', 'scale'])
        cards.append(f'<article {attrs}><h3>{row.horizon} years · {html.escape(row.feature_set)} · '
                     f'{html.escape(row.kind)} · {html.escape(row.arm)} · {row.scale}</h3>'
                     f'<p>Members: {html.escape(row.members)}<br>Top three: {html.escape(row.top_three)}<br>'
                     f'Thyroid share: {row.thyroid_attribution_share_pct:.2f}% · '
                     f'Mean probability: {row.mean_test_probability:.4f}</p>'
                     f'<img loading="lazy" src="{row.figure}" alt="SHAP {html.escape(row.arm)}"></article>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>SHAP classification</title>'
    document += '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#173345} '
    document += 'article{border-top:1px solid #ddd;padding:20px 0}img{max-width:100%}select{padding:6px;margin:6px} '
    document += 'nav{position:sticky;top:0;background:white;padding:12px;border:1px solid #ddd}pre{white-space:pre-wrap;line-height:1.55}</style>'
    document += '<h1>Classification SHAP</h1><pre>' + html.escape(notes) + '</pre>'
    document += '<img src="individual_thyroid_share.png" alt="Thyroid attribution shares across classifier families">'
    document += f'<p>{len(checks)} completed groups · {len(table)} option/scale explanations.</p><nav>{selectors}</nav>'
    document += ''.join(cards)
    document += '''<script>const keys=['horizon','feature_set','kind','scale'];function filter(){document.querySelectorAll('article').forEach(a=>{a.hidden=keys.some(k=>document.getElementById(k).value&&a.getAttribute('data-'+k)!==document.getElementById(k).value)});}keys.forEach(k=>document.getElementById(k).addEventListener('change',filter));</script></html>'''
    (OUT / 'all_results.html').write_text(document, encoding='utf-8')
    print(f'{len(checks)} groups, {len(table)} explanations, {len(records)} figures', flush=True)

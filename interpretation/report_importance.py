"""Tables, figures and browsable reports from executed interpretation artifacts."""

import html
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from .classification import RESULTS

ROOT = config.ROOT

OUT = RESULTS / "importance"
METHODS = """# Classification ablation and permutation extension

All eight individual families and the selected voting/stacking options are
evaluated across five predictor sets and two horizons. The complete frozen
test cohorts contain 878 patients at seven years and 523 at ten years.
The estimator, calibrator and classification threshold remain fixed during
each perturbation. No predictive selection is based on the test outcomes.

Training-mean ablation replaces predictors with their training means. It does
not remove columns or refit the model. Binary means can be fractional and
derived relationships can be broken. Every predictor, seven hierarchical
training Spearman clusters and the applicable cardiovascular, complete thyroid,
continuous, state and ratio blocks are evaluated.

Joint permutation uses ten repeats and a common patient order for all columns
within a target. Orders are shared across targets, configurations and predictor
sets within each horizon, using seed 42. Dependence inside a permuted block
is preserved; dependence with retained predictors can be broken.

Positive drops mean deterioration: baseline minus altered macro-F1, AUROC or
average precision (AP), and altered minus baseline Brier score. Negative values
are retained. Ablation macro-F1 drops and baseline/altered macro-F1 ratios have
paired percentile 95% intervals from 2000 patient bootstrap replicates. Draws
with one outcome class are excluded. These conditional fixed-fit intervals
are descriptive and unadjusted for multiplicity. AUROC, AP and Brier ablation
differences are point estimates. Permutation SD describes variability across
ten permutations; it is not a population confidence interval.

Raw fixed uses 0.5. Individual raw_threshold and sigmoid thresholds are
selected on their own validation predictions. Their sigmoid calibrators use
only existing training OOF probabilities. Voting raw_threshold applies the
saved calibrated threshold to raw probabilities, matching the original
evaluation. Stacking uses its frozen raw, sigmoid and isotonic thresholds.
The separate same_members_reselected_C sensitivity retains baseline members
and reselects C on validation. The existing same_baseline policy also fixes C.

Perturbation importance describes use of predictors by a fitted function.
It does not estimate the difference between separately fitted CV17 and
thyroid-extended predictors. It does not replace the primary locked AUROC
contrast. SHAP explains probabilities on a 64-patient sample; these scores
use thresholded predictions or probability metrics on the complete test cohort.

The original seven-year CV17_THY_CONT reference-voting ablation intervals and
permutation scores are checked against the original tables. The original
notebook and outputs remain available. All scales and targets appear in CSV
tables and the HTML report. Figures show the sigmoid cases. Repeated scores
are stored as losslessly compressed CSV files.
"""
KEYS = ['horizon', 'feature_set', 'kind', 'arm', 'mode']


def overview(block, column, title, filename):
    data = block[block.kind.eq('individual') & block['mode'].eq('sigmoid')]
    sets = ['CV17_THY_CONT', 'CV17_THY_STATES', 'CV17_THY_CONT_STATES', 'CV17_THY_CONT_RATIO']
    classifiers = sorted(data.arm.unique())
    limit = max(.001, data[column].abs().max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for h, ax in zip([7, 10], axes):
        values = data[data.horizon.eq(h)].pivot(index='arm', columns='feature_set', values=column).reindex(
            index=classifiers, columns=sets)
        im = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_title(f'{h} years')
        ax.set_xticks(range(4), ['Continuous', 'States', 'Continuous + states', 'Continuous + ratio'],
                      rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(8), classifiers, fontsize=10)
        for i in range(8):
            for j in range(4):
                value = values.iloc[i, j]
                ax.text(j, i, f'{value:+.3f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(value) > limit * .55 else '#173345')
    fig.suptitle(title + '\nIndividual classifiers, sigmoid probability, full frozen test', fontsize=12)
    fig.subplots_adjust(left=.17, right=.86, bottom=.2, top=.85, wspace=.15)
    cax = fig.add_axes([.89, .25, .018, .52])
    fig.colorbar(im, cax=cax, label='Macro-F1 decrease (positive = deterioration)')
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def check_original(ablation, permutation):
    selected = (ablation.horizon.eq(7) & ablation.feature_set.eq('CV17_THY_CONT')
                & ablation.kind.eq('voting') & ablation.arm.eq('paper') & ablation['mode'].eq('sigmoid'))
    actual = ablation[selected].set_index('target')
    old = pd.read_csv(ROOT / 'results/ablation_h7_CV17_THY_CONT_paper.csv').set_index('name')
    pairs = {'baseline_f1_macro': 'baseline_f1_macro', 'altered_f1_macro': 'ablated_f1_macro',
             'f1_macro_drop': 'absolute_f1_drop', 'importance_ratio': 'importance_ratio',
             'f1_drop_ci_lo': 'absolute_f1_drop_ci_lo', 'f1_drop_ci_hi': 'absolute_f1_drop_ci_hi',
             'importance_ratio_ci_lo': 'importance_ratio_ci_lo',
             'importance_ratio_ci_hi': 'importance_ratio_ci_hi'}
    error = 0.
    for name, original in old.iterrows():
        row = actual.loc[name]
        for current, previous in pairs.items():
            error = max(error, abs(row[current] - original[previous]))
    if error > 1e-8:
        raise ValueError(f'Original ablation results not reproduced: {error}')
    selected = (permutation.horizon.eq(7) & permutation.feature_set.eq('CV17_THY_CONT')
                & permutation.kind.eq('voting') & permutation.arm.eq('paper') & permutation['mode'].eq('sigmoid'))
    actual = permutation[selected].set_index('target')
    sources = [ROOT / 'results/permutation_importance_h7_CV17_THY_CONT_paper.csv',
               ROOT / 'results/grouped_permutation_importance_h7_CV17_THY_CONT_paper.csv']
    perm_error = 0.
    for source in sources:
        for _, row in pd.read_csv(source).iterrows():
            name = 'complete thyroid block' if row['group'] == 'all thyroid variables' else row['group']
            other = actual.loc[name]
            perm_error = max(perm_error, abs(other.f1_macro_drop_mean - row.f1_drop_mean),
                             abs(other.f1_macro_drop_std - row.f1_drop_std))
    if perm_error > 1e-8:
        raise ValueError(f'Original permutation results not reproduced: {perm_error}')
    return {'ablation_max_error': error, 'permutation_max_error': perm_error}


def plot_case(abl, perm, title, path):
    if path.exists():
        return
    single = perm[perm.target_kind.eq('single variable')].sort_values('f1_macro_drop_mean').tail(12)
    names = single.target.to_list()
    matched = abl.set_index('target').loc[names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    y = np.arange(len(names))
    axes[0].hlines(y, matched.f1_drop_ci_lo, matched.f1_drop_ci_hi, color='#326c8a', linewidth=1)
    axes[0].plot(matched.f1_macro_drop, y, 'o', color='#326c8a', markersize=4)
    axes[0].set_title('Training-mean ablation: paired 95% interval')
    axes[1].errorbar(single.f1_macro_drop_mean, y, xerr=single.f1_macro_drop_std,
                     fmt='o', color='#b15e39', capsize=2, markersize=4)
    axes[1].set_title('Permutation: mean ± SD across 10 repeats')
    axes[0].set_yticks(y, names, fontsize=9)
    for ax in axes:
        ax.axvline(0, color='grey', linewidth=.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Decrease in macro-F1')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    folders = sorted((OUT / 'groups').glob('*'))
    if len([p for p in folders if (p / 'complete.json').exists()]) != 10:
        raise RuntimeError('All ten groups must finish before assembling the final report')
    frames = {name: pd.concat([pd.read_csv(p / (f'{name}.csv.gz' if name == 'permutation_repeats' else f'{name}.csv')) for p in folders], ignore_index=True)
              for name in ('ablation', 'permutation', 'permutation_repeats', 'baseline_metrics', 'configurations')}
    ablation, permutation = frames['ablation'], frames['permutation']
    checks = check_original(ablation, permutation)
    infos = pd.DataFrame([json.loads((p / 'complete.json').read_text()) for p in folders])
    for name, frame in frames.items():
        frame.to_csv(OUT / (f'all_{name}.csv.gz' if name == 'permutation_repeats' else f'all_{name}.csv'), index=False)
    infos.to_csv(OUT / 'coverage.csv', index=False)
    (OUT / 'reference_reproduction.json').write_text(json.dumps(checks, indent=2))
    a = ablation[ablation.target.eq('complete thyroid block')]
    p = permutation[permutation.target.eq('complete thyroid block')]
    block = a.merge(p[KEYS + ['f1_macro_drop_mean', 'f1_macro_drop_std', 'roc_auc_drop_mean',
                             'auprc_drop_mean', 'brier_drop_mean']], on=KEYS, validate='one_to_one')
    block.to_csv(OUT / 'thyroid_block_all_configurations.csv', index=False)
    overview(block, 'f1_macro_drop', 'Training-mean ablation of the complete thyroid block', 'thyroid_ablation.png')
    overview(block, 'f1_macro_drop_mean', 'Joint permutation of the complete thyroid block: mean of 10 repeats',
             'thyroid_permutation.png')
    summary = METHODS
    summary += f"\nCompleted groups: {len(infos)}; evaluation cases: {len(frames['configurations'])}; ablations: {len(ablation)}; permutation repeats: {len(frames['permutation_repeats'])}.\n"
    primary = block[block.feature_set.eq('CV17_THY_CONT') & block['mode'].eq('sigmoid')]
    if (primary.f1_drop_ci_lo.le(0) & primary.f1_drop_ci_hi.ge(0)).all():
        summary += "\nAll continuous-thyroid sigmoid ablation macro-F1 intervals include zero. This does not demonstrate equivalence.\n"
    sigmoid = block[block['mode'].eq('sigmoid')]
    selected = sigmoid[sigmoid.f1_drop_ci_lo.gt(0) | sigmoid.f1_drop_ci_hi.lt(0)]
    summary += "\nUnadjusted intervals excluding zero, complete thyroid block:\n\n"
    summary += "| Horizon | Predictor set | Kind | Option | Macro-F1 drop [95% CI] |\n|---:|:---|:---|:---|---:|\n"
    for _, row in selected.sort_values(KEYS).iterrows():
        summary += f"| {row.horizon} | {row.feature_set} | {row.kind} | {row.arm} | {row.f1_macro_drop:+.4f} [{row.f1_drop_ci_lo:+.4f}, {row.f1_drop_ci_hi:+.4f}] |\n"
    (OUT / 'summary.md').write_text(summary, encoding='utf-8')
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    cards = []
    for i, (keys, group) in enumerate(ablation.groupby(KEYS, sort=True)):
        h, fs, kind, arm, mode = keys
        selected = permutation
        for key, value in zip(KEYS, keys):
            selected = selected[selected[key].eq(value)]
        title = f'{h} years / {fs} / {kind} / {arm} / {mode}'
        image = ''
        if mode == 'sigmoid':
            path = figures / f'case_{i:03d}.png'
            plot_case(group, selected, title, path)
            image = f'<img loading="lazy" src="figures/{path.name}" alt="Ablation and permutation importance">'
        metadata = group.iloc[0]
        joined = group.merge(selected[['target', 'f1_macro_drop_mean', 'f1_macro_drop_std',
                                        'roc_auc_drop_mean', 'auprc_drop_mean', 'brier_drop_mean']],
                             on='target', validate='one_to_one').sort_values(['target_kind', 'target'])
        columns = ['target', 'target_kind', 'f1_macro_drop', 'f1_drop_ci_lo', 'f1_drop_ci_hi',
                   'f1_macro_drop_mean', 'f1_macro_drop_std', 'roc_auc_drop', 'roc_auc_drop_mean',
                   'auprc_drop', 'auprc_drop_mean', 'brier_drop', 'brier_drop_mean']
        attributes = ' '.join(f'data-{key}="{html.escape(str(value))}"' for key, value in zip(KEYS, keys))
        cards.append(f'<article {attributes}><h3>{html.escape(title)}</h3><p>Threshold: {metadata.threshold:.2f}; '
                     f'Baseline macro-F1: {metadata.baseline_f1_macro:.4f}</p>{image}'
                     f'<div class="table">{joined[columns].to_html(index=False, float_format=lambda x: f"{x:+.4f}")}</div></article>')
    selectors = ''
    defaults = {'horizon': 7, 'feature_set': 'CV17_THY_CONT', 'kind': '', 'arm': '', 'mode': 'sigmoid'}
    for key in KEYS:
        selectors += f'<label>{key} <select id="{key}"><option value="">All</option>'
        for value in sorted(ablation[key].unique()):
            selectors += f'<option{(" selected" if value == defaults[key] else "")}>{html.escape(str(value))}</option>'
        selectors += '</select></label> '
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Classification importance</title>'
    document += '<style>body{font:16px system-ui;margin:25px auto;max-width:1350px;padding:20px;color:#173345}'
    document += 'pre{white-space:pre-wrap;line-height:1.5}nav{position:sticky;top:0;background:white;padding:12px;border:1px solid #ddd}'
    document += 'select{padding:6px;margin:4px}article{border-top:1px solid #ddd;padding:15px 0}img{max-width:100%}'
    document += '.table{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}</style>'
    document += '<h1>Ablation and permutation importance</h1><pre>' + html.escape(summary) + '</pre><nav>' + selectors + '</nav>'
    document += '<img src="thyroid_ablation.png" alt="Full thyroid-block ablation">'
    document += '<img src="thyroid_permutation.png" alt="Full thyroid-block permutation importance">'
    document += ''.join(cards)
    document += '<script>const keys=' + json.dumps(KEYS) + ';function filter(){document.querySelectorAll("article").forEach(a=>{a.hidden=keys.some(k=>document.getElementById(k).value&&a.getAttribute("data-"+k)!==document.getElementById(k).value);});}keys.forEach(k=>document.getElementById(k).addEventListener("change",filter));filter();</script></html>'
    (OUT / 'all_results.html').write_text(document, encoding='utf-8')
    print(f'{len(infos)} groups, {len(ablation)} ablations, {len(permutation)} permutation targets; reference verified', flush=True)

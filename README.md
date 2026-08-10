# ML4CAD: thyroid variables and cardiac-death risk in ischemic heart disease

## Objective

Does adding thyroid function to a cardiovascular predictor set improve the
classification of cardiac death within a fixed horizon, and the stratification of
long-term risk, in patients with ischemic heart disease?

The prespecified primary comparison is `CV17` against `CV17_THY_CONT`, that is
the cardiovascular baseline with and without the continuous thyroid biomarkers
TSH, fT3 and fT4. Everything else is secondary.

## Relationship to Pingitore et al.

This repository is an adapted replication and methodological extension of

> Pingitore A, Zhang C, Vassalle C, et al. Machine learning to identify a
> composite indicator to predict cardiac death in ischemic heart disease.
> *International Journal of Cardiology* 404 (2024) 131981.

The reference implementation is <https://github.com/orientino/ml4cad>, whose
compact notebook-driven layout this repository follows.

### Differences from the paper

| Paper | Here | Why |
|---|---|---|
| 18 cardiovascular predictors including creatinine | 17; creatinine excluded | creatinine is missing for a large share of this cohort. The baseline is a 17-variable model and is never described as the paper's 18-variable model. |
| Target `survive7Y`, positive class is survival | `y_event`, positive class is cardiac death within the horizon | the clinical question is who dies. `p_survive = 1 - p_event` is derived where paper terminology needs it. |
| Sampler chosen inside the hyperparameter search | hyperparameters searched first, sampler chosen afterwards on validation | keeps the search affordable and keeps sampler selection out of the training folds. Reported as a deviation. |
| Ensemble members fixed at logistic regression, random forest and AdaBoost | that ensemble is kept and always reported, plus an adapted ensemble chosen on validation from a closed candidate set | the paper selected its three members empirically; both the prespecified and the adapted choice are shown. |
| No calibration model | sigmoid calibration fitted on training out-of-fold probabilities, isotonic as a sensitivity | the uncalibrated ensemble over-predicts risk by roughly a factor of two, which makes any absolute-risk or decision-analytic reading meaningless. |
| Survival analysis only through the classifier output | that analysis is kept, plus a cause-specific Random Survival Forest on the full time-to-event cohort | a survival model can use right-censored patients that the fixed-horizon cohort has to drop. |
| Single 7-year horizon | 7 and 10 years under the primary profile | the paper reports the 10-year variant as a robustness check. |

## Repository structure

```
config.py     every experimental option, including the active profile
utils.py      paths, splits, metrics, paired bootstrap, caching
train.py      models, sampling, tuning, calibration, thresholds, frozen test predictions
ensemble.py   the paper ensemble and adapted-ensemble selection
survival.py   ML indicator, Cox, Kaplan-Meier, Random Survival Forest, Aalen-Johansen

1_data_process.ipynb                  cohorts, features, splits
2_classifiers.ipynb                   development-stage model comparison
3_sampling_ensemble_calibration.ipynb  tuning, ensembles, calibration, the single test evaluation
4_feature_cluster_explainability.ipynb clustering, ablation, permutation importance, SHAP
5_survival_analysis.ipynb             paper-aligned indicator and the survival extension

tests/test_pipeline.py                cohort, split and leakage acceptance tests

data/raw          source workbooks, never modified
data/processed    cohorts and the stored split assignment
models            fitted estimators and their manifests
predictions       frozen validation and test predictions
results           result tables
figures           generated figures
cache             resumable intermediate results, not tracked
```

`models` and `cache` are not versioned: they are regenerated from the data and
the cached search results, and the fitted estimators alone weigh 177 MB.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Linux and WSL: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ml4cad
```

Place the three source workbooks in `data/raw`: `raw_data.xlsx`,
`data_prelievo.xlsx` and `creatina_more_columns.xlsx`. Paths are resolved by
walking up from the notebook to the directory containing `config.py`, so the
notebooks work from any working directory and under WSL.

## Execution

Run the notebooks in order, from a fresh kernel:

```bash
jupyter lab
```

Expensive steps are cached by a signature covering the data fingerprint, the
feature list, the target, the horizon, the split, the model, its
hyperparameters, sampling, calibration, threshold strategy, seed, protocol
version and dependency versions. Loading a cache never triggers a fit; a cache
whose signature no longer matches is ignored rather than reused.

To run the tests:

```bash
python -m pytest tests/test_pipeline.py -q
```

## Profiles

Change one line in `config.py`:

```python
PROFILE = "smoke"
```

| Profile | Search draws | Horizons | Feature sets tuned | Purpose |
|---|---|---|---|---|
| `smoke` | 10 | 7 | primary contrast | end-to-end validation of every code path, minutes |
| `primary` | 5000 | 7 and 10 | primary contrast | the paper's search budget on the primary contrast alone; see the cost note below before using it |
| `full` | 1000 | 7 and 10 | all five | the executed analysis: the complete primary contrast plus the secondary feature sets, 11.1 hours |

`full` is a superset of `primary`, so there is no reason to run both.

Nothing else needs to change. Notebooks read horizons, feature sets, model
lists, sampler lists, search budget, bootstrap repetitions and threshold
strategy from `config.py`.

### Measured cost

The `full` profile was executed end to end on 6-7 August 2026: **11.1 hours** on
a 12-core machine, all five notebooks exiting 0. The breakdown was 5.4 hours for
the 80 search jobs, 0.3 hours for the rest of notebook 3, 0.1 hours for
notebooks 1, 2 and 4 together, and 5.3 hours for the survival extension.

The search budget is 1000 draws rather than the paper's 5000. That is a measured
decision, not a shortcut: at 5000 draws these search spaces cost 8.45 hours per
cell, so ten cells would take 84 hours. The paper reports that 10000 draws gave
results very similar to 5000, which places the search inside its plateau well
below 5000. The deviation is recorded in `config.py` next to the profile.

Two search spaces were also bounded after measurement. `SVC` originally ran
without an iteration cap, and a single 5000-draw job took 10.8 hours; it now uses
`max_iter=500000` and `C` capped at 100, which bounds pathological candidates and
lets them surface as convergence warnings instead of running unchecked. The
search itself runs with `probability=False`, because the macro-F1 scorer calls
`predict`; only the winning configuration is refitted with probabilities.

Because the run is long, launch it once and rely on the cache: each
(horizon, feature set, model) job is cached independently, and the search-space
version is tracked per model, so revising one model's space leaves every other
cached job usable.

## Analyses

**Primary.** Frozen test evaluation of the paper ensemble and the adapted
ensemble, on `CV17` and `CV17_THY_CONT`, reported with both uncalibrated and
calibrated probabilities. Incremental value is measured two ways, both paired on
identical test patients:

- *independently optimized*, where each arm keeps its own hyperparameters,
  sampler, calibrator and threshold;
- *locked pipeline*, where model family, hyperparameters, sampler, calibration
  method and threshold rule are held at the baseline configuration and only the
  feature set changes. The primary threshold-dependent comparison applies the
  baseline validation threshold to both arms; letting each arm pick its own
  threshold is reported separately as partially adapted.

Differences in macro-F1, AUROC, AUPRC and Brier score carry paired bootstrap
intervals. The primary contrast is reported uncorrected; secondary feature sets
carry Benjamini-Hochberg q-values.

**Secondary.** The other thyroid representations, the sampling comparison, the
adapted ensemble, isotonic calibration, ablation, permutation importance, SHAP,
and the strict-cohort Random Survival Forest sensitivity.

All eight models are tuned, but out-of-fold probabilities and the sampling
comparison are computed only for the models that can enter a candidate ensemble.
Producing them for a model no ensemble can use would cost hours under the
primary profile and change nothing.

### Feature sets

| Set | Role | Question |
|---|---|---|
| `CV17` | primary baseline | cardiovascular predictors alone |
| `CV17_THY_CONT` | primary thyroid | do continuous TSH, fT3 and fT4 add value |
| `CV17_THY_STATES` | secondary | do the clinical thyroid categories add value |
| `CV17_THY_CONT_STATES` | secondary | do the categories add anything beyond the continuous values |
| `CV17_THY_CONT_RATIO` | secondary | does the fT3/fT4 ratio add value |

Euthyroid is the reference category and is never entered as a dummy. Feature
sets that merely restate one of these questions were deliberately not kept:
every extra set costs multiplicity, computation and interpretation clarity.

### Survival

Two analyses with different estimands, kept visibly separate.

*Analysis A, strict cohort.* The paper-aligned part. Out-of-sample fixed-horizon
probabilities become an ML indicator, used in a single-covariate Cox model and
in Kaplan-Meier stratification at the paper cut of `p_survive = 0.6`. That cut
is kept for comparability and is not a clinically validated threshold. The
median sensitivity uses the median of the validation population, applied as a
fixed value to the test patients. A significant log-rank test shows separation
between strata; it is not a test of incremental thyroid value.

*Analysis B, full time-to-event cohort.* A Cox reference and a cause-specific
Random Survival Forest, with identical outer folds for both feature sets.
Cardiac death is the event of interest and non-cardiac death is treated as
censoring, so these are cause-specific results and not competing-risk cumulative
incidence. Every nuisance quantity needed by Uno's C-index, the time-dependent
AUROC and the integrated Brier score is estimated from the outer training fold
alone. Aalen-Johansen is reported separately for cardiac cumulative incidence in
the presence of non-cardiac death, alongside the `1 - Kaplan-Meier` value so the
size of that bias is visible.

## Results

Every number below comes from one `full`-profile run. The `smoke` profile was
used only to validate the code paths end to end; its artifacts were deleted
before this run and no smoke number is reported anywhere.

### Definitive primary result

The prespecified contrast is `CV17` versus `CV17_THY_CONT` at 7 and 10 years,
paired on identical test patients and reported **without** multiplicity
correction. The locked-pipeline comparison is the one that isolates the feature
effect: model family, hyperparameters, sampler, calibration method and threshold
rule are held at the baseline configuration and only the feature set changes.
Lower is better for the Brier score, so a negative difference favours the
thyroid arm there and a positive difference favours it everywhere else.

| Horizon | delta-AUROC | delta-AUPRC | delta-Brier | delta-F1-macro |
|---|---|---|---|---|
| 7 | +0.0026 (-0.0026, 0.0081) | +0.0041 (-0.0096, 0.0173) | -0.0003 (-0.0022, 0.0015) | -0.0141 (-0.0308, 0.0012) |
| 10 | +0.0010 (-0.0046, 0.0061) | +0.0048 (-0.0033, 0.0132) | -0.0010 (-0.0040, 0.0020) | -0.0057 (-0.0204, 0.0082) |

All eight intervals include zero.

The independently optimized comparison, in which each arm keeps its own tuning,
sampler, calibrator and threshold, gives two nominally significant results out
of eight: delta-F1-macro +0.0212 (0.0005, 0.0431) at 7 years and delta-AUPRC +0.0088
(0.0009, 0.0175) at 10 years. These do not survive scrutiny. At 7 years
delta-F1-macro changes sign between the two comparisons, -0.0141 locked against
+0.0212 unlocked, on the same patients: when each arm may choose a different
model family, sampler and threshold, a positive difference can come from a
luckier configuration rather than from thyroid information. Two nominal hits in
sixteen uncorrected primary tests is what chance produces.

For the paper-aligned survival analysis on the strict cohort, the ML indicator
reproduces the published result: Harrell's C of 0.822 for the baseline on the
frozen test partition against 0.82 in the paper, and Kaplan-Meier separation of
89.6 percent against 35.5 percent seven-year survival at the paper's 0.6 cut
against 88.8 and 29.1 percent. Adding the thyroid biomarkers changes the paired
C-index by +0.0043 (-0.0016, 0.0105) and moves four patients out of 878 between
risk strata.

**Conclusion: no convincing evidence that TSH, fT3 and fT4 add incremental value
for classification or for paper-aligned risk stratification.**

### Secondary and extension results

Secondary feature sets carry Benjamini-Hochberg q-values within each comparison
type and metric. None reaches significance for classification.

The survival extension on the full time-to-event cohort is the one place where a
signal appears. With identical outer folds for both arms, the cause-specific
Random Survival Forest gives a paired delta-C-index of +0.0085 (0.0031, 0.0140) for
the primary contrast, the only primary interval in the study that excludes zero.
Four reasons to treat it as a hypothesis rather than a finding:

- the Cox reference on the same folds sees nothing, +0.0012 (-0.0014, 0.0036);
- the time-dependent AUROC moves the other way for the thyroid arm, 0.8300 to
  0.8265, and the integrated Brier score is unchanged;
- the effect is 1.1 percent relative on a C-index of 0.785;
- the paired bootstrap treats out-of-fold predictions as independent although
  they share training folds, so this interval is anti-conservative. The
  frozen-test intervals in the primary result are not affected.

A non-linear thyroid effect is biologically plausible, low-T3 syndrome being the
obvious candidate, but this evidence is far too thin to support it.

Competing risks matter for absolute numbers: Aalen-Johansen puts seven-year
cardiac cumulative incidence at 11.73 percent against 12.56 percent for
`1 - Kaplan-Meier`, and 14.87 against 16.48 percent at ten years, with 1399
non-cardiac deaths in the cohort.

Calibration was the largest practical gain. Uncalibrated, the paper ensemble has
an observed-to-expected ratio of 0.48 at seven years and 0.78 at ten, that is it
over-predicts absolute risk by roughly a factor of two at the shorter horizon,
with a calibration slope near 1.9 and an expected calibration error above 0.13.
After sigmoid calibration fitted on training out-of-fold probabilities the ratio
is 0.99 at seven years and 0.94 at ten, the slope sits between 0.90 and 1.00 and
the error drops to 0.02 to 0.05. Absolute-risk and decision-curve readings are
meaningful only on the calibrated scale.

## How the test set is protected

- The split is created once in notebook 1 and stored as patient identifiers in
  `data/processed/splits.json`. Every later step loads it.
- Hyperparameters are searched on the training partition only.
- The sampler, the ensemble and the calibration method are selected on
  validation or on training cross-validation, never on test.
- The calibrator is fitted on training out-of-fold probabilities, so validation
  outcomes are spent only on the decision threshold.
- The test partition is evaluated once, by a single routine that writes patient
  identifiers, uncalibrated and calibrated probabilities, predictions, the
  threshold and a manifest to `predictions/`.
- Notebooks 4 and 5 read those frozen artifacts. They do not refit the
  classifier, refit calibration, move a threshold or select a configuration.
  Ablation, permutation importance, SHAP and the ML indicator are post-
  development analyses and cannot feed back into model selection.
- `tests/test_pipeline.py` asserts these properties, including a spy that fails
  if the hyperparameter search ever receives a validation or test row.

## Reproducibility notes

- One seed, `config.SEED`, drives splits, folds, estimators and every bootstrap.
  Paired comparisons share bootstrap draws, so differences are computed on the
  same resamples.
- Warnings are not suppressed. Convergence warnings, failed candidates and NaN
  search scores are counted and written to `results/hyperparameter_search.csv`.
- Superseded artifacts from the previous implementation were moved out of the
  tracked tree into `legacy_cache/`, which is ignored by git.
  `results/legacy_cache_manifest.json` records what they were, their fingerprint
  and why they cannot be reused. They are never counted as results.
- Results in `results/` and `figures/` are only those the current configuration
  produced. Nothing in this repository reports a number from a run that did not
  happen.

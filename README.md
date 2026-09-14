# ML4CAD: thyroid variables and cardiac-death risk in a hospitalised cardiac population

## Objective

Does adding thyroid function to a cardiovascular predictor set improve the
classification of cardiac death within a fixed horizon, and the stratification of
long-term risk, in a hospitalised cardiac population?

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
| 3987 patients meeting an ischemic-heart-disease definition, non-cardiac deaths excluded by design | a distinct cohort of 8065 patients hospitalised with or without established cardiac disease at the same clinical centre, non-cardiac deaths retained | both cohorts were collected at the same clinical centre, but they share neither the patient sample nor the diagnostic composition: about 53 percent of this cohort is flagged by a recorded-field proxy of the paper's IHD definition (`results/cohort_diagnosis_composition.csv`). Retaining non-cardiac deaths is what makes the time-to-event and competing-risks analyses possible. |
| 18 cardiovascular predictors including creatinine | 17; creatinine excluded | creatinine is missing for 7.4 percent of this cohort and was left out of the prespecified baseline. The pipeline does impute, so the exclusion is a protocol choice rather than a technical necessity, the baseline is therefore a 17-variable model that differs from the paper's predictor set by the omission of creatinine. |
| Target `survive7Y`, positive class is survival | `y_event`, positive class is cardiac death within the horizon | the clinical question is who dies. `p_survive = 1 - p_event` is derived where paper terminology needs it. |
| Sampler chosen inside the hyperparameter search | hyperparameters searched first, sampler chosen afterwards on validation | keeps the search affordable and keeps sampler selection out of the training folds. Reported as a deviation. |
| Ensemble members fixed at logistic regression, random forest and AdaBoost | that ensemble is kept and always reported, plus an adapted ensemble chosen on validation from a closed candidate set | the paper selected its three members empirically; both the prespecified and the adapted choice are shown. |
| No calibration model | sigmoid calibration fitted on training out-of-fold probabilities, isotonic as a sensitivity | the uncalibrated ensemble over-predicts risk by roughly a factor of two, which makes any absolute-risk or decision-analytic reading unreliable. |
| Survival analysis only through the classifier output | that analysis is kept, plus a cause-specific Random Survival Forest on the time-to-event cohort | a survival model can use right-censored patients that the fixed-horizon cohort has to drop. |
| Single 7-year horizon | 7 and 10 years under the primary profile | the paper reports the 10-year variant as a robustness check. |

### Endpoint

The event is the source field `CVD Death`, mapped once during preprocessing to
`death_cardiac` and referred to throughout as cardiac death. The rename
operationalises the endpoint and does not assert that cardiovascular death and
cardiac death are interchangeable. The clinical definition the source field
encodes is stated in the thesis.

## Repository structure

```
config.py     every experimental option, including the active profile
utils.py      paths, splits, metrics, paired bootstrap, caching
train.py      models, sampling, tuning, calibration, thresholds, frozen test predictions
ensemble.py   the paper ensemble and adapted-ensemble selection
survival.py   ML indicator, Cox, Kaplan-Meier, Random Survival Forest, Aalen-Johansen, cause-specific CIF

1_data_process.ipynb                  cohorts, features, splits
2_classifiers.ipynb                   development-stage model comparison
3_sampling_ensemble_calibration.ipynb  tuning, ensembles, calibration, the single test evaluation
4_feature_cluster_explainability.ipynb clustering, ablation, permutation importance, SHAP
5_survival_analysis.ipynb             paper-aligned indicator and the survival extension
6_competing_risks.ipynb               cause-specific absolute risk under competing events
7_robustness.py                       post hoc robustness analyses of the frozen results

tests/test_pipeline.py                cohort, split and leakage acceptance tests

data/raw          source workbooks, never modified
data/processed    cohorts and the stored split assignment
models            fitted estimators and their manifests
predictions       frozen validation and test predictions
results           result tables; results/robustness holds the outputs of 7_robustness.py
figures           generated figures
cache             resumable intermediate results, not tracked
```

`models` and `cache` are not versioned: they are regenerated from the data and
the cached search results, and the fitted estimators alone weigh 177 MB.

## Setup

The reported analyses were executed under Python 3.11.

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

Expensive steps are cached. Every signature covers the seed, the protocol
version and the dependency versions, plus the parameters of the step itself.
Coverage is not uniform: the hyperparameter-search cache is the strongest, and
also fingerprints the data and the feature list. The development
cross-validation, the sampling comparison and the frozen test predictions are
keyed on their configuration but not on a data fingerprint, and the survival
cache fingerprints the time and event columns only. Those caches therefore
detect a configuration change but would not detect a silent change to the
underlying data; delete `cache/` and `models/` if the inputs are edited.
Loading a cache never triggers a fit, and a cache whose signature no longer
matches is ignored rather than reused.

To run the tests:

```bash
python -m pytest tests/test_pipeline.py -q
```

The robustness analyses are a separate script, run after the notebooks on the
frozen artifacts they leave behind:

```bash
python 7_robustness.py
```

It writes only into `results/robustness` and `figures`, is resumable phase by
phase, and starts with a reproduction gate that must recover the frozen test
predictions and intervals exactly before any analysis runs. `--smoke` exercises
every code path at reduced sizes in a separate directory;
`--summary --tables --figures` rebuilds the derived outputs from the stored
CSV files without recomputing anything.

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

The `full` profile covers every analysis of the primary contrast together with
the secondary feature sets, at the computational budget specified for that
profile. The two profiles are not computationally nested: `primary` uses a larger
search budget on fewer feature sets.

Nothing else needs to change. Notebooks read horizons, feature sets, model
lists, sampler lists, search budget, bootstrap repetitions and threshold
strategy from `config.py`.

### Measured cost

The `full` profile was executed end to end on 6-7 August 2026: **11.1 hours** on
a 12-core machine, the five notebooks then present exiting 0; the
competing-risks notebook added subsequently runs in roughly an additional
hour and a half. The breakdown was 5.4 hours for
the 80 search jobs, 0.3 hours for the rest of notebook 3, 0.1 hours for
notebooks 1, 2 and 4 together, and 5.3 hours for the survival extension.

The search budget is 1000 draws rather than the paper's 5000: at 5000 draws
these search spaces cost 8.45 hours per cell, so ten cells would have taken 84
hours. This is a documented computational compromise, and it is recorded in
`config.py` next to the profile. The paper observes that 10000 draws gave
results very similar to 5000, which shows a plateau between those two budgets;
it does not by itself establish that the plateau already extends down to 1000.
Running `primary`, which keeps the paper's 5000 draws on the primary contrast,
is the way to check that directly.

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

The robustness script (`7_robustness.py`) ran on 13 September 2026 in about
eight hours on the same machine: 25 minutes for the 100 repeated partitions,
about three hours for the survival analyses (dominated by the forest over ten
fold assignments and the two refit bootstraps), and about four hours for the
three- and five-fold hyperparameter searches. It is resumable, so an
interrupted run continues from the first phase without an output file.

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

Three analyses with different estimands, kept visibly separate.

*Analysis A, strict cohort.* The paper-aligned part. Out-of-sample fixed-horizon
probabilities become an ML indicator, used in a single-covariate Cox model and
in Kaplan-Meier stratification at the paper cut of `p_survive = 0.6`, applied
on the uncalibrated probability scale on which the published threshold was
defined. That cut is kept for comparability and is not a clinically validated
threshold. The
median sensitivity uses the median of the validation population, applied as a
fixed value to the test patients. A significant log-rank test shows separation
between strata; it is not a test of incremental thyroid value.

*Analysis B, time-to-event cohort.* A Cox reference and a cause-specific
Random Survival Forest, with identical outer folds for both feature sets.
Cardiac death is the event of interest and non-cardiac death is treated as
censoring, so these are cause-specific results and not competing-risk cumulative
incidence. Every nuisance quantity needed by Uno's C-index, the time-dependent
AUROC and the integrated Brier score is estimated from the outer training fold
alone. Aalen-Johansen is reported separately for cardiac cumulative incidence in
the presence of non-cardiac death, alongside the `1 - Kaplan-Meier` value so the
size of that bias is visible.

*Analysis C, competing-risks absolute risk.* The estimand analyses A and B do
not touch: each patient's cardiac cumulative incidence in the presence of
non-cardiac death, on the time-to-event cohort. Two cause-specific Cox models per
feature set are combined through their Breslow baselines into a per-patient
`F1(t)`; the outer folds are stratified on the three-level cause code and
shared by every feature set, so the thyroid contrast stays paired. Ranking is
scored cause-specifically for comparability with analysis B; the absolute
scale is checked against Aalen-Johansen within quintiles of predicted risk.

## Results

Every number below comes from one `full`-profile run. The `smoke` profile was
used only to validate the code paths end to end; its artifacts were deleted
before this run and no smoke number is reported anywhere.

### Definitive primary result

The prespecified contrast is `CV17` versus `CV17_THY_CONT` at 7 and 10 years,
paired on identical test patients and reported **without** multiplicity
correction. The locked comparison estimates the feature-set difference under common
baseline-derived development choices: model family, hyperparameters, sampler,
calibration method and threshold rule are held at the baseline configuration
and only the feature set changes. The calibration mapping is refitted per arm,
as it must be.
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
+0.0212 unlocked, on the same patients: when each arm may choose its own
hyperparameters, sampler, calibrator and threshold, a positive difference may
reflect the selected configuration rather than thyroid information. The
reference ensemble members (logistic regression, random forest and AdaBoost)
are the same in both arms. These are two nominally
significant results among sixteen uncorrected comparisons, so they should be
read with caution.

For the survival analysis following the reference study on the strict cohort, the ML indicator
attains concordance comparable with the published value: Harrell's C of 0.822
for the baseline on the frozen test partition against 0.82 in the paper. Kaplan-Meier stratification
at the paper's 0.6 cut on the uncalibrated scale separates the strata at 95.1
against 63.2 percent seven-year survival (published: 88.8 against 29.1 percent,
in a distinct cohort; the uncalibrated ensemble here over-predicts risk, so the
same nominal cut assigns more patients to the low-survival stratum, which may
partly contribute to the less extreme separation). Adding the thyroid biomarkers changes the paired C-index by +0.0043
(-0.0016, 0.0105). The high-predicted-survival stratum goes from 487 to 516
patients, and that net figure hides movement in both directions: patient by
patient, 41 move up and 12 move down, so 53 of 878 are reclassified
(`results/indicator_reclassification.csv`).

**Conclusion: no robust evidence of improved incremental predictive value from TSH, fT3 and fT4
for classification or for paper-aligned risk stratification.**

### Secondary and extension results

Secondary feature sets carry Benjamini-Hochberg q-values within each comparison
type and metric. No secondary thyroid representation showed a statistically
supported improvement in classification. One comparison is significant in the
opposite direction: under the locked pipeline at 7 years, `CV17_THY_STATES` has
a delta-AUPRC of -0.0106 (-0.0182, -0.0030) with q = 0.030, that is the
thyroid-state representation performs worse than the baseline. This does not
support incremental improvement.

The survival extension on the time-to-event cohort is where the positive
signals concentrate. With identical outer folds for both arms, the
cause-specific Random Survival Forest gives a paired delta-C-index of +0.0085
(0.0031, 0.0140) for the primary contrast: among the survival-model
comparisons, this interval excludes zero. Three secondary comparisons
also survive correction, all small: Cox with thyroid states q = 0.018, Random
Survival Forest with thyroid states q = 0.036, and Random Survival Forest with
continuous plus states q = 0.036. They carry the same caveats as the primary one
and none exceeds +0.007 in absolute terms.

Four reasons to treat all of this as a hypothesis rather than a finding:

- the Cox reference on the same folds sees nothing, +0.0012 (-0.0014, 0.0036);
- the time-dependent AUROC moves the other way for the thyroid arm, 0.8300 to
  0.8265, and the integrated Brier score is unchanged;
- the absolute difference is +0.0085 in concordance;
- the paired bootstrap treats out-of-fold predictions as independent although
  they share training folds, so this interval is not accounting for shared-training-fold dependence. The
  frozen-test intervals in the primary result are not affected.

A non-linear thyroid effect is biologically plausible, low-T3 syndrome being the
plausible candidate, but this evidence is far too thin to support it.

Competing risks matter for absolute numbers: Aalen-Johansen puts seven-year
cardiac cumulative incidence at 11.72 percent against 12.56 percent for
`1 - Kaplan-Meier`, and 14.87 against 16.48 percent at ten years, with 1399
non-cardiac deaths in the cohort.

Analysis C models that estimand directly and is consistent with the primary
conclusion. The
cause-specific framework closely reproduces the population incidence
(mean predicted 11.84 percent at seven years against 11.72 observed,
observed-to-predicted 0.99 for both arms, quintile ratios within 0.88-1.12)
and the paired thyroid contrast on the predicted incidence, with Harrell's
concordance administratively censored at the evaluated horizon, is +0.0014
(-0.0011, 0.0034) at seven years and +0.0007 (-0.0017, 0.0026) at ten. The
clinical-states set repeats the small Cox-side signal (about +0.004,
q = 0.027 and 0.036); the combined continuous-plus-states set shows a similar
increment without surviving correction. The out-of-fold intervals carry the
same shared-training-folds caveat as analysis B.

### Robustness analyses

`7_robustness.py` was added after the analysis above was frozen, and none of
its outputs replaces a reported number or was used for selection. It starts by re-implementing the notebook-3
flow and reproducing the frozen test predictions and intervals to within
1e-16, then runs four groups of post hoc analyses on the primary contrast.
The full tables are in `results/robustness/SUMMARY.md` and in the thesis
appendix; the headline results are:

- *Repeated partitions.* The 60/20/20 procedure repeated over 50 random
  partitions per horizon with frozen hyperparameters and samplers. The
  standard deviation of the paired delta across partitions is of similar
  magnitude to the bootstrap standard error of the frozen test (ratio
  0.73-1.27). The locked seven-year delta-AUROC averages +0.0023 and is
  positive in 88 percent of partitions, but its within-partition interval
  excludes zero in 10 percent. The two nominally significant independently
  optimized results of the frozen test sit at the 96th and 92nd percentiles of
  their partition distributions, and the locked seven-year delta-F1-macro
  (-0.0141) at the 2nd: the sign change between the two modes is
  split-sensitive, both modes averaging about +0.003 across partitions.
- *Survival fold assignments.* With ten different outer-fold assignments, the
  between-assignment standard deviation of the paired delta-C-index is 0.35 to
  0.46 of the within-assignment bootstrap standard error for Cox, for the
  random survival forest at its default configuration and for analysis C. The
  forest gives +0.0096 on average, positive with an interval excluding zero in
  all ten assignments (+0.0083 on the reported assignment against +0.0085 for
  the tuned forest), while the time-dependent AUROC and the integrated Brier
  score again do not move: a small, internally reproducible and metric-specific
  concordance signal.
- *Bootstrap with refitting.* Refitting the Cox model (analysis B) or the
  cause-specific pair (analysis C) inside each of 2000 resamples and scoring
  the out-of-bag patients widens the intervals by 39 percent (Cox) and by 66
  and 84 percent (analysis C at seven and ten years), widening being the ratio
  of the refit percentile-interval width to the reported interval width (the
  ratio of the refit standard deviation to the standard error implied by the
  reported interval is 1.36, 1.65 and 1.84); every interval still includes
  zero. The no-refit bootstrap used above understates uncertainty by a
  measurable but moderate amount that changes no conclusion.
- *Inner folds of the search.* Re-running the search of the three reference
  members with 3 and 5 inner folds on the same 1000 candidates changes the
  winner in all 12 cells, yet on a training-only 5x3 repeated cross-validation
  the ensemble macro-F1 differs from the two-fold winners by -0.0034 to +0.0037,
  within the fold-to-fold standard deviation. On the frozen test (a second
  reading, reported in full) the locked comparison stays null for every fold
  count while the nominally significant independently optimized comparisons
  change with it: F1 at 7 years and AUPRC at 10 with two folds, AUROC, AUPRC
  and Brier at 7 years with three, none with five.

Calibration was the largest practical gain. Uncalibrated, the paper ensemble has
an observed-to-expected ratio of 0.48 at seven years and 0.78 at ten, that is it
over-predicts absolute risk by roughly a factor of two at the shorter horizon,
with a calibration slope near 1.9 and an expected calibration error above 0.13.
After sigmoid calibration fitted on training out-of-fold probabilities the ratio
is 0.99 at seven years and 0.94 at ten, the slope sits between 0.91 and 1.01 and
the error drops to 0.02 to 0.05. Probability-scale and decision-curve
interpretations should rely on the calibrated predictions.

## How the test set is protected

- The split is created once in notebook 1 and stored as patient identifiers in
  `data/processed/splits.json`. Every later step loads it.
- Hyperparameters are searched on the training partition only.
- The sampler and the adapted ensemble are selected during model development
  on validation or on training cross-validation, never on test. The
  calibration method is prespecified (sigmoid, with isotonic as a
  sensitivity); only the calibrator parameters are estimated.
- The calibrator is fitted on training out-of-fold probabilities, so
  calibration does not use validation outcomes. Validation is used for the
  development decisions the protocol prescribes: the sampler, the adapted
  ensemble and the decision threshold.
- Test outcomes are used for none of those decisions. A single routine writes
  patient identifiers, uncalibrated and calibrated probabilities, predictions,
  the threshold and a manifest to `predictions/`; test probabilities are
  computed earlier alongside the validation ones, but no test outcome enters
  model development or configuration selection.
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
- XGBoost tunes `scale_pos_weight` while the development stage may also apply
  `RandomOverSampler`, so some XGBoost configurations combine two imbalance
  corrections. This is an implementation peculiarity of the XGBoost branch: it
  does not affect the reference ensemble (logistic regression, random forest,
  AdaBoost) on which the formal incremental comparisons are computed. A future
  revision should select one imbalance mechanism at a time.
- Warnings are not suppressed. Failed candidates and NaN search scores are
  counted and written to `results/hyperparameter_search.csv`, together with the
  convergence warnings raised by the final refit of each winning configuration;
  warnings raised inside the parallel search workers are not aggregated.
- Superseded artifacts from the previous implementation were moved out of the
  tracked tree into `legacy_cache/`, which is ignored by git.
- The robustness script reads the frozen artifacts and writes only under
  `results/robustness` and `figures`; its run log (`run_log.txt`) and phase
  status are stored with its outputs. The hyperparameters it selects with 3
  and 5 inner folds are stored for inspection and adopted nowhere.
  `results/legacy_cache_manifest.json` records what they were, their fingerprint
  and why they cannot be reused. They are never counted as results.
- Results in `results/` and `figures/` are only those the current configuration
  produced. Nothing in this repository reports a number from a run that did not
  happen.

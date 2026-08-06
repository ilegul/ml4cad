# Migration report

Refactor of a repository that held two parallel implementations of the same
science into one compact pipeline in the layout of `orientino/ml4cad`.

A reversible checkpoint precedes every destructive change: commit
`Checkpoint before compact refactor`, tag `pre-refactor`. Everything tracked
below is recoverable from that tag.

The `legacy_cache/` directory that was to hold the never-tracked superseded
artifacts was destroyed during the refactor, together with `docs/`. The thesis
documents in `docs/` were restored from the checkpoint commit and verified
byte-identical. The superseded artifacts were only partly recoverable: the 82
tracked files under `reports/`, including all 57 paper-tuning JSON and CSV
artifacts, remain in the `pre-refactor` tag and can be restored with
`git checkout pre-refactor -- reports/`. The git-ignored part is gone: the
fitted `.joblib` models of the superseded tuning run, `reports/figures/*.png`,
the superseded `data/processed` CSVs and the `tmp/` scratch directory. None of
it was usable for the current analysis, for the reasons listed under Superseded
outputs, but the loss was accidental rather than intended and is recorded here
as such. `results/legacy_cache_manifest.json` states the current situation.

## Files removed from the active project

### Independent pipeline

| Removed | Contained | Disposition |
|---|---|---|
| `src/preprocessing/build_dataset.py` | raw load, merge, corrections, time-to-event, cohorts | migrated into `1_data_process.ipynb` |
| `src/features/feature_engineering.py` | derived thyroid features, feature-set access, extractors | migrated into `1_data_process.ipynb` and `utils.extract_xy` |
| `src/classification/screening.py` | model and sampler factories, pipeline builder | migrated into `train.get_model`, `train.get_sampler`, `train.make_pipeline` |
| `src/classification/robust_cv.py` | 5-fold cross-validation, inner-validation threshold selection | pattern migrated into `train.tune` and `utils.optimize_threshold` |
| `src/classification/tuning.py` | random search, threshold, result table | discarded and rewritten: it searched on the whole cohort and selected the threshold in-sample |
| `src/evaluation/calibration.py` | calibrated-model comparison | discarded and rewritten: it inherited the selection above and evaluated it on a split of the same rows |
| `src/evaluation/shap_analysis.py` | SHAP and permutation importance | migrated into `4_feature_cluster_explainability.ipynb` |
| `src/survival/survival_analysis.py` | Kaplan-Meier, Aalen-Johansen, Cox, survival cross-validation | migrated into `survival.py`; the circular risk-stratification log-rank was rewritten |
| `src/survival/survival_cohort.py` | strict and competing cohorts with administrative censoring | folded into `survival.survival_frame` |
| `src/visualization/eda.py` | cohort flow, univariate association, correlation clustering, patient clustering | cohort flow and the baseline description migrated into `1_data_process.ipynb`; feature clustering into notebook 4; KMeans and PCA patient clustering dropped, see omissions |
| `run_pipeline.py` | driver that also suppressed all `UserWarning` | dropped; notebooks are the entry points and warnings are counted, not hidden |
| `notebooks/01`-`05` | the independent pipeline notebooks | replaced by the five numbered notebooks |
| `configs/config.py` | paths, eight feature sets, legacy cache-name aliases | migrated into `config.py`; the alias tables were dropped with the cache they protected |
| `Pipeline.md` | the original specification, with several numbers that no longer matched the data | content folded into this report and the README |

### Paper-alignment pipeline

| Removed | Contained | Disposition |
|---|---|---|
| `src/alignment/cohorts10.py` | strict cohort at an arbitrary horizon | migrated into `1_data_process.ipynb` |
| `src/alignment/features_align.py` | Euthyroid-as-reference encoding, NaN-preserving extraction | migrated into `config.FEATURE_SETS` and `utils.extract_xy` |
| `src/alignment/holdout_split.py` | stratified 60/20/20 | migrated into `utils.make_splits`, now keyed by patient identifier and persisted |
| `src/alignment/ensemble.py` | soft-vote ensemble plus six invented ensembles | the paper ensemble migrated into `ensemble.py`; the six extra ensembles and the whole "winner" branch were dropped |
| `src/alignment/evaluation.py` | two evaluation schemes, paired bootstrap | migrated into `utils.classification_metrics` and `utils.paired_delta` |
| `src/alignment/paper_tuning.py` | 1414 lines: random search, sampling refinement, cache signatures | the search, the signature discipline and the paired bootstrap migrated into `train.py` and `utils.py`; the rest was dropped |
| `src/alignment/ablation.py` | knock-out ablation, importance ratio | migrated into `4_feature_cluster_explainability.ipynb`, now with bootstrap intervals |
| `src/alignment/calibration.py` | Brier table | superseded by `train.calibration_report` |
| `src/alignment/clinical_utility.py` | calibration summary, decision curve, clinical impact | migrated into `train.calibration_report`, `train.decision_curve`, `train.clinical_impact` |
| `src/alignment/ml_indicator.py` | ML indicator, Cox, Kaplan-Meier, paired delta-C-index | migrated into `survival.py` |
| `src/alignment/run_alignment.py` | caching and orchestration, `pick_winner` | orchestration moved into the notebooks; `pick_winner` dropped, see corrections |
| `notebooks/paper_alignment_pingitore.ipynb` | 38 cells | split across the five notebooks |
| `tests/test_alignment_clinical_utility.py`, `tests/test_alignment_paper_tuning.py` | clinical-utility unit tests and one leakage spy | the leakage spy pattern migrated into `tests/test_pipeline.py` |

### Superseded outputs

Removed from the active tree. The tracked portion survives in the
`pre-refactor` tag; the git-ignored portion was lost, see the note above.

| Moved | Size | Why it cannot be reused |
|---|---|---|
| `reports/alignment/paper_tuning/` | 105 files, 13.9 MB, 7.56 hours of compute | built for `CV17_THY_CONT_STATES` rather than the new primary `CV17_THY_CONT`; XGBoost's `scale_pos_weight` range assumed the opposite class orientation; calibration, threshold and ensemble selection followed a superseded protocol. Recorded in `results/legacy_cache_manifest.json` with its content hash. |
| `reports/` (remainder) | result tables and 90 figures | produced by the leaking tuning and calibration code; not traceable to the current configuration |
| `data/processed/` legacy files | competing cohort, horizon-less strict cohort, `results_*.csv`, SHAP tables | superseded cohort definitions and result schemas |
| `tmp/` | 359 scratch files, including a clone of the reference repository | one-off scripts and intermediate renders |

The four thesis documents `reports/Report_*.docx` were untracked and are now in
`docs/`. They were not generated by this pipeline and are preserved unchanged.
Raw workbooks in `data/raw` were never read for writing.

## Files added

| File | Role |
|---|---|
| `config.py` | the only file with experimental options: profile, seed, horizons, feature sets, models, samplers, calibration, thresholds, budgets, cache flags, and the patient-level corrections |
| `utils.py` | paths, seeding, input and output, feature extraction, splits keyed by patient identifier, alignment assertions, metrics, shared paired bootstrap, Benjamini-Hochberg, cache signatures, manifests |
| `train.py` | model and sampler factories, sampler-specific pipelines, search spaces, tuning, training out-of-fold calibration, threshold selection, calibration and decision-analytic reports, the single frozen test routine |
| `ensemble.py` | the prespecified paper ensemble, the closed adapted-ensemble candidate set, alignment checks |
| `survival.py` | analysis A (ML indicator, Cox, Kaplan-Meier, paired delta-C-index) and analysis B (cause-specific Cox and Random Survival Forest, time-dependent metrics, permutation importance, Aalen-Johansen) |
| five numbered notebooks | the pipeline, thin, cache-first, driven entirely by `config.py` |
| `tests/test_pipeline.py` | the acceptance tests |

Two additions sit outside the compact reference layout, both deliberate:
`tests/test_pipeline.py`, because the acceptance criteria require automated
checks, and `docs/`, to keep the untracked thesis documents separate from
generated output.

## Methodological corrections

1. **Hyperparameter search no longer sees the whole cohort.** `tuning.py:205`
   fitted `RandomizedSearchCV` on every row and then reported cross-validated
   metrics on those same rows. The search now runs on the training partition
   only, and a spy in the test suite fails if a validation or test row ever
   reaches it.
2. **The decision threshold is no longer selected in-sample.**
   `tuning.py:117-123` optimised it on the model's own training predictions,
   contradicting its docstring. It is now selected on validation.
3. **Calibration no longer inherits a contaminated selection.**
   `calibration.py:67-90` read the model, hyperparameters and threshold from a
   table produced on the full cohort and evaluated them on a split of the same
   rows. The calibrator is now fitted on training out-of-fold probabilities,
   sigmoid is prespecified, isotonic is a separately reported sensitivity, and
   the two are never compared on the observations used to fit them.
4. **The uncalibrated over-prediction is now visible and corrected.** On the
   frozen test partition the uncalibrated ensemble has an observed-to-expected
   ratio near 0.47, that is it over-predicts absolute risk by roughly a factor of
   two. After calibration the ratio is near 1.01 with a slope near 0.98. The old
   pipeline reported decision curves on the uncalibrated scale.
5. **Ensemble selection no longer uses test rows.** `run_alignment.pick_winner`
   claimed in its docstring that test rows were never used, but selected on
   out-of-fold metrics computed over the whole cohort, including the rows later
   reported as test, and excluded the paper ensemble from its own candidate pool
   so that a runner-up was labelled the winner. The candidate set is now closed
   before validation is read, contains the paper ensemble, and is ranked by
   training-only cross-validation and validation.
6. **XGBoost can now up-weight the minority class.** The `scale_pos_weight` grid
   was copied from the paper, whose positive class was the majority, so with the
   event as positive the appropriate value fell outside the grid. It is now
   anchored on the negative-to-positive ratio of the training partition and
   recorded in the model manifest.
7. **Preprocessing order is sampler-specific.** One universal
   impute-scale-sample order is invalid for SMOTENC, whose categorical indices
   must refer to unscaled binary columns. SMOTENC now runs before scaling; plain
   SMOTE variants, which interpolate dummies into fractional values, are labelled
   paper-replication sensitivities rather than treated as preferred.
8. **Ablation carries uncertainty and does not re-split the test set.** The old
   ablation used one split and one seed, producing rankings that flipped between
   horizons yet were reported as findings. Knock-outs now run against the frozen
   model on the frozen test patients, with a paired bootstrap over those patients
   for both the importance ratio and the absolute macro-F1 difference. The ratio
   is computed without rounding, and a ratio above 1 is not presented as a test.
9. **Feature clustering is estimated on training data only** and described as
   descriptive, since it defines cluster ablations evaluated on test.
10. **Explainability is separated from incremental value.** Permutation
    importance is computed for the frozen ensemble as a whole; SHAP explains one
    named tree member and says so. The SHAP block share is reported next to the
    per-feature mean, because a sum over 17 cardiovascular columns mechanically
    exceeds a sum over 3 thyroid columns.
11. **Risk groups are no longer defined and evaluated on the same fit.** The old
    stratification derived tertiles from a Cox model fitted on the same patients
    and events and then reported a log-rank p-value on them. Cut-points now come
    from the predicted risk of the training fold and are applied to held-out
    patients.
12. **Competing risks are handled explicitly.** Per-group horizon risk used
    `1 - Kaplan-Meier` despite roughly 1400 competing non-cardiac deaths.
    Aalen-Johansen is now reported next to it, with the gap shown. The
    cause-specific Random Survival Forest is never described as a competing-risk
    estimator.
13. **Missing thyroid status is no longer imputed as normal.**
    `feature_engineering.py:42,55` mapped a missing `Euthyroid` to "not abnormal"
    and an unknown state to euthyroid.
14. **Merges are validated.** The auxiliary workbooks are merged with
    `validate="m:1"`, so a duplicate key would fail loudly instead of fanning one
    patient across splits.
15. **Warnings are no longer globally suppressed.** `run_pipeline.py:38-39`
    silenced every `UserWarning`, which includes scikit-learn convergence
    warnings. Convergence warnings, failed candidates and NaN search scores are
    now counted per search and written to `results/hyperparameter_search.csv`.
16. **A real bug in the integrated Brier score was fixed.** The first
    implementation called `predict_survival_function` on a scikit-learn
    `Pipeline`, which does not forward that method, so the metric silently became
    NaN. Preprocessing steps are now applied explicitly, and metric failures are
    recorded with their reason instead of being swallowed.
17. **Multiplicity is controlled and the primary contrast is prespecified.** The
    previous 28 uncorrected paired comparisons are replaced by one prespecified
    primary contrast, reported uncorrected, with the secondary feature sets
    carrying Benjamini-Hochberg q-values.
18. **Stale narrative was discarded rather than patched.** The previous reports
    quoted a superseded cross-validation table: the headline delta was written as
    `+0.0003` where the current tables gave `+0.0015`, one delta had the wrong
    sign, all nine calibration numbers were stale, and "Tuning 12/12 complete"
    was claimed while the block was 58 of 64. No number in this repository is
    carried over from those files.
19. **An unbounded SVC search space was bounded.** Removing the old
    `max_iter=5000` cap was correct, since that cap truncated the solver and the
    resulting F1 was non-convergence presented as a result. But with `C` reaching
    1000 and no cap at all, a single 5000-draw search ran for 10.8 hours. The cap
    is now `max_iter=500000` with `C` limited to 100, which bounds pathological
    candidates and lets them appear as convergence warnings.
20. **The search no longer fits probabilities it never uses.** The scorer is
    macro-F1, which calls `predict`, so fitting Platt probabilities for every
    candidate was wasted work; the search runs with `probability=False` and only
    the winning configuration is refitted with probabilities.
21. **Search-space versions are tracked per model.** A single global version
    meant that revising one model's space invalidated every cached job. On a
    multi-hour run that is an expensive mistake, so `SEARCH_SPACE_VERSION` is now
    a mapping and `tests/test_pipeline.py` asserts the per-model signatures stay
    distinct.
22. **Concurrent runs are refused.** Stopping a run left its shell alive, and a
    second run then wrote the same model files, so a `.joblib` from one run could
    pair with a manifest from the other. The runner now takes a PID lock and
    refuses to start while another run is live; the affected cache was discarded
    rather than trusted.


## Analyses added

- AUPRC alongside AUROC, and both uncalibrated and calibrated test probabilities.
- The locked-pipeline comparison, which holds model family, hyperparameters,
  sampler, calibration and threshold rule fixed and changes only the feature set.
  It resolves an ambiguity the independently optimized comparison cannot: under
  the smoke profile the independently optimized macro-F1 difference is negative
  with an interval excluding zero, driven by each arm choosing its own threshold,
  while the locked difference is essentially zero.
- Whole-thyroid-block ablation and grouped permutation importance for the thyroid
  block, the continuous biomarkers and the state indicators.
- Analysis B: a cause-specific Random Survival Forest and a Cox reference on the
  full time-to-event cohort, with identical outer folds for both feature sets,
  Uno's C-index, time-dependent AUROC, the integrated Brier score and
  out-of-sample permutation importance, every nuisance quantity estimated from
  the outer training fold alone.
- Aalen-Johansen cumulative incidence reported against `1 - Kaplan-Meier`.
- A strict-cohort Random Survival Forest, labelled a strict-cohort conditional
  sensitivity analysis.
- Persisted split assignments, per-artifact manifests with dependency versions,
  and signature-checked caches.

## Deliberate omissions

- **KMeans and PCA patient clustering** from the old exploratory notebook. It
  reported a silhouette near 0.12, that is no cluster structure, and bears on
  neither the replication nor the incremental-value question.
- **The competing-risks classification cohort**, in which non-cardiac deaths
  before the horizon were labelled negative. It answers a different question from
  the strict cohort and was analysed downstream under the opposite convention.
  Competing risks are now handled where they belong, in the time-to-event
  analysis, through Aalen-Johansen.
- **Three thyroid feature sets** kept by the old pipeline: the single
  any-abnormality flag, the ordinal hypothyroid-to-hyperthyroid axis, and
  continuous plus states plus ratio. Each restates a question already covered by
  the five retained sets, and each adds multiplicity and computation.
- **A DeLong test for the AUROC difference.** The paired bootstrap is applied
  uniformly to macro-F1, AUROC, AUPRC, Brier score, C-index and ablation
  differences. A hand-written fast DeLong could not be validated against a
  trusted reference in this environment, and the specification makes it optional,
  so it was omitted rather than added as unverified statistical code.
- **The Fine-Gray subdistribution model.** Competing risks are described with
  Aalen-Johansen; no covariate model for the subdistribution hazard is fitted.

## Experiments actually run

### Smoke validation

The `smoke` profile was used only to prove that every code path executes: 10
search draws, one horizon, the primary contrast, 100 bootstrap resamples. All
five notebooks ran in order from a fresh kernel and `tests/test_pipeline.py`
passed. Its artifacts were then deleted, so no smoke number appears in
`results/` or `figures/` and none is reported anywhere.

### Definitive full-profile run

Executed 6-7 August 2026 under `PROFILE = "full"`: 5 feature sets, 2 horizons,
8 models, 1000 search draws, 2000 bootstrap resamples. Total **11.05 hours**,
every notebook exiting 0.

| Notebook | Exit | Elapsed |
|---|---|---|
| `1_data_process` | 0 | 41 s |
| `2_classifiers` | 0 | 189 s |
| `3_sampling_ensemble_calibration` | 0 | 20362 s |
| `4_feature_cluster_explainability` | 0 | 292 s |
| `5_survival_analysis` | 0 | 18901 s |

Nothing failed anywhere: 80 of 80 search jobs completed with 0 failed candidates
and 0 convergence warnings on the final in-process refits, 450 of 450 sampler
comparisons returned `ok`, and no survival metric failed in any fold. Artifacts:
35 result tables, 4 manifests, 11 figures, 103 fitted models, 46 prediction
files. `tests/test_pipeline.py` passes with 26 tests and 2 skips.

The definitive primary result, the secondary feature-set results and the
survival extension are reported in the README under "Results". In one sentence:
the locked-pipeline comparison that isolates the feature effect shows no
incremental value for `CV17_THY_CONT` at either horizon on any of the four
metrics, and the paper-aligned ML indicator replicates the published C-index and
stratification while the thyroid biomarkers move four patients out of 878
between risk strata.

Three findings worth recording separately because they show the rebuilt
machinery doing what it was rebuilt to do:

- **Calibration was the largest practical gain.** Uncalibrated, the paper
  ensemble has an observed-to-expected ratio of 0.48 at 7 years with a slope of
  1.88 and an expected calibration error of 0.21. After training out-of-fold
  sigmoid calibration those become 0.99, 1.00 and 0.02. The old pipeline read
  decision curves off the uncalibrated scale.
- **The locked-pipeline comparison earns its place.** At 7 years the macro-F1
  difference is -0.0141 locked and +0.0212 unlocked, on the same test patients.
  The sign depends entirely on whether each arm may choose its own model family,
  sampler and threshold, which is exactly the confound the locked comparison
  removes.
- **The competing-risk correction is not cosmetic.** Aalen-Johansen puts 7-year
  cardiac cumulative incidence at 0.1173 against 0.1256 for `1 - Kaplan-Meier`,
  and 0.1487 against 0.1648 at 10 years, over 1399 competing non-cardiac deaths.

### Not run

The `primary` profile was not executed and is not needed: `full` contains the
complete prespecified primary contrast at both horizons plus the secondary
analyses, so running `primary` separately would repeat a strict subset of the
same work.

The paper's 5000-draw search budget was not used. Measured cost over these
search spaces is 8.45 h per cell at 5000 draws, so ten cells would have taken 84
hours; 1000 draws cost 17 hours in projection and 5.4 hours in fact. The paper
reports 10000 draws giving results very similar to 5000, which places the search
inside its plateau. The deviation is recorded in `config.py` beside the profile
and in the README.

## Known limitations

- The one comparison whose primary interval excludes zero is the Random Survival
  Forest delta-C-index of +0.0085 (0.0031, 0.0140). It should be read as a
  hypothesis, not a finding: the Cox reference on the same folds sees nothing,
  the time-dependent AUROC moves the other way, the effect is 1.1 percent
  relative, and the interval is anti-conservative for the reason given below.
- The strict cohort drops patients censored before the horizon, about 46 percent
  of the full cohort at 7 years, and median follow-up is shorter than the
  horizon. Its absolute risks are conditional on outcome observability. This is
  inherent in the paper's design and is why analysis B uses the full cohort.
- The sampler is chosen after the hyperparameter search rather than inside it, a
  documented deviation from the paper.
- Creatinine is excluded for missingness, so the baseline holds 17
  cardiovascular variables rather than the paper's 18.
- In the strict cohort censoring is a single administrative point at the horizon,
  so Uno's C-index and Harrell's C-index nearly coincide there and the
  strict-cohort survival sensitivity carries little information beyond the
  classification result.
- The bootstrap on out-of-fold predictions treats them as independent, although
  they share training folds. Intervals from analysis B are therefore
  anti-conservative; the frozen-test intervals in notebook 3 are not affected.
  This matters specifically for the Random Survival Forest result above.
- The search budget is 1000 draws rather than the paper's 5000, and the SVC
  space is bounded in `max_iter` and `C`. Both are measured decisions recorded
  in `config.py`, but both are deviations from the published protocol.
- Convergence warnings raised inside the parallel search workers cannot be
  captured from the parent process, so `final_fit_convergence_warnings` counts
  only the final in-process refit. Candidates that fail outright are counted
  separately and were zero throughout.

## Final active tree

```
config.py  utils.py  train.py  ensemble.py  survival.py
1_data_process.ipynb
2_classifiers.ipynb
3_sampling_ensemble_calibration.ipynb
4_feature_cluster_explainability.ipynb
5_survival_analysis.ipynb
README.md  requirements.txt  MIGRATION_REPORT.md  .gitignore
tests/test_pipeline.py
data/raw/  data/processed/
models/  predictions/  results/  figures/  cache/  docs/
legacy_cache/   (git-ignored, superseded artifacts)
```

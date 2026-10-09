# Confidence intervals for thesis Tables 5.2 to 5.5

Each interval is the 2.5th/97.5th percentile of 2000 patient bootstrap resamples
generated with seed 42. Fixed predictions are resampled; no model, sampler,
calibrator or threshold is selected inside a resample. These are marginal
intervals, not new comparisons or significance tests.

- Table 5.2: eight CV17 classifiers at both horizons. The completed analysis
  regenerated the five-fold training out-of-fold predictions using the frozen
  training partition, RandomOverSampler and threshold 0.5. Metrics and
  intervals use pooled OOF patients, not fold-average metrics. The intervals
  therefore condition on the regenerated cross-fitted predictions and do not
  measure refitting or model-selection uncertainty.
- Table 5.3: the four thesis configurations use frozen selected models and the
  final reference-voting bundle. No new tuning or selection occurs. Raw
  validation probabilities and threshold 0.5 are used. These intervals are
  descriptive and conditional on the selected configuration and validation
  sample; they do not correct for selection using those same validation
  patients.
- Table 5.4: the stored raw and sigmoid test probabilities are resampled.
  Calibration statistics are recomputed within each resample using the current
  ten-bin ECE definition. The calibration slope is an evaluation statistic
  re-estimated from the stored probabilities, not a refitted probability
  calibrator.
- Table 5.5: the existing calibrated reference-ensemble test probabilities are
  used for all five predictor sets and both horizons. Each validation-selected
  threshold remains fixed. AP is average precision, not trapezoidal PR area.

All point estimates were checked against the corresponding original result
files before interval export. Table 5.3 was additionally checked against the
four thesis values at four-decimal rounding tolerance. All 108 confidence
intervals use 2000 valid resamples.

The four `table_5_*_ci.csv` files contain full-precision numeric values.
`copy_paste_tables.md` and `tables_5_2_to_5_5_ci.tex` use direct rounding
from those verified full-precision values.

Three point labels in the current thesis PDF differ from direct three-decimal
rounding of the verified CSV values:

- 7-year CV17_THY_CONT AUROC: PDF 0.848, direct rounding 0.847
- 10-year CV17_THY_CONT AP: PDF 0.794, direct rounding 0.793
- 10-year CV17_THY_CONT_RATIO AUROC: PDF 0.857, direct rounding 0.856

These are display differences only and are documented in
`thesis_display_differences.csv`. The confidence intervals always use the
full-precision verified source estimates.

`original_artifact_integrity.json` records that the completed CI-generation
run did not alter the pre-existing analysis artifacts.

## Reproduction and intermediate artifacts

Run `python 13_table_confidence_intervals.py` from the repository root with
the unchanged full profile and pinned dependencies. All point estimates must
reproduce before any CI export; discrepancies stop the procedure and are
recorded in `point_estimate_verification.csv`.

Regenerated training OOF probabilities, patient identifiers and fold assignments
are retained in `predictions/table_5_2_*.csv` with generation signatures.
The four raw validation prediction sets are retained in
`predictions/table_5_3_*_validation.csv`. No new model selection occurs.
All bootstrap metric arrays are retained under `bootstrap/`.

`all_table_ci.csv` combines all 108 marginal intervals. `n_boot_used` records
finite replicates separately for each metric. Classification replicates with
one outcome class are excluded, consistently with the existing bootstrap.
Non-finite calibration slopes are excluded only from the slope interval;
other finite calibration statistics remain available. ECE reuses the
existing ten-bin implementation.

`run_manifest.json` records the seed, requested replicate count and checksums
of input data, fitted sources, predictions and implementation files.
`copy_paste_tables_thesis_labels.md` preserves the original PDF point labels
as an alternative to the direct-rounding Markdown and LaTeX tables. Both
views use the same verified numerical estimates and confidence intervals.

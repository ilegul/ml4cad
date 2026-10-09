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

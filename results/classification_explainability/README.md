# Classification interpretation extension

Executed supplementary SHAP, training-mean ablation and permutation importance
cover all eight individual classifier families and selected voting/stacking
options across five predictor sets and two horizons. Original analyses remain
available separately. Policy labels are retained when their functions coincide.

## Coverage

| Quantity | Count |
|:---|---:|
| Horizon/predictor-set groups | 10 |
| Individual fitted options | 80 |
| Voting options | 28 |
| Stacking policy options | 44 |
| Labelled options | 152 |
| Probability-scale SHAP explanations | 348 |
| Threshold/scale evaluation cases | 500 |
| Ablation target/case rows | 15,707 |
| Permutation target/case summaries | 15,707 |
| Individual permutation repeats | 157,070 |

SHAP uses the same 64 randomly sampled test patients and 16 training background
patients per horizon. Ablation and permutation use all 878 seven-year and
523 ten-year strict test patients. These are different evaluation samples.

## Files

- `shap/all_results.html`: filtered report with every explanation's figure.
- `shap/all_explanations.csv`: probability scales, attribution shares and ranks.
- `shap/all_feature_importance.csv`: all individual predictor contributions.
- `shap/groups/`: patient identifiers, original inputs, output metadata,
  compressed attribution arrays and source manifests. Arrays include SHAP
  values, base values, probabilities and path variability.
- `shap/replay_verification.json`: executed direct-versus-batched SHAP check.
- `shap/structural_checks.json`: saved AdaBoost thyroid-split inspection.
- `importance/all_results.html`: all targets, scales and classification thresholds.
- `importance/all_configurations.csv` and `all_baseline_metrics.csv`: frozen
  evaluation definitions and unperturbed scores.
- `importance/all_ablation.csv`: target-level drops and paired macro-F1 intervals.
- `importance/all_permutation.csv`: mean drops and SD across ten repeats.
- `importance/all_permutation_repeats.csv.gz`: complete repeated scores.
- `importance/thyroid_block_all_configurations.csv`: complete-thyroid block summaries.
- `importance/groups/`: training clusters, reference probabilities, patient
  identifiers, training means, permutation orders and detailed tables.
- `importance/reference_reproduction.json`: checks against original outputs.
- `matched_C_validation_selection.csv`: validation-only fixed-members C selection.
- `verification.csv`: frozen probability, patient alignment and numerical checks.
- `artifact_inventory.csv`: byte counts and SHA-256 checksums for extension artifacts.

Meta-models and calibrators are preserved in the separately versioned
`cache/classification_explainability/prepared_h*_*.joblib` bundles, linked
to original fitted classifiers and OOF caches through development signatures
and training identifiers. Corresponding reselected-C test probabilities are
in `predictions/classification_explainability/`.

## Interpretation

SHAP uses an independent background, an identity probability link and two
antithetic forward/backward permutation cycles, with seed 20261002 plus the
horizon. It can break dependence and deterministic biomarker/state/ratio
relationships. Additivity and path variability do not establish Monte Carlo
convergence. Attribution share is not an AUROC or accuracy increment.

Ablation replaces predictors with training means without refitting. Binary
means can be fractional. Joint permutation preserves dependence within the
permuted block but can break relationships with retained predictors. Positive
drops denote deterioration, with the Brier direction reversed to retain this
convention. Negative drops are preserved.

Macro-F1 ablation intervals use 2000 paired patient bootstrap draws with fixed
fits and thresholds. They are descriptive and unadjusted for multiplicity.
AUROC, AP and Brier ablation drops are point estimates. Permutation SD
describes variation across ten repeats, not population uncertainty. These
quantities do not replace incremental-value comparisons against CV17.

Individual sigmoid mappings use stored training OOF probabilities; individual
thresholds use validation. Voting and stacking retain saved calibrators and
thresholds. Voting raw_threshold applies its saved calibrated threshold to raw
probabilities. The existing same_baseline stack retains baseline membership
and C. The separate same_members_reselected_C option selects C from 0.01,
0.1, 1, 10 and 100 using validation raw macro-F1 at 0.5, with AUROC and
configuration name as tie breakers. Calibration uses nested training OOF
inputs and threshold selection uses validation. The policies remain distinct.

## Reproduction

Use the pinned environment, original fitted caches and versioned extension
bundles. Run `python 11_classification_explainability.py --verify` to check
sources and reconstruct probabilities, and `--summary` to rebuild reports.
`--stage shap` and `--stage importance` retain completed compatible groups.
Different resampling settings do not silently reuse completed outputs.
Missing/stale sources raise an error without fitting base classifiers.
The optional `--pilot --patients 6` validates seeded replay separately.

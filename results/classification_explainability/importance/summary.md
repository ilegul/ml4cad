# Classification ablation and permutation extension

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

Completed groups: 10; evaluation cases: 500; ablations: 15707; permutation repeats: 157070.

All continuous-thyroid sigmoid ablation macro-F1 intervals include zero. This does not demonstrate equivalence.

Unadjusted intervals excluding zero, complete thyroid block:

| Horizon | Predictor set | Kind | Option | Macro-F1 drop [95% CI] |
|---:|:---|:---|:---|---:|
| 7 | CV17_THY_CONT_RATIO | individual | GradientBoosting | +0.0351 [+0.0070, +0.0623] |
| 7 | CV17_THY_STATES | individual | GradientBoosting | +0.0208 [+0.0029, +0.0405] |
| 7 | CV17_THY_STATES | individual | MLP | -0.0272 [-0.0512, -0.0040] |
| 7 | CV17_THY_STATES | stacking | locked_baseline | -0.0215 [-0.0385, -0.0075] |
| 10 | CV17_THY_CONT_RATIO | voting | top3_train_cv | -0.0112 [-0.0212, -0.0036] |
| 10 | CV17_THY_STATES | individual | RandomForest | -0.0108 [-0.0197, -0.0036] |

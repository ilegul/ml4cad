# Classification SHAP extension

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

Completed groups: 10; labelled options: 152; explanations: 348.

Continuous thyroid predictors, sigmoid attribution share (%):

| Classifier | Seven years | Ten years |
|:---|---:|---:|
| AdaBoost | 10.14 | 5.25 |
| GradientBoosting | 17.66 | 11.38 |
| KNeighbors | 5.56 | 4.35 |
| LogisticRegression | 1.68 | 2.15 |
| MLP | 2.34 | 6.25 |
| RandomForest | 15.21 | 9.48 |
| SVC | 2.36 | 3.13 |
| XGBoost | 16.04 | 2.83 |

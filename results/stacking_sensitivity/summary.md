# Supplementary stacking sensitivity

All membership, meta-model settings, calibrators and thresholds were fixed before new test evaluation.
Sigmoid and isotonic calibration use nested training OOF predictions of the entire stack.
The analysis is post hoc: intervals condition on selected configurations and the stored split.
They do not account for configuration-selection uncertainty or previous test inspection.
Holm adjustment covers the independent and same-baseline macro-F1 comparisons across feature sets
and horizons, separately for each probability/threshold mode. Other intervals are unadjusted.

## Selected configurations

- 7 years, CV17: KNeighbors+RandomForest+AdaBoost+MLP;C=1; validation macro-F1 0.7480.
- 7 years, CV17_THY_CONT: LogisticRegression+KNeighbors+AdaBoost;C=1; validation macro-F1 0.7362.
- 7 years, CV17_THY_STATES: KNeighbors+RandomForest+MLP+GradientBoosting;C=0.1; validation macro-F1 0.7436.
- 7 years, CV17_THY_CONT_STATES: MLP+GradientBoosting+XGBoost;C=0.1; validation macro-F1 0.7451.
- 7 years, CV17_THY_CONT_RATIO: AdaBoost+MLP;C=100; validation macro-F1 0.7418.
- 10 years, CV17: KNeighbors+AdaBoost+MLP+GradientBoosting;C=0.01; validation macro-F1 0.7519.
- 10 years, CV17_THY_CONT: LogisticRegression+SVC+KNeighbors+GradientBoosting+XGBoost;C=10; validation macro-F1 0.7459.
- 10 years, CV17_THY_STATES: GradientBoosting+XGBoost;C=0.01; validation macro-F1 0.7559.
- 10 years, CV17_THY_CONT_STATES: RandomForest+AdaBoost+MLP+GradientBoosting;C=10; validation macro-F1 0.7483.
- 10 years, CV17_THY_CONT_RATIO: LogisticRegression+SVC+AdaBoost;C=1; validation macro-F1 0.7445.

## Sigmoid calibration

| Horizon | Features | Policy | Test macro-F1 | AUROC | AP | Brier |
|---|---|---|---:|---:|---:|---:|
| 7 | CV17 | own | 0.7298 | 0.8426 | 0.6238 | 0.1067 |
| 7 | CV17 | paper | 0.7350 | 0.8422 | 0.6034 | 0.1091 |
| 7 | CV17_THY_CONT | own | 0.7554 | 0.8452 | 0.6027 | 0.1092 |
| 7 | CV17_THY_CONT | paper | 0.7571 | 0.8476 | 0.6237 | 0.1069 |
| 7 | CV17_THY_CONT | same_baseline | 0.7545 | 0.8468 | 0.6266 | 0.1070 |
| 7 | CV17_THY_CONT | locked_baseline | 0.7294 | 0.8485 | 0.6414 | 0.1052 |
| 7 | CV17_THY_STATES | own | 0.7232 | 0.8437 | 0.6141 | 0.1070 |
| 7 | CV17_THY_STATES | paper | 0.7375 | 0.8405 | 0.5866 | 0.1114 |
| 7 | CV17_THY_STATES | same_baseline | 0.7371 | 0.8426 | 0.6087 | 0.1085 |
| 7 | CV17_THY_STATES | locked_baseline | 0.7196 | 0.8401 | 0.6109 | 0.1084 |
| 7 | CV17_THY_CONT_STATES | own | 0.7481 | 0.8440 | 0.6270 | 0.1071 |
| 7 | CV17_THY_CONT_STATES | paper | 0.7534 | 0.8455 | 0.6265 | 0.1070 |
| 7 | CV17_THY_CONT_STATES | same_baseline | 0.7520 | 0.8471 | 0.6273 | 0.1067 |
| 7 | CV17_THY_CONT_STATES | locked_baseline | 0.7219 | 0.8466 | 0.6340 | 0.1060 |
| 7 | CV17_THY_CONT_RATIO | own | 0.7362 | 0.8419 | 0.5988 | 0.1109 |
| 7 | CV17_THY_CONT_RATIO | paper | 0.7536 | 0.8479 | 0.6221 | 0.1076 |
| 7 | CV17_THY_CONT_RATIO | same_baseline | 0.7541 | 0.8466 | 0.6196 | 0.1083 |
| 7 | CV17_THY_CONT_RATIO | locked_baseline | 0.7267 | 0.8457 | 0.6353 | 0.1067 |
| 10 | CV17 | own | 0.7760 | 0.8577 | 0.7905 | 0.1496 |
| 10 | CV17 | paper | 0.7505 | 0.8526 | 0.7816 | 0.1504 |
| 10 | CV17_THY_CONT | own | 0.7720 | 0.8509 | 0.7863 | 0.1506 |
| 10 | CV17_THY_CONT | paper | 0.7537 | 0.8585 | 0.7958 | 0.1480 |
| 10 | CV17_THY_CONT | same_baseline | 0.7652 | 0.8595 | 0.7985 | 0.1471 |
| 10 | CV17_THY_CONT | locked_baseline | 0.7768 | 0.8588 | 0.7929 | 0.1474 |
| 10 | CV17_THY_STATES | own | 0.7702 | 0.8555 | 0.7877 | 0.1499 |
| 10 | CV17_THY_STATES | paper | 0.7675 | 0.8542 | 0.7847 | 0.1502 |
| 10 | CV17_THY_STATES | same_baseline | 0.7647 | 0.8548 | 0.7855 | 0.1506 |
| 10 | CV17_THY_STATES | locked_baseline | 0.7624 | 0.8588 | 0.7945 | 0.1499 |
| 10 | CV17_THY_CONT_STATES | own | 0.7688 | 0.8575 | 0.7920 | 0.1479 |
| 10 | CV17_THY_CONT_STATES | paper | 0.7673 | 0.8554 | 0.7913 | 0.1482 |
| 10 | CV17_THY_CONT_STATES | same_baseline | 0.7638 | 0.8511 | 0.7876 | 0.1503 |
| 10 | CV17_THY_CONT_STATES | locked_baseline | 0.7673 | 0.8568 | 0.7952 | 0.1485 |
| 10 | CV17_THY_CONT_RATIO | own | 0.7679 | 0.8533 | 0.7906 | 0.1493 |
| 10 | CV17_THY_CONT_RATIO | paper | 0.7597 | 0.8548 | 0.7888 | 0.1494 |
| 10 | CV17_THY_CONT_RATIO | same_baseline | 0.7569 | 0.8554 | 0.7929 | 0.1484 |
| 10 | CV17_THY_CONT_RATIO | locked_baseline | 0.7709 | 0.8581 | 0.7924 | 0.1474 |

## Thyroid differences

- 7 years, CV17_THY_CONT, own: +0.0256, 95% CI [-0.0025, +0.0544].
- 7 years, CV17_THY_CONT, paper: +0.0221, 95% CI [-0.0014, +0.0478].
- 7 years, CV17_THY_CONT, same_baseline: +0.0247, 95% CI [+0.0012, +0.0488].
- 7 years, CV17_THY_CONT, locked_baseline: -0.0004, 95% CI [-0.0239, +0.0220].
- 7 years, CV17_THY_STATES, own: -0.0066, 95% CI [-0.0374, +0.0236].
- 7 years, CV17_THY_STATES, paper: +0.0025, 95% CI [-0.0224, +0.0276].
- 7 years, CV17_THY_STATES, same_baseline: +0.0073, 95% CI [-0.0155, +0.0307].
- 7 years, CV17_THY_STATES, locked_baseline: -0.0103, 95% CI [-0.0271, +0.0054].
- 7 years, CV17_THY_CONT_STATES, own: +0.0182, 95% CI [-0.0067, +0.0440].
- 7 years, CV17_THY_CONT_STATES, paper: +0.0184, 95% CI [-0.0104, +0.0484].
- 7 years, CV17_THY_CONT_STATES, same_baseline: +0.0221, 95% CI [-0.0065, +0.0546].
- 7 years, CV17_THY_CONT_STATES, locked_baseline: -0.0079, 95% CI [-0.0330, +0.0156].
- 7 years, CV17_THY_CONT_RATIO, own: +0.0064, 95% CI [-0.0160, +0.0305].
- 7 years, CV17_THY_CONT_RATIO, paper: +0.0185, 95% CI [-0.0118, +0.0490].
- 7 years, CV17_THY_CONT_RATIO, same_baseline: +0.0243, 95% CI [-0.0028, +0.0513].
- 7 years, CV17_THY_CONT_RATIO, locked_baseline: -0.0031, 95% CI [-0.0265, +0.0191].
- 10 years, CV17_THY_CONT, own: -0.0040, 95% CI [-0.0241, +0.0140].
- 10 years, CV17_THY_CONT, paper: +0.0032, 95% CI [-0.0147, +0.0224].
- 10 years, CV17_THY_CONT, same_baseline: -0.0108, 95% CI [-0.0330, +0.0105].
- 10 years, CV17_THY_CONT, locked_baseline: +0.0008, 95% CI [-0.0172, +0.0186].
- 10 years, CV17_THY_STATES, own: -0.0058, 95% CI [-0.0279, +0.0142].
- 10 years, CV17_THY_STATES, paper: +0.0170, 95% CI [-0.0020, +0.0355].
- 10 years, CV17_THY_STATES, same_baseline: -0.0113, 95% CI [-0.0289, +0.0059].
- 10 years, CV17_THY_STATES, locked_baseline: -0.0136, 95% CI [-0.0316, +0.0046].
- 10 years, CV17_THY_CONT_STATES, own: -0.0072, 95% CI [-0.0280, +0.0135].
- 10 years, CV17_THY_CONT_STATES, paper: +0.0168, 95% CI [-0.0009, +0.0343].
- 10 years, CV17_THY_CONT_STATES, same_baseline: -0.0122, 95% CI [-0.0335, +0.0075].
- 10 years, CV17_THY_CONT_STATES, locked_baseline: -0.0087, 95% CI [-0.0291, +0.0118].
- 10 years, CV17_THY_CONT_RATIO, own: -0.0081, 95% CI [-0.0297, +0.0134].
- 10 years, CV17_THY_CONT_RATIO, paper: +0.0092, 95% CI [-0.0093, +0.0272].
- 10 years, CV17_THY_CONT_RATIO, same_baseline: -0.0191, 95% CI [-0.0424, +0.0025].
- 10 years, CV17_THY_CONT_RATIO, locked_baseline: -0.0051, 95% CI [-0.0220, +0.0121].

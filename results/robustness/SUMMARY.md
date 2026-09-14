# Robustness analyses

Post hoc analyses of the frozen results; nothing here alters the main analysis. Produced by `7_robustness.py`; see the module docstring for the design of each phase.

## Phase status

| phase | ok | elapsed_s |
|---|---|---|
| D3_k2_check | True | 0 |
| A | True | 0 |
| B1_cox | True | 0 |
| B2_cif | True | 0 |
| B3_oob_cox | True | 0 |
| B4_oob_cif | True | 1944 |
| D1_k3 | True | 5277 |
| D3_k3 | True | 192 |
| B5_rsf | True | 5249 |
| D1_k5 | True | 8965 |
| D2_yardstick | True | 80 |
| D3_k5 | True | 148 |

## Reproduction gate (k = 2 flow against the frozen artifacts)

| horizon | artifact | n | max_abs_diff | mismatches |
|---|---|---|---|---|
| 7 | h7_CV17_paper | 878 | 0.00000000 | 0 |
| 7 | h7_CV17_THY_CONT_paper | 878 | 0.00000000 | 0 |
| 7 | h7_CV17_THY_CONT_paper_locked | 878 | 0.00000000 | 0 |
| 10 | h10_CV17_paper | 523 | 0.00000000 | 0 |
| 10 | h10_CV17_THY_CONT_paper | 523 | 0.00000000 | 0 |
| 10 | h10_CV17_THY_CONT_paper_locked | 523 | 0.00000000 | 0 |
| all | incremental_value.delta | 16 | 0.00000000 | 0 |
| all | incremental_value.ci_lo | 16 | 0.00000000 | 0 |
| all | incremental_value.ci_hi | 16 | 0.00000000 | 0 |
| all | sampler_choice | 12 | 0.00000000 | 0 |

## A. Repeated partitions of the fixed-horizon analysis

`reported_pct` is the share of partitions with a delta at or below the frozen-test value.

| horizon | comparison | metric | n | mean | sd | p2_5 | p97_5 | share_pos | share_ci_excl0 | reported_delta | reported_lo | reported_hi | p_bootstrap | reported_se | sd_over_reported_se | reported_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | independently optimized | auprc | 50 | 0.0047 | 0.0075 | -0.0051 | 0.0216 | 0.7000 | 0.1000 | 0.0069 | -0.0120 | 0.0262 | 0.4940 | 0.0097 | 0.7744 | 0.7000 |
| 7 | independently optimized | brier | 50 | -0.0009 | 0.0008 | -0.0025 | 0.0004 | 0.1200 | 0.0800 | -0.0009 | -0.0030 | 0.0014 | 0.4860 | 0.0011 | 0.7297 | 0.4800 |
| 7 | independently optimized | f1_macro | 50 | 0.0040 | 0.0123 | -0.0138 | 0.0263 | 0.6200 | 0.0600 | 0.0212 | 0.0005 | 0.0431 | 0.0450 | 0.0109 | 1.1346 | 0.9600 |
| 7 | independently optimized | roc_auc | 50 | 0.0024 | 0.0026 | -0.0024 | 0.0073 | 0.7800 | 0.1000 | 0.0034 | -0.0029 | 0.0102 | 0.3190 | 0.0033 | 0.7760 | 0.6800 |
| 7 | locked pipeline | auprc | 50 | 0.0054 | 0.0069 | -0.0051 | 0.0197 | 0.8000 | 0.0600 | 0.0041 | -0.0096 | 0.0173 | 0.5750 | 0.0069 | 1.0086 | 0.4400 |
| 7 | locked pipeline | brier | 50 | -0.0008 | 0.0008 | -0.0025 | 0.0007 | 0.1200 | 0.1400 | -0.0003 | -0.0022 | 0.0015 | 0.7600 | 0.0009 | 0.8252 | 0.7600 |
| 7 | locked pipeline | f1_macro | 50 | 0.0029 | 0.0081 | -0.0119 | 0.0166 | 0.7000 | 0.0600 | -0.0141 | -0.0308 | 0.0012 | 0.0760 | 0.0082 | 0.9927 | 0.0200 |
| 7 | locked pipeline | roc_auc | 50 | 0.0023 | 0.0021 | -0.0015 | 0.0064 | 0.8800 | 0.1000 | 0.0026 | -0.0026 | 0.0081 | 0.3740 | 0.0027 | 0.7613 | 0.5400 |
| 10 | independently optimized | auprc | 50 | 0.0030 | 0.0039 | -0.0052 | 0.0094 | 0.8000 | 0.0800 | 0.0088 | 0.0009 | 0.0175 | 0.0280 | 0.0042 | 0.9226 | 0.9200 |
| 10 | independently optimized | brier | 50 | -0.0012 | 0.0012 | -0.0028 | 0.0016 | 0.2200 | 0.0600 | -0.0019 | -0.0046 | 0.0010 | 0.1790 | 0.0014 | 0.8535 | 0.2800 |
| 10 | independently optimized | f1_macro | 50 | -0.0019 | 0.0099 | -0.0215 | 0.0145 | 0.4800 | 0.0600 | 0.0028 | -0.0125 | 0.0180 | 0.7330 | 0.0078 | 1.2656 | 0.6800 |
| 10 | independently optimized | roc_auc | 50 | 0.0023 | 0.0024 | -0.0020 | 0.0065 | 0.8400 | 0.0800 | 0.0034 | -0.0017 | 0.0082 | 0.1780 | 0.0025 | 0.9537 | 0.6800 |
| 10 | locked pipeline | auprc | 50 | 0.0007 | 0.0043 | -0.0088 | 0.0070 | 0.6000 | 0.0200 | 0.0048 | -0.0033 | 0.0132 | 0.2440 | 0.0042 | 1.0300 | 0.8000 |
| 10 | locked pipeline | brier | 50 | -0.0004 | 0.0015 | -0.0033 | 0.0030 | 0.3600 | 0.0600 | -0.0010 | -0.0040 | 0.0020 | 0.4950 | 0.0015 | 0.9894 | 0.3200 |
| 10 | locked pipeline | f1_macro | 50 | 0.0006 | 0.0086 | -0.0150 | 0.0182 | 0.5200 | 0.0600 | -0.0057 | -0.0204 | 0.0082 | 0.4240 | 0.0073 | 1.1831 | 0.3000 |
| 10 | locked pipeline | roc_auc | 50 | 0.0012 | 0.0025 | -0.0041 | 0.0063 | 0.7800 | 0.0400 | 0.0010 | -0.0046 | 0.0061 | 0.6970 | 0.0027 | 0.9304 | 0.4600 |

## B1, B2, B5. Survival analyses: dependence on the fold partition

| analysis | n_partitions | mean | sd_across | within_se | sd_over_within_se | share_pos | share_ci_excl0 | reported | reported_lo | reported_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| Analysis B, Cox | 10 | 0.0019 | 0.0004 | 0.0011 | 0.3731 | 1.0000 | 0.2000 | 0.0012 | -0.0014 | 0.0036 |
| Analysis B, RSF (default configuration) | 10 | 0.0096 | 0.0010 | 0.0028 | 0.3455 | 1.0000 | 1.0000 | 0.0085 | 0.0031 | 0.0140 |
| Analysis C, 7 years | 10 | 0.0013 | 0.0006 | 0.0013 | 0.4619 | 1.0000 | 0.1000 | 0.0014 | -0.0011 | 0.0034 |
| Analysis C, 10 years | 10 | 0.0008 | 0.0005 | 0.0012 | 0.4494 | 0.9000 | 0.0000 | 0.0007 | -0.0017 | 0.0026 |

## B3, B4. Bootstrap with model refitting (out-of-bag evaluation)

| analysis | n_draws | mean | sd | p2_5 | p97_5 | p_two_sided | mean_n_oob | reported | reported_lo | reported_hi | width_ratio | sd_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Analysis B, Cox | 2000 | 0.0019 | 0.0017 | -0.0018 | 0.0051 | 0.2060 | 2966.5225 | 0.0012 | -0.0014 | 0.0036 | 1.3883 | 1.3573 |
| Analysis C, 7 years | 2000 | 0.0016 | 0.0019 | -0.0023 | 0.0053 | 0.3880 | 2966.5225 | 0.0014 | -0.0011 | 0.0034 | 1.6586 | 1.6481 |
| Analysis C, 10 years | 2000 | 0.0010 | 0.0020 | -0.0031 | 0.0047 | 0.5840 | 2966.5225 | 0.0007 | -0.0017 | 0.0026 | 1.8413 | 1.8441 |

## D1. Search winners per inner-fold count

| k | horizon | feature_set | model | best_cv_f1_macro | elapsed_seconds | n_failed_candidates | best_params | same_as_k2 |
|---|---|---|---|---|---|---|---|---|
| 2 | 7 | CV17 | AdaBoost | 0.7113 | 478.0100 | 0 | {"model__learning_rate": 0.3674722291218063, "model__n_estimators": 249} | True |
| 3 | 7 | CV17 | AdaBoost | 0.7110 | 615.7300 | 0 | {"model__learning_rate": 1.1595183275654903, "model__n_estimators": 201} | False |
| 5 | 7 | CV17 | AdaBoost | 0.7110 | 1009.4900 | 0 | {"model__learning_rate": 0.36672729149665356, "model__n_estimators": 328} | False |
| 2 | 7 | CV17 | LogisticRegression | 0.7027 | 19.1300 | 0 | {"model__C": 0.0033857396754547486, "model__l1_ratio": 0.15156937158767358, "model__penalty": "elasticnet", "model__solver": "saga"} | True |
| 3 | 7 | CV17 | LogisticRegression | 0.7047 | 31.5600 | 0 | {"model__C": 0.001011922841853118, "model__penalty": "l2", "model__solver": "lbfgs"} | False |
| 5 | 7 | CV17 | LogisticRegression | 0.7040 | 38.5300 | 0 | {"model__C": 0.004982752357076452, "model__l1_ratio": 0.29214464853521815, "model__penalty": "elasticnet", "model__solver": "saga"} | False |
| 2 | 7 | CV17 | RandomForest | 0.7171 | 688.8500 | 0 | {"model__max_depth": 8, "model__max_features": "sqrt", "model__min_samples_leaf": 3, "model__min_samples_split": 7, "model__n_estimators": 555} | True |
| 3 | 7 | CV17 | RandomForest | 0.7109 | 966.9400 | 0 | {"model__max_depth": 8, "model__max_features": "sqrt", "model__min_samples_leaf": 3, "model__min_samples_split": 2, "model__n_estimators": 203} | False |
| 5 | 7 | CV17 | RandomForest | 0.7124 | 1468.1300 | 0 | {"model__max_depth": 20, "model__max_features": 0.3, "model__min_samples_leaf": 4, "model__min_samples_split": 4, "model__n_estimators": 588} | False |
| 2 | 7 | CV17_THY_CONT | AdaBoost | 0.7034 | 510.8200 | 0 | {"model__learning_rate": 0.24847215786017884, "model__n_estimators": 159} | True |
| 3 | 7 | CV17_THY_CONT | AdaBoost | 0.7096 | 665.2500 | 0 | {"model__learning_rate": 1.1478834275351926, "model__n_estimators": 412} | False |
| 5 | 7 | CV17_THY_CONT | AdaBoost | 0.7084 | 1185.5100 | 0 | {"model__learning_rate": 0.17700888306825902, "model__n_estimators": 377} | False |
| 2 | 7 | CV17_THY_CONT | LogisticRegression | 0.7014 | 11.0400 | 0 | {"model__C": 0.0034110937106195086, "model__l1_ratio": 0.140018538660449, "model__penalty": "elasticnet", "model__solver": "saga"} | True |
| 3 | 7 | CV17_THY_CONT | LogisticRegression | 0.7031 | 14.0500 | 0 | {"model__C": 0.006723053807889171, "model__l1_ratio": 0.033879160246594986, "model__penalty": "elasticnet", "model__solver": "saga"} | False |
| 5 | 7 | CV17_THY_CONT | LogisticRegression | 0.7019 | 25.3900 | 0 | {"model__C": 0.004982752357076452, "model__l1_ratio": 0.29214464853521815, "model__penalty": "elasticnet", "model__solver": "saga"} | False |
| 2 | 7 | CV17_THY_CONT | RandomForest | 0.7310 | 664.8400 | 0 | {"model__max_depth": 12, "model__max_features": "sqrt", "model__min_samples_leaf": 1, "model__min_samples_split": 13, "model__n_estimators": 775} | True |
| 3 | 7 | CV17_THY_CONT | RandomForest | 0.7252 | 909.7800 | 0 | {"model__max_depth": null, "model__max_features": "sqrt", "model__min_samples_leaf": 3, "model__min_samples_split": 2, "model__n_estimators": 669} | False |
| 5 | 7 | CV17_THY_CONT | RandomForest | 0.7277 | 1660.9600 | 0 | {"model__max_depth": 20, "model__max_features": "log2", "model__min_samples_leaf": 2, "model__min_samples_split": 10, "model__n_estimators": 102} | False |
| 2 | 10 | CV17 | AdaBoost | 0.7791 | 336.0500 | 0 | {"model__learning_rate": 0.33156365156332046, "model__n_estimators": 257} | True |
| 3 | 10 | CV17 | AdaBoost | 0.7773 | 430.6400 | 0 | {"model__learning_rate": 1.1478834275351926, "model__n_estimators": 412} | False |
| 5 | 10 | CV17 | AdaBoost | 0.7774 | 741.3000 | 0 | {"model__learning_rate": 1.226395208819354, "model__n_estimators": 350} | False |
| 2 | 10 | CV17 | LogisticRegression | 0.7790 | 6.5400 | 0 | {"model__C": 0.0014340571602734114, "model__penalty": "l2", "model__solver": "lbfgs"} | True |
| 3 | 10 | CV17 | LogisticRegression | 0.7865 | 9.0300 | 0 | {"model__C": 0.01148746495643705, "model__penalty": "l2", "model__solver": "lbfgs"} | False |
| 5 | 10 | CV17 | LogisticRegression | 0.7814 | 15.2300 | 0 | {"model__C": 0.005756046128009331, "model__l1_ratio": 0.06134962711066816, "model__penalty": "elasticnet", "model__solver": "saga"} | False |
| 2 | 10 | CV17 | RandomForest | 0.7856 | 394.5600 | 0 | {"model__max_depth": 8, "model__max_features": "log2", "model__min_samples_leaf": 2, "model__min_samples_split": 2, "model__n_estimators": 622} | True |
| 3 | 10 | CV17 | RandomForest | 0.7844 | 530.8700 | 0 | {"model__max_depth": null, "model__max_features": "sqrt", "model__min_samples_leaf": 3, "model__min_samples_split": 7, "model__n_estimators": 616} | False |
| 5 | 10 | CV17 | RandomForest | 0.7816 | 909.3800 | 0 | {"model__max_depth": 20, "model__max_features": "log2", "model__min_samples_leaf": 2, "model__min_samples_split": 10, "model__n_estimators": 102} | False |
| 2 | 10 | CV17_THY_CONT | AdaBoost | 0.7790 | 355.7000 | 0 | {"model__learning_rate": 0.2718359536006563, "model__n_estimators": 499} | True |
| 3 | 10 | CV17_THY_CONT | AdaBoost | 0.7779 | 477.1800 | 0 | {"model__learning_rate": 0.38590426192511307, "model__n_estimators": 192} | False |
| 5 | 10 | CV17_THY_CONT | AdaBoost | 0.7811 | 817.0700 | 0 | {"model__learning_rate": 0.5843642821496238, "model__n_estimators": 599} | False |
| 2 | 10 | CV17_THY_CONT | LogisticRegression | 0.7844 | 7.8500 | 0 | {"model__C": 0.005756046128009331, "model__l1_ratio": 0.06134962711066816, "model__penalty": "elasticnet", "model__solver": "saga"} | True |
| 3 | 10 | CV17_THY_CONT | LogisticRegression | 0.7862 | 16.2900 | 0 | {"model__C": 0.008353233151447025, "model__l1_ratio": 0.09033696057760787, "model__penalty": "elasticnet", "model__solver": "saga"} | False |
| 5 | 10 | CV17_THY_CONT | LogisticRegression | 0.7842 | 22.5300 | 0 | {"model__C": 0.014165465220555132, "model__l1_ratio": 0.20618616451842642, "model__penalty": "elasticnet", "model__solver": "saga"} | False |
| 2 | 10 | CV17_THY_CONT | RandomForest | 0.7823 | 455.1300 | 0 | {"model__max_depth": 12, "model__max_features": "log2", "model__min_samples_leaf": 3, "model__min_samples_split": 11, "model__n_estimators": 419} | True |
| 3 | 10 | CV17_THY_CONT | RandomForest | 0.7841 | 608.0900 | 0 | {"model__max_depth": null, "model__max_features": 0.3, "model__min_samples_leaf": 1, "model__min_samples_split": 22, "model__n_estimators": 617} | False |
| 5 | 10 | CV17_THY_CONT | RandomForest | 0.7825 | 1071.3300 | 0 | {"model__max_depth": 8, "model__max_features": "sqrt", "model__min_samples_leaf": 4, "model__min_samples_split": 11, "model__n_estimators": 166} | False |

## D2. Training-only yardstick (5 x 3-fold repeated CV, macro-F1 at 0.5)

| horizon | feature_set | model | k | f1_macro_mean | f1_macro_std | roc_auc_mean | roc_auc_std | f1_diff_vs_k2 | f1_diff_vs_k2_sd |
|---|---|---|---|---|---|---|---|---|---|
| 7 | CV17 | AdaBoost | 2 | 0.6994 | 0.0090 | 0.8330 | 0.0071 |  |  |
| 7 | CV17 | AdaBoost | 3 | 0.7004 | 0.0116 | 0.8272 | 0.0076 | 0.0010 | 0.0072 |
| 7 | CV17 | AdaBoost | 5 | 0.6993 | 0.0081 | 0.8334 | 0.0072 | -0.0001 | 0.0031 |
| 7 | CV17 | LogisticRegression | 2 | 0.6997 | 0.0090 | 0.8365 | 0.0082 |  |  |
| 7 | CV17 | LogisticRegression | 3 | 0.7029 | 0.0097 | 0.8355 | 0.0087 | 0.0032 | 0.0062 |
| 7 | CV17 | LogisticRegression | 5 | 0.6974 | 0.0078 | 0.8356 | 0.0081 | -0.0023 | 0.0049 |
| 7 | CV17 | RandomForest | 2 | 0.7051 | 0.0095 | 0.8287 | 0.0077 |  |  |
| 7 | CV17 | RandomForest | 3 | 0.7054 | 0.0098 | 0.8287 | 0.0082 | 0.0003 | 0.0063 |
| 7 | CV17 | RandomForest | 5 | 0.7066 | 0.0109 | 0.8248 | 0.0102 | 0.0015 | 0.0097 |
| 7 | CV17 | paper_ensemble | 2 | 0.7087 | 0.0064 | 0.8372 | 0.0070 |  |  |
| 7 | CV17 | paper_ensemble | 3 | 0.7114 | 0.0082 | 0.8368 | 0.0075 | 0.0027 | 0.0051 |
| 7 | CV17 | paper_ensemble | 5 | 0.7124 | 0.0096 | 0.8366 | 0.0079 | 0.0036 | 0.0073 |
| 7 | CV17_THY_CONT | AdaBoost | 2 | 0.6962 | 0.0080 | 0.8314 | 0.0062 |  |  |
| 7 | CV17_THY_CONT | AdaBoost | 3 | 0.6966 | 0.0095 | 0.8233 | 0.0088 | 0.0005 | 0.0105 |
| 7 | CV17_THY_CONT | AdaBoost | 5 | 0.6993 | 0.0111 | 0.8338 | 0.0070 | 0.0032 | 0.0052 |
| 7 | CV17_THY_CONT | LogisticRegression | 2 | 0.6994 | 0.0087 | 0.8360 | 0.0089 |  |  |
| 7 | CV17_THY_CONT | LogisticRegression | 3 | 0.7004 | 0.0113 | 0.8368 | 0.0087 | 0.0010 | 0.0055 |
| 7 | CV17_THY_CONT | LogisticRegression | 5 | 0.6968 | 0.0078 | 0.8348 | 0.0087 | -0.0025 | 0.0053 |
| 7 | CV17_THY_CONT | RandomForest | 2 | 0.7150 | 0.0118 | 0.8336 | 0.0088 |  |  |
| 7 | CV17_THY_CONT | RandomForest | 3 | 0.7189 | 0.0093 | 0.8353 | 0.0089 | 0.0039 | 0.0075 |
| 7 | CV17_THY_CONT | RandomForest | 5 | 0.7172 | 0.0115 | 0.8317 | 0.0081 | 0.0022 | 0.0078 |
| 7 | CV17_THY_CONT | paper_ensemble | 2 | 0.7202 | 0.0105 | 0.8411 | 0.0077 |  |  |
| 7 | CV17_THY_CONT | paper_ensemble | 3 | 0.7239 | 0.0106 | 0.8424 | 0.0082 | 0.0037 | 0.0068 |
| 7 | CV17_THY_CONT | paper_ensemble | 5 | 0.7203 | 0.0097 | 0.8406 | 0.0074 | 0.0000 | 0.0048 |
| 10 | CV17 | AdaBoost | 2 | 0.7708 | 0.0154 | 0.8567 | 0.0089 |  |  |
| 10 | CV17 | AdaBoost | 3 | 0.7682 | 0.0122 | 0.8516 | 0.0080 | -0.0027 | 0.0100 |
| 10 | CV17 | AdaBoost | 5 | 0.7692 | 0.0128 | 0.8528 | 0.0083 | -0.0017 | 0.0125 |
| 10 | CV17 | LogisticRegression | 2 | 0.7747 | 0.0153 | 0.8579 | 0.0067 |  |  |
| 10 | CV17 | LogisticRegression | 3 | 0.7758 | 0.0151 | 0.8609 | 0.0071 | 0.0011 | 0.0071 |
| 10 | CV17 | LogisticRegression | 5 | 0.7758 | 0.0147 | 0.8610 | 0.0061 | 0.0011 | 0.0063 |
| 10 | CV17 | RandomForest | 2 | 0.7731 | 0.0166 | 0.8580 | 0.0087 |  |  |
| 10 | CV17 | RandomForest | 3 | 0.7718 | 0.0187 | 0.8580 | 0.0098 | -0.0013 | 0.0062 |
| 10 | CV17 | RandomForest | 5 | 0.7714 | 0.0202 | 0.8561 | 0.0095 | -0.0017 | 0.0068 |
| 10 | CV17 | paper_ensemble | 2 | 0.7803 | 0.0176 | 0.8639 | 0.0064 |  |  |
| 10 | CV17 | paper_ensemble | 3 | 0.7769 | 0.0174 | 0.8649 | 0.0067 | -0.0034 | 0.0088 |
| 10 | CV17 | paper_ensemble | 5 | 0.7789 | 0.0168 | 0.8643 | 0.0064 | -0.0014 | 0.0090 |
| 10 | CV17_THY_CONT | AdaBoost | 2 | 0.7733 | 0.0123 | 0.8570 | 0.0079 |  |  |
| 10 | CV17_THY_CONT | AdaBoost | 3 | 0.7726 | 0.0146 | 0.8553 | 0.0076 | -0.0007 | 0.0055 |
| 10 | CV17_THY_CONT | AdaBoost | 5 | 0.7696 | 0.0140 | 0.8526 | 0.0076 | -0.0037 | 0.0067 |
| 10 | CV17_THY_CONT | LogisticRegression | 2 | 0.7769 | 0.0133 | 0.8620 | 0.0060 |  |  |
| 10 | CV17_THY_CONT | LogisticRegression | 3 | 0.7760 | 0.0155 | 0.8624 | 0.0060 | -0.0009 | 0.0049 |
| 10 | CV17_THY_CONT | LogisticRegression | 5 | 0.7750 | 0.0137 | 0.8624 | 0.0054 | -0.0019 | 0.0074 |
| 10 | CV17_THY_CONT | RandomForest | 2 | 0.7764 | 0.0126 | 0.8573 | 0.0085 |  |  |
| 10 | CV17_THY_CONT | RandomForest | 3 | 0.7745 | 0.0145 | 0.8526 | 0.0093 | -0.0019 | 0.0087 |
| 10 | CV17_THY_CONT | RandomForest | 5 | 0.7798 | 0.0146 | 0.8578 | 0.0087 | 0.0034 | 0.0053 |
| 10 | CV17_THY_CONT | paper_ensemble | 2 | 0.7834 | 0.0136 | 0.8652 | 0.0056 |  |  |
| 10 | CV17_THY_CONT | paper_ensemble | 3 | 0.7810 | 0.0125 | 0.8633 | 0.0055 | -0.0024 | 0.0061 |
| 10 | CV17_THY_CONT | paper_ensemble | 5 | 0.7821 | 0.0134 | 0.8650 | 0.0055 | -0.0014 | 0.0056 |

## D3. Frozen-test result per inner-fold count (a second reading of the test partition)

| k | horizon | comparison | metric | delta | ci_lo | ci_hi | p_bootstrap | excludes_zero | threshold_base | threshold_other |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 7 | locked pipeline | f1_macro | -0.0141 | -0.0308 | 0.0012 | 0.0760 | False | 0.4400 | 0.4400 |
| 2 | 7 | locked pipeline | roc_auc | 0.0026 | -0.0026 | 0.0081 | 0.3740 | False | 0.4400 | 0.4400 |
| 2 | 7 | locked pipeline | auprc | 0.0041 | -0.0096 | 0.0173 | 0.5750 | False | 0.4400 | 0.4400 |
| 2 | 7 | locked pipeline | brier | -0.0003 | -0.0022 | 0.0015 | 0.7600 | False | 0.4400 | 0.4400 |
| 2 | 7 | independently optimized | f1_macro | 0.0212 | 0.0005 | 0.0431 | 0.0450 | True | 0.4400 | 0.3900 |
| 2 | 7 | independently optimized | roc_auc | 0.0034 | -0.0029 | 0.0102 | 0.3190 | False | 0.4400 | 0.3900 |
| 2 | 7 | independently optimized | auprc | 0.0069 | -0.0120 | 0.0262 | 0.4940 | False | 0.4400 | 0.3900 |
| 2 | 7 | independently optimized | brier | -0.0009 | -0.0030 | 0.0014 | 0.4860 | False | 0.4400 | 0.3900 |
| 2 | 10 | locked pipeline | f1_macro | -0.0057 | -0.0204 | 0.0082 | 0.4240 | False | 0.3800 | 0.3800 |
| 2 | 10 | locked pipeline | roc_auc | 0.0010 | -0.0046 | 0.0061 | 0.6970 | False | 0.3800 | 0.3800 |
| 2 | 10 | locked pipeline | auprc | 0.0048 | -0.0033 | 0.0132 | 0.2440 | False | 0.3800 | 0.3800 |
| 2 | 10 | locked pipeline | brier | -0.0010 | -0.0040 | 0.0020 | 0.4950 | False | 0.3800 | 0.3800 |
| 2 | 10 | independently optimized | f1_macro | 0.0028 | -0.0125 | 0.0180 | 0.7330 | False | 0.3800 | 0.3800 |
| 2 | 10 | independently optimized | roc_auc | 0.0034 | -0.0017 | 0.0082 | 0.1780 | False | 0.3800 | 0.3800 |
| 2 | 10 | independently optimized | auprc | 0.0088 | 0.0009 | 0.0175 | 0.0280 | True | 0.3800 | 0.3800 |
| 2 | 10 | independently optimized | brier | -0.0019 | -0.0046 | 0.0010 | 0.1790 | False | 0.3800 | 0.3800 |
| 3 | 7 | locked pipeline | f1_macro | -0.0134 | -0.0360 | 0.0069 | 0.2170 | False | 0.4200 | 0.4200 |
| 3 | 7 | locked pipeline | roc_auc | 0.0016 | -0.0048 | 0.0086 | 0.6380 | False | 0.4200 | 0.4200 |
| 3 | 7 | locked pipeline | auprc | -0.0007 | -0.0188 | 0.0166 | 0.9490 | False | 0.4200 | 0.4200 |
| 3 | 7 | locked pipeline | brier | 0.0004 | -0.0020 | 0.0027 | 0.7660 | False | 0.4200 | 0.4200 |
| 3 | 7 | independently optimized | f1_macro | 0.0094 | -0.0128 | 0.0319 | 0.4340 | False | 0.4200 | 0.4100 |
| 3 | 7 | independently optimized | roc_auc | 0.0088 | 0.0006 | 0.0172 | 0.0290 | True | 0.4200 | 0.4100 |
| 3 | 7 | independently optimized | auprc | 0.0256 | 0.0055 | 0.0446 | 0.0080 | True | 0.4200 | 0.4100 |
| 3 | 7 | independently optimized | brier | -0.0032 | -0.0058 | -0.0006 | 0.0150 | True | 0.4200 | 0.4100 |
| 3 | 10 | locked pipeline | f1_macro | -0.0067 | -0.0231 | 0.0080 | 0.3760 | False | 0.4500 | 0.4500 |
| 3 | 10 | locked pipeline | roc_auc | -0.0017 | -0.0079 | 0.0043 | 0.5870 | False | 0.4500 | 0.4500 |
| 3 | 10 | locked pipeline | auprc | 0.0005 | -0.0080 | 0.0087 | 0.9240 | False | 0.4500 | 0.4500 |
| 3 | 10 | locked pipeline | brier | 0.0003 | -0.0029 | 0.0036 | 0.8460 | False | 0.4500 | 0.4500 |
| 3 | 10 | independently optimized | f1_macro | -0.0123 | -0.0345 | 0.0095 | 0.2930 | False | 0.4500 | 0.3700 |
| 3 | 10 | independently optimized | roc_auc | -0.0016 | -0.0079 | 0.0046 | 0.6090 | False | 0.4500 | 0.3700 |
| 3 | 10 | independently optimized | auprc | -0.0020 | -0.0120 | 0.0075 | 0.6600 | False | 0.4500 | 0.3700 |
| 3 | 10 | independently optimized | brier | 0.0003 | -0.0034 | 0.0038 | 0.8760 | False | 0.4500 | 0.3700 |
| 5 | 7 | locked pipeline | f1_macro | -0.0140 | -0.0335 | 0.0028 | 0.1240 | False | 0.4400 | 0.4400 |
| 5 | 7 | locked pipeline | roc_auc | 0.0055 | -0.0008 | 0.0119 | 0.0860 | False | 0.4400 | 0.4400 |
| 5 | 7 | locked pipeline | auprc | 0.0118 | -0.0045 | 0.0279 | 0.1680 | False | 0.4400 | 0.4400 |
| 5 | 7 | locked pipeline | brier | -0.0013 | -0.0034 | 0.0008 | 0.2260 | False | 0.4400 | 0.4400 |
| 5 | 7 | independently optimized | f1_macro | -0.0079 | -0.0305 | 0.0136 | 0.4790 | False | 0.4400 | 0.4000 |
| 5 | 7 | independently optimized | roc_auc | 0.0014 | -0.0053 | 0.0087 | 0.7200 | False | 0.4400 | 0.4000 |
| 5 | 7 | independently optimized | auprc | 0.0100 | -0.0112 | 0.0300 | 0.3910 | False | 0.4400 | 0.4000 |
| 5 | 7 | independently optimized | brier | -0.0002 | -0.0026 | 0.0023 | 0.9280 | False | 0.4400 | 0.4000 |
| 5 | 10 | locked pipeline | f1_macro | -0.0018 | -0.0168 | 0.0129 | 0.8180 | False | 0.3800 | 0.3800 |
| 5 | 10 | locked pipeline | roc_auc | 0.0022 | -0.0041 | 0.0079 | 0.4890 | False | 0.3800 | 0.3800 |
| 5 | 10 | locked pipeline | auprc | 0.0068 | -0.0020 | 0.0160 | 0.1560 | False | 0.3800 | 0.3800 |
| 5 | 10 | locked pipeline | brier | -0.0018 | -0.0052 | 0.0018 | 0.3010 | False | 0.3800 | 0.3800 |
| 5 | 10 | independently optimized | f1_macro | 0.0075 | -0.0066 | 0.0219 | 0.3370 | False | 0.3800 | 0.3900 |
| 5 | 10 | independently optimized | roc_auc | 0.0029 | -0.0026 | 0.0084 | 0.3140 | False | 0.3800 | 0.3900 |
| 5 | 10 | independently optimized | auprc | 0.0066 | -0.0031 | 0.0163 | 0.1930 | False | 0.3800 | 0.3900 |
| 5 | 10 | independently optimized | brier | -0.0028 | -0.0060 | 0.0006 | 0.1090 | False | 0.3800 | 0.3900 |

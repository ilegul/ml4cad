# Post hoc predictive sensitivity

Supplementary analysis; the prespecified primary results are unchanged.
Selection uses raw validation macro-F1 at 0.5, AUROC, then name.
Sigmoid calibration is fitted on training OOF probabilities; thresholds
are selected on validation or inherited from the baseline in locked arms.
All configurations and thresholds are fixed before test evaluation.
Top-three ensemble membership remains derived from CV17 training ranks.
Bootstrap intervals condition on the selected configurations and the stored
split; they do not account for selection uncertainty or repeated test inspection.
Raw fixed-threshold results and calibrated results are separate evaluations.

## Horizon 7

- CV17: RandomForest (validation macro-F1 0.7189).
- CV17_THY_CONT: RandomForest (validation macro-F1 0.7313).
- CV17_THY_STATES: top3_train_cv (validation macro-F1 0.7338).
- CV17_THY_CONT_STATES: paper (validation macro-F1 0.7250).
- CV17_THY_CONT_RATIO: top3_train_cv (validation macro-F1 0.7539).

- CV17_THY_CONT, independent ensemble selection: macro-F1 difference +0.0212, 95% CI [+0.0005, +0.0431].
- CV17_THY_CONT, independent configuration selection: macro-F1 difference -0.0118, 95% CI [-0.0389, +0.0154].
- CV17_THY_CONT, same baseline configuration: macro-F1 difference -0.0118, 95% CI [-0.0389, +0.0154].
- CV17_THY_CONT, locked baseline configuration: macro-F1 difference -0.0111, 95% CI [-0.0355, +0.0120].
- CV17_THY_STATES, independent ensemble selection: macro-F1 difference +0.0002, 95% CI [-0.0245, +0.0262].
- CV17_THY_STATES, independent configuration selection: macro-F1 difference -0.0184, 95% CI [-0.0445, +0.0075].
- CV17_THY_STATES, same baseline configuration: macro-F1 difference -0.0091, 95% CI [-0.0304, +0.0109].
- CV17_THY_STATES, locked baseline configuration: macro-F1 difference -0.0125, 95% CI [-0.0285, +0.0024].
- CV17_THY_STATES, selected ensemble versus paper: macro-F1 difference -0.0026, 95% CI [-0.0239, +0.0198].
- CV17_THY_CONT_STATES, independent ensemble selection: macro-F1 difference +0.0056, 95% CI [-0.0125, +0.0249].
- CV17_THY_CONT_STATES, independent configuration selection: macro-F1 difference -0.0130, 95% CI [-0.0382, +0.0114].
- CV17_THY_CONT_STATES, same baseline configuration: macro-F1 difference -0.0365, 95% CI [-0.0699, -0.0054].
- CV17_THY_CONT_STATES, locked baseline configuration: macro-F1 difference -0.0111, 95% CI [-0.0337, +0.0107].
- CV17_THY_CONT_RATIO, independent ensemble selection: macro-F1 difference +0.0083, 95% CI [-0.0144, +0.0328].
- CV17_THY_CONT_RATIO, independent configuration selection: macro-F1 difference -0.0104, 95% CI [-0.0409, +0.0168].
- CV17_THY_CONT_RATIO, same baseline configuration: macro-F1 difference -0.0091, 95% CI [-0.0332, +0.0145].
- CV17_THY_CONT_RATIO, locked baseline configuration: macro-F1 difference -0.0273, 95% CI [-0.0539, -0.0029].
- CV17_THY_CONT_RATIO, selected ensemble versus paper: macro-F1 difference -0.0089, 95% CI [-0.0322, +0.0131].

## Horizon 10

- CV17: XGBoost (validation macro-F1 0.7484).
- CV17_THY_CONT: paper (validation macro-F1 0.7430).
- CV17_THY_STATES: LogisticRegression (validation macro-F1 0.7445).
- CV17_THY_CONT_STATES: top3_train_cv (validation macro-F1 0.7460).
- CV17_THY_CONT_RATIO: top3_train_cv (validation macro-F1 0.7527).

- CV17_THY_CONT, independent ensemble selection: macro-F1 difference -0.0076, 95% CI [-0.0284, +0.0150].
- CV17_THY_CONT, independent configuration selection: macro-F1 difference -0.0041, 95% CI [-0.0274, +0.0190].
- CV17_THY_CONT, same baseline configuration: macro-F1 difference -0.0028, 95% CI [-0.0228, +0.0182].
- CV17_THY_CONT, locked baseline configuration: macro-F1 difference +0.0054, 95% CI [+0.0000, +0.0125].
- CV17_THY_STATES, independent ensemble selection: macro-F1 difference -0.0004, 95% CI [-0.0176, +0.0167].
- CV17_THY_STATES, independent configuration selection: macro-F1 difference -0.0069, 95% CI [-0.0216, +0.0076].
- CV17_THY_STATES, same baseline configuration: macro-F1 difference -0.0082, 95% CI [-0.0324, +0.0154].
- CV17_THY_STATES, locked baseline configuration: macro-F1 difference +0.0040, 95% CI [-0.0053, +0.0136].
- CV17_THY_STATES, selected ensemble versus paper: macro-F1 difference +0.0108, 95% CI [-0.0103, +0.0315].
- CV17_THY_CONT_STATES, independent ensemble selection: macro-F1 difference +0.0026, 95% CI [-0.0152, +0.0197].
- CV17_THY_CONT_STATES, independent configuration selection: macro-F1 difference +0.0061, 95% CI [-0.0130, +0.0246].
- CV17_THY_CONT_STATES, same baseline configuration: macro-F1 difference +0.0003, 95% CI [-0.0217, +0.0223].
- CV17_THY_CONT_STATES, locked baseline configuration: macro-F1 difference +0.0058, 95% CI [-0.0018, +0.0152].
- CV17_THY_CONT_STATES, selected ensemble versus paper: macro-F1 difference +0.0030, 95% CI [-0.0162, +0.0233].
- CV17_THY_CONT_RATIO, independent ensemble selection: macro-F1 difference +0.0008, 95% CI [-0.0139, +0.0153].
- CV17_THY_CONT_RATIO, independent configuration selection: macro-F1 difference +0.0043, 95% CI [-0.0120, +0.0200].
- CV17_THY_CONT_RATIO, same baseline configuration: macro-F1 difference +0.0063, 95% CI [-0.0018, +0.0151].
- CV17_THY_CONT_RATIO, locked baseline configuration: macro-F1 difference +0.0054, 95% CI [+0.0000, +0.0125].
- CV17_THY_CONT_RATIO, selected ensemble versus paper: macro-F1 difference +0.0102, 95% CI [-0.0090, +0.0284].

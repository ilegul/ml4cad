"""Patient-bootstrap intervals for thesis Tables 5.2 to 5.5.

All selection decisions are read from frozen artifacts. Development OOF
predictions reproduce notebook 2 before tuning. Bootstrap samples keep every
probability and classification threshold fixed. Original results are read only.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits

import config
import predictive as pred
import train
import utils

OUTPUT = config.RESULTS_DIR / "table_confidence_intervals"
N_BOOT = 2000
SEED = 42
TOLERANCE = 1e-10
VERSION = "v1"
LABELS = {"f1_macro": "macro-F1", "roc_auc": "AUROC", "auprc": "AP",
          "brier": "Brier score", "observed_to_expected": "O:E",
          "calibration_slope": "calibration slope", "ece": "ECE"}
MODEL_LABELS = {"LogisticRegression": "Logistic Regression", "SVC": "SVC",
                "KNeighbors": "k-Nearest Neighbours", "RandomForest": "Random Forest",
                "AdaBoost": "AdaBoost", "MLP": "Multilayer Perceptron",
                "GradientBoosting": "Gradient Boosting", "XGBoost": "XGBoost",
                "paper": "Reference soft voting"}
VALIDATION_ROWS = ((7, "CV17", "RandomForest", .7189),
                   (7, "CV17_THY_CONT", "RandomForest", .7313),
                   (10, "CV17", "XGBoost", .7484),
                   (10, "CV17_THY_CONT", "paper", .7430))
# These labels preserve the attached thesis PDF. Full-precision CSV estimates
# remain authoritative; the display differences are exported separately.
THESIS_DISPLAY_OVERRIDES = {("5.5", 7, "CV17_THY_CONT", "AUROC"): "0.848",
                            ("5.5", 10, "CV17_THY_CONT", "AP"): "0.794",
                            ("5.5", 10, "CV17_THY_CONT_RATIO", "AUROC"): "0.857"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_snapshot():
    paths = [path for root in (config.DATA_PROC, config.MODELS_DIR, config.PRED_DIR,
                              config.CACHE_DIR, config.RESULTS_DIR, config.FIGURES_DIR)
             for path in root.rglob("*") if path.is_file() and not path.is_relative_to(OUTPUT)]
    paths += list(config.ROOT.glob("*.ipynb"))
    paths += [config.ROOT / name for name in ("config.py", "train.py", "utils.py", "README.md")]
    return {str(path.relative_to(config.ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in paths}


def read_prediction(path):
    frame = pd.read_csv(path)
    utils.check_unique_ids(frame)
    return frame


def verify_points(metadata, actual, expected, metrics, verification, tolerance=TOLERANCE):
    """Record every reproduced value; a mismatch blocks all CI exports."""
    errors = []
    for metric in metrics:
        difference = abs(float(actual[metric]) - float(expected[metric]))
        matches = bool(np.isfinite(difference) and difference <= tolerance)
        verification.append({**metadata, "metric": metric, "expected": expected[metric],
                             "reproduced": actual[metric], "absolute_error": difference,
                             "tolerance": tolerance, "matches": matches})
        if not matches:
            errors.append(f"{metric}: expected {expected[metric]}, reproduced {actual[metric]}")
    if errors:
        pd.DataFrame(verification).to_csv(OUTPUT / "point_estimate_verification.csv", index=False)
        raise ValueError(f"Point estimate mismatch for {metadata}: " + "; ".join(errors))


def check_alignment(frame, data, split):
    _, outcomes, ids = pred.partition(data, split)
    utils.assert_same_patients(ids, frame.patient_id.to_numpy())
    if not np.array_equal(outcomes, frame.y_true.to_numpy()):
        raise ValueError("Stored patient outcomes do not match the requested development partition")


def validation_predictions(verification):
    scores = pd.read_csv(config.RESULTS_DIR / "predictive_sensitivity/validation_scores.csv")
    arms = []
    for horizon, feature_set, model, rounded in VALIDATION_ROWS:
        data = pred.load_partition(horizon, feature_set)
        Xv, yv, ids = pred.partition(data, "valid")
        if model == "paper":
            source = config.MODELS_DIR / f"final_h{horizon}_{feature_set}_paper.joblib"
            fitted = joblib.load(source)
            members = fitted["members_order"]
            if members != config.PAPER_ENSEMBLE:
                raise ValueError("The stored voting composition is not the reference ensemble")
            probability = np.column_stack([fitted["members"][m].predict_proba(Xv)[:, 1]
                                           for m in members]).mean(axis=1)
            saved = read_prediction(config.PRED_DIR / f"h{horizon}_{feature_set}_paper_valid.csv")
            check_alignment(saved, data, "valid")
            np.testing.assert_allclose(probability, saved.proba_raw, rtol=0, atol=TOLERANCE)
        else:
            row = scores[scores.horizon.eq(horizon) & scores.feature_set.eq(feature_set)
                         & scores.configuration.eq(model)].iloc[0]
            spec = pred.member_spec(horizon, feature_set, model, row.samplers)
            signature = pred.member_signature(data, spec)
            source = config.CACHE_DIR / "predictive_sensitivity" / f"member_{signature}.joblib"
            fitted = pred.cached(source, signature)
            if fitted is None:
                raise ValueError(f"Required frozen selected classifier cache missing: {source}")
            if fitted["spec"] != spec:
                raise ValueError("The stored classifier does not match the frozen sampler/tuning specification")
            utils.assert_same_patients(ids, fitted["valid_ids"])
            np.testing.assert_array_equal(yv, fitted["valid_y"])
            probability = fitted["estimator"].predict_proba(Xv)[:, 1]
            np.testing.assert_allclose(probability, fitted["valid_raw"], rtol=0, atol=TOLERANCE)
        frame = pd.DataFrame({"patient_id": ids, "y_true": yv, "probability": probability})
        point = utils.classification_metrics(yv, probability, .5)
        metadata = {"table": "5.3", "horizon": horizon, "feature_set": feature_set, "model": model}
        verify_points(metadata, point, {"f1_macro": rounded}, ["f1_macro"], verification, .00005)
        selected = scores[scores.horizon.eq(horizon) & scores.feature_set.eq(feature_set)
                          & scores.configuration.eq(model)].iloc[0]
        verify_points(metadata, point, selected, ["f1_macro"], verification)
        path = OUTPUT / "predictions" / f"table_5_3_h{horizon}_{feature_set}_{model}_validation.csv"
        frame.to_csv(path, index=False)
        arms.append({**metadata, "frame": frame, "threshold": .5, "point": point,
                     "source": str(source.relative_to(config.ROOT)).replace("\\", "/"),
                     "persisted_predictions": str(path.relative_to(config.ROOT)).replace("\\", "/")})
    return arms


def frozen_test_predictions(verification):
    classification = pd.read_csv(config.RESULTS_DIR / "final_test_classification.csv")
    calibration = pd.read_csv(config.RESULTS_DIR / "final_test_calibration.csv")
    performance_arms, calibration_arms = [], []
    for horizon in (7, 10):
        for feature_set in config.FEATURE_SETS:
            source = config.PRED_DIR / f"h{horizon}_{feature_set}_paper_test.csv"
            frame = read_prediction(source)
            row = classification[classification.horizon.eq(horizon) & classification.feature_set.eq(feature_set)
                                 & classification.ensemble.eq("paper") & classification.probabilities.eq("calibrated")]
            if len(row) != 1:
                raise ValueError("Expected one frozen reference ensemble test row")
            row = row.iloc[0]
            manifest_path = source.with_name(source.stem + "_manifest.json")
            manifest = pred.read_json(manifest_path)
            if abs(float(manifest["threshold"]) - row.threshold) > TOLERANCE:
                raise ValueError("Frozen threshold differs between the test result and its manifest")
            point = utils.classification_metrics(frame.y_true, frame.proba_calibrated, row.threshold)
            metadata = {"table": "5.5", "horizon": horizon, "feature_set": feature_set, "model": "paper"}
            verify_points(metadata, point, row, ["n", "n_events", "threshold", "roc_auc", "auprc",
                                                "f1_macro", "brier"], verification)
            arm = {**metadata, "frame": frame.rename(columns={"proba_calibrated": "probability"}),
                   "threshold": row.threshold, "point": point,
                   "source": str(source.relative_to(config.ROOT)).replace("\\", "/")}
            performance_arms.append(arm)
            if feature_set in (config.PRIMARY_BASELINE, config.PRIMARY_THYROID):
                for scale, column, label in (("raw", "proba_raw", "uncalibrated"),
                                              ("sigmoid", "proba_calibrated", "calibrated")):
                    subset = calibration[calibration.horizon.eq(horizon) & calibration.feature_set.eq(feature_set)
                                         & calibration.ensemble.eq("paper") & calibration.probabilities.eq(label)
                                         & calibration.method.eq("sigmoid")]
                    if len(subset) != 1:
                        raise ValueError("Expected one reference probability-calibration result")
                    actual = train.calibration_report(frame.y_true, frame[column])
                    metadata = {"table": "5.4", "horizon": horizon, "feature_set": feature_set,
                                "model": "paper", "probability_scale": scale}
                    verify_points(metadata, actual, subset.iloc[0], ["observed_to_expected", "calibration_slope",
                                                                    "ece", "brier"], verification)
                    calibration_arms.append({**metadata, "frame": frame[["patient_id", "y_true", column]].rename(
                        columns={column: "probability"}), "point": actual, "threshold": None,
                        "source": str(source.relative_to(config.ROOT)).replace("\\", "/")})
    return performance_arms, calibration_arms


def development_oof(verification):
    source = config.RESULTS_DIR / "model_development_training_cv.csv"
    reference = pd.read_csv(source)
    arms = []
    for horizon in (7, 10):
        data = pred.load_partition(horizon, config.PRIMARY_BASELINE)
        Xt, yt, ids = pred.partition(data, "train")
        folds = StratifiedKFold(n_splits=config.OUTER_CV, shuffle=True, random_state=config.SEED)
        assignments = np.empty(len(yt), dtype=int)
        for fold, (_, held) in enumerate(folds.split(Xt, yt)):
            assignments[held] = fold
        pd.DataFrame({"patient_id": ids, "fold": assignments}).to_csv(
            OUTPUT / "predictions" / f"table_5_2_h{horizon}_CV17_folds.csv", index=False)
        for model in config.MODELS:
            pipe = train.make_pipeline(model, config.DEV_SAMPLER, config.PRIMARY_BASELINE)
            signature = utils.cache_signature(stage="table_ci_development_oof", version=VERSION,
                data=data["signature"], estimator=joblib.hash(pipe), outer_cv=config.OUTER_CV,
                seed=config.SEED, notebook=sha256(config.ROOT / "2_classifiers.ipynb"))
            path = OUTPUT / "predictions" / f"table_5_2_h{horizon}_CV17_{model}_oof.csv"
            manifest_path = path.with_suffix(".json")
            if path.exists() and manifest_path.exists() and pred.read_json(manifest_path)["signature"] == signature:
                frame = read_prediction(path)
                check_alignment(frame, data, "train")
            else:
                pred.log(f"Regenerating notebook-2 OOF: h{horizon} CV17 {model}")
                probability = cross_val_predict(pipe, Xt, yt, cv=folds, method="predict_proba",
                                                 n_jobs=config.N_JOBS)[:, 1]
                frame = pd.DataFrame({"patient_id": ids, "y_true": yt, "probability": probability,
                                      "fold": assignments})
                frame.to_csv(path, index=False)
                manifest_path.write_text(json.dumps({"signature": signature, "model": model,
                    "sampler": config.DEV_SAMPLER, "outer_cv": config.OUTER_CV, "seed": config.SEED,
                    "parameters": repr(pipe.get_params(deep=True))}, indent=2), encoding="utf-8")
            point = utils.classification_metrics(frame.y_true, frame.probability, .5)
            expected = reference[reference.horizon.eq(horizon) & reference.feature_set.eq("CV17")
                                 & reference.model.eq(model)].iloc[0]
            metadata = {"table": "5.2", "horizon": horizon, "feature_set": "CV17", "model": model}
            verify_points(metadata, point, expected, ["n", "n_events", "f1_macro", "roc_auc"], verification)
            pred.log(f"Verified notebook-2 OOF: h{horizon} {model}")
            arms.append({**metadata, "frame": frame, "threshold": .5, "point": point,
                         "source": str(path.relative_to(config.ROOT)).replace("\\", "/")})
    return arms


def classification_draw(y, p, threshold, indices, metrics):
    yb, pb = y[indices], p[indices]
    if len(np.unique(yb)) < 2:
        return {metric: np.nan for metric in metrics}
    return {metric: float(utils.METRIC_FUNCS[metric](yb, pb, threshold)) for metric in metrics}


def calibration_draw(y, p, indices):
    yb, pb = y[indices], p[indices]
    try:
        report = train.calibration_report(yb, pb)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        # A failed evaluation-slope fit must not discard the other finite metrics.
        points = utils.reliability_points(yb, pb, config.CALIBRATION_BINS)
        expected = pb.mean()
        report = {"observed_to_expected": yb.mean() / expected if expected > 0 else np.nan,
                  "calibration_slope": np.nan, "ece": float((points.n / points.n.sum() *
                     (points.observed_rate - points.mean_predicted).abs()).sum()),
                  "brier": np.mean((pb - yb) ** 2)}
    return {metric: float(report[metric]) for metric in ("observed_to_expected", "calibration_slope", "ece", "brier")}


def percentile_interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi), len(finite)


def bootstrap_arm(arm, metrics, calibration=False):
    frame = arm["frame"]
    y, p = frame.y_true.to_numpy(), frame.probability.to_numpy(dtype=float)
    draws = utils.bootstrap_indices(len(frame), N_BOOT, SEED)
    pred.log(f"Bootstrap table {arm['table']}: h{arm['horizon']} {arm['feature_set']} {arm['model']} "
             f"{arm.get('probability_scale', 'raw' if arm['table'] in ('5.2', '5.3') else 'sigmoid')}")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=config.N_JOBS)(
            delayed(calibration_draw)(y, p, idx) if calibration else
            delayed(classification_draw)(y, p, arm["threshold"], idx, metrics) for idx in draws)
    distributions = pd.DataFrame(results)
    scale = arm.get("probability_scale", "raw" if arm["table"] in ("5.2", "5.3") else "sigmoid")
    path = OUTPUT / "bootstrap" / f"table_{arm['table'].replace('.', '_')}_h{arm['horizon']}_{arm['feature_set']}_{arm['model']}_{scale}.npz"
    np.savez_compressed(path, **{metric: distributions[metric].to_numpy() for metric in metrics})
    rows = []
    for metric in metrics:
        lo, hi, used = percentile_interval(distributions[metric])
        rows.append({"table": arm["table"], "horizon": arm["horizon"], "feature_set": arm["feature_set"],
                     "model_or_configuration": MODEL_LABELS[arm["model"]], "probability_scale": scale,
                     "metric": LABELS[metric], "metric_code": metric, "estimate": arm["point"][metric],
                     "ci_lo": lo, "ci_hi": hi, "n_boot_used": used, "bootstrap_seed": SEED,
                     "n_boot_requested": N_BOOT, "threshold": arm["threshold"], "n": len(frame),
                     "n_events": int(y.sum()), "prediction_source": arm["source"]})
    return rows


def display_ci(rows, horizon, feature_set, model, metric, scale=None, digits=3, preserve_thesis_labels=False):
    selected = rows[rows.horizon.eq(horizon) & rows.feature_set.eq(feature_set)
                    & rows.model_or_configuration.eq(MODEL_LABELS[model]) & rows.metric.eq(metric)]
    if scale is not None:
        selected = selected[selected.probability_scale.eq(scale)]
    if len(selected) != 1:
        raise ValueError("The copy-paste table does not have a unique interval row")
    row = selected.iloc[0]
    label = f"{row.estimate:.{digits}f}"
    if preserve_thesis_labels:
        label = THESIS_DISPLAY_OVERRIDES.get((row.table, horizon, feature_set, metric), label)
    return f"{label} [{row.ci_lo:.{digits}f}, {row.ci_hi:.{digits}f}]"


def display_audit(combined):
    rows = []
    for (table, horizon, feature_set, metric), label in THESIS_DISPLAY_OVERRIDES.items():
        row = combined[combined.table.eq(table) & combined.horizon.eq(horizon)
                       & combined.feature_set.eq(feature_set) & combined.metric.eq(metric)].iloc[0]
        rows.append({"table": table, "horizon": horizon, "feature_set": feature_set,
                     "metric": metric, "source_csv_estimate": row.estimate,
                     "source_direct_three_decimal_label": f"{row.estimate:.3f}",
                     "thesis_pdf_label": label, "source_pdf": "Gulino_Thesis_a.pdf, printed page 67",
                     "action": "preserve the thesis label in the copy-paste view; keep the CSV estimate unchanged"})
    pd.DataFrame(rows).to_csv(OUTPUT / "thesis_display_differences.csv", index=False)


def copy_tables(tables, preserve_thesis_labels=False):
    text = "Values are point estimates [95% percentile patient-bootstrap CI].\n\n"
    for number, rows in tables.items():
        text += f"## Table {number}\n\n"
        if number == "5.2":
            text += "| Classifier | macro-F1, 7 years | AUROC, 7 years | macro-F1, 10 years | AUROC, 10 years |\n|:---|---:|---:|---:|---:|\n"
            order = ["LogisticRegression", "SVC", "KNeighbors", "RandomForest", "AdaBoost", "GradientBoosting", "XGBoost", "MLP"]
            for model in order:
                values = [display_ci(rows, h, "CV17", model, metric) for h in (7, 10) for metric in ("macro-F1", "AUROC")]
                text += "| " + " | ".join([MODEL_LABELS[model]] + values) + " |\n"
        elif number == "5.3":
            text += "| Years | Predictors | Selected configuration | macro-F1 |\n|---:|:---|:---|---:|\n"
            for h, fs, model, _ in VALIDATION_ROWS:
                text += f"| {h} | {fs} | {MODEL_LABELS[model]} | {display_ci(rows, h, fs, model, 'macro-F1', digits=4)} |\n"
        elif number == "5.4":
            text += "| Years | Predictors | Scale | O:E | Slope | ECE | Brier |\n|---:|:---|:---|---:|---:|---:|---:|\n"
            for h in (7, 10):
                for fs in (config.PRIMARY_BASELINE, config.PRIMARY_THYROID):
                    for scale in ("raw", "sigmoid"):
                        values = [display_ci(rows, h, fs, "paper", metric, scale) for metric in
                                  ("O:E", "calibration slope", "ECE", "Brier score")]
                        text += "| " + " | ".join([str(h), fs, scale.title()] + values) + " |\n"
        else:
            text += "| Years | Predictors | Threshold | AUROC | AP | macro-F1 | Brier |\n|---:|:---|---:|---:|---:|---:|---:|\n"
            for h in (7, 10):
                for fs in config.FEATURE_SETS:
                    threshold = rows[rows.horizon.eq(h) & rows.feature_set.eq(fs)].iloc[0].threshold
                    values = [display_ci(rows, h, fs, "paper", metric, preserve_thesis_labels=preserve_thesis_labels)
                              for metric in ("AUROC", "AP", "macro-F1", "Brier score")]
                    text += "| " + " | ".join([str(h), fs, f"{threshold:.2f}"] + values) + " |\n"
        text += "\n"
    return text


def run():
    if config.PROFILE != "full" or config.SEED != SEED or config.OUTER_CV != 5 or config.DEV_SAMPLER != "RandomOverSampler":
        raise ValueError("Use the unchanged current full-profile development protocol")
    before = original_snapshot()
    for path in (OUTPUT, OUTPUT / "predictions", OUTPUT / "bootstrap"):
        path.mkdir(parents=True, exist_ok=True)
    verification = []
    # All point estimates must reproduce before a single CI is exported.
    validation = validation_predictions(verification)
    performance, calibration = frozen_test_predictions(verification)
    development = development_oof(verification)
    pd.DataFrame(verification).to_csv(OUTPUT / "point_estimate_verification.csv", index=False)
    plans = {"5.2": (development, ["f1_macro", "roc_auc"], False),
             "5.3": (validation, ["f1_macro"], False),
             "5.4": (calibration, ["observed_to_expected", "calibration_slope", "ece", "brier"], True),
             "5.5": (performance, ["roc_auc", "auprc", "f1_macro", "brier"], False)}
    tables = {}
    with threadpool_limits(limits=1):
        for number, (arms, metrics, calibration_metrics) in plans.items():
            rows = [row for arm in arms for row in bootstrap_arm(arm, metrics, calibration_metrics)]
            tables[number] = pd.DataFrame(rows)
            tables[number]["threshold"] = pd.to_numeric(tables[number]["threshold"], errors="coerce")
    for number, rows in tables.items():
        rows.to_csv(OUTPUT / f"table_{number.replace('.', '_')}_ci.csv", index=False)
    combined = pd.concat(tables.values(), ignore_index=True)
    combined.to_csv(OUTPUT / "all_table_ci.csv", index=False)
    display_audit(combined)
    (OUTPUT / "copy_paste_tables.md").write_text(copy_tables(tables) +
        "Point estimates and intervals are rounded directly from the verified CSV values. "
        "Three thesis PDF display differences are documented in thesis_display_differences.csv.\n", encoding="utf-8")
    (OUTPUT / "copy_paste_tables_thesis_labels.md").write_text(copy_tables(tables, preserve_thesis_labels=True) +
        "The existing thesis point labels are retained in this alternative view. "
        "Numeric estimates and intervals remain those of the verified CSV files.\n", encoding="utf-8")
    readme = """# Confidence intervals for thesis Tables 5.2 to 5.5

Each interval is the 2.5th/97.5th percentile of 2000 patient bootstrap resamples
generated by utils.bootstrap_indices with seed 42. Fixed predictions are
resampled; no model, sampler, calibrator or threshold is selected in a resample.
These are marginal intervals, not new comparisons or significance tests.

- Table 5.2: eight CV17 default classifiers at both horizons. Notebook 2's
  five-fold StratifiedKFold (shuffle=True, seed 42), train.make_pipeline,
  RandomOverSampler and threshold 0.5 are reproduced exactly. Probabilities
  are regenerated with cross_val_predict on the frozen training partition.
  Metrics and intervals use pooled OOF patients, not fold-average metrics
  or the five fold scores. This conditions on the regenerated cross-fitted
  predictions and does not measure refitting or model-selection uncertainty.
- Table 5.3: the four thesis configurations are read from frozen selected-model
  caches and the final reference-voting bundle. No new tuning or selection
  occurs. Raw validation probabilities and threshold 0.5 are used. Persisted
  validation predictions are in predictions/table_5_3_*.csv. Intervals are
  descriptive, conditional on the selected configuration and validation sample;
  they do not correct for selection using those same validation patients.
- Table 5.4: predictions/h{horizon}_{feature_set}_paper_test.csv supplies raw
  and sigmoid probabilities. train.calibration_report is called within each
  resample, with the current ten-bin ECE definition. The logistic calibration
  slope is an evaluation statistic re-estimated from the stored probabilities,
  not a refitted probability calibrator. Non-finite slopes are omitted only
  for that metric. Other finite metrics are retained.
- Table 5.5: the existing *_paper_test.csv calibrated probabilities supply all
  five predictor sets and both horizons. Each validation-selected threshold
  is checked against the frozen test manifest and remains fixed. AP is average
  precision (the source column auprc), not trapezoidal PR area.

All point estimates are checked before any interval export against the
corresponding original result files. Table 5.3 is additionally checked against
the four thesis values at four-decimal rounding tolerance. The complete audit
is in point_estimate_verification.csv. n_boot_used records finite resamples
separately for each metric. Classification draws with a single outcome class
are excluded, consistently with the existing classification bootstrap.

The four table_5_*_ci.csv files and all_table_ci.csv use one metric per row,
with full precision numeric values and explicit prediction_source. The
copy_paste_tables.md file follows the thesis order and display precision,
rounding directly from the full-precision estimates. The alternative
copy_paste_tables_thesis_labels.md retains the original PDF point labels.
OOF probabilities, fold assignments and generation signatures are preserved
under predictions/. Bootstrap metric arrays are preserved under bootstrap/.
The original thesis, numerical results and predictive artifacts are unchanged.

The alternative thesis-label view retains the existing PDF point labels. Three Table 5.5
labels differ from direct three-decimal rounding of the full-precision CSV:
7-year CV17_THY_CONT AUROC (PDF 0.848 versus direct 0.847), 10-year
CV17_THY_CONT AP (0.794 versus 0.793) and 10-year CV17_THY_CONT_RATIO
AUROC (0.857 versus 0.856). These are display differences, not failed
reproduction of the underlying result files. They are documented in
thesis_display_differences.csv. No original number was replaced. Every CI
uses the full-precision verified source estimate and stored probabilities.

Run from the repository root: python 13_table_confidence_intervals.py.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    after = original_snapshot()
    changed = [path for path, metadata in before.items() if after.get(path) != metadata]
    (OUTPUT / "original_artifact_integrity.json").write_text(json.dumps(
        {"checked_files": len(before), "changed": changed,
         "check": "file sizes and nanosecond modification times"}, indent=2), encoding="utf-8")
    if changed:
        raise RuntimeError(f"Original artifacts changed: {changed}")
    sources = [config.ROOT / "config.py", config.ROOT / "train.py", config.ROOT / "utils.py",
               config.ROOT / "predictive.py", config.ROOT / "table_confidence_intervals.py",
               config.ROOT / "13_table_confidence_intervals.py", config.ROOT / "2_classifiers.ipynb",
               config.RESULTS_DIR / "model_development_training_cv.csv",
               config.RESULTS_DIR / "final_test_classification.csv", config.RESULTS_DIR / "final_test_calibration.csv",
               config.RESULTS_DIR / "predictive_sensitivity/validation_scores.csv", config.SPLITS_FILE]
    sources += [Path(str(config.COHORT_STRICT).format(horizon=h)) for h in (7, 10)]
    sources += [config.ROOT / name for name in combined.prediction_source.unique()]
    manifest = {"version": VERSION, "bootstrap_seed": SEED, "n_boot_requested": N_BOOT,
                "all_original_csv_points_reproduced": True, "ci_metric_rows": len(combined),
                "thesis_pdf_display_differences": len(THESIS_DISPLAY_OVERRIDES),
                "source_checksums": {str(path.relative_to(config.ROOT)).replace("\\", "/"): sha256(path)
                                     for path in sorted(set(sources))}}
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pred.log(f"Complete: {len(combined)} confidence intervals; all points reproduced")
    return combined

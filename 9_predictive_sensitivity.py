"""Post hoc predictive sensitivities, supplementary to notebook 3.

Compare the eight tuned classifiers and the three existing ensembles on
validation, then calibrate and evaluate the selected configurations. All
choices are frozen before test evaluation; the original artifacts are read
only. The top-three ensemble retains its CV17-derived membership.

Usage:
    python 9_predictive_sensitivity.py
    python 9_predictive_sensitivity.py --smoke
    python 9_predictive_sensitivity.py --summary

The smoke run uses the primary contrast at seven years and 20 bootstrap
draws in separate directories. It retains the fitted model specifications.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone

import config
import ensemble as ens
import train
import utils

VERSION = "v1"
METRICS = ["f1_macro", "roc_auc", "auprc", "brier"]
BASE = config.PRIMARY_BASELINE


# ---------------------------------------------------------------------------
# Paths, signatures and partition integrity
# ---------------------------------------------------------------------------

def output_paths(smoke=False):
    name = "predictive_sensitivity_smoke" if smoke else "predictive_sensitivity"
    return {key: root / name for key, root in [
        ("results", config.RESULTS_DIR), ("models", config.MODELS_DIR),
        ("predictions", config.PRED_DIR), ("cache", config.CACHE_DIR),
        ("figures", config.FIGURES_DIR)]}


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached(path, signature, force=False):
    if force or not path.exists():
        return None
    try:
        payload = load(path)
    except (EOFError, ValueError, pickle.UnpicklingError):
        log(f"Incomplete cache; recomputing {path.name}")
        return None
    return payload if payload.get("signature") == signature else None


def store(payload, path, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dump({**payload, "signature": signature}, temporary)
    temporary.replace(path)


def load_partition(horizon, feature_set):
    frame = utils.read_frame(str(config.COHORT_STRICT).format(horizon=horizon))
    utils.check_unique_ids(frame)
    splits = utils.load_splits(horizon)
    utils.assert_disjoint(splits)
    if any(len(values) != len(set(values)) for values in splits.values()):
        raise ValueError("Stored partitions contain duplicate patient identifiers")
    if set(splits) != {"train", "valid", "test"}:
        raise ValueError("Expected the stored train, valid and test partitions")
    X, y, ids = utils.extract_xy(frame, feature_set)
    if set(ids) != set().union(*(set(values) for values in splits.values())):
        raise ValueError("Stored partitions do not cover exactly the cohort")
    masks = utils.split_masks(ids, splits)
    if not np.all(sum(masks.values()) == 1):
        raise ValueError("Each patient must belong to exactly one partition")
    development = masks["train"] | masks["valid"]
    columns = [config.ID_COL, config.TARGET] + utils.feature_list(feature_set)
    signature = utils.cache_signature(
        stage="predictive_development", version=VERSION, horizon=horizon,
        features=utils.feature_list(feature_set), splits=splits,
        fingerprint=utils.data_fingerprint(frame.loc[development], columns))
    return {"X": X, "y": y, "ids": ids, "masks": masks,
            "signature": signature}


def partition(data, split):
    mask = data["masks"][split]
    return data["X"].loc[mask], data["y"][mask], data["ids"][mask]


def rank_scores(scores, name="configuration"):
    """The notebook-3 selection rule, with a deterministic final tie break."""
    if scores.empty or not np.isfinite(scores[METRICS].to_numpy()).all():
        raise ValueError("Selection requires finite scores for every candidate")
    if not np.allclose(scores["threshold"], 0.5, rtol=0, atol=0):
        raise ValueError("Configuration selection uses the fixed raw threshold")
    if scores[["n", "n_events"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Candidates must cover the same validation cohort")
    return scores.sort_values(["f1_macro", "roc_auc", name],
                              ascending=[False, False, True])


def check_scores(actual, expected, context):
    """Fail on incompatible original inputs rather than silently reranking."""
    for metric in METRICS + ["n", "n_events", "threshold"]:
        if not np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-10):
            raise ValueError(f"Cannot reproduce {context}: {metric} "
                             f"{actual[metric]} != {expected[metric]}")


# ---------------------------------------------------------------------------
# Tuned single-model probabilities and validation comparison
# ---------------------------------------------------------------------------

def member_spec(horizon, feature_set, model, sampler, params_source=None):
    source = params_source or feature_set
    manifest = train.model_manifest(f"tuned_h{horizon}_{source}_{model}")
    if (manifest["horizon"] != horizon or manifest["feature_set"] != source
            or manifest["model"] != model or manifest["seed"] != config.SEED):
        raise ValueError("Tuning manifest does not match the requested model")
    return {"model": model, "sampler": sampler,
            "params": manifest["best_params"], "params_source": source,
            "tuning_signature": manifest["signature"]}


def fit_member(data, feature_set, spec, paths, force=False):
    signature = utils.cache_signature(stage="sensitivity_member", version=VERSION,
                                      development=data["signature"], spec=spec)
    path = paths["cache"] / f"member_{signature}.joblib"
    result = cached(path, signature, force)
    Xv, yv, iv = partition(data, "valid")
    if result is not None:
        utils.assert_same_patients(iv, result["valid_ids"])
        if not np.array_equal(yv, result["valid_y"]):
            raise ValueError("Cached validation outcomes do not match")
        return result
    Xt, yt, _ = partition(data, "train")
    pipe = train.make_pipeline(spec["model"], spec["sampler"], feature_set)
    pipe.set_params(**spec["params"])
    pipe.fit(Xt, yt)
    proba = pipe.predict_proba(Xv)[:, 1]
    result = {"estimator": pipe, "valid_raw": proba, "valid_ids": iv,
              "valid_y": yv, "spec": spec, "signature": signature,
              "metrics": utils.classification_metrics(yv, proba, 0.5)}
    store(result, path, signature)
    return result


def select_sampler(horizon, feature_set, model, data, sampling, paths, force=False):
    rows = sampling[(sampling["horizon"] == horizon)
                    & (sampling["feature_set"] == feature_set)
                    & (sampling["model"] == model)
                    & (sampling["status"] == "ok")
                    & (~sampling["replication_only"].fillna(False))
                    & (sampling["sampler"].isin(config.SAMPLERS))].copy()
    completed = []
    # Notebook 3 compares only models entering an ensemble. Complete missing
    # models with the same frozen tuning parameters and sampler policy.
    if rows.empty:
        for sampler in config.SAMPLERS:
            if sampler == "class_weight" and not train.supports_class_weight(model):
                continue
            spec = member_spec(horizon, feature_set, model, sampler)
            fitted = fit_member(data, feature_set, spec, paths, force)
            completed.append({"horizon": horizon, "feature_set": feature_set,
                              "model": model, "sampler": sampler, "status": "ok",
                              "replication_only": False, **fitted["metrics"]})
        rows = pd.DataFrame(completed)
    chosen = rank_scores(rows, "sampler").iloc[0]
    spec = member_spec(horizon, feature_set, model, str(chosen["sampler"]))
    fitted = fit_member(data, feature_set, spec, paths, force)
    check_scores(fitted["metrics"], chosen, f"h{horizon} {feature_set} {model}")
    return fitted, completed


def validation_comparison(horizons, feature_sets, paths, force=False):
    sampling = utils.read_frame(config.RESULTS_DIR / "sampling_comparison.csv")
    candidates = utils.read_frame(config.RESULTS_DIR / "ensemble_candidates.csv")
    reference = utils.read_frame(config.RESULTS_DIR / "ensemble_validation.csv")
    data, fitted, rows, completed = {}, {}, [], []
    for horizon in horizons:
        for feature_set in feature_sets:
            log(f"Validation comparison: h{horizon} {feature_set}")
            d = load_partition(horizon, feature_set)
            data[(horizon, feature_set)] = d
            for model in config.MODELS:
                result, missing = select_sampler(horizon, feature_set, model, d,
                                                 sampling, paths, force)
                fitted[(horizon, feature_set, model)] = result
                completed.extend(missing)
                rows.append({"horizon": horizon, "feature_set": feature_set,
                             "configuration": model, "kind": "single",
                             "members": model, "samplers": result["spec"]["sampler"],
                             **result["metrics"]})
            for _, candidate in candidates[candidates["horizon"] == horizon].iterrows():
                members = candidate["members"].split(",")
                parts = [fitted[(horizon, feature_set, m)] for m in members]
                utils.assert_same_patients(*(p["valid_ids"] for p in parts))
                proba = ens.mean_proba([p["valid_raw"] for p in parts])
                metrics = utils.classification_metrics(parts[0]["valid_y"], proba, 0.5)
                old = reference[(reference["horizon"] == horizon)
                                & (reference["feature_set"] == feature_set)
                                & (reference["candidate"] == candidate["candidate"])]
                if len(old) != 1 or old.iloc[0]["members"] != candidate["members"]:
                    raise ValueError("Expected exactly one matching ensemble reference")
                check_scores(metrics, old.iloc[0],
                             f"h{horizon} {feature_set} {candidate['candidate']}")
                rows.append({"horizon": horizon, "feature_set": feature_set,
                             "configuration": candidate["candidate"], "kind": "ensemble",
                             "members": candidate["members"],
                             "samplers": ",".join(p["spec"]["sampler"] for p in parts),
                             **metrics})
    scores = pd.DataFrame(rows)
    for _, subset in scores.groupby(["horizon", "feature_set"]):
        if len(subset) != len(config.MODELS) + 3:
            raise ValueError("Expected eight classifiers and three ensembles")
        rank_scores(subset)
    utils.write_frame(pd.DataFrame(completed), paths["results"] / "sampling_completed.csv")
    return scores, data, fitted


# ---------------------------------------------------------------------------
# Freeze comparison definitions without test scores
# ---------------------------------------------------------------------------

def comparison_plan(scores):
    arms, comparisons, selected = {}, [], []

    def add_arm(row, params_source=None, threshold_from=None):
        locked = params_source is not None
        name = f"h{row['horizon']}_{row['feature_set']}_{row['configuration']}"
        if locked:
            name += "_locked"
        arms[name] = {"name": name, "horizon": int(row["horizon"]),
                      "feature_set": row["feature_set"],
                      "configuration": row["configuration"], "kind": row["kind"],
                      "members": row["members"].split(","),
                      "params_source": params_source or row["feature_set"],
                      "threshold_from": threshold_from}
        return name

    def add_pair(horizon, feature_set, policy, base, other):
        comparisons.append({"horizon": int(horizon), "thyroid_set": feature_set,
                            "policy": policy, "base_arm": base, "other_arm": other})

    for horizon, group in scores.groupby("horizon", sort=True):
        baseline = group[group["feature_set"] == BASE]
        base_config = rank_scores(baseline).iloc[0]
        base_ensemble = rank_scores(baseline[baseline["kind"] == "ensemble"]).iloc[0]
        base_c = add_arm(base_config)
        base_e = add_arm(base_ensemble)
        for feature_set, subset in group.groupby("feature_set", sort=False):
            own_config = rank_scores(subset).iloc[0]
            own_ensemble = rank_scores(subset[subset["kind"] == "ensemble"]).iloc[0]
            selected.extend([
                {"horizon": int(horizon), "feature_set": feature_set,
                 "scope": scope, "configuration": row["configuration"],
                 "members": row["members"], "f1_macro": row["f1_macro"],
                 "roc_auc": row["roc_auc"]}
                for scope, row in [("all configurations", own_config),
                                   ("ensembles", own_ensemble)]])
            if feature_set == BASE:
                continue
            own_c, own_e = add_arm(own_config), add_arm(own_ensemble)
            fixed = subset[subset["configuration"] == base_config["configuration"]].iloc[0]
            same = add_arm(fixed)
            locked = add_arm(fixed, params_source=BASE, threshold_from=base_c)
            add_pair(horizon, feature_set, "independent ensemble selection", base_e, own_e)
            add_pair(horizon, feature_set, "independent configuration selection", base_c, own_c)
            add_pair(horizon, feature_set, "same baseline configuration", base_c, same)
            add_pair(horizon, feature_set, "locked baseline configuration", base_c, locked)
            if own_ensemble["configuration"] != "paper":
                paper = subset[subset["configuration"] == "paper"].iloc[0]
                add_pair(horizon, feature_set, "selected ensemble versus paper",
                         add_arm(paper), own_e)
    return arms, comparisons, pd.DataFrame(selected)


def member_oof(data, member, paths, jobs, force=False):
    signature = utils.cache_signature(stage="sensitivity_oof", version=VERSION,
                                      member=member["signature"], cv=config.CALIBRATION_CV)
    path = paths["cache"] / f"oof_{signature}.joblib"
    result = cached(path, signature, force)
    if result is None:
        Xt, yt, ids = partition(data, "train")
        oof = train.training_oof_proba(clone(member["estimator"]), Xt, yt, n_jobs=jobs)
        result = {"oof": oof, "train_ids": ids, "train_y": yt}
        store(result, path, signature)
    _, yt, ids = partition(data, "train")
    utils.assert_same_patients(ids, result["train_ids"])
    if not np.array_equal(yt, result["train_y"]):
        raise ValueError("Cached OOF outcomes do not match")
    return result["oof"]


def prepare_arm(spec, data, fitted, ready, paths, jobs, force=False):
    h, fs = spec["horizon"], spec["feature_set"]
    d = data[(h, fs)]
    parts = {}
    for model in spec["members"]:
        if spec["params_source"] == fs:
            parts[model] = fitted[(h, fs, model)]
        else:
            original = fitted[(h, spec["params_source"], model)]["spec"]
            member = member_spec(h, fs, model, original["sampler"], spec["params_source"])
            parts[model] = fit_member(d, fs, member, paths, force)
    threshold_override = (ready[spec["threshold_from"]]["threshold"]
                          if spec["threshold_from"] else None)
    signature = utils.cache_signature(
        stage="sensitivity_arm", version=VERSION, spec=spec,
        members={m: p["signature"] for m, p in parts.items()},
        calibration=config.CALIBRATION_PRIMARY, cv=config.CALIBRATION_CV,
        threshold_strategy=config.THRESHOLD_STRATEGY,
        threshold_grid=config.THRESHOLD_GRID.tolist(), threshold_override=threshold_override)
    path = paths["models"] / f"{spec['name']}.joblib"
    result = cached(path, signature, force)
    if result is None:
        log(f"OOF calibration: {spec['name']}")
        oof = ens.mean_proba([member_oof(d, p, paths, jobs, force) for p in parts.values()])
        valid_raw = ens.mean_proba([p["valid_raw"] for p in parts.values()])
        _, yt, _ = partition(d, "train")
        _, yv, iv = partition(d, "valid")
        calibrator = train.fit_calibrator(oof, yt, config.CALIBRATION_PRIMARY)
        valid_cal = train.apply_calibrator(calibrator, valid_raw)
        threshold = (utils.optimize_threshold(yv, valid_cal)
                     if threshold_override is None else threshold_override)
        result = {**spec, "members_fitted": {m: p["estimator"] for m, p in parts.items()},
                  "member_specs": {m: p["spec"] for m, p in parts.items()},
                  "calibrator": calibrator, "threshold": threshold,
                  "valid_raw": valid_raw, "valid_cal": valid_cal,
                  "valid_ids": iv, "valid_y": yv, "signature": signature}
        store(result, path, signature)
    _, yv, iv = partition(d, "valid")
    utils.assert_same_patients(iv, result["valid_ids"])
    if not np.array_equal(yv, result["valid_y"]):
        raise ValueError("Cached arm validation outcomes do not match")
    expected_threshold = (utils.optimize_threshold(yv, result["valid_cal"])
                          if threshold_override is None else threshold_override)
    if result["threshold"] != expected_threshold:
        raise ValueError("Cached threshold does not match its validation rule")
    manifest = path.with_name(f"{path.stem}_manifest.json")
    valid_path = paths["predictions"] / f"{spec['name']}_valid.csv"
    if force or not manifest.exists() or read_json(manifest).get("signature") != signature:
        utils.write_manifest(manifest, analysis="post hoc predictive sensitivity", **spec,
                             signature=signature, members_spec=result["member_specs"],
                             calibration=config.CALIBRATION_PRIMARY,
                             calibration_cv=config.CALIBRATION_CV,
                             threshold=result["threshold"],
                             threshold_source=spec["threshold_from"] or "own validation")
    if force or not valid_path.exists():
        utils.write_frame(pd.DataFrame({config.ID_COL: iv, "y_true": yv,
                                       "proba_raw": result["valid_raw"],
                                       "proba_calibrated": result["valid_cal"]}), valid_path)
    return result


# ---------------------------------------------------------------------------
# Test evaluation and paired inference, after every arm has been prepared
# ---------------------------------------------------------------------------

def evaluate_arm(arm, data, paths, force=False):
    X, y, ids = partition(data, "test")
    fingerprint = utils.data_fingerprint(pd.DataFrame({
        config.ID_COL: ids, "y_true": y, **{c: X[c].to_numpy() for c in X.columns}}))
    signature = utils.cache_signature(stage="sensitivity_test", version=VERSION,
                                      arm=arm["signature"], test=fingerprint)
    path = paths["predictions"] / f"{arm['name']}_test.csv"
    manifest = path.with_name(f"{path.stem}_manifest.json")
    if (not force and path.exists() and manifest.exists()
            and read_json(manifest)["signature"] == signature):
        frame = pd.read_csv(path)
    else:
        raw = ens.mean_proba([p.predict_proba(X)[:, 1] for p in arm["members_fitted"].values()])
        calibrated = train.apply_calibrator(arm["calibrator"], raw)
        frame = pd.DataFrame({config.ID_COL: ids, "y_true": y, "proba_raw": raw,
                              "proba_calibrated": calibrated,
                              "prediction": (calibrated >= arm["threshold"]).astype(int),
                              "p_survive": 1 - calibrated})
        utils.write_frame(frame, path)
        utils.write_manifest(manifest, signature=signature, arm=arm["name"],
                             threshold=arm["threshold"], n_test=len(frame),
                             n_events=int(y.sum()), selection_source="validation only")
    utils.check_unique_ids(frame)
    utils.assert_same_patients(ids, frame[config.ID_COL].to_numpy())
    if not np.array_equal(y, frame["y_true"]):
        raise ValueError("Test outcomes do not match cached predictions")
    p = frame[["proba_raw", "proba_calibrated"]].to_numpy()
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid cached test probabilities")
    return frame, signature


def check_original_test(arm, frame):
    """Reproduce existing ensemble artifacts without using them for selection."""
    if arm["kind"] != "ensemble" or arm["threshold_from"]:
        return None
    path = config.PRED_DIR / f"{arm['name']}_test.csv"
    if not path.exists():
        return None
    original = utils.read_frame(path)
    utils.assert_same_patients(frame[config.ID_COL].to_numpy(),
                               original[config.ID_COL].to_numpy())
    if not np.array_equal(frame["y_true"], original["y_true"]):
        raise ValueError("Original ensemble test outcomes do not match")
    differences = {}
    for column in ["proba_raw", "proba_calibrated"]:
        differences[f"{column}_max_difference"] = float(
            np.max(np.abs(frame[column].to_numpy() - original[column].to_numpy())))
    manifest = read_json(path.with_name(f"{path.stem}_manifest.json"))
    differences["threshold_difference"] = abs(arm["threshold"] - manifest["threshold"])
    if any(value > 1e-10 for value in differences.values()):
        raise ValueError(f"Cannot reproduce the original frozen ensemble: {arm['name']}")
    return {"arm": arm["name"], **differences}


def paired_comparison(pair, ready, evaluated, paths, n_boot, force=False):
    base, other = evaluated[pair["base_arm"]], evaluated[pair["other_arm"]]
    utils.assert_same_patients(base[0][config.ID_COL].to_numpy(), other[0][config.ID_COL].to_numpy())
    if not np.array_equal(base[0]["y_true"], other[0]["y_true"]):
        raise ValueError("Paired test outcomes differ")
    signature = utils.cache_signature(stage="sensitivity_delta", version=VERSION,
                                      base=base[1], other=other[1],
                                      n_boot=n_boot, metrics=METRICS)
    path = paths["cache"] / f"delta_{signature}.joblib"
    payload = cached(path, signature, force)
    if payload is not None:
        table = payload["table"].copy()
    else:
        tables = []
        for label, column in [("calibrated", "proba_calibrated"),
                              ("uncalibrated_fixed", "proba_raw")]:
            tb = ready[pair["base_arm"]]["threshold"] if label == "calibrated" else 0.5
            to = ready[pair["other_arm"]]["threshold"] if label == "calibrated" else 0.5
            delta = utils.paired_delta(base[0]["y_true"], base[0][column], other[0][column],
                                       tb, to, n_boot=n_boot)
            delta["probabilities"] = label
            delta["threshold_base"], delta["threshold_other"] = tb, to
            tables.append(delta)
        table = pd.concat(tables, ignore_index=True)
        store({"table": table}, path, signature)
    for key, value in reversed(list(pair.items())):
        table.insert(0, key, value)
    return table


# ---------------------------------------------------------------------------
# Supplementary summaries and figures
# ---------------------------------------------------------------------------

def write_summary(paths):
    manifest = read_json(paths["results"] / "run_manifest.json")
    if not manifest.get("complete"):
        raise ValueError("Complete the sensitivity run before rebuilding its summaries")
    metrics = utils.read_frame(paths["results"] / "test_metrics.csv")
    deltas = utils.read_frame(paths["results"] / "paired_deltas.csv")
    selected = utils.read_frame(paths["results"] / "selected_configurations.csv")
    if len(metrics) != manifest["n_arms"] * 2 or len(deltas) != manifest["n_comparisons"] * 8:
        raise ValueError("Supplementary output tables are incomplete")
    lines = ["# Post hoc predictive sensitivity", "",
             "Supplementary analysis; the prespecified primary results are unchanged.",
             "Selection uses raw validation macro-F1 at 0.5, AUROC, then name.",
             "Sigmoid calibration is fitted on training OOF probabilities; thresholds",
             "are selected on validation or inherited from the baseline in locked arms.",
             "All configurations and thresholds are fixed before test evaluation.",
             "Top-three ensemble membership remains derived from CV17 training ranks.",
             "Bootstrap intervals condition on the selected configurations and the stored",
             "split; they do not account for selection uncertainty or repeated test inspection.",
             "Raw fixed-threshold results and calibrated results are separate evaluations.", ""]
    for horizon in sorted(selected["horizon"].unique()):
        lines.append(f"## Horizon {horizon}")
        lines.append("")
        choices = selected[(selected["horizon"] == horizon)
                           & (selected["scope"] == "all configurations")]
        for _, row in choices.iterrows():
            lines.append(f"- {row['feature_set']}: {row['configuration']} "
                         f"(validation macro-F1 {row['f1_macro']:.4f}).")
        lines.append("")
        for _, row in deltas[(deltas["horizon"] == horizon)
                              & (deltas["probabilities"] == "calibrated")
                              & (deltas["metric"] == "f1_macro")].iterrows():
            lines.append(f"- {row['thyroid_set']}, {row['policy']}: macro-F1 difference "
                         f"{row['delta']:+.4f}, 95% CI [{row['ci_lo']:+.4f}, "
                         f"{row['ci_hi']:+.4f}].")
        lines.append("")
    (paths["results"] / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    # Compact tables for inspection without deriving a new winner from test.
    utils.write_frame(metrics[metrics["probabilities"] == "calibrated"],
                      paths["results"] / "calibrated_test_metrics.csv")
    utils.write_frame(deltas[deltas["probabilities"] == "calibrated"],
                      paths["results"] / "calibrated_paired_deltas.csv")


def write_figures(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = utils.read_frame(paths["results"] / "paired_deltas.csv")
    policies = ["independent configuration selection", "locked baseline configuration"]
    for horizon in sorted(deltas["horizon"].unique()):
        rows = deltas[(deltas["horizon"] == horizon)
                      & (deltas["probabilities"] == "calibrated")
                      & (deltas["policy"].isin(policies))]
        height = max(3, 1.5 + 0.55 * len(rows[rows["metric"] == "f1_macro"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, height), layout="constrained")
        for ax, metric in zip(axes, ["f1_macro", "roc_auc"]):
            subset = rows[rows["metric"] == metric].reset_index(drop=True)
            labels = [f"{r.thyroid_set}\n{r.policy}" for r in subset.itertuples()]
            positions = np.arange(len(subset))
            ax.hlines(positions, subset["ci_lo"], subset["ci_hi"], color="tab:blue")
            ax.plot(subset["delta"], positions, "o", color="tab:blue")
            ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
            ax.set_yticks(positions, labels, fontsize=8)
            ax.set_ylim(len(subset) - 0.5, -0.5)
            ax.set_xlabel(f"Thyroid minus CV17: {metric}")
            ax.set_title(f"{horizon} years; paired 95% intervals")
        fig.savefig(paths["figures"] / f"h{horizon}_configuration_deltas.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver: prepare every arm, then open the test evaluation stage
# ---------------------------------------------------------------------------

def original_artifacts():
    paths = [p for root in [config.RESULTS_DIR, config.MODELS_DIR,
                            config.PRED_DIR, config.DATA_PROC]
             for p in root.rglob("*") if p.is_file()
             and not any(part.startswith("predictive_sensitivity") for part in p.parts)]
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def run(args):
    paths = output_paths(args.smoke)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.summary:
        write_summary(paths)
        write_figures(paths)
        return
    original = original_artifacts()
    utils.write_manifest(paths["results"] / "run_manifest.json", complete=False,
                         analysis="post hoc predictive sensitivity")
    horizons = (7,) if args.smoke else config.HORIZONS
    feature_sets = ((BASE, config.PRIMARY_THYROID) if args.smoke
                    else config.TUNING_FEATURE_SETS)
    n_boot = 20 if args.smoke else config.BOOTSTRAP
    scores, data, fitted = validation_comparison(horizons, feature_sets, paths, args.force)
    specs, pairs, selected = comparison_plan(scores)
    utils.write_frame(scores.sort_values(["horizon", "feature_set", "f1_macro"],
                                        ascending=[True, True, False]),
                      paths["results"] / "validation_scores.csv")
    utils.write_frame(selected, paths["results"] / "selected_configurations.csv")
    ready = {}
    for name, spec in specs.items():
        ready[name] = prepare_arm(spec, data, fitted, ready, paths, args.jobs, args.force)
    selection = {name: {**spec, "signature": ready[name]["signature"],
                        "threshold": ready[name]["threshold"]} for name, spec in specs.items()}
    utils.write_manifest(paths["results"] / "frozen_selection.json",
                         analysis="post hoc predictive sensitivity", version=VERSION,
                         horizons=list(horizons), feature_sets=list(feature_sets),
                         selection_rule="raw validation f1_macro, roc_auc, name",
                         calibration=config.CALIBRATION_PRIMARY,
                         calibration_cv=config.CALIBRATION_CV,
                         threshold_strategy=config.THRESHOLD_STRATEGY,
                         threshold_grid=config.THRESHOLD_GRID.tolist(), n_boot=n_boot,
                         arms=selection, comparisons=pairs)
    log(f"All {len(ready)} configurations and thresholds frozen; evaluating test")
    evaluated, metric_rows, reproduction = {}, [], []
    for name, arm in ready.items():
        frame, signature = evaluate_arm(arm, data[(arm["horizon"], arm["feature_set"])],
                                        paths, args.force)
        evaluated[name] = (frame, signature)
        check = check_original_test(arm, frame)
        if check is not None:
            reproduction.append(check)
        for label, column in [("calibrated", "proba_calibrated"),
                              ("uncalibrated_fixed", "proba_raw")]:
            threshold = arm["threshold"] if label == "calibrated" else 0.5
            threshold_source = ((arm["threshold_from"] or "own validation")
                                if label == "calibrated" else "fixed 0.5")
            metric_rows.append({"horizon": arm["horizon"], "feature_set": arm["feature_set"],
                                "arm": name, "configuration": arm["configuration"],
                                "kind": arm["kind"], "members": ",".join(arm["members"]),
                                "params_source": arm["params_source"],
                                "threshold_source": threshold_source,
                                "probabilities": label,
                                **utils.classification_metrics(
                                    frame["y_true"], frame[column], threshold)})
    utils.write_frame(pd.DataFrame(metric_rows), paths["results"] / "test_metrics.csv")
    utils.write_frame(pd.DataFrame(reproduction), paths["results"] / "reproduction_check.csv")
    tables = []
    for i, pair in enumerate(pairs, 1):
        log(f"Paired comparison {i}/{len(pairs)}: h{pair['horizon']} "
            f"{pair['thyroid_set']}, {pair['policy']}")
        tables.append(paired_comparison(pair, ready, evaluated, paths, n_boot, args.force))
        utils.write_frame(pd.concat(tables, ignore_index=True), paths["results"] / "paired_deltas.csv")
    if original != original_artifacts():
        raise RuntimeError("An original artifact changed during the sensitivity run")
    utils.write_manifest(paths["results"] / "run_manifest.json", complete=True,
                         selection_manifest="frozen_selection.json", n_arms=len(ready),
                         n_comparisons=len(pairs), original_artifacts_unchanged=len(original))
    write_summary(paths)
    write_figures(paths)
    log("Predictive sensitivities complete; original artifacts unchanged")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Primary contrast, seven years, separate outputs")
    parser.add_argument("--summary", action="store_true",
                        help="Rebuild summaries and figures without fitting")
    parser.add_argument("--force", action="store_true", help="Recompute supplementary artifacts")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Workers for training OOF predictions (default: 1)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run(args)


if __name__ == "__main__":
    main()

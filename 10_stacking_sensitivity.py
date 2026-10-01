"""Supplementary stacking selection, calibration and frozen test sensitivities.

Run after 9_predictive_sensitivity.py. The original analyses are read only.
All subsets of two to eight classifiers are compared on validation, with a
regularized logistic meta-model fitted on training OOF probabilities. Nested
training OOF predictions calibrate the selected stacks before test evaluation.

Usage:
    python 10_stacking_sensitivity.py
    python 10_stacking_sensitivity.py --smoke
    python 10_stacking_sensitivity.py --summary
"""

import argparse

import numpy as np
import pandas as pd
from joblib import parallel_config
from threadpoolctl import threadpool_limits

import config
import predictive as pred
import stacking as stack
import train
import utils

BASE = config.PRIMARY_BASELINE
VERSION = stack.VERSION
POLICIES = ("own", "same_baseline", "locked_baseline", "paper")


# ---------------------------------------------------------------------------
# Original inputs and resumable development matrices
# ---------------------------------------------------------------------------

def output_paths(smoke=False):
    name = "stacking_sensitivity_smoke" if smoke else "stacking_sensitivity"
    return {key: root / name for key, root in [
        ("results", config.RESULTS_DIR), ("models", config.MODELS_DIR),
        ("predictions", config.PRED_DIR), ("cache", config.CACHE_DIR),
        ("figures", config.FIGURES_DIR)]}


def reference_scores():
    root = config.RESULTS_DIR / "predictive_sensitivity"
    manifest = pred.read_json(root / "run_manifest.json")
    if not manifest.get("complete"):
        raise ValueError("Complete 9_predictive_sensitivity.py first")
    return utils.read_frame(root / "validation_scores.csv")


def fitted_member(data, feature_set, definition, paths, jobs=1, force=False):
    signature = pred.member_signature(data, definition)
    source = config.CACHE_DIR / "predictive_sensitivity"
    result = pred.cached(source / f"member_{signature}.joblib", signature)
    if result is None:
        result = pred.fit_member(data, feature_set, definition, paths, force)
    _, y, ids = pred.partition(data, "valid")
    utils.assert_same_patients(ids, result["valid_ids"])
    if not np.array_equal(y, result["valid_y"]) or result["spec"] != definition:
        raise ValueError("Stored base model does not match its development specification")
    oof_signature = pred.oof_signature(result)
    old = pred.cached(source / f"oof_{oof_signature}.joblib", oof_signature)
    if old is None:
        oof = pred.member_oof(data, result, paths, jobs=jobs, force=force)
    else:
        _, yt, it = pred.partition(data, "train")
        utils.assert_same_patients(it, old["train_ids"])
        if not np.array_equal(yt, old["train_y"]):
            raise ValueError("Stored base OOF outcomes do not match")
        oof = old["oof"]
    return result, oof


def development(horizon, feature_set, scores, paths, jobs=1, force=False):
    data = pred.load_partition(horizon, feature_set)
    rows = scores[(scores.horizon == horizon) & (scores.feature_set == feature_set)
                  & (scores.kind == "single")]
    if set(rows.configuration) != set(config.MODELS) or len(rows) != len(config.MODELS):
        raise ValueError("Expected one reference score for each base classifier")
    definitions = {model: pred.member_spec(horizon, feature_set, model,
                   rows[rows.configuration == model].iloc[0].samplers) for model in config.MODELS}
    signature = utils.cache_signature(stage="stacking_development", version=VERSION,
                                      development=data["signature"], definitions=definitions)
    path = paths["cache"] / f"development_{signature}.joblib"
    result = pred.cached(path, signature, force)
    _, yt, it = pred.partition(data, "train")
    _, yv, iv = pred.partition(data, "valid")
    if result is None:
        members, oofs = {}, []
        for model in config.MODELS:
            pred.log(f"Training OOF: h{horizon} {feature_set} {model}")
            member, oof = fitted_member(data, feature_set, definitions[model], paths, jobs, force)
            pred.check_scores(member["metrics"], rows[rows.configuration == model].iloc[0],
                              f"h{horizon} {feature_set} {model}")
            members[model] = member
            oofs.append(oof)
        result = {"members": members, "train": np.column_stack(oofs),
                  "valid": np.column_stack([members[m]["valid_raw"] for m in config.MODELS]),
                  "train_y": yt, "train_ids": it, "valid_y": yv, "valid_ids": iv,
                  "signature": signature}
        pred.store(result, path, signature)
    utils.assert_same_patients(it, result["train_ids"])
    utils.assert_same_patients(iv, result["valid_ids"])
    if not np.array_equal(yt, result["train_y"]) or not np.array_equal(yv, result["valid_y"]):
        raise ValueError("Development outcomes do not match the cache")
    for key in ("train", "valid"):
        if not np.isfinite(result[key]).all():
            raise ValueError("Invalid development probability matrix")
    return data, result


def validation_search(horizon, feature_set, development, paths, force=False):
    signature = utils.cache_signature(stage="stacking_search", version=VERSION,
        development=development["signature"], models=config.MODELS,
        C=stack.REGULARIZATION, subset_sizes=[2, len(config.MODELS)], threshold=0.5)
    path = paths["cache"] / f"search_{signature}.joblib"
    result = pred.cached(path, signature, force)
    if result is None:
        rows = []
        for members in stack.candidate_subsets(config.MODELS):
            indices = [config.MODELS.index(m) for m in members]
            for C in stack.REGULARIZATION:
                meta = stack.meta_model(C).fit(development["train"][:, indices], development["train_y"])
                p = meta.predict_proba(development["valid"][:, indices])[:, 1]
                rows.append({"horizon": horizon, "feature_set": feature_set,
                             "configuration": stack.candidate_name(members, C),
                             "members": ",".join(members), "C": C,
                             "n": len(p), "n_events": int(development["valid_y"].sum()),
                             "prevalence": float(development["valid_y"].mean()), "threshold": 0.5,
                             **{metric: function(development["valid_y"], p, 0.5)
                                for metric, function in utils.METRIC_FUNCS.items()}})
        result = {"scores": pd.DataFrame(rows)}
        pred.store(result, path, signature)
    expected = len(stack.candidate_subsets(config.MODELS)) * len(stack.REGULARIZATION)
    if len(result["scores"]) != expected:
        raise ValueError("Incomplete stacking candidate search")
    return pred.rank_scores(result["scores"])


# ---------------------------------------------------------------------------
# Selected models, nested OOF calibration and validation thresholds
# ---------------------------------------------------------------------------

def arm_name(horizon, feature_set, policy):
    return f"h{horizon}_{feature_set}_{policy}"


def prepare_group(horizon, feature_set, data, development, own, baseline,
                  paper, ready, scores, paths, jobs=1, force=False):
    definitions = {"own": own, "paper": paper}
    if feature_set != BASE:
        definitions.update(same_baseline=baseline, locked_baseline=baseline)
    members = dict(development["members"])
    columns = {model: development["train"][:, i] for i, model in enumerate(config.MODELS)}
    valid = {model: development["valid"][:, i] for i, model in enumerate(config.MODELS)}
    nested_definitions = {}
    for policy, definition in definitions.items():
        chosen = definition["members"].split(",")
        aliases = chosen
        if policy == "locked_baseline":
            aliases = [model + "_locked" for model in chosen]
            for model, alias in zip(chosen, aliases):
                row = scores[(scores.horizon == horizon) & (scores.feature_set == BASE)
                             & (scores.configuration == model)].iloc[0]
                spec = pred.member_spec(horizon, feature_set, model, row.samplers, params_source=BASE)
                members[alias], columns[alias] = fitted_member(data, feature_set, spec, paths, jobs, force)
                valid[alias] = members[alias]["valid_raw"]
        nested_definitions[policy] = {"members": aliases, "C": float(definition["C"])}
    Xt, yt, it = pred.partition(data, "train")
    signature = utils.cache_signature(stage="stacking_nested", version=VERSION,
        development=development["signature"], definitions=nested_definitions,
        members={m: members[m]["signature"] for m in members}, cv=config.CALIBRATION_CV)
    path = paths["cache"] / f"stack_oof_{signature}.joblib"
    nested = pred.cached(path, signature, force)
    if nested is None:
        oof = stack.nested_stack_oof(Xt, yt, members, nested_definitions, paths,
                                     jobs=jobs, force=force)
        nested = {"oof": oof, "train_ids": it, "train_y": yt}
        pred.store(nested, path, signature)
    utils.assert_same_patients(it, nested["train_ids"])
    if not np.array_equal(yt, nested["train_y"]):
        raise ValueError("Nested OOF outcomes do not match")
    for policy, definition in definitions.items():
        name = arm_name(horizon, feature_set, policy)
        chosen = definition["members"].split(",")
        aliases = nested_definitions[policy]["members"]
        meta = stack.meta_model(float(definition["C"])).fit(
            np.column_stack([columns[m] for m in aliases]), yt)
        p = meta.predict_proba(np.column_stack([valid[m] for m in aliases]))[:, 1]
        calibrators = {method: train.fit_calibrator(nested["oof"][policy], yt, method)
                       for method in ("sigmoid", "isotonic")}
        probabilities = {"raw_fixed": p, "raw_threshold": p,
                         **{method: train.apply_calibrator(cal, p) for method, cal in calibrators.items()}}
        thresholds = {mode: (0.5 if mode == "raw_fixed" else utils.optimize_threshold(
                      development["valid_y"], probability)) for mode, probability in probabilities.items()}
        if policy == "locked_baseline":
            thresholds = dict(ready[arm_name(horizon, BASE, "own")]["thresholds"])
        model = stack.StackedPredictor({m: members[a]["estimator"] for m, a in zip(chosen, aliases)}, meta)
        arm_signature = utils.cache_signature(stage="stacking_arm", version=VERSION,
            nested=signature, policy=policy, definition=definition, thresholds=thresholds,
            threshold_grid=config.THRESHOLD_GRID.tolist(), threshold_strategy=config.THRESHOLD_STRATEGY)
        arm = {"name": name, "horizon": horizon, "feature_set": feature_set, "policy": policy,
               "configuration": definition["configuration"], "members": chosen, "C": float(definition["C"]),
               "model": model, "calibrators": calibrators, "thresholds": thresholds,
               "signature": arm_signature, "nested_signature": signature,
               "validation": probabilities}
        pred.store(arm, paths["models"] / f"{name}.joblib", arm_signature)
        utils.write_manifest(paths["models"] / f"{name}_manifest.json", artifact=name,
                             signature=arm_signature, members=chosen, C=arm["C"], policy=policy,
                             calibration="nested training OOF", thresholds=thresholds)
        frame = pd.DataFrame({config.ID_COL: development["valid_ids"], "y_true": development["valid_y"],
                              **{f"proba_{mode}": probability for mode, probability in probabilities.items()}})
        utils.write_frame(frame, paths["predictions"] / f"{name}_valid.csv")
        ready[name] = arm


def evaluate_arm(arm, data, paths, force=False):
    X, y, ids = pred.partition(data, "test")
    fingerprint = utils.data_fingerprint(pd.DataFrame({config.ID_COL: ids, "y_true": y,
                                                       **{c: X[c].to_numpy() for c in X}}))
    signature = utils.cache_signature(stage="stacking_test", version=VERSION,
                                      arm=arm["signature"], test=fingerprint)
    path = paths["predictions"] / f"{arm['name']}_test.csv"
    manifest = path.with_name(f"{path.stem}_manifest.json")
    if not force and path.exists() and manifest.exists() and pred.read_json(manifest)["signature"] == signature:
        frame = pd.read_csv(path)
    else:
        raw = arm["model"].raw_proba(X)
        frame = pd.DataFrame({config.ID_COL: ids, "y_true": y, "proba_raw_fixed": raw,
                              "proba_raw_threshold": raw})
        for method, calibrator in arm["calibrators"].items():
            frame[f"proba_{method}"] = train.apply_calibrator(calibrator, raw)
        for mode, threshold in arm["thresholds"].items():
            frame[f"prediction_{mode}"] = (frame[f"proba_{mode}"] >= threshold).astype(int)
        utils.write_frame(frame, path)
        utils.write_manifest(manifest, artifact=arm["name"], signature=signature,
                             thresholds=arm["thresholds"], selection_source="validation only")
    utils.check_unique_ids(frame)
    utils.assert_same_patients(ids, frame[config.ID_COL].to_numpy())
    p = frame[[f"proba_{mode}" for mode in stack.MODES]].to_numpy()
    if not np.array_equal(y, frame.y_true) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid frozen stacking test predictions")
    return frame


# ---------------------------------------------------------------------------
# Paired comparisons, calibration and clinical utility
# ---------------------------------------------------------------------------

def evaluate_comparisons(ready, evaluated, paths, n_boot):
    comparisons, existing = [], []
    counts, resampled = {}, {}

    def compare(horizon, base_name, other_name, mode, reference=None):
        other = evaluated[other_name]
        base = evaluated[base_name] if reference is None else reference[0]
        utils.assert_same_patients(base[config.ID_COL].to_numpy(), other[config.ID_COL].to_numpy())
        y = other.y_true.to_numpy()
        if not np.array_equal(y, base.y_true):
            raise ValueError("Paired outcomes do not match")
        if horizon not in counts:
            counts[horizon] = stack.bootstrap_counts(y, n_boot)
        po = other[f"proba_{mode}"].to_numpy()
        to = ready[other_name]["thresholds"][mode]
        pb = base[f"proba_{mode}"].to_numpy() if reference is None else base.proba_calibrated.to_numpy()
        tb = ready[base_name]["thresholds"][mode] if reference is None else reference[1]
        for name, p, t in [(base_name, pb, tb), (other_name, po, to)]:
            key = (name, mode)
            if key not in resampled:
                resampled[key] = stack.bootstrap_metrics(y, p, t, counts[horizon])
        return stack.paired_metrics(y, pb, po, tb, to, counts[horizon],
                                    resampled[base_name, mode], resampled[other_name, mode])

    for name, arm in ready.items():
        if arm["feature_set"] == BASE:
            continue
        base_policy = "paper" if arm["policy"] == "paper" else "own"
        base_name = arm_name(arm["horizon"], BASE, base_policy)
        for mode in stack.MODES:
            delta = compare(arm["horizon"], base_name, name, mode)
            comparisons.append(delta.assign(horizon=arm["horizon"], feature_set=arm["feature_set"],
                                             policy=arm["policy"], mode=mode,
                                             base_arm=base_name, other_arm=name))
    deltas = pd.concat(comparisons, ignore_index=True)
    deltas["p_holm"] = np.nan
    for mode in stack.MODES:
        mask = (deltas.metric == "f1_macro") & (deltas["mode"] == mode) & deltas.policy.isin(["own", "same_baseline"])
        deltas.loc[mask, "p_holm"] = stack.holm_adjust(deltas.loc[mask, "p_bootstrap"])
    utils.write_frame(deltas, paths["results"] / "paired_deltas.csv")
    source = config.RESULTS_DIR / "predictive_sensitivity"
    selected = utils.read_frame(source / "selected_configurations.csv")
    previous = utils.read_frame(source / "test_metrics.csv")
    for name, arm in ready.items():
        if arm["policy"] not in ("own", "paper"):
            continue
        h, fs = arm["horizon"], arm["feature_set"]
        if arm["policy"] == "own":
            chosen = selected[(selected.horizon == h) & (selected.feature_set == fs)
                              & (selected.scope == "all configurations")].iloc[0]
            ref = previous[(previous.horizon == h) & (previous.feature_set == fs)
                           & (previous.configuration == chosen.configuration) & (previous.params_source == fs)
                           & (previous.probabilities == "calibrated")].iloc[0]
            reference_name, reference_configuration, reference_threshold = ref.arm, chosen.configuration, float(ref.threshold)
            frame = utils.read_frame(config.PRED_DIR / "predictive_sensitivity" / f"{ref.arm}_test.csv")
        else:
            reference_name, reference_configuration = f"h{h}_{fs}_paper", "paper"
            frame = utils.read_frame(config.PRED_DIR / f"{reference_name}_test.csv")
            reference_threshold = float(pred.read_json(config.PRED_DIR / f"{reference_name}_test_manifest.json")["threshold"])
        for mode in ("raw_threshold", "sigmoid", "isotonic"):
            delta = compare(h, f"reference_{reference_name}", name, mode, (frame, reference_threshold))
            existing.append(delta.assign(horizon=h, feature_set=fs, mode=mode,
                                         policy=arm["policy"], reference=reference_configuration, other_arm=name))
    utils.write_frame(pd.concat(existing, ignore_index=True), paths["results"] / "versus_existing.csv")


def write_summary(paths):
    manifest = pred.read_json(paths["results"] / "run_manifest.json")
    if not manifest.get("complete"):
        raise ValueError("Complete the stacking run before rebuilding its summary")
    metrics = utils.read_frame(paths["results"] / "test_metrics.csv")
    deltas = utils.read_frame(paths["results"] / "paired_deltas.csv")
    choices = utils.read_frame(paths["results"] / "selected_configurations.csv")
    frozen = pred.read_json(paths["results"] / "frozen_selection.json")
    validation, reliability = [], []
    for name, arm in frozen["arms"].items():
        valid = utils.read_frame(paths["predictions"] / f"{name}_valid.csv")
        test = utils.read_frame(paths["predictions"] / f"{name}_test.csv")
        metadata = {"arm": name, "horizon": arm["horizon"], "feature_set": arm["feature_set"],
                    "policy": arm["policy"], "configuration": arm["configuration"]}
        for mode in stack.MODES:
            p = valid[f"proba_{mode}"]
            validation.append({**metadata, "mode": mode,
                **utils.classification_metrics(valid.y_true, p, arm["thresholds"][mode]),
                **train.calibration_report(valid.y_true, p)})
            reliability.append(utils.reliability_points(test.y_true, test[f"proba_{mode}"]).assign(
                               **metadata, mode=mode))
    utils.write_frame(pd.DataFrame(validation), paths["results"] / "validation_metrics.csv")
    utils.write_frame(pd.concat(reliability, ignore_index=True), paths["results"] / "test_reliability.csv")
    lines = ["# Supplementary stacking sensitivity", "",
             "All membership, meta-model settings, calibrators and thresholds were fixed before new test evaluation.",
             "Sigmoid and isotonic calibration use nested training OOF predictions of the entire stack.",
             "The analysis is post hoc: intervals condition on selected configurations and the stored split.",
             "They do not account for configuration-selection uncertainty or previous test inspection.",
             "Holm adjustment covers the independent and same-baseline macro-F1 comparisons across feature sets",
             "and horizons, separately for each probability/threshold mode. Other intervals are unadjusted.", "",
             "## Selected configurations", ""]
    for row in choices.itertuples():
        lines.append(f"- {row.horizon} years, {row.feature_set}: {row.configuration}; validation macro-F1 {row.f1_macro:.4f}.")
    lines.extend(["", "## Sigmoid calibration", "",
                  "| Horizon | Features | Policy | Test macro-F1 | AUROC | AP | Brier |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in metrics[metrics["mode"] == "sigmoid"].itertuples():
        lines.append(f"| {row.horizon} | {row.feature_set} | {row.policy} | {row.f1_macro:.4f} | "
                     f"{row.roc_auc:.4f} | {row.auprc:.4f} | {row.brier:.4f} |")
    lines.extend(["", "## Thyroid differences", ""])
    for row in deltas[(deltas.metric == "f1_macro") & (deltas["mode"] == "sigmoid")].itertuples():
        lines.append(f"- {row.horizon} years, {row.feature_set}, {row.policy}: {row.delta:+.4f}, "
                     f"95% CI [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}].")
    (paths["results"] / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_figures(paths)


def write_figures(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = utils.read_frame(paths["results"] / "paired_deltas.csv")
    for horizon in sorted(table.horizon.unique()):
        subset = table[(table.horizon == horizon) & (table.metric == "f1_macro")
                       & table.policy.isin(["own", "same_baseline"])]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
        for ax, mode in zip(axes, ["raw_threshold", "sigmoid"]):
            rows = subset[subset["mode"] == mode].reset_index(drop=True)
            positions = np.arange(len(rows))
            ax.hlines(positions, rows.ci_lo, rows.ci_hi, color="tab:blue")
            ax.plot(rows.delta, positions, "o", color="tab:blue")
            ax.axvline(0, linestyle="--", color="0.5", linewidth=1)
            ax.set_yticks(positions, [f"{r.feature_set}\n{r.policy}" for r in rows.itertuples()], fontsize=8)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            ax.set_title(f"{horizon} years; {mode}")
            ax.set_xlabel("Thyroid minus CV17: macro-F1, paired 95% interval")
        fig.savefig(paths["figures"] / f"h{horizon}_stacking_deltas.png", dpi=150)
        plt.close(fig)
    reliability = utils.read_frame(paths["results"] / "test_reliability.csv")
    curves = utils.read_frame(paths["results"] / "decision_curves.csv")
    for horizon in sorted(table.horizon.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        axes[0].plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
        for feature_set, color in [(BASE, "tab:blue"), (config.PRIMARY_THYROID, "tab:orange")]:
            for mode, style in [("sigmoid", "-"), ("isotonic", ":")]:
                mask = ((reliability.horizon == horizon) & (reliability.feature_set == feature_set)
                        & (reliability.policy == "own") & (reliability["mode"] == mode))
                points = reliability[mask]
                label = f"{feature_set}, {mode}"
                axes[0].plot(points.mean_predicted, points.observed_rate, style, marker="o",
                             color=color, label=label, markersize=4)
                mask = ((curves.horizon == horizon) & (curves.feature_set == feature_set)
                        & (curves.policy == "own") & (curves["mode"] == mode))
                points = curves[mask]
                axes[1].plot(points.threshold, points.net_benefit_model, style, color=color, label=label)
        axes[0].set(xlabel="Mean predicted probability", ylabel="Observed event rate",
                    title=f"{horizon} years: test calibration", xlim=(0, 1), ylim=(0, 1))
        reference = curves[(curves.horizon == horizon) & (curves.feature_set == BASE)
                           & (curves.policy == "own") & (curves["mode"] == "sigmoid")]
        axes[1].plot(reference.threshold, reference.net_benefit_treat_all, "--", color="0.5", label="Treat all")
        axes[1].axhline(0, color="0.5", linestyle=":", linewidth=1, label="Treat none")
        axes[1].set(xlabel="Risk threshold", ylabel="Net benefit", title=f"{horizon} years: decision curves")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.savefig(paths["figures"] / f"h{horizon}_stacking_calibration.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver: complete development before evaluating any selected test arm
# ---------------------------------------------------------------------------

def original_artifacts():
    return {p: (p.stat().st_size, p.stat().st_mtime_ns)
            for root in [config.DATA_PROC, config.RESULTS_DIR, config.MODELS_DIR,
                         config.PRED_DIR, config.CACHE_DIR, config.FIGURES_DIR]
            for p in root.rglob("*") if p.is_file()
            and not any(part.startswith("stacking_sensitivity") for part in p.parts)}


def run(args):
    paths = output_paths(args.smoke)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.summary:
        write_summary(paths)
        return
    before = original_artifacts()
    utils.write_manifest(paths["results"] / "run_manifest.json", complete=False,
                         analysis="post hoc stacking sensitivity")
    horizons = (7,) if args.smoke else config.HORIZONS
    feature_sets = (BASE, config.PRIMARY_THYROID) if args.smoke else config.TUNING_FEATURE_SETS
    n_boot = 20 if args.smoke else config.BOOTSTRAP
    reference = reference_scores()
    all_scores, selected, ready, signatures = [], [], {}, {}
    for horizon in horizons:
        baseline = None
        for feature_set in feature_sets:
            pred.log(f"Stacking development: h{horizon} {feature_set}")
            data, dev = development(horizon, feature_set, reference, paths, args.jobs, args.force)
            scores = validation_search(horizon, feature_set, dev, paths, args.force)
            winner = scores.iloc[0].to_dict()
            paper_members = ",".join(model for model in config.MODELS if model in config.PAPER_ENSEMBLE)
            paper = pred.rank_scores(scores[scores.members == paper_members]).iloc[0].to_dict()
            if feature_set == BASE:
                baseline = winner
            if baseline is None:
                raise ValueError("CV17 must precede the thyroid feature sets")
            all_scores.append(scores)
            selected.append(winner)
            signatures[horizon, feature_set] = data["signature"]
            pred.log(f"Selected h{horizon} {feature_set}: {winner['configuration']}")
            prepare_group(horizon, feature_set, data, dev, winner, baseline, paper,
                          ready, reference, paths, args.jobs, args.force)
            del data, dev
    utils.write_frame(pd.concat(all_scores, ignore_index=True), paths["results"] / "validation_scores.csv")
    utils.write_frame(pd.DataFrame(selected), paths["results"] / "selected_configurations.csv")
    selection = {name: {key: arm[key] for key in ["horizon", "feature_set", "policy", "configuration",
                        "members", "C", "signature", "nested_signature", "thresholds"]}
                 for name, arm in ready.items()}
    utils.write_manifest(paths["results"] / "frozen_selection.json", version=VERSION,
                         analysis="post hoc stacking sensitivity", candidate_count=sum(len(t) for t in all_scores),
                         regularization=list(stack.REGULARIZATION), base_models=config.MODELS,
                         subset_sizes=[2, len(config.MODELS)], selection_rule="raw validation f1_macro, roc_auc, name",
                         calibration="nested training OOF", inner_cv=config.CALIBRATION_CV,
                         outer_cv=config.CALIBRATION_CV, threshold_grid=config.THRESHOLD_GRID.tolist(),
                         threshold_strategy=config.THRESHOLD_STRATEGY, n_boot=n_boot, arms=selection)
    pred.log(f"All {len(ready)} stacks and thresholds frozen; evaluating test")
    evaluated, metrics, calibration, curves, impact = {}, [], [], [], []
    for name, arm in ready.items():
        data = pred.load_partition(arm["horizon"], arm["feature_set"])
        if data["signature"] != signatures[arm["horizon"], arm["feature_set"]]:
            raise ValueError("Development inputs changed before test evaluation")
        frame = evaluate_arm(arm, data, paths, args.force)
        evaluated[name] = frame
        metadata = {"arm": name, "horizon": arm["horizon"], "feature_set": arm["feature_set"],
                    "policy": arm["policy"], "configuration": arm["configuration"]}
        for mode in stack.MODES:
            p, threshold = frame[f"proba_{mode}"], arm["thresholds"][mode]
            metrics.append({**metadata, "mode": mode,
                            **utils.classification_metrics(frame.y_true, p, threshold)})
            calibration.append({**metadata, "mode": mode, **train.calibration_report(frame.y_true, p)})
            curves.append(train.decision_curve(frame.y_true, p).assign(**metadata, mode=mode))
            impact.append(train.clinical_impact(frame.y_true, p).assign(**metadata, mode=mode))
    for name, rows in [("test_metrics", metrics), ("test_calibration", calibration)]:
        utils.write_frame(pd.DataFrame(rows), paths["results"] / f"{name}.csv")
    for name, rows in [("decision_curves", curves), ("clinical_impact", impact)]:
        utils.write_frame(pd.concat(rows, ignore_index=True), paths["results"] / f"{name}.csv")
    evaluate_comparisons(ready, evaluated, paths, n_boot)
    if before != original_artifacts():
        raise RuntimeError("An original artifact changed during the stacking run")
    utils.write_manifest(paths["results"] / "run_manifest.json", complete=True,
                         analysis="post hoc stacking sensitivity", n_arms=len(ready),
                         candidate_count=sum(len(t) for t in all_scores),
                         original_artifacts_unchanged=len(before), selection_manifest="frozen_selection.json")
    write_summary(paths)
    pred.log("Stacking sensitivities complete; original artifacts unchanged")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Primary contrast at seven years, separate outputs")
    parser.add_argument("--summary", action="store_true", help="Rebuild summary and figures without fitting")
    parser.add_argument("--force", action="store_true", help="Recompute supplementary stacking artifacts")
    parser.add_argument("--jobs", type=int, default=1, help="Workers for training OOF and nested base-model fits")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    with threadpool_limits(1), parallel_config(backend="loky", inner_max_num_threads=1):
        run(args)


if __name__ == "__main__":
    main()

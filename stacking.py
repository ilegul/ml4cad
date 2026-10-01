"""Stacking models, nested training OOF predictions and paired inference."""

import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import predictive as pred
import train
import utils

VERSION = "v1"
REGULARIZATION = (0.01, 0.1, 1.0, 10.0, 100.0)
MODES = ("raw_fixed", "raw_threshold", "sigmoid", "isotonic")


def candidate_subsets(models):
    """All subsets containing at least two distinct base classifiers."""
    if len(models) != len(set(models)) or len(models) < 2:
        raise ValueError("Stacking requires distinct base classifiers")
    return [list(members) for size in range(2, len(models) + 1)
            for members in itertools.combinations(models, size)]


def meta_model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, max_iter=3000, solver="lbfgs", random_state=config.SEED))


def candidate_name(members, C):
    return "+".join(members) + f";C={C:g}"


class StackedPredictor:
    """Frozen base pipelines followed by a meta-model and optional calibration."""

    def __init__(self, members, meta, calibrator=None):
        self.members = members
        self.meta = meta
        self.calibrator = calibrator

    def raw_proba(self, X):
        inputs = np.column_stack([model.predict_proba(X)[:, 1]
                                  for model in self.members.values()])
        return self.meta.predict_proba(inputs)[:, 1]

    def predict_proba(self, X):
        p = train.apply_calibrator(self.calibrator, self.raw_proba(X))
        return np.column_stack([1 - p, p])


def nested_member_fold(estimator, X, y, fit_indices, held_indices, fold,
                       member_signature, paths, jobs=1, force=False):
    """Inner OOF inputs and held-out inputs excluding the outer fold entirely."""
    signature = utils.cache_signature(
        stage="stacking_nested_member", version=VERSION, member=member_signature,
        outer_fold=fold, fit_indices=fit_indices.tolist(), held_indices=held_indices.tolist(),
        outer_cv=config.CALIBRATION_CV, inner_cv=config.CALIBRATION_CV)
    path = paths["cache"] / f"nested_{signature}.joblib"
    result = pred.cached(path, signature, force)
    if result is None:
        Xi, yi = X.iloc[fit_indices], y[fit_indices]
        inner = train.training_oof_proba(clone(estimator), Xi, yi, n_jobs=jobs)
        fitted = clone(estimator).fit(Xi, yi)
        result = {"inner": inner, "held": fitted.predict_proba(X.iloc[held_indices])[:, 1],
                  "fit_indices": fit_indices, "held_indices": held_indices}
        pred.store(result, path, signature)
    if (not np.array_equal(result["fit_indices"], fit_indices)
            or not np.array_equal(result["held_indices"], held_indices)):
        raise ValueError("Nested OOF partition does not match its cache")
    return result


def nested_stack_oof(X, y, members, definitions, paths, jobs=1, force=False):
    """Cross-fit the entire stacking procedure, sharing base fits across arms.

    Hyperparameters, samplers and membership remain frozen. The outer held-out
    patient is absent from every inner base fit and from the meta-model fit.
    """
    folds = StratifiedKFold(n_splits=config.CALIBRATION_CV, shuffle=True,
                           random_state=config.SEED)
    outputs = {name: np.full(len(y), np.nan) for name in definitions}
    visits = np.zeros(len(y), dtype=int)
    for fold, (fit_indices, held_indices) in enumerate(folds.split(X, y)):
        pred.log(f"Nested stacking OOF: fold {fold + 1}/{config.CALIBRATION_CV}")
        needed = sorted(set(m for definition in definitions.values() for m in definition["members"]))
        fitted = Parallel(n_jobs=jobs)(
            delayed(nested_member_fold)(members[name]["estimator"], X, y, fit_indices,
                                       held_indices, fold, members[name]["signature"],
                                       paths, 1, force) for name in needed)
        inputs = dict(zip(needed, fitted))
        for name, definition in definitions.items():
            chosen = definition["members"]
            inner = np.column_stack([inputs[m]["inner"] for m in chosen])
            held = np.column_stack([inputs[m]["held"] for m in chosen])
            meta = meta_model(definition["C"]).fit(inner, y[fit_indices])
            outputs[name][held_indices] = meta.predict_proba(held)[:, 1]
        visits[held_indices] += 1
    if not np.all(visits == 1) or any(not np.isfinite(p).all() for p in outputs.values()):
        raise ValueError("Incomplete nested stacking OOF coverage")
    return outputs


def bootstrap_counts(y, n_boot, seed=config.SEED):
    """Represent the same patient resamples as utils.bootstrap_indices."""
    y = np.asarray(y, dtype=int)
    draws = utils.bootstrap_indices(len(y), n_boot, seed)
    counts = np.stack([np.bincount(draw, minlength=len(y)) for draw in draws])
    positives = counts @ y
    return counts[(positives > 0) & (positives < len(y))]


def bootstrap_metrics(y, p, threshold, counts):
    """Exact weighted bootstrap metrics, including tied probability ranks."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if (counts.ndim != 2 or counts.shape[1] != len(y) or len(p) != len(y)
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Invalid aligned bootstrap inputs")
    positive = counts @ y
    negative = counts.sum(axis=1) - positive
    if np.any((positive == 0) | (negative == 0)):
        raise ValueError("Bootstrap metrics require both classes in each draw")
    predicted = p >= threshold
    tp = counts @ (y * predicted)
    fp = counts @ ((1 - y) * predicted)
    fn, tn = positive - tp, negative - fp
    f1 = 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order]) != 0) + 1]
    weights = counts[:, order]
    pos_group = np.add.reduceat(weights * y[order], starts, axis=1)
    neg_group = np.add.reduceat(weights * (1 - y[order]), starts, axis=1)
    auc = (pos_group * (np.cumsum(neg_group, axis=1) - 0.5 * neg_group)).sum(axis=1)
    auc /= positive * negative
    pos_desc, all_desc = pos_group[:, ::-1], (pos_group + neg_group)[:, ::-1]
    cumulative = np.cumsum(all_desc, axis=1)
    precision = np.divide(np.cumsum(pos_desc, axis=1), cumulative,
                          out=np.zeros_like(pos_desc, dtype=float), where=cumulative > 0)
    ap = (pos_desc * precision).sum(axis=1) / positive
    brier = counts @ ((p - y) ** 2) / counts.sum(axis=1)
    return {"f1_macro": f1, "roc_auc": auc, "auprc": ap, "brier": brier}


def paired_metrics(y, base, other, threshold_base, threshold_other,
                   counts, base_draws=None, other_draws=None):
    base_draws = (bootstrap_metrics(y, base, threshold_base, counts)
                  if base_draws is None else base_draws)
    other_draws = (bootstrap_metrics(y, other, threshold_other, counts)
                   if other_draws is None else other_draws)
    rows = []
    for metric, function in utils.METRIC_FUNCS.items():
        delta = other_draws[metric] - base_draws[metric]
        lo, hi = np.percentile(delta, [2.5, 97.5]) if len(delta) else (np.nan, np.nan)
        rows.append({"metric": metric,
                     "delta": function(y, other, threshold_other) - function(y, base, threshold_base),
                     "ci_lo": lo, "ci_hi": hi, "p_bootstrap": utils.bootstrap_pvalue(delta),
                     "n_boot_used": len(delta), "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm adjustment requires finite probabilities")
    order = np.argsort(p, kind="stable")
    corrected = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = corrected
    return result

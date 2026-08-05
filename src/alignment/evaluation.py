"""
src/alignment/evaluation.py
───────────────────────────
Classification evaluation for the two validation schemes plus the paired
incremental-value analysis (paper-alignment).

Two schemes, both leakage-free and both reusing the same metrics:

  Scheme A — 5-fold stratified CV (the repo's scheme).
      For every fold the threshold is tuned on an *inner* validation split of
      the training fold and applied to the held-out fold. Out-of-fold (OOF)
      probabilities are collected for every patient (used later as the ML
      indicator in survival).

  Scheme B — single 60/20/20 split (the paper's scheme).
      Threshold is tuned on the validation part; metrics are reported on the
      test part; test probabilities are returned.

Metrics: F1-macro (primary), AUROC, precision/recall per class, Brier.

The incremental value of a thyroid feature set over CV17 is measured **paired**:
identical folds / identical split, with a 95% bootstrap CI on Δ F1-macro and
Δ AUROC.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE, CV_FOLDS, THRESHOLD_GRID
from src.alignment.ensemble import build_pipeline
from src.alignment.holdout_split import split_60_20_20


# ─── helpers ─────────────────────────────────────────────────────────────

def optimize_threshold(y_true, y_proba, grid=THRESHOLD_GRID):
    """Return (best_threshold, best_f1macro) maximising F1-macro on the grid."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = f1_score(y_true, (y_proba >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def _metrics(y_true, y_proba, threshold) -> dict:
    """Compute the alignment metric panel at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1_macro":    f1_score(y_true, y_pred, average="macro"),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "precision_1": precision_score(y_true, y_pred, pos_label=1,
                                       zero_division=0),
        "recall_1":    recall_score(y_true, y_pred, pos_label=1,
                                    zero_division=0),
        "precision_0": precision_score(y_true, y_pred, pos_label=0,
                                       zero_division=0),
        "recall_0":    recall_score(y_true, y_pred, pos_label=0,
                                    zero_division=0),
        "brier":       brier_score_loss(y_true, y_proba),
        "threshold":   threshold,
    }


def _tune_threshold_inner(X_train, y_train, model_name, sampler, seed):
    """Tune the threshold on an inner 75/25 split of the training fold."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=seed)
    pipe = build_pipeline(model_name, sampler=sampler, random_state=seed)
    pipe.fit(X_fit, y_fit)
    proba = pipe.predict_proba(X_val)[:, 1]
    thr, _ = optimize_threshold(y_val, proba)
    return thr


# ─── Scheme A: 5-fold CV ────────────────────────────────────────────────

def evaluate_cv(X, y, model_name, sampler="RandomOverSampler",
                n_splits=CV_FOLDS, seed=RANDOM_STATE, tune_threshold=True):
    """
    5-fold stratified CV evaluation.

    Returns dict with:
      - per-metric ``{m}_mean`` / ``{m}_std`` across folds,
      - ``oof_proba`` : np.array of out-of-fold P(y=1) aligned to X rows,
      - ``oof_index`` : the index of X (for joining to survival data),
      - ``oof_pred`` : hard predictions using each outer fold's independently
        tuned threshold,
      - ``oof_threshold`` : threshold applied to each OOF observation.
    """
    X = X.reset_index(drop=False)
    index_col = X.columns[0]
    orig_index = X[index_col].values
    X = X.drop(columns=[index_col])
    y = pd.Series(np.asarray(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_threshold = np.full(len(y), np.nan)
    fold_rows = []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        if tune_threshold:
            thr = _tune_threshold_inner(X_tr, y_tr, model_name, sampler, seed)
        else:
            thr = 0.5

        pipe = build_pipeline(model_name, sampler=sampler, random_state=seed)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        oof_proba[te] = proba
        oof_pred[te] = (proba >= thr).astype(int)
        oof_threshold[te] = thr
        fold_rows.append(_metrics(y_te, proba, thr))

    fm = pd.DataFrame(fold_rows)
    out = {}
    for col in fm.columns:
        out[f"{col}_mean"] = float(fm[col].mean())
        out[f"{col}_std"] = float(fm[col].std())
    out["oof_proba"] = oof_proba
    out["oof_pred"] = oof_pred
    out["oof_threshold"] = oof_threshold
    out["oof_index"] = orig_index
    out["scheme"] = "A_5fold_cv"
    out["model"] = model_name
    return out


# ─── Scheme B: 60/20/20 holdout ─────────────────────────────────────────

def evaluate_holdout(X, y, model_name, sampler="RandomOverSampler",
                     seed=RANDOM_STATE, tune_threshold=True):
    """
    60/20/20 holdout evaluation: tune threshold on val, report on test.

    Returns dict with the metric panel (flat) plus:
      - ``test_proba``  : P(y=1) on the test part,
      - ``test_index``  : original X index of the test rows,
      - ``threshold``   : threshold tuned on the validation part.
    """
    X = X.reset_index(drop=False)
    index_col = X.columns[0]
    orig_index = X[index_col].values
    X = X.drop(columns=[index_col])
    y = pd.Series(np.asarray(y))

    idx_tr, idx_val, idx_te = split_60_20_20(y, seed=seed)
    X_tr, X_val, X_te = X.iloc[idx_tr], X.iloc[idx_val], X.iloc[idx_te]
    y_tr, y_val, y_te = y.iloc[idx_tr], y.iloc[idx_val], y.iloc[idx_te]

    pipe = build_pipeline(model_name, sampler=sampler, random_state=seed)
    pipe.fit(X_tr, y_tr)

    if tune_threshold:
        proba_val = pipe.predict_proba(X_val)[:, 1]
        thr, _ = optimize_threshold(y_val, proba_val)
    else:
        thr = 0.5

    proba_te = pipe.predict_proba(X_te)[:, 1]
    out = _metrics(y_te, proba_te, thr)
    out["test_proba"] = proba_te
    out["test_pred"] = (proba_te >= thr).astype(int)
    out["test_index"] = orig_index[idx_te]   # original labels (survival join)
    out["test_pos"] = idx_te                  # positional (paired Δ on y order)
    out["test_y"] = y_te.values
    out["n_train"] = len(idx_tr)
    out["n_val"] = len(idx_val)
    out["n_test"] = len(idx_te)
    out["scheme"] = "B_60_20_20"
    out["model"] = model_name
    return out


# ─── Paired incremental value with bootstrap CI ─────────────────────────

def _paired_delta(y_true, proba_base, proba_thy, pred_base, pred_thy,
                  n_boot=1000, seed=RANDOM_STATE):
    """
    Bootstrap the paired Δ between a thyroid set and the base set on the SAME
    samples/predictions. Hard predictions must already use thresholds selected
    without seeing the evaluated observations: inner validation for scheme A,
    validation split for scheme B. No threshold is re-tuned on OOF/test labels.

    Returns dict with Δ F1-macro and Δ AUROC point estimates + 95% CI.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba_base = np.asarray(proba_base)
    proba_thy = np.asarray(proba_thy)
    pred_base = np.asarray(pred_base)
    pred_thy = np.asarray(pred_thy)

    def _f1(y, pred):
        return f1_score(y, pred, average="macro")

    d_f1_point = _f1(y_true, pred_thy) - _f1(y_true, pred_base)
    d_auc_point = (roc_auc_score(y_true, proba_thy)
                   - roc_auc_score(y_true, proba_base))

    n = len(y_true)
    d_f1s, d_aucs = [], []
    for _ in range(n_boot):
        bs = rng.integers(0, n, n)
        yb = y_true[bs]
        if yb.min() == yb.max():       # need both classes for AUROC
            continue
        d_f1s.append(_f1(yb, pred_thy[bs]) - _f1(yb, pred_base[bs]))
        d_aucs.append(roc_auc_score(yb, proba_thy[bs])
                      - roc_auc_score(yb, proba_base[bs]))

    def _ci(arr):
        return (float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)))

    f1_lo, f1_hi = _ci(d_f1s)
    auc_lo, auc_hi = _ci(d_aucs)
    return {
        "delta_f1_macro": float(d_f1_point),
        "delta_f1_ci_lo": f1_lo, "delta_f1_ci_hi": f1_hi,
        "delta_auroc": float(d_auc_point),
        "delta_auroc_ci_lo": auc_lo, "delta_auroc_ci_hi": auc_hi,
    }


def incremental_value_cv(X_base, X_thy, y, model_name="ENSEMBLE",
                         sampler="RandomOverSampler", n_splits=CV_FOLDS,
                         seed=RANDOM_STATE, n_boot=1000):
    """
    Paired Δ (thyroid set vs CV17) under scheme A.

    Both feature sets are evaluated on the *same* folds (same seed), OOF
    probabilities are aligned by row, then the paired bootstrap Δ is computed.
    ``X_base`` and ``X_thy`` must share the same row order / index.
    """
    res_b = evaluate_cv(X_base, y, model_name, sampler, n_splits, seed)
    res_t = evaluate_cv(X_thy, y, model_name, sampler, n_splits, seed)
    y_arr = np.asarray(pd.Series(np.asarray(y)))
    delta = _paired_delta(
        y_arr, res_b["oof_proba"], res_t["oof_proba"],
        res_b["oof_pred"], res_t["oof_pred"],
        n_boot=n_boot, seed=seed)
    delta["scheme"] = "A_5fold_cv"
    return delta


def incremental_value_holdout(X_base, X_thy, y, model_name="ENSEMBLE",
                              sampler="RandomOverSampler", seed=RANDOM_STATE,
                              n_boot=1000):
    """
    Paired Δ (thyroid set vs CV17) under scheme B, on the identical test split.
    """
    res_b = evaluate_holdout(X_base, y, model_name, sampler, seed)
    res_t = evaluate_holdout(X_thy, y, model_name, sampler, seed)
    y_te = res_b["test_y"]              # same split (same seed) -> same test rows
    delta = _paired_delta(
        y_te, res_b["test_proba"], res_t["test_proba"],
        res_b["test_pred"], res_t["test_pred"],
        n_boot=n_boot, seed=seed)
    delta["scheme"] = "B_60_20_20"
    return delta

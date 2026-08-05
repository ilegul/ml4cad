"""
src/alignment/ablation.py
─────────────────────────
Knock-out ablation test, in the style of the paper (Table 4 / Table 5),
extended to ALL features including the thyroid-derived ones (fT3/fT4 ratio,
ordinal, binary, one-hot states).

Method (per feature / per group):
    1. Fit the ensemble (leakage-free pipeline) on the training part.
    2. Baseline F1-macro on the test part at a threshold tuned on validation.
    3. Knock the feature out by replacing it with its (training) MEAN value on
       the test part, recompute F1' with the same fitted model and threshold.
    4. importance = F1 / F1'   ( >1 = useful, <1 = noise ).

Two flavours, both run on the SAME fitted model:
  - single-variable: one feature at a time.
  - multi-variable: all predictors are grouped by hierarchical clustering,
    matching the paper's seven-cluster ablation workflow.

A 60/20/20 split (scheme B) provides train / val(threshold) / test, so the
knock-out is evaluated on held-out rows and never leaks.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import f1_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE, CARDIO_17
from src.alignment.ensemble import build_pipeline
from src.alignment.evaluation import optimize_threshold
from src.alignment.holdout_split import split_60_20_20


def _thyroid_columns(cols) -> list:
    """Thyroid / thyroid-derived columns = everything not in CARDIO_17."""
    return [c for c in cols if c not in CARDIO_17]


def _cluster_features(X_train: pd.DataFrame, n_clusters=7) -> dict:
    """
    Hierarchical clustering of all predictors by |Spearman correlation|.

    Distance = 1 - |rho|; average linkage; the dendrogram is cut into at most
    ``n_clusters`` groups, as in the paper's seven-cluster analysis.
    Returns {cluster_label: [columns]}.
    """
    cols = list(X_train.columns)
    if len(cols) <= 1:
        return {1: cols}
    corr = X_train.corr(method="spearman").abs().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr
    dist = (dist + dist.T) / 2.0          # enforce symmetry
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=min(n_clusters, len(cols)), criterion="maxclust")
    groups = {}
    for col, lab in zip(cols, labels):
        groups.setdefault(int(lab), []).append(col)
    return groups


def _fit_and_baseline(X, y, model_name, sampler, seed):
    """Fit on train, tune threshold on val, return (pipe, X_test, y_test, thr,
    train_means, baseline_f1)."""
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    idx_tr, idx_val, idx_te = split_60_20_20(y, seed=seed)
    X_tr, X_val, X_te = X.iloc[idx_tr], X.iloc[idx_val], X.iloc[idx_te]
    y_tr, y_val, y_te = y.iloc[idx_tr], y.iloc[idx_val], y.iloc[idx_te]

    pipe = build_pipeline(model_name, sampler=sampler, random_state=seed)
    pipe.fit(X_tr, y_tr)
    thr, _ = optimize_threshold(y_val, pipe.predict_proba(X_val)[:, 1])

    proba_te = pipe.predict_proba(X_te)[:, 1]
    base_f1 = f1_score(y_te, (proba_te >= thr).astype(int), average="macro")
    train_means = X_tr.mean(numeric_only=True)
    return pipe, X_tr, X_te, y_te, thr, train_means, base_f1


def _knockout_f1(pipe, X_te, y_te, thr, cols_to_knock, train_means):
    """F1-macro after replacing ``cols_to_knock`` with their training mean."""
    X_mod = X_te.copy()
    for c in cols_to_knock:
        X_mod[c] = train_means[c]
    proba = pipe.predict_proba(X_mod)[:, 1]
    return f1_score(y_te, (proba >= thr).astype(int), average="macro")


def run_ablation(X, y, feature_set_name, model_name="ENSEMBLE",
                 sampler="RandomOverSampler", seed=RANDOM_STATE,
                 n_clusters=7) -> pd.DataFrame:
    """
    Single- and multi-variable knock-out ablation for one feature set.

    Returns a ranking DataFrame with columns:
      feature_set, mode (single|multi), feature/group, members,
      baseline_f1, knockout_f1, importance (=baseline/knockout), is_thyroid.
    """
    pipe, X_tr, X_te, y_te, thr, train_means, base_f1 = _fit_and_baseline(
        X, y, model_name, sampler, seed)
    thy_cols = _thyroid_columns(X.columns)
    rows = []

    # ── single-variable ──
    for col in X.columns:
        ko = _knockout_f1(pipe, X_te, y_te, thr, [col], train_means)
        rows.append({
            "feature_set": feature_set_name, "mode": "single",
            "feature": col, "members": col,
            "baseline_f1": base_f1, "knockout_f1": ko,
            "importance": base_f1 / ko if ko > 0 else np.inf,
            "is_thyroid": col in thy_cols,
        })

    # ── multi-variable: cluster ALL predictors, matching the paper ──
    groups = {
        f"cluster_{lab}": members
        for lab, members in _cluster_features(X_tr, n_clusters).items()
    }

    for gname, members in groups.items():
        ko = _knockout_f1(pipe, X_te, y_te, thr, members, train_means)
        rows.append({
            "feature_set": feature_set_name, "mode": "multi",
            "feature": gname, "members": ", ".join(members),
            "baseline_f1": base_f1, "knockout_f1": ko,
            "importance": base_f1 / ko if ko > 0 else np.inf,
            "is_thyroid": any(m in thy_cols for m in members),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["mode", "importance"], ascending=[True, False])
    return out.reset_index(drop=True)

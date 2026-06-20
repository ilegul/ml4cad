"""
src/classification/robust_cv.py
────────────────────────────────
Robust cross-validation with decision-threshold optimisation.
Pipeline.md section 6 -- robust CV.

StratifiedKFold 5-fold on strong models (LR, RF, HistGB, XGB) with SMOTE.
Threshold optimised on np.linspace(0.1, 0.9, 41) -> max F1-macro.
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, RESULTS_CV_CSV, RANDOM_STATE, CV_FOLDS,
    ROBUST_CV_MODELS, ROBUST_CV_SAMPLER, THRESHOLD_GRID,
)
from src.features.feature_engineering import extract_Xy, add_derived_features
from src.classification.screening import _build_pipeline


def _optimize_threshold(y_true, y_proba, grid=THRESHOLD_GRID):
    """Find the threshold that maximises F1-macro."""
    best_thr, best_f1 = 0.5, 0.0
    for thr in grid:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def _load_existing() -> set:
    """Load already-computed combinations."""
    if RESULTS_CV_CSV.exists():
        df = pd.read_csv(RESULTS_CV_CSV)
        if ("threshold_strategy" not in df.columns or
                (df["threshold_strategy"] != "inner_validation").any()):
            print("[ROBUST CV] Existing results_cv.csv uses an old threshold "
                  "strategy; recomputing robust CV results.")
            RESULTS_CV_CSV.unlink()
            return set()
        return {(r["cohort"], r["feature_set"], r["model"])
                for _, r in df.iterrows()}
    return set()


def _optimize_threshold_inner_validation(X_train: pd.DataFrame,
                                         y_train: pd.Series,
                                         model_name: str,
                                         sampler_name: str) -> float:
    """
    Select the decision threshold on an inner validation split.

    The outer-fold model is later refit on the full outer training fold; this
    keeps threshold selection separate from the held-out outer test fold while
    avoiding in-sample threshold tuning.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train,
        random_state=RANDOM_STATE)
    threshold_pipe = _build_pipeline(model_name, sampler_name)
    threshold_pipe.fit(X_fit, y_fit)
    y_val_proba = threshold_pipe.predict_proba(X_val)[:, 1]
    best_thr, _ = _optimize_threshold(y_val, y_val_proba)
    return best_thr


def run(cohorts: dict, force: bool = False):
    """
    Robust CV: StratifiedKFold 5-fold.
    cohorts: {"strict": df, "competing": df}
    """
    print("\n" + "=" * 60)
    print("[ROBUST CV] Cross-validation with threshold optimisation")
    print("=" * 60)

    if force and RESULTS_CV_CSV.exists():
        RESULTS_CV_CSV.unlink()

    done = set() if force else _load_existing()
    cohort_names = ["strict", "competing"]
    fs_names = list(FEATURE_SETS.keys())

    for coh_name in cohort_names:
        df = cohorts[coh_name].copy()
        df = add_derived_features(df)

        for fs_name in fs_names:
            try:
                X, y = extract_Xy(df, fs_name)
            except Exception as e:
                print(f"  [SKIP] {coh_name}/{fs_name}: {e}")
                continue

            if len(X) == 0:
                continue

            for model_name in ROBUST_CV_MODELS:
                key = (coh_name, fs_name, model_name)
                if key in done:
                    continue

                print(f"  {coh_name}/{fs_name}/{model_name} ...", end=" ")

                skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                      random_state=RANDOM_STATE)
                fold_metrics = []

                for fold_i, (train_idx, test_idx) in enumerate(
                        skf.split(X, y)):
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    X_test  = X.iloc[test_idx]
                    y_test  = y.iloc[test_idx]

                    try:
                        best_thr = _optimize_threshold_inner_validation(
                            X_train, y_train, model_name, ROBUST_CV_SAMPLER)

                        pipe = _build_pipeline(model_name, ROBUST_CV_SAMPLER)
                        pipe.fit(X_train, y_train)
                        y_proba = pipe.predict_proba(X_test)[:, 1]

                        # Apply optimal threshold on test
                        y_pred = (y_proba >= best_thr).astype(int)
                        f1m = f1_score(y_test, y_pred, average="macro")
                        auc = roc_auc_score(y_test, y_proba)

                        # Also F1 with default threshold 0.5
                        y_pred_def = (y_proba >= 0.5).astype(int)
                        f1m_default = f1_score(y_test, y_pred_def,
                                                average="macro")

                        fold_metrics.append({
                            "f1_macro_opt": f1m,
                            "f1_macro_default": f1m_default,
                            "roc_auc": auc,
                            "threshold": best_thr,
                        })
                    except Exception as e:
                        print(f"\n    [ERROR fold {fold_i}] {e}")
                        continue

                if not fold_metrics:
                    print("SKIP (no successful fold)")
                    continue

                fm = pd.DataFrame(fold_metrics)
                row = {
                    "cohort": coh_name,
                    "feature_set": fs_name,
                    "model": model_name,
                    "sampler": ROBUST_CV_SAMPLER,
                    "cv_folds": CV_FOLDS,
                    "threshold_strategy": "inner_validation",
                    "f1_macro_opt_mean":   fm["f1_macro_opt"].mean(),
                    "f1_macro_opt_std":    fm["f1_macro_opt"].std(),
                    "f1_macro_def_mean":   fm["f1_macro_default"].mean(),
                    "f1_macro_def_std":    fm["f1_macro_default"].std(),
                    "roc_auc_mean":        fm["roc_auc"].mean(),
                    "roc_auc_std":         fm["roc_auc"].std(),
                    "threshold_mean":      fm["threshold"].mean(),
                }

                # Append to CSV
                row_df = pd.DataFrame([row])
                if RESULTS_CV_CSV.exists():
                    row_df.to_csv(RESULTS_CV_CSV, mode="a",
                                  header=False, index=False)
                else:
                    row_df.to_csv(RESULTS_CV_CSV, index=False)

                done.add(key)
                print(f"F1opt={row['f1_macro_opt_mean']:.3f}+-"
                      f"{row['f1_macro_opt_std']:.3f} "
                      f"AUC={row['roc_auc_mean']:.3f}+-"
                      f"{row['roc_auc_std']:.3f} "
                      f"thr={row['threshold_mean']:.2f}")

    print(f"\n[ROBUST CV] Complete. Results in {RESULTS_CV_CSV}")


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run({"strict": cohorts["strict"], "competing": cohorts["competing"]})

"""
src/classification/tuning.py
─────────────────────────────
Fine-tuning of leader models.
Pipeline.md section 6 -- fine-tuning.

RandomizedSearchCV(scoring='f1_macro', cv=4, n_iter~15) + threshold + SVMSMOTE
on at least {CV17, CV17_THY26, CV17_CONT}.
Also produces the AUC-by-feature-set comparison plot.
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.base import clone

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, RESULTS_TUNE_CSV, RESULTS_CV_CSV,
    RANDOM_STATE, FIGURES_DIR, THRESHOLD_GRID, FEATURE_SET_ORDER,
)
from src.features.feature_engineering import extract_Xy, add_derived_features
from src.classification.robust_cv import _optimize_threshold


# ─── Hyperparameter search spaces ─────────────────────────────────────

def _get_param_distributions(model_name: str) -> dict:
    """Hyperparameter distributions for RandomizedSearchCV."""
    if model_name == "LogisticRegression":
        return {
            "clf__C": [0.01, 0.1, 0.5, 1, 5, 10],
            "clf__penalty": ["l2"],
        }
    elif model_name == "RandomForest":
        return {
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__max_depth": [5, 10, 15, 20, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5],
        }
    elif model_name == "HistGradientBoosting":
        return {
            "clf__max_iter": [100, 200, 300],
            "clf__max_depth": [3, 5, 7, None],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__min_samples_leaf": [10, 20, 40],
        }
    elif model_name == "XGBoost":
        return {
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__max_depth": [3, 5, 7, 9],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
    return {}


def _load_existing() -> set:
    if RESULTS_TUNE_CSV.exists():
        df = pd.read_csv(RESULTS_TUNE_CSV)
        required_cols = {
            "cv_f1_macro_threshold", "cv_auc", "cv_threshold_mean",
        }
        if not required_cols.issubset(df.columns):
            print("[TUNING] Existing results_tune.csv has an old schema; "
                  "recomputing tuning results.")
            RESULTS_TUNE_CSV.unlink()
            return set()
        return {(r["cohort"], r["feature_set"], r["model"])
                for _, r in df.iterrows()}
    return set()


def _append_result_row(row: dict):
    """Append a tuning row while preserving/expanding the CSV schema."""
    row_df = pd.DataFrame([row])
    if RESULTS_TUNE_CSV.exists():
        existing = pd.read_csv(RESULTS_TUNE_CSV)
        combined = pd.concat([existing, row_df], ignore_index=True,
                             sort=False)
        combined.to_csv(RESULTS_TUNE_CSV, index=False)
    else:
        row_df.to_csv(RESULTS_TUNE_CSV, index=False)


def _cross_validated_threshold_metrics(best_estimator, X, y,
                                       n_splits: int = 4) -> dict:
    """
    Estimate threshold-tuned performance for the selected hyperparameters.

    Threshold selection is done on each training fold and evaluated on the
    held-out fold, avoiding the in-sample optimism of train_f1_opt_thr.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    rows = []

    for train_idx, val_idx in skf.split(X, y):
        model = clone(best_estimator)
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        threshold, _ = _optimize_threshold(y_train, y_train_proba)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= threshold).astype(int)
        y_val_pred_default = (y_val_proba >= 0.5).astype(int)

        rows.append({
            "threshold": threshold,
            "f1_macro_threshold": f1_score(
                y_val, y_val_pred, average="macro"),
            "f1_macro_default": f1_score(
                y_val, y_val_pred_default, average="macro"),
            "roc_auc": roc_auc_score(y_val, y_val_proba),
        })

    metrics = pd.DataFrame(rows)
    return {
        "cv_f1_macro_threshold": metrics["f1_macro_threshold"].mean(),
        "cv_f1_macro_threshold_std": metrics["f1_macro_threshold"].std(),
        "cv_f1_macro_default": metrics["f1_macro_default"].mean(),
        "cv_f1_macro_default_std": metrics["f1_macro_default"].std(),
        "cv_auc": metrics["roc_auc"].mean(),
        "cv_auc_std": metrics["roc_auc"].std(),
        "cv_threshold_mean": metrics["threshold"].mean(),
    }


def run(cohorts: dict, force: bool = False):
    """Fine-tuning with RandomizedSearchCV."""
    from src.classification.screening import _build_pipeline

    print("\n" + "=" * 60)
    print("[TUNING] Fine-tuning of leader models")
    print("=" * 60)

    if force and RESULTS_TUNE_CSV.exists():
        RESULTS_TUNE_CSV.unlink()

    done = set() if force else _load_existing()

    tune_fs = list(FEATURE_SETS.keys())
    tune_models = ["LogisticRegression", "RandomForest",
                   "HistGradientBoosting", "XGBoost"]
    cohort_names = ["strict", "competing"]

    for coh_name in cohort_names:
        df = cohorts[coh_name].copy()
        df = add_derived_features(df)

        for fs_name in tune_fs:
            try:
                X, y = extract_Xy(df, fs_name)
            except Exception as e:
                print(f"  [SKIP] {coh_name}/{fs_name}: {e}")
                continue

            if len(X) == 0:
                continue

            for model_name in tune_models:
                key = (coh_name, fs_name, model_name)
                if key in done:
                    continue

                param_dist = _get_param_distributions(model_name)
                if not param_dist:
                    continue

                print(f"  Tuning {coh_name}/{fs_name}/{model_name} ...",
                      end=" ", flush=True)

                try:
                    pipe = _build_pipeline(model_name, "SVMSMOTE")

                    search = RandomizedSearchCV(
                        pipe,
                        param_distributions=param_dist,
                        n_iter=15,
                        scoring="f1_macro",
                        cv=StratifiedKFold(n_splits=4, shuffle=True,
                                            random_state=RANDOM_STATE),
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(X, y)

                    # In-sample audit metrics only; use the CV columns below
                    # for model comparison.
                    y_proba = search.predict_proba(X)[:, 1]
                    best_thr, best_f1_train = _optimize_threshold(y, y_proba)

                    auc = roc_auc_score(y, y_proba)
                    cv_threshold_metrics = _cross_validated_threshold_metrics(
                        search.best_estimator_, X, y)

                    row = {
                        "cohort": coh_name,
                        "feature_set": fs_name,
                        "model": model_name,
                        "sampler": "SVMSMOTE",
                        "best_params": str(search.best_params_),
                        "cv_f1_macro": search.best_score_,
                        **cv_threshold_metrics,
                        "train_f1_opt_thr": best_f1_train,
                        "optimal_threshold": best_thr,
                        "train_auc": auc,
                    }

                    _append_result_row(row)

                    done.add(key)
                    print(f"CV-F1m={search.best_score_:.3f} "
                          f"CV-F1thr="
                          f"{cv_threshold_metrics['cv_f1_macro_threshold']:.3f} "
                          f"CV-AUC={cv_threshold_metrics['cv_auc']:.3f} "
                          f"thr={cv_threshold_metrics['cv_threshold_mean']:.2f}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

    print(f"\n[TUNING] Complete. Results in {RESULTS_TUNE_CSV}")


# ─── AUC by feature set plot ─────────────────────────────────────────

def plot_auc_by_feature_set(save: bool = True):
    """
    Plot AUC (5-fold CV) per feature set, one line per model,
    two panels (strict / competing).
    """
    if not RESULTS_CV_CSV.exists():
        print("[PLOT] results_cv.csv not found, skipping.")
        return

    df = pd.read_csv(RESULTS_CV_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cohort_names = ["strict", "competing"]

    for ax, coh in zip(axes, cohort_names):
        sub = df[df["cohort"] == coh]
        if sub.empty:
            ax.set_title(f"{coh} (no data)")
            continue

        for model in sub["model"].unique():
            m_data = sub[sub["model"] == model].copy()
            m_data["feature_set"] = pd.Categorical(
                m_data["feature_set"], categories=FEATURE_SET_ORDER,
                ordered=True)
            m_data = m_data.sort_values("feature_set")
            ax.errorbar(
                m_data["feature_set"],
                m_data["roc_auc_mean"],
                yerr=m_data["roc_auc_std"],
                marker="o", capsize=3, label=model,
            )

        ax.set_xlabel("Feature Set", fontsize=11)
        ax.set_ylabel("ROC-AUC (5-fold CV)", fontsize=11)
        ax.set_title(f"Cohort: {coh}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    fig.suptitle("AUC by Feature Set and Model",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "auc_by_feature_set.png",
                    bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] AUC by feature set saved.")


def plot_f1_by_feature_set(save: bool = True):
    """
    Plot F1-macro (primary metric) per feature set, one line per model,
    two panels (strict / competing).
    """
    if not RESULTS_CV_CSV.exists():
        print("[PLOT] results_cv.csv not found, skipping.")
        return

    df = pd.read_csv(RESULTS_CV_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cohort_names = ["strict", "competing"]

    for ax, coh in zip(axes, cohort_names):
        sub = df[df["cohort"] == coh]
        if sub.empty:
            ax.set_title(f"{coh} (no data)")
            continue

        for model in sub["model"].unique():
            m_data = sub[sub["model"] == model].copy()
            m_data["feature_set"] = pd.Categorical(
                m_data["feature_set"], categories=FEATURE_SET_ORDER,
                ordered=True)
            m_data = m_data.sort_values("feature_set")
            ax.errorbar(
                m_data["feature_set"],
                m_data["f1_macro_opt_mean"],
                yerr=m_data["f1_macro_opt_std"],
                marker="o", capsize=3, label=model,
            )

        ax.set_xlabel("Feature Set", fontsize=11)
        ax.set_ylabel("F1-macro optimized threshold (5-fold CV)", fontsize=11)
        ax.set_title(f"Cohort: {coh}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    fig.suptitle("F1-macro by Feature Set and Model",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "f1_by_feature_set.png",
                    bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] F1-macro by feature set saved.")


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run({"strict": cohorts["strict"], "competing": cohorts["competing"]})
    plot_f1_by_feature_set()
    plot_auc_by_feature_set()

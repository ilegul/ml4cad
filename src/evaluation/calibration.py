"""
src/evaluation/calibration.py
--------------------------------
Final model calibration.

Runs calibration on classification cohorts only (strict and competing) and,
by default, across all feature sets. For each cohort/feature-set pair the
best tuned model is used when available; otherwise the best robust-CV model
is used as a fallback.
"""

import sys, ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, FIGURES_DIR, RANDOM_STATE, TEST_SIZE,
    RESULTS_TUNE_CSV, RESULTS_CV_CSV,
)
from src.features.feature_engineering import extract_Xy, add_derived_features
from src.classification.screening import _build_pipeline


def _parse_best_params(value) -> dict:
    """Parse the best_params string saved by RandomizedSearchCV."""
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_tuned_pipeline(model_name: str, sampler_name: str,
                          best_params: dict = None):
    """Build the pipeline and apply tuned hyperparameters when available."""
    pipe = _build_pipeline(model_name, sampler_name)
    if best_params:
        pipe.set_params(**best_params)
    return pipe


def _select_final_config(cohort_name: str, feature_set_name: str,
                         default_model: str, default_sampler: str) -> dict:
    """
    Select the best available config for one cohort/feature-set pair.

    Priority:
    1. tuning results for the exact cohort + feature set;
    2. robust-CV results for the exact cohort + feature set;
    3. explicit function defaults.
    """
    if RESULTS_TUNE_CSV.exists():
        tune = pd.read_csv(RESULTS_TUNE_CSV)
        sub = tune[
            (tune["cohort"] == cohort_name) &
            (tune["feature_set"] == feature_set_name)
        ].copy()
        if not sub.empty:
            score_col = (
                "cv_f1_macro_threshold"
                if "cv_f1_macro_threshold" in sub.columns
                else "cv_f1_macro"
            )
            best = sub.sort_values(score_col, ascending=False).iloc[0]
            return {
                "model": best["model"],
                "sampler": best.get("sampler", default_sampler),
                "best_params": _parse_best_params(
                    best.get("best_params", "{}")),
                "selected_threshold": best.get(
                    "cv_threshold_mean",
                    best.get("optimal_threshold", 0.5)),
                "selection_source": "tuning",
                "selection_score": best.get(score_col, np.nan),
            }

    if RESULTS_CV_CSV.exists():
        cv = pd.read_csv(RESULTS_CV_CSV)
        sub = cv[
            (cv["cohort"] == cohort_name) &
            (cv["feature_set"] == feature_set_name)
        ].copy()
        if not sub.empty:
            best = sub.sort_values(
                "f1_macro_opt_mean", ascending=False).iloc[0]
            return {
                "model": best["model"],
                "sampler": best.get("sampler", default_sampler),
                "best_params": {},
                "selected_threshold": best.get("threshold_mean", 0.5),
                "selection_source": "robust_cv",
                "selection_score": best.get("f1_macro_opt_mean", np.nan),
            }

    return {
        "model": default_model,
        "sampler": default_sampler,
        "best_params": {},
        "selected_threshold": 0.5,
        "selection_source": "default",
        "selection_score": np.nan,
    }


def _calibrate_one(df: pd.DataFrame, cohort_name: str,
                   feature_set_name: str, model_name: str,
                   sampler_name: str, save: bool = True) -> list:
    """Calibrate one cohort/feature-set pair and return result rows."""
    config = _select_final_config(
        cohort_name, feature_set_name, model_name, sampler_name)
    selected_model = config["model"]
    selected_sampler = config["sampler"]
    best_params = config["best_params"]
    selected_threshold = config["selected_threshold"]
    if not np.isfinite(selected_threshold):
        selected_threshold = 0.5

    print(f"  {cohort_name}/{feature_set_name}: "
          f"{selected_model}/{selected_sampler} "
          f"({config['selection_source']})")

    X, y = extract_Xy(df, feature_set_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    model_base = _build_tuned_pipeline(
        selected_model, selected_sampler, best_params)
    model_base.fit(X_train, y_train)
    y_proba_base = model_base.predict_proba(X_test)[:, 1]

    methods = {
        "base": {
            "probas": y_proba_base,
            "predictor": model_base,
        }
    }

    for method in ["isotonic", "sigmoid"]:
        try:
            cal = CalibratedClassifierCV(
                estimator=_build_tuned_pipeline(
                    selected_model, selected_sampler, best_params),
                method=method, cv=5)
            cal.fit(X_train, y_train)
            methods[method] = {
                "probas": cal.predict_proba(X_test)[:, 1],
                "predictor": cal,
            }
        except Exception as e:
            print(f"    [WARN] {method} calibration failed: {e}")

    rows = []
    for method, data in methods.items():
        probas = data["probas"]
        predictor = data["predictor"]
        y_pred_threshold = (probas >= selected_threshold).astype(int)
        rows.append({
            "cohort": cohort_name,
            "feature_set": feature_set_name,
            "method": method,
            "brier_score": brier_score_loss(y_test, probas),
            "roc_auc": roc_auc_score(y_test, probas),
            "f1_macro_default": f1_score(
                y_test, predictor.predict(X_test), average="macro"),
            "selected_threshold": selected_threshold,
            "f1_macro_selected_threshold": f1_score(
                y_test, y_pred_threshold, average="macro"),
            "model": selected_model,
            "sampler": selected_sampler,
            "best_params": str(best_params),
            "selection_source": config["selection_source"],
            "selection_score": config["selection_score"],
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

    if save:
        fig, ax = plt.subplots(figsize=(7, 6))
        for method, data in methods.items():
            fraction_pos, mean_pred = calibration_curve(
                y_test, data["probas"], n_bins=10, strategy="uniform")
            brier = brier_score_loss(y_test, data["probas"])
            ax.plot(mean_pred, fraction_pos, "o-",
                    label=f"{method} (Brier={brier:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(
            f"Calibration - {cohort_name}/{feature_set_name}\n"
            f"{selected_model} + {selected_sampler}",
            fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR /
            f"calibration_curve_{cohort_name}_{feature_set_name}.png",
            bbox_inches="tight")
        plt.close(fig)

    return rows


def run(cohorts,
        feature_set_name: str = None,
        model_name: str = "XGBoost",
        sampler_name: str = "SMOTE",
        save: bool = True):
    """Run calibration on classification cohorts and feature sets."""
    print("\n" + "=" * 60)
    print("[CALIBRATION] Final model calibration")
    print("=" * 60)

    if isinstance(cohorts, dict):
        cohort_items = [(name, df) for name, df in cohorts.items()
                        if name in ["strict", "competing"]]
    else:
        cohort_items = [("strict", cohorts)]

    feature_sets = (
        [feature_set_name] if feature_set_name is not None
        else list(FEATURE_SETS.keys())
    )

    all_rows = []
    for cohort_name, cohort_df in cohort_items:
        df = add_derived_features(cohort_df.copy())
        for fs_name in feature_sets:
            try:
                all_rows.extend(_calibrate_one(
                    df, cohort_name, fs_name, model_name, sampler_name, save))
            except Exception as e:
                print(f"  [ERROR] {cohort_name}/{fs_name}: {e}")

    results_df = pd.DataFrame(all_rows)
    if not results_df.empty:
        out = FIGURES_DIR.parent / "calibration_results.csv"
        results_df.to_csv(out, index=False)
        print(f"\n[CALIBRATION] Results saved to {out}")

    return results_df


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run({"strict": cohorts["strict"], "competing": cohorts["competing"]})

"""
src/classification/screening.py
────────────────────────────────
Classification screening: full grid
  2 cohorts x 8 feature sets x 8 models x 5 samplers
Pipeline.md section 6 -- screening.

Resumable: skips combinations already computed in results_clf.csv.
"""

import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, classification_report,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, RESULTS_CLF_CSV, RANDOM_STATE, TEST_SIZE,
    MODEL_NAMES, SAMPLER_NAMES,
)
from src.features.feature_engineering import extract_Xy, add_derived_features


# ─── Model factory ────────────────────────────────────────────────────

def _get_model(name: str):
    """Create a classifier instance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, AdaBoostClassifier,
        HistGradientBoostingClassifier,
    )
    from sklearn.neural_network import MLPClassifier

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"),
        "SVC": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
            max_iter=5000),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=RANDOM_STATE),
        "XGBoost": None,  # lazy import
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=RANDOM_STATE, early_stopping=True),
    }

    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
            eval_metric="logloss", n_jobs=1, verbosity=0)

    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name]


def _get_sampler(name: str):
    """Create a sampler instance."""
    if name == "none":
        return None
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE

    samplers = {
        "RandomUnderSampler": RandomUnderSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=RANDOM_STATE),
        "SVMSMOTE": SVMSMOTE(random_state=RANDOM_STATE),
    }
    return samplers[name]


def _build_pipeline(model_name: str, sampler_name: str):
    """Build an imblearn pipeline with scaler, sampler, classifier."""
    from imblearn.pipeline import Pipeline as ImbPipeline

    steps = [("scaler", StandardScaler())]
    sampler = _get_sampler(sampler_name)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", _get_model(model_name)))
    return ImbPipeline(steps)


def _load_existing_results() -> set:
    """Load already-computed combinations."""
    if RESULTS_CLF_CSV.exists():
        df = pd.read_csv(RESULTS_CLF_CSV)
        done = set()
        for _, row in df.iterrows():
            key = (row["cohort"], row["feature_set"],
                   row["model"], row["sampler"])
            done.add(key)
        return done
    return set()


def _evaluate(pipe, X_train, y_train, X_test, y_test) -> dict:
    """Train the pipeline and compute metrics."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Probabilities for AUC (if available)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)
    except Exception:
        roc_auc = np.nan
        pr_auc  = np.nan

    return {
        "f1_macro":     f1_score(y_test, y_pred, average="macro"),
        "f1_class0":    f1_score(y_test, y_pred, pos_label=0),
        "f1_class1":    f1_score(y_test, y_pred, pos_label=1),
        "roc_auc":      roc_auc,
        "pr_auc":       pr_auc,
        "precision_0":  precision_score(y_test, y_pred, pos_label=0,
                                        zero_division=0),
        "precision_1":  precision_score(y_test, y_pred, pos_label=1,
                                        zero_division=0),
        "recall_0":     recall_score(y_test, y_pred, pos_label=0,
                                     zero_division=0),
        "recall_1":     recall_score(y_test, y_pred, pos_label=1,
                                     zero_division=0),
    }


# ─── Entry point ────────────────────────────────────────────────────────

def run(cohorts: dict, force: bool = False):
    """
    Run the full screening.
    cohorts: {"strict": df, "competing": df}
    """
    print("\n" + "=" * 60)
    print("[SCREENING] Classification -- full grid")
    print("=" * 60)

    if force and RESULTS_CLF_CSV.exists():
        RESULTS_CLF_CSV.unlink()

    done = set() if force else _load_existing_results()
    cohort_names = ["strict", "competing"]
    fs_names = list(FEATURE_SETS.keys())

    total = len(cohort_names) * len(fs_names) * len(MODEL_NAMES) * len(SAMPLER_NAMES)
    done_count = len(done)
    print(f"  Total combinations: {total}, already computed: {done_count}")

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

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y,
                random_state=RANDOM_STATE)

            for model_name in MODEL_NAMES:
                for sampler_name in SAMPLER_NAMES:
                    key = (coh_name, fs_name, model_name, sampler_name)
                    if key in done:
                        continue

                    try:
                        t0 = time.time()
                        pipe = _build_pipeline(model_name, sampler_name)
                        metrics = _evaluate(pipe, X_train, y_train,
                                            X_test, y_test)
                        elapsed = time.time() - t0

                        row = {
                            "cohort": coh_name,
                            "feature_set": fs_name,
                            "model": model_name,
                            "sampler": sampler_name,
                            "n_train": len(X_train),
                            "n_test": len(X_test),
                            "elapsed_s": round(elapsed, 2),
                            **metrics,
                        }

                        # Append to CSV
                        row_df = pd.DataFrame([row])
                        if RESULTS_CLF_CSV.exists():
                            row_df.to_csv(RESULTS_CLF_CSV, mode="a",
                                          header=False, index=False)
                        else:
                            row_df.to_csv(RESULTS_CLF_CSV, index=False)

                        done.add(key)
                        done_count += 1
                        print(f"  [{done_count}/{total}] {coh_name}/{fs_name}/"
                              f"{model_name}/{sampler_name} -> "
                              f"F1m={metrics['f1_macro']:.3f} "
                              f"AUC={metrics['roc_auc']:.3f} "
                              f"({elapsed:.1f}s)")

                    except Exception as e:
                        print(f"  [ERROR] {key}: {e}")
                        continue

    print(f"\n[SCREENING] Complete. Results in {RESULTS_CLF_CSV}")


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run({"strict": cohorts["strict"], "competing": cohorts["competing"]})

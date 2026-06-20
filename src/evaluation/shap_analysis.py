"""
src/evaluation/shap_analysis.py
────────────────────────────────
SHAP analysis on the best model.
Pipeline.md section 8.

- TreeExplainer for tree-based models
- LinearExplainer for LR
- Beeswarm, bar plot, dependence plot
- Cardiac vs thyroid importance quantification
- Comparison with permutation importance
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, FIGURES_DIR, SHAP_CSV, SHAP_PERM_CORR_CSV,
    RANDOM_STATE, TEST_SIZE, CARDIO_17, THYROID_RAW9,
)
from src.features.feature_engineering import (
    extract_Xy, add_derived_features,
)
from src.classification.screening import _get_model, _get_sampler


def _train_model_for_shap(df: pd.DataFrame, feature_set_name: str,
                           model_name: str = "XGBoost",
                           sampler_name: str = "SMOTE"):
    """
    Train a model on scaled data (without the sampler in the pipeline,
    since SHAP needs the explainer applied to the classifier on scaled data).
    """
    X, y = extract_Xy(df, feature_set_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Scale
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
    X_test_sc = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns,
                              index=X_test.index)

    # Apply sampler to training set
    sampler = _get_sampler(sampler_name)
    if sampler is not None:
        X_train_res, y_train_res = sampler.fit_resample(X_train_sc, y_train)
    else:
        X_train_res, y_train_res = X_train_sc, y_train

    # Train the model
    model = _get_model(model_name)
    model.fit(X_train_res, y_train_res)

    return model, X_train_sc, X_test_sc, y_train, y_test, scaler


def _get_explainer(model, X_background, model_name: str):
    """Create the appropriate SHAP explainer."""
    if model_name in ["XGBoost", "RandomForest", "HistGradientBoosting",
                       "AdaBoost"]:
        return shap.TreeExplainer(model)
    elif model_name == "LogisticRegression":
        return shap.LinearExplainer(model, X_background)
    else:
        # Fallback: KernelExplainer on a sample
        background = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model.predict_proba, background)


def run_shap_for_feature_set(df: pd.DataFrame, feature_set_name: str,
                              model_name: str = "XGBoost",
                              cohort_name: str = "strict",
                              save: bool = True) -> dict:
    """Run SHAP analysis for a single feature set."""
    print(f"\n  [SHAP] {cohort_name}/{feature_set_name}/{model_name}")

    model, X_train, X_test, y_train, y_test, scaler = \
        _train_model_for_shap(df, feature_set_name, model_name)

    explainer = _get_explainer(model, X_train, model_name)

    # Compute SHAP values
    shap_values = explainer(X_test)

    # For binary models, take the positive class
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # Beeswarm
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm -- {feature_set_name} ({model_name})",
              fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(
            FIGURES_DIR / f"shap_beeswarm_{cohort_name}_{feature_set_name}.png",
            bbox_inches="tight")
    plt.close(fig)

    # Bar plot
    fig = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.title(f"SHAP Importance -- {feature_set_name} ({model_name})",
              fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(
            FIGURES_DIR / f"shap_bar_{cohort_name}_{feature_set_name}.png",
            bbox_inches="tight")
    plt.close(fig)

    # Mean importance
    feat_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # Cardiac vs thyroid quantification
    thyroid_feats = set(THYROID_RAW9 + ["fT3_fT4_ratio", "Thyroid_abnormal",
                                         "thyroid_ord"])
    cardio_feats = set(CARDIO_17)

    fi = feat_importance.copy()
    fi["group"] = fi["feature"].apply(
        lambda f: "thyroid" if f in thyroid_feats else
                  ("cardio" if f in cardio_feats else "other"))

    total_shap = fi["mean_abs_shap"].sum()
    thyroid_shap = fi.loc[fi["group"] == "thyroid", "mean_abs_shap"].sum()
    cardio_shap = fi.loc[fi["group"] == "cardio", "mean_abs_shap"].sum()

    pct_thyroid = 100 * thyroid_shap / total_shap if total_shap > 0 else 0
    pct_cardio = 100 * cardio_shap / total_shap if total_shap > 0 else 0

    print(f"    Cardiac importance: {cardio_shap:.4f} ({pct_cardio:.1f}%)")
    print(f"    Thyroid importance: {thyroid_shap:.4f} ({pct_thyroid:.1f}%)")

    fi["feature_set"] = feature_set_name
    fi["cohort"] = cohort_name
    fi["model"] = model_name

    # Dependence plot for top features
    top_feats = feat_importance["feature"].head(3).tolist()
    # Add best thyroid feature
    thyroid_ranks = fi[fi["group"] == "thyroid"]
    if not thyroid_ranks.empty:
        best_thyroid = thyroid_ranks.iloc[0]["feature"]
        if best_thyroid not in top_feats:
            top_feats.append(best_thyroid)

    for feat in top_feats:
        if feat in X_test.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_idx = list(X_test.columns).index(feat)
            shap.plots.scatter(
                shap_values[:, feat_idx], color=shap_values,
                ax=ax, show=False)
            ax.set_title(f"SHAP Dependence -- {feat} ({feature_set_name})",
                         fontweight="bold")
            fig.tight_layout()
            if save:
                fig.savefig(FIGURES_DIR /
                            f"shap_dep_{cohort_name}_{feature_set_name}_{feat}.png",
                            bbox_inches="tight")
            plt.close(fig)

    return {
        "importance": fi,
        "pct_thyroid": pct_thyroid,
        "pct_cardio": pct_cardio,
    }


def run_permutation_importance(df: pd.DataFrame, feature_set_name: str,
                                model_name: str = "XGBoost",
                                cohort_name: str = "strict",
                                save: bool = True) -> pd.DataFrame:
    """Permutation importance as a robustness check vs SHAP."""
    print(f"  [PERM] {cohort_name}/{feature_set_name}/{model_name}")

    model, X_train, X_test, y_train, y_test, scaler = \
        _train_model_for_shap(df, feature_set_name, model_name)

    pi = permutation_importance(model, X_test, y_test,
                                 n_repeats=10, random_state=RANDOM_STATE,
                                 scoring="f1_macro", n_jobs=1)

    pi_df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance_mean": pi.importances_mean,
        "perm_importance_std": pi.importances_std,
        "feature_set": feature_set_name,
        "cohort": cohort_name,
        "model": model_name,
    }).sort_values("perm_importance_mean", ascending=False)

    return pi_df


def _correlation_summary(shap_df: pd.DataFrame,
                         perm_df: pd.DataFrame) -> pd.DataFrame:
    """Compare SHAP and permutation importance for each feature set."""
    rows = []
    keys = sorted(
        set(map(tuple, shap_df[["cohort", "feature_set"]].drop_duplicates()
                .values)) &
        set(map(tuple, perm_df[["cohort", "feature_set"]].drop_duplicates()
                .values))
    )
    for cohort_name, fs_name in keys:
        shap_fs = shap_df[
            (shap_df["cohort"] == cohort_name) &
            (shap_df["feature_set"] == fs_name)]
        perm_fs = perm_df[
            (perm_df["cohort"] == cohort_name) &
            (perm_df["feature_set"] == fs_name)]
        merged = shap_fs[["feature", "mean_abs_shap"]].merge(
            perm_fs[["feature", "perm_importance_mean"]],
            on="feature")
        if len(merged) < 2:
            continue

        pearson = merged["mean_abs_shap"].corr(
            merged["perm_importance_mean"], method="pearson")
        # Rank-Pearson avoids relying on scipy for Spearman.
        spearman = merged["mean_abs_shap"].rank().corr(
            merged["perm_importance_mean"].rank(), method="pearson")
        rows.append({
            "cohort": cohort_name,
            "feature_set": fs_name,
            "n_features": len(merged),
            "pearson_corr": pearson,
            "spearman_corr": spearman,
        })
    return pd.DataFrame(rows)


# ─── Entry point ────────────────────────────────────────────────────────

def run(cohorts, force: bool = False):
    """Run the complete SHAP analysis."""
    print("\n" + "=" * 60)
    print("[SHAP] SHAP Analysis")
    print("=" * 60)

    if isinstance(cohorts, dict):
        cohort_items = [(name, df) for name, df in cohorts.items()
                        if name in ["strict", "competing"]]
    else:
        cohort_items = [("strict", cohorts)]

    # Tree model for exact TreeExplainer. XGBoost 3.x serializes base_score
    # as a vector that current SHAP versions may fail to parse, so RandomForest
    # is the robust tree-based explainer for this project.
    all_importance = []
    all_perm = []

    fs_names = list(FEATURE_SETS.keys())
    shap_model = "RandomForest"

    for cohort_name, cohort_df in cohort_items:
        df = add_derived_features(cohort_df.copy())
        for fs_name in fs_names:
            try:
                result = run_shap_for_feature_set(
                    df, fs_name, shap_model, cohort_name=cohort_name)
                all_importance.append(result["importance"])

                perm = run_permutation_importance(
                    df, fs_name, shap_model, cohort_name=cohort_name)
                all_perm.append(perm)
            except Exception as e:
                print(f"  [ERROR] {cohort_name}/{fs_name}/{shap_model}: {e}")

    # Save CSV
    if all_importance:
        imp_df = pd.concat(all_importance, ignore_index=True)
        imp_df.to_csv(SHAP_CSV, index=False)
        print(f"\n[SHAP] Importance saved to {SHAP_CSV}")

    if all_perm:
        perm_df = pd.concat(all_perm, ignore_index=True)
        perm_df.to_csv(SHAP_CSV.parent / "permutation_importance.csv",
                        index=False)

    # Comparison SHAP vs Permutation importance
    if all_importance and all_perm:
        corr_df = _correlation_summary(imp_df, perm_df)
        corr_df.to_csv(SHAP_PERM_CORR_CSV, index=False)
        print("\n[SHAP] SHAP vs Permutation Importance comparison:")
        for _, row in corr_df.iterrows():
            print(f"  {row['cohort']}/{row['feature_set']}: "
                  f"Pearson={row['pearson_corr']:.3f}, "
                  f"Spearman={row['spearman_corr']:.3f}")
        print(f"  Correlations saved to {SHAP_PERM_CORR_CSV}")

    print("\n[SHAP] Complete.")


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run({"strict": cohorts["strict"], "competing": cohorts["competing"]})

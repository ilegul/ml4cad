"""
src/survival/survival_cohort.py
────────────────────────────────
Survival analysis restricted to classification cohorts.
Pipeline.md section 7-bis.

- Administrative censoring at 7 years
- c-index 5-fold CV for Cox and RSF across all feature sets
- Caveat: estimates are NOT unbiased absolute risks
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, RESULTS_SURV_CSV, RANDOM_STATE, CV_FOLDS,
    HORIZON_YEARS, RSF_PARAMS, CONTINUOUS_FEATURES,
    COX_COHORT_MULTIVARIATE_CSV, COX_COHORT_SCHOENFELD_CSV,
)
from src.features.feature_engineering import (
    add_derived_features, get_all_feature_set_names,
)


def _prepare_cohort_survival(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for survival on the classification cohort.
    Administrative censoring at 7 years:
        t = min(time_years, 7)
        event = (event_cvd & time_years <= 7)
    """
    df = df.copy()
    df["surv_time"] = df["time_years"].clip(upper=HORIZON_YEARS)
    df["surv_event"] = ((df["event_cvd"] == 1) &
                        (df["time_years"] <= HORIZON_YEARS)).astype(int)
    return df


def _load_existing() -> set:
    if RESULTS_SURV_CSV.exists():
        existing = pd.read_csv(RESULTS_SURV_CSV)
        return {(r["cohort"], r["feature_set"], r["model"])
                for _, r in existing.iterrows()
                if r.get("cohort", "") in ["strict", "competing"]}
    return set()


def _cox_covariates(df: pd.DataFrame, feature_set_name: str) -> list:
    """
    Covariates for interpretable cohort-level Cox models.

    For feature sets with thyroid-state dummy variables, Euthyroid is used as
    the reference category and is dropped before fitting.
    """
    requested = [c for c in FEATURE_SETS.get(feature_set_name, [])
                 if c in df.columns]
    thyroid_state_cols = {
        "Euthyroid", "SCH", "SCT", "Low_T3",
        "Hypothyroid", "Hyperthyroid",
    }
    if "Euthyroid" in requested and len(thyroid_state_cols & set(requested)) > 1:
        requested = [c for c in requested if c != "Euthyroid"]
    return requested


def _fit_cox_interpretation(df: pd.DataFrame,
                            cohort_name: str,
                            feature_set_name: str) -> tuple:
    """
    Fit an interpretable Cox model for a classification cohort.

    Returns (cox_summary, schoenfeld_summary). The model uses administrative
    7-year censoring columns created by _prepare_cohort_survival.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except Exception as e:
        print(f"  [COX {cohort_name}] lifelines unavailable: {e}")
        return None, None

    features = _cox_covariates(df, feature_set_name)
    cols = features + ["surv_time", "surv_event"]
    subset = df[cols].dropna().copy()

    if subset["surv_event"].sum() == 0 or len(subset) < 50:
        print(f"  [COX {cohort_name}] insufficient data.")
        return None, None

    cont_cols = [c for c in CONTINUOUS_FEATURES if c in subset.columns]
    if cont_cols:
        scaler = StandardScaler()
        subset[cont_cols] = scaler.fit_transform(subset[cont_cols])

    penalizer_used = 0.0
    cph = CoxPHFitter()
    try:
        cph.fit(subset, duration_col="surv_time", event_col="surv_event")
    except Exception as e:
        penalizer_used = 0.01
        print(f"  [COX {cohort_name}] unpenalized fit failed ({e}); "
              f"retrying with penalizer={penalizer_used}.")
        cph = CoxPHFitter(penalizer=penalizer_used)
        cph.fit(subset, duration_col="surv_time", event_col="surv_event")

    summary = cph.summary.reset_index().rename(columns={"index": "covariate"})
    summary.insert(0, "cohort", cohort_name)
    summary.insert(1, "feature_set", feature_set_name)
    summary.insert(2, "n", len(subset))
    summary.insert(3, "events", int(subset["surv_event"].sum()))
    summary.insert(4, "penalizer", penalizer_used)

    try:
        ph = proportional_hazard_test(
            cph, subset, time_transform="rank")
        ph_summary = ph.summary.reset_index().rename(
            columns={"index": "covariate"})
        ph_summary.insert(0, "cohort", cohort_name)
        ph_summary.insert(1, "feature_set", feature_set_name)
    except Exception as e:
        print(f"  [COX {cohort_name}] Schoenfeld test failed: {e}")
        ph_summary = None

    print(f"  [COX {cohort_name}/{feature_set_name}] fitted on "
          f"n={len(subset)}, events={int(subset['surv_event'].sum())}.")
    return summary, ph_summary


def run(cohorts: dict, force: bool = False):
    """
    Survival analysis on classification cohorts (strict + competing).
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored

    print("\n" + "=" * 60)
    print("[SURV-COHORT] Survival on classification cohorts")
    print("=" * 60)
    print("  CAVEAT: this cohort removes patients censored before 7 years.")
    print("  Estimates are NOT unbiased absolute risks.")
    print("  This variant serves comparability with classification.\n")

    if force and RESULTS_SURV_CSV.exists():
        existing = pd.read_csv(RESULTS_SURV_CSV)
        existing = existing[~existing["cohort"].isin(["strict", "competing"])]
        if existing.empty:
            RESULTS_SURV_CSV.unlink()
        else:
            existing.to_csv(RESULTS_SURV_CSV, index=False)

    done = set() if force else _load_existing()
    cox_summaries = []
    schoenfeld_summaries = []

    models_dict = {
        "CoxPH": lambda: CoxPHSurvivalAnalysis(),
        "RSF": lambda: RandomSurvivalForest(**RSF_PARAMS),
    }

    fs_names = get_all_feature_set_names()

    for coh_name in ["strict", "competing"]:
        if coh_name not in cohorts:
            continue

        df = cohorts[coh_name].copy()
        df = add_derived_features(df)
        df = _prepare_cohort_survival(df)

        for fs_name in fs_names:
            cox_summary, schoenfeld_summary = _fit_cox_interpretation(
                df, coh_name, fs_name)
            if cox_summary is not None:
                cox_summaries.append(cox_summary)
            if schoenfeld_summary is not None:
                schoenfeld_summaries.append(schoenfeld_summary)

            # Prepare features
            features = FEATURE_SETS.get(fs_name, [])
            available = [f for f in features if f in df.columns]
            cols = available + ["surv_time", "surv_event"]
            subset = df[cols].dropna()

            if len(subset) < 50:
                continue

            X = subset[available]
            events = subset["surv_event"].astype(int)

            y_surv = np.array(
                [(bool(e), t) for e, t in
                 zip(subset["surv_event"], subset["surv_time"])],
                dtype=[("event", bool), ("time", float)]
            )

            for model_name, model_factory in models_dict.items():
                key = (coh_name, fs_name, model_name)
                if key in done:
                    continue

                print(f"  {coh_name}/{fs_name}/{model_name} ...",
                      end=" ", flush=True)

                skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                       random_state=RANDOM_STATE)
                c_indices = []

                for fold_i, (train_idx, test_idx) in enumerate(
                        skf.split(X, events)):
                    try:
                        X_train = X.iloc[train_idx].values
                        X_test  = X.iloc[test_idx].values
                        y_train = y_surv[train_idx]
                        y_test  = y_surv[test_idx]

                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test  = scaler.transform(X_test)

                        model = model_factory()
                        model.fit(X_train, y_train)
                        risk = model.predict(X_test)

                        ci = concordance_index_censored(
                            y_test["event"], y_test["time"], risk)[0]
                        c_indices.append(ci)

                    except Exception as e:
                        print(f"\n    [ERROR fold {fold_i}] {e}")
                        continue

                if not c_indices:
                    print("SKIP")
                    continue

                mean_ci = np.mean(c_indices)
                std_ci  = np.std(c_indices)

                row = {
                    "cohort": coh_name,
                    "feature_set": fs_name,
                    "model": model_name,
                    "cv_folds": CV_FOLDS,
                    "c_index_mean": mean_ci,
                    "c_index_std": std_ci,
                }

                row_df = pd.DataFrame([row])
                if RESULTS_SURV_CSV.exists():
                    row_df.to_csv(RESULTS_SURV_CSV, mode="a",
                                   header=False, index=False)
                else:
                    row_df.to_csv(RESULTS_SURV_CSV, index=False)

                done.add(key)
                print(f"c-index={mean_ci:.4f}+-{std_ci:.4f}")

    if cox_summaries:
        pd.concat(cox_summaries, ignore_index=True).to_csv(
            COX_COHORT_MULTIVARIATE_CSV, index=False)
        print(f"\n[SURV-COHORT] Cox summaries saved to "
              f"{COX_COHORT_MULTIVARIATE_CSV}")

    if schoenfeld_summaries:
        pd.concat(schoenfeld_summaries, ignore_index=True).to_csv(
            COX_COHORT_SCHOENFELD_CSV, index=False)
        print(f"[SURV-COHORT] Schoenfeld tests saved to "
              f"{COX_COHORT_SCHOENFELD_CSV}")

    print(f"\n[SURV-COHORT] Complete.")


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run({"strict": cohorts["strict"], "competing": cohorts["competing"]})

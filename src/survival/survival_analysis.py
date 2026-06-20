"""
src/survival/survival_analysis.py
──────────────────────────────────
Survival analysis on all patients.
Pipeline.md section 7.

- Kaplan-Meier, 1-KM (cumulative incidence of CVD death)
- Aalen-Johansen for competing risks
- Stratification by Cox partial hazard tertiles
- Univariate and multivariate Cox (HR, p)
- Schoenfeld test
- c-index in 5-fold CV for CoxPH, CoxNet, RSF, GBSurv
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import (
    KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter,
)
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS, FIGURES_DIR, RESULTS_SURV_CSV,
    RANDOM_STATE, CV_FOLDS, HORIZON_YEARS,
    CARDIO_17, RSF_PARAMS, GBSURV_PARAMS,
    CONTINUOUS_FEATURES, FEATURE_SET_ORDER,
    SURVIVAL_SUMMARY_CSV, RISK_STRATIFICATION_CSV,
)
from src.features.feature_engineering import (
    add_derived_features, extract_Xy_survival, get_all_feature_set_names,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ─── Kaplan-Meier and cumulative incidence ─────────────────────────────

def plot_km_and_cumulative_incidence(df: pd.DataFrame, save: bool = True):
    """
    Plot KM (survival) and 1-KM (cumulative incidence of CVD death).
    Highlight the value at 7 years.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # KM
    kmf = KaplanMeierFitter()
    kmf.fit(df["time_years"], event_observed=df["event_cvd"],
            label="CVD Survival")

    kmf.plot_survival_function(ax=axes[0], ci_show=True)
    axes[0].axvline(x=HORIZON_YEARS, color="red", linestyle="--", alpha=0.5)
    # Value at 7 years
    s7 = kmf.predict(HORIZON_YEARS)
    axes[0].scatter([HORIZON_YEARS], [s7], color="red", zorder=5, s=50)
    axes[0].annotate(f"S(7) = {s7:.3f}",
                     xy=(HORIZON_YEARS, s7),
                     xytext=(HORIZON_YEARS + 0.5, s7 + 0.03),
                     fontsize=10, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
    axes[0].set_title("Kaplan-Meier: CVD Survival", fontweight="bold")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Survival probability")

    # 1-KM: cumulative incidence
    ci = 1 - kmf.survival_function_
    ci.columns = ["CVD Cumulative Incidence"]
    ci.plot(ax=axes[1], color="crimson")
    axes[1].fill_between(ci.index, 0, ci.iloc[:, 0], alpha=0.1, color="crimson")
    f7 = 1 - s7
    axes[1].axvline(x=HORIZON_YEARS, color="red", linestyle="--", alpha=0.5)
    axes[1].scatter([HORIZON_YEARS], [f7], color="red", zorder=5, s=50)
    axes[1].annotate(f"F(7) = {f7:.3f}",
                     xy=(HORIZON_YEARS, f7),
                     xytext=(HORIZON_YEARS + 0.5, f7 + 0.02),
                     fontsize=10, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
    axes[1].set_title("Cumulative Incidence of CVD Death (1-KM)",
                      fontweight="bold")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("CVD death probability")

    fig.suptitle("Kaplan-Meier Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "km_cumulative_incidence.png",
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[SURV] KM: S(7)={s7:.4f}, F(7)={f7:.4f}")
    return {"km_s7": float(s7), "km_f7": float(f7)}


# ─── Aalen-Johansen (competing risks) ────────────────────────────────

def plot_aalen_johansen(df: pd.DataFrame, save: bool = True):
    """
    Aalen-Johansen for competing risks.
    Event 1 = CVD death, Event 2 = non-CVD death, 0 = censored.
    """
    df_aj = df.copy()
    # Event column for AJ: 0=censored, 1=CVD death, 2=non-CVD death
    df_aj["event_aj"] = 0
    df_aj.loc[df_aj["event_cvd"] == 1, "event_aj"] = 1
    df_aj.loc[df_aj["event_noncvd"] == 1, "event_aj"] = 2

    def _aj_cif_at_horizon(data: pd.DataFrame) -> float:
        data = data[["time_years", "event_aj"]].sort_values("time_years")
        survival = 1.0
        cif = 0.0
        for time, group in data.groupby("time_years", sort=True):
            if time > HORIZON_YEARS:
                break
            at_risk = int((data["time_years"] >= time).sum())
            if at_risk == 0:
                continue
            d_interest = int((group["event_aj"] == 1).sum())
            d_all = int((group["event_aj"] != 0).sum())
            cif += survival * d_interest / at_risk
            survival *= (1 - d_all / at_risk)
        return float(cif)

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        # AJ for CVD death (event of interest = 1)
        aj = AalenJohansenFitter(calculate_variance=True)
        aj.fit(df_aj["time_years"], df_aj["event_aj"], event_of_interest=1)
        aj.plot(ax=ax, label="AJ: CVD death")

        # Comparison with 1-KM
        kmf = KaplanMeierFitter()
        kmf.fit(df_aj["time_years"], event_observed=df_aj["event_cvd"])
        ci_km = 1 - kmf.survival_function_
        ax.plot(ci_km.index, ci_km.values, "--", color="orange",
                label="1-KM (overestimates)")
        f7_km = float(1 - kmf.predict(HORIZON_YEARS))
        f7_aj = _aj_cif_at_horizon(df_aj)

        ax.axvline(x=HORIZON_YEARS, color="gray", linestyle=":", alpha=0.5)
        ax.scatter([HORIZON_YEARS], [f7_aj], color="navy", zorder=5, s=35)
        ax.annotate(f"AJ F(7)={f7_aj:.3f}",
                    xy=(HORIZON_YEARS, f7_aj),
                    xytext=(HORIZON_YEARS + 0.4, f7_aj - 0.025),
                    fontsize=9, color="navy")
        ax.set_title("Competing Risks: Aalen-Johansen vs 1-KM",
                     fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Cumulative incidence of CVD death")
        ax.legend()
        fig.tight_layout()

        if save:
            fig.savefig(FIGURES_DIR / "aalen_johansen.png",
                        bbox_inches="tight")
        over_abs = f7_km - f7_aj
        over_rel = 100 * over_abs / f7_aj if f7_aj > 0 else np.nan
        print(f"[SURV] Aalen-Johansen: F(7)={f7_aj:.4f}; "
              f"1-KM overestimate={over_abs:.4f} ({over_rel:.1f}%)")
        plt.close(fig)
        return {
            "aj_f7": f7_aj,
            "km_f7_for_aj": f7_km,
            "km_minus_aj_abs": over_abs,
            "km_minus_aj_relative_pct": over_rel,
        }
    except Exception as e:
        print(f"[SURV] Aalen-Johansen error: {e}")

    plt.close(fig)
    return {}


# ─── Risk stratification by Cox tertiles ──────────────────────────────

def plot_risk_stratification(df: pd.DataFrame, save: bool = True):
    """
    Stratification by Cox partial hazard tertiles on CV17.
    Plot cumulative incidence curves by risk group.
    """
    # Prepare data for Cox
    df_cox = df.copy()
    df_cox = add_derived_features(df_cox)
    feats = [f for f in CARDIO_17 if f in df_cox.columns]
    cols = feats + ["time_years", "event_cvd"]
    df_cox = df_cox[cols].dropna()

    if len(df_cox) == 0:
        print("[SURV] Insufficient data for stratification.")
        return

    # Fit Cox
    cph = CoxPHFitter()
    try:
        cph.fit(df_cox, duration_col="time_years", event_col="event_cvd")
    except Exception as e:
        print(f"[SURV] Cox fit error: {e}")
        return

    # Partial hazard
    ph = cph.predict_partial_hazard(df_cox)
    df_cox["risk_score"] = ph.values

    # Tertiles
    df_cox["risk_group"] = pd.qcut(df_cox["risk_score"], q=3,
                                    labels=["Low", "Medium", "High"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Survival
    for grp, color in zip(["Low", "Medium", "High"],
                          ["#4CAF50", "#FF9800", "#F44336"]):
        mask = df_cox["risk_group"] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(df_cox.loc[mask, "time_years"],
                df_cox.loc[mask, "event_cvd"],
                label=f"{grp} risk (n={mask.sum()})")
        kmf.plot_survival_function(ax=axes[0], ci_show=True, color=color)

    axes[0].axvline(x=HORIZON_YEARS, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_title("CVD Survival by Risk Group", fontweight="bold")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Survival probability")

    # Cumulative incidence of CVD death
    for grp, color in zip(["Low", "Medium", "High"],
                          ["#4CAF50", "#FF9800", "#F44336"]):
        mask = df_cox["risk_group"] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(df_cox.loc[mask, "time_years"],
                df_cox.loc[mask, "event_cvd"],
                label=f"{grp} risk")
        ci = 1 - kmf.survival_function_
        axes[1].plot(ci.index, ci.values, color=color, label=f"{grp} risk")
        # Value at 7 years
        f7 = 1 - kmf.predict(HORIZON_YEARS)
        axes[1].scatter([HORIZON_YEARS], [f7], color=color, zorder=5, s=30)
        axes[1].annotate(f"{f7:.3f}", xy=(HORIZON_YEARS, f7),
                         fontsize=8, color=color)

    axes[1].axvline(x=HORIZON_YEARS, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_title("Cumulative Incidence of CVD Death by Risk Group",
                      fontweight="bold")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("CVD death probability")
    axes[1].legend()

    # Log-rank tests
    groups = df_cox["risk_group"]
    low  = df_cox[groups == "Low"]
    high = df_cox[groups == "High"]
    lr = logrank_test(low["time_years"], high["time_years"],
                      low["event_cvd"], high["event_cvd"])
    lr_global = multivariate_logrank_test(
        df_cox["time_years"], df_cox["risk_group"], df_cox["event_cvd"])
    fig.suptitle(f"Risk Stratification (Log-rank Low vs High: "
                 f"p={lr.p_value:.2e})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "risk_stratification.png",
                    bbox_inches="tight")
    plt.close(fig)
    summary_rows = []
    for grp in ["Low", "Medium", "High"]:
        mask = df_cox["risk_group"] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(df_cox.loc[mask, "time_years"],
                df_cox.loc[mask, "event_cvd"], label=grp)
        f7 = float(1 - kmf.predict(HORIZON_YEARS))
        summary_rows.append({
            "risk_group": grp,
            "n": int(mask.sum()),
            "cvd_death_risk_7y_1km": f7,
        })
    summary = pd.DataFrame(summary_rows)
    summary["logrank_low_vs_high_p"] = lr.p_value
    summary["logrank_global_p"] = lr_global.p_value
    summary.to_csv(RISK_STRATIFICATION_CSV, index=False)
    print(f"[SURV] Stratification: Log-rank Low vs High p={lr.p_value:.2e}; "
          f"global p={lr_global.p_value:.2e}")
    return summary


# ─── Cox univariate and multivariate ──────────────────────────────────

def cox_analysis(df: pd.DataFrame, save: bool = True) -> dict:
    """
    Univariate and multivariate Cox analysis.
    HR per SD, p-value, CI.
    Schoenfeld test.
    """
    df = df.copy()
    df = add_derived_features(df)

    all_feats = sorted(set(CARDIO_17 + ['TSH', 'fT3', 'fT4',
                                         'Euthyroid', 'SCH', 'SCT',
                                         'Low_T3', 'Hypothyroid',
                                         'Hyperthyroid']))
    available = [f for f in all_feats if f in df.columns]
    cols = available + ["time_years", "event_cvd"]
    df_cox = df[cols].dropna()

    # Standardise continuous features for HR per SD
    scaler = StandardScaler()
    cont = [f for f in CONTINUOUS_FEATURES if f in df_cox.columns]
    df_cox[cont] = scaler.fit_transform(df_cox[cont])

    results_univ = []
    results_multi = {}

    # -- Univariate --
    print("\n[SURV] Cox Univariate:")
    for feat in available:
        try:
            cph = CoxPHFitter()
            cph.fit(df_cox[[feat, "time_years", "event_cvd"]],
                    duration_col="time_years", event_col="event_cvd")
            summary = cph.summary
            hr = summary["exp(coef)"].iloc[0]
            p  = summary["p"].iloc[0]
            ci_lo = summary["exp(coef) lower 95%"].iloc[0]
            ci_hi = summary["exp(coef) upper 95%"].iloc[0]
            results_univ.append({
                "feature": feat, "HR": hr, "p": p,
                "CI_lo": ci_lo, "CI_hi": ci_hi,
                "significant": p < 0.05,
            })
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                  ("*" if p < 0.05 else ""))
            print(f"  {feat:25s}: HR={hr:.3f} [{ci_lo:.3f}-{ci_hi:.3f}] "
                  f"p={p:.4f} {sig}")
        except Exception as e:
            print(f"  {feat}: ERROR {e}")

    univ_df = pd.DataFrame(results_univ)
    if save:
        univ_df.to_csv(FIGURES_DIR.parent / "cox_univariate.csv", index=False)

    # -- Multivariate --
    # Euthyroid is the reference category for thyroid-state dummy variables.
    thyroid_state_covariates = ["SCH", "SCT", "Low_T3",
                                "Hypothyroid", "Hyperthyroid"]
    multivariate_feats = [
        f for f in CARDIO_17 + ["TSH", "fT3", "fT4"] +
        thyroid_state_covariates
        if f in df_cox.columns
    ]

    print("\n[SURV] Cox Multivariate (CV17 + thyroid labs; "
          "Euthyroid as reference):")
    try:
        cph_multi = CoxPHFitter()
        cph_multi.fit(df_cox[multivariate_feats + ["time_years", "event_cvd"]],
                      duration_col="time_years", event_col="event_cvd")
        print(cph_multi.print_summary())
        results_multi["all"] = cph_multi.summary

        if save:
            cph_multi.summary.to_csv(
                FIGURES_DIR.parent / "cox_multivariate.csv")

        # Schoenfeld test
        print("\n[SURV] Schoenfeld test:")
        try:
            schoenfeld = cph_multi.check_assumptions(df_cox[
                multivariate_feats + ["time_years", "event_cvd"]],
                p_value_threshold=0.05, show_plots=False)
        except Exception as e:
            print(f"  Schoenfeld test: {e}")

    except Exception as e:
        print(f"  Cox multivariate error: {e}")

    return {"univariate": univ_df, "multivariate": results_multi}


# ─── c-index in 5-fold CV ────────────────────────────────────────────

def survival_cv(df: pd.DataFrame, force: bool = False):
    """
    c-index in 5-fold CV for CoxPH, CoxNet, RSF, GBSurv
    across all feature sets.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.ensemble import (
        RandomSurvivalForest, GradientBoostingSurvivalAnalysis,
    )
    from sksurv.metrics import concordance_index_censored

    print("\n" + "=" * 60)
    print("[SURV] c-index in 5-fold CV")
    print("=" * 60)

    if force and RESULTS_SURV_CSV.exists():
        existing = pd.read_csv(RESULTS_SURV_CSV)
        existing = existing[existing["cohort"] != "full"]
        if existing.empty:
            RESULTS_SURV_CSV.unlink()
        else:
            existing.to_csv(RESULTS_SURV_CSV, index=False)

    df = df.copy()
    df = add_derived_features(df)

    # Load existing results
    done = set()
    if not force and RESULTS_SURV_CSV.exists():
        existing = pd.read_csv(RESULTS_SURV_CSV)
        done = {(r["feature_set"], r["model"]) for _, r in existing.iterrows()
                if r.get("cohort", "full") == "full"}

    models_dict = {
        "CoxPH": lambda: CoxPHSurvivalAnalysis(),
        "CoxNet": lambda: CoxnetSurvivalAnalysis(
            l1_ratio=0.5, alphas=[0.001],
            max_iter=10000),
        "RSF": lambda: RandomSurvivalForest(**RSF_PARAMS),
        "GBSurv": lambda: GradientBoostingSurvivalAnalysis(**GBSURV_PARAMS),
    }

    fs_names = get_all_feature_set_names()

    for fs_name in fs_names:
        try:
            X, y_surv = extract_Xy_survival(df, fs_name)
        except Exception as e:
            print(f"  [SKIP] {fs_name}: {e}")
            continue

        if len(X) == 0:
            continue

        # Create event vector for stratification
        events = y_surv["event"].astype(int)

        for model_name, model_factory in models_dict.items():
            key = (fs_name, model_name)
            if key in done:
                continue

            print(f"  {fs_name}/{model_name} ...", end=" ", flush=True)

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

                    # Scale
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test  = scaler.transform(X_test)

                    model = model_factory()
                    model.fit(X_train, y_train)

                    # Predictions (risk score)
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
                "cohort": "full",
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

    print(f"\n[SURV] c-index saved to {RESULTS_SURV_CSV}")


def plot_survival_results(save: bool = True):
    """Plot c-index by feature set and model."""
    if not RESULTS_SURV_CSV.exists():
        print("[PLOT] results_surv.csv not found.")
        return

    df = pd.read_csv(RESULTS_SURV_CSV)
    df_full = df[df["cohort"] == "full"]

    if df_full.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for model in df_full["model"].unique():
        m = df_full[df_full["model"] == model].copy()
        m["feature_set"] = pd.Categorical(
            m["feature_set"], categories=FEATURE_SET_ORDER, ordered=True)
        m = m.sort_values("feature_set")
        ax.errorbar(m["feature_set"], m["c_index_mean"],
                    yerr=m["c_index_std"], marker="o", capsize=3,
                    label=model)

    ax.set_xlabel("Feature Set", fontsize=11)
    ax.set_ylabel("c-index (5-fold CV)", fontsize=11)
    ax.set_title("Survival c-index by Feature Set and Model",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "survival_cindex.png", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Survival c-index saved.")


# ─── Entry point ────────────────────────────────────────────────────────

def run(df_survival: pd.DataFrame, force: bool = False):
    """Run all survival analyses."""
    print("\n" + "=" * 60)
    print("[SURVIVAL] Survival Analysis -- full sample")
    print("=" * 60)

    # 1. KM and cumulative incidence
    km_summary = plot_km_and_cumulative_incidence(df_survival)

    # 2. Aalen-Johansen
    aj_summary = plot_aalen_johansen(df_survival)
    summary = {**km_summary, **aj_summary}
    if summary:
        pd.DataFrame([summary]).to_csv(SURVIVAL_SUMMARY_CSV, index=False)

    # 3. Risk stratification
    plot_risk_stratification(df_survival)

    # 4. Cox analysis
    cox_results = cox_analysis(df_survival)

    # 5. c-index CV
    survival_cv(df_survival, force=force)

    # 6. Plot
    plot_survival_results()

    print("\n[SURVIVAL] Complete.")
    return cox_results


if __name__ == "__main__":
    from src.preprocessing.build_dataset import run as build
    cohorts = build()
    run(cohorts["survival"])

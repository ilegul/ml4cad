"""
src/alignment/ml_indicator.py
─────────────────────────────
The ML composite indicator in a survival model (replica of the paper's idea).

Steps (paper-alignment):
  1.  Take the ensemble's predicted probability of the *event* (CVD death within
      the horizon). The paper's indicator is the probability of *surviving* the
      horizon, so we also expose ``p_survive = 1 - p_event``.
  2.  Use the predicted risk (``p_event``) as the **single covariate** in a Cox
      model and report Harrell's C-index.
  3.  Kaplan–Meier stratification at a probability threshold (0.6, as in the
      paper, on ``p_survive``) and at a data-driven threshold (the median),
      with a log-rank test and per-group survival at the horizon.
  4.  Compare the C-index of CV17 vs CV17+thyroid (Δ).

CAVEAT (printed in the notebook): this survival analysis runs on the **strict**
cohort, i.e. selecting on the outcome (patients censored before the horizon are
dropped). The estimates are therefore *conditional* — consistent with the paper,
but DIFFERENT from the competing-risk (Aalen–Johansen) survival used elsewhere
in the thesis on the full cohort. It does not overwrite or contradict that
analysis; it exists only for comparability with the paper.

Probabilities are produced leakage-free: OOF (scheme A) or test-only (scheme B).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE
from src.alignment.features_align import extract_Xy_align
from src.alignment.evaluation import evaluate_cv, evaluate_holdout


def _survival_frame(strict_df: pd.DataFrame, horizon_years: float) -> pd.DataFrame:
    """
    Administrative censoring at the horizon (same recipe as survival_cohort):
        surv_time  = min(time_years, horizon)
        surv_event = event_cvd & time_years <= horizon
    Returns a frame indexed like ``strict_df`` with surv_time / surv_event.
    """
    df = strict_df.copy()
    df["surv_time"] = df["time_years"].clip(upper=horizon_years)
    df["surv_event"] = ((df["event_cvd"] == 1) &
                        (df["time_years"] <= horizon_years)).astype(int)
    return df


def predicted_risk(strict_df, feature_set_name, target_col,
                   scheme="A", model_name="ENSEMBLE",
                   sampler="RandomOverSampler", seed=RANDOM_STATE):
    """
    Return a DataFrame (indexed by the strict cohort's index) with the ensemble
    predicted ``p_event`` and ``p_survive`` for the patients that received a
    prediction (all patients for scheme A OOF; the test rows for scheme B).
    """
    X, y = extract_Xy_align(strict_df, feature_set_name, target_col)
    if scheme == "A":
        res = evaluate_cv(X, y, model_name, sampler=sampler, seed=seed)
        idx, p_event = res["oof_index"], res["oof_proba"]
    elif scheme == "B":
        res = evaluate_holdout(X, y, model_name, sampler=sampler, seed=seed)
        idx, p_event = res["test_index"], res["test_proba"]
    else:
        raise ValueError("scheme must be 'A' or 'B'")

    out = pd.DataFrame({"p_event": p_event}, index=idx)
    out["p_survive"] = 1.0 - out["p_event"]
    return out


def cox_cindex(strict_df, risk_df, horizon_years) -> dict:
    """
    Fit a single-covariate Cox model (covariate = p_event) and return its
    C-index, N and number of events.
    """
    from lifelines import CoxPHFitter

    surv = _survival_frame(strict_df, horizon_years)
    data = surv[["surv_time", "surv_event"]].join(risk_df["p_event"],
                                                  how="inner").dropna()
    cph = CoxPHFitter()
    cph.fit(data, duration_col="surv_time", event_col="surv_event")
    return {
        "c_index": float(cph.concordance_index_),
        "n": int(len(data)),
        "events": int(data["surv_event"].sum()),
        "coef_p_event": float(cph.params_["p_event"]),
    }


def paired_cindex_delta(strict_df, base_risk, thy_risk, horizon_years,
                        n_boot=1000, seed=RANDOM_STATE) -> dict:
    """Paired bootstrap CI for Δ Harrell C-index (thyroid - baseline).

    Both risk scores are evaluated on the same patients. ``p_event`` is
    negated because ``lifelines.utils.concordance_index`` expects larger scores
    to indicate longer survival.
    """
    from lifelines.utils import concordance_index

    surv = _survival_frame(strict_df, horizon_years)
    data = (surv[["surv_time", "surv_event"]]
            .join(base_risk[["p_event"]].rename(
                columns={"p_event": "p_event_base"}), how="inner")
            .join(thy_risk[["p_event"]].rename(
                columns={"p_event": "p_event_thy"}), how="inner")
            .dropna())

    def _cindex(frame, col):
        return concordance_index(
            frame["surv_time"], -frame[col], frame["surv_event"])

    base = _cindex(data, "p_event_base")
    thy = _cindex(data, "p_event_thy")
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(data)
    for _ in range(n_boot):
        sample = data.iloc[rng.integers(0, n, n)]
        if sample["surv_event"].nunique() < 2:
            continue
        try:
            deltas.append(
                _cindex(sample, "p_event_thy")
                - _cindex(sample, "p_event_base"))
        except ZeroDivisionError:
            continue
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_c_index": float(thy - base),
        "delta_c_index_ci_lo": float(lo),
        "delta_c_index_ci_hi": float(hi),
        "n": int(n),
        "events": int(data["surv_event"].sum()),
    }


def km_stratify(strict_df, risk_df, horizon_years, threshold,
                threshold_name) -> dict:
    """
    Kaplan–Meier stratification on ``p_survive`` at ``threshold``.

    Group "high_predicted_survival" = p_survive >= threshold (low risk);
    "low_predicted_survival" = p_survive < threshold. Returns survival at the
    horizon per group, group sizes, and the log-rank p-value.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    surv = _survival_frame(strict_df, horizon_years)
    data = surv[["surv_time", "surv_event"]].join(risk_df["p_survive"],
                                                  how="inner").dropna()
    high = data["p_survive"] >= threshold
    out = {
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "n_high": int(high.sum()),
        "n_low": int((~high).sum()),
    }

    kmf = KaplanMeierFitter()
    curves = {}
    for label, mask in [("high_predicted_survival", high),
                        ("low_predicted_survival", ~high)]:
        if mask.sum() == 0:
            out[f"surv_at_horizon_{label}"] = np.nan
            curves[label] = None
            continue
        kmf.fit(data.loc[mask, "surv_time"], data.loc[mask, "surv_event"],
                label=label)
        out[f"surv_at_horizon_{label}"] = float(
            kmf.predict(horizon_years))
        curves[label] = (data.loc[mask, "surv_time"].values,
                         data.loc[mask, "surv_event"].values)

    if high.sum() > 0 and (~high).sum() > 0:
        lr = logrank_test(
            data.loc[high, "surv_time"], data.loc[~high, "surv_time"],
            event_observed_A=data.loc[high, "surv_event"],
            event_observed_B=data.loc[~high, "surv_event"])
        out["logrank_p"] = float(lr.p_value)
    else:
        out["logrank_p"] = np.nan

    out["_curves"] = curves
    return out


def compare_indicator(strict_df, horizon_years, base_set="CV17",
                      thy_set="CV17_THY_CONT_STATES", target_col=None,
                      scheme="A", model_name="ENSEMBLE", seed=RANDOM_STATE,
                      n_boot=1000):
    """
    Full ML-indicator comparison CV17 vs a thyroid set for one horizon/scheme,
    using ``model_name`` to produce the predicted risk.

    Returns a dict with the Cox C-index for each set, the Δ C-index, and the KM
    stratification (0.6 and median thresholds) for both sets.
    """
    h = int(round(horizon_years))
    if target_col is None:
        target_col = f"y{h}"

    results = {"horizon": h, "scheme": scheme, "model": model_name}
    risks = {}
    for tag, fs in [("base", base_set), ("thy", thy_set)]:
        risk = predicted_risk(strict_df, fs, target_col, scheme=scheme,
                              model_name=model_name, seed=seed)
        risks[tag] = risk
        cidx = cox_cindex(strict_df, risk, horizon_years)
        med = float(risk["p_survive"].median())
        km_06 = km_stratify(strict_df, risk, horizon_years, 0.6, "fixed_0.6")
        km_md = km_stratify(strict_df, risk, horizon_years, med, "median")
        results[tag] = {
            "feature_set": fs,
            "cox": cidx,
            "km_0.6": km_06,
            "km_median": km_md,
        }
    delta = paired_cindex_delta(
        strict_df, risks["base"], risks["thy"], horizon_years,
        n_boot=n_boot, seed=seed)
    results["delta_c_index"] = delta["delta_c_index"]
    results["delta_c_index_ci_lo"] = delta["delta_c_index_ci_lo"]
    results["delta_c_index_ci_hi"] = delta["delta_c_index_ci_hi"]
    return results

"""Survival analyses.

Analysis A adapts the paper's indicator-based survival analysis to the
strict fixed-horizon cohort: the ML
indicator comes from out-of-sample classification probabilities and is then used
in Cox and Kaplan-Meier models.

Analysis B is the model extension on the full time-to-event cohort: a Cox
reference and a cause-specific Random Survival Forest. Non-cardiac death is
treated as censoring, so nothing in analysis B is a competing-risk cumulative
incidence.

Analysis C targets that estimand directly: cause-specific Cox models for
cardiac and non-cardiac death are combined through their Breslow baselines
into each patient's cardiac cumulative incidence, with Aalen-Johansen as the
non-parametric reference for the absolute scale.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import randint
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import cumulative_incidence_competing_risks
from sksurv.util import Surv

import config
import utils


# ---------------------------------------------------------------------------
# Shared time-to-event construction
# ---------------------------------------------------------------------------

def survival_frame(df: pd.DataFrame, horizon: float = None) -> pd.DataFrame:
    """Administrative censoring at the horizon; no horizon means full follow-up."""
    out = df.copy()
    time = out["survival_time_years"].to_numpy(dtype=float)
    event = (out["event_cardiac"] == 1).to_numpy()
    if horizon is not None:
        event = event & (time <= horizon)
        time = np.minimum(time, horizon)
    out["surv_time"] = time
    out["surv_event"] = event.astype(int)
    return out


def make_surv_y(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(event=frame["surv_event"].astype(bool),
                            time=frame["surv_time"].astype(float))


def _time_bounds(y_train, y_test) -> tuple:
    """Common estimable support of the training and held-out folds.

    The upper bound is the largest training event time rather than the horizon,
    so the censoring distribution stays estimable, and the held-out range is
    intersected so that every requested time is observable in both. The bounds
    therefore depend on the follow-up support of both folds; the censoring
    distribution used by the inverse-probability weights is estimated from the
    training fold alone.
    """
    lo = max(y_train["time"].min(), y_test["time"].min())
    train_events = y_train["time"][y_train["event"]]
    hi = min(train_events.max() if train_events.size else y_train["time"].max(),
             y_test["time"].max())
    return float(lo), float(hi)


def safe_times(y_train, y_test, requested) -> np.ndarray:
    """Requested evaluation times restricted to the estimable range."""
    requested = np.atleast_1d(np.asarray(requested, dtype=float))
    lo, hi = _time_bounds(y_train, y_test)
    keep = requested[(requested > lo) & (requested < hi)]
    if keep.size == 0:
        keep = np.array([lo + 0.99 * (hi - lo)])
    return keep


def safe_time_grid(y_train, y_test, n_points: int = 50) -> np.ndarray:
    """Grid for the integrated Brier score, inside the estimable range."""
    lo, hi = _time_bounds(y_train, y_test)
    span = hi - lo
    return np.linspace(lo + 0.01 * span, hi - 0.01 * span, n_points)


# ---------------------------------------------------------------------------
# Analysis A: paper-aligned ML indicator on the strict cohort
# ---------------------------------------------------------------------------

def indicator_frame(predictions: pd.DataFrame, strict: pd.DataFrame,
                    horizon: float) -> pd.DataFrame:
    """Join frozen out-of-sample probabilities to the survival outcome."""
    surv = survival_frame(strict, horizon)
    cols = [config.ID_COL, "surv_time", "surv_event"]
    merged = predictions.merge(surv[cols], on=config.ID_COL, how="inner",
                               validate="one_to_one")
    if len(merged) != len(predictions):
        raise ValueError("Some predicted patients are absent from the strict cohort")
    return merged


def cox_indicator(frame: pd.DataFrame, score_col: str = "proba_calibrated") -> dict:
    """Harrell's C for the indicator plus a descriptive hazard ratio.

    The C-index of a single monotone covariate does not depend on the fitted
    coefficient, so the Cox fit only supplies the hazard ratio.
    """
    cindex = concordance_index_censored(
        frame["surv_event"].astype(bool).to_numpy(),
        frame["surv_time"].to_numpy(dtype=float),
        frame[score_col].to_numpy(dtype=float))[0]

    data = frame[["surv_time", "surv_event", score_col]].rename(
        columns={score_col: "indicator"})
    cph = CoxPHFitter().fit(data, duration_col="surv_time", event_col="surv_event")
    return {
        "n": int(len(frame)),
        "events": int(frame["surv_event"].sum()),
        "c_index": float(cindex),
        "hazard_ratio_per_unit": float(np.exp(cph.params_["indicator"])),
        "hazard_ratio_p": float(cph.summary.loc["indicator", "p"]),
    }


def km_stratify(frame: pd.DataFrame, cut: float, cut_origin: str,
                horizon: float, score_col: str = "p_survive") -> dict:
    """Kaplan-Meier stratification at a cut on predicted survival probability."""
    high = frame[score_col].to_numpy(dtype=float) >= cut
    out = {
        "cut": float(cut),
        "cut_origin": cut_origin,
        "n_high_predicted_survival": int(high.sum()),
        "n_low_predicted_survival": int((~high).sum()),
        "events_high_predicted_survival": int(frame.loc[high, "surv_event"].sum()),
        "events_low_predicted_survival": int(frame.loc[~high, "surv_event"].sum()),
    }
    kmf = KaplanMeierFitter()
    for label, mask in [("high_predicted_survival", high),
                        ("low_predicted_survival", ~high)]:
        if mask.sum() == 0:
            out[f"survival_at_horizon_{label}"] = np.nan
            out[f"survival_ci_lo_{label}"] = np.nan
            out[f"survival_ci_hi_{label}"] = np.nan
            continue
        kmf.fit(frame.loc[mask, "surv_time"], frame.loc[mask, "surv_event"])
        out[f"survival_at_horizon_{label}"] = float(kmf.predict(horizon))
        ci = kmf.confidence_interval_survival_function_
        row = ci.index[ci.index <= horizon]
        if len(row):
            lo, hi = ci.loc[row[-1]].to_numpy()
            out[f"survival_ci_lo_{label}"] = float(lo)
            out[f"survival_ci_hi_{label}"] = float(hi)
        else:
            out[f"survival_ci_lo_{label}"] = np.nan
            out[f"survival_ci_hi_{label}"] = np.nan

    if high.sum() and (~high).sum():
        test = logrank_test(
            frame.loc[high, "surv_time"], frame.loc[~high, "surv_time"],
            event_observed_A=frame.loc[high, "surv_event"],
            event_observed_B=frame.loc[~high, "surv_event"])
        # Separation between strata, not a test of incremental thyroid value.
        out["descriptive_logrank_p"] = float(test.p_value)
    else:
        out["descriptive_logrank_p"] = np.nan
    return out


def paired_cindex_delta(frame_base: pd.DataFrame, frame_other: pd.DataFrame,
                        score_col: str = "proba_calibrated",
                        n_boot: int = None, seed: int = config.SEED) -> dict:
    """Paired bootstrap of the C-index difference on identical patients."""
    n_boot = config.BOOTSTRAP if n_boot is None else n_boot
    utils.assert_same_patients(frame_base[config.ID_COL].to_numpy(),
                               frame_other[config.ID_COL].to_numpy())

    event = frame_base["surv_event"].astype(bool).to_numpy()
    time = frame_base["surv_time"].to_numpy(dtype=float)
    risk_base = frame_base[score_col].to_numpy(dtype=float)
    risk_other = frame_other[score_col].to_numpy(dtype=float)

    def cindex(mask_event, mask_time, risk):
        return concordance_index_censored(mask_event, mask_time, risk)[0]

    point = cindex(event, time, risk_other) - cindex(event, time, risk_base)
    draws = utils.bootstrap_indices(len(time), n_boot, seed)
    deltas = []
    for idx in draws:
        if event[idx].sum() < 2 or event[idx].all():
            continue
        try:
            deltas.append(cindex(event[idx], time[idx], risk_other[idx])
                          - cindex(event[idx], time[idx], risk_base[idx]))
        except (ZeroDivisionError, ValueError):
            continue
    lo, hi = utils.bootstrap_ci(deltas)
    return {
        "delta_c_index": float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_bootstrap": utils.bootstrap_pvalue(deltas),
        "n_boot_used": int(len(deltas)),
        "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
    }


# ---------------------------------------------------------------------------
# Analysis B: survival models on the full time-to-event cohort
# ---------------------------------------------------------------------------

def survival_pipeline(model_name: str, seed: int = config.SEED) -> Pipeline:
    """Imputation and scaling only; resampling is meaningless for a survival target."""
    if model_name == "RSF":
        # A leaf floor keeps the untuned default affordable on this cohort size;
        # the per-fold search overrides it.
        estimator = RandomSurvivalForest(
            n_estimators=config.RSF_N_ESTIMATORS, min_samples_leaf=15,
            random_state=seed, n_jobs=config.N_JOBS)
    elif model_name == "Cox":
        estimator = CoxPHSurvivalAnalysis(alpha=1e-4)
    else:
        raise KeyError(f"Unknown survival model {model_name!r}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("model", estimator)])


def rsf_search_space() -> dict:
    return {
        "model__max_depth": [None, 4, 6, 8, 12],
        "model__min_samples_split": randint(4, 40),
        "model__min_samples_leaf": randint(5, 60),
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__max_samples": [None, 0.6, 0.8],
    }


def _sample_params(space: dict, n_iter: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_iter):
        params = {}
        for key, dist in space.items():
            if hasattr(dist, "rvs"):
                params[key] = dist.rvs(random_state=int(rng.integers(1 << 31)))
            else:
                params[key] = dist[int(rng.integers(len(dist)))]
        draws.append(params)
    return draws


def survival_metrics(y_train, y_test, risk, surv_probs, times, grid) -> dict:
    """All nuisance quantities are estimated from the training fold only."""
    harrell = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    tau = float(times.max())
    out = {"c_harrell": float(harrell), "metric_failures": ""}
    failures = []

    try:
        out["c_uno"] = float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])
    except Exception as error:
        out["c_uno"] = np.nan
        failures.append(f"c_uno: {type(error).__name__}")
    try:
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
        out["auc_time_dependent"] = float(mean_auc)
        out["auc_at_last_time"] = float(auc[-1])
    except Exception as error:
        out["auc_time_dependent"] = np.nan
        out["auc_at_last_time"] = np.nan
        failures.append(f"auc: {type(error).__name__}")
    try:
        out["integrated_brier"] = float(
            integrated_brier_score(y_train, y_test, surv_probs, grid))
    except Exception as error:
        out["integrated_brier"] = np.nan
        failures.append(f"integrated_brier: {type(error).__name__}")

    out["metric_failures"] = "; ".join(failures)
    return out


def _survival_probabilities(pipeline, X, times) -> np.ndarray:
    """Survival probabilities at the given times.

    A scikit-learn Pipeline does not forward predict_survival_function, so the
    preprocessing steps are applied explicitly before calling the estimator.
    """
    data = X
    for _, step in pipeline.steps[:-1]:
        data = step.transform(data)
    functions = pipeline.steps[-1][1].predict_survival_function(data)
    return np.vstack([fn(times) for fn in functions])


def evaluate_survival_cv(X, frame: pd.DataFrame, model_name: str, folds,
                         horizons=None, n_iter: int = None,
                         seed: int = config.SEED) -> dict:
    """Out-of-fold survival evaluation with per-fold tuning on the training fold.

    Both feature sets must be passed the same fold list so the comparison is
    paired, and the evaluation times come from the training fold alone.
    """
    horizons = config.HORIZONS if horizons is None else horizons
    n_iter = config.RSF_SEARCH_ITER if n_iter is None else n_iter

    y_all = make_surv_y(frame)
    rows = []
    oof_risk = np.full(len(frame), np.nan)

    for fold, (tr, te) in enumerate(folds):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_all[tr], y_all[te]

        best, best_score = None, -np.inf
        if model_name == "RSF" and n_iter > 0:
            inner = utils.make_folds(frame["surv_event"].to_numpy()[tr],
                                     n_splits=2, seed=seed)
            for params in _sample_params(rsf_search_space(), n_iter, seed + fold):
                pipe = survival_pipeline(model_name, seed).set_params(**params)
                scores = []
                for itr, ite in inner:
                    pipe.fit(X_tr.iloc[itr], y_tr[itr])
                    scores.append(concordance_index_censored(
                        y_tr[ite]["event"], y_tr[ite]["time"],
                        pipe.predict(X_tr.iloc[ite]))[0])
                score = float(np.mean(scores))
                if score > best_score:
                    best, best_score = params, score
        pipe = survival_pipeline(model_name, seed)
        if best:
            pipe.set_params(**best)
        pipe.fit(X_tr, y_tr)

        risk = pipe.predict(X_te)
        oof_risk[te] = risk
        times = safe_times(y_tr, y_te, horizons)
        grid = safe_time_grid(y_tr, y_te)
        surv_probs = _survival_probabilities(pipe, X_te, grid)
        rows.append({
            "fold": fold, "model": model_name,
            "n_train": len(tr), "n_test": len(te),
            "events_test": int(y_te["event"].sum()),
            "eval_times": ",".join(f"{t:.3f}" for t in times),
            "brier_grid": f"{grid.min():.3f}-{grid.max():.3f}",
            "inner_best_c": best_score if np.isfinite(best_score) else np.nan,
            **survival_metrics(y_tr, y_te, risk, surv_probs, times, grid),
        })

    return {"folds": pd.DataFrame(rows), "oof_risk": oof_risk}


def permutation_importance_surv(estimator, X, y_struct, n_repeats: int = 5,
                                seed: int = config.SEED) -> pd.DataFrame:
    """Out-of-sample permutation importance on Harrell's C."""
    rng = np.random.default_rng(seed)
    base = concordance_index_censored(y_struct["event"], y_struct["time"],
                                      estimator.predict(X))[0]
    rows = []
    for column in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = concordance_index_censored(
                y_struct["event"], y_struct["time"],
                estimator.predict(shuffled))[0]
            drops.append(base - score)
        rows.append({"feature": column,
                     "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


# ---------------------------------------------------------------------------
# Analysis C: cause-specific absolute risk under competing events
# ---------------------------------------------------------------------------

def competing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full-follow-up frame with a single cause code per patient.

    0 = censored alive, 1 = cardiac death, 2 = non-cardiac death. The code
    also serves as the stratification label of the outer folds, so every
    fold carries its share of both event types.
    """
    out = df.copy()
    out["cause_code"] = np.where(out["event_cardiac"] == 1, 1,
                                 np.where(out["event_noncardiac"] == 1, 2, 0))
    out["surv_time"] = out["survival_time_years"].astype(float)
    out["surv_event"] = (out["cause_code"] == 1).astype(int)
    return out


def fit_cause_specific(X, frame: pd.DataFrame, seed: int = config.SEED) -> dict:
    """One Cox pipeline per cause on shared covariates.

    Each fit treats the competing cause as censoring, which is the
    definition of the cause-specific hazard; the competing-risk correction
    happens where the two fitted hazards are combined, in
    predict_cardiac_cif.
    """
    models = {}
    for code in (1, 2):
        y = Surv.from_arrays(event=frame["cause_code"].to_numpy() == code,
                             time=frame["surv_time"].to_numpy(dtype=float))
        pipe = survival_pipeline("Cox", seed)
        pipe.fit(X, y)
        models[code] = pipe
    return models


def predict_cardiac_cif(models: dict, X, times) -> np.ndarray:
    """Cardiac cumulative incidence F1 at the requested times.

    F1(t | x) accumulates, over the cardiac event times u <= t, the product
    of all-cause survival just before u and the cardiac hazard increment
    dH1(u | x), with S = exp(-(H1 + H2)) built from both Breslow
    baselines: the plug-in form of F1(t) = int S(u-) dH1(u). Left limits
    matter: each increment must be weighted by survival before the jump,
    or F1 would exceed the all-cause risk. The running total is truncated
    at one, which the discrete sum can exceed only when an extreme
    covariate value drives the linear predictor far outside the fitted
    range.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    base1 = models[1].steps[-1][1].cum_baseline_hazard_
    base2 = models[2].steps[-1][1].cum_baseline_hazard_

    # H1 jumps only at cardiac event times, so that grid carries the sum.
    grid = np.asarray(base1.x, dtype=float)
    inc1 = np.diff(np.concatenate([[0.0], np.asarray(base1.y, dtype=float)]))
    left1 = np.concatenate([[0.0], np.asarray(base1.y, dtype=float)[:-1]])
    idx2 = np.searchsorted(np.asarray(base2.x, dtype=float), grid,
                           side="left") - 1
    left2 = np.where(idx2 >= 0,
                     np.asarray(base2.y, dtype=float)[np.clip(idx2, 0, None)],
                     0.0)

    hazard1 = np.exp(models[1].predict(X))
    hazard2 = np.exp(models[2].predict(X))
    columns = np.searchsorted(grid, times, side="right") - 1
    out = np.empty((len(hazard1), len(times)))
    for start in range(0, len(hazard1), 2000):
        e1 = hazard1[start:start + 2000, None]
        e2 = hazard2[start:start + 2000, None]
        survival_left = np.exp(-(e1 * left1[None, :] + e2 * left2[None, :]))
        jump = e1 * inc1[None, :]
        cif = np.minimum(np.cumsum(survival_left * jump, axis=1), 1.0)
        out[start:start + 2000] = np.where(
            columns[None, :] >= 0, cif[:, np.clip(columns, 0, None)], 0.0)
    return out


def evaluate_cif_cv(X, frame: pd.DataFrame, folds, horizons=None,
                    seed: int = config.SEED) -> dict:
    """Out-of-fold predicted cardiac incidence with per-fold refits.

    Both feature sets must be passed the same fold list so the comparison
    stays paired. Ranking metrics remain cause-specific, as in analysis B,
    so the thyroid contrast is comparable across analyses; the absolute
    scale of the predictions is assessed separately in cif_calibration.
    """
    horizons = config.HORIZONS if horizons is None else horizons
    horizons = [float(h) for h in horizons]

    y_all = make_surv_y(frame)
    rows = []
    oof = {h: np.full(len(frame), np.nan) for h in horizons}
    for fold, (tr, te) in enumerate(folds):
        models = fit_cause_specific(X.iloc[tr], frame.iloc[tr], seed)
        cif = predict_cardiac_cif(models, X.iloc[te], horizons)
        y_tr, y_te = y_all[tr], y_all[te]
        for j, horizon in enumerate(horizons):
            risk = cif[:, j]
            oof[horizon][te] = risk
            tau = float(safe_times(y_tr, y_te, [horizon])[-1])
            # The score targets this horizon, so Harrell's concordance is
            # administratively censored at it; Uno is truncated via tau.
            event_h = y_te["event"] & (y_te["time"] <= horizon)
            time_h = np.minimum(y_te["time"], horizon)
            harrell = concordance_index_censored(event_h, time_h, risk)[0]
            try:
                uno = float(concordance_index_ipcw(y_tr, y_te, risk,
                                                   tau=tau)[0])
            except Exception:
                uno = np.nan
            rows.append({"fold": fold, "horizon": horizon,
                         "n_test": len(te),
                         "events_cardiac_test": int(y_te["event"].sum()),
                         "tau": tau, "c_harrell": float(harrell),
                         "c_uno": uno,
                         "mean_predicted_cif": float(np.mean(risk))})
    return {"folds": pd.DataFrame(rows), "oof_cif": oof}


def cif_calibration(frame: pd.DataFrame, cif, horizon: float,
                    n_groups: int = 5, seed: int = config.SEED) -> pd.DataFrame:
    """Mean predicted incidence against Aalen-Johansen in risk groups.

    Groups are quantiles of the out-of-fold predicted incidence. This is
    the calibration check that matches the competing-risk estimand: within
    each group the observed quantity is itself a cumulative incidence, so
    a 1 - Kaplan-Meier reference would overstate every observed value.
    """
    cif = np.asarray(cif, dtype=float)
    edges = np.quantile(cif, np.linspace(0.0, 1.0, n_groups + 1))
    group = np.clip(np.searchsorted(edges[1:-1], cif, side="right"),
                    0, n_groups - 1)
    rows = []
    for g in range(n_groups):
        mask = group == g
        aj = aalen_johansen_cif(frame.loc[mask], horizon, seed)
        rows.append({"group": str(g + 1), "n": int(mask.sum()),
                     "mean_predicted_cif": float(cif[mask].mean()),
                     "observed_cif_aalen_johansen": aj["cif_aalen_johansen"],
                     "events_cardiac": aj["events_cardiac"],
                     "events_noncardiac": aj["events_noncardiac"]})
    overall = aalen_johansen_cif(frame, horizon, seed)
    rows.append({"group": "overall", "n": int(len(frame)),
                 "mean_predicted_cif": float(cif.mean()),
                 "observed_cif_aalen_johansen": overall["cif_aalen_johansen"],
                 "events_cardiac": overall["events_cardiac"],
                 "events_noncardiac": overall["events_noncardiac"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Competing risks
# ---------------------------------------------------------------------------

def aalen_johansen_cif(df: pd.DataFrame, horizon: float, seed: int = config.SEED) -> dict:
    """Cardiac cumulative incidence with non-cardiac death as a competing event.

    This is the estimator to quote for absolute cardiac risk; 1 - Kaplan-Meier
    censors the competing deaths and overstates it. The scikit-survival
    estimator handles tied event times exactly, where the lifelines fitter
    would break ties with a random jitter; the seed argument is kept for
    signature stability and is unused.
    """
    time = df["survival_time_years"].to_numpy(dtype=float)
    code = np.where(df["event_cardiac"] == 1, 1,
                    np.where(df["event_noncardiac"] == 1, 2, 0))

    grid, incidence = cumulative_incidence_competing_risks(code, time)
    at_horizon = grid <= horizon
    aj_risk = float(incidence[1][at_horizon][-1]) if at_horizon.any() else np.nan

    kmf = KaplanMeierFitter().fit(time, (code == 1).astype(int))
    km_risk = float(1.0 - kmf.predict(horizon))
    return {
        "horizon": float(horizon),
        "n": int(len(df)),
        "events_cardiac": int((code == 1).sum()),
        "events_noncardiac": int((code == 2).sum()),
        "cif_aalen_johansen": aj_risk,
        "risk_one_minus_km": km_risk,
        "overestimate_of_one_minus_km": km_risk - aj_risk,
    }

"""Decision-analytic summaries for out-of-sample event probabilities.

The functions in this module accept predictions that have already been
generated out of sample (for example, held-out test predictions or
out-of-fold predictions).  They do not fit or select a prediction model and
they never optimise a clinical threshold.

Calibration, decision-curve net benefit, and classification counts are
decision-analytic proxies.  They can describe the potential consequences of
using a model under stated assumptions, but they are not evidence that using
the model improves realised patient outcomes.  Demonstrating realised impact
requires prospective evaluation in the intended clinical workflow.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


DEFAULT_BOOTSTRAP_SEED = 42

__all__ = [
    "calibration_summary",
    "calibration_curve_table",
    "decision_curve",
    "paired_bootstrap_delta_net_benefit",
    "clinical_impact_table",
]


def _materialize_array(values: Iterable) -> np.ndarray:
    """Convert ordinary array-likes and one-pass iterables to an ndarray."""
    if isinstance(values, np.ndarray) or hasattr(values, "__array__"):
        return np.asarray(values)
    if np.isscalar(values):
        return np.asarray(values)
    try:
        return np.asarray(list(values))
    except TypeError:
        return np.asarray(values)


def _validate_binary_inputs(
    y_true: Iterable,
    y_proba: Iterable,
    *,
    probability_name: str = "y_proba",
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional binary outcomes and probabilities."""
    y_raw = _materialize_array(y_true)
    p_raw = _materialize_array(y_proba)

    if y_raw.ndim != 1 or p_raw.ndim != 1:
        raise ValueError("y_true and probability arrays must be one-dimensional")
    if len(y_raw) == 0:
        raise ValueError("y_true and probability arrays must not be empty")
    if len(y_raw) != len(p_raw):
        raise ValueError("y_true and probability arrays must have equal length")

    try:
        y_float = y_raw.astype(float)
        p = p_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and probabilities must be numeric") from exc

    if not np.all(np.isfinite(y_float)) or not np.all(np.isfinite(p)):
        raise ValueError("y_true and probabilities must contain only finite values")
    if not np.all(np.isin(y_float, (0.0, 1.0))):
        raise ValueError("y_true must contain only binary values 0 and 1")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"{probability_name} must lie in the closed interval [0, 1]")

    return y_float.astype(np.int8), p


def _validate_thresholds(thresholds: Iterable) -> np.ndarray:
    """Validate event-risk thresholds, preserving their supplied order."""
    try:
        values = _materialize_array(thresholds).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must be numeric") from exc
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("thresholds must contain only finite values")
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("event-risk thresholds must be strictly between 0 and 1")
    return values


def _bin_edges(probabilities: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if (
        isinstance(n_bins, (bool, np.bool_))
        or not isinstance(n_bins, (int, np.integer))
        or n_bins < 1
    ):
        raise ValueError("n_bins must be a positive integer")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        quantiles = np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1))
        # Include the full probability scale and collapse tied quantiles.  This
        # can yield fewer than n_bins occupied intervals, which is preferable
        # to manufacturing zero-width bins.
        return np.unique(np.concatenate(([0.0], quantiles[1:-1], [1.0])))
    raise ValueError("strategy must be either 'uniform' or 'quantile'")


def calibration_curve_table(
    y_true: Iterable,
    y_proba: Iterable,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Return a transparent reliability table for out-of-sample predictions.

    Only occupied bins are returned.  ``calibration_gap`` is observed event
    rate minus mean predicted risk.  ``weight`` is the bin's share of all
    observations and is used by :func:`calibration_summary` to calculate ECE.

    This table is a decision-analytic diagnostic, not evidence of realised
    benefit to patients.
    """
    y, p = _validate_binary_inputs(y_true, y_proba)
    edges = _bin_edges(p, n_bins, strategy)
    # side="right" puts an exact upper-bound value in the next interval; clip
    # ensures p == 1 belongs to the final interval.
    bin_id = np.searchsorted(edges, p, side="right") - 1
    bin_id = np.clip(bin_id, 0, len(edges) - 2)

    rows: list[dict[str, float | int]] = []
    total = len(y)
    for idx in range(len(edges) - 1):
        selected = bin_id == idx
        count = int(np.sum(selected))
        if count == 0:
            continue
        mean_predicted = float(np.mean(p[selected]))
        observed_rate = float(np.mean(y[selected]))
        gap = observed_rate - mean_predicted
        rows.append(
            {
                "bin_index": idx + 1,
                "bin_lower": float(edges[idx]),
                "bin_upper": float(edges[idx + 1]),
                "n": count,
                "weight": count / total,
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "calibration_gap": gap,
                "absolute_calibration_gap": abs(gap),
            }
        )

    return pd.DataFrame(rows)


def _logistic_recalibration(
    y: np.ndarray,
    p: np.ndarray,
    clip_eps: float,
) -> tuple[float, float, str]:
    """Estimate recalibration coefficients and return an identification status."""
    if not 0.0 < clip_eps < 0.5:
        raise ValueError("clip_eps must be strictly between 0 and 0.5")

    p_clipped = np.clip(p, clip_eps, 1.0 - clip_eps)
    logits = np.log(p_clipped / (1.0 - p_clipped))
    # Both outcome classes and variation in predicted risk are required to
    # identify an intercept and a slope separately.
    if np.unique(y).size < 2:
        return float("nan"), float("nan"), "one_class"
    if np.ptp(logits) <= np.finfo(float).eps:
        return float("nan"), float("nan"), "constant_predictions"

    # In a one-predictor logistic model, non-overlapping (or just-touching)
    # class ranges imply complete or quasi-complete separation.  The
    # unpenalised maximum-likelihood coefficients are then not finite; a
    # numerical solver may nevertheless stop at arbitrary large values.
    logits_0 = logits[y == 0]
    logits_1 = logits[y == 1]
    separated = (
        np.max(logits_0) <= np.min(logits_1)
        or np.max(logits_1) <= np.min(logits_0)
    )
    if separated:
        return float("nan"), float("nan"), "complete_or_quasi_separation"

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=10_000,
        tol=1e-12,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        model.fit(logits.reshape(-1, 1), y)

    did_not_converge = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    if did_not_converge:
        return float("nan"), float("nan"), "nonconvergence"

    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    if not np.isfinite(intercept) or not np.isfinite(slope):
        return float("nan"), float("nan"), "nonfinite_coefficients"
    return intercept, slope, "ok"


def calibration_summary(
    y_true: Iterable,
    y_proba: Iterable,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    clip_eps: float = 1e-6,
) -> dict[str, float | int | str]:
    """Summarise calibration of out-of-sample event probabilities.

    The returned mapping contains Brier score, unpenalised logistic
    recalibration intercept and slope based on clipped probability logits,
    expected calibration error (ECE), observed and expected event rates, and
    their ratio.  ECE is the sample-size-weighted absolute bin calibration
    gap using the requested binning strategy.  ``recalibration_status`` makes
    non-identifiability or solver nonconvergence explicit rather than silently
    reporting arbitrary large coefficients.

    If the outcome has one class only, predictions have no variation, or the
    classes are completely/quasi-completely separated on the prediction-logit
    scale, a finite recalibration intercept and slope are not identifiable and
    are returned as ``NaN``.  If expected risk is zero, the observed/expected
    ratio is returned as ``NaN``.

    These measures are calibration diagnostics and decision-analytic proxies;
    they do not demonstrate realised improvement in patient outcomes.
    """
    y, p = _validate_binary_inputs(y_true, y_proba)
    curve = calibration_curve_table(y, p, n_bins=n_bins, strategy=strategy)
    intercept, slope, recalibration_status = _logistic_recalibration(
        y, p, clip_eps
    )

    observed_rate = float(np.mean(y))
    expected_rate = float(np.mean(p))
    expected_events = float(np.sum(p))
    observed_to_expected = (
        observed_rate / expected_rate if expected_rate > 0.0 else float("nan")
    )
    ece = float(
        np.sum(curve["weight"] * curve["absolute_calibration_gap"])
    )

    return {
        "n": int(len(y)),
        "observed_events": int(np.sum(y)),
        "expected_events": expected_events,
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "recalibration_status": recalibration_status,
        "ece": ece,
        "observed_rate": observed_rate,
        "expected_rate": expected_rate,
        "observed_to_expected_ratio": float(observed_to_expected),
    }


def _net_benefit_arrays(
    y: np.ndarray,
    predictions: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Vectorised net benefit for a prediction matrix (n x thresholds)."""
    positive = y.astype(bool)[:, None]
    tp = np.sum(predictions & positive, axis=0)
    fp = np.sum(predictions & ~positive, axis=0)
    odds = thresholds / (1.0 - thresholds)
    return tp / len(y) - fp / len(y) * odds


def paired_bootstrap_delta_net_benefit(
    y_true: Iterable,
    y_proba_model: Iterable,
    y_proba_comparator: Iterable,
    thresholds: Iterable,
    *,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Paired bootstrap difference in net benefit for two models.

    The same patient indices are resampled for both models on every bootstrap
    iteration.  Thresholds must have been selected externally (for example,
    from clinical policy or training/validation data), never by optimising the
    evaluated test or OOF outcomes.

    Net benefit is a decision-analytic proxy under an explicit threshold-odds
    trade-off.  It is not evidence of realised clinical impact.
    """
    # Materialise once so a one-pass y_true iterator can be validated against
    # both paired probability vectors.
    y_values = _materialize_array(y_true)
    y, p_model = _validate_binary_inputs(
        y_values, y_proba_model, probability_name="y_proba_model"
    )
    y_comparator, p_comparator = _validate_binary_inputs(
        y_values, y_proba_comparator, probability_name="y_proba_comparator"
    )
    # The second validation call also protects against future input coercion
    # changes; both copies must represent the identical paired outcomes.
    if not np.array_equal(y, y_comparator):
        raise ValueError("the two models must be evaluated on identical outcomes")
    threshold_values = _validate_thresholds(thresholds)
    if (
        isinstance(n_boot, (bool, np.bool_))
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 1
    ):
        raise ValueError("n_boot must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    pred_model = p_model[:, None] >= threshold_values[None, :]
    pred_comparator = p_comparator[:, None] >= threshold_values[None, :]
    nb_model = _net_benefit_arrays(y, pred_model, threshold_values)
    nb_comparator = _net_benefit_arrays(y, pred_comparator, threshold_values)
    delta = nb_model - nb_comparator

    rng = np.random.default_rng(seed)
    bootstrap_delta = np.empty((n_boot, len(threshold_values)), dtype=float)
    n = len(y)
    for iteration in range(n_boot):
        sampled = rng.integers(0, n, size=n)
        y_boot = y[sampled]
        nb_model_boot = _net_benefit_arrays(
            y_boot, pred_model[sampled], threshold_values
        )
        nb_comparator_boot = _net_benefit_arrays(
            y_boot, pred_comparator[sampled], threshold_values
        )
        bootstrap_delta[iteration] = nb_model_boot - nb_comparator_boot

    alpha = (1.0 - confidence_level) / 2.0
    ci_low = np.quantile(bootstrap_delta, alpha, axis=0)
    ci_high = np.quantile(bootstrap_delta, 1.0 - alpha, axis=0)
    return pd.DataFrame(
        {
            "threshold": threshold_values,
            "net_benefit_model": nb_model,
            "net_benefit_comparator": nb_comparator,
            "delta_net_benefit": delta,
            "delta_ci_low": ci_low,
            "delta_ci_high": ci_high,
            "n_bootstrap": int(n_boot),
        }
    )


def decision_curve(
    y_true: Iterable,
    y_proba: Iterable,
    thresholds: Iterable,
    *,
    y_proba_comparator: Iterable | None = None,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Calculate model, treat-all, and treat-none net benefit.

    ``thresholds`` are event-risk thresholds and must be supplied externally;
    this function deliberately provides no test-set threshold optimisation.
    If ``y_proba_comparator`` is supplied, paired bootstrap differences
    (model minus comparator) and percentile confidence intervals are appended.

    Decision curves quantify a decision-analytic proxy.  They do not establish
    that deployment changes treatment decisions or improves patient outcomes.
    """
    y, p = _validate_binary_inputs(y_true, y_proba)
    threshold_values = _validate_thresholds(thresholds)
    predictions = p[:, None] >= threshold_values[None, :]
    nb_model = _net_benefit_arrays(y, predictions, threshold_values)

    prevalence = float(np.mean(y))
    odds = threshold_values / (1.0 - threshold_values)
    result = pd.DataFrame(
        {
            "threshold": threshold_values,
            "net_benefit_model": nb_model,
            "net_benefit_treat_all": prevalence - (1.0 - prevalence) * odds,
            "net_benefit_treat_none": np.zeros(len(threshold_values)),
        }
    )

    if y_proba_comparator is not None:
        paired = paired_bootstrap_delta_net_benefit(
            y,
            p,
            y_proba_comparator,
            threshold_values,
            n_boot=n_boot,
            confidence_level=confidence_level,
            seed=seed,
        )
        for column in (
            "net_benefit_comparator",
            "delta_net_benefit",
            "delta_ci_low",
            "delta_ci_high",
            "n_bootstrap",
        ):
            result[column] = paired[column].to_numpy()

    return result


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def clinical_impact_table(
    y_true: Iterable,
    y_proba: Iterable,
    thresholds: Iterable,
) -> pd.DataFrame:
    """Tabulate classification consequences at external risk thresholds.

    Thresholds must be prespecified using clinical policy or independent
    training/validation data.  They are never optimised here on the evaluated
    outcomes.  Event is coded as ``y_true == 1`` and high risk as predicted
    event probability greater than or equal to the threshold.

    Per-1000 quantities use the entire evaluated population as denominator,
    not only patients with events.  The table is a decision-analytic proxy for
    possible workflow consequences; it is not evidence of realised patient
    impact, treatment effectiveness, or clinical utility after deployment.
    """
    y, p = _validate_binary_inputs(y_true, y_proba)
    threshold_values = _validate_thresholds(thresholds)
    n = len(y)
    event = y == 1
    rows: list[dict[str, float | int]] = []

    for threshold in threshold_values:
        high_risk = p >= threshold
        tp = int(np.sum(high_risk & event))
        fp = int(np.sum(high_risk & ~event))
        tn = int(np.sum(~high_risk & ~event))
        fn = int(np.sum(~high_risk & event))
        high_risk_n = tp + fp
        threshold_odds = threshold / (1.0 - threshold)

        rows.append(
            {
                "threshold": float(threshold),
                "n": n,
                "events_n": int(np.sum(event)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": _safe_ratio(tp, tp + fn),
                "specificity": _safe_ratio(tn, tn + fp),
                "ppv": _safe_ratio(tp, tp + fp),
                "npv": _safe_ratio(tn, tn + fn),
                "high_risk_n": high_risk_n,
                "high_risk_pct": 100.0 * high_risk_n / n,
                "events_detected": tp,
                "events_missed": fn,
                "tp_per_1000": 1000.0 * tp / n,
                "fp_per_1000": 1000.0 * fp / n,
                "tn_per_1000": 1000.0 * tn / n,
                "fn_per_1000": 1000.0 * fn / n,
                "high_risk_per_1000": 1000.0 * high_risk_n / n,
                "events_detected_per_1000": 1000.0 * tp / n,
                "events_missed_per_1000": 1000.0 * fn / n,
                "net_benefit": tp / n - fp / n * threshold_odds,
            }
        )

    return pd.DataFrame(rows)

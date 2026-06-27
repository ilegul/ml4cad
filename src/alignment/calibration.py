"""
src/alignment/calibration.py
────────────────────────────
Brier score and calibration curves for the alignment models (paper-alignment).

For a given cohort/feature-set, this evaluates every model (including the
ENSEMBLE) under both schemes and returns the Brier score, plus a helper to draw
calibration (reliability) curves. Probabilities come from the leakage-free
pipeline: OOF probabilities for scheme A, test probabilities for scheme B.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE
from src.alignment.evaluation import evaluate_cv, evaluate_holdout


def brier_table(X, y, models, sampler="RandomOverSampler",
                seed=RANDOM_STATE) -> pd.DataFrame:
    """
    Brier score for every model under scheme A (OOF) and scheme B (test).
    Returns a tidy DataFrame: model x scheme x brier.
    """
    y_arr = np.asarray(pd.Series(np.asarray(y)))
    rows = []
    for m in models:
        res_a = evaluate_cv(X, y, m, sampler=sampler, seed=seed)
        rows.append({"model": m, "scheme": "A_5fold_cv",
                     "brier": brier_score_loss(y_arr, res_a["oof_proba"])})
        res_b = evaluate_holdout(X, y, m, sampler=sampler, seed=seed)
        rows.append({"model": m, "scheme": "B_60_20_20",
                     "brier": brier_score_loss(res_b["test_y"],
                                               res_b["test_proba"])})
    return pd.DataFrame(rows)


def calibration_points(y_true, y_proba, n_bins=10):
    """Return (mean_predicted, fraction_positive, brier) for a reliability curve."""
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos, brier_score_loss(y_true, y_proba)


def plot_calibration(ax, curves, title):
    """
    Draw reliability curves on ``ax``.
    ``curves`` is a list of (label, y_true, y_proba) tuples.
    """
    for label, y_true, y_proba in curves:
        mean_pred, frac_pos, brier = calibration_points(y_true, y_proba)
        ax.plot(mean_pred, frac_pos, "o-", label=f"{label} (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction (class 1)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax

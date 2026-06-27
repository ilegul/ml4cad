"""
src/alignment/cohorts10.py
──────────────────────────
Strict cohort at an arbitrary horizon (paper-alignment).

The repo's ``build_dataset`` produces the strict cohort at 7 years only. The
paper reports results at multiple horizons, so here we rebuild the *strict*
cohort with the **same logic** at any horizon (used for 7 and 10 years):

    cvd_within  = event_cvd & time_days <= horizon
    event_free  = time_days >= horizon & ~cvd_within
    strict_mask = cvd_within | event_free            (censored-before-horizon dropped)

The classification target follows the *paper's* "survived vs cardiac death"
framing.  To stay numerically identical to the repo's existing target we keep
the repo convention ``y{h} = 1 -> CVD death within horizon`` (positive class =
event) and additionally expose ``survive{h} = 1 - y{h}`` (1 = survived beyond
the horizon), which is the quantity the paper's composite indicator predicts.

This module only READS ``cohort_full.parquet`` and derives in-memory frames; it
never rewrites any cohort file.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import COHORT_FULL_FILE
from src.features.feature_engineering import add_derived_features


def load_full() -> pd.DataFrame:
    """Load the full cohort produced by build_dataset (read-only)."""
    return pd.read_parquet(COHORT_FULL_FILE)


def build_strict(full_df: pd.DataFrame, horizon_years: float) -> pd.DataFrame:
    """
    Build the strict cohort at ``horizon_years`` from the full cohort.

    Adds, for horizon h (in whole years used as the column suffix):
      - ``y{h}``       : 1 = CVD death within h years, 0 = event-free at h years
      - ``survive{h}`` : 1 - y{h}  (1 = survived beyond h, paper framing)
    Derived thyroid features are added via ``add_derived_features``.
    """
    h = int(round(horizon_years))
    horizon_days = horizon_years * 365.25

    df = full_df.copy()
    cvd_within = (df["event_cvd"] == 1) & (df["time_days"] <= horizon_days)
    event_free = (df["time_days"] >= horizon_days) & (~cvd_within)
    strict_mask = cvd_within | event_free

    strict = df[strict_mask].copy()
    ycol = f"y{h}"
    strict[ycol] = np.nan
    strict.loc[cvd_within[strict_mask], ycol] = 1
    strict.loc[event_free[strict_mask], ycol] = 0
    strict[ycol] = strict[ycol].astype(int)
    strict[f"survive{h}"] = (1 - strict[ycol]).astype(int)

    strict = add_derived_features(strict)
    return strict


def summarize(strict_df: pd.DataFrame, horizon_years: float) -> dict:
    """Return N, n_events (CVD deaths), prevalence for a strict cohort."""
    h = int(round(horizon_years))
    ycol = f"y{h}"
    n = len(strict_df)
    n_events = int(strict_df[ycol].sum())
    return {
        "horizon_years": h,
        "N": n,
        "n_events_cvd": n_events,
        "n_survivors": n - n_events,
        "prevalence": round(n_events / n, 4) if n else float("nan"),
    }


def build_strict_cohorts(horizons=(7, 10)) -> dict:
    """
    Build strict cohorts for the requested horizons.
    Returns {horizon_int: strict_df}.
    """
    full = load_full()
    return {int(round(h)): build_strict(full, h) for h in horizons}

"""
src/alignment/features_align.py
───────────────────────────────
Feature-set views used for the paper-alignment notebook.

This reuses the canonical feature sets from ``configs.config.FEATURE_SETS`` and
the derived-feature engineering from
``src.features.feature_engineering.add_derived_features`` — it does NOT redefine
them. The only alignment-specific adjustments are:

1.  **5-dummy thyroid-state encoding.**  The paper-alignment uses the eutiroideo
    (Euthyroid) state as the *reference* category, so the ``Euthyroid`` dummy is
    dropped from any feature set that carries the full block of mutually
    exclusive thyroid-state indicators (``CV17_THY_STATES``,
    ``CV17_THY_CONT_STATES`` and ``CV17_THY_CONT_STATES_RATIO``). This mirrors
    what ``survival_cohort._cox_covariates`` already does for the Cox
    models, and produces 5 dummies (SCH, SCT, Low_T3, Hypothyroid, Hyperthyroid)
    with Euthyroid as the implicit reference.

2.  **Imputation instead of row-dropping.**  ``extract_Xy_align`` keeps rows with
    missing *features* (only the target must be present); missingness is handled
    downstream by a leakage-free imputer fitted on the training fold. The
    repo-wide ``extract_Xy`` drops those rows instead — we keep them so the
    cohort N stays constant across feature sets.

Note (matching the thesis constraint): CV17 stays at 17 cardiac variables.
Creatinine and eGFR remain EXCLUDED (too many missing). The paper used 18
variables including creatinine — this is a documented difference, not added.
"""

import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FEATURE_SETS,
    CONTINUOUS_FEATURES,
    canonical_feature_set_name,
)

# The full block of mutually exclusive thyroid-state dummies.
_THYROID_STATE_DUMMIES = {
    "Euthyroid", "SCH", "SCT", "Low_T3", "Hypothyroid", "Hyperthyroid",
}
_REFERENCE_STATE = "Euthyroid"


def get_align_feature_set(name: str) -> list:
    """
    Return the alignment view of a feature set.

    Identical to ``configs.config.FEATURE_SETS[name]`` except that, when the set
    carries the full thyroid-state dummy block, the Euthyroid reference dummy is
    dropped (5-dummy encoding, eutiroideo = reference).
    """
    canonical = canonical_feature_set_name(name)
    if canonical not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {name}. Available: {list(FEATURE_SETS)}")
    feats = list(FEATURE_SETS[canonical])
    has_full_block = _THYROID_STATE_DUMMIES.issubset(set(feats))
    if has_full_block and _REFERENCE_STATE in feats:
        feats = [f for f in feats if f != _REFERENCE_STATE]
    return feats


def get_all_align_feature_set_names() -> list:
    """All feature-set names, in canonical order."""
    return list(FEATURE_SETS.keys())


def describe_feature_sets() -> pd.DataFrame:
    """
    Tabulate every alignment feature set: name, #features, and the explicit
    feature list. Used in the notebook's verification cell.
    """
    rows = []
    for name in FEATURE_SETS:
        feats = get_align_feature_set(name)
        rows.append({
            "feature_set": name,
            "n_features": len(feats),
            "features": ", ".join(feats),
        })
    return pd.DataFrame(rows)


def continuous_in(cols) -> list:
    """Continuous features present among ``cols`` (subset of CONTINUOUS_FEATURES)."""
    cols = list(cols)
    return [c for c in CONTINUOUS_FEATURES if c in cols]


def extract_Xy_align(df: pd.DataFrame, feature_set_name: str,
                     target_col: str) -> tuple:
    """
    Extract (X, y) for a feature set, keeping rows with missing *features*.

    Only rows with a missing ``target_col`` are dropped. Missing features are
    left as NaN to be imputed inside the leakage-free pipeline. Returns
    (X: DataFrame, y: Series[int]).
    """
    feats = get_align_feature_set(feature_set_name)
    cols = [c for c in feats if c in df.columns]
    missing = set(feats) - set(cols)
    if missing:
        print(f"  [WARN] Missing columns for {feature_set_name}: {missing}")

    subset = df[df[target_col].notna()].copy()
    X = subset[cols]
    y = subset[target_col].astype(int)
    return X, y

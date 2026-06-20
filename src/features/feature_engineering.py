"""
src/features/feature_engineering.py
───────────────────────────────────
Feature engineering and definition of the 8 feature sets.
Pipeline.md section 4.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    CARDIO_17, THYROID_RAW9, FEATURE_SETS,
    CONTINUOUS_FEATURES, THYROID_ORD_MAP,
)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features:
    - fT3_fT4_ratio = fT3 / fT4
    - Thyroid_abnormal = (Euthyroid == 0)
    - thyroid_state (single mutually exclusive category)
    - thyroid_ord (ordinal map)
    """
    df = df.copy()

    # fT3 / fT4 ratio (handle division by zero)
    df["fT3_fT4_ratio"] = np.where(
        df["fT4"] != 0,
        df["fT3"] / df["fT4"],
        np.nan
    )

    # Thyroid abnormal (binary)
    df["Thyroid_abnormal"] = (df["Euthyroid"] == 0).astype(int)

    # thyroid_state -- single category (the 6 states are mutually exclusive)
    thyroid_cats = ["Euthyroid", "SCH", "SCT", "Low_T3",
                    "Hypothyroid", "Hyperthyroid"]
    def _get_thyroid_state(row):
        for cat in thyroid_cats:
            if cat in row.index and row[cat] == 1:
                return cat
        return "Unknown"
    df["thyroid_state"] = df.apply(_get_thyroid_state, axis=1)

    # thyroid_ord -- ordinal map
    df["thyroid_ord"] = df["thyroid_state"].map(THYROID_ORD_MAP).fillna(0).astype(int)

    return df


def get_feature_set(name: str) -> list:
    """Return the feature list for a given set name."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {name}. "
                         f"Available: {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[name]


def extract_Xy(df: pd.DataFrame, feature_set_name: str,
               target_col: str = "y7") -> tuple:
    """
    Extract X and y from a dataframe for a given feature set.
    Drops rows with NaN in target or features.
    Returns (X: pd.DataFrame, y: pd.Series).
    """
    features = get_feature_set(feature_set_name)
    cols = [c for c in features if c in df.columns]
    missing = set(features) - set(cols)
    if missing:
        print(f"  [WARN] Missing features in {feature_set_name}: {missing}")

    subset = df[cols + [target_col]].dropna()
    X = subset[cols]
    y = subset[target_col].astype(int)
    return X, y


def extract_Xy_survival(df: pd.DataFrame, feature_set_name: str,
                        time_col: str = "time_years",
                        event_col: str = "event_cvd") -> tuple:
    """
    Extract X and structured array (event, time) for scikit-survival.
    Returns (X: pd.DataFrame, y_surv: np.structured_array).
    """
    features = get_feature_set(feature_set_name)
    cols = [c for c in features if c in df.columns]

    subset = df[cols + [time_col, event_col]].dropna()
    X = subset[cols]

    # Structured array for scikit-survival
    y_surv = np.array(
        [(bool(e), t) for e, t in zip(subset[event_col], subset[time_col])],
        dtype=[("event", bool), ("time", float)]
    )
    return X, y_surv


def scale_continuous(X: pd.DataFrame,
                     scaler: StandardScaler = None,
                     fit: bool = True) -> tuple:
    """
    Standardise continuous features in X.
    Returns (X_scaled: pd.DataFrame, scaler: StandardScaler).
    """
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in X.columns]
    if not cont_cols:
        return X, scaler

    X = X.copy()
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X[cont_cols] = scaler.fit_transform(X[cont_cols])
    else:
        X[cont_cols] = scaler.transform(X[cont_cols])

    return X, scaler


def get_all_feature_set_names() -> list:
    """Return a list of all feature set names."""
    return list(FEATURE_SETS.keys())


def prepare_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the cohort dataframe by adding derived features
    and reporting missingness.
    """
    df = add_derived_features(df)

    # Report missing in main features
    all_feats = set()
    for fs in FEATURE_SETS.values():
        all_feats.update(fs)

    available = [f for f in all_feats if f in df.columns]
    miss = df[available].isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        print(f"  [INFO] Missing values in features:")
        for col, n in miss.items():
            print(f"    {col}: {n} ({100*n/len(df):.1f}%)")
    else:
        print("  [INFO] No missing values in main features.")

    return df

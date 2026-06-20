"""
src/preprocessing/build_dataset.py
──────────────────────────────────
Load raw data, merge, build cohorts.
Pipeline.md sections 2-3: handles right-censoring, builds y7 target,
produces strict, competing, and survival cohorts.

Resumable: if parquet files exist, the script skips them.
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    RAW_DATA_FILE, DATA_PRELIEVO_FILE, CREATINA_FILE,
    COHORT_STRICT_FILE, COHORT_COMPETING_FILE,
    COHORT_SURVIVAL_FILE, COHORT_FULL_FILE,
    COHORT_STRICT_CSV, COHORT_COMPETING_CSV,
    COHORT_SURVIVAL_CSV, COHORT_FULL_CSV,
    DATA_QUALITY_CSV, RENAME, DATE_COLS, HORIZON_DAYS,
)


# ─── Utility functions ────────────────────────────────────────────────

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names (keep internal \\n)."""
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw_data() -> pd.DataFrame:
    """Load raw_data.xlsx, filter non-null Number, cast to int."""
    print("[build_dataset] Loading raw_data.xlsx ...")
    df = pd.read_excel(RAW_DATA_FILE, engine="openpyxl")
    df = _clean_column_names(df)
    n_before = len(df)
    df = df.dropna(subset=["Number"])
    df["Number"] = df["Number"].astype(int)
    print(f"  raw_data: {n_before} -> {len(df)} rows (after non-null Number filter)")
    return df


def load_data_prelievo() -> pd.DataFrame:
    """Load data_prelievo.xlsx with enrolment dates."""
    print("[build_dataset] Loading data_prelievo.xlsx ...")
    df = pd.read_excel(DATA_PRELIEVO_FILE, engine="openpyxl")
    df = _clean_column_names(df)
    df["Number"] = df["Number"].astype(int)
    print(f"  data_prelievo: {len(df)} rows")
    return df[["Number", "Data prelievo"]]


def load_creatina() -> pd.DataFrame:
    """Load creatina_more_columns.xlsx."""
    print("[build_dataset] Loading creatina_more_columns.xlsx ...")
    df = pd.read_excel(CREATINA_FILE, engine="openpyxl")
    df = _clean_column_names(df)
    df["Number"] = df["Number"].astype(int)
    keep = ["Number", "Total cholesterol", "HDL", "LDL",
            "Triglycerides", "Creatinina"]
    # Keep only columns that exist
    keep = [c for c in keep if c in df.columns]
    print(f"  creatina: {len(df)} rows, columns: {keep}")
    return df[keep]


def merge_datasets(raw: pd.DataFrame,
                   prelievo: pd.DataFrame,
                   creatina: pd.DataFrame) -> pd.DataFrame:
    """Merge on Number (left join)."""
    print("[build_dataset] Merging datasets ...")
    df = raw.merge(prelievo, on="Number", how="left")
    df = df.merge(creatina, on="Number", how="left")
    print(f"  After merge: {len(df)} rows, {len(df.columns)} columns")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using the RENAME dictionary."""
    df = df.rename(columns=RENAME)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime."""
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    # Also handle internal / renamed date columns
    for col in ["Follow Up Data", "Data of death"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


def apply_data_quality_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply explicit patient-level corrections without modifying raw files.

    Corrections are logged to reports/data_quality_corrections.csv:
    - Number 6850 has conflicting thyroid flags. TSH=4.76 is coherent with
      SCH, not hyperthyroidism, so keep SCH=1 and set Hyperthyroid=0.
    - Number 7286 has a death date but Total mortality=0. Set mortality=1.
    """
    df = df.copy()
    corrections = []

    def _capture(number, issue, before_cols, updates, rationale):
        mask = df["Number"] == number
        if not mask.any():
            corrections.append({
                "Number": number,
                "issue": issue,
                "status": "not_found",
                "rationale": rationale,
            })
            return

        before = df.loc[mask, before_cols].iloc[0].to_dict()
        for col, value in updates.items():
            if col in df.columns:
                df.loc[mask, col] = value
        after = df.loc[mask, before_cols].iloc[0].to_dict()

        corrections.append({
            "Number": number,
            "issue": issue,
            "status": "corrected",
            "before": before,
            "updates": updates,
            "after": after,
            "rationale": rationale,
        })

    thyroid_cols = [
        "Number", "TSH", "fT3", "fT4", "Euthyroid", "SCH", "SCT",
        "Low_T3", "Hypothyroid", "Hyperthyroid",
    ]
    thyroid_cols = [c for c in thyroid_cols if c in df.columns]
    _capture(
        6850,
        "Conflicting thyroid flags: SCH=1 and Hyperthyroid=1",
        thyroid_cols,
        {"SCH": 1, "Hyperthyroid": 0},
        "TSH=4.76 is elevated and is more coherent with SCH than hyperthyroidism.",
    )

    mortality_cols = [
        "Number", "Data prelievo", "Follow Up Data", "Data of death",
        "Total mortality", "CVD Death", "Cause of death",
    ]
    mortality_cols = [c for c in mortality_cols if c in df.columns]
    _capture(
        7286,
        "Death date is present but Total mortality=0",
        mortality_cols,
        {"Total mortality": 1},
        "A populated death date should be consistent with Total mortality=1; CVD Death remains 0.",
    )

    DATA_QUALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(corrections).to_csv(DATA_QUALITY_CSV, index=False)
    print(f"[build_dataset] Data-quality corrections saved to {DATA_QUALITY_CSV}")
    return df


def compute_time_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-to-event and event indicators.
    Pipeline.md section 3:
        t_end = Data of death if present, otherwise Follow Up Data
        time_days = (t_end - Data prelievo).days, clipped to >= 0
        event_cvd  = (CVD Death == 1)
        event_death = (Total mortality == 1)
        event_noncvd = event_death & ~event_cvd
    """
    df = df.copy()

    # t_end: date of death if available, otherwise follow-up date
    df["t_end"] = df["Data of death"].fillna(df["Follow Up Data"])

    # time_days
    df["time_days"] = (df["t_end"] - df["Data prelievo"]).dt.days
    df["time_days"] = df["time_days"].clip(lower=0)
    df["time_years"] = df["time_days"] / 365.25

    # Events
    df["event_cvd"]    = (df["CVD Death"] == 1).astype(int)
    df["event_death"]  = (df["Total mortality"] == 1).astype(int)
    df["event_noncvd"] = ((df["event_death"] == 1) &
                          (df["event_cvd"] == 0)).astype(int)

    # A same-day CVD death is a valid event for the 7-year endpoint. For
    # survival models that require positive durations, shift only this
    # survival-analysis time to 1 day while retaining the original time_days.
    df["zero_day_cvd_event"] = ((df["event_cvd"] == 1) &
                                (df["time_days"] == 0)).astype(int)
    df["survival_time_days"] = df["time_days"].copy()
    df.loc[df["zero_day_cvd_event"] == 1, "survival_time_days"] = 1
    df["survival_time_years"] = df["survival_time_days"] / 365.25

    return df


def build_target_y7(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build classification target: CVD death within 7 years (y7).
    Pipeline.md section 3:
        y7 = 1 if event_cvd & time_days <= HORIZON_DAYS
        y7 = 0 if time_days >= HORIZON_DAYS & not(cvd within 7y)
        otherwise censored -> NaN (to be removed)
    """
    df = df.copy()

    cvd_within = (df["event_cvd"] == 1) & (df["time_days"] <= HORIZON_DAYS)
    event_free = (df["time_days"] >= HORIZON_DAYS) & (~cvd_within)

    df["y7"] = np.nan
    df.loc[cvd_within, "y7"]   = 1
    df.loc[event_free, "y7"]   = 0

    # Statistics
    n_pos   = int(cvd_within.sum())
    n_neg   = int(event_free.sum())
    n_cens  = int(df["y7"].isna().sum())
    print(f"  y7: positives={n_pos}, negatives={n_neg}, censored={n_cens}")

    return df


def build_cohorts(df: pd.DataFrame) -> dict:
    """
    Build cohorts:
    - strict: event + event-free only (censored & non-CVD deaths <7y removed)
    - competing: strict + non-CVD deaths <7y as negatives
    - survival: all patients with time_days > 0
    """
    df_full = df.copy()

    # -- Strict cohort (cause-specific) --
    cvd_within   = (df["event_cvd"] == 1) & (df["time_days"] <= HORIZON_DAYS)
    event_free   = (df["time_days"] >= HORIZON_DAYS) & (~cvd_within)
    strict_mask  = cvd_within | event_free
    df_strict    = df[strict_mask].copy()
    df_strict["y7"] = df_strict["y7"].astype(int)

    # -- Competing cohort --
    # Add non-CVD deaths <7y as negatives
    noncvd_early = ((df["event_noncvd"] == 1) &
                    (df["time_days"] < HORIZON_DAYS))
    competing_mask = strict_mask | noncvd_early
    df_competing = df[competing_mask].copy()
    # Non-CVD deaths <7y are negatives
    df_competing.loc[noncvd_early, "y7"] = 0
    df_competing["y7"] = df_competing["y7"].astype(int)

    # -- Survival cohort --
    # Keep positive follow-up and include same-day CVD deaths as events at
    # survival_time_days=1. Same-day censored records still add no risk time.
    survival_mask = ((df["time_days"] > 0) |
                     ((df["event_cvd"] == 1) & (df["time_days"] == 0)))
    df_surv = df[survival_mask].copy()
    df_surv["time_days_original"] = df_surv["time_days"]
    df_surv["time_years_original"] = df_surv["time_years"]
    df_surv["time_days"] = df_surv["survival_time_days"]
    df_surv["time_years"] = df_surv["survival_time_years"]

    print(f"\n[build_dataset] Cohorts built:")
    print(f"  Strict:    N={len(df_strict)}, "
          f"positives={int(df_strict['y7'].sum())} "
          f"({100*df_strict['y7'].mean():.1f}%)")
    print(f"  Competing: N={len(df_competing)}, "
          f"positives={int(df_competing['y7'].sum())} "
          f"({100*df_competing['y7'].mean():.1f}%)")
    print(f"  Survival:  N={len(df_surv)}, "
          f"CVD events={int(df_surv['event_cvd'].sum())}")

    return {
        "full":      df_full,
        "strict":    df_strict,
        "competing": df_competing,
        "survival":  df_surv,
    }


def save_cohorts(cohorts: dict):
    """Save analysis cohorts as parquet plus CSV audit copies."""
    outputs = {
        "full": (COHORT_FULL_FILE, COHORT_FULL_CSV),
        "strict": (COHORT_STRICT_FILE, COHORT_STRICT_CSV),
        "competing": (COHORT_COMPETING_FILE, COHORT_COMPETING_CSV),
        "survival": (COHORT_SURVIVAL_FILE, COHORT_SURVIVAL_CSV),
    }
    for name, (parquet_path, csv_path) in outputs.items():
        cohorts[name].to_parquet(parquet_path, index=False)
        cohorts[name].to_csv(csv_path, index=False)


# ─── Entry point ────────────────────────────────────────────────────────

def run(force: bool = False):
    """
    Run the dataset building pipeline.
    If output files exist and force=False, skip.
    """
    output_files = [COHORT_STRICT_FILE, COHORT_COMPETING_FILE,
                    COHORT_SURVIVAL_FILE, COHORT_FULL_FILE]

    if not force and all(f.exists() for f in output_files):
        print("[build_dataset] Files already exist, skipping.")
        return {
            "full":      pd.read_parquet(COHORT_FULL_FILE),
            "strict":    pd.read_parquet(COHORT_STRICT_FILE),
            "competing": pd.read_parquet(COHORT_COMPETING_FILE),
            "survival":  pd.read_parquet(COHORT_SURVIVAL_FILE),
        }

    print("=" * 60)
    print("[build_dataset] Building dataset ...")
    print("=" * 60)

    # 1. Load
    raw      = load_raw_data()
    prelievo = load_data_prelievo()
    creatina = load_creatina()

    # 2. Merge
    df = merge_datasets(raw, prelievo, creatina)

    # 3. Rename
    df = rename_columns(df)

    # 4. Parse dates
    df = parse_dates(df)

    # 5. Explicit patient-level data-quality corrections
    df = apply_data_quality_corrections(df)

    # 6. Time-to-event
    df = compute_time_event(df)

    # 7. Target y7
    df = build_target_y7(df)

    # 8. Cohorts
    cohorts = build_cohorts(df)

    # 9. Save
    print("\n[build_dataset] Saving parquet and CSV files ...")
    save_cohorts(cohorts)
    print("[build_dataset] Done.")

    return cohorts


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)

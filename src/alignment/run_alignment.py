"""
src/alignment/run_alignment.py
──────────────────────────────
Orchestration / caching layer for the paper-alignment notebook.

All heavy computation lives here (not in the notebook). Each ``build_*`` function
computes a result table once and caches it under ``reports/alignment/``; on a
re-run it loads the cache unless ``force=True``. The notebook imports these
functions, calls them, and only *displays* the returned DataFrames + draws
figures.

Cohort: strict only. Horizons: 7 and 10 years. Schemes: A = 5-fold CV,
B = 60/20/20 holdout. Model focus for the ensemble analyses: the paper's
ENSEMBLE (LR+RF+AdaBoost soft-vote).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE, REPORTS_DIR
from src.alignment.cohorts10 import build_strict_cohorts, summarize
from src.alignment.features_align import (
    get_all_align_feature_set_names, extract_Xy_align,
)
from src.alignment.ensemble import ALIGN_MODELS
from src.alignment.evaluation import (
    evaluate_cv, evaluate_holdout,
    incremental_value_cv, incremental_value_holdout,
)
from src.alignment.ablation import run_ablation
from src.alignment.ml_indicator import compare_indicator

ALIGN_DIR = REPORTS_DIR / "alignment"
ALIGN_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = ["f1_macro", "roc_auc", "precision_1", "recall_1",
               "precision_0", "recall_0", "brier", "threshold"]


# ─── 1. Classification metrics table (all sets x models x schemes x horizons) ─

CLF_CSV = ALIGN_DIR / "classification_metrics.csv"


def build_classification_table(cohorts, models=None, horizons=(7, 10),
                               seed=RANDOM_STATE, force=False) -> pd.DataFrame:
    """
    Full classification table: feature_set x model x scheme x horizon with
    F1-macro / AUROC / precision / recall / Brier. Cached to CLF_CSV.
    """
    if CLF_CSV.exists() and not force:
        return pd.read_csv(CLF_CSV)

    models = models or ALIGN_MODELS
    fs_names = get_all_align_feature_set_names()
    rows = []
    for h in horizons:
        df = cohorts[int(h)]
        target = f"y{int(h)}"
        for fs in fs_names:
            X, y = extract_Xy_align(df, fs, target)
            for m in models:
                ra = evaluate_cv(X, y, m, seed=seed)
                rows.append({
                    "horizon": int(h), "feature_set": fs, "model": m,
                    "scheme": "A_5fold_cv",
                    **{k: ra[f"{k}_mean"] for k in METRIC_KEYS},
                    **{f"{k}_std": ra[f"{k}_std"] for k in METRIC_KEYS},
                })
                rb = evaluate_holdout(X, y, m, seed=seed)
                rows.append({
                    "horizon": int(h), "feature_set": fs, "model": m,
                    "scheme": "B_60_20_20",
                    **{k: rb[k] for k in METRIC_KEYS},
                })
                print(f"  [clf] h{h} {fs} {m} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CLF_CSV, index=False)
    return out


# ─── 2. Incremental value of thyroid vs CV17 (ensemble, paired bootstrap) ────

INCR_CSV = ALIGN_DIR / "incremental_thyroid.csv"


def build_incremental_table(cohorts, base="CV17", horizons=(7, 10),
                            model_name="ENSEMBLE", n_boot=1000,
                            seed=RANDOM_STATE, force=False) -> pd.DataFrame:
    """
    Paired Δ F1-macro and Δ AUROC (thyroid set vs CV17) with 95% bootstrap CI,
    for each thyroid set x horizon x scheme. Cached to INCR_CSV.
    """
    if INCR_CSV.exists() and not force:
        return pd.read_csv(INCR_CSV)

    thy_sets = [fs for fs in get_all_align_feature_set_names() if fs != base]
    rows = []
    for h in horizons:
        df = cohorts[int(h)]
        target = f"y{int(h)}"
        Xb, y = extract_Xy_align(df, base, target)
        for fs in thy_sets:
            Xt, _ = extract_Xy_align(df, fs, target)
            da = incremental_value_cv(Xb, Xt, y, model_name=model_name,
                                      n_boot=n_boot, seed=seed)
            db = incremental_value_holdout(Xb, Xt, y, model_name=model_name,
                                           n_boot=n_boot, seed=seed)
            for d in (da, db):
                rows.append({"horizon": int(h), "base": base,
                             "feature_set": fs, **d})
            print(f"  [incr] h{h} {fs} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(INCR_CSV, index=False)
    return out


# ─── 3. ML indicator in survival (Cox C-index + KM) ─────────────────────────

CINDEX_CSV = ALIGN_DIR / "ml_indicator_cindex.csv"
KM_CSV = ALIGN_DIR / "ml_indicator_km.csv"


def build_ml_indicator(cohorts, base="CV17", thy="CV17_THY26",
                       horizons=(7, 10), schemes=("A", "B"),
                       seed=RANDOM_STATE, force=False):
    """
    ML-indicator survival analysis: single-covariate Cox C-index for CV17 and
    CV17+thyroid, Δ C-index, and KM stratification (0.6 + median). Caches the
    C-index table and the KM-summary table. Returns (cindex_df, km_df).
    """
    if CINDEX_CSV.exists() and KM_CSV.exists() and not force:
        return pd.read_csv(CINDEX_CSV), pd.read_csv(KM_CSV)

    cidx_rows, km_rows = [], []
    for h in horizons:
        df = cohorts[int(h)]
        for scheme in schemes:
            res = compare_indicator(df, h, base_set=base, thy_set=thy,
                                    scheme=scheme, seed=seed)
            for tag in ("base", "thy"):
                cox = res[tag]["cox"]
                cidx_rows.append({
                    "horizon": int(h), "scheme": scheme,
                    "feature_set": res[tag]["feature_set"],
                    "c_index": cox["c_index"], "n": cox["n"],
                    "events": cox["events"],
                })
                for km_key in ("km_0.6", "km_median"):
                    km = res[tag][km_key]
                    km_rows.append({
                        "horizon": int(h), "scheme": scheme,
                        "feature_set": res[tag]["feature_set"],
                        "threshold_name": km["threshold_name"],
                        "threshold": km["threshold"],
                        "n_high": km["n_high"], "n_low": km["n_low"],
                        "surv_high": km["surv_at_horizon_high_predicted_survival"],
                        "surv_low": km["surv_at_horizon_low_predicted_survival"],
                        "logrank_p": km["logrank_p"],
                    })
            # delta row
            cidx_rows.append({
                "horizon": int(h), "scheme": scheme,
                "feature_set": f"DELTA({thy}-{base})",
                "c_index": res["delta_c_index"], "n": np.nan, "events": np.nan,
            })
            print(f"  [mlind] h{h} scheme {scheme} done", flush=True)

    cindex_df = pd.DataFrame(cidx_rows)
    km_df = pd.DataFrame(km_rows)
    cindex_df.to_csv(CINDEX_CSV, index=False)
    km_df.to_csv(KM_CSV, index=False)
    return cindex_df, km_df


# ─── 4. Ablation ranking (single + multi) per feature set, per horizon ───────

ABLATION_CSV = ALIGN_DIR / "ablation_ranking.csv"


def build_ablation(cohorts, horizons=(7, 10), model_name="ENSEMBLE",
                   seed=RANDOM_STATE, force=False) -> pd.DataFrame:
    """
    Knock-out ablation (single + multi-variable) for every feature set and
    horizon, on the ensemble. Cached to ABLATION_CSV.
    """
    if ABLATION_CSV.exists() and not force:
        return pd.read_csv(ABLATION_CSV)

    fs_names = get_all_align_feature_set_names()
    frames = []
    for h in horizons:
        df = cohorts[int(h)]
        target = f"y{int(h)}"
        for fs in fs_names:
            X, y = extract_Xy_align(df, fs, target)
            ab = run_ablation(X, y, fs, model_name=model_name, seed=seed)
            ab.insert(0, "horizon", int(h))
            frames.append(ab)
            print(f"  [abl] h{h} {fs} done", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(ABLATION_CSV, index=False)
    return out


# ─── Driver ─────────────────────────────────────────────────────────────────

def build_all(horizons=(7, 10), seed=RANDOM_STATE, n_boot=1000, force=False):
    """Build and cache every alignment result table."""
    cohorts = build_strict_cohorts(horizons)
    print("[run_alignment] cohort summary:")
    for h in horizons:
        print("  ", summarize(cohorts[int(h)], h))

    print("[run_alignment] 1/4 classification table ...", flush=True)
    build_classification_table(cohorts, horizons=horizons, seed=seed,
                               force=force)
    print("[run_alignment] 2/4 incremental table ...", flush=True)
    build_incremental_table(cohorts, horizons=horizons, n_boot=n_boot,
                            seed=seed, force=force)
    print("[run_alignment] 3/4 ML indicator ...", flush=True)
    build_ml_indicator(cohorts, horizons=horizons, seed=seed, force=force)
    print("[run_alignment] 4/4 ablation ...", flush=True)
    build_ablation(cohorts, horizons=horizons, seed=seed, force=force)
    print("[run_alignment] all caches built under", ALIGN_DIR, flush=True)


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_all(force=force)

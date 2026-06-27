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
from src.alignment.ensemble import ALIGN_MODELS, ENSEMBLE_NAMES
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

    Row-incremental: existing (horizon, feature_set, model, scheme) rows are kept
    untouched and only the missing combinations are computed. New models (e.g.
    extra ensembles) are therefore added without recomputing the cached ones.
    Pass force=True to recompute everything from scratch.
    """
    models = models or ALIGN_MODELS
    fs_names = get_all_align_feature_set_names()

    if CLF_CSV.exists() and not force:
        existing = pd.read_csv(CLF_CSV)
        done = {(int(r["horizon"]), r["feature_set"], r["model"], r["scheme"])
                for _, r in existing.iterrows()}
    else:
        existing = pd.DataFrame()
        done = set()

    rows = []
    for h in horizons:
        df = cohorts[int(h)]
        target = f"y{int(h)}"
        for fs in fs_names:
            X = y = None
            for m in models:
                need_a = (int(h), fs, m, "A_5fold_cv") not in done
                need_b = (int(h), fs, m, "B_60_20_20") not in done
                if not (need_a or need_b):
                    continue
                if X is None:
                    X, y = extract_Xy_align(df, fs, target)
                if need_a:
                    ra = evaluate_cv(X, y, m, seed=seed)
                    rows.append({
                        "horizon": int(h), "feature_set": fs, "model": m,
                        "scheme": "A_5fold_cv",
                        **{k: ra[f"{k}_mean"] for k in METRIC_KEYS},
                        **{f"{k}_std": ra[f"{k}_std"] for k in METRIC_KEYS},
                    })
                if need_b:
                    rb = evaluate_holdout(X, y, m, seed=seed)
                    rows.append({
                        "horizon": int(h), "feature_set": fs, "model": m,
                        "scheme": "B_60_20_20",
                        **{k: rb[k] for k in METRIC_KEYS},
                    })
                print(f"  [clf] h{h} {fs} {m} done", flush=True)

    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True,
                        sort=False)
        out.to_csv(CLF_CSV, index=False)
    else:
        out = existing
    return out


# ─── 2. Incremental value of thyroid vs CV17 (ensemble, paired bootstrap) ────

INCR_CSV = ALIGN_DIR / "incremental_thyroid.csv"


def _suffixed(path: Path, suffix: str) -> Path:
    """reports/alignment/foo.csv + 'BAR' -> reports/alignment/foo_BAR.csv."""
    return path if not suffix else path.with_name(
        f"{path.stem}_{suffix}{path.suffix}")


def build_incremental_table(cohorts, base="CV17", horizons=(7, 10),
                            model_name="ENSEMBLE", n_boot=1000,
                            seed=RANDOM_STATE, force=False,
                            out_suffix="") -> pd.DataFrame:
    """
    Paired Δ F1-macro and Δ AUROC (thyroid set vs CV17) with 95% bootstrap CI,
    for each thyroid set x horizon x scheme, on ``model_name``. Cached to
    INCR_CSV (or a suffixed file when ``out_suffix`` is given, e.g. the winner
    ensemble — so the paper-ENSEMBLE cache is left untouched).
    """
    csv = _suffixed(INCR_CSV, out_suffix)
    if csv.exists() and not force:
        return pd.read_csv(csv)

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
    out.insert(0, "model", model_name)
    out.to_csv(csv, index=False)
    return out


# ─── 3. ML indicator in survival (Cox C-index + KM) ─────────────────────────

CINDEX_CSV = ALIGN_DIR / "ml_indicator_cindex.csv"
KM_CSV = ALIGN_DIR / "ml_indicator_km.csv"


def build_ml_indicator(cohorts, base="CV17", thy="CV17_THY26",
                       horizons=(7, 10), schemes=("A", "B"),
                       model_name="ENSEMBLE",
                       seed=RANDOM_STATE, force=False, out_suffix=""):
    """
    ML-indicator survival analysis on ``model_name``: single-covariate Cox
    C-index for CV17 and CV17+thyroid, Δ C-index, and KM stratification
    (0.6 + median). Caches the C-index and KM-summary tables (suffixed when
    ``out_suffix`` is given). Returns (cindex_df, km_df).
    """
    cindex_csv = _suffixed(CINDEX_CSV, out_suffix)
    km_csv = _suffixed(KM_CSV, out_suffix)
    if cindex_csv.exists() and km_csv.exists() and not force:
        return pd.read_csv(cindex_csv), pd.read_csv(km_csv)

    cidx_rows, km_rows = [], []
    for h in horizons:
        df = cohorts[int(h)]
        for scheme in schemes:
            res = compare_indicator(df, h, base_set=base, thy_set=thy,
                                    scheme=scheme, model_name=model_name,
                                    seed=seed)
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
    cindex_df.insert(0, "model", model_name)
    km_df.insert(0, "model", model_name)
    cindex_df.to_csv(cindex_csv, index=False)
    km_df.to_csv(km_csv, index=False)
    return cindex_df, km_df


# ─── 4. Ablation ranking (single + multi) per feature set, per horizon ───────

ABLATION_CSV = ALIGN_DIR / "ablation_ranking.csv"


def build_ablation(cohorts, horizons=(7, 10), model_name="ENSEMBLE",
                   seed=RANDOM_STATE, force=False, out_suffix="") -> pd.DataFrame:
    """
    Knock-out ablation (single + multi-variable) for every feature set and
    horizon, on ``model_name``. Cached to ABLATION_CSV (suffixed when
    ``out_suffix`` is given).
    """
    csv = _suffixed(ABLATION_CSV, out_suffix)
    if csv.exists() and not force:
        return pd.read_csv(csv)

    fs_names = get_all_align_feature_set_names()
    frames = []
    for h in horizons:
        df = cohorts[int(h)]
        target = f"y{int(h)}"
        for fs in fs_names:
            X, y = extract_Xy_align(df, fs, target)
            ab = run_ablation(X, y, fs, model_name=model_name, seed=seed)
            ab.insert(0, "horizon", int(h))
            ab.insert(0, "model", model_name)
            frames.append(ab)
            print(f"  [abl] h{h} {fs} done", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(csv, index=False)
    return out


# ─── Winner selection + winner pipeline ──────────────────────────────────────

def pick_winner(clf_df, sets=("CV17",), candidates=None,
                exclude=("ENSEMBLE",)) -> str:
    """
    Pick the best ensemble by mean F1-macro over {horizons}x{schemes} on the
    given feature ``sets`` (AUROC as tiebreak). Candidates default to all
    ensemble names except the paper ENSEMBLE (kept as the reference). Returns
    the winning model name.
    """
    if candidates is None:
        candidates = [m for m in ENSEMBLE_NAMES if m not in exclude]
    sub = clf_df[clf_df["model"].isin(candidates) &
                 clf_df["feature_set"].isin(sets)]
    agg = (sub.groupby("model")[["f1_macro", "roc_auc"]].mean()
           .sort_values(["f1_macro", "roc_auc"], ascending=False))
    return agg.index[0]


def build_winner_pipeline(cohorts, winner, base="CV17", thy="CV17_THY26",
                          horizons=(7, 10), n_boot=1000, seed=RANDOM_STATE,
                          force=False):
    """
    Run the full downstream pipeline (incremental value, ML indicator, ablation)
    for the winning ensemble, caching to ``*_<winner>.csv`` files so the
    paper-ENSEMBLE caches stay untouched.
    """
    print(f"[run_alignment] winner pipeline for {winner} ...", flush=True)
    build_incremental_table(cohorts, base=base, horizons=horizons,
                            model_name=winner, n_boot=n_boot, seed=seed,
                            force=force, out_suffix=winner)
    build_ml_indicator(cohorts, base=base, thy=thy, horizons=horizons,
                       model_name=winner, seed=seed, force=force,
                       out_suffix=winner)
    build_ablation(cohorts, horizons=horizons, model_name=winner, seed=seed,
                   force=force, out_suffix=winner)
    print(f"[run_alignment] winner pipeline for {winner} done.", flush=True)


# ─── Driver ─────────────────────────────────────────────────────────────────

def build_all(horizons=(7, 10), seed=RANDOM_STATE, n_boot=1000, force=False):
    """Build and cache every alignment result table."""
    cohorts = build_strict_cohorts(horizons)
    print("[run_alignment] cohort summary:")
    for h in horizons:
        print("  ", summarize(cohorts[int(h)], h))

    print("[run_alignment] 1/5 classification table ...", flush=True)
    clf = build_classification_table(cohorts, horizons=horizons, seed=seed,
                                     force=force)
    print("[run_alignment] 2/5 incremental table (paper ENSEMBLE) ...", flush=True)
    build_incremental_table(cohorts, horizons=horizons, n_boot=n_boot,
                            seed=seed, force=force)
    print("[run_alignment] 3/5 ML indicator (paper ENSEMBLE) ...", flush=True)
    build_ml_indicator(cohorts, horizons=horizons, seed=seed, force=force)
    print("[run_alignment] 4/5 ablation (paper ENSEMBLE) ...", flush=True)
    build_ablation(cohorts, horizons=horizons, seed=seed, force=force)

    winner = pick_winner(clf)
    print(f"[run_alignment] 5/5 winner pipeline ({winner}) ...", flush=True)
    build_winner_pipeline(cohorts, winner, horizons=horizons, n_boot=n_boot,
                          seed=seed, force=force)
    print("[run_alignment] all caches built under", ALIGN_DIR, flush=True)


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_all(force=force)

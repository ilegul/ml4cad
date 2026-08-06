"""Shared low-level utilities: paths, splits, metrics, bootstrap, caching."""

import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create the output directories used by the pipeline."""
    for d in [config.DATA_PROC, config.MODELS_DIR, config.PRED_DIR,
              config.RESULTS_DIR, config.FIGURES_DIR, config.CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = config.SEED) -> None:
    """Seed the Python and numpy generators."""
    random.seed(seed)
    np.random.seed(seed)


def dependency_versions() -> dict:
    """Versions that can change numerical results, for cache signatures."""
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    for name in ["imblearn", "xgboost", "lifelines", "sksurv"]:
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "absent"
    return versions


def write_manifest(path: Path, **fields) -> Path:
    """Write a JSON manifest with a timestamp and the dependency versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": config.PROFILE,
        "protocol_version": config.PROTOCOL_VERSION,
        "seed": config.SEED,
        "dependencies": dependency_versions(),
        **fields,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------------

def read_frame(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the earlier notebooks in order first."
        )
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write_frame(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def save_result(df: pd.DataFrame, name: str) -> Path:
    """Write a result table to results/<name>.csv."""
    return write_frame(df, config.RESULTS_DIR / f"{name}.csv")


def save_fig(fig, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def feature_list(feature_set: str) -> list:
    if feature_set not in config.FEATURE_SETS:
        raise KeyError(
            f"Unknown feature set {feature_set!r}. "
            f"Available: {list(config.FEATURE_SETS)}"
        )
    return list(config.FEATURE_SETS[feature_set])


def extract_xy(df: pd.DataFrame, feature_set: str, target: str = config.TARGET):
    """Return (X, y, patient_ids). Missing features raise instead of degrading."""
    features = feature_list(feature_set)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Feature set {feature_set!r} requires missing columns: {missing}")
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not in frame")
    return (df[features].copy(),
            df[target].astype(int).to_numpy(),
            df[config.ID_COL].to_numpy())


def categorical_indices(feature_set: str) -> list:
    """Positions of binary columns, for samplers that must respect categories."""
    features = feature_list(feature_set)
    return [i for i, f in enumerate(features) if f not in config.CONTINUOUS_FEATURES]


# ---------------------------------------------------------------------------
# Splits, keyed by patient identifier
# ---------------------------------------------------------------------------

def check_unique_ids(df: pd.DataFrame) -> None:
    dup = df[config.ID_COL].duplicated().sum()
    if dup:
        raise ValueError(f"{dup} duplicated {config.ID_COL} values")


def make_splits(df: pd.DataFrame, target: str = config.TARGET,
                seed: int = config.SEED) -> dict:
    """Stratified 60/20/20 split returning patient identifiers."""
    check_unique_ids(df)
    ids = df[config.ID_COL].to_numpy()
    y = df[target].astype(int).to_numpy()

    rest_frac = config.VALID_FRAC + config.TEST_FRAC
    ids_train, ids_rest, y_train, y_rest = train_test_split(
        ids, y, test_size=rest_frac, stratify=y, random_state=seed)
    ids_valid, ids_test = train_test_split(
        ids_rest, test_size=config.TEST_FRAC / rest_frac,
        stratify=y_rest, random_state=seed)
    return {
        "train": sorted(int(i) for i in ids_train),
        "valid": sorted(int(i) for i in ids_valid),
        "test": sorted(int(i) for i in ids_test),
    }


def save_splits(splits_by_horizon: dict, path=None) -> Path:
    path = Path(path or config.SPLITS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": config.SEED,
        "fractions": [config.TRAIN_FRAC, config.VALID_FRAC, config.TEST_FRAC],
        "splits": splits_by_horizon,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_splits(horizon: int, path=None) -> dict:
    path = Path(path or config.SPLITS_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 1_data_process.ipynb first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"h{int(horizon)}"
    if key not in payload["splits"]:
        raise KeyError(f"No split stored for horizon {horizon}")
    return payload["splits"][key]


def split_masks(patient_ids, splits: dict) -> dict:
    """Boolean masks over an array of patient identifiers."""
    ids = np.asarray(patient_ids)
    return {part: np.isin(ids, np.asarray(members))
            for part, members in splits.items()}


def assert_disjoint(splits: dict) -> None:
    parts = list(splits)
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise AssertionError(
                    f"{a} and {b} share {len(overlap)} patients")


def assert_same_patients(*id_arrays) -> None:
    """Every paired comparison must run on identical patients."""
    reference = np.asarray(id_arrays[0])
    for other in id_arrays[1:]:
        other = np.asarray(other)
        if reference.shape != other.shape or not np.array_equal(reference, other):
            raise AssertionError("Compared arms do not cover identical patients")


def make_folds(y, n_splits: int, seed: int = config.SEED) -> list:
    """Stratified fold indices, identical for every arm given the same y."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "n_events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision_event": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_event": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_free": float(precision_score(y_true, y_pred, pos_label=0,
                                                zero_division=0)),
        "recall_free": float(recall_score(y_true, y_pred, pos_label=0,
                                          zero_division=0)),
    }


def optimize_threshold(y_true, y_proba, grid=None, strategy=None) -> float:
    """Best threshold on the given (never test) observations."""
    grid = config.THRESHOLD_GRID if grid is None else grid
    strategy = config.THRESHOLD_STRATEGY if strategy is None else strategy
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if strategy == "f1_macro":
        def score(threshold):
            return f1_score(y_true, (y_proba >= threshold).astype(int),
                            average="macro")
    elif strategy == "youden":
        def score(threshold):
            predicted = (y_proba >= threshold).astype(int)
            sensitivity = recall_score(y_true, predicted, zero_division=0)
            specificity = recall_score(y_true, predicted, pos_label=0,
                                       zero_division=0)
            return sensitivity + specificity - 1.0
    else:
        raise KeyError(f"Unknown threshold strategy {strategy!r}")

    return float(grid[int(np.argmax([score(t) for t in grid]))])


METRIC_FUNCS = {
    "f1_macro": lambda y, p, t: f1_score(y, (p >= t).astype(int), average="macro"),
    "roc_auc": lambda y, p, t: roc_auc_score(y, p),
    "auprc": lambda y, p, t: average_precision_score(y, p),
    "brier": lambda y, p, t: brier_score_loss(y, p),
}


# ---------------------------------------------------------------------------
# Paired inference
# ---------------------------------------------------------------------------

def bootstrap_indices(n: int, n_boot: int, seed: int = config.SEED) -> np.ndarray:
    """One shared resampler so every paired delta uses identical draws."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def paired_delta(y_true, proba_base, proba_other, threshold_base: float,
                 threshold_other: float, n_boot: int = None,
                 seed: int = config.SEED, metrics=None) -> pd.DataFrame:
    """Paired bootstrap of (other - base) on identical observations."""
    n_boot = config.BOOTSTRAP if n_boot is None else n_boot
    metrics = list(METRIC_FUNCS) if metrics is None else list(metrics)
    y_true = np.asarray(y_true)
    proba_base = np.asarray(proba_base)
    proba_other = np.asarray(proba_other)
    if not (len(y_true) == len(proba_base) == len(proba_other)):
        raise ValueError("Paired delta requires identical observation counts")

    point = {m: (METRIC_FUNCS[m](y_true, proba_other, threshold_other)
                 - METRIC_FUNCS[m](y_true, proba_base, threshold_base))
             for m in metrics}

    draws = bootstrap_indices(len(y_true), n_boot, seed)
    collected = {m: [] for m in metrics}
    for idx in draws:
        yb = y_true[idx]
        if yb.min() == yb.max():
            continue
        for m in metrics:
            collected[m].append(
                METRIC_FUNCS[m](yb, proba_other[idx], threshold_other)
                - METRIC_FUNCS[m](yb, proba_base[idx], threshold_base))

    rows = []
    for m in metrics:
        values = np.asarray(collected[m], dtype=float)
        lo, hi = (np.percentile(values, [2.5, 97.5])
                  if values.size else (np.nan, np.nan))
        rows.append({
            "metric": m,
            "delta": float(point[m]),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "p_bootstrap": bootstrap_pvalue(values),
            "n_boot_used": int(values.size),
            "excludes_zero": bool(values.size and (lo > 0 or hi < 0)),
        })
    return pd.DataFrame(rows)


def bootstrap_pvalue(deltas) -> float:
    """Two-sided bootstrap tail probability that the difference is zero."""
    values = np.asarray(deltas, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    below = float(np.mean(values <= 0))
    above = float(np.mean(values >= 0))
    return float(min(1.0, 2.0 * min(below, above)))


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Step-up FDR q-values for the designated secondary comparisons."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def bootstrap_ci(values, level: float = 95.0) -> tuple:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    tail = (100.0 - level) / 2.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def data_fingerprint(df: pd.DataFrame, columns=None) -> str:
    cols = sorted(columns or df.columns)
    hashed = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def cache_signature(**parts) -> str:
    """Signature covering data, protocol and dependencies."""
    payload = dict(parts)
    payload["protocol_version"] = config.PROTOCOL_VERSION
    payload["seed"] = config.SEED
    payload["dependencies"] = dependency_versions()
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def cache_path(name: str, signature: str, suffix: str = ".csv") -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{name}_{signature}{suffix}"


def load_cache(name: str, signature: str):
    """Return the cached frame or None. Never triggers computation."""
    if not config.USE_CACHE:
        return None
    path = cache_path(name, signature)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "signature" not in frame.columns or not (frame["signature"] == signature).all():
        return None
    return frame.drop(columns=["signature"])


def save_cache(frame: pd.DataFrame, name: str, signature: str) -> Path:
    stamped = frame.copy()
    stamped["signature"] = signature
    return write_frame(stamped, cache_path(name, signature))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def reliability_points(y_true, y_proba, n_bins: int = None) -> pd.DataFrame:
    n_bins = config.CALIBRATION_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_proba, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_predicted": float(y_proba[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)

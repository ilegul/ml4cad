"""Shared development partitions, selection rules and model caches."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone

import config
import train
import utils

VERSION = "v1"
METRICS = ["f1_macro", "roc_auc", "auprc", "brier"]


# ---------------------------------------------------------------------------
# Atomic caches and development partition integrity
# ---------------------------------------------------------------------------

def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached(path, signature, force=False):
    if force or not path.exists():
        return None
    try:
        payload = load(path)
    except (EOFError, ValueError, pickle.UnpicklingError):
        log(f"Incomplete cache; recomputing {path.name}")
        return None
    return payload if payload.get("signature") == signature else None


def store(payload, path, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dump({**payload, "signature": signature}, temporary)
    temporary.replace(path)


def load_partition(horizon, feature_set):
    frame = utils.read_frame(str(config.COHORT_STRICT).format(horizon=horizon))
    utils.check_unique_ids(frame)
    splits = utils.load_splits(horizon)
    utils.assert_disjoint(splits)
    if any(len(values) != len(set(values)) for values in splits.values()):
        raise ValueError("Stored partitions contain duplicate patient identifiers")
    if set(splits) != {"train", "valid", "test"}:
        raise ValueError("Expected the stored train, valid and test partitions")
    X, y, ids = utils.extract_xy(frame, feature_set)
    if set(ids) != set().union(*(set(values) for values in splits.values())):
        raise ValueError("Stored partitions do not cover exactly the cohort")
    masks = utils.split_masks(ids, splits)
    if not np.all(sum(masks.values()) == 1):
        raise ValueError("Each patient must belong to exactly one partition")
    development = masks["train"] | masks["valid"]
    columns = [config.ID_COL, config.TARGET] + utils.feature_list(feature_set)
    signature = utils.cache_signature(
        stage="predictive_development", version=VERSION, horizon=horizon,
        features=utils.feature_list(feature_set), splits=splits,
        fingerprint=utils.data_fingerprint(frame.loc[development], columns))
    return {"X": X, "y": y, "ids": ids, "masks": masks,
            "signature": signature}


def partition(data, split):
    mask = data["masks"][split]
    return data["X"].loc[mask], data["y"][mask], data["ids"][mask]


# ---------------------------------------------------------------------------
# Validation selection and reproducibility checks
# ---------------------------------------------------------------------------

def rank_scores(scores, name="configuration"):
    """The notebook-3 selection rule, with a deterministic final tie break."""
    if scores.empty or not np.isfinite(scores[METRICS].to_numpy()).all():
        raise ValueError("Selection requires finite scores for every candidate")
    if not np.allclose(scores["threshold"], 0.5, rtol=0, atol=0):
        raise ValueError("Configuration selection uses the fixed raw threshold")
    if scores[["n", "n_events"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Candidates must cover the same validation cohort")
    return scores.sort_values(["f1_macro", "roc_auc", name],
                              ascending=[False, False, True])


def check_scores(actual, expected, context):
    """Fail on incompatible original inputs rather than silently reranking."""
    for metric in METRICS + ["n", "n_events", "threshold"]:
        if not np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-10):
            raise ValueError(f"Cannot reproduce {context}: {metric} "
                             f"{actual[metric]} != {expected[metric]}")


# ---------------------------------------------------------------------------
# Frozen base-model specifications and training OOF probabilities
# ---------------------------------------------------------------------------

def member_spec(horizon, feature_set, model, sampler, params_source=None):
    source = params_source or feature_set
    manifest = train.model_manifest(f"tuned_h{horizon}_{source}_{model}")
    if (manifest["horizon"] != horizon or manifest["feature_set"] != source
            or manifest["model"] != model or manifest["seed"] != config.SEED):
        raise ValueError("Tuning manifest does not match the requested model")
    return {"model": model, "sampler": sampler,
            "params": manifest["best_params"], "params_source": source,
            "tuning_signature": manifest["signature"]}


def member_signature(data, spec):
    return utils.cache_signature(stage="sensitivity_member", version=VERSION,
                                 development=data["signature"], spec=spec)


def oof_signature(member):
    return utils.cache_signature(stage="sensitivity_oof", version=VERSION,
                                 member=member["signature"], cv=config.CALIBRATION_CV)


def fit_member(data, feature_set, spec, paths, force=False):
    signature = member_signature(data, spec)
    path = paths["cache"] / f"member_{signature}.joblib"
    result = cached(path, signature, force)
    Xv, yv, iv = partition(data, "valid")
    if result is not None:
        utils.assert_same_patients(iv, result["valid_ids"])
        if not np.array_equal(yv, result["valid_y"]):
            raise ValueError("Cached validation outcomes do not match")
        return result
    Xt, yt, _ = partition(data, "train")
    pipe = train.make_pipeline(spec["model"], spec["sampler"], feature_set)
    pipe.set_params(**spec["params"])
    pipe.fit(Xt, yt)
    proba = pipe.predict_proba(Xv)[:, 1]
    result = {"estimator": pipe, "valid_raw": proba, "valid_ids": iv,
              "valid_y": yv, "spec": spec, "signature": signature,
              "metrics": utils.classification_metrics(yv, proba, 0.5)}
    store(result, path, signature)
    return result


def member_oof(data, member, paths, jobs, force=False):
    signature = oof_signature(member)
    path = paths["cache"] / f"oof_{signature}.joblib"
    result = cached(path, signature, force)
    if result is None:
        Xt, yt, ids = partition(data, "train")
        oof = train.training_oof_proba(clone(member["estimator"]), Xt, yt, n_jobs=jobs)
        result = {"oof": oof, "train_ids": ids, "train_y": yt}
        store(result, path, signature)
    _, yt, ids = partition(data, "train")
    utils.assert_same_patients(ids, result["train_ids"])
    if not np.array_equal(yt, result["train_y"]):
        raise ValueError("Cached OOF outcomes do not match")
    return result["oof"]

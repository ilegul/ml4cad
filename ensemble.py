"""Prespecified paper ensemble and adapted ensemble selection."""

import numpy as np
import pandas as pd

import config
import utils


def mean_proba(probas) -> np.ndarray:
    """Ensemble output is the simple mean of member probabilities."""
    stacked = np.vstack([np.asarray(p, dtype=float) for p in probas])
    return stacked.mean(axis=0)


def check_members_aligned(ids_by_member: dict) -> None:
    """Every member must cover identical patients in identical order."""
    arrays = list(ids_by_member.values())
    if not arrays:
        raise ValueError("No ensemble members provided")
    utils.assert_same_patients(*arrays)


def candidate_ensembles(train_cv_rank: pd.DataFrame) -> dict:
    """Closed candidate set, fixed before any validation outcome is read.

    train_cv_rank must be ordered by training-only cross-validation score and
    carry a 'model' column.
    """
    top3 = list(train_cv_rank["model"].head(3))
    candidates = {
        "paper": list(config.PAPER_ENSEMBLE),
        "top3_train_cv": top3,
        "diverse": list(config.DIVERSE_ENSEMBLE),
    }
    return {name: members for name, members in candidates.items() if len(members) >= 2}


def select_ensemble(validation_scores: pd.DataFrame) -> str:
    """Pick among already-defined candidates using validation only.

    Ties break on AUROC, then on the candidate name, so the choice is
    reproducible.
    """
    ranked = validation_scores.sort_values(
        ["f1_macro", "roc_auc", "candidate"], ascending=[False, False, True])
    return str(ranked.iloc[0]["candidate"])


def evaluate_candidates(candidates: dict, probas: dict, y_true,
                        threshold: float) -> pd.DataFrame:
    """Validation metrics for each candidate ensemble."""
    rows = []
    for name, members in candidates.items():
        missing = [m for m in members if m not in probas]
        if missing:
            continue
        proba = mean_proba([probas[m] for m in members])
        metrics = utils.classification_metrics(y_true, proba, threshold)
        rows.append({"candidate": name, "members": ",".join(members), **metrics})
    return pd.DataFrame(rows)

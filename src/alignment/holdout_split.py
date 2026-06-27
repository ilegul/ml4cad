"""
src/alignment/holdout_split.py
──────────────────────────────
Stratified 60/20/20 train/validation/test split (paper-alignment scheme B).

The paper uses a single 60/20/20 split: hyper-parameters / decision threshold
are tuned on the validation set, and final metrics are reported on the test set.
The repo uses 5-fold CV (scheme A). The alignment notebook reports BOTH, so this
module provides the 60/20/20 split as a clean alternative.

The split returns **index positions only** — no data is transformed here, so
imputation/scaling/oversampling can be fitted on the training part downstream,
keeping the split leakage-free. The same ``seed`` yields the same split, so
paired comparisons (CV17 vs CV17+thyroid) use identical train/val/test rows.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import RANDOM_STATE


def split_60_20_20(y, seed: int = RANDOM_STATE):
    """
    Stratified 60/20/20 split of the rows indexed by ``y``.

    Returns (idx_train, idx_val, idx_test) as numpy arrays of the *positional*
    indices into ``y`` (0..len(y)-1), stratified on the class label.
    """
    y = pd.Series(np.asarray(y))
    pos = np.arange(len(y))

    # First carve off 60% train, 40% temp (stratified).
    idx_train, idx_temp = train_test_split(
        pos, test_size=0.40, stratify=y.values, random_state=seed)
    # Split the 40% temp evenly into 20% val + 20% test (stratified).
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y.values[idx_temp],
        random_state=seed)
    return idx_train, idx_val, idx_test


def describe_split(y, idx_train, idx_val, idx_test) -> pd.DataFrame:
    """Tabulate N and positive prevalence of each split part (for the notebook)."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rows = []
    for name, idx in [("train", idx_train), ("val", idx_val),
                      ("test", idx_test)]:
        yy = y.iloc[idx]
        rows.append({
            "part": name,
            "n": len(yy),
            "fraction": round(len(yy) / len(y), 3),
            "n_pos": int(yy.sum()),
            "prevalence": round(float(yy.mean()), 4),
        })
    return pd.DataFrame(rows)

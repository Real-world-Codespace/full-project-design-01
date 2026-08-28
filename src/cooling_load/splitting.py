"""Time-based holdout and expanding-window cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_holdout(
    frame: pd.DataFrame,
    timestamp_column: str,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    timestamps = np.sort(pd.to_datetime(frame[timestamp_column], utc=True).unique())
    validation_start = timestamps[int(len(timestamps) * (1 - validation_fraction - test_fraction))]
    test_start = timestamps[int(len(timestamps) * (1 - test_fraction))]
    time = pd.to_datetime(frame[timestamp_column], utc=True)
    train = frame.loc[time < validation_start].copy()
    validation = frame.loc[(time >= validation_start) & (time < test_start)].copy()
    test = frame.loc[time >= test_start].copy()
    return train, validation, test


class ExpandingTimestampSplit:
    """Sklearn-compatible split that never divides the same timestamp across folds."""

    def __init__(self, n_splits: int = 3, min_train_fraction: float = 0.5):
        self.n_splits = n_splits
        self.min_train_fraction = min_train_fraction

    def split(self, X: pd.DataFrame, y=None, groups=None):
        if "__timestamp" not in X.columns:
            raise ValueError("X must include an internal __timestamp column.")
        timestamps = pd.to_datetime(X["__timestamp"], utc=True)
        unique = np.sort(timestamps.unique())
        first_validation = int(len(unique) * self.min_train_fraction)
        remaining = len(unique) - first_validation
        fold_size = remaining // self.n_splits
        if fold_size < 1:
            raise ValueError("Not enough timestamps for requested cross-validation splits.")
        for fold in range(self.n_splits):
            train_end = first_validation + fold * fold_size
            validation_end = len(unique) if fold == self.n_splits - 1 else train_end + fold_size
            train_mask = timestamps < unique[train_end]
            validation_mask = (timestamps >= unique[train_end]) & (timestamps <= unique[validation_end - 1])
            yield np.flatnonzero(train_mask), np.flatnonzero(validation_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

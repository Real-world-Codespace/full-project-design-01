"""Regression metrics and slice-level evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual).sum()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "wape": float(np.abs(actual - predicted).sum() / denominator) if denominator else float("nan"),
        "r2": float(r2_score(actual, predicted)),
    }


def metrics_by_group(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    group_column: str,
) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby(group_column):
        rows.append({group_column: name, **regression_metrics(group[actual_column], group[prediction_column])})
    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)

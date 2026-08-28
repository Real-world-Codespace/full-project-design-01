"""Lightweight post-deployment data and prediction monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(reference_values) == 0 or len(current_values) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_hist = np.histogram(reference_values, bins=edges)[0] / len(reference_values)
    current_hist = np.histogram(current_values, bins=edges)[0] / len(current_values)
    reference_hist = np.clip(reference_hist, epsilon, None)
    current_hist = np.clip(current_hist, epsilon, None)
    return float(np.sum((current_hist - reference_hist) * np.log(current_hist / reference_hist)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for column in columns:
        psi = population_stability_index(reference[column], current[column])
        status = "stable" if psi < 0.1 else "monitor" if psi < 0.25 else "retrain"
        rows.append({"feature": column, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

"""Model comparison, time-aware tuning and feature importance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from cooling_load.splitting import ExpandingTimestampSplit


@dataclass
class ModelSelectionResult:
    leaderboard: pd.DataFrame
    champion_name: str


def candidate_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "median_baseline": DummyRegressor(strategy="median"),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=random_state
        ),
    }


def build_pipeline(feature_columns: list[str], estimator: Any) -> Pipeline:
    transformer = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", add_indicator=True), feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", transformer), ("model", estimator)])


def select_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timestamp_column: str,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    models: dict[str, Any] | None = None,
    random_state: int = 42,
) -> ModelSelectionResult:
    working_features = ["__timestamp", *feature_columns]
    X = train[[timestamp_column, *feature_columns]].rename(
        columns={timestamp_column: "__timestamp"}
    )
    y = train[target_column]
    splitter = ExpandingTimestampSplit(n_splits=n_splits, min_train_fraction=min_train_fraction)
    estimators = models or candidate_models(random_state=random_state)
    rows: list[dict[str, object]] = []
    best_score = float("inf")
    best_name = ""
    for name, estimator in estimators.items():
        fold_scores: list[float] = []
        for train_index, validation_index in splitter.split(X[working_features]):
            pipeline = build_pipeline(feature_columns, estimator)
            pipeline.fit(X.iloc[train_index][feature_columns], y.iloc[train_index])
            prediction = pipeline.predict(X.iloc[validation_index][feature_columns])
            fold_scores.append(
                float(mean_squared_error(y.iloc[validation_index], prediction) ** 0.5)
            )
        score = float(np.mean(fold_scores))
        rows.append(
            {
                "model": name,
                "cv_rmse_mean": score,
                "cv_rmse_std": float(np.std(fold_scores)),
            }
        )
        if score < best_score:
            best_score, best_name = score, name
    leaderboard = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    return ModelSelectionResult(leaderboard=leaderboard, champion_name=best_name)


def _tuning_candidates(model_name: str, random_state: int) -> list[tuple[Any, dict[str, object]]]:
    """Return a small, deterministic search space for the selected model family."""
    parameter_sets: dict[str, list[dict[str, object]]] = {
        "median_baseline": [{}],
        "random_forest": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": 1.0},
            {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 3, "max_features": 0.8},
            {"n_estimators": 450, "max_depth": 24, "min_samples_leaf": 2, "max_features": 0.7},
            {"n_estimators": 450, "max_depth": None, "min_samples_leaf": 4, "max_features": 0.9},
        ],
        "extra_trees": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": 1.0},
            {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 3, "max_features": 0.8},
            {"n_estimators": 450, "max_depth": 24, "min_samples_leaf": 2, "max_features": 0.7},
            {"n_estimators": 450, "max_depth": None, "min_samples_leaf": 4, "max_features": 0.9},
        ],
        "hist_gradient_boosting": [
            {
                "max_iter": 200,
                "learning_rate": 0.04,
                "max_leaf_nodes": 15,
                "l2_regularization": 0.0,
            },
            {
                "max_iter": 250,
                "learning_rate": 0.06,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.0,
            },
            {
                "max_iter": 350,
                "learning_rate": 0.04,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.1,
            },
            {
                "max_iter": 250,
                "learning_rate": 0.08,
                "max_leaf_nodes": 63,
                "l2_regularization": 0.1,
            },
        ],
    }
    if model_name not in parameter_sets:
        raise ValueError(f"Unsupported model family for tuning: {model_name}")

    candidates: list[tuple[Any, dict[str, object]]] = []
    for params in parameter_sets[model_name]:
        if model_name == "median_baseline":
            estimator = DummyRegressor(strategy="median")
        elif model_name == "random_forest":
            estimator = RandomForestRegressor(**params, n_jobs=-1, random_state=random_state)
        elif model_name == "extra_trees":
            estimator = ExtraTreesRegressor(**params, n_jobs=-1, random_state=random_state)
        else:
            estimator = HistGradientBoostingRegressor(**params, random_state=random_state)
        candidates.append((estimator, params))
    return candidates


def tune_selected_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    random_state: int = 42,
    max_trials: int | None = None,
) -> tuple[Pipeline, pd.DataFrame]:
    """Tune only the model family selected by temporal cross-validation."""
    candidates = _tuning_candidates(model_name, random_state)
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        candidates = candidates[:max_trials]

    trials: list[dict[str, object]] = []
    best_pipeline: Pipeline | None = None
    best_rmse = float("inf")
    for estimator, params in candidates:
        pipeline = build_pipeline(feature_columns, estimator)
        pipeline.fit(train[feature_columns], train[target_column])
        prediction = pipeline.predict(validation[feature_columns])
        rmse = float(mean_squared_error(validation[target_column], prediction) ** 0.5)
        trials.append({"model": model_name, **params, "validation_rmse": rmse})
        if rmse < best_rmse:
            best_rmse, best_pipeline = rmse, pipeline
    assert best_pipeline is not None
    return best_pipeline, pd.DataFrame(trials).sort_values("validation_rmse").reset_index(drop=True)


def tree_feature_importance(pipeline: Pipeline, top_n: int = 30) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": names[: len(importances)], "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

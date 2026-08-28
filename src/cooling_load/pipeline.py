"""Reusable application-level stages called by notebooks and tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cooling_load.artifacts import create_bundle, save_bundle
from cooling_load.clustering import OperatingRegimeClusterer
from cooling_load.config import ProjectConfig
from cooling_load.eda import EDAReport, generate_eda_report
from cooling_load.evaluation import regression_metrics
from cooling_load.features import CoolingLoadFeatureBuilder, model_feature_columns
from cooling_load.ingestion import ingest_dataset
from cooling_load.io import read_frame, write_frame
from cooling_load.modeling import select_model, tune_tree_model
from cooling_load.preprocessing import preprocess_sensor_data
from cooling_load.registry import publish_model_bundle
from cooling_load.splitting import temporal_holdout
from cooling_load.validation import validate_raw_data


def run_ingestion_and_validation(config: ProjectConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    ingestion = ingest_dataset(config)
    frames = [read_frame(path) for path in ingestion.files]
    raw = pd.concat(frames, ignore_index=True)
    report = validate_raw_data(raw, config.data)
    return raw, report.to_dict()


def run_preprocessing(config: ProjectConfig, raw: pd.DataFrame) -> pd.DataFrame:
    settings = config.raw["preprocessing"]
    cleaned = preprocess_sensor_data(
        raw,
        config.data,
        max_ffill_steps=int(settings["max_ffill_steps"]),
        clip_quantiles=tuple(settings["numeric_clip_quantiles"]),
    )
    write_frame(cleaned, config.path("interim_data"))
    return cleaned


def run_eda(config: ProjectConfig, raw: pd.DataFrame) -> EDAReport:
    """Persist the standard EDA chart set and machine-readable summary."""
    return generate_eda_report(
        raw,
        config.data,
        output_dir=config.path("eda_report_dir"),
        summary_path=config.path("eda_summary"),
    )


def run_feature_engineering(config: ProjectConfig, cleaned: pd.DataFrame) -> pd.DataFrame:
    builder = CoolingLoadFeatureBuilder(config.data, config.features)
    featured = builder.transform(cleaned)
    write_frame(featured, config.path("feature_data"))
    return featured


def run_training(config: ProjectConfig, featured: pd.DataFrame) -> dict[str, object]:
    timestamp = config.data.timestamp_column
    target = config.data.target_column
    train, validation, test = temporal_holdout(featured, timestamp)

    clusterer = OperatingRegimeClusterer(
        feature_columns=config.clustering.features,
        n_clusters=config.clustering.n_clusters,
        random_state=int(config.raw["project"]["random_state"]),
    )
    ordering = f"{target}_lag_{config.features.forecast_horizon_hours}"
    train = clusterer.fit_transform(train, ordering)
    validation = clusterer.transform(validation)
    test = clusterer.transform(test)
    feature_columns = model_feature_columns(train, config.data)
    # K-Means labels and distance are available to the supervised model.
    feature_columns.extend([column for column in ["operating_cluster", "cluster_distance"] if column not in feature_columns])

    training = config.raw["training"]
    selection = select_model(
        train,
        feature_columns,
        target,
        timestamp,
        n_splits=int(training["validation_periods"]),
        min_train_fraction=float(training["min_train_fraction"]),
    )
    tuned_model, tuning_results = tune_tree_model(
        train,
        validation,
        feature_columns,
        target,
        random_state=int(config.raw["project"]["random_state"]),
    )
    validation_prediction = tuned_model.predict(validation[feature_columns])
    validation_metrics = regression_metrics(validation[target], validation_prediction)

    # Refit the selected hyperparameters on train + validation; test stays untouched.
    train_validation = pd.concat([train, validation], ignore_index=True)
    tuned_model.fit(train_validation[feature_columns], train_validation[target])
    test_prediction = tuned_model.predict(test[feature_columns])
    test_metrics = regression_metrics(test[target], test_prediction)

    bundle = create_bundle(
        model=tuned_model,
        clusterer=clusterer,
        feature_columns=feature_columns,
        project_config=config.raw,
        metrics=test_metrics,
        model_name="tuned_extra_trees",
    )
    model_path = save_bundle(bundle, config.path("model"))
    registry_settings = config.raw.get("model_registry", {})
    published_model = None
    if registry_settings.get("enabled", False):
        published_model = publish_model_bundle(
            local_model_path=model_path,
            bundle=bundle,
            bucket=registry_settings["bucket"],
            prefix=registry_settings["prefix"],
            profile=registry_settings.get("profile"),
            server_side_encryption=registry_settings.get("server_side_encryption"),
        )
    metrics_path = config.path("metrics")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"validation": validation_metrics, "test": test_metrics}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    test_predictions = test[[timestamp, config.data.entity_column, target, "operating_regime"]].copy()
    test_predictions["prediction"] = test_prediction
    return {
        "leaderboard": selection.leaderboard,
        "tuning_results": tuning_results,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_predictions": test_predictions,
        "model_path": model_path,
        "published_model": published_model,
        "feature_columns": feature_columns,
        "config_snapshot": asdict(config.features),
    }

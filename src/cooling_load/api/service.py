"""Inference service that reuses training-time feature logic."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cooling_load.artifacts import ModelBundle
from cooling_load.config import ClusteringConfig, DataConfig, FeatureConfig
from cooling_load.features import CoolingLoadFeatureBuilder


@dataclass
class ForecastService:
    bundle: ModelBundle

    def __post_init__(self) -> None:
        raw = self.bundle.project_config
        data = raw["data"]
        features = raw["features"]
        self.data_config = DataConfig(
            timestamp_column=data["timestamp_column"],
            entity_column=data["entity_column"],
            target_column=data["target_column"],
            frequency=data["frequency"],
            required_columns=tuple(data["required_columns"]),
        )
        self.feature_config = FeatureConfig(
            forecast_horizon_hours=int(features["forecast_horizon_hours"]),
            target_lags=tuple(features["target_lags"]),
            rolling_windows=tuple(features["rolling_windows"]),
            sensor_columns=tuple(features["sensor_columns"]),
        )
        clustering = raw["clustering"]
        self.clustering_config = ClusteringConfig(
            n_clusters=int(clustering["n_clusters"]), features=tuple(clustering["features"])
        )
        self.feature_builder = CoolingLoadFeatureBuilder(self.data_config, self.feature_config)

    def forecast(self, observations: list[dict[str, object]]) -> dict[str, object]:
        frame = pd.DataFrame(observations)
        entity = self.data_config.entity_column
        timestamp = self.data_config.timestamp_column
        target = self.data_config.target_column
        if frame[entity].nunique() != 1:
            raise ValueError("A request must contain observations for exactly one building.")
        frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
        frame = frame.sort_values(timestamp).drop_duplicates(timestamp, keep="last")
        horizon = self.feature_config.forecast_horizon_hours
        last_timestamp = frame[timestamp].max()
        future_rows = []
        for step in range(1, horizon + 1):
            future = {column: None for column in self.data_config.required_columns}
            future[timestamp] = last_timestamp + pd.Timedelta(hours=step)
            future[entity] = frame[entity].iloc[-1]
            future_rows.append(future)
        future_timestamp = future_rows[-1][timestamp]
        frame = pd.concat([frame, pd.DataFrame(future_rows)], ignore_index=True)
        engineered = self.feature_builder.transform(frame, drop_incomplete=False)
        row = engineered.loc[engineered[timestamp] == future_timestamp].copy()
        if row.empty:
            raise ValueError("Unable to construct a forecast row from supplied observations.")
        # Training creates one dummy per known building. A single-building API
        # request naturally lacks the other dummy columns, so add them as zero.
        for column in self.bundle.feature_columns:
            if column.startswith("building_") and column not in row:
                row[column] = 0
        row = self.bundle.clusterer.transform(row)
        missing_columns = sorted(set(self.bundle.feature_columns) - set(row.columns))
        if missing_columns:
            raise ValueError(f"Request cannot produce required model features: {missing_columns}")
        prediction = float(self.bundle.model.predict(row[list(self.bundle.feature_columns)])[0])
        return {
            "building_id": str(row[entity].iloc[0]),
            "forecast_timestamp": future_timestamp.to_pydatetime(),
            "forecast_horizon_hours": horizon,
            "cooling_load_kwh": max(0.0, prediction),
            "operating_regime": str(row["operating_regime"].iloc[0]),
            "model_name": self.bundle.model_name,
            "model_created_at": self.bundle.created_at,
        }

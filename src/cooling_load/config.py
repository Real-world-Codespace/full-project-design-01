"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    timestamp_column: str
    entity_column: str
    target_column: str
    frequency: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class FeatureConfig:
    forecast_horizon_hours: int
    target_lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    sensor_columns: tuple[str, ...]


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    data: DataConfig
    features: FeatureConfig
    clustering: ClusteringConfig
    config_path: Path

    @property
    def root(self) -> Path:
        return self.config_path.parent.parent

    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]


def load_config(path: str | Path = "configs/base.yaml") -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    data = raw["data"]
    feature = raw["features"]
    clustering = raw["clustering"]
    return ProjectConfig(
        raw=raw,
        data=DataConfig(
            timestamp_column=data["timestamp_column"],
            entity_column=data["entity_column"],
            target_column=data["target_column"],
            frequency=data["frequency"],
            required_columns=tuple(data["required_columns"]),
        ),
        features=FeatureConfig(
            forecast_horizon_hours=int(feature["forecast_horizon_hours"]),
            target_lags=tuple(int(value) for value in feature["target_lags"]),
            rolling_windows=tuple(int(value) for value in feature["rolling_windows"]),
            sensor_columns=tuple(feature["sensor_columns"]),
        ),
        clustering=ClusteringConfig(
            n_clusters=int(clustering["n_clusters"]),
            features=tuple(clustering["features"]),
        ),
        config_path=config_path,
    )

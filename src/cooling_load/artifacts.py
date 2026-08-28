"""Versioned model bundle persisted as one deployable artifact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelBundle:
    model: Any
    clusterer: Any
    feature_columns: tuple[str, ...]
    project_config: dict[str, Any]
    metrics: dict[str, float]
    model_name: str
    created_at: str


def create_bundle(
    model: Any,
    clusterer: Any,
    feature_columns: list[str],
    project_config: dict[str, Any],
    metrics: dict[str, float],
    model_name: str,
) -> ModelBundle:
    return ModelBundle(
        model=model,
        clusterer=clusterer,
        feature_columns=tuple(feature_columns),
        project_config=project_config,
        metrics=metrics,
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(destination)
    return destination


def load_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(Path(path))

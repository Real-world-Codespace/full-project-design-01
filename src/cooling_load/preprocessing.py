"""Time-aware cleaning and hourly alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cooling_load.config import DataConfig


def preprocess_sensor_data(
    frame: pd.DataFrame,
    config: DataConfig,
    max_ffill_steps: int = 3,
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
) -> pd.DataFrame:
    """Clean and resample without using future observations for imputation.

    Sensor columns are forward-filled by building for short gaps. The target is
    never imputed, so training rows without ground truth remain visibly missing.
    """

    timestamp = config.timestamp_column
    entity = config.entity_column
    target = config.target_column
    result = frame.copy()
    result[timestamp] = pd.to_datetime(result[timestamp], utc=True)
    result = result.sort_values([entity, timestamp]).drop_duplicates([entity, timestamp], keep="last")

    numeric = [column for column in config.required_columns if column not in {timestamp, entity}]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    pieces: list[pd.DataFrame] = []
    for building_id, building in result.groupby(entity, sort=False):
        indexed = building.set_index(timestamp).sort_index()
        resampled = indexed[numeric].resample(config.frequency).mean()
        resampled[entity] = building_id
        sensor_columns = [column for column in numeric if column != target]
        resampled[sensor_columns] = resampled[sensor_columns].ffill(limit=max_ffill_steps)
        pieces.append(resampled.reset_index())

    result = pd.concat(pieces, ignore_index=True).sort_values([timestamp, entity]).reset_index(drop=True)
    lower_q, upper_q = clip_quantiles
    sensor_columns = [column for column in numeric if column != target]
    for column in sensor_columns:
        bounds = result.groupby(entity)[column].quantile([lower_q, upper_q]).unstack()
        lower = result[entity].map(bounds[lower_q])
        upper = result[entity].map(bounds[upper_q])
        result[column] = result[column].clip(lower=lower, upper=upper)

    result[target] = result[target].where(result[target] >= 0, np.nan)
    return result

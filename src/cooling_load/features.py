"""Leakage-safe feature engineering for multi-building forecasting."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from cooling_load.config import DataConfig, FeatureConfig


class CoolingLoadFeatureBuilder:
    def __init__(self, data: DataConfig, features: FeatureConfig):
        self.data = data
        self.features = features

    def transform(self, frame: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
        timestamp = self.data.timestamp_column
        entity = self.data.entity_column
        target = self.data.target_column
        horizon = self.features.forecast_horizon_hours
        result = frame.copy().sort_values([entity, timestamp])
        result[timestamp] = pd.to_datetime(result[timestamp], utc=True)

        dt = result[timestamp].dt
        result["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
        result["dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        result["dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)
        result["month_sin"] = np.sin(2 * np.pi * (dt.month - 1) / 12)
        result["month_cos"] = np.cos(2 * np.pi * (dt.month - 1) / 12)
        result["is_weekend"] = (dt.dayofweek >= 5).astype(int)

        grouped_target = result.groupby(entity, sort=False)[target]
        for lag in sorted(set(self.features.target_lags) | {horizon}):
            result[f"{target}_lag_{lag}"] = grouped_target.shift(lag)
        for window in self.features.rolling_windows:
            shifted = grouped_target.shift(horizon)
            result[f"{target}_mean_{window}"] = shifted.groupby(result[entity]).transform(
                lambda series: series.rolling(window, min_periods=max(2, window // 3)).mean()
            )
            result[f"{target}_std_{window}"] = shifted.groupby(result[entity]).transform(
                lambda series: series.rolling(window, min_periods=max(2, window // 3)).std()
            )

        for column in self.features.sensor_columns:
            result[f"{column}_lag_{horizon}"] = result.groupby(entity, sort=False)[column].shift(horizon)

        supply = f"chilled_water_supply_c_lag_{horizon}"
        returned = f"chilled_water_return_c_lag_{horizon}"
        flow = f"chilled_water_flow_m3h_lag_{horizon}"
        if {supply, returned}.issubset(result.columns):
            result["water_temperature_delta_c"] = result[returned] - result[supply]
        if {flow, "water_temperature_delta_c"}.issubset(result.columns):
            result["thermal_load_proxy"] = result[flow] * result["water_temperature_delta_c"]

        entity_dummies = pd.get_dummies(result[entity], prefix="building", dtype=int)
        result = pd.concat([result, entity_dummies], axis=1)
        if drop_incomplete:
            required = [f"{target}_lag_{max(self.features.target_lags)}"]
            result = result.dropna(subset=required)
        return result.reset_index(drop=True)

    def metadata(self) -> dict[str, object]:
        return {"data": asdict(self.data), "features": asdict(self.features)}


def model_feature_columns(frame: pd.DataFrame, data: DataConfig) -> list[str]:
    excluded = {data.timestamp_column, data.entity_column, data.target_column, "operating_regime"}
    return [column for column in frame.columns if column not in excluded]

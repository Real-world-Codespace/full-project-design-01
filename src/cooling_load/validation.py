"""Dataset contract and quality checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from cooling_load.config import DataConfig


class DataValidationError(ValueError):
    """Raised when a dataset violates a required contract."""


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    columns: int
    duplicate_keys: int
    invalid_timestamps: int
    missing_fraction: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_raw_data(frame: pd.DataFrame, config: DataConfig) -> ValidationReport:
    missing_columns = sorted(set(config.required_columns) - set(frame.columns))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if frame.empty:
        raise DataValidationError("Dataset is empty.")

    parsed = pd.to_datetime(frame[config.timestamp_column], errors="coerce", utc=True)
    invalid_timestamps = int(parsed.isna().sum())
    keys = frame.assign(**{config.timestamp_column: parsed})[
        [config.entity_column, config.timestamp_column]
    ]
    duplicate_keys = int(keys.duplicated().sum())

    numeric_columns = [config.target_column, *config.required_columns[3:]]
    non_numeric = [
        column
        for column in numeric_columns
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().sum() == 0
    ]
    if non_numeric:
        raise DataValidationError(f"Columns expected to be numeric: {non_numeric}")
    if invalid_timestamps:
        raise DataValidationError(f"Found {invalid_timestamps} invalid timestamps.")
    if frame[config.entity_column].isna().any():
        raise DataValidationError("building_id contains missing values.")

    return ValidationReport(
        rows=len(frame),
        columns=len(frame.columns),
        duplicate_keys=duplicate_keys,
        invalid_timestamps=invalid_timestamps,
        missing_fraction={name: float(value) for name, value in frame.isna().mean().items()},
    )

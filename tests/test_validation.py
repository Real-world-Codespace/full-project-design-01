import pandas as pd
import pytest

from cooling_load.config import load_config
from cooling_load.synthetic import generate_synthetic_data
from cooling_load.validation import DataValidationError, validate_raw_data


def test_validation_accepts_canonical_dataset():
    config = load_config("configs/base.yaml")
    frame = generate_synthetic_data(periods=48, buildings=("A",))
    report = validate_raw_data(frame, config.data)
    assert report.rows == len(frame)
    assert report.duplicate_keys > 0


def test_validation_rejects_missing_required_column():
    config = load_config("configs/base.yaml")
    frame = generate_synthetic_data(periods=24, buildings=("A",)).drop(columns=["building_id"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw_data(frame, config.data)

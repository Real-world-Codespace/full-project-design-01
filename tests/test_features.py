import pandas as pd

from cooling_load.config import load_config
from cooling_load.features import CoolingLoadFeatureBuilder
from cooling_load.preprocessing import preprocess_sensor_data
from cooling_load.synthetic import generate_synthetic_data


def test_lag_features_do_not_use_current_target():
    config = load_config("configs/base.yaml")
    raw = generate_synthetic_data(periods=220, buildings=("A",))
    clean = preprocess_sensor_data(raw, config.data)
    featured = CoolingLoadFeatureBuilder(config.data, config.features).transform(clean)
    source = clean.set_index("timestamp")["cooling_load_kwh"]
    row = featured.iloc[-1]
    expected = source.loc[row["timestamp"] - pd.Timedelta(hours=3)]
    assert row["cooling_load_kwh_lag_3"] == expected


def test_feature_builder_creates_cyclical_and_hvac_features():
    config = load_config("configs/base.yaml")
    raw = generate_synthetic_data(periods=220, buildings=("A",))
    clean = preprocess_sensor_data(raw, config.data)
    featured = CoolingLoadFeatureBuilder(config.data, config.features).transform(clean)
    assert {"hour_sin", "hour_cos", "water_temperature_delta_c", "thermal_load_proxy"}.issubset(
        featured.columns
    )

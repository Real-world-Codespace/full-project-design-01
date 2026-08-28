from cooling_load.splitting import temporal_holdout
from cooling_load.synthetic import generate_synthetic_data


def test_temporal_holdout_has_strict_time_boundaries():
    frame = generate_synthetic_data(periods=100, buildings=("A", "B"))
    train, validation, test = temporal_holdout(frame, "timestamp")
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()

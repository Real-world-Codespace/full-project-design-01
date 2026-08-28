from cooling_load.clustering import OperatingRegimeClusterer
from cooling_load.config import load_config
from cooling_load.features import CoolingLoadFeatureBuilder, model_feature_columns
from cooling_load.modeling import select_model
from cooling_load.preprocessing import preprocess_sensor_data
from cooling_load.synthetic import generate_synthetic_data


def test_model_selection_runs_with_temporal_cv():
    config = load_config("configs/base.yaml")
    raw = generate_synthetic_data(periods=400, buildings=("A", "B"))
    clean = preprocess_sensor_data(raw, config.data)
    featured = CoolingLoadFeatureBuilder(config.data, config.features).transform(clean)
    clusterer = OperatingRegimeClusterer(config.clustering.features, n_clusters=4)
    featured = clusterer.fit_transform(featured, "cooling_load_kwh_lag_3")
    columns = model_feature_columns(featured, config.data)
    result = select_model(
        featured,
        columns,
        config.data.target_column,
        config.data.timestamp_column,
        n_splits=2,
        models={"baseline": __import__("sklearn.dummy").dummy.DummyRegressor(strategy="mean")},
    )
    assert result.champion_name == "baseline"
    assert result.leaderboard["cv_rmse_mean"].notna().all()

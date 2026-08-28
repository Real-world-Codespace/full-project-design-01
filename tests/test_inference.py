import pandas as pd

from cooling_load.api.service import ForecastService
from cooling_load.artifacts import create_bundle
from cooling_load.clustering import OperatingRegimeClusterer
from cooling_load.config import load_config
from cooling_load.features import CoolingLoadFeatureBuilder, model_feature_columns
from cooling_load.modeling import build_pipeline
from cooling_load.preprocessing import preprocess_sensor_data
from cooling_load.synthetic import generate_synthetic_data
from sklearn.ensemble import ExtraTreesRegressor


def test_service_builds_future_row_and_predicts():
    config = load_config("configs/base.yaml")
    raw = generate_synthetic_data(periods=300, buildings=("A", "B"))
    clean = preprocess_sensor_data(raw, config.data)
    featured = CoolingLoadFeatureBuilder(config.data, config.features).transform(clean)
    clusterer = OperatingRegimeClusterer(config.clustering.features, n_clusters=4)
    featured = clusterer.fit_transform(featured, "cooling_load_kwh_lag_3")
    columns = model_feature_columns(featured, config.data)
    model = build_pipeline(columns, ExtraTreesRegressor(n_estimators=10, random_state=1))
    model.fit(featured[columns], featured[config.data.target_column])
    bundle = create_bundle(model, clusterer, columns, config.raw, {}, "test-model")
    service = ForecastService(bundle)
    observations = clean[clean.building_id.eq("A")].tail(200).to_dict(orient="records")
    response = service.forecast(observations)
    assert response["forecast_horizon_hours"] == 3
    assert response["cooling_load_kwh"] >= 0
    assert pd.Timestamp(response["forecast_timestamp"]) > pd.Timestamp(observations[-1]["timestamp"])

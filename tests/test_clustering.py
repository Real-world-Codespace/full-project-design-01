from cooling_load.clustering import OperatingRegimeClusterer
from cooling_load.config import load_config
from cooling_load.features import CoolingLoadFeatureBuilder
from cooling_load.preprocessing import preprocess_sensor_data
from cooling_load.synthetic import generate_synthetic_data


def test_clusterer_adds_regime_features():
    config = load_config("configs/base.yaml")
    raw = generate_synthetic_data(periods=300, buildings=("A", "B"))
    clean = preprocess_sensor_data(raw, config.data)
    featured = CoolingLoadFeatureBuilder(config.data, config.features).transform(clean)
    clusterer = OperatingRegimeClusterer(config.clustering.features, n_clusters=4)
    result = clusterer.fit_transform(featured, "cooling_load_kwh_lag_3")
    assert result["operating_cluster"].nunique() == 4
    assert result["operating_regime"].notna().all()
    assert (result["cluster_distance"] >= 0).all()

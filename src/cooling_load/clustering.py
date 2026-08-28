"""Operating-regime discovery using training-only K-Means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class OperatingRegimeClusterer:
    feature_columns: tuple[str, ...]
    n_clusters: int = 4
    random_state: int = 42

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "kmeans",
                    KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state),
                ),
            ]
        )
        self.label_map_: dict[int, str] = {}

    def fit(self, frame: pd.DataFrame, ordering_feature: str) -> "OperatingRegimeClusterer":
        self._require_features(frame)
        labels = self.pipeline.fit_predict(frame[list(self.feature_columns)])
        ordering = pd.DataFrame({"cluster": labels, "value": frame[ordering_feature].to_numpy()})
        ordered_clusters = ordering.groupby("cluster")["value"].median().sort_values().index.tolist()
        names = self._regime_names(len(ordered_clusters))
        self.label_map_ = dict(zip(ordered_clusters, names, strict=True))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_features(frame)
        result = frame.copy()
        labels = self.pipeline.predict(result[list(self.feature_columns)])
        distances = self.pipeline.transform(result[list(self.feature_columns)])
        result["operating_cluster"] = labels.astype(int)
        result["operating_regime"] = pd.Series(labels, index=result.index).map(self.label_map_)
        result["cluster_distance"] = np.min(distances, axis=1)
        return result

    def fit_transform(self, frame: pd.DataFrame, ordering_feature: str) -> pd.DataFrame:
        return self.fit(frame, ordering_feature).transform(frame)

    def _require_features(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing clustering features: {missing}")

    @staticmethod
    def _regime_names(count: int) -> list[str]:
        if count == 4:
            return ["low_load", "normal_load", "high_load", "peak_load"]
        return [f"regime_{index}" for index in range(count)]

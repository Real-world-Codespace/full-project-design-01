from datetime import datetime, timezone

from cooling_load.artifacts import ModelBundle
from cooling_load.registry import model_version, parse_s3_uri, publish_model_bundle


class FakeS3Client:
    def __init__(self):
        self.uploads = []
        self.objects = []

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads.append((filename, bucket, key, ExtraArgs))

    def put_object(self, **kwargs):
        self.objects.append(kwargs)


def test_parse_s3_uri():
    assert parse_s3_uri("s3://models/cooling/v1/model.joblib") == (
        "models",
        "cooling/v1/model.joblib",
    )


def test_publish_uses_versioned_key_and_latest_pointer(tmp_path):
    artifact = tmp_path / "champion.joblib"
    artifact.write_bytes(b"model-data")
    bundle = ModelBundle(
        model=None,
        clusterer=None,
        feature_columns=("a", "b"),
        project_config={},
        metrics={"rmse": 1.5},
        model_name="extra-trees",
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc).isoformat(),
    )
    fake = FakeS3Client()
    published = publish_model_bundle(
        artifact,
        bundle,
        bucket="model-bucket",
        prefix="cooling/models",
        client=fake,
    )
    assert model_version(bundle) in published.model_uri
    assert fake.uploads[0][2].endswith("/champion.joblib")
    assert {item["Key"] for item in fake.objects} == {
        f"cooling/models/{model_version(bundle)}/metadata.json",
        "cooling/models/latest.json",
    }

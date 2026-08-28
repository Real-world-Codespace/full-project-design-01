"""Versioned model-bundle storage backed by Amazon S3."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from cooling_load.artifacts import ModelBundle
from cooling_load.ingestion.manifest import sha256_file


@dataclass(frozen=True)
class PublishedModel:
    model_uri: str
    metadata_uri: str
    latest_uri: str
    version: str
    sha256: str


def model_version(bundle: ModelBundle) -> str:
    created = bundle.created_at.replace(":", "").replace("+", "_").replace("-", "")
    return f"{created}-{bundle.model_name}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client(profile: str | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install the project with the 'aws' extra to use the S3 registry.") from exc
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3")


def publish_model_bundle(
    local_model_path: str | Path,
    bundle: ModelBundle,
    bucket: str,
    prefix: str,
    profile: str | None = None,
    server_side_encryption: str | None = "AES256",
    client=None,
) -> PublishedModel:
    """Upload an immutable model version, metadata and a mutable latest pointer."""

    source = Path(local_model_path)
    if not source.exists():
        raise FileNotFoundError(source)
    s3 = client or _s3_client(profile)
    version = model_version(bundle)
    root = prefix.strip("/")
    model_key = f"{root}/{version}/champion.joblib"
    metadata_key = f"{root}/{version}/metadata.json"
    latest_key = f"{root}/latest.json"
    checksum = sha256_file(source)
    extra_args = {"ServerSideEncryption": server_side_encryption} if server_side_encryption else {}
    s3.upload_file(os.fspath(source), bucket, model_key, ExtraArgs=extra_args)

    metadata = {
        "version": version,
        "model_name": bundle.model_name,
        "created_at": bundle.created_at,
        "model_uri": f"s3://{bucket}/{model_key}",
        "sha256": checksum,
        "size_bytes": source.stat().st_size,
        "metrics": bundle.metrics,
        "feature_count": len(bundle.feature_columns),
    }
    put_args = {"ServerSideEncryption": server_side_encryption} if server_side_encryption else {}
    s3.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
        **put_args,
    )
    s3.put_object(
        Bucket=bucket,
        Key=latest_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
        **put_args,
    )
    return PublishedModel(
        model_uri=metadata["model_uri"],
        metadata_uri=f"s3://{bucket}/{metadata_key}",
        latest_uri=f"s3://{bucket}/{latest_key}",
        version=version,
        sha256=checksum,
    )


def download_model_bundle(
    uri: str,
    destination: str | Path,
    profile: str | None = None,
    expected_sha256: str | None = None,
    client=None,
) -> Path:
    """Atomically download a model artifact and optionally verify its SHA-256."""

    bucket, key = parse_s3_uri(uri)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    s3 = client or _s3_client(profile)
    s3.download_file(bucket, key, os.fspath(partial))
    if expected_sha256 and sha256_file(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise IOError(f"Checksum verification failed for {uri}")
    partial.replace(target)
    return target

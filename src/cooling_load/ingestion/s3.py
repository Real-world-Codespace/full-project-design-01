"""Safe, idempotent S3 downloader."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cooling_load.ingestion.manifest import sha256_file


@dataclass(frozen=True)
class DownloadedObject:
    key: str
    local_path: str
    size: int
    etag: str
    sha256: str
    downloaded: bool


def _safe_destination(root: Path, key: str, prefix: str) -> Path:
    relative = PurePosixPath(key).relative_to(PurePosixPath(prefix))
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe S3 object key: {key}")
    destination = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError(f"Object key escapes destination directory: {key}")
    return destination


def download_prefix(
    bucket: str,
    prefix: str,
    destination: str | Path,
    profile: str | None = None,
    allowed_extensions: tuple[str, ...] = (".csv", ".parquet"),
    max_workers: int = 8,
) -> list[DownloadedObject]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install the project with the 'aws' extra to ingest from S3.") from exc

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if Path(item["Key"]).suffix.lower() in allowed_extensions:
                objects.append(item)
    if not objects:
        raise FileNotFoundError(f"No supported objects found at s3://{bucket}/{prefix}")

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / ".s3_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    def download(item: dict[str, Any]) -> DownloadedObject:
        key = item["Key"]
        local_path = _safe_destination(root, key, prefix)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(item["Size"])
        etag = str(item.get("ETag", "")).strip('"')
        previous = state.get(key, {})
        should_download = (
            not local_path.exists()
            or local_path.stat().st_size != expected_size
            or previous.get("etag") != etag
        )
        if should_download:
            partial = local_path.with_suffix(local_path.suffix + ".partial")
            client.download_file(bucket, key, os.fspath(partial))
            if partial.stat().st_size != expected_size:
                partial.unlink(missing_ok=True)
                raise IOError(f"Size verification failed for s3://{bucket}/{key}")
            partial.replace(local_path)
        return DownloadedObject(
            key=key,
            local_path=os.fspath(local_path),
            size=expected_size,
            etag=etag,
            sha256=sha256_file(local_path),
            downloaded=should_download,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download, objects))
    new_state = {item.key: {"etag": item.etag, "size": item.size} for item in results}
    state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return results

"""Application service coordinating local and S3 ingestion."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from cooling_load.config import ProjectConfig
from cooling_load.ingestion.manifest import sha256_file, write_manifest
from cooling_load.ingestion.s3 import download_prefix


@dataclass(frozen=True)
class IngestionResult:
    mode: str
    files: tuple[Path, ...]
    manifest: Path


def ingest_dataset(config: ProjectConfig) -> IngestionResult:
    settings = config.raw["ingestion"]
    mode = settings["mode"].lower()
    manifest_path = config.path("manifest")

    if mode == "s3":
        s3 = settings["s3"]
        raw_root = config.path("raw_data").parent
        downloaded = download_prefix(
            bucket=s3["bucket"],
            prefix=s3["prefix"],
            destination=raw_root,
            profile=s3.get("profile"),
            allowed_extensions=tuple(s3["allowed_extensions"]),
            max_workers=int(s3["max_workers"]),
        )
        files = tuple(Path(item.local_path) for item in downloaded)
        manifest = write_manifest(
            manifest_path,
            source={"mode": "s3", "bucket": s3["bucket"], "prefix": s3["prefix"]},
            files=[asdict(item) for item in downloaded],
        )
        return IngestionResult(mode=mode, files=files, manifest=manifest)

    if mode == "local":
        source = config.root / settings["local_source"]
        destination = config.path("raw_data")
        if not source.exists():
            raise FileNotFoundError(
                f"Local source not found: {source}. Run `make sample` or configure S3 ingestion."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        file_record = {
            "key": source.name,
            "local_path": str(destination),
            "size": destination.stat().st_size,
            "etag": "",
            "sha256": sha256_file(destination),
            "downloaded": source.resolve() != destination.resolve(),
        }
        manifest = write_manifest(
            manifest_path,
            source={"mode": "local", "path": str(source)},
            files=[file_record],
        )
        return IngestionResult(mode=mode, files=(destination,), manifest=manifest)

    raise ValueError(f"Unsupported ingestion mode: {mode}")

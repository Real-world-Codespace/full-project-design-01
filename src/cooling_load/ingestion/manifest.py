"""Reproducibility manifest for downloaded objects."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    source: dict[str, Any],
    files: list[dict[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "source": source,
        "files": files,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination

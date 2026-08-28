"""Data-frame IO helpers with explicit, predictable formats."""

from pathlib import Path

import pandas as pd


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported data format: {suffix}")


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(destination, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    else:
        raise ValueError(f"Unsupported data format: {suffix}")
    return destination

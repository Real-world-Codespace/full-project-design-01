"""Reproducible EDA report generation with persisted chart artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cooling_load.config import DataConfig


@dataclass(frozen=True)
class EDAReport:
    figures: tuple[Path, ...]
    summary_path: Path


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def generate_eda_report(
    frame: pd.DataFrame,
    data_config: DataConfig,
    output_dir: str | Path,
    summary_path: str | Path,
    scatter_sample_size: int = 8_000,
) -> EDAReport:
    """Generate a deterministic set of PNG charts and a JSON data summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = data_config.timestamp_column
    entity = data_config.entity_column
    target = data_config.target_column
    working = frame.copy()
    working[timestamp] = pd.to_datetime(working[timestamp], errors="coerce", utc=True)
    numeric = working.select_dtypes(include="number")
    sns.set_theme(style="whitegrid")
    figures: list[Path] = []

    missing = working.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(11, 5))
    sns.barplot(x=missing.index, y=missing.values, color="#4472C4")
    plt.title("Missing-value fraction by column")
    plt.xlabel("Feature")
    plt.ylabel("Missing fraction")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(0.05, min(1.0, float(missing.max()) * 1.15)))
    figures.append(_save_figure(output / "01_missing_values.png"))

    plt.figure(figsize=(10, 5))
    sns.histplot(data=working, x=target, hue=entity, bins=45, element="step", stat="density")
    plt.title("Cooling-load distribution by building")
    figures.append(_save_figure(output / "02_target_distribution.png"))

    hourly = (
        working.assign(hour=working[timestamp].dt.hour)
        .groupby([entity, "hour"], as_index=False)[target]
        .mean()
    )
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=hourly, x="hour", y=target, hue=entity, marker="o")
    plt.title("Average intraday cooling-load profile")
    plt.xticks(range(0, 24, 2))
    figures.append(_save_figure(output / "03_intraday_profile.png"))

    daily = (
        working.set_index(timestamp)
        .groupby(entity)[target]
        .resample("D")
        .mean()
        .rename("daily_cooling_load_kwh")
        .reset_index()
    )
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=daily, x=timestamp, y="daily_cooling_load_kwh", hue=entity)
    plt.title("Daily average cooling load")
    figures.append(_save_figure(output / "04_daily_load_timeseries.png"))

    sample = working.sample(min(len(working), scatter_sample_size), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="outdoor_temperature_c",
        y=target,
        hue=entity,
        alpha=0.45,
        s=24,
    )
    plt.title("Cooling load versus outdoor temperature")
    figures.append(_save_figure(output / "05_load_vs_temperature.png"))

    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Numeric feature correlation")
    figures.append(_save_figure(output / "06_correlation_heatmap.png"))

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=working, x=entity, y=target)
    plt.title("Cooling-load range by building")
    figures.append(_save_figure(output / "07_target_by_building.png"))

    summary = {
        "rows": int(len(working)),
        "columns": int(len(working.columns)),
        "buildings": sorted(working[entity].dropna().astype(str).unique().tolist()),
        "time_start": working[timestamp].min().isoformat(),
        "time_end": working[timestamp].max().isoformat(),
        "duplicate_entity_timestamps": int(working.duplicated([entity, timestamp]).sum()),
        "missing_fraction": {name: float(value) for name, value in missing.items()},
        "target_statistics": {
            name: float(value) for name, value in working[target].describe().items()
        },
        "figures": [str(path) for path in figures],
    }
    summary_destination = Path(summary_path)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return EDAReport(figures=tuple(figures), summary_path=summary_destination)

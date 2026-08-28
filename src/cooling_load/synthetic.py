"""Generate a realistic local dataset for smoke tests and notebook walkthroughs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(
    periods: int = 24 * 120,
    buildings: tuple[str, ...] = ("A", "B", "C", "D"),
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    frames: list[pd.DataFrame] = []
    for index, building in enumerate(buildings):
        hour = timestamps.hour.to_numpy()
        day = np.arange(periods) / 24
        outdoor = 18 + 8 * np.sin(2 * np.pi * (hour - 8) / 24) + 3 * np.sin(2 * np.pi * day / 30)
        outdoor += rng.normal(0, 1.2, periods)
        humidity = np.clip(68 - 1.1 * (outdoor - 18) + rng.normal(0, 5, periods), 25, 98)
        weekday = timestamps.dayofweek.to_numpy() < 5
        occupied_hours = (hour >= 7) & (hour <= 19) & weekday
        occupancy = np.where(occupied_hours, 0.75 + 0.15 * rng.random(periods), 0.08)
        occupancy *= 0.85 + index * 0.07
        active = np.clip(np.ceil((outdoor - 16) / 6 + occupancy * 2), 1, 4).astype(float)
        supply = 6.5 + rng.normal(0, 0.3, periods)
        returned = supply + 2.5 + 0.22 * np.maximum(outdoor - 18, 0) + occupancy * 2
        flow = 35 + active * 28 + occupancy * 45 + rng.normal(0, 5, periods)
        scale = 0.85 + index * 0.12
        load = scale * (55 + 9 * np.maximum(outdoor - 17, 0) + 160 * occupancy + 0.9 * flow)
        load += 20 * np.sin(2 * np.pi * day / 7) + rng.normal(0, 18, periods)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "building_id": building,
                "cooling_load_kwh": np.maximum(load, 0),
                "outdoor_temperature_c": outdoor,
                "relative_humidity_pct": humidity,
                "chilled_water_supply_c": supply,
                "chilled_water_return_c": returned,
                "chilled_water_flow_m3h": flow,
                "active_chillers": active,
                "occupancy_proxy": occupancy,
            }
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "building_id"])
    missing_mask = rng.random(len(result)) < 0.006
    result.loc[missing_mask, "relative_humidity_pct"] = np.nan
    duplicate = result.sample(12, random_state=seed)
    return pd.concat([result, duplicate], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/cooling_load.csv")
    parser.add_argument("--periods", type=int, default=24 * 120)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data(periods=args.periods).to_csv(output, index=False)
    print(f"Wrote synthetic dataset to {output}")


if __name__ == "__main__":
    main()

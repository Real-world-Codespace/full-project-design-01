"""HTTP request and response contracts."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SensorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    building_id: str = Field(min_length=1, max_length=64)
    cooling_load_kwh: float | None = Field(default=None, ge=0)
    outdoor_temperature_c: float | None = Field(default=None, ge=-60, le=70)
    relative_humidity_pct: float | None = Field(default=None, ge=0, le=100)
    chilled_water_supply_c: float | None = Field(default=None, ge=-10, le=50)
    chilled_water_return_c: float | None = Field(default=None, ge=-10, le=70)
    chilled_water_flow_m3h: float | None = Field(default=None, ge=0)
    active_chillers: float | None = Field(default=None, ge=0, le=20)
    occupancy_proxy: float | None = Field(default=None, ge=0, le=2)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[SensorObservation] = Field(min_length=24)


class ForecastResponse(BaseModel):
    building_id: str
    forecast_timestamp: datetime
    forecast_horizon_hours: int
    cooling_load_kwh: float
    operating_regime: str
    model_name: str
    model_created_at: str


class ModelInfoResponse(BaseModel):
    model_name: str
    created_at: str
    metrics: dict[str, float]
    feature_count: int

"""FastAPI application factory and endpoints."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from cooling_load.api.schemas import ForecastRequest, ForecastResponse, ModelInfoResponse
from cooling_load.api.service import ForecastService
from cooling_load.artifacts import load_bundle
from cooling_load.registry import download_model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "models/champion.joblib"))
    model_s3_uri = os.getenv("MODEL_S3_URI")
    if model_s3_uri:
        download_model_bundle(
            model_s3_uri,
            model_path,
            profile=os.getenv("AWS_PROFILE") or None,
            expected_sha256=os.getenv("MODEL_SHA256") or None,
        )
    app.state.forecast_service = ForecastService(load_bundle(model_path)) if model_path.exists() else None
    yield


app = FastAPI(title="Cooling Load Forecasting API", version="0.1.0", lifespan=lifespan)


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def readiness(request: Request) -> dict[str, str]:
    if request.app.state.forecast_service is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    return {"status": "ready"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = request.app.state.forecast_service
    if service is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    bundle = service.bundle
    return ModelInfoResponse(
        model_name=bundle.model_name,
        created_at=bundle.created_at,
        metrics=bundle.metrics,
        feature_count=len(bundle.feature_columns),
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
    service = request.app.state.forecast_service
    if service is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    try:
        observations = [item.model_dump() for item in payload.observations]
        return ForecastResponse(**service.forecast(observations))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

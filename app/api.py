from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request

from app.schemas import (
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from marketsignal.data.schema import records_to_frame
from marketsignal.inference.predictor import Predictor
from marketsignal.utils.logging import configure_logging

MAX_REQUEST_BYTES = 2 * 1024 * 1024
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _authorize(provided: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("MARKETSIGNAL_API_KEY", "")
    if expected and (provided is None or not secrets.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _allowed_hosts() -> list[str]:
    value = os.getenv("MARKETSIGNAL_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")
    return [host.strip() for host in value.split(",") if host.strip()]


def create_app(model_path: str | Path | None = None) -> FastAPI:
    configure_logging()
    production = os.getenv("MARKETSIGNAL_ENV", "development").lower() == "production"
    if production and not os.getenv("MARKETSIGNAL_API_KEY"):
        raise RuntimeError("MARKETSIGNAL_API_KEY is required in production.")
    application = FastAPI(
        title="MarketSignal Forecast API",
        version="2.0.0",
        description="Probabilistic next-session direction forecasts from OHLCV history.",
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts())

    artifact_path = Path(
        model_path or os.getenv("MARKETSIGNAL_MODEL_PATH", "artifacts/latest/model")
    )
    predictor: Predictor | None = None
    try:
        predictor = Predictor.from_artifact(artifact_path)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass

    @application.middleware("http")
    async def limit_request_size(request: Request, call_next: Callable):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return JSONResponse(status_code=413, content={"detail": "Request too large."})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
        return await call_next(request)

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ready" if predictor is not None else "model_unavailable",
            model_loaded=predictor is not None,
        )

    @application.get(
        "/model",
        response_model=ModelInfo,
        dependencies=[Depends(_authorize)],
    )
    def model_metadata() -> ModelInfo:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable.")
        metadata = predictor.model.metadata
        return ModelInfo(
            project=str(metadata["project"]),
            version=str(metadata["version"]),
            trained_at=str(metadata["trained_at"]),
            run_id=str(metadata["run_id"]),
            symbol=str(metadata["symbol"]),
            metrics=dict(metadata["metrics"]),
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        dependencies=[Depends(_authorize)],
    )
    def predict(request: PredictionRequest) -> PredictionResponse:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable.")
        try:
            frame = records_to_frame(candle.model_dump() for candle in request.candles)
            result = predictor.predict_latest(frame)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        metadata = predictor.model.metadata
        return PredictionResponse(
            **result.__dict__,
            model_version=str(metadata["version"]),
            run_id=str(metadata["run_id"]),
        )

    return application


app = create_app()

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: datetime
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)
    volume: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_range(self) -> Candle:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, close, and high")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candles: list[Candle] = Field(min_length=80, max_length=2_500)


class PredictionResponse(BaseModel):
    timestamp: str
    probability_up: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    direction: Literal["up", "down", "uncertain"]
    decision_threshold: float = Field(gt=0, lt=1)
    forecast_horizon: int = Field(ge=1)
    model_version: str
    run_id: str


class ModelInfo(BaseModel):
    project: str
    version: str
    trained_at: str
    run_id: str
    symbol: str
    metrics: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ready", "model_unavailable"]
    model_loaded: bool

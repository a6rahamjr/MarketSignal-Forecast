from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketsignal.features.technical import build_feature_frame
from marketsignal.models.artifact import ForestArtifact, load_artifact


@dataclass(frozen=True)
class Prediction:
    timestamp: str
    probability_up: float
    confidence: float
    direction: str
    decision_threshold: float
    forecast_horizon: int


class Predictor:
    def __init__(self, model: ForestArtifact):
        self.model = model

    @classmethod
    def from_artifact(cls, path: str | Path) -> "Predictor":
        return cls(load_artifact(path))

    def predict_latest(self, market_data: pd.DataFrame) -> Prediction:
        features = build_feature_frame(market_data).dropna()
        if features.empty:
            raise ValueError("At least 80 valid observations are required for prediction.")

        latest = features.iloc[[-1]]
        probability = float(self.model.predict_probability(latest).iloc[0])
        distance = probability - self.model.decision_threshold
        if abs(distance) < self.model.abstain_margin:
            direction = "uncertain"
        else:
            direction = "up" if distance > 0 else "down"

        confidence = max(probability, 1 - probability)
        return Prediction(
            timestamp=latest.index[-1].isoformat(),
            probability_up=probability,
            confidence=confidence,
            direction=direction,
            decision_threshold=self.model.decision_threshold,
            forecast_horizon=self.model.forecast_horizon,
        )

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "MarketSignal Forecast"
    random_seed: int = 42
    artifact_dir: str = "artifacts"
    data_dir: str = "data/generated"


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    symbol: str = "DEMO"
    rows: int = 1800
    start_date: str = "2018-01-01"
    csv_path: str = ""
    period: str = "10y"
    interval: str = "1d"


@dataclass(frozen=True)
class FeatureConfig:
    forecast_horizon: int = 1
    minimum_history: int = 80


@dataclass(frozen=True)
class TrainingConfig:
    test_fraction: float = 0.20
    tune: bool = True
    cv_splits: int = 5
    search_iterations: int = 10
    search_estimators: int = 160
    abstain_margin: float = 0.04
    transaction_cost_bps: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 400
    min_samples_leaf: int = 8
    max_features: float = 0.75
    max_depth: int = 0


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    section = values.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section [{name}] must be a table.")
    return section


def load_config(path: str | Path = "configs/default.toml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        values = tomllib.load(handle)

    config = AppConfig(
        project=ProjectConfig(**_section(values, "project")),
        data=DataConfig(**_section(values, "data")),
        features=FeatureConfig(**_section(values, "features")),
        training=TrainingConfig(**_section(values, "training")),
        model=ModelConfig(**_section(values, "model")),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.data.source not in {"synthetic", "csv", "yfinance"}:
        raise ValueError("data.source must be one of: synthetic, csv, yfinance")
    if config.data.rows < config.features.minimum_history + 100:
        raise ValueError("data.rows is too small for feature generation and evaluation")
    if not 0.05 <= config.training.test_fraction <= 0.5:
        raise ValueError("training.test_fraction must be between 0.05 and 0.5")
    if not 0.0 <= config.training.abstain_margin < 0.25:
        raise ValueError("training.abstain_margin must be between 0 and 0.25")
    if config.training.transaction_cost_bps < 0:
        raise ValueError("training.transaction_cost_bps cannot be negative")
    if config.training.search_estimators < 50:
        raise ValueError("training.search_estimators must be at least 50")
    if config.features.forecast_horizon < 1:
        raise ValueError("features.forecast_horizon must be at least 1")
    if config.model.n_estimators < 50:
        raise ValueError("model.n_estimators must be at least 50")
    if not 0.1 <= config.model.max_features <= 1.0:
        raise ValueError("model.max_features must be between 0.1 and 1.0")

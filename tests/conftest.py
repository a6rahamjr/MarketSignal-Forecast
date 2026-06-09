from __future__ import annotations

import pytest

from marketsignal.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from marketsignal.data.synthetic import generate_synthetic_ohlcv
from marketsignal.training.pipeline import TrainingResult, train_model


@pytest.fixture(scope="session")
def synthetic_market():
    return generate_synthetic_ohlcv(rows=720, seed=17)


@pytest.fixture(scope="session")
def trained_result(tmp_path_factory, synthetic_market) -> TrainingResult:
    artifact_root = tmp_path_factory.mktemp("artifacts")
    config = AppConfig(
        project=ProjectConfig(random_seed=17, artifact_dir=str(artifact_root)),
        data=DataConfig(rows=720),
        training=TrainingConfig(
            test_fraction=0.2,
            tune=False,
            cv_splits=3,
            search_iterations=2,
            abstain_margin=0.03,
        ),
        model=ModelConfig(
            n_estimators=100,
            min_samples_leaf=6,
            max_features=0.75,
            max_depth=12,
        ),
    )
    return train_model(synthetic_market, config, artifact_root=artifact_root)

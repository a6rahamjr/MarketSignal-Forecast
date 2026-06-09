from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from marketsignal.config import AppConfig
from marketsignal.evaluation.explainability import permutation_feature_importance
from marketsignal.evaluation.metrics import (
    classification_metrics,
    selective_metrics,
    strategy_metrics,
)
from marketsignal.features.technical import FEATURE_COLUMNS, build_supervised_dataset
from marketsignal.models.artifact import save_artifact
from marketsignal.models.factory import build_baseline, build_forest, search_space
from marketsignal.utils.io import write_json
from marketsignal.utils.seeding import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    run_dir: Path
    model_path: Path
    metrics: dict[str, Any]


def chronological_split(
    dataset: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = int(len(dataset) * (1 - test_fraction))
    if split_at < 160 or len(dataset) - split_at < 30:
        raise ValueError("Dataset is too small for a reliable chronological holdout.")

    train = dataset.iloc[:split_at].copy()
    test = dataset.iloc[split_at:].copy()
    if train["target"].nunique() != 2 or test["target"].nunique() != 2:
        raise ValueError("Training and holdout partitions must both contain two classes.")
    return train, test


def _time_series_split(config: AppConfig, row_count: int) -> TimeSeriesSplit:
    split_count = min(config.training.cv_splits, max(2, row_count // 120))
    return TimeSeriesSplit(
        n_splits=split_count,
        gap=config.features.forecast_horizon,
    )


def _fit_forest(
    features: pd.DataFrame,
    target: pd.Series,
    config: AppConfig,
) -> tuple[object, dict[str, Any]]:
    estimator = build_forest(config.model, config.project.random_seed)
    if not config.training.tune:
        estimator.fit(features, target)
        return estimator, {}

    search_estimator = clone(estimator).set_params(
        classifier__n_estimators=min(
            config.model.n_estimators,
            config.training.search_estimators,
        )
    )
    search = RandomizedSearchCV(
        estimator=search_estimator,
        param_distributions=search_space(),
        n_iter=config.training.search_iterations,
        scoring="roc_auc",
        cv=_time_series_split(config, len(features)),
        random_state=config.project.random_seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(features, target)
    estimator.set_params(**search.best_params_)
    return estimator, {
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "search_estimators": min(
            config.model.n_estimators,
            config.training.search_estimators,
        ),
    }


def _out_of_fold_probabilities(
    estimator: object,
    features: pd.DataFrame,
    target: pd.Series,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.full(len(features), np.nan, dtype=np.float64)
    for train_index, validation_index in _time_series_split(config, len(features)).split(features):
        fold_model = clone(estimator)
        fold_model.fit(features.iloc[train_index], target.iloc[train_index])
        probabilities[validation_index] = fold_model.predict_proba(
            features.iloc[validation_index]
        )[:, 1]

    valid = np.isfinite(probabilities)
    if valid.sum() < 100 or target.iloc[valid].nunique() != 2:
        raise ValueError("Not enough temporal validation predictions for calibration.")
    return probabilities[valid], target.to_numpy()[valid]


def _fit_calibrator(
    raw_probability: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000, solver="lbfgs")
    calibrator.fit(logits, target)
    calibrated = calibrator.predict_proba(logits)[:, 1]
    return (
        float(calibrator.coef_[0, 0]),
        float(calibrator.intercept_[0]),
        calibrated,
    )


def _apply_calibration(
    raw_probability: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    values = intercept + slope * logits
    return np.where(
        values >= 0,
        1 / (1 + np.exp(-values)),
        np.exp(values) / (1 + np.exp(values)),
    )


def _select_threshold(probability: np.ndarray, target: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probability, np.linspace(0.1, 0.9, 161)))
    scored = [
        (
            balanced_accuracy_score(target, probability >= threshold),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored)[2]


def _publish_latest(run_dir: Path, artifact_root: Path) -> None:
    latest = artifact_root / "latest"
    staging = artifact_root / ".latest"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copytree(run_dir / "model", staging / "model")
    for filename in ("metrics.json", "feature_importance.csv", "predictions.csv"):
        shutil.copy2(run_dir / filename, staging / filename)

    if latest.exists():
        shutil.rmtree(latest)
    staging.replace(latest)


def train_model(
    market_data: pd.DataFrame,
    config: AppConfig,
    artifact_root: str | Path | None = None,
) -> TrainingResult:
    seed_everything(config.project.random_seed)
    dataset = build_supervised_dataset(
        market_data,
        forecast_horizon=config.features.forecast_horizon,
        minimum_history=config.features.minimum_history,
    )
    train, test = chronological_split(dataset, config.training.test_fraction)
    train_features = train[FEATURE_COLUMNS]
    test_features = test[FEATURE_COLUMNS]
    train_target = train["target"]
    test_target = test["target"]

    LOGGER.info("training_rows=%d holdout_rows=%d", len(train), len(test))

    baseline = build_baseline(config.project.random_seed)
    baseline.fit(train_features, train_target)
    baseline_probability = baseline.predict_proba(test_features)[:, 1]
    baseline_metrics = classification_metrics(test_target, baseline_probability)

    forest, tuning = _fit_forest(train_features, train_target, config)
    oof_probability, oof_target = _out_of_fold_probabilities(
        forest,
        train_features,
        train_target,
        config,
    )
    slope, intercept, calibrated_oof = _fit_calibrator(oof_probability, oof_target)
    decision_threshold = _select_threshold(calibrated_oof, oof_target)

    forest.fit(train_features, train_target)
    raw_holdout = forest.predict_proba(test_features)[:, 1]
    holdout_probability = _apply_calibration(raw_holdout, slope, intercept)
    model_metrics = classification_metrics(
        test_target,
        holdout_probability,
        decision_threshold,
    )
    confidence_metrics = selective_metrics(
        test_target,
        holdout_probability,
        decision_threshold,
        config.training.abstain_margin,
    )
    backtest_metrics = strategy_metrics(
        test["future_return"],
        holdout_probability,
        threshold=decision_threshold,
        abstain_margin=config.training.abstain_margin,
        transaction_cost_bps=config.training.transaction_cost_bps,
    )
    importance = permutation_feature_importance(
        forest,
        test_features,
        test_target,
        config.project.random_seed,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_root = Path(artifact_root or config.project.artifact_dir)
    run_dir = artifact_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "model"

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "data": {
            "source": config.data.source,
            "symbol": config.data.symbol,
            "total_rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
        },
        "baseline": baseline_metrics,
        "model": model_metrics,
        "confidence": confidence_metrics,
        "decision_threshold": decision_threshold,
        "abstain_margin": config.training.abstain_margin,
        "improvement": {
            "accuracy_delta": model_metrics["accuracy"] - baseline_metrics["accuracy"],
            "f1_delta": model_metrics["f1"] - baseline_metrics["f1"],
            "roc_auc_delta": model_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        },
        "strategy": backtest_metrics,
        "tuning": tuning,
    }
    model_metadata = {
        "project": config.project.name,
        "version": "2.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "symbol": config.data.symbol,
        "metrics": model_metrics,
        "config": asdict(config),
    }
    save_artifact(
        forest,
        model_path,
        feature_columns=FEATURE_COLUMNS.copy(),
        calibration_slope=slope,
        calibration_intercept=intercept,
        decision_threshold=decision_threshold,
        abstain_margin=config.training.abstain_margin,
        forecast_horizon=config.features.forecast_horizon,
        metadata=model_metadata,
    )
    write_json(metrics, run_dir / "metrics.json")
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": test.index,
            "actual": test_target.to_numpy(),
            "probability_up": holdout_probability,
            "direction": np.where(
                np.abs(holdout_probability - decision_threshold)
                < config.training.abstain_margin,
                "uncertain",
                np.where(holdout_probability >= decision_threshold, "up", "down"),
            ),
            "future_return": test["future_return"].to_numpy(),
        }
    ).to_csv(run_dir / "predictions.csv", index=False)
    _publish_latest(run_dir, artifact_root)

    LOGGER.info("model_path=%s", model_path)
    return TrainingResult(run_id, run_dir, model_path, metrics)

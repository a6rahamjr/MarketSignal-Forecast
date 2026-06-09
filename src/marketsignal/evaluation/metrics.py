from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    predicted = (probability >= threshold).astype(int)
    has_both_classes = np.unique(actual).size == 2

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, probability)) if has_both_classes else math.nan,
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual, probability)),
    }


def strategy_metrics(
    future_returns: pd.Series,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
    abstain_margin: float = 0.0,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    probability = pd.Series(np.asarray(probabilities), index=future_returns.index)
    upper = threshold + abstain_margin
    lower = threshold - abstain_margin
    position = pd.Series(0.0, index=future_returns.index)
    position.loc[probability >= upper] = 1.0
    position.loc[probability <= lower] = -1.0
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * transaction_cost_bps / 10_000
    strategy_returns = position * future_returns - costs
    equity_curve = (1 + strategy_returns).cumprod()
    benchmark_curve = (1 + future_returns).cumprod()

    annualized_return = equity_curve.iloc[-1] ** (periods_per_year / len(equity_curve)) - 1
    annualized_volatility = strategy_returns.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (
        strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(periods_per_year)
        if strategy_returns.std(ddof=0) > 0
        else 0.0
    )
    drawdown = equity_curve / equity_curve.cummax() - 1

    return {
        "total_return": float(equity_curve.iloc[-1] - 1),
        "benchmark_return": float(benchmark_curve.iloc[-1] - 1),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "coverage": float((position != 0).mean()),
        "turnover": float(turnover.sum()),
        "transaction_cost_bps": float(transaction_cost_bps),
        "observations": int(len(equity_curve)),
    }


def selective_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
    abstain_margin: float,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    selected = np.abs(probability - threshold) >= abstain_margin
    if not selected.any():
        return {"coverage": 0.0, "selective_accuracy": math.nan}
    predicted = (probability[selected] >= threshold).astype(int)
    return {
        "coverage": float(selected.mean()),
        "selective_accuracy": float(accuracy_score(actual[selected], predicted)),
    }

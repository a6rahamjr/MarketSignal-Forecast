from __future__ import annotations

import numpy as np
import pandas as pd

from marketsignal.data.schema import normalize_ohlcv

FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return_1d",
    "momentum_acceleration",
    "close_to_sma_5",
    "close_to_sma_10",
    "close_to_sma_20",
    "close_to_sma_50",
    "bollinger_z_20",
    "trend_slope_10",
    "trend_slope_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "downside_volatility_20",
    "return_skew_20",
    "return_autocorr_10",
    "rsi_14",
    "macd",
    "macd_histogram",
    "atr_14",
    "stochastic_14",
    "overnight_gap",
    "intraday_return",
    "range_fraction",
    "close_location",
    "upper_wick",
    "lower_wick",
    "volume_change_1d",
    "volume_zscore_20",
    "volume_trend_5",
    "price_volume_correlation_20",
    "chaikin_money_flow_20",
]


def _relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    losses = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def _average_true_range(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean() / frame["close"]


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = np.dot(x, x)
    return series.rolling(window).apply(
        lambda values: np.dot(values - values.mean(), x) / denominator,
        raw=True,
    )


def _feature_frame(market: pd.DataFrame) -> pd.DataFrame:
    close = market["close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=market.index)
    features["return_1d"] = returns
    for window in (2, 5, 10, 20):
        features[f"return_{window}d"] = close.pct_change(window)
    features["log_return_1d"] = np.log(close).diff()
    features["momentum_acceleration"] = features["return_5d"] - features["return_20d"] / 4

    for window in (5, 10, 20, 50):
        moving_average = close.rolling(window).mean()
        features[f"close_to_sma_{window}"] = close / moving_average - 1

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    features["bollinger_z_20"] = (close - rolling_mean) / rolling_std
    log_close = np.log(close)
    features["trend_slope_10"] = _rolling_slope(log_close, 10)
    features["trend_slope_20"] = _rolling_slope(log_close, 20)

    for window in (5, 10, 20):
        features[f"volatility_{window}"] = returns.rolling(window).std()

    downside_returns = returns.where(returns < 0, 0)
    features["downside_volatility_20"] = downside_returns.rolling(20).std()
    features["return_skew_20"] = returns.rolling(20).skew()
    features["return_autocorr_10"] = returns.rolling(10).corr(returns.shift(1))

    features["rsi_14"] = _relative_strength_index(close) / 100
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_absolute = ema_fast - ema_slow
    features["macd"] = macd_absolute / close
    macd_signal = macd_absolute.ewm(span=9, adjust=False).mean()
    features["macd_histogram"] = (macd_absolute - macd_signal) / close
    features["atr_14"] = _average_true_range(market)

    rolling_low = market["low"].rolling(14).min()
    rolling_high = market["high"].rolling(14).max()
    daily_range = (market["high"] - market["low"]).replace(0, np.nan)
    features["stochastic_14"] = (close - rolling_low) / (rolling_high - rolling_low)
    features["overnight_gap"] = market["open"] / close.shift(1) - 1
    features["intraday_return"] = close / market["open"] - 1
    features["range_fraction"] = daily_range / close
    features["close_location"] = (2 * close - market["high"] - market["low"]) / daily_range
    features["upper_wick"] = (market["high"] - market[["open", "close"]].max(axis=1)) / close
    features["lower_wick"] = (market[["open", "close"]].min(axis=1) - market["low"]) / close

    volume_mean = market["volume"].rolling(20).mean()
    volume_std = market["volume"].rolling(20).std()
    features["volume_change_1d"] = market["volume"].pct_change()
    features["volume_zscore_20"] = (market["volume"] - volume_mean) / volume_std
    features["volume_trend_5"] = market["volume"] / market["volume"].rolling(5).mean() - 1
    features["price_volume_correlation_20"] = returns.rolling(20).corr(
        market["volume"].pct_change()
    )
    money_flow_multiplier = (2 * close - market["high"] - market["low"]) / daily_range
    money_flow_volume = money_flow_multiplier * market["volume"]
    features["chaikin_money_flow_20"] = (
        money_flow_volume.rolling(20).sum() / market["volume"].rolling(20).sum()
    )

    return features.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _feature_frame(normalize_ohlcv(frame))


def build_supervised_dataset(
    frame: pd.DataFrame,
    forecast_horizon: int = 1,
    minimum_history: int = 60,
) -> pd.DataFrame:
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1")

    market = normalize_ohlcv(frame)
    if len(market) < minimum_history + forecast_horizon:
        raise ValueError(
            f"At least {minimum_history + forecast_horizon} rows are required."
        )

    features = _feature_frame(market)
    future_return = market["close"].shift(-forecast_horizon) / market["close"] - 1
    dataset = features.copy()
    dataset["future_return"] = future_return
    dataset["target"] = (future_return > 0).astype(float)
    dataset.loc[future_return.isna(), "target"] = np.nan
    dataset = dataset.dropna()
    dataset["target"] = dataset["target"].astype(int)

    if dataset["target"].nunique() < 2:
        raise ValueError("The generated target contains only one class.")
    return dataset

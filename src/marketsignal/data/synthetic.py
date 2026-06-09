from __future__ import annotations

import numpy as np
import pandas as pd

from marketsignal.data.schema import normalize_ohlcv


def generate_synthetic_ohlcv(
    rows: int = 1800,
    start_date: str = "2018-01-01",
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate regime-switching OHLCV data with volatility clustering.

    The process intentionally contains momentum, mean-reversion, and volatility
    interactions so a nonlinear model can be evaluated against a linear baseline.
    """
    if rows < 100:
        raise ValueError("Synthetic generation requires at least 100 rows.")

    rng = np.random.default_rng(seed)
    regimes = np.zeros(rows, dtype=int)
    returns = np.zeros(rows)
    volatility = np.zeros(rows)
    volatility[0] = 0.012

    transition = np.array(
        [
            [0.965, 0.025, 0.010],
            [0.035, 0.945, 0.020],
            [0.075, 0.075, 0.850],
        ]
    )
    regime_drift = np.array([0.0008, -0.0006, 0.0])
    regime_noise = np.array([0.0055, 0.007, 0.016])

    for index in range(1, rows):
        regimes[index] = rng.choice(3, p=transition[regimes[index - 1]])
        recent_returns = returns[max(0, index - 5) : index]
        recent_direction = 1.0 if recent_returns.sum() >= 0 else -1.0
        momentum = 0.16 * returns[index - 1]
        volatility[index] = (
            0.0008
            + 0.86 * volatility[index - 1]
            + 0.12 * abs(returns[index - 1])
        )
        if volatility[index] < 0.012:
            nonlinear_signal = 0.0045 * recent_direction
        elif abs(returns[index - 1]) > 1.15 * volatility[index]:
            nonlinear_signal = -0.006 * np.sign(returns[index - 1])
        else:
            nonlinear_signal = 0.0015 * recent_direction
        shock_scale = regime_noise[regimes[index]] + 0.35 * volatility[index]
        returns[index] = (
            regime_drift[regimes[index]]
            + momentum
            + nonlinear_signal
            + rng.normal(0.0, shock_scale)
        )
        returns[index] = np.clip(returns[index], -0.18, 0.18)

    close = start_price * np.exp(np.cumsum(returns))
    overnight = rng.normal(0, 0.0025 + volatility * 0.12, rows)
    open_price = np.r_[start_price, close[:-1]] * np.exp(overnight)
    intraday_range = np.abs(rng.normal(0.008, 0.004, rows)) + 0.45 * np.abs(returns)
    high = np.maximum(open_price, close) * (1 + intraday_range)
    low = np.minimum(open_price, close) * np.maximum(0.01, 1 - intraday_range)

    base_volume = 1_200_000 * (1 + 18 * np.abs(returns) + 8 * volatility)
    regime_volume = np.choose(regimes, [1.0, 1.1, 1.7])
    volume = rng.lognormal(np.log(base_volume * regime_volume), 0.22).astype(np.int64)

    index = pd.bdate_range(start=start_date, periods=rows, tz="UTC")
    frame = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    frame.index.name = "date"
    return normalize_ohlcv(frame)
